import threading
import websockets
import keyboard
import asyncio
import queue
import time
import json
import ast

from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager
from core.tools.registry import ToolRegistry

from core.common.config import ConfigManager
from core.common.logger import LogManager

#配置
USER = "Zostime"
ENABLE_STT = False
ENABLE_TOOLS = True    #某些LLM不支持tool_calls则设为False
WEBSOCKET_PORT = 8085

#全局变量
SYSTEM_MEMORY = ""
SYSTEM_EMOTION = {
    "开心":0.5,
    "悲伤":0.5,
    "愤怒":0.5,
    "恐惧":0.5,
    "惊讶":0.5,
    "厌恶":0.5,
    "信任":0.5,
    "期待":0.5,
    "爱":0.5,
    "嫉妒":0.5
}   # 0.00~1.00

class BehaviorController:
    def __init__(self,llm: LLMClient):
        self.llm = llm
        self.messages=[]

    def response_policy(self):
        pass

    def emotion_policy(self):
        global SYSTEM_MEMORY, SYSTEM_EMOTION
        self.messages = [
            {"role": "system", "content": f"记忆上下文:{SYSTEM_MEMORY}"},
            {"role": "user","content": f"当前情绪:{SYSTEM_EMOTION},根据记忆上下文更新情绪,只返回一个字典,值范围在0.00~1.00间"}
        ]
        for attempt in range(3):
            res = self.llm.chat(messages=self.messages)
            raw_content = res["full_content"]
            try:
                emotion_dict = ast.literal_eval(raw_content)
                if not isinstance(emotion_dict, dict):
                    raise ValueError("返回内容不是字典")
                for k, v in emotion_dict.items():
                    if not (0.0 <= v <= 1.0):
                        raise ValueError(f"情绪'{k}'的值{v}超出[0.00, 1.00]范围")
                SYSTEM_EMOTION = emotion_dict
                break
            except Exception as e:
                self.messages.append({"role": "assistant", "content": raw_content})
                self.messages.append({"role": "user", "content": f"错误: {e},请重新生成一个合法的情绪字典."})
        else:
            raise RuntimeError("失败次数过多,无法获取合法情绪字典")

        SYNC.push({
            "type": "emotion",
            "data": SYSTEM_EMOTION
        })

class InterruptManager:
    def __init__(self):
        self.event = threading.Event()
        threading.Thread(target=self.listener, daemon=True).start()

    def trigger(self):
        TTS.interrupt()
        self.event.set()

    def clear(self):
        self.event.clear()

    def is_interrupted(self):
        return self.event.is_set()

    def listener(self):
        while True:
            if ENABLE_STT:
                pass
            else:
                keyboard.wait("ctrl+f1")
                self.trigger()
                handle_user_input()

def build_memory_context(user_input):
    user_ltm = MEMORY.search_ltm(user_input,USER)    # [{'memory': str, 'score': datetime}, ...]
    assistant_ltm = MEMORY.search_ltm(user_input, 'Airis')
    user_stm = MEMORY.search_stm(USER)               # [{"memory": str, "timestamp": datetime}, ...]
    assistant_stm = MEMORY.search_stm('Airis')

    # STM排序(最新在前)
    user_stm = sorted(user_stm, key=lambda x: x.get("timestamp", 0), reverse=True)
    assistant_stm = sorted(assistant_stm, key=lambda x: x.get("timestamp", 0), reverse=True)

    # LTM排序(相关性高在前)
    user_ltm = sorted(user_ltm, key=lambda x: x.get("score", 0.5), reverse=True)
    assistant_ltm = sorted(assistant_ltm, key=lambda x: x.get("score", 0.5), reverse=True)

    parts = ["[记忆上下文]"]

    if user_stm:
        parts.append("\n用户短期上下文:")
        for i, entry in enumerate(user_stm):
            tag = "最新" if i == 0 else "较新"
            memory = entry.get("memory", "")
            parts.append(f"- [{tag}] {memory}")

    if assistant_stm:
        parts.append("\n助手短期上下文:")
        for i, entry in enumerate(assistant_stm):
            tag = "最新" if i == 0 else "较新"
            memory = entry.get("memory", "")
            parts.append(f"- [{tag}] {memory}")

    if user_ltm:
        parts.append("\n用户长期记忆:")
        for entry in user_ltm:
            memory = entry.get("memory")
            score = entry.get("score")
            parts.append(f"- [{score:.2f}] {memory}")

    if assistant_ltm:
        parts.append("\n助手长期记忆:")
        for entry in assistant_ltm:
            memory = entry.get("memory")
            score = entry.get("score")
            parts.append(f"- [{score:.2f}] {memory}")

    if len(parts) == 1:
        return "没有相关记忆"

    return "".join(parts)

def handle_user_input():
    print()
    if ENABLE_STT:
        while True:
            user_input = STT.listen_and_transcribe()
            if user_input is not None:
                print(f"\r{USER}:{user_input}")
                break
            else:
                print("\r未识别到音频", end='')
                time.sleep(1)
    else:
        user_input = input(f"{USER}:")

    global SYSTEM_MEMORY
    SYSTEM_MEMORY = build_memory_context(user_input)

    INTERRUPT.clear()
    llm_queue.put(user_input)

class StateSyncManager:
    def __init__(self):
        self.clients = set()
        self.queue = queue.Queue()

    def push(self, data):
        self.queue.put(data)

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.clients.remove(websocket)

    async def sender_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            data = await loop.run_in_executor(None, self.queue.get)
            if self.clients:
                msg = json.dumps(data)
                await asyncio.gather(*(ws.send(msg) for ws in self.clients))

    async def run(self):
        async with websockets.serve(self.handler, "localhost", WEBSOCKET_PORT):
            await self.sender_loop()

def llm_worker():
    while True:
        task = llm_queue.get()
        if task is None:
            break
        llm_input = task
        MEMORY.add_memory(llm_input, user_id=USER)

        system_prompt=f"""
        情绪:{SYSTEM_EMOTION}
        记忆上下文:{SYSTEM_MEMORY}
        记忆上下文 使用规则：
        - 短期上下文 表示最近对话，优先使用
        - 长期记忆 表示长期信息，按相关性使用
        - 优先参考标记为“最新”或高score的内容
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "name": USER, "content": llm_input}
        ]
        result = None
        while True:
            gen = LLM.chat_stream(
                messages=messages,
                tools=TOOLS.get_tools() if ENABLE_TOOLS else None,
                tool_choice="auto"
            )
            while True:
                try:
                    if INTERRUPT.is_interrupted():
                        gen.close()
                        break
                    chunk = next(gen)
                    print(chunk, end='')
                    SYNC.push({
                        "type": "llm_stream",
                        "data": chunk
                    })
                except StopIteration as e:
                    result = e.value
                    tts_queue.put(result['full_content'])
                    MEMORY.add_memory(result['full_content'], user_id="Airis")
                    CONTROLLER.emotion_policy()
                    break

            if INTERRUPT.is_interrupted():
                break

            tool_calls = result.get("tool_calls") or []

            if len(tool_calls) == 0:
                break

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                        }
                    }
                    for tc in tool_calls
                ]
            })

            for tool_call in tool_calls:
                try:
                    output = TOOLS.run_tool(tool_call)
                except Exception as e:
                    output = f"工具执行失败:{e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(output)
                })

                SYNC.push({
                    "type": "tool_call",
                    "data": tool_call["name"]
                })

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        TTS.stream_tts(text)

async def main_loop():
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        CONFIG = ConfigManager()
        LOGGER = LogManager("system")

        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryManager()
        TOOLS = ToolRegistry()

        INTERRUPT = InterruptManager()
        CONTROLLER = BehaviorController(LLM)
        SYNC = StateSyncManager()

        llm_queue = queue.Queue()
        tts_queue = queue.Queue()

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(SYNC.run()), daemon=True).start()

        asyncio.run(main_loop())  # 等效空循环

    except KeyboardInterrupt:
        exit()