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
    user_search = MEMORY.search_memory(user_input, USER)
    llm_search = MEMORY.search_memory(user_input, "Airis")
    user_memories = []
    for entry in user_search.get("results", []):
        if entry is None:
            continue
        user_memories.append(f"- {entry['memory']}")
    assistant_memories = []
    for entry in llm_search.get("results", []):
        if entry is None:
            continue
        assistant_memories.append(f"- {entry['memory']}")
    memory_sections = []
    if user_memories:
        user_items = "\n".join(f"{i + 1}. {mem}" for i, mem in enumerate(user_memories))
        memory_sections.append(f"[用户历史记忆]\n{user_items}")
    if assistant_memories:
        assistant_items = "\n".join(f"{i + 1}. {mem}" for i, mem in enumerate(assistant_memories))
        memory_sections.append(f"[助手历史记忆]\n{assistant_items}")

    global SYSTEM_MEMORY
    if not memory_sections:
        SYSTEM_MEMORY = "暂无相关历史记忆."
    else:
        SYSTEM_MEMORY = "\n".join(memory_sections)

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
        messages = [
            {"role": "system", "content": f"情绪:{SYSTEM_EMOTION},记忆上下文:{SYSTEM_MEMORY}"},
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
                    MEMORY.add_memory(llm_input, user_id=USER)
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

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        exit()