from typing import Any, Literal
import websockets
import threading
import datetime
import keyboard
import asyncio
import queue
import time
import json

from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager
from core.prompts.manager import PromptBuilder
from core.tools.registry import ToolRegistry

from core.common.config import ConfigManager
from core.common.logger import LogManager

#配置
USER = "Zostime"
ENABLE_STT = False
ENABLE_TOOLS = True    # 某些LLM不支持tool_calls则设为False
WEBSOCKET_PORT = 8090

class State:
    class Agent:
        def __init__(self):
            self.memory: str = "无相关记忆"
            self.is_silent: bool = True

    class Env:
        def __init__(self):
            self.is_speaking: bool = False
            self.input: dict = {
                "content": "",
                "source": "",
                "timestamp": None
            }

    def __init__(self):
        self.agent = State.Agent()
        self.env = State.Env()

        self._lock = threading.Lock()

    def update_memory(self, memory: dict):
        with self._lock:
            self.agent.memory = memory

class EventManager:
    PRIORITY_MAP = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }  # EVENT 优先级
    def __init__(self):
        self._event_queue = queue.PriorityQueue()
        threading.Thread(target=self.event_loop, daemon=True).start()
        self._handlers = {}
        self._lock = threading.Lock()

    def on(self, event_type: str, handler) -> None:
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def add_event(self,
                  event_type: str,
                  data: Any = None,
                  priority: Literal["low", "medium", "high", "critical"] = "low"
                  ) -> None:
        if priority not in self.PRIORITY_MAP:
            raise ValueError(f"未定义的EVENT优先级: {priority}")

        priority_val = self.PRIORITY_MAP[priority]
        timestamp = time.time()
        event = {
            "timestamp": timestamp,
            "priority": priority,
            "type": event_type,
            "data": data
        }
        # (优先级, 时间戳, event)
        self._event_queue.put((priority_val, timestamp, event))

    def event_loop(self):
        while True:
            _, _, event = self._event_queue.get()
            if event is None:
                break

            with self._lock:
                handlers = list(self._handlers.get(event["type"], [])) + \
                           list(self._handlers.get("*", []))

            for handler in handlers:
                try:
                    threading.Thread(
                        target=handler,
                        args=(event,),
                        daemon=True
                    ).start()
                except Exception as e:
                    LOGGER.logger.error(f"[Event Error] {event['type']} -> {e}")

class EventHandler:
    @staticmethod
    def input_handler(event):
        INTERRUPT.clear()
        data = event["data"]
        STATE.update_memory(build_memory_context(data['input']))

        STATE.env.input={
            "content": data['input'],
            "source": data['source'],
            "timestamp": time.time()
        }

        llm_queue.put({
            "source": USER,
            "input": STATE.env.input['content']
        })

    # noinspection PyUnusedLocal
    @staticmethod
    def interrupt_handler(event):
        TTS.interrupt()
        STATE.agent.is_silent = True

class InterruptManager:
    def __init__(self):
        self.event = threading.Event()
        threading.Thread(target=self.listener, daemon=True).start()

    def trigger(self):
        self.event.set()
        EVENT.add_event(
            event_type="interrupt",
            data=None,
            priority="critical",
        )

    def clear(self):
        self.event.clear()

    def is_interrupted(self):
        return self.event.is_set()

    def listener(self):
        while True:
            if ENABLE_STT:
                STT.detect()
                self.trigger()
                handle_user_input()
            else:
                keyboard.wait("ctrl+f1")
                self.trigger()
                handle_user_input()

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

def build_memory_context(user_input) -> str:
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

    return "\n".join(parts)

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
    STATE.env.is_speaking = True

    EVENT.add_event(
        event_type="input",
        data={
            "source": USER,
            "input": user_input,
        },
        priority="high"
    )

def llm_worker():
    while True:
        task = llm_queue.get()
        if task is None:
            break
        STATE.agent.is_silent = False

        system_prompt = PROMPT.build({
            "system.md": {},
            "personality.md": {},
            "memory.md": {
                "memory": STATE.agent.memory,
            },
            "runtime_state.md": {
                "current_time": datetime.datetime.now(),
                "unread_events": None
            }
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "name": task['source'], "content": task['input']}
        ]
        result = None
        try:
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
                        print(chunk, end='', flush=True)
                        SYNC.push({
                            "type": "llm_stream",
                            "data": chunk
                        })
                    except StopIteration as e:
                        result = e.value
                        tts_queue.put(result['full_content'])
                        MEMORY.add_memory(result['full_content'], user_id="Airis")
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
                    if INTERRUPT.is_interrupted():
                        break
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
                if INTERRUPT.is_interrupted():
                    break

            MEMORY.add_memory(task['input'], user_id=task['source'])

        except Exception as e:  # noqa
            print(f"Someone tell Zostime there is a problem with my AI.", flush=True)
            tts_queue.put("Someone tell Zostime there is a problem with my AI.")
            LOGGER.logger.error(f"处理请求时发生未知错误: {e}", exc_info=True)
        finally:
            STATE.env.is_speaking = False
            STATE.agent.is_silent = True

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        STATE.agent.is_silent = False
        TTS.stream_tts(text)
        STATE.agent.is_silent = True

async def main_loop():
    while True:
        await asyncio.Event().wait()

if __name__ == '__main__':
    llm_queue = None
    tts_queue = None
    try:
        CONFIG = ConfigManager()
        LOGGER = LogManager("system")

        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryManager()
        PROMPT = PromptBuilder()
        TOOLS = ToolRegistry()

        STATE = State()
        EVENT = EventManager()
        EVENT.on("input", EventHandler.input_handler)
        EVENT.on("interrupt", EventHandler.interrupt_handler)

        INTERRUPT = InterruptManager()
        SYNC = StateSyncManager()

        llm_queue = queue.Queue()
        tts_queue = queue.Queue()

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(SYNC.run()), daemon=True).start()

        asyncio.run(main_loop())

    except KeyboardInterrupt:
        if llm_queue is not None and tts_queue is not None:
            llm_queue.put(None)
            tts_queue.put(None)
        exit()