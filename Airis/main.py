from websockets.server import WebSocketServerProtocol # noqa
from typing import Any, Literal, Optional, Dict
from collections import defaultdict
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

from core.common.config import ConfigManager
from core.common.logger import LogManager

#配置
USER = "Zostime"
ENABLE_STT = False
ENABLE_TOOLS = True    # 某些LLM不支持tool_calls则设为False

class State:
    class Agent:
        def __init__(self):
            self.memory: str = "无相关记忆"
            self.is_silent: bool = True
            self.unread_events: list = []

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
        STATE.agent.memory=build_memory_context(data['input'])

        STATE.env.input = {
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

class EventBus:
    def __init__(self):
        self.clients = defaultdict(dict)
        self.queue = asyncio.Queue()
        self.handlers = {}
        self.loop = None

    def on(self, path: str, handler):
        self.handlers[path] = handler

    async def run(self):
        self.loop = asyncio.get_running_loop()
        websocket_port = CONFIG.get_json("system.websocket_port")
        async with websockets.serve(self._connection_handler, "localhost", websocket_port):
            await self._sender_loop()

    async def _connection_handler(self, websocket: WebSocketServerProtocol):
        path = websocket.path.lstrip("/")
        client_id = id(websocket)
        self.clients[path][client_id] = websocket

        try:
            if path in self.handlers:
                await self.handlers[path](websocket, client_id)
        except websockets.ConnectionClosed:
            LOGGER.logger.debug(f"[websocket] {client_id} -> {path} connection closed.")
        except Exception:  # noqa
            pass
        finally:
            self.clients[path].pop(client_id, None)

    async def publish_async(self, path: str, data: Any, client_id: int = None):
        await self.queue.put((path, client_id, data))

    def publish(self, path: str, data: Any, client_id: int = None):
        if self.loop is None:
            raise RuntimeError("Event loop not ready")
        asyncio.run_coroutine_threadsafe(
            self.queue.put((path, client_id, data)),
            self.loop
        )

    async def _sender_loop(self):
        while True:
            path, client_id, data = await self.queue.get()

            targets = []

            if client_id is not None:
                ws = self.clients.get(path, {}).get(client_id)
                if ws:
                    targets.append(ws)
            else:
                targets = list(self.clients.get(path, {}).values())

            msg = json.dumps(data)

            for ws in targets:
                try:
                    await ws.send(msg)
                except:  # noqa
                    pass

class EventRouter:
    class State:
        @staticmethod
        async def handle(websocket: WebSocketServerProtocol, client_id: int): # noqa
            async for msg in websocket: pass # noqa

        @staticmethod
        def emit(data: Any):
            EVENT_BUS.publish(
                path = "state",
                data = data
            )

    class Game:
        sessions: Dict[int, 'EventRouter.Game.Session'] = {}  # client_id -> Session

        class Session:
            def __init__(self, client_id: int, websocket: WebSocketServerProtocol):
                self.client_id = client_id
                self.websocket = websocket
                self.game_name: Optional[str] = None
                self.registered_actions: dict = {}
                self.pending_action_id: Optional[str] = None
                self.pending_actions: Dict[str, asyncio.Future] = {}
                self.forced_action_names: list = []
                self.force_payload: Optional[dict] = None

        @staticmethod
        async def handle(websocket: WebSocketServerProtocol, client_id: int):
            session = EventRouter.Game.Session(client_id, websocket)
            EventRouter.Game.sessions[client_id] = session

            try:
                async for raw_msg in websocket:
                    try:
                        data = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        LOGGER.logger.warning(f"Invalid JSON: {raw_msg}")
                        continue
                    await EventRouter.Game._handle_message(session, data)
            except websockets.ConnectionClosed:
                pass
            finally:
                EventRouter.Game.sessions.pop(client_id, None)

        @staticmethod
        async def _handle_message(session, message):
            command = message.get("command")
            data = message.get("data")
            game = message.get("game")

            if command == "startup":
                session.game_name = game

            elif command == "context":
                message = data.get("message", "")
                silent = data.get("silent", True)
                if silent:
                    STATE.agent.unread_events.append(f"来自{game}的msg:{message}")
                else:
                    STATE.agent.unread_events.append(f"[应该回复]来自{game}的msg:{message}")

            elif command == "actions/register":
                for act in data.get("actions", []):
                    session.registered_actions[act["name"]] = act

            elif command == "actions/unregister":
                for name in data.get("action_names", []):
                    session.registered_actions.pop(name, None)

            elif command == "actions/force":
                query = data.get("query", "")
                action_names = data.get("action_names", [])
                state = data.get("state", "")
                priority = data.get("priority", "low")
                ephemeral_context = data.get("ephemeral_context", True)

                session.forced_action_names = action_names
                if not ephemeral_context:
                    session.force_payload = {
                        "query": query,
                        "state": state
                    }
                    STATE.agent.unread_events.append(f"来自{game}的[force]:{session.force_payload}")

                EVENT.add_event(
                    event_type="input",
                    data={
                        "source": game,
                        "input": query
                    },
                    priority=priority
                )   # 触发LLM

            elif command == "action/result":
                action_id = data.get("id")
                if not action_id:
                    return
                future = session.pending_actions.pop(action_id, None)
                if future and not future.done():
                    future.set_result(data)

        @staticmethod
        async def run_action(session, data):
            action_id = data["data"]["id"]

            loop = asyncio.get_running_loop()
            future = loop.create_future()

            session.pending_actions[action_id] = future

            await EVENT_BUS.publish_async(
                path="game",
                client_id=session.client_id,
                data=data
            )

            return await future

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

def build_memory_context(user_input) -> str:
    user_ltm = MEMORY.search_ltm(user_input,USER)    # [{'memory': str, 'score': datetime}, ...]
    assistant_ltm = MEMORY.search_ltm(user_input, 'Airis')
    user_stm = MEMORY.search_stm(USER)               # [{"memory": str, "timestamp": datetime}, ...]
    assistant_stm = MEMORY.search_stm('Airis')

    note = MEMORY.note.read()

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

    if note:
        parts.append("\nNOTE:")
        parts.append(note)

    if len(parts) == 1:
        return "没有相关记忆"

    return "\n".join(parts)

def handle_user_input():
    STATE.env.is_speaking = True
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
        system_prompt = PROMPT.build({
            "system.md": {},
            "personality.md": {},
            "memory.md": {
                "memory": STATE.agent.memory,
            },
            "runtime_state.md": {
                "current_time": datetime.datetime.now(),
                "unread_events": STATE.agent.unread_events,
            }
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "name": task['source'], "content": task['input']}
        ]
        result = None
        STATE.agent.is_silent = False
        try:
            while True:
                tools = []
                tool_map = {}
                tool_choice = "auto"
                forced_tools = []

                for session in EventRouter.Game.sessions.values():
                    for action in session.registered_actions.values():
                        safe_name = f"{session.client_id}_{action['name']}"

                        tools.append({
                            "type": "function",
                            "function": {
                                "name": safe_name,
                                "description": action["description"],
                                "parameters": action.get("schema") or {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        })

                        tool_map[safe_name] = (session, action["name"])

                    for name in session.forced_action_names:
                        safe_name = f"{session.client_id}_{name}"
                        forced_tools.append(safe_name)

                if forced_tools:
                    tool_choice = "required"

                gen = LLM.chat_stream(
                    messages=messages,
                    tools=tools if ENABLE_TOOLS else None,
                    tool_choice = tool_choice
                )
                buf = ""
                chars = ['.', '。', '!', '！', '?', '？', '\n']
                while True:
                    try:
                        if INTERRUPT.is_interrupted():
                            gen.close()
                            break

                        chunk = next(gen)
                        buf += chunk

                        if any(_ in buf for _ in chars):
                            TTS.stream_feed(buf)
                            buf = ""

                        EventRouter.State.emit(
                            data={
                                "type": "llm_stream",
                                "data": chunk
                            }
                        )
                    except StopIteration as e:
                        result = e.value
                        TTS.stream_feed(buf)
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
                        session, real_name = tool_map[tool_call["name"]]

                        future = asyncio.run_coroutine_threadsafe(
                            EventRouter.Game.run_action(
                                session=session,
                                data={
                                    "command": "action",
                                    "data": {
                                        "id": tool_call["id"],
                                        "name": real_name,
                                        "data": json.dumps(tool_call.get("arguments", {}))
                                    }
                                }
                            ),
                            EVENT_BUS.loop
                        )
                        result = future.result()
                        if result.get("success"):
                            session.forced_action_names.clear()
                            session.force_payload = None
                        else:
                            pass
                        output = result.get("message", "")

                    except Exception as e:
                        output = f"工具执行失败: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(output)
                    })

                    EventRouter.State.emit(
                            data={
                            "type": "tool_call",
                            "data": tool_call["name"]
                        }
                    )
                if INTERRUPT.is_interrupted():
                    break

            MEMORY.add_memory(task['input'], user_id=task['source'])

        except Exception as e:
            print(f"Someone tell Zostime there is a problem with my AI.", flush=True)
            TTS.stream_feed("Someone tell Zostime there is a problem with my AI.")
            LOGGER.logger.error(f"处理 LLM 请求时发生未知错误: {e}", exc_info=True)
        finally:
            STATE.env.is_speaking = False
            STATE.agent.is_silent = True
            STATE.agent.unread_events = []  # 清空未读消息

def tts_worker():
    while True:
        try:
            word = TTS.subtitle_queue.get()
            print(word, end='', flush=True)
        except queue.Empty:
            pass

if __name__ == '__main__':
    llm_queue = queue.Queue()

    try:
        CONFIG = ConfigManager()
        LOGGER = LogManager("system")

        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryManager()
        PROMPT = PromptBuilder()

        STATE = State()
        EVENT = EventManager()
        EVENT.on("input", EventHandler.input_handler)
        EVENT.on("interrupt", EventHandler.interrupt_handler)

        INTERRUPT = InterruptManager()
        EVENT_BUS = EventBus()
        EVENT_BUS.on("state", EventRouter.State.handle)
        EVENT_BUS.on("game", EventRouter.Game.handle)

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(EVENT_BUS.run()), daemon=True).start()

        threading.Event().wait() # loop

    except KeyboardInterrupt:
        llm_queue.put(None)
        exit()