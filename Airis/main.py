import threading
import datetime
import keyboard
import queue
import time
import json

from src.core.prompts.manager import PromptBuilder
from src.core.memory.manager import MemoryManager
from src.core.common.config import ConfigManager
from src.core.common.logger import LogManager
from src.core.tts.client import TTSClient
from src.core.stt.client import STTClient
from src.core.llm.client import LLMClient

import src.runtime as runtime
from src.gateway import ProtocolRouter
from src.event_bus import EventBus
from src.state import State

# 配置
USER = "Zostime"
ENABLE_STT = False
ENABLE_TOOLS = True  # 某些 LLM 不支持 tool_calls 则设为 False

class EventHandlers:
    @staticmethod
    def input_handler(event):
        INTERRUPT.clear()
        data = event["data"]
        runtime.STATE.agent.memory = build_memory_context(data['content'])

        runtime.STATE.env.input = {
            "content": data['content'],
            "source": data['source'],
            "ephemeral_context": data['ephemeral_context'],
            "timestamp": time.time()
        }

        llm_queue.put({
            "source": data['source'],
            "content": data['content'],
            "ephemeral_context": data['ephemeral_context']
        })

    # noinspection PyUnusedLocal
    @staticmethod
    def interrupt_handler(event):
        TTS.interrupt()

class InterruptManager:
    def __init__(self):
        self.event = threading.Event()
        threading.Thread(target=self.listener, daemon=True).start()

    def trigger(self):
        self.event.set()
        runtime.EVENT_BUS.emit(
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
                self.handle_user_input()
            else:
                keyboard.wait("ctrl+f1")
                self.trigger()
                self.handle_user_input()

    @staticmethod
    def handle_user_input():
        runtime.STATE.env.is_speaking = True
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

        runtime.EVENT_BUS.emit(
            event_type="input",
            data={
                "source": USER,
                "content": user_input,
                "ephemeral_context": False,
            },
            priority="high"
        )

def build_memory_context(user_input) -> str:
    user_ltm = MEMORY.search_ltm(user_input, USER)  # [{'memory': str, 'score': datetime}, ...]
    assistant_ltm = MEMORY.search_ltm(user_input, 'Airis')

    note = MEMORY.note.read()

    # LTM排序(相关性高在前)
    user_ltm = sorted(user_ltm, key=lambda x: x.get("score", 0.5), reverse=True)
    assistant_ltm = sorted(assistant_ltm, key=lambda x: x.get("score", 0.5), reverse=True)

    parts = ["[记忆上下文]"]

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

def llm_worker():
    while True:
        task = llm_queue.get()
        if task is None:
            break

        source = task["source"]
        content = task["content"]
        ephemeral_context = task["ephemeral_context"]

        system_prompt = PROMPT.build({
            "system.md": {},
            "personality.md": {},
            "memory.md": {
                "memory": runtime.STATE.agent.memory,
            },
            "runtime_state.md": {
                "current_time": datetime.datetime.now(),
                "unread_events": runtime.STATE.agent.unread_events,
            }
        })

        messages = [{"role": "system", "content": system_prompt}] # 预备 messages

        # STM排序(最新在前)
        user_stm = MEMORY.search_stm(USER)  # [{"memory": str, "timestamp": datetime}, ...]
        assistant_stm = MEMORY.search_stm('Airis')
        user_stm = sorted(user_stm, key=lambda x: x.get("timestamp", 0))
        assistant_stm = sorted(assistant_stm, key=lambda x: x.get("timestamp", 0))
        for user, assistant in zip(user_stm, assistant_stm):
            messages.extend([
                {"role": "user", "name": USER, "content": user["memory"]},
                {"role": "assistant", "content": assistant["memory"]}
            ])

        messages.append({
            "role": "user",
            "name": source,
            "content": content
        })

        result = None
        try:
            while True:
                tools = []
                tool_map = {}
                tool_choice = "auto"
                forced_tools = []

                for session in ProtocolRouter.Game.sessions.values():
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
                    tool_choice=tool_choice
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
                            TTS.feed(buf)
                            buf = ""

                        ProtocolRouter.Runtime.emit(
                            event="llm/stream",
                            data={
                                "chunk": chunk
                            }
                        )

                    except StopIteration as e:
                        result = e.value
                        TTS.feed(buf)
                        if not ephemeral_context:
                            MEMORY.add_memory(result['full_content'], user_id="Airis")
                            MEMORY.add_memory(content, user_id=source)
                        break

                if INTERRUPT.is_interrupted():
                    break

                tool_calls = result.get("tool_calls") or []

                if len(tool_calls) == 0:
                    break

                messages.append({
                    "role": "assistant",
                    "content": result.get("full_content"),
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

                        result = ProtocolRouter.Game.run_action(
                            session=session,
                            data={
                                "command": "action",
                                "data": {
                                    "id": tool_call["id"],
                                    "name": real_name,
                                    "data": json.dumps(tool_call.get("arguments", {}))
                                }
                            }
                        )

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

                    ProtocolRouter.Runtime.emit(
                        event="llm/tool_call",
                        data={
                            "name": tool_call["name"]
                        }
                    )

                if INTERRUPT.is_interrupted():
                    break

        except Exception as e:
            TTS.feed(f"Someone tell {USER} there is a problem with my AI.")
            runtime.LOGGER.logger.error(f"处理 LLM 请求时发生未知错误: {e}", exc_info=True)
        finally:
            runtime.STATE.env.is_speaking = False
            runtime.STATE.agent.unread_events.clear()  # 清空未读消息

def tts_worker():
    while True:
        word = TTS.subtitle_queue.get()
        print(word, end='', flush=True)

if __name__ == '__main__':
    llm_queue = queue.Queue()

    try:
        runtime.CONFIG = ConfigManager()
        runtime.LOGGER = LogManager("system")

        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryManager()
        PROMPT = PromptBuilder()

        runtime.STATE = State()
        runtime.EVENT_BUS = EventBus()
        runtime.EVENT_BUS.on("input", EventHandlers.input_handler)
        runtime.EVENT_BUS.on("interrupt", EventHandlers.interrupt_handler)

        INTERRUPT = InterruptManager()
        ProtocolRouter.setup()

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()

        runtime.LOGGER.logger.info("初始化完成")
        threading.Event().wait()  # loop

    except KeyboardInterrupt:
        llm_queue.put(None)
        exit()
