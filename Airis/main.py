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
from src.output import OutputManager

# 配置
USER = "Zostime"
ENABLE_STT = False
ENABLE_TOOLS = True  # 某些 LLM 不支持 tool_calls 则设为 False

def write_stt_status(message: str, end: str = '') -> None:
    message = message.lstrip("\r")
    OUTPUT.write("stdout", f"\r\x1b[2K{message}", end=end)

def emit_tts_state(state: str) -> None:
    ProtocolRouter.Runtime.emit(
        event=f"tts/{state}",
        data={"state": state}
    )

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
        OUTPUT.write("stdout", "")
        if ENABLE_STT:
            while True:
                write_stt_status("开始录音...")
                user_input = STT.listen_and_transcribe()
                if user_input is not None:
                    write_stt_status(f"{USER}:{user_input}", end='\n')
                    break
                else:
                    write_stt_status("未识别到音频")
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

def build_conversation_messages(task):
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

    messages = [{"role": "system", "content": system_prompt}]
    user_stm = MEMORY.search_stm(USER)
    assistant_stm = MEMORY.search_stm("Airis")
    user_stm = sorted(user_stm, key=lambda item: item.get("timestamp", 0))
    assistant_stm = sorted(
        assistant_stm,
        key=lambda item: item.get("timestamp", 0)
    )

    for user_entry, assistant_entry in zip(user_stm, assistant_stm):
        messages.extend([
            {
                "role": "user",
                "name": USER,
                "content": user_entry["memory"]
            },
            {"role": "assistant", "content": assistant_entry["memory"]}
        ])

    messages.append({
        "role": "user",
        "name": task["source"],
        "content": task["content"]
    })
    return messages

def collect_tools():
    tools = []
    tool_map = {}
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
            forced_tools.append(f"{session.client_id}_{name}")

    tool_choice = "required" if forced_tools else "auto"
    return tools, tool_map, tool_choice

def stream_llm_response(messages, tools, tool_choice, task):
    generator = LLM.chat_stream(
        messages=messages,
        tools=tools if ENABLE_TOOLS else None,
        tool_choice=tool_choice
    )
    buffer = ""
    sentence_endings = (".", "。", "!", "！", "?", "？", "\n")

    while True:
        if INTERRUPT.is_interrupted():
            generator.close()
            return None

        try:
            chunk = next(generator)
        except StopIteration as stop:
            result = stop.value
            TTS.feed(buffer)
            if not task["ephemeral_context"]:
                MEMORY.add_memory(result["full_content"], user_id="Airis")
                MEMORY.add_memory(task["content"], user_id=task["source"])
            return result

        buffer += chunk
        if any(ending in buffer for ending in sentence_endings):
            TTS.feed(buffer)
            buffer = ""

        ProtocolRouter.Runtime.emit(
            event="llm/stream", data={"chunk": chunk}
        )

def append_assistant_tool_message(messages, result, tool_calls):
    messages.append({
        "role": "assistant",
        "content": result.get("full_content"),
        "tool_calls": [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["arguments"])
                }
            }
            for tool_call in tool_calls
        ]
    })

def execute_tool_calls(tool_calls, tool_map, messages):
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
            output = result.get("message", "")
        except Exception as error:
            output = f"工具执行失败: {error}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(output)
        })
        ProtocolRouter.Runtime.emit(
            event="llm/tool_call",
            data={"name": tool_call["name"]}
        )

def process_llm_task(task):
    messages = build_conversation_messages(task)

    try:
        while True:
            tools, tool_map, tool_choice = collect_tools()
            result = stream_llm_response(messages, tools, tool_choice, task)
            if result is None or INTERRUPT.is_interrupted():
                break

            tool_calls = result.get("tool_calls") or []
            if not tool_calls:
                break

            append_assistant_tool_message(messages, result, tool_calls)
            execute_tool_calls(tool_calls, tool_map, messages)

            if INTERRUPT.is_interrupted():
                break

            TTS.finished()
            TTS.player_idle.wait()
    except Exception as error:
        TTS.feed(f"Someone tell {USER} there is a problem with my AI.")
        runtime.LOGGER.logger.error(
            f"处理 LLM 请求时发生未知错误: {error}", exc_info=True
        )
    finally:
        runtime.STATE.env.is_speaking = False
        runtime.STATE.agent.unread_events.clear()
        TTS.finished()
        TTS.player_idle.wait()

def llm_worker():
    while True:
        task = llm_queue.get()
        if task is None:
            break
        process_llm_task(task)

def tts_worker():
    while True:
        word = TTS.subtitle_queue.get()
        OUTPUT.write("stdout", word, end='')

if __name__ == '__main__':
    llm_queue = queue.Queue()

    try:
        runtime.CONFIG = ConfigManager()
        runtime.LOGGER = LogManager("system")

        OUTPUT = OutputManager()
        OUTPUT.register("stdout", enabled=True)

        LLM = LLMClient()
        TTS = TTSClient(state_callback=emit_tts_state)
        STT = STTClient(
            status_callback=write_stt_status
        )
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
