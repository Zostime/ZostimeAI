import threading
import datetime
import keyboard
import queue
import time
import json
import sys
from html.parser import HTMLParser

from src.core.prompts.manager import PromptBuilder
from src.core.memory.manager import MemoryManager
from src.core.common.config import ConfigManager
from src.core.common.logger import LogManager
from src.core.tts.client import TTSClient
from src.core.stt.client import STTClient
from src.core.llm.client import LLMClient

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
        data={}
    )

class EventHandlers:
    @staticmethod
    def input_handler(event):
        data = event["data"]
        if data.get("type") == "force_action":
            action_names = data.get("action_names")
            if (
                not data.get("session_id")
                or not isinstance(action_names, list)
                or not action_names
                or any(not isinstance(name, str) or not name.strip() for name in action_names)
            ):
                LOGGER.logger.error(
                    "拒绝无效的 force_action 输入: 必须提供 session_id 和非空 action_names"
                )
                return

        INTERRUPT.clear()
        STATE.agent.memory = AGENT.build_memory_context(data['content'])

        STATE.env.input = {
            "type": data['type'],
            "source": data['source'],
            "content": data['content'],
            "ephemeral_context": data['ephemeral_context'],
            "timestamp": time.time()
        }

        task = {
            "type": data['type'],
            "source": data['source'],
            "content": data['content'],
            "ephemeral_context": data['ephemeral_context'],
            "priority": event['priority']
        }
        if data['type'] == "force_action":
            task["session_id"] = data["session_id"]
            task["action_names"] = data.get("action_names", [])
        llm_queue.put(task)

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
        EVENT_BUS.emit(
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
        STATE.env.is_speaking = True
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

        EVENT_BUS.emit(
            event_type="input",
            data={
                "type": "user_input",
                "source": USER,
                "content": user_input,
                "ephemeral_context": False,
            },
            priority="high"
        )

class Agent:
    def __init__(self):
        self.memory_filter_ssml = True
        self.pending_inputs = queue.Queue()
        self.last_env = {
            "pending_inputs": list(self.pending_inputs.queue),
            "unread_events": STATE.agent.unread_events.copy()
        }

        self.llm_task_lock = threading.Lock()
        
        threading.Thread(target=self.environmental_assessment, daemon=True).start()

    def filter_memory_content(self, content: str) -> str:
        filtered_content = content
        if self.memory_filter_ssml:
            text_parts = []
            parser = HTMLParser(convert_charrefs=True)
            parser.handle_data = text_parts.append
            parser.feed(content)
            parser.close()
            filtered_content = "".join(text_parts)
        return filtered_content

    def main(self, tasks):
        """接受 tasks 并处理 LLM + func_call 请求"""
        with self.llm_task_lock:
            if isinstance(tasks, dict):
                tasks = [tasks]

            STATE.agent.idle = False
            try:
                messages = self.build_conversation_messages(tasks)
                force_action_pending = any(
                    task.get("type") == "force_action" for task in tasks
                )
                while True:
                    tools, tool_map, tool_choice = self.collect_tools(
                        tasks, force_action_pending
                    )
                    result = self.stream_llm_response(messages, tools, tool_choice, tasks)
                    if result is None or INTERRUPT.is_interrupted():
                        break

                    tool_calls = result.get("tool_calls") or []
                    if not tool_calls:
                        break

                    self.append_assistant_tool_message(messages, result, tool_calls)
                    if self.execute_tool_calls(tool_calls, tool_map, messages):
                        force_action_pending = False

                    if INTERRUPT.is_interrupted():
                        break

                    TTS.finished()
                    TTS.player_idle.wait()
            except Exception as error:
                TTS.feed(f"Someone tell {USER} there is a problem with my AI.")
                LOGGER.logger.error(
                    f"处理 LLM 请求时发生未知错误: {error}", exc_info=True
                )
            finally:
                STATE.env.is_speaking = False
                try:
                    STATE.agent.unread_events.clear()
                    TTS.finished()
                    TTS.player_idle.wait()
                finally:
                    STATE.agent.idle = True

    def environmental_assessment(self):
        """积累 env 数据, Main LLM 处于空闲时 由 Environmental Assessment LLM 判断是否触发 Main LLM"""
        while True:
            env = {
                "pending_inputs": list(self.pending_inputs.queue),
                "unread_events": STATE.agent.unread_events.copy()
            }

            env_changed = (env != self.last_env)
            if STATE.agent.idle and env_changed:
                ProtocolRouter.Runtime.emit(
                    event="debug/info", data=f"env has changed"
                )
                
                response = LLM.chat(
                    messages=[
                        {
                            "role": "system", 
                            "content": PROMPT.build(
                             profile="environmental_assessment", 
                             context={"environmental_assessment.md": {}}
                            )
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                env,
                                ensure_ascii=False
                            )
                        }
                    ]
                )
                if response["full_content"].strip() == "IGNORE":
                    self.last_env = env
                    continue

                ProtocolRouter.Runtime.emit(
                    event="debug/info", data=f"triggle main LLM"
                )

                tasks = []
                for _ in range(len(env["pending_inputs"])):
                    try:
                        tasks.append(self.pending_inputs.get_nowait())
                    except queue.Empty:
                        break

                if not tasks:
                    tasks = [{
                        "source": "Airis",
                        "content": "",
                        "ephemeral_context": False
                    }]

                self.main(tasks)
                self.last_env = {
                    "pending_inputs": [],
                    "unread_events": [],
                }
            else:
                time.sleep(0.1)

    def build_memory_context(self, user_input) -> str:
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

    def build_conversation_messages(self, tasks):
        system_prompt = PROMPT.build(profile="main", context={
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

        for task in tasks:
            messages.append({
                "role": "user",
                "name": task["source"],
                "content": task["content"]
            })
        return messages

    def collect_tools(self, tasks, force_action_pending):
        tools = []
        tool_map = {}
        forced_actions = {}

        for task_index, task in enumerate(tasks):
            if task.get("type") == "force_action":
                session_id = task.get("session_id")
                action_names = task.get("action_names") or []
                if session_id is None or not action_names:
                    raise ValueError(
                        "force_action task must include session_id and action_names "
                        f"(task_index={task_index}, source={task.get('source')!r}, "
                        f"session_id={session_id!r}, "
                        f"action_names={task.get('action_names')!r})"
                    )
                forced_actions.setdefault(session_id, set()).update(action_names)

        for session in ProtocolRouter.Game.sessions.values():
            for action in session.registered_actions.values():
                allowed_names = forced_actions.get(session.client_id)
                if forced_actions and (
                    allowed_names is None or action["name"] not in allowed_names
                ):
                    continue

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

        if forced_actions and not tools:
            raise ValueError("force_action task has no registered action names")

        tool_choice = (
            "required" if forced_actions and force_action_pending else "auto"
        )
        return tools, tool_map, tool_choice

    def stream_llm_response(self, messages, tools, tool_choice, tasks):
        sources = [task["source"] for task in tasks]
        stream_data = {"source": sources[0]}
        if len(sources) > 1:
            stream_data["sources"] = sources
        response = ""

        ProtocolRouter.Runtime.emit(
            event="llm/stream_start",
            data=stream_data
        )
        try:
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
                    if not result.get("tool_calls"):
                        persistent_tasks = [
                            task for task in tasks
                            if not task["ephemeral_context"]
                        ]
                        if persistent_tasks:
                            if len(persistent_tasks) == len(tasks):
                                MEMORY.add_memory(
                                    self.filter_memory_content(
                                        result["full_content"]
                                    ),
                                    user_id="Airis"
                                )
                            for task in persistent_tasks:
                                MEMORY.add_memory(
                                    task["content"], user_id=task["source"]
                                )
                    return result

                buffer += chunk
                if any(ending in buffer for ending in sentence_endings):
                    TTS.feed(buffer)
                    response += buffer
                    buffer = ""

                ProtocolRouter.Runtime.emit(
                    event="llm/stream", data={"chunk": chunk}
                )
        finally:
            ProtocolRouter.Runtime.emit(
                event="llm/stream_end",
                data={"sources": stream_data, "response": response}
            )

    def append_assistant_tool_message(self, messages, result, tool_calls):
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

    def execute_tool_calls(self, tool_calls, tool_map, messages):
        action_succeeded = False
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
                    action_succeeded = True
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
        return action_succeeded

def llm_worker():
    while True:
        task = llm_queue.get()
        if task is None:
            break
        
        if task['type'] == "force_action":  # 不经过 Environmental Assessment LLM
            priority = task['priority']

            tasks = []  # tasks 组装
            while True:
                try:
                    tasks.append(AGENT.pending_inputs.get_nowait())
                except queue.Empty:
                    break
            tasks.append(task)

            if TTS.player_idle.is_set():
                pass

            elif priority == "low":
                TTS.player_idle.wait()

            elif priority == "medium":
                TTS.sentence_boundary_idle.wait()
                TTS.interrupt()

            elif priority == "high":
                TTS.word_boundary_idle.wait()
                TTS.interrupt()

            elif priority == "critical":
                TTS.interrupt()
                
            AGENT.main(tasks)

        else:
            AGENT.pending_inputs.put(task)

def tts_worker():
    text_parts = []
    parser = HTMLParser(convert_charrefs=True)
    parser.handle_data = text_parts.append
    while True:
        word = TTS.subtitle_queue.get()
        OUTPUT.write("stderr", word, end='')
        parser.feed(word)
        plain_text = "".join(text_parts)
        text_parts.clear()
        if plain_text:
            OUTPUT.write("stdout", plain_text, end='')

if __name__ == '__main__':
    llm_queue = queue.Queue()

    try:
        CONFIG = ConfigManager()
        LOGGER = LogManager("system")
        
        OUTPUT = OutputManager()
        OUTPUT.register("stdout", enabled=True)
        OUTPUT.register("stderr", enabled=False, stream=sys.stderr)

        LLM = LLMClient()
        TTS = TTSClient(state_callback=emit_tts_state)
        STT = STTClient(status_callback=write_stt_status)
        MEMORY = MemoryManager()
        PROMPT = PromptBuilder()

        STATE = State()
        EVENT_BUS = EventBus(LOGGER.logger)
        EVENT_BUS.on("input", EventHandlers.input_handler)
        EVENT_BUS.on("interrupt", EventHandlers.interrupt_handler)

        INTERRUPT = InterruptManager()
        ProtocolRouter.setup(
            config=CONFIG,
            logger=LOGGER.logger,
            state=STATE,
            event_bus=EVENT_BUS,
        )

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()

        AGENT = Agent()

        LOGGER.logger.info("初始化完成")
        threading.Event().wait()  # loop

    except KeyboardInterrupt:
        llm_queue.put(None)
        exit()
