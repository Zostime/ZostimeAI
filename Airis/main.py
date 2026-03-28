import threading
import keyboard
import time
import json
import queue
import sounddevice as sd
import numpy as np

from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager
from core.tools.registry import ToolRegistry

#配置
USER = "Zostime"
ENABLE_STT = False
ENABLE_TOOLS = True    #某些LLM不支持tool_calls则设为False

class InterruptManager:
    def __init__(self):
        self.event = threading.Event()
        threading.Thread(target=self.listener, daemon=True).start()

    def trigger(self):
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
                print("\n[Ctrl+F1 打断]")
                self.trigger()

if __name__ == '__main__':
    try:
        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryManager()

        TOOLS = ToolRegistry()
        INTERRUPT = InterruptManager()

        while True:
            INTERRUPT.clear()

            if ENABLE_STT:
                while True:
                    user_input=STT.listen_and_transcribe()
                    if user_input is not None:
                        print(f"\r{USER}:{user_input}")
                        break
                    else:
                        print("\r未识别到音频",end='')
                        time.sleep(1)
            else:
                user_input=input(f"{USER}:")
                print()

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

            if not memory_sections:
                system_memory = "暂无相关历史记忆."
            else:
                system_memory = "\n\n".join(memory_sections)

            messages = [
                {"role": "system", "content": f"记忆上下文:{system_memory}"},
                {"role": "user", "name": USER, "content": user_input}
            ]
            text=""
            result = None
            while True:
                gen = LLM.chat_stream(
                    messages=messages,
                    tools=TOOLS.get_tools() if ENABLE_TOOLS else None,
                    tool_choice="auto"
                )
                while True:
                    try:
                        chunk = next(gen)
                        print(chunk, end='')
                        text+=chunk
                        if INTERRUPT.is_interrupted():
                            gen.close()
                            break
                    except StopIteration as e:
                        result = e.value
                        #stream TTS
                        audio_queue = queue.Queue()
                        stop_event = threading.Event()
                        def play_worker():
                            with sd.OutputStream(samplerate=24000, channels=1, dtype=np.float32) as stream:
                                while True:
                                    if stop_event.is_set() and audio_queue.empty():
                                        break
                                    try:
                                        audio_data = audio_queue.get(timeout=0.1)
                                        stream.write(audio_data)
                                        audio_queue.task_done()
                                    except queue.Empty:
                                        continue
                                # 退出前清空剩余任务
                                while not audio_queue.empty():
                                    try:
                                        audio_data = audio_queue.get_nowait()
                                        stream.write(audio_data)
                                        audio_queue.task_done()
                                    except queue.Empty:
                                        break
                        play_thread = threading.Thread(target=play_worker, daemon=True)
                        play_thread.start()
                        generator = TTS.stream_tts(result['full_content'])
                        for i, (gs, ps, audio) in enumerate(generator):
                            if INTERRUPT.is_interrupted():
                                stop_event.set()
                                break
                            audio_queue.put(audio)
                        stop_event.set()
                        play_thread.join()
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
            print()

            MEMORY.add_memory(user_input, user_id=USER)
            MEMORY.add_memory(text,user_id="Airis")

    except KeyboardInterrupt:
        exit()