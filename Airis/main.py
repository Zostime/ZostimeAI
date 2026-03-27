import time
import json

from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager
from core.tools.registry import ToolRegistry

ENABLE_STT = False
USER = "Zostime"

ENABLE_TOOLS = True    #某些LLM不支持tool_calls则设为False

if __name__ == '__main__':
    try:
        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryManager()
        TOOLS = ToolRegistry()

        while True:
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

            user_conversations = []
            for entry in user_search.get("results", []):
                if entry is None:
                    continue
                user_conversations.append(f"- {entry['memory']}")

            assistant_conversations = []
            for entry in llm_search.get("results", []):
                if entry is None:
                    continue
                assistant_conversations.append(f"- {entry['memory']}")

            system_memory=(
                "用户历史对话:" + "|".join(user_conversations[:5]) +
                "助手历史对话:" + "|".join(assistant_conversations[:5])
            )
            messages = [
                {"role": "system", "content": f"记忆上下文:{system_memory}"},
                {"role": "user", "name": USER, "content": user_input}
            ]
            text=""
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
                    except StopIteration as e:
                        result = e.value
                        TTS.stream_tts(result['full_content'])
                        text += result['full_content']
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