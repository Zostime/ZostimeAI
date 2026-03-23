import time
from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient
from core.memory.manager import MemoryManager

ENABLE_STT = False
USER = "Zostime"

if __name__ == '__main__':
    try:
        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryManager()
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

            gen = LLM.chat_stream(
                messages=[
                    {"role": "system", "content": f"记忆上下文:{system_memory}"},
                    {"role": "user", "name": USER, "content": user_input}
                ]
            )
            while True:
                try:
                    chunk = next(gen)
                    print(chunk, end='')
                except StopIteration as e:
                    print()
                    result = e.value
                    break
            text = result['full_content']
            TTS.stream_tts(text)

            MEMORY.add_memory(user_input, user_id=USER)
            MEMORY.add_memory(text,user_id="Airis")

    except KeyboardInterrupt:
        exit()