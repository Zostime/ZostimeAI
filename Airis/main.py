import time
from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient
from core.memory.client import MemoryClient

ENABLE_STT = False
USER = "Zostime"

if __name__ == '__main__':
    try:
        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        MEMORY = MemoryClient()
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

            user_search = MEMORY.memory_search(user_input, USER, 20)
            llm_search = MEMORY.memory_search(user_input, "Airis", 20)

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
                "\n\n用户历史对话:\n" + "\n".join(user_conversations[:5]) +
                "\n\n助手历史对话:\n" + "\n".join(assistant_conversations[:5])
            )

            MEMORY.memory_add(user_input, user_id=USER)

            gen = LLM.chat_stream(
                messages=[
                    {"role": "system", "content": f"Memory:{system_memory}"},
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
            MEMORY.memory_add(text,user_id="Airis")

    except KeyboardInterrupt:
        exit()