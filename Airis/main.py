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
            user_memory = "\n".join(
                f"- {entry['memory']}"
                for entry in MEMORY.memory_search(user_input,USER,5)["results"]
            )
            llm_memory = "\n".join(
                f"- {entry['memory']}"
                for entry in MEMORY.memory_search(user_input,"Airis",5)["results"]
            )

            MEMORY.memory_add(user_input, user_id=USER)
            gen = LLM.chat_stream(
                messages=[
                    {"role": "system", "content": f"Memory:[User:{user_memory},Assistant:{llm_memory}]"},
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