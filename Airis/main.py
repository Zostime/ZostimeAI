import time
from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient

ENABLE_STT = False

if __name__ == '__main__':
    try:
        LLM = LLMClient()
        TTS = TTSClient()
        STT = STTClient()
        while True:
            if ENABLE_STT:
                while True:
                    text=STT.listen_and_transcribe()
                    if text is not None:
                        print(f"\rUSER:{text}")
                        break
                    else:
                        print("\r未识别到音频",end='')
                        time.sleep(1)
            else:
                text=input("USER:")
                print()
            gen = LLM.chat_stream(
                messages=[
                    {"role": "system", "content": " "},
                    {"role": "assistant", "content": " "},
                    {"role": "user", "name": "Zostime", "content": text}
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

    except KeyboardInterrupt:
        exit()