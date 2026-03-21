import time
from core.tts.client import TTSClient
from core.stt.client import STTClient
from core.llm.client import LLMClient
import pyaudio
import wave
import threading
import os

def play_tts_audio():
    """在一个新线程中播放tts音频，实现非阻塞"""
    def _play():
        i=0
        while True:
            try:
                wf = wave.open(rf'.\Files\cache\tts\output_{i}.wav', 'rb')
            except FileNotFoundError:
                break
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            os.remove(rf'.\Files\cache\tts\output_{i}.wav')
            i+=1

    thread = threading.Thread(target=_play)
    thread.start()
    return thread #返回线程对象

ENABLE_STT = True

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
            TTS.text_to_speech(text)
            play_tts_audio()

    except KeyboardInterrupt:
        exit()