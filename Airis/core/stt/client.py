import whisper
import pyaudio
import numpy as np
from typing import Optional

from ..common.config import ConfigManager   #配置管理器
from ..common.logger import LogManager      #日志管理器

# ===============================
# STT客户端
# ===============================
class STTClient:
    def __init__(self):
        self.config = ConfigManager()
        self.logger = LogManager("stt").get_logger()

        self.model = self._init_client()

        # 音频参数（从配置读取，提供默认值）
        self.sample_rate = self.config.get_json("stt.sample_rate", 16000)
        self.channels = self.config.get_json("stt.channels", 1)
        self.chunk_size = self.config.get_json("stt.chunk_size", 1024)
        self.format = pyaudio.paInt16
        self.silence_threshold = self.config.get_json("stt.silence_threshold", 500)   # RMS 能量阈值
        self.silence_timeout = self.config.get_json("stt.silence_timeout", 1.0)       # 静音超时（秒）

        self.language = self.config.get_json("stt.language")
        self.prompt=self.config.get_json("stt.prompt")
        self.temperature = self.config.get_json("stt.temperature")
        self.best_of = self.config.get_json("stt.best_of")
        self.beam_size = self.config.get_json("stt.beam_size")
        #PyAudio实例
        self.p = pyaudio.PyAudio()

        self.stream = None
        self.detect_chunk = None

    def _init_client(self) -> whisper.Whisper:
        model_name = self.config.get_json("stt.model")
        models_dir = self.config.get_path("stt.models_dir")
        try:
            model=whisper.load_model(model_name, download_root=str(models_dir))
        except Exception as e:
            raise ValueError(f"whisper初始化错误:{e}")
        return model

    def _is_silence(self, data: bytes) -> bool:
        """基于RMS能量的静音检测"""
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        return rms < self.silence_threshold

    def detect(self) -> bool:
        """
        录制音频：等待声音出现
        :return: 返回bool: 是否检测到音频
        """
        self.detect_chunk = None
        if self.stream:
            self.stream.close()

        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        while True:
            try:
                self.detect_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
            except IOError:
                continue
            if not self._is_silence(self.detect_chunk):
                return True

    def _record_until_silence(self) -> Optional[np.ndarray]:
        """
        录制音频：先等待声音出现，然后录音直到连续静音超过 silence_timeout 秒。
        返回归一化的 float32 音频数组，若未捕获到有效语音则返回 None。
        """
        if not self.stream or self.detect_chunk is None:
            raise RuntimeError("必须先调用 detect() 方法")

        print("\r开始录音...",end='')

        #录音直到连续静音超时
        frames = [self.detect_chunk]
        silent_chunks = 0
        # 静音阈值对应的块数
        silence_limit = int((self.silence_timeout * self.sample_rate) / self.chunk_size)

        while True:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            except IOError:
                continue
            frames.append(data)

            if self._is_silence(data):
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks > silence_limit:
                print(f"\r检测到{self.silence_timeout}秒静音,录音结束.",end='')
                break

        self.stream.stop_stream()
        self.stream.close()

        if not frames:
            return None

        #将字节数据转为归一化的float32数组
        audio_bytes = b''.join(frames)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

    def listen_and_transcribe(self) -> Optional[str]:
        """
        主入口:录音->转写->返回文本。
        若未检测到有效语音或转写失败则返回 None。
        """
        audio = self._record_until_silence()
        if audio is None or len(audio) == 0:
            print("\r未捕获到有效语音",end='')
            return None

        print("\r正在转写...",end='')
        # 直接传入numpy数组
        result = self.model.transcribe(
            audio=audio,
            fp16=False,
            language=self.language,
            prompt = self.prompt,
            temperature = self.temperature,
            best_of = self.best_of,
            beam_size = self.beam_size
        )   #type:ignore
        text = result["text"].strip()
        if text:
            return text
        else:
            return None

    def __del__(self):
        """释放 PyAudio 资源"""
        if hasattr(self, 'p'):
            self.p.terminate()
