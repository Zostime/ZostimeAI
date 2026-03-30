import queue
import threading
import numpy as np
import sounddevice as sd
from kokoro import KPipeline
import logging
import os

from ..common.config import ConfigManager   #配置管理器
from ..common.logger import LogManager      #日志管理器

# ===============================
# 缓存管理器
# ===============================
class CacheManager:
    def __init__(self, config: ConfigManager, service_name: str):
        """初始化缓存管理器
        Args:
            config: 配置管理器实例
            service_name: 服务名称,用于缓存目录命名
        """
        self.config = config
        self.service_name = service_name
        self.cache_dir = self._setup_cache_dir()

    def _setup_cache_dir(self):
        try:
            cache_dir = self.config.get_path(
                f"cache.{self.service_name}_dir"
            )
            # 创建缓存目录
            cache_dir.mkdir(exist_ok=True)
            return cache_dir

        except Exception as e:
            raise ValueError(f"设置缓存目录失败:{e}")


class TTSClient:
    def __init__(self):
        self.config = ConfigManager()
        self.logger = LogManager("tts").get_logger()
        self.cache = CacheManager(self.config,"tts")

        # 常见voice列表见 https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
        self.voice = self.config.get_json("tts.voice")
        self.language = self.config.get_json("tts.language")
        self.speed = self.config.get_json("tts.speed")
        self.pipeline = self._init_client()

        logging.getLogger('jieba').setLevel(logging.WARNING)  # 禁用jieba的日志

        self._stop_event = None
        self._play_thread = None
        self._audio_queue = None
        self._generator = None
        self._audio_stream = None

    def _init_client(self):
        os.environ["HF_TOKEN"] = self.config.get_env("HF_TOKEN")
        # 指定基础语言
        return KPipeline(lang_code=self.language, repo_id="hexgrad/Kokoro-82M")

    def stream_tts(self, text: str):
        audio_queue = queue.Queue()
        stop_event = threading.Event()

        self._audio_queue = audio_queue
        self._stop_event = stop_event

        def play_worker():
            try:
                # 打开音频流
                with sd.OutputStream(samplerate=24000, channels=1, dtype=np.float32) as stream:
                    self._audio_stream = stream
                    while True:
                        try:
                            audio_data = audio_queue.get(timeout=0.1)
                            if audio_data is None:
                                audio_queue.task_done()
                                break
                            try:
                                stream.write(audio_data)
                            except (sd.PortAudioError, OSError):
                                audio_queue.task_done()
                                break
                            audio_queue.task_done()
                        except queue.Empty:
                            continue
            except Exception as e:
                self.logger.error(f"音频播放错误: {e}")
            finally:
                self._audio_stream = None

        play_thread = threading.Thread(target=play_worker, daemon=True)
        play_thread.start()
        self._play_thread = play_thread

        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r'\n+',
        )
        self._generator = generator
        interrupted = False
        try:
            for _, _, audio in generator:
                if stop_event.is_set():
                    interrupted = True
                    break
                audio_queue.put(audio)
        finally:
            generator.close()
            self._generator = None

        if not interrupted:
            audio_queue.join()
            audio_queue.put(None)
        else:
            while True:
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                except queue.Empty:
                    break
            audio_queue.put(None)

        play_thread.join()
        self._stop_event = None
        self._play_thread = None
        self._audio_queue = None

    def interrupt(self):
        if self._audio_stream is not None:
            try:
                self._audio_stream.close()
            except Exception as e:
                self.logger.warning(f"{e}")
            self._audio_stream = None

        if self._stop_event is not None:
            self._stop_event.set()

        if self._audio_queue is not None:
            while True:
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.task_done()
                except queue.Empty:
                    break
            self._audio_queue.put(None)

        if self._play_thread is not None:
            self._play_thread.join(timeout=1.0)

        self._stop_event = None
        self._play_thread = None
        self._audio_queue = None
        self._generator = None
        self._audio_stream = None