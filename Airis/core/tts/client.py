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

        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._audio_stream = None

    def _init_client(self):
        os.environ["HF_TOKEN"] = self.config.get_env("HF_TOKEN")
        # 指定基础语言
        return KPipeline(lang_code=self.language, repo_id="hexgrad/Kokoro-82M")

    def stream_tts(self, text: str):
        self._stop_event.clear()

        def play_worker():
            chunk_duration = 0.05  # 每次写入的时长（秒），越小响应越快
            sample_rate = 24000
            chunk_size = int(sample_rate * chunk_duration)

            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype=np.float32) as stream:
                self._audio_stream = stream
                try:
                    while not self._stop_event.is_set():
                        try:
                            audio_block = self._audio_queue.get(timeout=0.5)
                            if audio_block is None:
                                break

                            for start in range(0, len(audio_block), chunk_size):
                                if self._stop_event.is_set():
                                    break
                                chunk = audio_block[start:start + chunk_size]
                                stream.write(chunk)

                            self._audio_queue.task_done()

                        except queue.Empty:
                            continue
                        except (Exception,KeyboardInterrupt):
                            break
                except (Exception,KeyboardInterrupt):
                    pass

        play_thread = threading.Thread(target=play_worker, daemon=True)
        play_thread.start()

        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r'\n+',
        )

        for i, (gs, ps, audio) in enumerate(generator):
            if self._stop_event.is_set():
                break
            self._audio_queue.put(audio)

        if not self._stop_event.is_set():
            self._audio_queue.join()

        self._stop_event.set()
        play_thread.join()

    def interrupt(self):
        self._stop_event.set()