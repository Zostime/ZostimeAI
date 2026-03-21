import queue
from kokoro import KPipeline
import sounddevice as sd
import threading
import os
import numpy as np

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
        self.logger = LogManager("tts")
        self.cache = CacheManager(self.config,"tts")

        # 常见voice列表见 https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
        self.voice = self.config.get_json("tts.voice")
        self.language = self.config.get_json("tts.language")
        self.pipeline = self._init_client()

    def _init_client(self):
        os.environ["HF_TOKEN"] = self.config.get_env("HF_TOKEN")
        # 指定基础语言
        return KPipeline(lang_code=self.language, repo_id="hexgrad/Kokoro-82M")

    def stream_tts(self, text: str):
        audio_queue = queue.Queue()
        stop_event = threading.Event()

        def play_worker():
            with sd.OutputStream(samplerate=24000, channels=1, dtype=np.float32) as stream:
                while not stop_event.is_set():
                    try:
                        audio_data = audio_queue.get(timeout=0.5)
                        stream.write(audio_data)
                        audio_queue.task_done()
                    except queue.Empty:
                        continue
                #退出前清空剩余任务
                while not audio_queue.empty():
                    try:
                        audio_data = audio_queue.get_nowait()
                        stream.write(audio_data)
                        audio_queue.task_done()
                    except queue.Empty:
                        break

        play_thread = threading.Thread(target=play_worker, daemon=True)
        play_thread.start()

        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=1,
            split_pattern=r'\n+'
        )

        for i, (gs, ps, audio) in enumerate(generator):
            audio_queue.put(audio)

        # 等待所有音频片段被播放并标记完成
        audio_queue.join()
        stop_event.set()
        play_thread.join()