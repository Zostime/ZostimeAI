from kokoro import KPipeline
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

    def _init_client(self):
        os.environ["HF_TOKEN"] = self.config.get_env("HF_TOKEN")
        # 指定基础语言
        return KPipeline(lang_code=self.language, repo_id="hexgrad/Kokoro-82M")

    def stream_tts(self, text: str):
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r'\n+',
        )
        return generator
