from kokoro import KPipeline
import soundfile as sf
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
        self.logger = LogManager("tts")
        self.cache = CacheManager(self.config,"tts")
        self.pipeline = self._init_client()

        # 常见voice列表见 https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
        self.voice = 'af_heart'

    def _init_client(self):
        os.environ["HF_TOKEN"] = self.config.get_env("HF_TOKEN")
        # 指定基础语言
        return KPipeline(lang_code='zh', repo_id="hexgrad/Kokoro-82M")

    def text_to_speech(self,text: str):
        """输入音频,输出.wav文件"""
        tts_cache = self.cache.cache_dir

        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=1,
            split_pattern=r'\n+'
        )
        for i,(gs, ps, audio) in enumerate(generator):
            sf.write(rf"{tts_cache}\output_{i}.wav", audio, 24000)