from kokoro import KPipeline
import soundfile as sf
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from typing import Any
import logging
from logging import handlers
import sys

# ===============================
# 配置管理器
# ===============================
class ConfigManager:
    def __init__(self):
        """初始化配置管理器"""
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "Files" / "config"
        self.json_config = {}
        self._load_config()

    def _load_config(self):
        """加载.env和config.json"""
        env_file = self.config_dir / ".env"
        json_file = self.config_dir / "config.json"
        #加载.env
        if not env_file.exists():
            raise ValueError(f"未找到.env文件,请创建:{env_file}")
        load_dotenv(env_file)

        required_keys = []
        missing_keys = []
        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        if missing_keys:
            raise ValueError(f"配置项:{', '.join(missing_keys)}不存在")

        #加载config.json
        if not json_file.exists():
            raise ValueError(f"未找到config.json文件,请创建:{json_file}")
        try:
            with open(json_file, encoding="utf-8") as f:
                self.json_config = json.load(f)
        except Exception as e:
            raise ValueError(f"config.json错误{e}")

    @staticmethod
    def get_env(key: str, default: str = None) -> str:
        """从.env获取配置"""
        value = os.getenv(key, default)
        return value

    def get_json(self, key_path: str, default: Any = None) -> Any:
        """从JSON配置获取值"""
        if not key_path:
            return self.json_config

        parts = key_path.split('.')
        value = self.json_config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_path(self, key_path: str) -> Path:
        """获取路径配置，转换为绝对路径"""
        path_str = self.get_json(key_path,"")

        if not path_str:
            raise ValueError(f"未找到路径配置:{key_path}")

        path = self.base_dir / path_str

        return path

# ===============================
# 日志管理器
# ===============================
class LogManager:
    def __init__(self, config: ConfigManager, service_name: str):
        """初始化日志管理器

        Args:
            config: 配置管理器实例
            service_name: 服务名称，用于日志文件命名
        """
        self.config = config
        self.service_name = service_name
        self.log_dir = None
        self._setup_logging()

    def _setup_logging(self):
        """设置日志系统"""
        # 获取日志配置
        log_level_str = self.config.get_json('logging.level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # 设置日志目录
        log_dir = self.config.get_path(
            f"logging.{self.service_name}_dir",
        )
        # 创建日志目录
        log_dir.mkdir(exist_ok=True)

        #配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        #清除现有处理器
        root_logger.handlers.clear()

        #创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        #控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        #文件处理器(按大小轮转)
        log_file = log_dir / f"{self.service_name.lower()}.log"
        max_size_mb = int(self.config.get_json('logging.max_file_size_mb', 10))
        backup_count = int(self.config.get_json('logging.backup_count', 5))

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"{self.service_name}日志初始化完成")
        logging.info(f"日志级别:{log_level_str}")

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
        self.cache = CacheManager(self.config,"tts")
        os.environ["HF_TOKEN"]=self.config.get_env("HF_TOKEN")
        # 指定基础语言
        self.pipeline = KPipeline(lang_code='zh')
        self.voice = 'af_heart'# 常见voice列表见 https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

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