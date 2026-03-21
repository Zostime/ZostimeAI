from kokoro import KPipeline
import soundfile as sf
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from typing import Any

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

        required_keys = ['HF_TOKEN']
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

    def get_env(self, key: str, default: str = None) -> str:
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

# ===============================
# 缓存管理器
# ===============================
class CacheManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.cache_dir = self.base_dir / "Files" / "cache" / "tts"

class TTSClient:
    def __init__(self):
        self.config = ConfigManager()
        self.cache = CacheManager()
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