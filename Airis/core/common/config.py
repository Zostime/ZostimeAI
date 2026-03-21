"""
配置管理器
"""
from dotenv import load_dotenv, dotenv_values
import os
from pathlib import Path
import json
from typing import Any

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if self._initialized:
            return
        self._initialized = True

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

        # 读取 .env 文件中的所有键值对
        env_vars = dotenv_values(env_file)

        # 筛选出值为空的键
        empty_keys = [key for key, value in env_vars.items() if not value]

        if empty_keys:
            raise ValueError(f"配置项:{', '.join(empty_keys)}不存在")

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
