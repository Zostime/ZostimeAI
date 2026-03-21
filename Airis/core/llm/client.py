"""
适用于OpenAI标准API格式
"""
import sys
from datetime import datetime
import openai
from dotenv import load_dotenv
from typing import Any,Generator,Dict,List
import os
from pathlib import Path
import json
import logging
import logging.handlers

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

        required_keys = ['LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL']
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
class CacheClient:
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
            #创建缓存目录
            cache_dir.mkdir(exist_ok=True)
            #创建子目录
            sub_dirs = ['conversations', 'messages']
            for sub_dir in sub_dirs:
                (cache_dir / sub_dir).mkdir(exist_ok=True)

            return cache_dir

        except Exception as e:
            raise ValueError(f"设置缓存目录失败:{e}")


# ===============================
# LLM客户端
# ===============================
class LLMClient:
    def __init__(self):
        """初始化LLMClient"""
        self.config=ConfigManager()
        self.client = self._init_client()
        self.model = self.config.get_env("LLM_MODEL")
        self.temperature = self.config.get_json('llm.temperature', 0.7)
        self.max_tokens = self.config.get_json('llm.max_tokens', 4096)
        self.top_p = self.config.get_json('llm.top_p', 1.0)
        self.frequency_penalty = self.config.get_json('llm.frequency_penalty', 0.0)
        self.presence_penalty = self.config.get_json('llm.presence_penalty', 0.0)

    def _init_client(self):
        """初始化OpenAI客户端"""
        api_key=self.config.get_env("LLM_API_KEY")
        base_url=self.config.get_env("LLM_BASE_URL")
        # 创建客户端
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        return client

    def chat_stream(
            self,
            messages: List[Dict[str, str]],
            stream_callback=None,
            **kwargs
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Args:
            messages: 消息列表
            stream_callback: 流式回调函数
            **kwargs: 其他参数，覆盖默认配置
        Yields:
            str: 流式返回的文本片段
        Returns:
            Dict[str, Any]: 完整的响应信息(通过生成器返回值)
        """
        try:
            # 准备请求参数
            request_params = {
                'model': kwargs.get('model', self.model),
                'messages': messages,
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p),
                'frequency_penalty': kwargs.get('frequency_penalty', self.frequency_penalty),
                'presence_penalty': kwargs.get('presence_penalty', self.presence_penalty),
                'stream': True
            }

            # 发送请求
            stream = self.client.chat.completions.create(**request_params)

            full_content = ""
            tokens_used = None
            response_model = request_params['model']

            # 处理流式响应
            for chunk in stream:
                # 获取模型信息
                if hasattr(chunk, 'model') and chunk.model:
                    response_model = chunk.model

                # 获取token使用情况
                if hasattr(chunk, 'usage') and chunk.usage:
                    tokens_used = chunk.usage

                # 获取内容片段
                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk

                    # 调用回调函数
                    if stream_callback is not None:
                        try:
                            stream_callback(content_chunk)
                        except Exception as cb_error:
                            raise ValueError(f"流式回调失败:{cb_error}")

                    # 生成器返回片段
                    yield content_chunk

            # 准备响应数据
            response_data = {
                'timestamp': datetime.now().isoformat(),
                'model': response_model,
                'request_params': {
                    'temperature': request_params['temperature'],
                    'max_tokens': request_params['max_tokens'],
                    'top_p': request_params['top_p'],
                    'frequency_penalty': request_params['frequency_penalty'],
                    'presence_penalty': request_params['presence_penalty']
                },
                'messages': messages,
                'response': {
                    'role': 'assistant',
                    'content': full_content
                },
                'usage': {
                    'prompt_tokens': tokens_used.prompt_tokens if tokens_used else 0,
                    'completion_tokens': tokens_used.completion_tokens if tokens_used else 0,
                    'total_tokens': tokens_used.total_tokens if tokens_used else 0
                }
            }

            # 返回最终结果
            return {
                'full_content': full_content,
                'tokens_used': tokens_used,
                'response_model': response_model,
                'usage': response_data['usage']
            }

        except Exception as e:
            raise ValueError(f"流式处理失败:{e}")