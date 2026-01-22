import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dotenv import load_dotenv
from openai import OpenAI


class ConfigManager:
    """统一配置管理器"""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "Files" / "config"
        self._load_configs()

    def _load_configs(self):
        """加载.env和config.json配置"""
        env_file = self.config_dir / ".env"
        if not env_file.exists():
            raise ValueError(f"未找到.env文件！请创建: {env_file}")

        load_dotenv(env_file)

        json_file = self.config_dir / "config.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                self.json_config = json.load(f)
        else:
            self.json_config = {}

        self._validate_config()

    def _validate_config(self):
        """验证必要配置"""
        if not os.getenv('LLM_API_KEY'):
            raise ValueError("未找到LLM_API_KEY！请配置Files/config/.env文件")

    def get_env(self, key: str, default: str = None) -> str:
        """从.env获取配置"""
        return os.getenv(key, default)

    def get_json(self, key_path: str, default: Any = None) -> Any:
        """从JSON配置获取值"""
        parts = key_path.split('.')
        value = self.json_config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        """通用获取配置方法"""
        env_key = key_path.upper().replace('.', '_')
        env_value = self.get_env(env_key)
        if env_value is not None:
            return env_value

        json_value = self.get_json(key_path)
        if json_value is not None:
            return json_value

        return default

    def get_path(self, key_path: str, ensure_exists: bool = True) -> Path:
        """获取路径配置，转换为绝对路径"""
        path_str = self.get(key_path, "")
        if not path_str:
            return Path("")

        path = self.base_dir / path_str

        if ensure_exists:
            path.mkdir(parents=True, exist_ok=True)

        return path


class LogManager:
    """日志管理器"""

    def __init__(self, config: ConfigManager, service_name: str = "LLM"):
        self.config = config
        self.service_name = service_name
        self.log_dir = self._setup_log_dir()
        self._setup_logging()

    def _setup_log_dir(self) -> Path:
        """设置日志目录"""
        log_dir_key = f"logging.{self.service_name.lower()}_log_dir"
        log_dir_str = self.config.get(log_dir_key, "Files/logs/LLM")
        log_dir = self.config.base_dir / log_dir_str
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _setup_logging(self):
        """设置日志系统"""
        log_level_str = self.config.get('logging.level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        log_file = self.log_dir / f"{self.service_name.lower()}.log"
        max_size_mb = int(self.config.get('logging.max_file_size_mb', 10))
        backup_count = int(self.config.get('logging.backup_count', 5))

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"{self.service_name}日志系统初始化完成")


class CacheManager:
    """缓存管理器"""

    def __init__(self, config: ConfigManager, service_name: str = "LLM"):
        self.config = config
        self.service_name = service_name
        self.cache_dir = self._setup_cache_dir()

    def _setup_cache_dir(self) -> Path:
        """设置缓存目录"""
        cache_dir_key = f"cache.{self.service_name.lower()}_dir"
        cache_dir_str = self.config.get(cache_dir_key, "Files/cache/LLM")
        cache_dir = self.config.base_dir / cache_dir_str
        cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"{self.service_name}缓存目录: {cache_dir}")
        return cache_dir

    def save_response(self, response_data: Dict) -> Optional[Path]:
        """保存响应到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_{timestamp}.json"
            filepath = self.cache_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            logging.info(f"响应已保存: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"保存响应失败: {e}")
            return None


class LLMClient:
    """LLM客户端（支持多种提供商）"""

    def __init__(self, provider: str = None):
        """初始化LLM客户端

        Args:
            provider: 提供商名称，如 'deepseek', 'openai'
        """
        self.config = ConfigManager()
        self.logger = logging.getLogger(__name__)
        self.cache_manager = CacheManager(self.config, "LLM")

        # 确定提供商
        self.provider = provider or self.config.get('llm.provider', 'deepseek')

        # 获取提供商特定配置
        provider_config = self.config.get_json(f'llm.{self.provider}')
        if not provider_config:
            raise ValueError(f"未找到 {self.provider} 的配置")

        # 初始化OpenAI客户端
        self.client = self._init_client(provider_config)

        # 配置参数
        self.model = provider_config.get('model', 'gpt-3.5-turbo')
        self.temperature = float(provider_config.get('temperature', 0.7))
        self.max_tokens = int(provider_config.get('max_tokens', 4096))

    def _init_client(self, provider_config: Dict):
        """初始化OpenAI兼容客户端"""
        api_key = self.config.get_env('LLM_API_KEY')
        base_url = provider_config.get('base_url', 'https://api.openai.com/v1')

        if not api_key:
            raise ValueError("LLM_API_KEY未配置")

        return OpenAI(api_key=api_key, base_url=base_url)

    def chat_stream(
            self,
            messages: List[Dict[str, str]],
            stream_callback=None
    ) -> Generator[str, None, Dict[str, Any]]:
        """流式聊天"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            full_content = ""
            tokens_used = None
            response_model = self.model

            for chunk in stream:
                if hasattr(chunk, 'model') and chunk.model:
                    response_model = chunk.model

                if hasattr(chunk, 'usage') and chunk.usage:
                    tokens_used = chunk.usage

                if chunk.choices and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk

                    print(content_chunk, end='', flush=True)

                    if stream_callback:
                        stream_callback(content_chunk)

                    yield content_chunk

            print()

            # 保存响应
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "provider": self.provider,
                "model": response_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "response": {
                    "role": "assistant",
                    "content": full_content
                },
                "usage": {
                    "prompt_tokens": tokens_used.prompt_tokens if tokens_used else 0,
                    "completion_tokens": tokens_used.completion_tokens if tokens_used else 0,
                    "total_tokens": tokens_used.total_tokens if tokens_used else 0
                }
            }

            self.cache_manager.save_response(response_data)

            return {
                "full_content": full_content,
                "tokens_used": tokens_used,
                "response_model": response_model,
                "provider": self.provider
            }

        except Exception as e:
            self.logger.error(f"流式处理失败: {e}")
            raise

    def chat_simple(self, prompt: str) -> str:
        """简单聊天（非流式）"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        content = response.choices[0].message.content

        response_data = {
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider,
            "model": response.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "response": {
                "role": "assistant",
                "content": content
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

        self.cache_manager.save_response(response_data)

        return content


def main():
    """主函数"""
    try:
        print("=" * 50)
        print("LLM 客户端示例")
        print("=" * 50)

        # 初始化客户端
        llm_client = LLMClient()

        print(f"\n提供商: {llm_client.provider}")
        print(f"模型: {llm_client.model}")
        print(f"温度: {llm_client.temperature}")
        print("-" * 30)

        # 测试简单聊天
        print("\n[测试简单聊天]")
        response = llm_client.chat_simple("你好，简单介绍一下你自己")
        print(f"AI回复: {response}")

        # 测试流式聊天
        print("\n[测试流式聊天]")
        messages = [
            {"role": "system", "content": "你是一个专业的AI助手"},
            {"role": "user", "content": "请用中文写一首关于春天的诗"}
        ]

        print("AI回复 (流式): ")

        result = llm_client.chat_stream(messages)

        full_content = ""
        try:
            while True:
                chunk = next(result)
                full_content += chunk
        except StopIteration as e:
            final_result = e.value
            print(f"\n\n流式输出完成！")
            print(f"总长度: {len(full_content)} 字符")
            print(f"使用模型: {final_result['response_model']}")

            if final_result['tokens_used']:
                print(f"Tokens使用: {final_result['tokens_used'].total_tokens}")

    except ValueError as e:
        print(f"\n配置错误: {e}")
    except Exception as e:
        print(f"\n程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()