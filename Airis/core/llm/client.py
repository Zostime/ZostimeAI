#!/usr/bin/env python3
"""
LLM客户端模块
支持多种AI提供商（DeepSeek、OpenAI等）
版本: 1.0.0
作者: Zostime
"""

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


# ===============================
# 配置管理器
# ===============================
class ConfigManager:
    """统一配置管理器"""

    def __init__(self):
        """初始化配置管理器"""
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "Files" / "config"
        self.json_config = {}
        self._load_configs()
        self._validate_config()

    def _load_configs(self):
        """加载.env和config.json配置"""
        # 加载.env文件
        env_file = self.config_dir / ".env"
        if not env_file.exists():
            raise ValueError(f"未找到.env文件！请创建: {env_file}")

        load_dotenv(env_file)

        # 加载JSON配置文件
        json_file = self.config_dir / "config.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.json_config = json.load(f)
                logging.info(f"配置文件加载成功: {json_file}")
            except json.JSONDecodeError as e:
                logging.error(f"配置文件格式错误: {e}")
                raise
        else:
            logging.warning(f"未找到配置文件: {json_file}，使用默认配置")
            self.json_config = {}

    def _validate_config(self):
        """验证必要配置"""
        required_env_vars = ['LLM_API_KEY']
        missing_vars = []

        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            error_msg = f"缺少必要的环境变量: {missing_vars}！请配置Files/config/.env文件"
            logging.error(error_msg)
            raise ValueError(error_msg)

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

    def get(self, key_path: str, default: Any = None) -> Any:
        """通用获取配置方法"""
        # 优先从环境变量获取
        env_key = key_path.upper().replace('.', '_')
        env_value = self.get_env(env_key)

        if env_value is not None:
            # 尝试转换为适当类型
            if env_value.lower() in ['true', 'false']:
                return env_value.lower() == 'true'
            if env_value.isdigit():
                return int(env_value)
            if env_value.replace('.', '', 1).isdigit():
                return float(env_value)
            return env_value

        # 其次从JSON配置获取
        json_value = self.get_json(key_path)
        if json_value is not None:
            return json_value

        # 返回默认值
        return default

    def get_path(self, key_path: str, ensure_exists: bool = True) -> Path:
        """获取路径配置，转换为绝对路径"""
        path_str = self.get(key_path, "")

        if not path_str:
            raise ValueError(f"未找到路径配置: {key_path}")

        path = self.base_dir / path_str

        if ensure_exists:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                logging.error(f"创建目录权限不足: {path} - {e}")
                raise

        return path

    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置信息"""
        return {
            "environment_variables": {
                k: v for k, v in os.environ.items()
                if k.startswith(('LLM_', 'OPENAI_'))
            },
            "json_config": self.json_config
        }

    def reload(self):
        """重新加载配置"""
        logging.info("重新加载配置...")
        self._load_configs()
        self._validate_config()


# ===============================
# 日志管理器
# ===============================
class LogManager:
    """日志管理器"""

    def __init__(self, config: ConfigManager, service_name: str = "LLM"):
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
        log_level_str = self.config.get('logging.level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # 设置日志目录
        self.log_dir = self.config.get_path(
            f"logging.{self.service_name.lower()}_log_dir",
            f"Files/logs/{self.service_name}"
        )

        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # 清除现有处理器
        root_logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 文件处理器（按大小轮转）
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

        # 设置特定模块的日志级别
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)

        logging.info(f"{self.service_name} 日志系统初始化完成")
        logging.info(f"日志级别: {log_level_str}")
        logging.info(f"日志目录: {self.log_dir}")

    def get_log_file_path(self) -> Path:
        """获取当前日志文件路径"""
        return self.log_dir / f"{self.service_name.lower()}.log"

    def clear_old_logs(self, days: int = 30):
        """清理旧日志文件"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)

            for file in self.log_dir.glob("*.log*"):  # 包括轮转文件
                if file.is_file():
                    file_time = file.stat().st_mtime
                    if file_time < cutoff_time:
                        file.unlink()
                        logging.info(f"删除旧日志文件: {file.name}")

        except Exception as e:
            logging.error(f"清理旧日志失败: {e}")


# ===============================
# 缓存管理器
# ===============================
class CacheManager:
    """缓存管理器"""

    def __init__(self, config: ConfigManager, service_name: str = "LLM"):
        """初始化缓存管理器

        Args:
            config: 配置管理器实例
            service_name: 服务名称，用于缓存目录命名
        """
        self.config = config
        self.service_name = service_name
        self.cache_dir = self._setup_cache_dir()

    def _setup_cache_dir(self) -> Path:
        """设置缓存目录"""
        try:
            cache_dir = self.config.get_path(
                f"cache.{self.service_name.lower()}_dir",
                f"Files/cache/{self.service_name}"
            )

            # 创建子目录
            subdirs = ['responses', 'conversations', 'templates']
            for subdir in subdirs:
                (cache_dir / subdir).mkdir(exist_ok=True)

            logging.info(f"{self.service_name} 缓存目录: {cache_dir}")
            return cache_dir

        except Exception as e:
            logging.error(f"设置缓存目录失败: {e}")
            raise

    def save_response(
        self,
        response_data: Dict[str, Any],
        conversation_id: str = None
    ) -> Optional[Path]:
        """保存响应到文件

        Args:
            response_data: 响应数据字典
            conversation_id: 会话ID，用于组织文件

        Returns:
            Optional[Path]: 保存的文件路径，失败返回None
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            provider = response_data.get('provider', 'unknown')
            model = response_data.get('model', 'unknown').replace('/', '_')

            if conversation_id:
                filename = f"{conversation_id}_{timestamp}_{provider}_{model}.json"
                subdir = 'conversations'
            else:
                filename = f"response_{timestamp}_{provider}_{model}.json"
                subdir = 'responses'

            # 创建文件路径
            filepath = self.cache_dir / subdir / filename

            # 添加元数据
            response_data['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'filepath': str(filepath)
            }

            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            logging.info(f"响应已保存: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"保存响应失败: {e}")
            return None

    def load_response(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """从文件加载响应"""
        try:
            if not filepath.exists():
                logging.warning(f"文件不存在: {filepath}")
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logging.debug(f"响应已加载: {filepath}")
            return data

        except Exception as e:
            logging.error(f"加载响应失败: {e}")
            return None

    def search_responses(
        self,
        provider: str = None,
        model: str = None,
        date_from: str = None,
        date_to: str = None
    ) -> List[Path]:
        """搜索响应文件"""
        try:
            response_dir = self.cache_dir / 'responses'
            files = list(response_dir.glob("*.json"))

            # 过滤条件
            filtered_files = []

            for file in files:
                # 按日期过滤
                if date_from:
                    file_date = datetime.fromtimestamp(file.stat().st_mtime)
                    from_date = datetime.strptime(date_from, "%Y%m%d")
                    if file_date < from_date:
                        continue

                if date_to:
                    file_date = datetime.fromtimestamp(file.stat().st_mtime)
                    to_date = datetime.strptime(date_to, "%Y%m%d")
                    if file_date > to_date:
                        continue

                # 按文件名内容过滤
                filename = file.stem
                if provider and provider not in filename:
                    continue
                if model and model not in filename:
                    continue

                filtered_files.append(file)

            return sorted(filtered_files, key=lambda x: x.stat().st_mtime, reverse=True)

        except Exception as e:
            logging.error(f"搜索响应文件失败: {e}")
            return []

    def clear_cache(self, days: int = None):
        """清理缓存文件"""
        try:
            import time
            current_time = time.time()

            for subdir in ['responses', 'conversations']:
                dir_path = self.cache_dir / subdir

                if not dir_path.exists():
                    continue

                for file in dir_path.glob("*.json"):
                    if days:
                        file_time = file.stat().st_mtime
                        if current_time - file_time > days * 24 * 60 * 60:
                            file.unlink()
                            logging.info(f"删除缓存文件: {file.name}")
                    else:
                        file.unlink()
                        logging.info(f"删除缓存文件: {file.name}")

            logging.info("缓存清理完成")

        except Exception as e:
            logging.error(f"清理缓存失败: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'by_type': {},
            'by_provider': {}
        }

        try:
            for subdir in ['responses', 'conversations', 'templates']:
                dir_path = self.cache_dir / subdir

                if not dir_path.exists():
                    continue

                files = list(dir_path.glob("*"))
                stats['by_type'][subdir] = len(files)

                for file in files:
                    if file.is_file():
                        stats['total_files'] += 1
                        stats['total_size_bytes'] += file.stat().st_size

                        # 统计提供商
                        if subdir in ['responses', 'conversations']:
                            if file.suffix == '.json':
                                try:
                                    with open(file, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    provider = data.get('provider', 'unknown')
                                    stats['by_provider'][provider] = \
                                        stats['by_provider'].get(provider, 0) + 1
                                except:
                                    pass

            stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)

        except Exception as e:
            logging.error(f"获取缓存统计失败: {e}")

        return stats


# ===============================
# LLM客户端
# ===============================
class LLMClient:
    """LLM客户端（支持多种提供商）"""

    def __init__(self, provider: str = None):
        """初始化LLM客户端

        Args:
            provider: 提供商名称，如 'deepseek', 'openai', 'zhipu'等
        """
        # 初始化配置
        self.config = ConfigManager()
        self.logger = logging.getLogger(__name__)

        # 确定提供商
        self.provider = provider or self.config.get('llm.provider', 'deepseek')
        self.logger.info(f"初始化LLM客户端，提供商: {self.provider}")

        # 获取提供商特定配置
        provider_config = self.config.get_json(f'llm.{self.provider}')
        if not provider_config:
            error_msg = f"未找到 {self.provider} 的配置，请检查config.json"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 初始化OpenAI客户端
        self.client = self._init_client(provider_config)

        # 配置参数
        self.model = provider_config.get('model', 'gpt-3.5-turbo')
        self.temperature = float(provider_config.get('temperature', 0.7))
        self.max_tokens = int(provider_config.get('max_tokens', 4096))
        self.top_p = float(provider_config.get('top_p', 1.0))
        self.frequency_penalty = float(provider_config.get('frequency_penalty', 0.0))
        self.presence_penalty = float(provider_config.get('presence_penalty', 0.0))

        # 初始化缓存管理器
        self.cache_manager = CacheManager(self.config, "LLM")

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'failed_requests': 0,
            'providers_used': [self.provider]
        }

        self.logger.info(f"LLM客户端初始化完成: {self.provider}/{self.model}")

    def _init_client(self, provider_config: Dict[str, Any]) -> OpenAI:
        """初始化OpenAI兼容客户端"""
        try:
            # 获取API密钥
            api_key_env = provider_config.get('api_key_env', 'LLM_API_KEY')
            api_key = self.config.get_env(api_key_env)

            if not api_key:
                raise ValueError(f"API密钥未配置: {api_key_env}")

            # 获取基础URL
            base_url = provider_config.get('base_url', 'https://api.openai.com/v1')

            # 创建客户端
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=provider_config.get('timeout', 30.0)
            )

            self.logger.info(f"OpenAI客户端初始化成功，base_url: {base_url}")
            return client

        except Exception as e:
            self.logger.error(f"初始化OpenAI客户端失败: {e}")
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        stream_callback=None,
        conversation_id: str = None,
        **kwargs
    ) -> Generator[str, None, Dict[str, Any]]:
        """流式聊天

        Args:
            messages: 消息列表
            stream_callback: 流式回调函数
            conversation_id: 会话ID，用于缓存
            **kwargs: 其他参数，覆盖默认配置

        Yields:
            str: 流式返回的文本片段

        Returns:
            Dict[str, Any]: 完整的响应信息
        """
        self.stats['total_requests'] += 1

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

            self.logger.debug(f"开始流式请求: {request_params['model']}")

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
                    if stream_callback:
                        try:
                            stream_callback(content_chunk)
                        except Exception as cb_error:
                            self.logger.error(f"流式回调失败: {cb_error}")

                    # 生成器返回片段
                    yield content_chunk

            # 更新统计
            if tokens_used:
                self.stats['total_tokens'] += tokens_used.total_tokens

            # 准备响应数据
            response_data = {
                'timestamp': datetime.now().isoformat(),
                'provider': self.provider,
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

            # 保存响应
            self.cache_manager.save_response(response_data, conversation_id)

            self.logger.info(
                f"流式请求完成: {len(full_content)}字符, "
                f"{response_data['usage']['total_tokens']} tokens"
            )

            # 返回最终结果
            return {
                'full_content': full_content,
                'tokens_used': tokens_used,
                'response_model': response_model,
                'provider': self.provider,
                'usage': response_data['usage']
            }

        except Exception as e:
            self.stats['failed_requests'] += 1
            self.logger.error(f"流式处理失败: {e}")
            raise

    def chat_simple(
        self,
        prompt: str,
        system_prompt: str = None,
        conversation_id: str = None,
        **kwargs
    ) -> str:
        """简单聊天（非流式）

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            conversation_id: 会话ID，用于缓存
            **kwargs: 其他参数

        Returns:
            str: AI回复内容
        """
        self.stats['total_requests'] += 1

        try:
            # 准备消息
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": "你是一个有用的AI助手。"})

            messages.append({"role": "user", "content": prompt})

            # 准备请求参数
            request_params = {
                'model': kwargs.get('model', self.model),
                'messages': messages,
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'top_p': kwargs.get('top_p', self.top_p),
                'frequency_penalty': kwargs.get('frequency_penalty', self.frequency_penalty),
                'presence_penalty': kwargs.get('presence_penalty', self.presence_penalty)
            }

            self.logger.debug(f"开始简单请求: {request_params['model']}")

            # 发送请求
            response = self.client.chat.completions.create(**request_params)

            # 提取内容
            content = response.choices[0].message.content

            # 准备响应数据
            response_data = {
                'timestamp': datetime.now().isoformat(),
                'provider': self.provider,
                'model': response.model,
                'request_params': request_params,
                'messages': messages,
                'response': {
                    'role': 'assistant',
                    'content': content
                },
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

            # 更新统计
            self.stats['total_tokens'] += response.usage.total_tokens

            # 保存响应
            self.cache_manager.save_response(response_data, conversation_id)

            self.logger.info(
                f"简单请求完成: {len(content)}字符, "
                f"{response.usage.total_tokens} tokens"
            )

            return content

        except Exception as e:
            self.stats['failed_requests'] += 1
            self.logger.error(f"简单请求失败: {e}")
            raise

    def chat_with_history(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        system_prompt: str = None,
        max_history: int = 10,
        **kwargs
    ) -> str:
        """带历史记录的聊天

        Args:
            user_input: 用户输入
            history: 历史记录列表
            system_prompt: 系统提示词
            max_history: 最大历史记录数
            **kwargs: 其他参数

        Returns:
            str: AI回复内容
        """
        # 准备消息
        messages = []

        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加历史记录（限制数量）
        if history:
            messages.extend(history[-max_history:])

        # 添加用户输入
        messages.append({"role": "user", "content": user_input})

        # 发送请求
        return self.chat_simple(
            "",
            system_prompt=None,  # 已经在messages中
            conversation_id=kwargs.get('conversation_id'),
            **kwargs
        )

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        cache_stats = self.cache_manager.get_cache_stats()

        return {
            'provider': self.provider,
            'model': self.model,
            'config': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'top_p': self.top_p,
                'frequency_penalty': self.frequency_penalty,
                'presence_penalty': self.presence_penalty
            },
            'statistics': self.stats,
            'cache': cache_stats,
            'timestamp': datetime.now().isoformat()
        }

    def test_connection(self) -> bool:
        """测试连接是否正常"""
        try:
            self.logger.info("测试API连接...")

            # 发送一个简单的测试请求
            response = self.chat_simple("Hello", system_prompt="请回复 'OK'")

            success = response.strip().upper() == 'OK'

            if success:
                self.logger.info("API连接测试成功")
            else:
                self.logger.warning(f"API连接测试异常，响应: {response}")

            return success

        except Exception as e:
            self.logger.error(f"API连接测试失败: {e}")
            return False

    def switch_provider(self, provider: str) -> bool:
        """切换提供商"""
        try:
            self.logger.info(f"尝试切换提供商: {provider}")

            # 保存旧配置
            old_provider = self.provider

            # 重新初始化客户端
            self.__init__(provider)  # 重新调用初始化

            # 测试新提供商
            if self.test_connection():
                self.logger.info(f"成功切换到提供商: {provider}")
                return True
            else:
                # 切换失败，恢复原提供商
                self.__init__(old_provider)
                self.logger.error(f"切换到提供商 {provider} 失败，已恢复")
                return False

        except Exception as e:
            self.logger.error(f"切换提供商失败: {e}")
            return False