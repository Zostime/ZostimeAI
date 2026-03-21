"""
适用于OpenAI标准API格式
"""
from datetime import datetime
import openai
from typing import Any,Generator,Dict,List

from ..common.config import ConfigManager   #配置管理器
from ..common.logger import LogManager      #日志管理器

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
        self.logger=LogManager("llm")

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