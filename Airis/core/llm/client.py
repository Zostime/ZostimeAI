"""
适用于OpenAI标准API格式
"""
from datetime import datetime
import openai
from typing import Any,Generator,Dict,List,Optional
from pathlib import Path
import json

from ..common.config import ConfigManager   #配置管理器
from ..common.logger import LogManager      #日志管理器

# ===============================
# 缓存管理器
# ===============================
class CacheManager:
    def __init__(self, service_name: str):
        """初始化缓存管理器

        Args:
            service_name: 服务名称,用于缓存目录命名
        """
        self.config = ConfigManager()
        self.logger = LogManager("llm").get_logger()
        self.service_name = service_name
        self.cache_dir = self._setup_cache_dir()
    def _setup_cache_dir(self) -> Path:
        try:
            cache_dir = self.config.get_path(
                f"cache.{self.service_name}_dir"
            )
            #创建缓存目录
            cache_dir.mkdir(exist_ok=True)
            #创建子目录
            sub_dirs = ['conversations']
            for sub_dir in sub_dirs:
                (cache_dir / sub_dir).mkdir(exist_ok=True)

            self.logger.info(f"{self.service_name}缓存目录:{cache_dir}")
            return cache_dir

        except Exception as e:
            self.logger.error(f"设置缓存目录失败:{e}")
            raise

    def save_response(
            self,
            response_data: Dict[str, Any],
    ) -> Optional[Path]:
        """保存响应到文件
        Args:
            response_data: 响应数据字典
        Returns:
            Optional[Path]: 保存的文件路径，失败返回None
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"{timestamp}.json"
            subdir = 'conversations'

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

            self.logger.debug(f"响应已保存:{filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"保存响应失败:{e}")
            return None

# ===============================
# LLM客户端
# ===============================
class LLMClient:
    def __init__(self):
        """初始化LLMClient"""
        self.config=ConfigManager()
        self.logger=LogManager("llm").get_logger()
        self.cache=CacheManager("llm")

        self.client = self._init_client()
        self.model = self.config.get_env("LLM_MODEL")
        self.temperature = self.config.get_json('llm.temperature', 0.7)
        self.max_tokens = self.config.get_json('llm.max_tokens', 4096)
        self.top_p = self.config.get_json('llm.top_p', 1.0)
        self.frequency_penalty = self.config.get_json('llm.frequency_penalty', 0.0)
        self.presence_penalty = self.config.get_json('llm.presence_penalty', 0.0)
        self.tool_choice = self.config.get_json('llm.tool_choice', 'auto')

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
                'tools': kwargs.get('tools', None),
                'tool_choice': kwargs.get('tool_choice', "auto"),
                'response_format': kwargs.get('response_format', {"type": "text"}),
                'stream': True
            }

            # 发送请求
            stream = self.client.chat.completions.create(**request_params)

            full_content = ""
            tool_calls_buffer = {}
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

                if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        idx = tool_call.index

                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tool_call.id,
                                "name": "",
                                "arguments": ""
                            }

                        if tool_call.function.name:
                            tool_calls_buffer[idx]["name"] += tool_call.function.name

                        if tool_call.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tool_call.function.arguments

            parsed_tool_calls = []

            for idx in sorted(tool_calls_buffer.keys()):
                tc = tool_calls_buffer[idx]
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    self.logger.warn("JSON解析失败: {tc['arguments']}")
                    args = tc["arguments"]

                parsed_tool_calls.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": args
                })

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
                'tool_calls': parsed_tool_calls if parsed_tool_calls else None,
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

            self.cache.save_response(response_data)

            # 返回最终结果
            return {
                'full_content': full_content,
                'tool_calls': parsed_tool_calls if parsed_tool_calls else None,
                'tokens_used': tokens_used,
                'response_model': response_model,
                'usage': response_data['usage']
            }

        except Exception as e:
            raise ValueError(f"流式处理失败:{e}")