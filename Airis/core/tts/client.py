import os
import sys
import json
import websocket
import datetime
import hashlib
import base64
import hmac
import time
import ssl
import queue
import threading
import pyaudio
from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
from time import mktime
import _thread as thread
import re
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from dotenv import load_dotenv
import logging
import logging.handlers


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
        required_keys = ['TTS_APP_ID', 'TTS_API_KEY', 'TTS_API_SECRET']
        missing_keys = []

        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"缺少必要配置项: {', '.join(missing_keys)}")

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
        """获取路径配置"""
        path_str = self.get(key_path, "")
        if not path_str:
            return Path("")

        path = self.base_dir / path_str

        if ensure_exists:
            path.mkdir(parents=True, exist_ok=True)

        return path


class LogManager:
    """日志管理器"""

    def __init__(self, config: ConfigManager, service_name: str = "TTS"):
        self.config = config
        self.service_name = service_name
        self.log_dir = self._setup_log_dir()
        self._setup_logging()

    def _setup_log_dir(self) -> Path:
        """设置日志目录"""
        log_dir_key = f"logging.{self.service_name.lower()}_log_dir"
        log_dir_str = self.config.get(log_dir_key, "Files/logs/TTS")
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
            datefmt='%Y-%m-d %H:%M:%S'
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

    def __init__(self, config: ConfigManager, service_name: str = "TTS"):
        self.config = config
        self.service_name = service_name
        self.cache_dir = self._setup_cache_dir()

    def _setup_cache_dir(self) -> Path:
        """设置缓存目录"""
        cache_dir_key = f"cache.{self.service_name.lower()}_dir"
        cache_dir_str = self.config.get(cache_dir_key, "Files/cache/TTS")
        cache_dir = self.config.base_dir / cache_dir_str
        cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"{self.service_name}缓存目录: {cache_dir}")
        return cache_dir

    def save_audio(self, audio_data: bytes, filename: str = None) -> Optional[Path]:
        """保存音频到文件"""
        try:
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"audio_{timestamp}.pcm"

            filepath = self.cache_dir / filename

            with open(filepath, 'wb') as f:
                f.write(audio_data)

            logging.info(f"音频已保存: {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"保存音频失败: {e}")
            return None


class AudioStreamManager:
    """音频流管理器（重构版）"""

    def __init__(self, config: ConfigManager, disable_internal_playback: bool = False):
        """
        初始化音频流管理器

        Args:
            config: 配置管理器
            disable_internal_playback: 是否禁用内部播放（用于避免双重播放）
        """
        self.config = config
        self.disable_internal_playback = disable_internal_playback

        # 读取配置
        audio_queue_size = int(config.get('streaming.audio_queue_size', 200))
        self.stream_chunk_size = int(config.get('streaming.stream_chunk_size', 2048))
        self.playback_buffer_ms = int(config.get('streaming.playback_buffer_ms', 200))

        # 读取实时播放配置，但根据disable_internal_playback参数决定是否启用
        realtime_playback_value = config.get('streaming.realtime_playback', True)
        if isinstance(realtime_playback_value, str):
            config_realtime_playback = realtime_playback_value.lower() == 'true'
        else:
            config_realtime_playback = bool(realtime_playback_value)

        # 最终决定是否启用实时播放
        self.realtime_playback_enabled = config_realtime_playback and not disable_internal_playback

        # 初始化队列
        self.audio_queue = queue.Queue(maxsize=audio_queue_size)

        # 播放状态
        self.is_playing = False
        self.should_play = False  # 控制是否应该播放的标志
        self.playback_thread = None
        self.audio_stream = None
        self.shutdown_flag = threading.Event()

        # 音频参数
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 24000
        self.chunk_size = 1024

        # 初始化PyAudio（但仅在需要播放时创建实例）
        self.pyaudio_instance = None
        if self.realtime_playback_enabled:
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
                logging.info("PyAudio初始化成功（内部播放已启用）")
            except Exception as e:
                logging.error(f"PyAudio初始化失败: {e}")
                self.pyaudio_instance = None
        else:
            logging.info("内部音频播放已禁用，仅用于音频数据管理")

    def enable_playback(self, enable: bool = True):
        """启用或禁用播放功能"""
        if enable and self.pyaudio_instance is None:
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
                logging.info("已启用内部音频播放")
            except Exception as e:
                logging.error(f"启用内部音频播放失败: {e}")
                return False
        elif not enable:
            self.stop_playback()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
                logging.info("已禁用内部音频播放")

        self.should_play = enable and self.pyaudio_instance is not None
        return self.should_play

    def add_audio_chunk(self, audio_data: bytes):
        """添加音频数据块到队列"""
        try:
            if self.shutdown_flag.is_set() or not audio_data:
                return False

            # 将数据分割成合适大小的块
            for i in range(0, len(audio_data), self.stream_chunk_size):
                chunk = audio_data[i:i + self.stream_chunk_size]
                if chunk:
                    try:
                        self.audio_queue.put(chunk, timeout=1)
                    except queue.Full:
                        try:
                            # 队列满时，丢弃最旧的数据
                            self.audio_queue.get_nowait()
                            self.audio_queue.put(chunk, timeout=1)
                        except queue.Empty:
                            pass

            # 如果启用实时播放且尚未开始播放，则启动播放
            if (self.realtime_playback_enabled and self.should_play and
                    not self.is_playing and self.pyaudio_instance and
                    not self.shutdown_flag.is_set()):
                self.start_playback()

            return True

        except Exception as e:
            logging.error(f"添加音频数据块失败: {e}")
            return False

    def start_playback(self):
        """开始播放音频"""
        if (self.is_playing or not self.pyaudio_instance or
                not self.should_play or self.shutdown_flag.is_set()):
            return False

        self.is_playing = True
        self.playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True,
            name="AudioPlayback"
        )
        self.playback_thread.start()
        return True

    def _playback_loop(self):
        """播放循环"""
        try:
            # 打开音频输出流
            self.audio_stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )

            playback_buffer = bytearray()
            buffer_target_bytes = int(self.sample_rate * 2 * self.playback_buffer_ms / 1000)

            logging.info("音频播放循环开始")

            while self.is_playing and not self.shutdown_flag.is_set():
                try:
                    # 从队列获取音频数据
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    playback_buffer.extend(audio_chunk)

                    # 当缓冲区达到目标大小时播放
                    if len(playback_buffer) >= buffer_target_bytes:
                        self.audio_stream.write(bytes(playback_buffer))
                        playback_buffer.clear()

                except queue.Empty:
                    # 队列为空时，播放缓冲区剩余数据
                    if playback_buffer:
                        self.audio_stream.write(bytes(playback_buffer))
                        playback_buffer.clear()

                    # 短暂休眠避免忙等待
                    time.sleep(0.01)

                except Exception as e:
                    logging.error(f"播放音频时出错: {e}")
                    break

            # 播放缓冲区剩余数据
            if playback_buffer:
                self.audio_stream.write(bytes(playback_buffer))

        except Exception as e:
            logging.error(f"播放循环出错: {e}")
        finally:
            self._cleanup_playback()
            self.is_playing = False
            logging.info("音频播放循环结束")

    def _cleanup_playback(self):
        """清理播放资源"""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
        except Exception as e:
            logging.error(f"清理播放资源时出错: {e}")

    def stop_playback(self):
        """停止播放"""
        logging.info("正在停止音频播放...")
        self.shutdown_flag.set()
        self.is_playing = False

        # 等待播放线程结束
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

        self._cleanup_playback()
        self.shutdown_flag.clear()

        # 清空队列
        self.clear_queue()

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """从队列获取音频数据块（供外部播放器使用）"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception as e:
            logging.error(f"获取音频数据块失败: {e}")
            return None

    def clear_queue(self):
        """清空音频队列"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.audio_queue.qsize()

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "is_playing": self.is_playing,
            "should_play": self.should_play,
            "queue_size": self.get_queue_size(),
            "realtime_playback_enabled": self.realtime_playback_enabled,
            "has_pyaudio_instance": self.pyaudio_instance is not None,
            "playback_thread_alive": self.playback_thread is not None and self.playback_thread.is_alive()
        }


def assemble_ws_auth_url(request_url: str, method: str = "GET",
                         api_key: str = "", api_secret: str = "") -> str:
    """组装WebSocket认证URL"""
    # 解析URL
    parsed = urlparse(request_url)
    host = parsed.hostname
    path = parsed.path

    # 获取当前时间（RFC1123格式）
    now = datetime.datetime.now()
    date = format_date_time(mktime(now.timetuple()))

    # 构造签名字符串
    signature_origin = f"host: {host}\ndate: {date}\n{method} {path} HTTP/1.1"

    # 计算签名
    signature_sha = hmac.new(
        api_secret.encode('utf-8'),
        signature_origin.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    signature_sha = base64.b64encode(signature_sha).decode('utf-8')

    # 构造authorization
    authorization_origin = (
        f'api_key="{api_key}", '
        f'algorithm="hmac-sha256", '
        f'headers="host date request-line", '
        f'signature="{signature_sha}"'
    )
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')

    # 构造查询参数
    query_params = {
        "authorization": authorization,
        "date": date,
        "host": host
    }

    # 构建最终的URL
    query_string = urlencode(query_params)
    return f"{request_url}?{query_string}"


class TTSClient:
    """TTS客户端（重构版）- 支持多种提供商"""

    def __init__(self, provider: str = None, disable_internal_playback: bool = False):
        """
        初始化TTS客户端

        Args:
            provider: 提供商名称，如 'xfyun', 'azure'
            disable_internal_playback: 是否禁用内部播放（避免双重播放）
        """
        print("=" * 60)
        print("初始化 TTS 客户端")
        print("=" * 60)

        # 初始化管理器
        self.config = ConfigManager()
        self.logger = LogManager(self.config, "TTS")
        self.cache_manager = CacheManager(self.config, "TTS")

        # 初始化音频流管理器，传递禁用内部播放标志
        self.audio_stream = AudioStreamManager(self.config, disable_internal_playback)

        # 确定提供商
        self.provider = provider or self.config.get('tts.provider', 'xfyun')
        self.disable_internal_playback = disable_internal_playback

        # 获取提供商特定配置
        self._load_provider_config()

        # 流式处理状态
        self.is_streaming = False
        self.current_stream_id = None
        self.stream_callback = None
        self.ws = None
        self.ws_thread = None
        self.completion_event = threading.Event()
        self.stream_lock = threading.Lock()
        self.stream_error = None
        self.current_text = ""

        logging.info(f"TTS客户端初始化完成 - 提供商: {self.provider}")
        if self.disable_internal_playback:
            logging.info("内部音频播放已禁用（避免双重播放）")

    def _load_provider_config(self):
        """加载提供商配置"""
        if self.provider == 'xfyun':
            self.app_id = self.config.get('TTS_APP_ID')
            self.api_key = self.config.get('TTS_API_KEY')
            self.api_secret = self.config.get('TTS_API_SECRET')

            provider_config = self.config.get_json('tts.xfyun')
            self.vcn = provider_config.get('vcn', 'x6_lingyuyan_pro')
            self.speed = int(provider_config.get('speed', 45))
            self.volume = int(provider_config.get('volume', 50))
            self.pitch = int(provider_config.get('pitch', 57))
            self.endpoint = provider_config.get('endpoint')

            # 验证必要配置
            self._validate_xfyun_config()

        elif self.provider == 'azure':
            # Azure配置
            provider_config = self.config.get_json('tts.azure')
            self.region = provider_config.get('region')
            self.voice = provider_config.get('voice')
            self.endpoint = provider_config.get('endpoint')
            self.api_key = provider_config.get('api_key')
        else:
            raise ValueError(f"不支持的TTS提供商: {self.provider}")

    def _validate_xfyun_config(self):
        """验证讯飞配置"""
        if not all([self.app_id, self.api_key, self.api_secret, self.endpoint]):
            missing = []
            if not self.app_id: missing.append("TTS_APP_ID")
            if not self.api_key: missing.append("TTS_API_KEY")
            if not self.api_secret: missing.append("TTS_API_SECRET")
            if not self.endpoint: missing.append("endpoint")
            raise ValueError(f"缺少必要TTS配置: {', '.join(missing)}")

    def _clean_text_for_tts(self, text: str) -> str:
        """清理文本，移除不适合TTS的内容"""
        if not text:
            return ""

        cleaned = text

        # 移除代码块
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
        cleaned = re.sub(r'`.*?`', '', cleaned)

        # 移除链接
        cleaned = re.sub(r'\[.*?\]\(.*?\)', '', cleaned)

        # 移除特殊字符，保留中英文、数字、标点
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；："\'（）《》.,!?;:"\'()\-\n]', '', cleaned)

        # 合并多个空格
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # 合并多个换行
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)

        return cleaned.strip()

    def text_to_speech_stream(self, text: str, stream_callback: Callable[[bytes], None] = None) -> bool:
        """
        文本转语音流式合成

        Args:
            text: 要合成的文本
            stream_callback: 音频数据回调函数

        Returns:
            bool: 是否成功启动合成
        """
        with self.stream_lock:
            if self.is_streaming:
                logging.warning("已有流式合成在进行中")
                return False

            if not text:
                logging.warning("未提供要合成的文本")
                return False

            # 清理文本
            cleaned_text = self._clean_text_for_tts(text)
            if not cleaned_text:
                logging.warning("清理后的文本为空")
                return False

            logging.info(f"开始流式合成文本（长度 {len(cleaned_text)}）")

            # 设置流式处理状态
            self.stream_callback = stream_callback
            self.completion_event.clear()
            self.audio_stream.clear_queue()
            self.stream_error = None
            self.is_streaming = True
            self.current_stream_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_text = cleaned_text

            # 根据提供商启动合成
            if self.provider == 'xfyun':
                return self._start_xfyun_stream(cleaned_text)
            elif self.provider == 'azure':
                return self._start_azure_stream(cleaned_text)
            else:
                logging.error(f"不支持的提供商: {self.provider}")
                return False

    def _start_xfyun_stream(self, text: str) -> bool:
        """启动讯飞流式合成"""
        try:
            # 生成WebSocket认证URL
            ws_url = assemble_ws_auth_url(
                self.endpoint,
                "GET",
                self.api_key,
                self.api_secret
            )

            logging.info(f"WebSocket URL 已生成，端点: {self.endpoint}")

        except Exception as e:
            logging.error(f"生成WebSocket URL失败: {e}")
            return False

        # WebSocket回调函数
        def on_message(ws, message):
            try:
                message = json.loads(message)
                code = message["header"]["code"]

                if "payload" in message:
                    audio = message["payload"]["audio"]['audio']
                    audio_data = base64.b64decode(audio)
                    status = message["payload"]['audio']["status"]

                    if status == 2:
                        logging.info("TTS流式合成完成")
                        with self.stream_lock:
                            self.is_streaming = False

                        # 等待播放完成（如果启用了内部播放）
                        self._wait_for_playback_completion()
                        self.completion_event.set()
                        ws.close()

                    if code != 0:
                        errMsg = message.get("message", "")
                        logging.error(f"调用错误:{errMsg} 错误码:{code}")
                        with self.stream_lock:
                            self.is_streaming = False
                            self.stream_error = f"API错误 {code}: {errMsg}"
                        self.completion_event.set()
                    else:
                        self._handle_audio_chunk(audio_data)

            except Exception as e:
                logging.error(f"接收消息时解析异常: {e}")
                with self.stream_lock:
                    self.is_streaming = False
                    self.stream_error = f"解析异常: {e}"
                self.completion_event.set()

        def on_error(ws, error):
            logging.error(f"WebSocket错误: {error}")
            with self.stream_lock:
                self.is_streaming = False
                self.stream_error = f"WebSocket错误: {error}"
            self.completion_event.set()

        def on_close(ws, close_status_code, close_msg):
            logging.info(f"WebSocket连接已关闭，状态码: {close_status_code}, 消息: {close_msg}")
            with self.stream_lock:
                self.is_streaming = False
            self.completion_event.set()

        def on_open(ws):
            def run(*args):
                params = self._create_xfyun_params(text)
                data = json.dumps(params)
                logging.info("正在发送文本数据到TTS服务...")
                ws.send(data)

            thread.start_new_thread(run, ())

        # 设置WebSocket
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self.ws.on_open = on_open

        logging.info("正在连接TTS WebSocket服务器...")

        # 启动WebSocket线程
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True,
            name="TTS-WebSocket"
        )
        self.ws_thread.start()

        return True

    def _create_xfyun_params(self, text: str) -> Dict:
        """创建讯飞API参数"""
        return {
            "header": {"app_id": self.app_id, "status": 2},
            "parameter": {
                "tts": {
                    "vcn": self.vcn,
                    "volume": self.volume,
                    "rhy": 0,
                    "speed": self.speed,
                    "pitch": self.pitch,
                    "bgs": 0,
                    "reg": 0,
                    "rdn": 0,
                    "audio": {
                        "encoding": "raw",
                        "sample_rate": 24000,
                        "channels": 1,
                        "bit_depth": 16,
                        "frame_size": 0
                    }
                }
            },
            "payload": {
                "text": {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "plain",
                    "status": 2,
                    "seq": 0,
                    "text": str(base64.b64encode(text.encode('utf-8')), "UTF8")
                }
            }
        }

    def _start_azure_stream(self, text: str) -> bool:
        """启动Azure流式合成（示例实现）"""
        logging.info(f"Azure TTS流式合成（暂未实现）: {text[:50]}...")
        # 这里需要根据Azure TTS API实现
        # 由于Azure实现较复杂，这里只提供框架
        return False

    def _wait_for_playback_completion(self):
        """等待音频播放完成（如果启用了内部播放）"""
        if not self.disable_internal_playback and self.audio_stream.is_playing:
            logging.info("等待音频播放完成...")
            for i in range(50):  # 最多等待5秒
                if not self.audio_stream.is_playing:
                    break
                time.sleep(0.1)

        # 停止播放
        self.audio_stream.stop_playback()

    def _handle_audio_chunk(self, audio_data: bytes):
        """处理音频数据块"""
        try:
            # 可选：保存音频到缓存
            if self.config.get('cache.save_audio', True):
                self.cache_manager.save_audio(audio_data)

            # 添加到音频流管理器
            self.audio_stream.add_audio_chunk(audio_data)

            # 调用用户提供的回调函数
            if self.stream_callback:
                try:
                    self.stream_callback(audio_data)
                except Exception as e:
                    logging.error(f"调用流式回调时出错: {e}")

        except Exception as e:
            logging.error(f"处理音频数据块时出错: {e}")

    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """
        等待流式合成完成

        Args:
            timeout: 超时时间（秒）

        Returns:
            bool: 是否成功完成
        """
        success = self.completion_event.wait(timeout)
        if self.stream_error:
            logging.error(f"流式合成过程中发生错误: {self.stream_error}")
            return False
        return success

    def stop_stream(self):
        """停止流式合成"""
        with self.stream_lock:
            if not self.is_streaming:
                return

            logging.info("正在停止流式合成...")

            # 关闭WebSocket连接
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    logging.error(f"关闭WebSocket时出错: {e}")

            self.is_streaming = False

        # 停止音频播放
        self.audio_stream.stop_playback()

        # 设置完成事件
        self.completion_event.set()

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "provider": self.provider,
            "is_streaming": self.is_streaming,
            "audio_queue_size": self.audio_stream.get_queue_size(),
            "is_playing": self.audio_stream.is_playing,
            "last_error": self.stream_error,
            "disable_internal_playback": self.disable_internal_playback,
            "config": {
                "vcn": self.vcn if hasattr(self, 'vcn') else None,
                "speed": self.speed if hasattr(self, 'speed') else None,
                "volume": self.volume if hasattr(self, 'volume') else None,
                "pitch": self.pitch if hasattr(self, 'pitch') else None
            },
            "audio_stream_status": self.audio_stream.get_status()
        }

    def enable_internal_playback(self, enable: bool = True) -> bool:
        """
        启用或禁用内部音频播放

        Args:
            enable: 是否启用内部播放

        Returns:
            bool: 操作是否成功
        """
        if enable and self.disable_internal_playback:
            logging.warning("无法启用内部播放，客户端初始化时已禁用")
            return False

        return self.audio_stream.enable_playback(enable)

    def get_audio_queue(self):
        """获取音频队列（供外部播放器使用）"""
        return self.audio_stream