#!/usr/bin/env python3
"""
Airis - LLM-TTS 流处理交互系统
集成STM/LTM记忆功能
版本: 2.0.0
作者: Zostime

主程序 - 重构版
包含完整的错误处理和模块化结构
"""

import os
import sys
import threading
import time
import queue
import json
import signal
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 全局日志记录器
logger = None


# ===============================
# 错误处理函数
# ===============================
class ErrorHandler:
    """全局错误处理器"""

    @staticmethod
    def handle_exception(e: Exception, context: str = "", level: str = "error") -> Dict[str, Any]:
        """
        统一异常处理函数

        Args:
            e: 异常对象
            context: 异常发生的上下文描述
            level: 日志级别 (error, warning, critical)

        Returns:
            Dict: 错误信息字典
        """
        error_info = {
            "type": type(e).__name__,
            "message": str(e),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc()
        }

        # 根据级别记录日志
        log_method = getattr(logger, level, logger.error)
        log_method(f"错误 [{context}]: {type(e).__name__}: {str(e)}")

        # 仅在调试模式或错误级别较高时记录完整堆栈
        if level == "error" or level == "critical":
            logger.debug(f"完整堆栈跟踪:\n{error_info['traceback']}")

        return error_info

    @staticmethod
    def safe_execute(func: Callable, *args, **kwargs) -> Any:
        """
        安全执行函数，捕获并记录所有异常

        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            Any: 函数返回值或None（如果出错）
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.handle_exception(e, f"执行函数 {func.__name__}")
            return None

    @staticmethod
    def check_dependencies() -> bool:
        """
        检查必要的依赖包

        Returns:
            bool: 依赖是否满足
        """
        required_packages = ["pyaudio"]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"缺少必要的依赖包: {missing_packages}")
            print(f"[错误] 缺少必要的依赖包: {missing_packages}")
            print("请运行以下命令安装依赖:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

        return True


# ===============================
# 音频队列管理器
# ===============================
class AudioQueueManager:
    """增强版音频队列管理器"""

    def __init__(self, buffer_duration_ms: int = 500):
        self.audio_queue = queue.Queue(maxsize=200)  # 增加队列容量
        self.is_playing = False
        self.total_bytes_received = 0
        self.playback_complete_event = threading.Event()
        self.has_data_event = threading.Event()  # 用于信号数据到达
        self.streaming_active = False  # 流式合成是否活跃
        self.buffer_duration_ms = buffer_duration_ms

        # 统计信息
        self.chunks_received = 0
        self.last_receive_time = 0

    def add_audio_chunk(self, audio_data: bytes) -> bool:
        """添加音频数据到队列"""
        if not audio_data:
            return False

        try:
            self.audio_queue.put(audio_data)
            self.total_bytes_received += len(audio_data)
            self.chunks_received += 1
            self.last_receive_time = time.time()

            # 设置数据到达信号
            self.has_data_event.set()
            self.playback_complete_event.clear()

            return True
        except Exception as e:
            logging.error(f"添加音频数据到队列失败: {e}")
            return False

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """从队列获取音频数据"""
        try:
            # 如果有数据信号，立即获取
            if self.has_data_event.is_set():
                return self.audio_queue.get(timeout=0.01)
            else:
                return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception as e:
            logging.error(f"从队列获取音频数据失败: {e}")
            return None

    def mark_streaming_started(self):
        """标记流式合成开始"""
        self.streaming_active = True
        self.has_data_event.clear()
        self.playback_complete_event.clear()
        logging.info(f"音频队列: 流式合成已开始")

    def mark_streaming_ended(self):
        """标记流式合成结束"""
        self.streaming_active = False
        logging.info(f"音频队列: 流式合成已结束，共收到 {self.chunks_received} 个数据块")

    def clear(self):
        """清空队列"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.total_bytes_received = 0
        self.chunks_received = 0
        self.has_data_event.clear()
        self.playback_complete_event.set()

    def mark_playback_complete(self):
        """标记播放完成"""
        self.playback_complete_event.set()

    def wait_for_playback_complete(self, timeout: float = 10.0) -> bool:
        """等待播放完成"""
        return self.playback_complete_event.wait(timeout)

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """等待数据到达"""
        return self.has_data_event.wait(timeout)

    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.audio_queue.qsize()

    def get_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "queue_size": self.get_queue_size(),
            "is_playing": self.is_playing,
            "total_bytes_received": self.total_bytes_received,
            "chunks_received": self.chunks_received,
            "streaming_active": self.streaming_active,
            "has_data": self.has_data_event.is_set(),
            "playback_complete": self.playback_complete_event.is_set()
        }


# ===============================
# 外部音频播放器
# ===============================
try:
    import pyaudio
    import struct
    import math

    PYAUDIO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyAudio导入失败: {e}")
    PYAUDIO_AVAILABLE = False


class ExternalAudioPlayer:
    """增强版外部音频播放器 - 支持持续播放和缓冲区"""

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1024
        if PYAUDIO_AVAILABLE:
            self.format = pyaudio.paInt16
        self.bytes_per_sample = 2

        # 播放控制
        self.p = None
        self.stream = None
        self.is_playing = False
        self.playback_thread = None
        self.shutdown_flag = threading.Event()
        self.pause_flag = threading.Event()  # 暂停标志
        self.pause_flag.set()  # 默认不暂停

        # 音频队列管理器引用
        self.audio_queue_manager = None

        # 统计信息
        self.total_bytes_played = 0
        self.playback_start_time = 0
        self.playback_end_time = 0
        self.consecutive_empty_count = 0
        self.max_consecutive_empty = 50  # 最多连续空50次（约5秒）

        # 缓冲区设置
        self.pre_buffer_duration = 0.5  # 预缓冲0.5秒
        self.min_buffer_duration = 0.3  # 最小缓冲区0.3秒
        self.max_wait_for_data = 10.0  # 最大等待数据时间10秒

        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio不可用，音频播放器将无法工作")
            return

        self._init_pyaudio()

    def _init_pyaudio(self) -> bool:
        """初始化PyAudio"""
        try:
            self.p = pyaudio.PyAudio()
            logger.info(f"外部音频播放器初始化成功")
            return True
        except Exception as e:
            logger.error(f"PyAudio初始化失败: {e}")
            return False

    def play_from_queue(self, audio_queue_manager: AudioQueueManager) -> bool:
        """
        从音频队列播放音频

        Args:
            audio_queue_manager: 音频队列管理器

        Returns:
            bool: 是否成功开始播放
        """
        if not self.p or self.is_playing or self.shutdown_flag.is_set():
            return False

        if not audio_queue_manager:
            logger.error("音频队列管理器为空")
            return False

        self.audio_queue_manager = audio_queue_manager
        self.is_playing = True
        self.playback_start_time = time.time()
        self.shutdown_flag.clear()
        self.pause_flag.clear()  # 清除暂停标志
        self.consecutive_empty_count = 0

        # 标记队列为播放状态
        self.audio_queue_manager.is_playing = True
        self.audio_queue_manager.playback_complete_event.clear()

        # 启动播放线程
        self.playback_thread = threading.Thread(
            target=self._enhanced_playback_loop,
            daemon=True,
            name="EnhancedAudioPlayer"
        )
        self.playback_thread.start()

        logger.info(f"外部音频播放器开始播放，缓冲区: {self.pre_buffer_duration}秒")
        return True

    def _enhanced_playback_loop(self):
        """增强版播放循环 - 支持缓冲和持续播放"""
        try:
            # 打开音频输出流
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )

            # 初始缓冲：等待一些数据积累
            self._initial_buffer()

            logger.info("音频播放循环开始")

            # 主播放循环
            while self.is_playing and not self.shutdown_flag.is_set():
                # 检查是否需要暂停
                if self.pause_flag.is_set():
                    time.sleep(0.1)
                    continue

                # 从队列获取音频数据
                audio_data = self._get_audio_with_buffer_strategy()

                if audio_data:
                    # 播放音频数据
                    self._write_audio_data(audio_data)
                    self.consecutive_empty_count = 0  # 重置空计数器
                else:
                    # 处理无数据情况
                    if not self._handle_no_audio_data():
                        break  # 应该结束播放

            # 播放完成
            self._finish_playback()

        except Exception as e:
            logger.error(f"音频播放循环出错: {e}")
            ErrorHandler.handle_exception(e, "音频播放循环")
        finally:
            self._cleanup_stream()
            self.is_playing = False
            if self.audio_queue_manager:
                self.audio_queue_manager.is_playing = False

    def _initial_buffer(self):
        """初始缓冲 - 等待一定量的数据积累"""
        buffer_start = time.time()
        target_buffer_bytes = int(self.sample_rate * 2 * self.pre_buffer_duration)  # 2字节/样本

        accumulated_bytes = 0
        accumulated_chunks = []

        logger.info(f"开始初始缓冲，目标: {self.pre_buffer_duration}秒 ({target_buffer_bytes}字节)")

        # 等待数据到达
        if self.audio_queue_manager:
            data_wait_start = time.time()
            while (time.time() - data_wait_start) < 3.0:  # 最多等3秒
                if self.audio_queue_manager.has_data_event.is_set():
                    break
                time.sleep(0.1)

        # 收集初始缓冲数据
        while accumulated_bytes < target_buffer_bytes:
            # 检查超时
            if (time.time() - buffer_start) > 5.0:
                logger.warning(f"初始缓冲超时，已缓冲 {accumulated_bytes} 字节")
                break

            # 获取数据
            chunk = self._get_audio_chunk_with_timeout(0.1)
            if chunk:
                accumulated_bytes += len(chunk)
                accumulated_chunks.append(chunk)
            else:
                # 短暂等待新数据
                time.sleep(0.05)

        # 将缓冲的数据写入流
        if accumulated_chunks:
            combined_data = b''.join(accumulated_chunks)
            self._write_audio_data(combined_data)
            logger.info(f"初始缓冲完成: {accumulated_bytes} 字节, 耗时: {time.time() - buffer_start:.2f}秒")
        else:
            logger.warning("初始缓冲未收集到任何数据")

    def _get_audio_with_buffer_strategy(self) -> Optional[bytes]:
        """使用缓冲区策略获取音频数据"""
        # 尝试立即获取数据
        audio_data = self.audio_queue_manager.get_audio_chunk(timeout=0.05)

        if audio_data:
            return audio_data

        # 如果没有数据，检查流式合成是否仍在进行
        if self.audio_queue_manager.streaming_active:
            # 流式合成仍在进行，等待更多数据
            if self.audio_queue_manager.wait_for_data(timeout=0.2):
                # 有新数据到达，重新尝试获取
                return self.audio_queue_manager.get_audio_chunk(timeout=0.05)
            else:
                # 等待超时，增加空计数器
                self.consecutive_empty_count += 1
                return None
        else:
            # 流式合成已结束，尝试获取剩余数据
            audio_data = self.audio_queue_manager.get_audio_chunk(timeout=0.1)
            if audio_data:
                self.consecutive_empty_count = 0
            else:
                self.consecutive_empty_count += 1
            return audio_data

    def _handle_no_audio_data(self) -> bool:
        """处理无音频数据的情况"""
        self.consecutive_empty_count += 1

        # 检查是否应该结束播放
        should_continue = True

        if self.consecutive_empty_count > self.max_consecutive_empty:
            # 长时间无数据，检查流式合成状态
            if not self.audio_queue_manager.streaming_active:
                # 流式合成已结束且队列为空，结束播放
                queue_size = self.audio_queue_manager.get_queue_size()
                if queue_size == 0:
                    logger.info(f"流式合成已结束且队列为空，结束播放 (连续空: {self.consecutive_empty_count})")
                    should_continue = False
                else:
                    logger.info(f"队列中仍有 {queue_size} 个数据块，继续等待")
            else:
                # 流式合成仍在进行，但长时间无数据，可能有问题
                logger.warning(f"流式合成进行中，但长时间无数据 (连续空: {self.consecutive_empty_count})")

                # 检查是否等待超时
                if (time.time() - self.playback_start_time) > self.max_wait_for_data:
                    logger.warning(f"等待数据超时 ({self.max_wait_for_data}秒)，结束播放")
                    should_continue = False

        # 短暂休眠避免忙等待
        if should_continue:
            time.sleep(0.1)

        return should_continue

    def _get_audio_chunk_with_timeout(self, timeout: float) -> Optional[bytes]:
        """带超时的获取音频数据"""
        try:
            return self.audio_queue_manager.get_audio_chunk(timeout=timeout)
        except Exception as e:
            logger.error(f"获取音频数据块失败: {e}")
            return None

    def _write_audio_data(self, audio_data: bytes):
        """写入音频数据到输出流"""
        try:
            if self.stream and audio_data:
                self.stream.write(audio_data)
                self.total_bytes_played += len(audio_data)
        except Exception as e:
            logger.error(f"写入音频数据失败: {e}")

    def _finish_playback(self):
        """完成播放"""
        self.playback_end_time = time.time()
        playback_duration = self.playback_end_time - self.playback_start_time

        # 标记播放完成
        if self.audio_queue_manager:
            self.audio_queue_manager.is_playing = False
            self.audio_queue_manager.mark_playback_complete()

        logger.info(f"音频播放完成，时长: {playback_duration:.2f}秒，总字节: {self.total_bytes_played}")

    def _cleanup_stream(self):
        """清理音频流"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            logger.error(f"清理音频流时出错: {e}")

    def pause(self):
        """暂停播放"""
        self.pause_flag.set()
        logger.info("音频播放已暂停")

    def resume(self):
        """恢复播放"""
        self.pause_flag.clear()
        logger.info("音频播放已恢复")

    def stop(self):
        """停止播放"""
        self.shutdown_flag.set()
        self.pause_flag.set()  # 确保暂停
        self.is_playing = False

        # 等待播放线程结束
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)

        # 清理音频流
        self._cleanup_stream()

        # 终止PyAudio
        if self.p:
            try:
                self.p.terminate()
                self.p = None
            except Exception as e:
                logger.error(f"终止PyAudio时出错: {e}")

        logger.info("外部音频播放器已停止")

    def test_playback(self, duration: float = 1.0, frequency: float = 440.0) -> bool:
        """测试音频播放"""
        if not self.p:
            return False

        try:
            # 生成测试音频（正弦波）
            test_audio = self._generate_test_tone(duration, frequency)

            # 临时打开流并播放
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )

            stream.write(test_audio)

            stream.stop_stream()
            stream.close()

            logger.info(f"测试音播放成功，频率: {frequency}Hz, 时长: {duration}秒")
            return True
        except Exception as e:
            logger.error(f"测试音频播放失败: {e}")
            return False

    def _generate_test_tone(self, duration: float, frequency: float) -> bytes:
        """生成测试音频（正弦波）"""
        frames = int(self.sample_rate * duration)
        audio_data = []

        for i in range(frames):
            value = int(32767.0 * 0.3 * math.sin(2 * math.pi * frequency * i / self.sample_rate))
            audio_data.append(value)

        return struct.pack('<' + 'h' * len(audio_data), *audio_data)

    def get_status(self) -> Dict[str, Any]:
        """获取播放器状态"""
        return {
            "is_playing": self.is_playing,
            "total_bytes_played": self.total_bytes_played,
            "has_pyaudio": self.p is not None,
            "has_stream": self.stream is not None,
            "thread_alive": self.playback_thread is not None and self.playback_thread.is_alive(),
            "consecutive_empty": self.consecutive_empty_count,
            "paused": self.pause_flag.is_set()
        }


# ===============================
# 模拟类（用于测试）
# ===============================
class MockLLMClient:
    """模拟LLM客户端"""

    def __init__(self):
        self.provider = "mock"
        self.model = "mock-chat"

    def chat_stream(self, messages):
        """模拟流式聊天"""
        responses = [
            "你好！我是Airis，一个AI助手。",
            "我注意到你正在测试系统。",
            "一切看起来都很正常！",
            "祝您使用愉快！"
        ]

        for response in responses:
            yield response
            time.sleep(0.5)


class MockMemoryManager:
    """模拟记忆管理器"""

    def __init__(self, *args, **kwargs):
        print("[警告] 使用模拟的记忆管理器（记忆功能已禁用）")
        self.enabled = False

    def process_conversation(self, *args, **kwargs):
        return {"stm_updated": False, "ltm_updated": False}

    def get_context(self, *args, **kwargs):
        return ""

    def search_memories(self, *args, **kwargs):
        return {"error": "记忆功能未启用"}

    def get_stats(self, *args, **kwargs):
        return {"error": "记忆功能未启用"}

    def clear_memory(self, *args, **kwargs):
        return False

    def consolidate_session(self, *args, **kwargs):
        return None


# ===============================
# 系统主类
# ===============================
class EnhancedStreamCoordinator:
    """增强版协调器 - 集成记忆功能"""

    def __init__(self, config_path: str = None):
        """
        初始化增强版协调器

        Args:
            config_path: 配置文件路径
        """
        self.project_root = Path(__file__).parent
        self.shutdown_flag = threading.Event()

        # 设置日志
        self._setup_logging()

        # 初始化音频队列管理器
        self.audio_queue_manager = AudioQueueManager()

        # 加载配置文件
        if config_path is None:
            config_path = self.project_root / "Files" / "config" / "config.json"
        self.config = self._load_config(config_path)

        # 检查功能开关
        self.tts_enabled = self.config.get("tts", {}).get("enabled", True)
        self.memory_enabled = self.config.get("memory", {}).get("enabled", True)

        # 初始化组件
        self._init_components()

        # 流式处理状态
        self.is_processing = False
        self.current_task_id = None
        self.processing_lock = threading.Lock()
        self.active_threads = []

        # 消息队列
        self.message_queue = queue.Queue()
        self.response_buffer = ""

        # 回调函数
        self.callbacks = {
            'llm_chunk': None,
            'tts_audio': None,
            'complete': None,
            'memory_update': None,
            'error': None
        }

        # 统计信息
        self.stats = {
            "llm_calls": 0,
            "tts_calls": 0,
            "memory_updates": 0,
            "total_chars": 0,
            "total_audio_ms": 0,
            "tts_enabled": self.tts_enabled,
            "memory_enabled": self.memory_enabled,
            "start_time": time.time(),
            "total_tasks": 0,
            "failed_tasks": 0,
            "audio_bytes_received": 0
        }

        # 会话历史
        self.conversation_history = []
        self.max_history = self.config.get("streaming", {}).get("conversation_history", 50)

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Airis 增强版协调器初始化完成")
        self._print_system_info()

    def _setup_logging(self):
        """设置日志系统"""
        log_dir = self.project_root / "Files" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 配置全局日志记录器
        global logger
        logger = logging.getLogger("Airis")
        logger.setLevel(logging.INFO)

        # 如果已有处理器，则不重复添加
        if not logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(log_dir / "system.log", encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 格式化器
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

    def _print_system_info(self):
        """打印系统信息"""
        print("\n" + "=" * 60)
        print("Airis 系统初始化完成！")
        print("=" * 60)
        print(f"系统状态:")
        print(f"  LLM: {getattr(self.llm_client, 'provider', 'unknown').upper()}")
        print(f"  TTS: {'启用' if self.tts_enabled else '禁用'}")
        print(f"  记忆系统: {'启用' if self.memory_enabled else '禁用'}")
        print(f"  会话历史: 最多保存 {self.max_history} 条")
        print("=" * 60)

    def _init_components(self):
        """初始化各个组件"""
        try:
            # 初始化LLM客户端
            print("[初始化] 正在初始化LLM客户端...")
            try:
                # 尝试从core.llm.client导入
                from core.llm.client import LLMClient
                self.llm_client = LLMClient()
                print(f"[成功] LLM客户端初始化完成 - 提供商: {getattr(self.llm_client, 'provider', 'unknown')}")
            except ImportError as e:
                print(f"[警告] LLM客户端导入失败: {e}")
                print("[信息] 使用模拟LLM客户端")
                self.llm_client = MockLLMClient()
            except Exception as e:
                ErrorHandler.handle_exception(e, "LLM客户端初始化")
                print("[信息] 使用模拟LLM客户端")
                self.llm_client = MockLLMClient()

            # 初始化TTS客户端（如果启用）
            if self.tts_enabled:
                print("[初始化] 正在初始化TTS客户端...")
                try:
                    # 导入重构后的TTSClient
                    from core.tts.client import TTSClient

                    # 创建TTS客户端时，禁用内部播放（避免双重播放）
                    tts_provider = self.config.get("tts", {}).get("provider", "xfyun")
                    self.tts_client = TTSClient(
                        provider=tts_provider,
                        disable_internal_playback=True  # 关键：禁用TTS内部播放
                    )

                    print(f"[成功] TTS客户端初始化完成 - 提供商: {getattr(self.tts_client, 'provider', 'unknown')}")
                    print("[配置] TTS客户端内部播放已禁用（避免双重播放）")

                except ImportError as e:
                    print(f"[错误] TTS客户端导入失败: {e}")
                    print("[信息] TTS功能将被禁用")
                    self.tts_enabled = False
                    self.tts_client = None
                except Exception as e:
                    ErrorHandler.handle_exception(e, "TTS客户端初始化")
                    print("[信息] TTS功能将被禁用")
                    self.tts_enabled = False
                    self.tts_client = None
            else:
                print("[信息] TTS功能已禁用，跳过初始化")
                self.tts_client = None

            # 初始化记忆管理器（如果启用）
            if self.memory_enabled:
                print("[初始化] 正在初始化记忆系统...")
                try:
                    from core.memory.manager import MemoryManager
                    memory_config_path = self.project_root / "Files" / "config" / "memory_config.json"
                    self.memory_manager = MemoryManager(str(memory_config_path))
                    print("[成功] 记忆系统初始化完成")
                except ImportError:
                    print("[警告] 记忆系统导入失败，使用模拟管理器")
                    self.memory_manager = MockMemoryManager()
                except Exception as e:
                    ErrorHandler.handle_exception(e, "记忆系统初始化")
                    print("[信息] 记忆功能将被禁用")
                    self.memory_enabled = False
                    self.memory_manager = None
            else:
                print("[信息] 记忆功能已禁用，跳过初始化")
                self.memory_manager = None

        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            print(f"[错误] 初始化失败: {e}")
            raise

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"配置文件不存在: {config_path}")
                print(f"[警告] 配置文件不存在，使用默认配置")
                return self._get_default_config()

            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            logger.info(f"配置文件加载成功: {config_path}")
            return config

        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            print(f"[错误] 配置文件格式错误，请检查JSON语法")
            print(f"   错误详情: {e}")
            return self._get_default_config()

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            print(f"[错误] 加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "llm": {
                "provider": "deepseek",
                "deepseek": {
                    "model": "deepseek-chat",
                    "temperature": 0.7,
                    "max_tokens": 4096
                }
            },
            "tts": {
                "enabled": True,
                "provider": "xfyun",
                "xfyun": {
                    "vcn": "x6_lingyuyan_pro",
                    "speed": 45,
                    "volume": 50,
                    "pitch": 57
                }
            },
            "memory": {
                "enabled": True
            },
            "streaming": {
                "conversation_history": 50,
                "realtime_playback": False
            }
        }

    def register_callback(self, callback_type: str, callback_func: Callable):
        """
        注册回调函数

        Args:
            callback_type: 回调类型 ('llm_chunk', 'tts_audio', 'complete', 'memory_update', 'error')
            callback_func: 回调函数
        """
        if callback_type in self.callbacks:
            self.callbacks[callback_type] = callback_func
            logger.debug(f"注册回调: {callback_type}")
        else:
            logger.warning(f"未知的回调类型: {callback_type}")

    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"收到信号 {signum}，正在关闭系统...")
        print(f"\n[信息] 收到关闭信号，正在清理资源...")
        self.shutdown()
        sys.exit(0)

    def process_user_input(self, user_input: str) -> bool:
        """
        处理用户输入

        Args:
            user_input: 用户输入的文本

        Returns:
            bool: 是否成功启动处理
        """
        with self.processing_lock:
            if self.is_processing:
                self._emit_message("当前已有处理任务进行中，请等待...", "warning")
                return False

            if not user_input or user_input.strip() == "":
                self._emit_message("输入为空，请重新输入", "warning")
                return False

            # 清理输入
            user_input = user_input.strip()

            # 创建任务ID
            self.current_task_id = f"task_{int(time.time())}_{hash(user_input) % 10000:04d}"

            # 添加到会话历史
            self._add_to_history("user", user_input)

            # 显示处理开始
            input_preview = user_input[:60] + "..." if len(user_input) > 60 else user_input
            self._emit_message(f"\n[开始] 开始处理任务: {self.current_task_id}", "info")
            self._emit_message(f"[输入] {input_preview}", "info")
            self._emit_message(
                f"[模式] TTS={'启用' if self.tts_enabled else '禁用'}, 记忆={'启用' if self.memory_enabled else '禁用'}",
                "info")

            # 重置状态
            self.is_processing = True
            self.response_buffer = ""
            self.stats["total_tasks"] += 1

            # 清空音频队列
            self.audio_queue_manager.clear()

            # 创建处理线程
            process_thread = threading.Thread(
                target=self._enhanced_process_chain,
                args=(user_input,),
                daemon=True,
                name=f"Process-{self.current_task_id}"
            )
            self.active_threads.append(process_thread)
            process_thread.start()

            logger.info(f"启动处理任务: {self.current_task_id}, 输入长度: {len(user_input)}")
            return True

    def _enhanced_process_chain(self, user_input: str):
        """
        增强版处理链 - 集成记忆系统

        Args:
            user_input: 用户输入
        """
        task_start_time = time.time()
        llm_response = ""
        success = False

        try:
            # 步骤1: 构建增强版LLM消息
            self._emit_message("[信息] 构建对话上下文...", "info")
            messages = self._build_enhanced_messages(user_input)

            # 步骤2: LLM流式生成
            self._emit_message("[信息] LLM正在生成回答...", "info")
            llm_response = self._process_llm_stream(messages)

            if not llm_response or llm_response.strip() == "":
                self._emit_message("[错误] LLM未生成有效响应", "error")
                raise ValueError("LLM响应为空")

            # 更新统计
            self.stats["llm_calls"] += 1
            self.stats["total_chars"] += len(llm_response)

            # 添加到会话历史
            self._add_to_history("assistant", llm_response)

            self._emit_message(f"[成功] LLM生成完成 (长度: {len(llm_response)}字符)", "success")

            # 步骤3: 更新记忆系统
            if self.memory_enabled and self.memory_manager:
                self._emit_message("[信息] 正在更新记忆系统...", "info")
                memory_result = self._update_memory_system(user_input, llm_response)
                self.stats["memory_updates"] += 1
                self._emit_message(f"[成功] 记忆更新完成", "success")

            # 步骤4: TTS合成（如果启用）
            if self.tts_enabled and self.tts_client:
                self._emit_message("[信息] 开始语音合成...", "info")
                tts_success = self._process_tts_synthesis(llm_response)

                if tts_success:
                    self.stats["tts_calls"] += 1
                    self._emit_message("[成功] 语音合成完成", "success")
                else:
                    self._emit_message("[警告] 语音合成失败，但文本响应已生成", "warning")

            # 步骤5: 显示最终结果
            success = True
            task_duration = time.time() - task_start_time
            self._emit_message(f"\n[成功] 处理完成！耗时: {task_duration:.2f}秒", "success")

            # 显示响应（TTS禁用时）
            if not self.tts_enabled:
                self._display_response(llm_response)

        except Exception as e:
            success = False
            self.stats["failed_tasks"] += 1
            error_msg = f"处理链错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._emit_message(f"[错误] 处理失败: {str(e)}", "error")

            if self.callbacks['error']:
                try:
                    self.callbacks['error'](e, self.current_task_id)
                except Exception as cb_error:
                    logger.error(f"错误回调失败: {cb_error}")

        finally:
            # 完成处理
            self._finish_processing(success, llm_response, task_start_time)

            # 巩固会话记忆（如果启用且成功）
            if success and self.memory_enabled and self.memory_manager:
                try:
                    self._consolidate_session_memory()
                except Exception as e:
                    logger.warning(f"巩固会话记忆失败: {e}")

    def _process_llm_stream(self, messages: List[Dict]) -> str:
        """处理LLM流式生成"""
        llm_response = ""

        try:
            # 获取流式生成器
            stream_gen = self.llm_client.chat_stream(messages)

            # 处理每个流式块
            for chunk in stream_gen:
                if isinstance(chunk, str):
                    llm_response += chunk

                    # 调用LLM回调（实时显示）
                    if self.callbacks['llm_chunk']:
                        try:
                            self.callbacks['llm_chunk'](chunk)
                        except Exception as e:
                            logger.error(f"LLM回调失败: {e}")
                    else:
                        # 如果没有回调，直接打印
                        print(chunk, end="", flush=True)

                    # 更新响应缓冲区
                    self.response_buffer = llm_response

            # 获取最终结果
            try:
                final_result = next(stream_gen)
                if isinstance(final_result, dict):
                    final_content = final_result.get('full_content', llm_response)
                    if final_content and final_content != llm_response:
                        llm_response = final_content
            except StopIteration:
                pass

            # 确保换行
            print()

        except Exception as e:
            logger.error(f"LLM流式处理错误: {e}")
            raise

        return llm_response

    def _update_memory_system(self, user_input: str, llm_response: str) -> Dict[str, Any]:
        """更新记忆系统"""
        try:
            memory_result = self.memory_manager.process_conversation(user_input, llm_response)

            # 调用记忆更新回调
            if self.callbacks['memory_update']:
                try:
                    self.callbacks['memory_update'](memory_result)
                except Exception as e:
                    logger.error(f"记忆回调失败: {e}")

            return memory_result
        except Exception as e:
            logger.error(f"更新记忆系统失败: {e}")
            return {"error": str(e)}

    def _process_tts_synthesis(self, text: str) -> bool:
        """处理TTS合成 - 改进版"""
        try:
            # 标记流式合成开始
            self.audio_queue_manager.mark_streaming_started()

            # 定义TTS回调函数
            def tts_callback(audio_data: bytes):
                """TTS音频数据回调"""
                # 将音频数据添加到队列
                if audio_data and len(audio_data) > 0:
                    self.stats["audio_bytes_received"] += len(audio_data)
                    self.audio_queue_manager.add_audio_chunk(audio_data)

                    # 调用TTS音频回调（外部可以处理播放）
                    if self.callbacks['tts_audio']:
                        try:
                            self.callbacks['tts_audio'](audio_data)
                        except Exception as e:
                            logging.error(f"TTS音频回调失败: {e}")

            # 启动TTS流式合成
            success = self.tts_client.text_to_speech_stream(text, tts_callback)

            if not success:
                logger.error("TTS流式合成启动失败")
                self.audio_queue_manager.mark_streaming_ended()
                return False

            # 等待TTS合成完成（增加超时时间）
            timeout = self.config.get("tts", {}).get("timeout", 120)  # 增加到120秒
            wait_success = self.tts_client.wait_for_completion(timeout)

            # 标记流式合成结束
            self.audio_queue_manager.mark_streaming_ended()

            if not wait_success:
                logger.warning(f"TTS合成等待超时或失败 (超时: {timeout}秒)")

            # 等待音频播放完成（增加等待时间）
            if self.audio_queue_manager.is_playing:
                logger.info("等待音频播放完成...")
                playback_timeout = min(30, len(text) * 0.1)  # 基于文本长度的动态超时
                playback_success = self.audio_queue_manager.wait_for_playback_complete(timeout=playback_timeout)

                if not playback_success:
                    logger.warning(f"音频播放等待超时 (超时: {playback_timeout}秒)")

            return wait_success

        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
            # 确保标记流式合成结束
            self.audio_queue_manager.mark_streaming_ended()
            return False

    def _build_enhanced_messages(self, user_input: str) -> List[Dict]:
        """构建增强版消息，包含记忆上下文"""
        # 基础系统提示
        system_prompt = """
                        你叫Airis,创作者是Zostime.
                        请保持回答简洁,模仿正常聊天语气进行回答：
                        """

        messages = [{"role": "system", "content": system_prompt}]

        # 添加记忆上下文（如果启用）
        if self.memory_enabled and self.memory_manager:
            try:
                memory_context = self.memory_manager.get_context(
                    query=user_input,
                    include_stm=True,
                    include_ltm=True,
                    top_k=3
                )

                if memory_context and memory_context.strip():
                    memory_prompt = f"""【记忆上下文 - 过往对话参考】\n{memory_context}\n\n请基于以上记忆和当前对话进行回答。如果记忆内容相关，请自然地引用或参考。"""
                    messages.append({"role": "system", "content": memory_prompt})
                    logger.debug(f"记忆上下文已添加: {len(memory_context)}字符")
            except Exception as e:
                logger.warning(f"获取记忆上下文失败: {e}")

        # 添加最近的会话历史（如果有）
        if self.conversation_history:
            recent_history = self.conversation_history[-5:]  # 最近5轮对话
            for entry in recent_history:
                if entry["role"] == "user":
                    messages.append({"role": "user", "content": entry["content"]})
                elif entry["role"] == "assistant" and entry["content"]:
                    messages.append({"role": "assistant", "content": entry["content"]})

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        # 记录消息结构
        logger.debug(f"构建消息完成，共 {len(messages)} 条消息")

        return messages

    def _display_response(self, response: str):
        """显示响应文本"""
        print("\n" + "=" * 70)
        print("Airis的回复:")
        print("=" * 70)
        print(response)
        print("=" * 70)

    def _add_to_history(self, role: str, content: str):
        """添加到会话历史"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "task_id": self.current_task_id
        }
        self.conversation_history.append(entry)

        # 维持历史长度限制
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def _consolidate_session_memory(self):
        """巩固会话记忆"""
        if not self.memory_enabled or not self.memory_manager:
            return

        try:
            consolidated_id = self.memory_manager.consolidate_session()
            if consolidated_id:
                logger.info(f"会话记忆已巩固: {consolidated_id}")
        except Exception as e:
            logger.warning(f"巩固会话记忆失败: {e}")

    def _finish_processing(self, success: bool, response: str = None, start_time: float = None):
        """完成处理任务"""
        with self.processing_lock:
            self.is_processing = False

            # 计算处理时间
            duration = 0
            if start_time:
                duration = time.time() - start_time

            # 清理任务ID
            task_id = self.current_task_id
            self.current_task_id = None

            # 调用完成回调
            if self.callbacks['complete']:
                try:
                    self.callbacks['complete'](success, response, task_id, duration)
                except Exception as e:
                    logger.error(f"完成回调失败: {e}")

            # 清理线程列表
            self.active_threads = [t for t in self.active_threads if t.is_alive()]

            logger.info(f"任务完成: {task_id}, 成功: {success}, 耗时: {duration:.2f}秒")

    def _emit_message(self, message: str, msg_type: str = "info"):
        """发送消息（用于UI显示）"""
        print(message)

    def search_memories(self, query: str, search_type: str = "hybrid") -> Dict[str, Any]:
        """
        搜索记忆

        Args:
            query: 搜索查询
            search_type: 搜索类型 (stm, ltm, hybrid)

        Returns:
            Dict[str, Any]: 搜索结果
        """
        if not self.memory_enabled or not self.memory_manager:
            return {"error": "记忆系统未启用"}

        try:
            results = self.memory_manager.search_memories(query, search_type)
            logger.info(
                f"记忆搜索完成: '{query}', 结果数量: {len(results.get('stm_results', [])) + len(results.get('ltm_results', []))}")
            return results
        except Exception as e:
            logger.error(f"记忆搜索失败: {e}")
            return {"error": str(e)}

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计"""
        if not self.memory_enabled or not self.memory_manager:
            return {"error": "记忆系统未启用"}

        try:
            stats = self.memory_manager.get_stats()
            return stats
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {"error": str(e)}

    def clear_memory(self, memory_type: str = "stm") -> bool:
        """
        清除记忆

        Args:
            memory_type: 记忆类型 (stm, ltm, session, all)

        Returns:
            bool: 是否成功
        """
        if not self.memory_enabled or not self.memory_manager:
            logger.warning("记忆系统未启用，无法清除")
            return False

        try:
            self.memory_manager.clear_memory(memory_type)

            if memory_type in ["session", "all"]:
                self.conversation_history.clear()

            logger.info(f"记忆清除成功: {memory_type}")
            return True
        except Exception as e:
            logger.error(f"清除记忆失败: {e}")
            return False

    def toggle_tts(self, enable: bool = None) -> bool:
        """
        切换TTS功能状态

        Args:
            enable: True启用，False禁用，None切换

        Returns:
            bool: 新的TTS状态
        """
        if enable is None:
            new_state = not self.tts_enabled
        else:
            new_state = bool(enable)

        if new_state == self.tts_enabled:
            self._emit_message(f"[信息] TTS已经是{'启用' if new_state else '禁用'}状态", "info")
            return self.tts_enabled

        # 更新状态
        self.tts_enabled = new_state
        self.stats["tts_enabled"] = new_state

        # 如果从禁用切换到启用，需要初始化TTS客户端
        if new_state and not hasattr(self, 'tts_client'):
            try:
                self._emit_message("[信息] 正在初始化TTS客户端...", "info")
                from core.tts.client import TTSClient
                tts_provider = self.config.get("tts", {}).get("provider", "xfyun")

                # 创建TTS客户端，禁用内部播放
                self.tts_client = TTSClient(
                    provider=tts_provider,
                    disable_internal_playback=True
                )

                self._emit_message("[成功] TTS客户端初始化完成", "success")
            except Exception as e:
                self._emit_message(f"[错误] TTS客户端初始化失败: {e}", "error")
                self.tts_enabled = False
                return False
        # 如果从启用切换到禁用，可以释放TTS客户端资源
        elif not new_state and hasattr(self, 'tts_client'):
            self._emit_message("[信息] 正在释放TTS客户端资源...", "info")
            self.tts_client.stop_stream()
            self.tts_client = None
            self._emit_message("[信息] TTS客户端资源已释放", "info")

        self._emit_message(f"[成功] TTS功能已{'启用' if new_state else '禁用'}", "success")
        return self.tts_enabled

    def toggle_memory(self, enable: bool = None) -> bool:
        """
        切换记忆功能状态

        Args:
            enable: True启用，False禁用，None切换

        Returns:
            bool: 新的记忆状态
        """
        if enable is None:
            new_state = not self.memory_enabled
        else:
            new_state = bool(enable)

        if new_state == self.memory_enabled:
            self._emit_message(f"[信息] 记忆系统已经是{'启用' if new_state else '禁用'}状态", "info")
            return self.memory_enabled

        # 更新状态
        self.memory_enabled = new_state
        self.stats["memory_enabled"] = new_state

        # 如果从禁用切换到启用，需要初始化记忆管理器
        if new_state and not hasattr(self, 'memory_manager'):
            try:
                self._emit_message("[信息] 正在初始化记忆系统...", "info")
                from core.memory.manager import MemoryManager
                memory_config_path = self.project_root / "Files" / "config" / "memory_config.json"
                self.memory_manager = MemoryManager(str(memory_config_path))
                self._emit_message("[成功] 记忆系统初始化完成", "success")
            except Exception as e:
                self._emit_message(f"[错误] 记忆系统初始化失败: {e}", "error")
                self.memory_enabled = False
                return False
        # 如果从启用切换到禁用，可以释放记忆管理器资源
        elif not new_state and hasattr(self, 'memory_manager'):
            self._emit_message("[信息] 正在释放记忆系统资源...", "info")
            self.memory_manager = None
            self._emit_message("[信息] 记忆系统资源已释放", "info")

        self._emit_message(f"[成功] 记忆系统已{'启用' if new_state else '禁用'}", "success")
        return self.memory_enabled

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = time.time() - self.stats["start_time"]

        # 获取TTS客户端状态
        tts_client_status = {}
        if self.tts_enabled and hasattr(self, 'tts_client') and self.tts_client:
            try:
                tts_client_status = self.tts_client.get_status()
            except Exception as e:
                tts_client_status = {"error": str(e)}

        # 获取音频队列状态
        audio_queue_status = self.audio_queue_manager.get_status()

        status = {
            "system": {
                "is_processing": self.is_processing,
                "current_task_id": self.current_task_id,
                "tts_enabled": self.tts_enabled,
                "memory_enabled": self.memory_enabled,
                "uptime": uptime,
                "uptime_formatted": self._format_duration(uptime),
                "conversation_history_count": len(self.conversation_history),
                "active_threads": len(self.active_threads)
            },
            "components": {
                "llm": {
                    "provider": getattr(self.llm_client, 'provider', 'unknown'),
                    "model": getattr(self.llm_client, 'model', 'unknown')
                },
                "tts": {
                    "enabled": self.tts_enabled,
                    "provider": getattr(self.tts_client, 'provider', 'unknown') if self.tts_client else None,
                    "client_status": tts_client_status
                },
                "memory": {
                    "enabled": self.memory_enabled
                },
                "audio_queue": audio_queue_status
            },
            "statistics": self.stats.copy()
        }

        # 添加成功率
        if self.stats["total_tasks"] > 0:
            success_rate = (self.stats["total_tasks"] - self.stats["failed_tasks"]) / self.stats["total_tasks"] * 100
            status["statistics"]["success_rate"] = f"{success_rate:.1f}%"

        return status

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """
        获取对话历史

        Args:
            limit: 返回的最大条目数

        Returns:
            List[Dict]: 对话历史
        """
        if not self.conversation_history:
            return []

        start_idx = max(0, len(self.conversation_history) - limit)
        return self.conversation_history[start_idx:]

    def stop_processing(self):
        """停止当前处理任务"""
        with self.processing_lock:
            if not self.is_processing:
                return

            self._emit_message("[警告] 正在停止当前处理...", "warning")

            # 停止TTS合成（如果启用）
            if self.tts_enabled and hasattr(self, 'tts_client') and self.tts_client:
                try:
                    self.tts_client.stop_stream()
                except Exception as e:
                    logger.error(f"停止TTS流失败: {e}")

            # 清空音频队列
            self.audio_queue_manager.clear()

            # 重置状态
            self.is_processing = False
            self.current_task_id = None

            # 清理线程
            for thread in self.active_threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)

            self.active_threads.clear()
            self._emit_message("[信息] 处理已停止", "info")

    def shutdown(self):
        """关闭系统，清理资源"""
        self._emit_message("[信息] 正在关闭系统，请稍候...", "warning")
        self.shutdown_flag.set()

        # 停止当前处理
        self.stop_processing()

        # 等待所有线程完成
        for thread in self.active_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

        # 保存状态（如果需要）
        self._save_state()

        # 记录关闭信息
        uptime = time.time() - self.stats["start_time"]
        logger.info(f"系统关闭，运行时间: {self._format_duration(uptime)}")
        logger.info(f"统计数据: {self.stats}")

        self._emit_message("[信息] 系统已关闭", "info")

    def _save_state(self):
        """保存系统状态"""
        try:
            state_dir = self.project_root / "Files" / "cache" / "system_state"
            state_dir.mkdir(parents=True, exist_ok=True)

            state_file = state_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            state_data = {
                "conversation_history": self.conversation_history,
                "statistics": self.stats,
                "saved_at": datetime.now().isoformat(),
                "version": "2.2.0"
            }

            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)

            logger.info(f"系统状态已保存: {state_file}")
        except Exception as e:
            logger.error(f"保存系统状态失败: {e}")

    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}小时{minutes}分{secs}秒"
        elif minutes > 0:
            return f"{minutes}分{secs}秒"
        else:
            return f"{secs}秒"

    def get_audio_queue(self) -> AudioQueueManager:
        """获取音频队列管理器（用于外部播放音频）"""
        return self.audio_queue_manager


# ===============================
# 用户界面函数
# ===============================
def display_banner():
    """显示程序横幅"""
    print("\n" + "=" * 70)
    print("AIRIS - v2.2.0")
    print("=" * 70)
    print("LLM + TTS + STM & LTM")
    print("创作者: Zostime")
    print("=" * 70)


def display_menu(coordinator: EnhancedStreamCoordinator, audio_player: ExternalAudioPlayer = None):
    """显示主菜单"""
    status = coordinator.get_status()
    tts_status = "启用" if coordinator.tts_enabled else "禁用"
    memory_status = "启用" if coordinator.memory_enabled else "禁用"

    print(f"\n系统状态:")
    print(f"  TTS: {tts_status}")
    print(f"  记忆: {memory_status}")
    print(f"  运行: {status['system']['uptime_formatted']}")
    print(f"  历史: {status['system']['conversation_history_count']} 条")

    # 显示音频状态
    audio_status = status['components']['audio_queue']
    print(f"  音频队列: {audio_status.get('queue_size', 0)} 块")
    print(f"  正在播放: {'是' if audio_status.get('is_playing', False) else '否'}")

    if coordinator.is_processing:
        print(f"  当前任务: {coordinator.current_task_id}")

    # 显示音频播放器状态
    if audio_player:
        player_status = audio_player.get_status()
        print(f"  外部播放器: {'运行中' if player_status['is_playing'] else '空闲'}")

    print("\n" + "-" * 70)
    print("主菜单:")
    print("  1. 开始对话")
    print("  2. 搜索记忆")
    print("  3. 查看系统状态")
    print("  4. 查看记忆统计")
    print("  5. 查看对话历史")
    print("  6. 系统设置")
    print("  7. 测试音频")
    print("  8. 退出程序")
    print("-" * 70)


def system_settings_menu(coordinator: EnhancedStreamCoordinator, audio_player: ExternalAudioPlayer = None):
    """系统设置菜单"""
    while True:
        print("\n" + "-" * 70)
        print("系统设置:")
        print("  1. 切换TTS功能")
        print("  2. 切换记忆系统")
        print("  3. 清除记忆")
        print("  4. 测试TTS功能")
        print("  5. 测试音频播放器")
        print("  6. 返回主菜单")
        print("-" * 70)

        try:
            choice = input("请选择 (1-6): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n返回主菜单")
            break

        if choice == "1":
            new_state = coordinator.toggle_tts()
            print(f"TTS功能已{'启用' if new_state else '禁用'}")

        elif choice == "2":
            new_state = coordinator.toggle_memory()
            print(f"记忆系统已{'启用' if new_state else '禁用'}")

        elif choice == "3":
            print("\n选择清除类型:")
            print("  1. 短期记忆 (STM)")
            print("  2. 长期记忆 (LTM)")
            print("  3. 会话历史")
            print("  4. 全部清除")
            print("  5. 取消")

            try:
                mem_choice = input("请选择 (1-5): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("取消操作")
                continue

            if mem_choice == "1":
                if coordinator.clear_memory("stm"):
                    print("[成功] 短期记忆已清除")
                else:
                    print("[错误] 清除失败")
            elif mem_choice == "2":
                if coordinator.clear_memory("ltm"):
                    print("[成功] 长期记忆已清除")
                else:
                    print("[错误] 清除失败")
            elif mem_choice == "3":
                if coordinator.clear_memory("session"):
                    print("[成功] 会话历史已清除")
                else:
                    print("[错误] 清除失败")
            elif mem_choice == "4":
                confirm = input("确认清除所有记忆？(y/N): ").strip().lower()
                if confirm == 'y':
                    if coordinator.clear_memory("all"):
                        print("[成功] 所有记忆已清除")
                    else:
                        print("[错误] 清除失败")
                else:
                    print("取消操作")

        elif choice == "4":
            # 测试TTS功能
            if not coordinator.tts_enabled:
                print("[错误] TTS功能已禁用，请先启用TTS功能")
                continue

            test_text = input("请输入测试文本（留空使用默认文本）: ").strip()
            if not test_text:
                test_text = "你好，这是一个测试语音。Airis系统正在运行。"

            print(f"[测试] 测试文本: {test_text}")
            print("[测试] 正在测试TTS合成...")

            # 定义测试回调
            audio_received = [0]

            def test_callback(audio_data):
                if audio_data:
                    audio_received[0] += len(audio_data)

            # 临时注册回调
            old_callback = coordinator.callbacks['tts_audio']
            coordinator.callbacks['tts_audio'] = test_callback

            # 启动TTS
            success = coordinator.tts_client.text_to_speech_stream(test_text, test_callback)

            if success:
                print("[测试] TTS合成已启动...")
                # 等待合成完成
                time.sleep(5)

                if audio_received[0] > 0:
                    print(f"[成功] 收到 {audio_received[0]} 字节音频数据")
                else:
                    print("[警告] 未收到音频数据")
            else:
                print("[错误] TTS合成启动失败")

            # 恢复原回调
            coordinator.callbacks['tts_audio'] = old_callback

        elif choice == "5":
            # 测试音频播放器
            if not audio_player:
                print("[错误] 音频播放器未初始化")
                continue

            print("[测试] 正在测试音频播放器...")
            success = audio_player.test_playback(duration=1.0, frequency=440.0)
            if success:
                print("[成功] 测试音播放完成，您应该听到了一个音调")
            else:
                print("[错误] 测试音播放失败")

        elif choice == "6":
            break

        else:
            print("无效选择，请重新输入")


# ===============================
# 主程序入口
# ===============================
def main():
    """主函数"""
    try:
        # 显示横幅
        display_banner()

        # 检查依赖
        print("\n[检查] 正在检查系统依赖...")
        if not ErrorHandler.check_dependencies():
            print("\n[错误] 依赖检查失败，程序无法启动")
            input("\n按Enter键退出...")
            return

        # 初始化协调器
        print("\n[初始化] 正在初始化系统...")
        coordinator = EnhancedStreamCoordinator()

    except Exception as e:
        error_info = ErrorHandler.handle_exception(e, "系统初始化", "critical")
        print(f"\n[致命错误] 系统初始化失败: {e}")
        print("请检查配置文件和日志文件获取更多信息。")
        traceback.print_exc()
        input("\n按Enter键退出...")
        return

    # 初始化外部音频播放器
    audio_player = None
    try:
        if PYAUDIO_AVAILABLE:
            audio_player = ExternalAudioPlayer()
            print("[成功] 外部音频播放器初始化完成")
        else:
            print("[警告] PyAudio不可用，音频播放功能将不可用")
    except Exception as e:
        error_info = ErrorHandler.handle_exception(e, "音频播放器初始化")
        print("[警告] 外部音频播放器初始化失败")

    # 定义回调函数
    def on_llm_chunk(chunk: str):
        print(chunk, end="", flush=True)

    def on_tts_audio(audio_data: bytes):
        # 当收到TTS音频时，启动外部音频播放器（如果还没有启动）
        if audio_player and not audio_player.is_playing:
            # 从队列播放音频
            success = audio_player.play_from_queue(coordinator.get_audio_queue())
            if success:
                print("[音频] 音频播放已启动")
            else:
                print("[警告] 音频播放启动失败")

    def on_complete(success: bool, response: str, task_id: str, duration: float):
        if success:
            print(f"\n\n[成功] 任务 {task_id} 完成，耗时: {duration:.2f}秒")
            if not coordinator.tts_enabled and response:
                print(f"响应长度: {len(response)} 字符")
        else:
            print(f"\n\n[错误] 任务 {task_id} 失败")

    def on_memory_update(result: Dict[str, Any]):
        # 这里可以添加记忆更新处理
        pass

    def on_error(error: Exception, task_id: str):
        print(f"\n[错误] 任务 {task_id} 发生错误: {error}")

    # 注册回调
    coordinator.register_callback("llm_chunk", on_llm_chunk)
    coordinator.register_callback("tts_audio", on_tts_audio)
    coordinator.register_callback("complete", on_complete)
    coordinator.register_callback("memory_update", on_memory_update)
    coordinator.register_callback("error", on_error)

    # 主循环
    try:
        while True:
            try:
                display_menu(coordinator, audio_player)

                try:
                    choice = input("\n请选择 (1-8): ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n\n检测到中断信号")
                    confirm = input("确定要退出吗？(y/N): ").strip().lower()
                    if confirm == 'y':
                        break
                    else:
                        continue

                if choice == "1":
                    # 开始对话
                    try:
                        print("\n请输入您的消息 (输入 'quit' 返回主菜单):")
                        print("提示: 您可以输入多行，以空行结束输入")

                        lines = []
                        while True:
                            try:
                                line = input("> " if not lines else "... ").strip()
                            except (KeyboardInterrupt, EOFError):
                                print("\n输入已取消")
                                lines = []
                                break

                            if not lines and line.lower() == 'quit':
                                print("返回主菜单")
                                break

                            if line == "":
                                if lines:
                                    break
                                else:
                                    continue

                            lines.append(line)

                        if lines:
                            user_input = "\n".join(lines)
                            input_preview = user_input[:80] + "..." if len(user_input) > 80 else user_input
                            print(f"\n正在发送: {input_preview}")

                            # 启动处理
                            success = coordinator.process_user_input(user_input)

                            if success:
                                # 等待处理完成
                                print("\n处理中...", end="", flush=True)
                                dots = 0
                                start_wait = time.time()
                                timeout = 120  # 2分钟超时

                                while coordinator.is_processing:
                                    elapsed = time.time() - start_wait
                                    if elapsed > timeout:
                                        print("\n[超时] 处理时间过长，已超时")
                                        coordinator.stop_processing()
                                        break

                                    print(".", end="", flush=True)
                                    dots += 1
                                    if dots % 40 == 0:
                                        print()

                                    time.sleep(0.5)

                                print()  # 换行
                            else:
                                print("[错误] 处理启动失败")

                    except Exception as e:
                        ErrorHandler.handle_exception(e, "对话处理")
                        print(f"[错误] 对话处理异常: {e}")

                elif choice == "2":
                    # 搜索记忆
                    if not coordinator.memory_enabled:
                        print("[错误] 记忆系统未启用")
                        continue

                    try:
                        query = input("\n请输入搜索关键词: ").strip()
                        if query:
                            print(f"正在搜索: '{query}'...")
                            results = coordinator.search_memories(query)

                            if "error" in results:
                                print(f"[错误] 搜索失败: {results['error']}")
                            else:
                                print("\n搜索结果:")

                                if results.get('stm_results'):
                                    print(f"\n短期记忆 ({len(results['stm_results'])} 条):")
                                    for i, item in enumerate(results['stm_results'], 1):
                                        print(f"  {i}. {item['content']}")
                                        print(f"     重要性: {item['importance']:.2f}, 时间: {item['timestamp']}")

                                if results.get('ltm_results'):
                                    print(f"\n长期记忆 ({len(results['ltm_results'])} 条):")
                                    for i, item in enumerate(results['ltm_results'], 1):
                                        print(f"  {i}. [{item['category']}] {item['content']}")
                                        print(f"     重要性: {item['importance']:.2f}, 访问: {item['access_count']}次")

                                if not results.get('stm_results') and not results.get('ltm_results'):
                                    print("未找到相关记忆")
                    except Exception as e:
                        ErrorHandler.handle_exception(e, "记忆搜索")
                        print(f"[错误] 搜索记忆异常: {e}")

                elif choice == "3":
                    # 查看系统状态
                    try:
                        status = coordinator.get_status()
                        print("\n" + "=" * 70)
                        print("系统状态详情:")
                        print("=" * 70)

                        print(f"运行状态:")
                        print(f"  • 正在处理: {'是' if status['system']['is_processing'] else '否'}")
                        if status['system']['current_task_id']:
                            print(f"  • 当前任务: {status['system']['current_task_id']}")
                        print(f"  • 运行时间: {status['system']['uptime_formatted']}")
                        print(f"  • 活跃线程: {status['system']['active_threads']}")
                        print(f"  • 对话历史: {status['system']['conversation_history_count']} 条")

                        print(f"\n功能状态:")
                        print(f"  • TTS功能: {'启用' if status['system']['tts_enabled'] else '禁用'}")
                        print(f"  • 记忆系统: {'启用' if status['system']['memory_enabled'] else '禁用'}")

                        print(f"\n组件信息:")
                        print(f"  • LLM提供商: {status['components']['llm']['provider']}")
                        print(f"  • LLM模型: {status['components']['llm']['model']}")
                        print(f"  • TTS提供商: {status['components']['tts']['provider'] or 'N/A'}")

                        # 音频状态
                        audio_status = status['components']['audio_queue']
                        print(f"  • 音频队列大小: {audio_status.get('queue_size', 0)}")
                        print(f"  • 音频总字节: {audio_status.get('total_bytes_received', 0)}")
                        print(f"  • 正在播放: {'是' if audio_status.get('is_playing', False) else '否'}")

                        print(f"\n统计数据:")
                        print(f"  • 总任务数: {status['statistics']['total_tasks']}")
                        print(f"  • LLM调用: {status['statistics']['llm_calls']}")
                        print(f"  • TTS调用: {status['statistics']['tts_calls']}")
                        print(f"  • 记忆更新: {status['statistics']['memory_updates']}")
                        print(f"  • 失败任务: {status['statistics']['failed_tasks']}")
                        if 'success_rate' in status['statistics']:
                            print(f"  • 成功率: {status['statistics']['success_rate']}")
                        print(f"  • 总字符数: {status['statistics']['total_chars']}")
                        print(f"  • 音频字节数: {status['statistics'].get('audio_bytes_received', 0)}")

                        # TTS客户端状态
                        if status['components']['tts']['client_status']:
                            tts_status = status['components']['tts']['client_status']
                            print(f"\nTTS客户端状态:")
                            print(f"  • 正在流式合成: {'是' if tts_status.get('is_streaming', False) else '否'}")
                            if 'config' in tts_status:
                                config = tts_status['config']
                                if config.get('vcn'):
                                    print(f"  • 语音模型: {config['vcn']}")
                                    print(f"  • 语速: {config['speed']}")
                                    print(f"  • 音量: {config['volume']}")
                                    print(f"  • 音高: {config['pitch']}")

                        print("=" * 70)
                    except Exception as e:
                        ErrorHandler.handle_exception(e, "获取系统状态")
                        print(f"[错误] 获取系统状态异常: {e}")

                elif choice == "4":
                    # 查看记忆统计
                    if not coordinator.memory_enabled:
                        print("[错误] 记忆系统未启用")
                        continue

                    try:
                        stats = coordinator.get_memory_stats()
                        if "error" in stats:
                            print(f"[错误] 获取统计失败: {stats['error']}")
                        else:
                            print("\n" + "=" * 70)
                            print("记忆系统统计:")
                            print("=" * 70)

                            if 'short_term_memory' in stats:
                                stm = stats['short_term_memory']
                                print(f"短期记忆:")
                                print(f"  • 当前条目: {stm.get('current_buffer_size', 0)}")
                                print(f"  • 总条目数: {stm.get('total_entries', 0)}")
                                print(f"  • 平均重要性: {stm.get('avg_importance', 0):.2f}")
                                print(f"  • 摘要数量: {stm.get('summary_count', 0)}")

                            if 'long_term_memory' in stats:
                                ltm = stats['long_term_memory']
                                print(f"\n长期记忆:")
                                print(f"  • 总记忆数: {ltm.get('total_memories', 0)}")
                                print(f"  • 分类数量: {ltm.get('categories_count', 0)}")
                                print(f"  • 平均重要性: {ltm.get('avg_importance', 0):.2f}")
                                print(f"  • 平均访问次数: {ltm.get('avg_access_count', 0):.1f}")
                                print(f"  • 最近记忆 (7天内): {ltm.get('recent_memories', 0)}")

                            if 'knowledge_graph' in stats:
                                kg = stats['knowledge_graph']
                                print(f"\n知识图谱:")
                                print(f"  • 实体数量: {kg.get('knowledge_graph_entities', 0)}")

                            if 'session_id' in stats:
                                print(f"\n会话信息:")
                                print(f"  • 会话ID: {stats['session_id']}")
                                print(f"  • 对话数量: {stats.get('conversation_count', 0)}")

                            print("=" * 70)
                    except Exception as e:
                        ErrorHandler.handle_exception(e, "获取记忆统计")
                        print(f"[错误] 获取记忆统计异常: {e}")

                elif choice == "5":
                    # 查看对话历史
                    try:
                        history = coordinator.get_conversation_history(10)
                        if not history:
                            print("暂无对话历史")
                        else:
                            print("\n" + "=" * 70)
                            print(f"最近 {len(history)} 条对话历史:")
                            print("=" * 70)

                            for i, entry in enumerate(history, 1):
                                role_icon = "用户" if entry["role"] == "user" else "助手"
                                time_str = entry.get("timestamp", "未知时间")
                                if "T" in time_str:
                                    time_str = time_str.split("T")[1].split(".")[0]

                                content_preview = entry["content"]
                                if len(content_preview) > 60:
                                    content_preview = content_preview[:57] + "..."

                                print(f"{i}. {role_icon} [{time_str}] {content_preview}")
                                if i < len(history):
                                    print("   ──")

                            print("=" * 70)
                    except Exception as e:
                        ErrorHandler.handle_exception(e, "获取对话历史")
                        print(f"[错误] 获取对话历史异常: {e}")

                elif choice == "6":
                    # 系统设置
                    try:
                        system_settings_menu(coordinator, audio_player)
                    except Exception as e:
                        ErrorHandler.handle_exception(e, "系统设置菜单")
                        print(f"[错误] 系统设置菜单异常: {e}")

                elif choice == "7":
                    # 测试音频
                    if audio_player:
                        try:
                            print("[测试] 正在测试音频播放...")
                            success = audio_player.test_playback(duration=1.0, frequency=440.0)
                            if success:
                                print("[成功] 测试音播放完成，您应该听到了一个音调")
                            else:
                                print("[错误] 测试音播放失败")
                        except Exception as e:
                            ErrorHandler.handle_exception(e, "音频测试")
                            print(f"[错误] 音频测试异常: {e}")
                    else:
                        print("[错误] 音频播放器未初始化")

                elif choice == "8":
                    # 退出程序
                    print("\n" + "=" * 70)
                    confirm = input("确定要退出Airis系统吗？(y/N): ").strip().lower()
                    if confirm == 'y':
                        print("\n正在关闭系统，请稍候...")
                        coordinator.shutdown()
                        if audio_player:
                            audio_player.stop()
                        print("感谢使用Airis，再见！")
                        break
                    else:
                        print("取消退出")

                else:
                    print("[错误] 无效选择，请重新输入")

                # 每次操作后暂停一下，让用户看清结果
                if choice not in ["1", "6"]:
                    input("\n按Enter键继续...")

            except Exception as e:
                ErrorHandler.handle_exception(e, "主循环")
                print(f"\n[错误] 主循环发生异常: {e}")
                print("系统将继续运行，请稍后重试...")
                time.sleep(1)

    except Exception as e:
        ErrorHandler.handle_exception(e, "主程序", "critical")
        print(f"\n[致命错误] 主程序发生未预期错误: {e}")
        traceback.print_exc()

    finally:
        # 确保资源被清理
        try:
            coordinator.shutdown()
        except:
            pass

        if audio_player:
            audio_player.stop()

        print("\n" + "=" * 70)
        print("Airis 系统已关闭")
        print("=" * 70)


# ===============================
# 启动程序
# ===============================
if __name__ == "__main__":
    # 修复Windows控制台编码问题
    if os.name == 'nt':
        try:
            import sys
            import io

            # 设置标准输出编码为UTF-8
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

            os.environ['PYTHONIOENCODING'] = 'utf-8'

            try:
                os.system('chcp 65001 > nul')
            except:
                pass
        except:
            pass

    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        error_info = ErrorHandler.handle_exception(e, "程序启动", "critical")
        print(f"\n[错误] 程序启动失败: {e}")
        traceback.print_exc()
        input("\n按Enter键退出...")