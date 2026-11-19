from abc import ABC, abstractmethod


class WordBoundaryEvent:
    __slots__ = ("text", "audio_offset_seconds")

    def __init__(self, text: str, audio_offset_seconds: float) -> None:
        self.text = text
        self.audio_offset_seconds = audio_offset_seconds


class ProviderCallbacks(ABC):
    """Provider 向上回调接口"""

    @abstractmethod
    def on_audio(self, data: bytes) -> None:
        """收到 PCM 音频 (int16 / 单声道 / sample_rate)"""

    @abstractmethod
    def on_word_boundary(self, event: WordBoundaryEvent) -> None:
        """收到单词边界事件"""

    @abstractmethod
    def on_completed(self) -> None:
        """合成正常完成"""

    @abstractmethod
    def on_canceled(self, error_details: str = "", is_error: bool = False) -> None:
        """合成被取消"""

    @abstractmethod
    def on_error(self, message: str) -> None:
        """Provider 内部错误日志"""


class TTSProvider(ABC):
    """TTS 服务商端通信的抽象层"""

    def __init__(self, callbacks: ProviderCallbacks, config) -> None:
        self.callbacks = callbacks
        self.config = config

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """输出 PCM 的采样率 (Hz)"""

    @abstractmethod
    def synthesize(self, ssml_fragment: str) -> None:
        """异步合成一个 SSML 片段;音频/边界通过 callbacks 流出"""

    @abstractmethod
    def stop(self) -> None:
        """停止当前合成"""

    @abstractmethod
    def close(self) -> None:
        """释放底层资源"""
