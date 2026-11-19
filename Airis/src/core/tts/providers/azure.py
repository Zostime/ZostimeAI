import textwrap
import azure.cognitiveservices.speech as speechsdk
import re

from .base import (
    ProviderCallbacks,
    TTSProvider,
    WordBoundaryEvent
)


class AzureTTSProvider(TTSProvider):
    """
    config.tts:
        provider: "azure"
        voice:    ""       # https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=tts
        lang:     ""       # voice 与 lang 同时设置时 lang 被忽略
        pitch:    "+0Hz"   # 格式 [+-]\\d+(Hz|%) 或命名级别 x-low/low/medium/high/x-high
        rate:     "+0%"    # 格式 [+-]\\d+% 或命名级别 x-slow/slow/medium/fast/x-fast
        volume:   "+0%"    # 格式 [+-]\\d+% 或 0.0–100.0 绝对值

    runtime 依赖:
        - azure-cognitiveservices-speech Python 包
        - 环境变量 $TTS_SPEECH_KEY / $TTS_SERVICE_REGION
    """

    OUTPUT_SAMPLE_RATE = 24000

    def __init__(self, callbacks: ProviderCallbacks, config) -> None:
        super().__init__(callbacks, config)

        self.voice = config.get_json("tts.voice")
        self.lang = config.get_json("tts.lang")
        self.pitch = config.get_json("tts.pitch")
        self.rate = config.get_json("tts.rate")
        self.volume = config.get_json("tts.volume")
        self.speech_key = config.get_env("TTS_SPEECH_KEY")
        self.service_region = config.get_env("TTS_SERVICE_REGION")

        self.synthesizer = None
        self._init()

    def _init(self) -> None:
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, region=self.service_region
        )
        speech_config.speech_synthesis_voice_name = self.voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
            value="true",
        )
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        self.synthesizer.synthesis_word_boundary.connect(self._on_word_boundary)
        self.synthesizer.synthesizing.connect(self._on_synthesizing)
        self.synthesizer.synthesis_completed.connect(self._on_synthesis_completed)
        self.synthesizer.synthesis_canceled.connect(self._on_synthesis_canceled)

    @property
    def sample_rate(self) -> int:
        return self.OUTPUT_SAMPLE_RATE

    # Azure Callbacks
    def _on_word_boundary(self, evt) -> None:
        self.callbacks.on_word_boundary(
            WordBoundaryEvent(
                text=evt.text,
                audio_offset_seconds=evt.audio_offset / 10_000_000,
            )
        )

    def _on_synthesizing(self, evt) -> None:
        try:
            data = evt.result.audio_data
        except Exception as error:  # noqa
            self.callbacks.on_error(f"Error in synthesizing callback: {error}")
            return
        if data:
            self.callbacks.on_audio(data)

    def _on_synthesis_completed(self, evt) -> None:  # noqa
        self.callbacks.on_completed()

    def _on_synthesis_canceled(self, evt) -> None:
        details = evt.result.cancellation_details
        is_error = details.reason == speechsdk.CancellationReason.Error
        self.callbacks.on_canceled(
            error_details=details.error_details if is_error else "",
            is_error=is_error,
        )

    def sanitize_ssml_fragment(raw: str) -> str:
        """清理输入中的 SSML 包装, 代码块标记和多余空白"""
        raw = re.sub(r'</?speak[^>]*>', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'</?voice[^>]*>', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'```[a-zA-Z]*', '', raw)
        raw = re.sub(r'[\s\u200B-\u200D\uFEFF]+', ' ', raw, flags=re.UNICODE)
        raw = re.sub(r' {2,}', ' ', raw).strip()
        return raw

    # TTSProvider
    def synthesize(self, ssml_fragment: str) -> None:
        body = self.sanitize_ssml_fragment(ssml_fragment)
        ssml_string = textwrap.dedent(f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{self.lang}">
            <voice name="{self.voice}">
                <prosody pitch="{self.pitch}" rate="{self.rate}" volume="{self.volume}">
                    {body}
                </prosody>
            </voice>
        </speak>
        """)
        self.synthesizer.speak_ssml_async(ssml_string)

    def stop(self) -> None:
        try:
            self.synthesizer.stop_speaking_async().get()
        except Exception as error:  # noqa
            self.callbacks.on_error(f"Error stopping synthesis: {error}")

    def close(self) -> None:
        self.synthesizer = None
