import queue
import threading
import azure.cognitiveservices.speech as speechsdk # noqa
import pyaudio

from core.common.config import ConfigManager
from core.common.logger import LogManager

class TTSClient:
    def __init__(self):
        self.config = ConfigManager()
        self.logger = LogManager("tts").get_logger()

        self.voice = self.config.get_json("tts.voice")
        self.lang = self.config.get_json("tts.lang")
        self.pitch = self.config.get_json("tts.pitch")
        self.rate = self.config.get_json("tts.rate")
        self.volume = self.config.get_json("tts.volume")

        self.speech_key = self.config.get_env("TTS_SPEECH_KEY")
        self.service_region = self.config.get_env("TTS_SERVICE_REGION")

        self.running = True
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.p = None
        self.player = None
        self.synthesizer = None

        self.synthesis_idle = threading.Event()
        self.synthesis_idle.set()
        self.stop_requested = False

        self._init()
        self._start_threads()

    def _init(self):
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, region=self.service_region
        )
        speech_config.speech_synthesis_voice_name = self.voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )

        self.synthesizer.synthesizing.connect(self._on_synthesizing)
        self.synthesizer.synthesis_completed.connect(self._on_synthesis_completed)
        self.synthesizer.synthesis_canceled.connect(self._on_synthesis_canceled)

        self.p = pyaudio.PyAudio()
        self.player = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=4096
        )

    def _start_threads(self):
        self.player_thread = threading.Thread(target=self._player_worker, daemon=True)
        self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self.player_thread.start()
        self.synth_thread.start()

    def _on_synthesizing(self, evt):
        try:
            data = evt.result.audio_data
            if data:
                self.audio_queue.put(data)
        except Exception as e:
            self.logger.error(f"Error in synthesizing callback: {e}")

    def _on_synthesis_completed(self, evt): # noqa
        self.synthesis_idle.set()

    def _on_synthesis_canceled(self, evt):
        self.synthesis_idle.set()
        cancellation_details = evt.result.cancellation_details
        self.logger.warning(
            f"Synthesis canceled: {cancellation_details.reason}"
        )
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            self.logger.error(f"Error details: {cancellation_details.error_details}")

    def _player_worker(self):
        while self.running:
            data = self.audio_queue.get()
            if data is None:
                break
            try:
                self.player.write(data)
            except Exception as e:
                self.logger.error(f"Audio write error: {e}")
                break

    def _synth_worker(self):
        while self.running:
            try:
                text = self.text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            if not text.strip():
                continue

            self.synthesis_idle.wait()
            if self.stop_requested:
                break
            self.synthesis_idle.clear()

            ssml_string = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.lang}">
                <voice name="{self.voice}">
                    <prosody pitch="{self.pitch}" rate="{self.rate}" volume="{self.volume}">{text}</prosody>
                </voice>
            </speak>
            """
            try:
                self.synthesizer.speak_ssml_async(ssml_string)
            except Exception as e:
                self.logger.error(f"Synthesis start error: {e}")
                self.synthesis_idle.set()

    def stream_feed(self, text: str) -> None:
        if not self.running:
            return
        self.text_queue.put(text)

    def interrupt(self):
        self.stop_requested = True
        try:
            self.synthesizer.stop_speaking_async().get()
        except Exception as e:
            self.logger.error(f"Error stopping synthesis: {e}")

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break

        self.synthesis_idle.set()
        self.stop_requested = False

    def close(self):
        self.running = False
        self.text_queue.put(None)
        self.audio_queue.put(None)
        if hasattr(self, 'synth_thread') and self.synth_thread.is_alive():
            self.synth_thread.join(timeout=2.0)
        if hasattr(self, 'player_thread') and self.player_thread.is_alive():
            self.player_thread.join(timeout=2.0)
        if self.player:
            self.player.stop_stream()
            self.player.close()
        if self.p:
            self.p.terminate()