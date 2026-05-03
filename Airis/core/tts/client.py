import queue
import threading
import azure.cognitiveservices.speech as speechsdk # noqa
import pyaudio

from core.common.config import ConfigManager
from core.common.logger import LogManager


class TTSClient:
    SENTENCE_END = object()

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
        self.boundary_queue = queue.Queue()
        self.subtitle_queue = queue.Queue()
        self.base_frame_queue = queue.Queue()

        self.played_frames = 0
        self.played_frames_lock = threading.Lock()
        self.sentence_start_marker = object()

        self._sentence_end_sent = False
        self._sentence_end_lock = threading.Lock()

        self.p = None
        self.player = None
        self.synthesizer = None

        self.synthesis_idle = threading.Event()
        self.synthesis_idle.set()
        self.stop_requested = False
        self.interrupt_audio = threading.Event()

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
        speech_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
            value="true"
        )
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        self.synthesizer.synthesis_word_boundary.connect(self._on_word_boundary)
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

    def _on_word_boundary(self, evt):
        try:
            self.boundary_queue.put_nowait(evt)
        except queue.Full:
            self.logger.warning("Word boundary queue is full, event dropped.")
        except Exception as e:
            self.logger.error(f"Error putting word boundary event: {e}")

    def _start_threads(self):
        self.player_thread = threading.Thread(target=self._player_worker, daemon=True)
        self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self.subtitle_thread = threading.Thread(target=self._subtitle_worker, daemon=True)
        self.player_thread.start()
        self.synth_thread.start()
        self.subtitle_thread.start()

    def _subtitle_worker(self):
        while self.running:
            try:
                base_frame = self.base_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if base_frame is None:
                break

            while self.running and not self.interrupt_audio.is_set():
                try:
                    evt = self.boundary_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if evt is None:
                    return
                if evt is self.SENTENCE_END:
                    break

                text = evt.text
                if not text:
                    continue

                offset_frames = round(evt.audio_offset * 24000 / 10_000_000)
                target_frame = base_frame + offset_frames

                while self.running and not self.interrupt_audio.is_set():
                    with self.played_frames_lock:
                        current_frame = self.played_frames
                    if current_frame >= target_frame:
                        self.subtitle_queue.put(text)
                        break

    def _on_synthesizing(self, evt):
        try:
            data = evt.result.audio_data
            if data:
                self.audio_queue.put(data)
        except Exception as e:
            self.logger.error(f"Error in synthesizing callback: {e}")

    def _maybe_send_sentence_end(self):
        with self._sentence_end_lock:
            if not self._sentence_end_sent:
                self._sentence_end_sent = True
                self.boundary_queue.put(self.SENTENCE_END)

    def _on_synthesis_completed(self, evt): # noqa
        self._maybe_send_sentence_end()
        self.synthesis_idle.set()

    def _on_synthesis_canceled(self, evt):
        self._maybe_send_sentence_end()
        self.synthesis_idle.set()
        cancellation_details = evt.result.cancellation_details
        self.logger.warning(f"Synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            self.logger.error(f"Error details: {cancellation_details.error_details}")

    def _player_worker(self):
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                if self.interrupt_audio.is_set():
                    continue
                continue

            if data is None:
                break

            if self.interrupt_audio.is_set():
                continue

            if data is self.sentence_start_marker:
                with self.played_frames_lock:
                    base_frame = self.played_frames
                try:
                    self.base_frame_queue.put_nowait(base_frame)
                except queue.Full:
                    pass
                continue

            try:
                self.player.write(data)
                frames = len(data) // 2
                with self.played_frames_lock:
                    self.played_frames += frames
            except Exception as e:
                if not self.running:
                    break
                self.logger.error(f"Audio write error: {e}")
                if self.interrupt_audio.is_set():
                    continue
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

            while self.running and not self.stop_requested:
                if self.synthesis_idle.wait(timeout=0.1):
                    break
            if self.stop_requested or not self.running:
                break
            self.synthesis_idle.clear()

            self.audio_queue.put(self.sentence_start_marker)

            with self._sentence_end_lock:
                self._sentence_end_sent = False

            ssml_string = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.lang}">
                <voice name="{self.voice}">
                    <prosody pitch="{self.pitch}" rate="{self.rate}" volume="{self.volume}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            try:
                self.synthesizer.speak_ssml_async(ssml_string)
            except Exception as e:
                self.logger.error(f"Synthesis start error: {e}")
                self._maybe_send_sentence_end()
                self.synthesis_idle.set()

    def stream_feed(self, text: str) -> None:
        if not self.running:
            return
        self.text_queue.put(text)

    def interrupt(self):
        self.stop_requested = True
        self.interrupt_audio.set()
        try:
            self.synthesizer.stop_speaking_async().get()
        except Exception as e:
            self.logger.error(f"Error stopping synthesis: {e}")

        if self.player and self.player.is_active():
            try:
                self.player.stop_stream()
            except Exception as e:
                self.logger.error(f"Error stopping audio stream: {e}")

        self._drain_queue(self.audio_queue)
        self._drain_queue(self.text_queue)
        self._drain_queue(self.boundary_queue)
        self._drain_queue(self.subtitle_queue)
        self._drain_queue(self.base_frame_queue)

        with self._sentence_end_lock:
            self._sentence_end_sent = False

        if self.player:
            try:
                self.player.close()
            except Exception as e:
                self.logger.error(f"Error closing audio stream: {e}")
            try:
                self.player = self.p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True,
                    frames_per_buffer=4096
                )
            except Exception as e:
                self.logger.error(f"Error reopening audio stream: {e}")

        self.synthesis_idle.set()
        self.stop_requested = False
        self.interrupt_audio.clear()

    @staticmethod
    def _drain_queue(q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def close(self):
        self.running = False
        self.text_queue.put(None)
        self.audio_queue.put(None)
        self.boundary_queue.put(None)
        self.base_frame_queue.put(None)

        if hasattr(self, 'synth_thread') and self.synth_thread.is_alive():
            self.synth_thread.join(timeout=2.0)
        if hasattr(self, 'player_thread') and self.player_thread.is_alive():
            self.player_thread.join(timeout=2.0)
        if hasattr(self, 'subtitle_thread') and self.subtitle_thread.is_alive():
            self.subtitle_thread.join(timeout=2.0)

        if self.player:
            self.player.stop_stream()
            self.player.close()
        if self.p:
            self.p.terminate()