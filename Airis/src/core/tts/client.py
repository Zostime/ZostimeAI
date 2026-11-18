import html
import queue
import re
import threading
import time
from typing import Callable, Optional
import numpy as np
import sounddevice as sd

from ..common.config import ConfigManager
from ..common.logger import LogManager
from .providers.base import ProviderCallbacks, TTSProvider, WordBoundaryEvent
from .providers.factory import create_provider


class TTSClient(ProviderCallbacks):
    SENTENCE_END = object()

    def __init__(self, state_callback: Optional[Callable[[str], None]] = None):
        self.state_callback = state_callback
        self.config = ConfigManager()
        self.logger = LogManager("tts").get_logger()

        self.running = True
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.finish_marker = object()
        self.playback_end_marker = object()
        self.sentence_playback_end_marker = object()
        self._has_pending_text = False
        self.boundary_queue = queue.Queue()
        self.subtitle_queue = queue.Queue()
        self.base_frame_queue = queue.Queue()

        self.played_frames = 0
        self.played_frames_lock = threading.Lock()
        self._playback_state_lock = threading.Lock()
        self._player_lock = threading.Lock()
        self._is_playing = False
        self.sentence_start_marker = object()

        self._sentence_end_sent = False
        self._sentence_end_lock = threading.Lock()

        self.p = None
        self.player: Optional[sd.OutputStream] = None
        self.provider: Optional[TTSProvider] = None
        self.sample_rate = 24000

        self.synthesis_idle = threading.Event()
        self.synthesis_idle.set()
        self.stop_requested = False
        self.interrupt_audio = threading.Event()

        self._text_cache = ""
        self._ssml_pending_text = ""
        self.sentence_boundary_idle = threading.Event()
        self.sentence_boundary_idle.set()
        self.word_boundary_idle = threading.Event()
        self.word_boundary_idle.set()
        self.player_idle = threading.Event()
        self.player_idle.set()

        self._init()

    def _init(self):
        self.provider = create_provider(self, self.config)
        self.sample_rate = self.provider.sample_rate

        self.player = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16'
        )
        self.player.start()

        self.player_thread = threading.Thread(target=self._player_worker, daemon=True)
        self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self.subtitle_thread = threading.Thread(target=self._subtitle_worker, daemon=True)
        self.player_thread.start()
        self.synth_thread.start()
        self.subtitle_thread.start()

    # Provider Callbacks

    def on_audio(self, data: bytes) -> None:
        self.audio_queue.put(data)

    def on_word_boundary(self, event: WordBoundaryEvent) -> None:
        self.word_boundary_idle.clear()
        self.boundary_queue.put_nowait(event)

    def on_completed(self) -> None:
        self._send_sentence_end()
        self.audio_queue.put(self.sentence_playback_end_marker)
        self.synthesis_idle.set()

    def on_canceled(self, error_details: str = "", is_error: bool = False) -> None:
        self._send_sentence_end()
        self.audio_queue.put(self.sentence_playback_end_marker)
        self.synthesis_idle.set()
        if is_error:
            self.logger.error(f"Error details: {error_details}")

    def on_error(self, message: str) -> None:
        self.logger.error(message)

    def _subtitle_worker(self):
        while self.running:  # noqa
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
                    self.word_boundary_idle.set()
                    break

                text = evt.text
                if not text:
                    continue

                offset_frames = round(evt.audio_offset_seconds * self.sample_rate)
                target_frame = base_frame + offset_frames

                while self.running and not self.interrupt_audio.is_set():
                    with self.played_frames_lock:
                        current_frame = self.played_frames
                    if current_frame >= target_frame:
                        index = self._text_cache.find(text)
                        if index != -1:
                            subtitle_chunk = self._text_cache[:index + len(text)]
                            self._text_cache = self._text_cache[index + len(text):]
                            self.subtitle_queue.put(subtitle_chunk)

                            filtered_text_cache = re.sub(
                                pattern=r'\s',
                                repl='',
                                string=self._text_cache
                            )
                            if filtered_text_cache[:1] == '\0':
                                filtered_index = self._text_cache.find('\0')
                                if filtered_index != -1:
                                    self._text_cache = self._text_cache[filtered_index + 1:]
                        self.word_boundary_idle.set()
                        break

    def _send_sentence_end(self):
        with self._sentence_end_lock:
            if not self._sentence_end_sent:
                self._sentence_end_sent = True
                self.boundary_queue.put(self.SENTENCE_END)

    def _emit_state(self, state: str) -> None:
        if self.state_callback is not None:
            try:
                self.state_callback(state)
            except Exception as error:
                self.logger.error(f"TTS state callback error: {error}")

    def _start_playback(self) -> None:
        with self._playback_state_lock:
            if self._is_playing:
                return
            self._is_playing = True
        self._emit_state("start")

    def _finish_playback(self, state: str) -> None:
        with self._playback_state_lock:
            was_playing = self._is_playing
            self._is_playing = False
        self.player_idle.set()
        if was_playing:
            self._emit_state(state)

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

            if data is self.playback_end_marker:
                self._finish_playback("end")
                continue

            if data is self.sentence_playback_end_marker:
                self.sentence_boundary_idle.set()
                self.word_boundary_idle.set()
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
                self.sentence_boundary_idle.clear()
                self.player_idle.clear()
                with self._player_lock:
                    if self.interrupt_audio.is_set():
                        continue
                    self._start_playback()
                    self.player.write(np.frombuffer(data, dtype=np.int16))
                frames = len(data) // 2
                with self.played_frames_lock:
                    self.played_frames += frames
            except Exception as e:
                if not self.running:
                    break
                self.logger.error(f"Audio write error: {e}")
                self.sentence_boundary_idle.set()
                self._finish_playback("end")
                if self.interrupt_audio.is_set():
                    continue
                break

    @staticmethod
    def _find_ssml_tag_end(text: str, start: int):
        quote = None
        for index in range(start + 1, len(text)):
            char = text[index]
            if quote:
                if char == quote:
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == ">":
                return index
        return None

    @classmethod
    def _scan_ssml_fragment(cls, text: str):
        stack = []
        safe_end = 0
        incomplete_at = None
        malformed = False
        index = 0

        while index < len(text):
            if text[index] != "<":
                index += 1
                if not stack:
                    safe_end = index
                continue

            if text.startswith("<!--", index):
                comment_end = text.find("-->", index + 4)
                if comment_end == -1:
                    incomplete_at = index
                    break
                index = comment_end + 3
                if not stack:
                    safe_end = index
                continue

            tag = re.match(r"</?([A-Za-z_][\w:.-]*)", text[index:])
            if not tag:
                if text[index:] in ("<", "</"):
                    incomplete_at = index
                    break
                index += 1
                if not stack:
                    safe_end = index
                continue

            tag_end = cls._find_ssml_tag_end(text, index)
            if tag_end is None:
                incomplete_at = index
                break

            tag_name = tag.group(1)
            is_closing = text[index + 1] == "/"
            is_self_closing = text[index:tag_end].rstrip().endswith("/")
            if is_closing:
                if not stack or stack[-1] != tag_name:
                    malformed = True
                    break
                stack.pop()
            elif not is_self_closing:
                stack.append(tag_name)

            index = tag_end + 1
            if not stack:
                safe_end = index

        return safe_end, stack, incomplete_at, malformed

    def _queue_pending_ssml(self):
        if not self._ssml_pending_text:
            return

        _, stack, incomplete_at, malformed = self._scan_ssml_fragment(
            self._ssml_pending_text
        )
        pending = self._ssml_pending_text
        self._ssml_pending_text = ""

        if malformed:
            self.logger.error(
                "Malformed SSML fragment; treating it as plain text"
            )
            pending = html.escape(pending, quote=False)
        elif incomplete_at is not None:
            self.logger.warning(
                "Incomplete SSML tag at end of stream; escaping the unfinished tag"
            )
            pending = (
                pending[:incomplete_at]
                + "".join(f"</{tag_name}>" for tag_name in reversed(stack))
                + html.escape(pending[incomplete_at:], quote=False)
            )
        elif stack:
            self.logger.warning(
                "Unclosed SSML elements at end of stream; closing them automatically"
            )
            pending += "".join(
                f"</{tag_name}>" for tag_name in reversed(stack)
            )

        self.text_queue.put(pending)

    def _synth_worker(self):
        while self.running:  # noqa
            try:
                text = self.text_queue.get()
            except queue.Empty:
                continue
            if text is None:
                break
            if text is self.finish_marker:
                while self.running and not self.stop_requested:
                    if self.synthesis_idle.wait(timeout=0.1):
                        break
                if self.stop_requested or not self.running:
                    continue
                self.audio_queue.put(self.playback_end_marker)
                continue
            if not text.strip():
                continue
            while self.running and not self.stop_requested:
                if self.synthesis_idle.wait(timeout=0.1):
                    break
            if self.stop_requested or not self.running:
                continue
            self.synthesis_idle.clear()

            self.audio_queue.put(self.sentence_start_marker)

            with self._sentence_end_lock:
                self._sentence_end_sent = False

            try:
                self.provider.synthesize(text)
            except Exception as e:
                self.logger.error(f"Synthesis start error: {e}")
                self._send_sentence_end()
                self.audio_queue.put(self.sentence_playback_end_marker)
                self.synthesis_idle.set()

    def feed(self, text: str) -> None:
        if not self.running:
            return
        self._text_cache += text
        self._has_pending_text = self._has_pending_text or bool(text.strip())
        self._ssml_pending_text += text
        safe_end, _, _, _ = self._scan_ssml_fragment(self._ssml_pending_text)
        if safe_end:
            self.text_queue.put(self._ssml_pending_text[:safe_end])
            self._ssml_pending_text = self._ssml_pending_text[safe_end:]

    def finished(self):
        if not self.running:
            return
        if not self._has_pending_text and not self._ssml_pending_text:
            return
        self._has_pending_text = False
        self._queue_pending_ssml()
        self._text_cache += '\0'
        self.player_idle.clear()
        self.text_queue.put(self.finish_marker)

    def interrupt(self):
        was_active = (
            self._has_pending_text
            or not self.synthesis_idle.is_set()
            or not self.player_idle.is_set()
        )
        self.stop_requested = True
        self._text_cache = ""
        self._ssml_pending_text = ""
        self._has_pending_text = False
        with self._playback_state_lock:
            self._is_playing = False
        self.player_idle.set()
        self.sentence_boundary_idle.set()
        self.word_boundary_idle.set()
        self.interrupt_audio.set()
        if was_active:
            self._emit_state("interrupt")

        if self.provider is not None:
            self.provider.stop()

        player_ready = True
        if self.player:
            with self._player_lock:
                try:
                    if self.player.active:
                        self.player.abort()
                except Exception as e:
                    self.logger.error(f"Error aborting audio stream: {e}")

                for attempt in range(5):
                    try:
                        if not self.player.active:
                            self.player.start()
                        break
                    except Exception as e:
                        if attempt == 4:
                            self.logger.error(
                                f"Error reopening audio stream after retries: {e}"
                            )
                            player_ready = False
                        else:
                            time.sleep(0.1 * (attempt + 1))

        self._drain_queue(self.audio_queue)
        self._drain_queue(self.text_queue)
        self._drain_queue(self.boundary_queue)
        self._drain_queue(self.subtitle_queue)
        self._drain_queue(self.base_frame_queue)

        with self._sentence_end_lock:
            self._sentence_end_sent = False

        self.synthesis_idle.set()
        self.stop_requested = False
        if player_ready:
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
        self._text_cache = ""
        self.sentence_boundary_idle.set()
        self.word_boundary_idle.set()
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

        if self.provider is not None:
            self.provider.close()

        if self.player:
            self.player.stop()
            self.player.close()
