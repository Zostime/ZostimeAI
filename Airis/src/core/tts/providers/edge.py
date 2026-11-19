import asyncio
import subprocess
import threading
import re

import edge_tts

from .base import (
    ProviderCallbacks,
    TTSProvider,
    WordBoundaryEvent,
)


class EdgeTTSProvider(TTSProvider):
    """
    config.tts:
        provider: "edge"  
        voice:    ""      # https://github.com/rany2/edge-tts
        pitch:    "+0Hz"  # 格式 [+-]\\d+Hz
        rate:     "+0%"   # 格式 [+-]\\d+%  
        volume:   "+0%"   # 格式 [+-]\\d+%  

    runtime 依赖:
        - edge-tts Python 包
        - ffmpeg 可执行文件 ($PATH 中可见)
    """

    OUTPUT_SAMPLE_RATE = 24000

    def __init__(self, callbacks: ProviderCallbacks, config) -> None:
        super().__init__(callbacks, config)

        self.voice = config.get_json("tts.voice")
        self.rate = config.get_json("tts.rate") 
        self.volume = config.get_json("tts.volume") 
        self.pitch = config.get_json("tts.pitch") 

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._thread.start()

        self._task = None
        self._task_lock = threading.Lock()

        self._ffmpeg = None
        self._ffmpeg_lock = threading.Lock()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def sample_rate(self) -> int:
        return self.OUTPUT_SAMPLE_RATE

    def _ensure_ffmpeg(self) -> subprocess.Popen:
        with self._ffmpeg_lock:
            if self._ffmpeg is None or self._ffmpeg.poll() is not None:
                try:
                    self._ffmpeg = subprocess.Popen(
                        [
                            "ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-i", "pipe:0",
                            "-f", "s16le", "-acodec", "pcm_s16le",
                            "-ac", "1", "-ar", str(self.OUTPUT_SAMPLE_RATE),
                            "pipe:1",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        bufsize=0,
                    )
                except FileNotFoundError:
                    raise RuntimeError(
                        "ffmpeg not found in PATH; EdgeTTSProvider requires ffmpeg"
                    ) from None
                threading.Thread(
                    target=self._pcm_reader, args=(self._ffmpeg,), daemon=True
                ).start()
            return self._ffmpeg

    def _pcm_reader(self, ffmpeg: subprocess.Popen) -> None:
        while True:
            try:
                data = ffmpeg.stdout.read(4096)
            except Exception as e:
                self.callbacks.on_error(f"ffmpeg stdout read error: {e}")
                break
            if not data:
                break
            self.callbacks.on_audio(data)

    def strip_ssml_tags(self, text: str) -> str:
        return re.sub(r'</?[A-Za-z][^>]*>', '', text)

    def synthesize(self, ssml_fragment: str) -> None:
        text = self.strip_ssml_tags(ssml_fragment)
        if not text.strip():
            self.callbacks.on_completed()
            return
        with self._task_lock:
            self._task = asyncio.run_coroutine_threadsafe(
                self._stream(text), self._loop
            )
            self._task.add_done_callback(self._on_task_done)
            
    def _on_task_done(self, fut):
        try:
            fut.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.callbacks.on_error(f"EdgeTTS async task failed: {e!r}")

    async def _stream(self, text: str) -> None:
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )
        word_boundary_seen = False
        sentence_boundaries = []
        try:
            ffmpeg = self._ensure_ffmpeg()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    ffmpeg.stdin.write(chunk["data"])
                    ffmpeg.stdin.flush()
                elif chunk["type"] == "WordBoundary":
                    word_boundary_seen = True
                    self.callbacks.on_word_boundary(
                        WordBoundaryEvent(
                            text=chunk["text"],
                            audio_offset_seconds=chunk["offset"] / 1e7,
                        )
                    )
                elif chunk["type"] == "SentenceBoundary":
                    if not word_boundary_seen:
                        sentence_boundaries.append(chunk)
        except asyncio.CancelledError:
            self._reset_ffmpeg()
            self.callbacks.on_canceled()
            raise
        except Exception as e:
            self._reset_ffmpeg()
            self.callbacks.on_canceled(
                error_details=f"{type(e).__name__}: {e}",
                is_error=True,
            )
        else:
            if not word_boundary_seen:
                for chunk in sentence_boundaries:
                    self.callbacks.on_word_boundary(
                        WordBoundaryEvent(
                            text=chunk["text"],
                            audio_offset_seconds=chunk["offset"] / 1e7,
                        )
                    )
            self._close_ffmpeg()
            self.callbacks.on_completed()

    def _close_ffmpeg(self) -> None:
        with self._ffmpeg_lock:
            ffmpeg, self._ffmpeg = self._ffmpeg, None
        if ffmpeg is None:
            return
        try:
            ffmpeg.stdin.close()
        except Exception:
            pass
        try:
            ffmpeg.wait(timeout=5)
        except Exception:
            ffmpeg.kill()

    def _reset_ffmpeg(self) -> None:
        with self._ffmpeg_lock:
            ffmpeg, self._ffmpeg = self._ffmpeg, None
        if ffmpeg is None:
            return
        try:
            ffmpeg.kill()
        except Exception:
            pass

    def stop(self) -> None:
        with self._task_lock:
            if self._task is not None and not self._task.done():
                self._task.cancel()
        self._reset_ffmpeg()

    def close(self) -> None:
        self.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
