import subprocess
import threading
import asyncio
import queue
from typing import Optional
import edge_tts

from core.common.config import ConfigManager
from core.common.logger import LogManager

class TTSClient:
    def __init__(self):
        self.config = ConfigManager()
        self.logger = LogManager("tts").get_logger()

        self.voice = self.config.get_json("tts.voice")
        self.pitch = self.config.get_json("tts.pitch")
        self.rate = self.config.get_json("tts.rate")
        self.volume = self.config.get_json("tts.volume")

        self._text_queue: queue.Queue = queue.Queue()
        self._boundary_queue: queue.Queue = queue.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

        self._stop_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._start_worker()

    def _start_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_worker_loop,
            daemon=True
        )
        self._worker_thread.start()

    def _stop_worker(self):
        if not self._worker_thread:
            return
        self._shutdown_event.set()
        self._text_queue.put(None)
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._worker_thread.join(timeout=3)
        self._worker_thread = None

    def get_boundary_queue(self) -> queue.Queue:
        return self._boundary_queue

    def stream_feed(self, text: str):
        if not text:
            return
        self._stop_event.clear()
        if not self._worker_thread or not self._worker_thread.is_alive():
            self._start_worker()
        self._text_queue.put(text)

    def interrupt(self):
        self._stop_event.set()
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break
        self._kill_player()

    def shutdown(self):
        self._stop_worker()
        self._kill_player()

    def _run_worker_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_process_queue())
        except Exception as e:
            self.logger.error(f"Worker loop error: {e}")
        finally:
            self._loop.close()
            self._loop = None
            self._kill_player()

    async def _async_process_queue(self):
        while not self._shutdown_event.is_set():
            try:
                text = await self._loop.run_in_executor(
                    None, self._text_queue.get
                )
            except Exception: # noqa
                break

            if text is None:
                break

            await self._play_text(text)

        self._kill_player()

    async def _play_text(self, text: str):
        if not self._ensure_player():
            self.logger.error("Failed to start ffplay player")
            return

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
                volume=self.volume,
            )
            async for chunk in communicate.stream():
                if self._stop_event.is_set():
                    break
                if chunk["type"] == "WordBoundary" or chunk["type"] == "SentenceBoundary":
                    try:
                        self._boundary_queue.put_nowait({
                            "type": chunk["type"],
                            "text": chunk["text"],
                            "offset": chunk["offset"],
                            "duration": chunk["duration"]
                        })
                    except queue.Full:
                        pass
                if chunk["type"] == "audio":
                    data = chunk["data"]
                    if self._process and self._process.stdin:
                        try:
                            self._process.stdin.write(data)
                            self._process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            self._kill_player()
                            return
            if self._process and self._process.stdin:
                try:
                    self._process.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._kill_player()
        except Exception as e:
            self.logger.error(f"TTS playback error: {e}")
        finally:
            if self._stop_event.is_set():
                self._kill_player()

    def _ensure_player(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        cmd = [
            "ffplay",
            "-f", "mp3",
            "-i", "pipe:0",
            "-nodisp",
            "-autoexit",
            "-loglevel", "quiet",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as e:
            self.logger.error(f"Cannot start ffplay: {e}")
            return False

    def _kill_player(self):
        if self._process:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None
