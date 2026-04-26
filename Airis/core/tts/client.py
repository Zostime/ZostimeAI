import subprocess
import threading
import asyncio
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

        self._process: Optional[subprocess.Popen] = None
        self._wait_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def stream_tts(self, text: str):
        self.interrupt()
        self._stop_event.clear()

        self._wait_thread = threading.Thread(target=self._run_tts, args=(text,), daemon=True)
        self._wait_thread.start()
        self._wait_thread.join()

    def _run_tts(self, text: str):
        cmd = [
            "ffplay",
            "-f", "mp3",
            "-i", "pipe:0",
            "-nodisp",
            "-autoexit",
            "-loglevel", "quiet",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            asyncio.run(self._stream_and_play(text))
        except Exception as e:
            self.logger.error(f"error: {e}")
        finally:
            if self._process and self._process.stdin:
                try:
                    self._process.stdin.close()
                except Exception: # noqa
                    pass
            if self._process:
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None

    async def _stream_and_play(self, text: str):
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
            if chunk["type"] == "audio":
                data = chunk["data"]
                if self._process and self._process.stdin:
                    try:
                        self._process.stdin.write(data)
                        self._process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        break
        if self._stop_event.is_set():
            self._kill_player()

    def _kill_player(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def interrupt(self):
        self._stop_event.set()
        self._kill_player()

        if self._wait_thread and self._wait_thread.is_alive():
            self._wait_thread.join(timeout=1)

        self._process = None