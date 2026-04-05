import subprocess
import threading
from typing import Optional

from ..common.config import ConfigManager
from ..common.logger import LogManager

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

    def stream_tts(self, text: str):
        self.interrupt()

        cmd = [
            "edge-playback",
            "--text", text,
            "--voice", self.voice,
            "--rate", self.rate,
            "--volume", self.volume,
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self._process.wait()
        self._process = None

    def interrupt(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

        if self._wait_thread and self._wait_thread.is_alive():
            self._wait_thread.join(timeout=0.5)