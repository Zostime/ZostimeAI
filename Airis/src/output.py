import sys
from dataclasses import dataclass
from typing import TextIO


@dataclass
class OutputChannel:
    name: str
    enabled: bool = True
    stream: TextIO = sys.stdout

class OutputManager:
    def __init__(self):
        self.channels: dict[str, OutputChannel] = {}

    def register(
        self,
        name: str,
        *,
        enabled: bool = True,
        stream: TextIO = sys.stdout,
    ) -> None:
        """Register a new output channel."""
        if name in self.channels:
            raise ValueError(f"Output channel already exists: {name}")

        self.channels[name] = OutputChannel(
            name=name,
            enabled=enabled,
            stream=stream,
        )

    def unregister(self, name: str) -> None:
        """Remove an output channel."""
        self.channels.pop(name)

    def set_enabled(self, name: str, enabled: bool) -> None:
        self.channels[name].enabled = enabled

    def write(
        self,
        name: str,
        *values,
        sep: str = " ",
        end: str = "\n",
    ) -> None:
        channel = self.channels.get(name)

        if channel is None:
            raise ValueError(f"Unknown output channel: {name}")

        if not channel.enabled:
            return

        print(
            *values,
            sep=sep,
            end=end,
            file=channel.stream,
            flush=True,
        )
