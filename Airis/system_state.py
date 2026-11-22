import websockets
import asyncio
import json
import time
from typing import Any, Optional


GATEWAY_URL = "ws://localhost:8000"
RUNTIME_URL = f"{GATEWAY_URL.rstrip('/')}/runtime"


async def send_runtime_message(event: str, data: dict[str, Any], meta: Optional[dict[str, Any]] = {}) -> None:
    message = {
        "event": event,
        "data": data,
        "meta": {
            "timestamp": time.time()
        } | meta
    }
    async with websockets.connect(RUNTIME_URL) as ws:
        await ws.send(json.dumps(message, ensure_ascii=False))


async def system_states():
    async with websockets.connect(RUNTIME_URL) as ws:
        while True:
            states=json.loads(await ws.recv())
            event = states.get("event").split('/')
            data = states.get("data")
            meta = states.get("meta")

            event_handler(event, data, meta)

def event_handler(event, data, meta):
    if event[0] == "llm":
        if event[1] == "stream":
            pass
        if event[1] == "stream_start":
            print(f"LLM stream started: {data}")
        if event[1] == "stream_end":
            print(f"LLM stream ended: {data}")
        if event[1] == "tool_call":
            print(f"Tool call: {data}")

    if event[0] == "tts":
        if event[1] == "interrupt":
            print(f"TTS interrupted: {data}")
        if event[1] == "end":
            print(f"TTS ended: {data}")
        if event[1] == "start":
            print(f"TTS started: {data}")

    if event[0] == "websocket":
        if event[1] == "connected":
            pass
        if event[1] == "message":
            direction = data.get("direction") if isinstance(data, dict) else "?"
            arrow = f"[{direction}]"
            print(f"{arrow} {data.get('path')}: {data.get('message')}")
        if event[1] == "disconnected":
            pass

    if event[0] == "debug":
        if event[1] == "info":
            print(f"[DEBUG] info: {data}, t={meta}",)


if __name__ == "__main__":
    asyncio.run(system_states())
