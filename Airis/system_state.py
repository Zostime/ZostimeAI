import websockets
import asyncio
import json

async def system_states():
    async with websockets.connect("ws://localhost:8000/runtime") as ws:
        while True:
            states=json.loads(await ws.recv())
            event = states.get("event").split('/')
            data = states.get("data")

            event_handler(event, data)

def event_handler(event, data):
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

asyncio.run(system_states())
