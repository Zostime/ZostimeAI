import websockets
import asyncio
import json

async def system_states():
    async with websockets.connect("ws://localhost:8085") as ws:
        while True:
            states=json.loads(await ws.recv())
            if states["data"].get("type") == "system_tick":
                continue
            if states.get("type") == "llm_stream":
                continue
            print(states)
asyncio.run(system_states())
