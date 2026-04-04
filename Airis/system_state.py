import websockets
import asyncio
import json

async def system_states():
    async with websockets.connect("ws://localhost:8085") as ws:
        while True:
            states=json.loads(await ws.recv())
            if states.get("type") == "llm_stream":
                chunk=states.get("data")
                print(chunk, flush=True, end="")

            if states.get("type") == "emotion":
                pass

asyncio.run(system_states())
