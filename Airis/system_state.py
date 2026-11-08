import websockets
import asyncio
import json

async def system_states():
    async with websockets.connect("ws://localhost:8000/runtime") as ws:
        while True:
            states=json.loads(await ws.recv())
            event = states.get("event")
            data = states.get("data")

            if event == "llm/stream":
                pass
            if event == "llm/tool_call":
                print(f"Tool call: {data}")

asyncio.run(system_states())
