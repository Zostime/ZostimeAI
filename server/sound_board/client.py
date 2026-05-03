from airis_sdk import Websocket, Action
from playsound3 import playsound
import os
from pathlib import Path
import json
import asyncio

audios_dir = f"{Path(__file__).parent}/audios"

async def main():
    client = Websocket()
    await client.connect("ws://localhost:8000/game")
    await client.startup("sound_board")

    await client.register_actions([
        Action(
            name="sound_board",
            description="播放指定音效",
            schema={
                "type": "object",
                "properties": {
                    "audio": {
                        "type": "string",
                        "description": f"音频名称,只可以是{os.listdir(audios_dir)}中的任意一个"
                    }
                },
                "required": ["audio"]
            }
        ).to_dict()
    ])

    async def on_action(payload):
        action_name=payload.get("name")
        action_id=payload.get("id")
        data=json.loads(payload.get("data"))

        if action_name=="sound_board":
            try:
                playsound(f"{audios_dir}/{data.get('audio')}")
                await client.send_action_result(action_id,True,"播放成功")
            except Exception as e:
                await client.send_action_result(action_id,False, f"播放失败: {e}")

    client.on_action(on_action)
    loop = asyncio.Future()
    await loop
    await client.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit()
