from airis_sdk import Websocket, Action
from simpleeval import simple_eval
import json
import asyncio

async def main():
    client = Websocket()
    await client.connect("ws://localhost:8000/game")
    await client.startup("calculator")

    await client.register_actions([
        Action(
            name="calculator",
            description="计算数学表达式",
            schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式,例如:2+2"
                    }
                },
                "required": ["expression"]
            }
        ).to_dict()
    ])

    async def on_action(payload):
        action_name=payload.get("name")
        action_id=payload.get("id")
        data=json.loads(payload.get("data"))

        if action_name=="calculator":
            try:
                res=simple_eval(data.get("expression"))
                await client.send_action_result(action_id,True, res)
            except Exception as e:
                await client.send_action_result(action_id,False, f"error: {e}")

    client.on_action(on_action)
    loop = asyncio.Future()
    await loop
    await client.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit()