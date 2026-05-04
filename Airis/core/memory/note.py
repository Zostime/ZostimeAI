import threading
import asyncio
import json
import os

from ..common.config import ConfigManager   #配置管理器
from airis_sdk import Websocket, Action

class NoteManager:
    def __init__(self):
        self.config = ConfigManager()
        self.websocket_url = self.config.get_json("system.websocket_url")
        self.max_char = int(self.config.get_json("memory.note.max_char", 128))
        self.file_path = self.config.get_json("memory.note.path")
        threading.Thread(
            target=lambda: asyncio.run(self._init()),
            daemon=True
        ).start()

    async def _init(self):
        os.makedirs(self.file_path, exist_ok=True)
        client = Websocket()
        await client.connect(f"{self.websocket_url}/game")
        await client.startup("note")

        await client.register_actions([
            Action(
                name="take_note",
                description="将笔记写入持久笔记存储, 你可以在prompt中看到",
                schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string"
                    },
                    "mode": {
                        "type": "string",
                        "description": "\"append\" | \"overwrite\"",
                    }
                },
                "required": ["content", "mode"]
            }
            ).to_dict()
        ])

        async def on_action(payload):
            action_name = payload.get("name")
            action_id = payload.get("id")
            data = json.loads(payload.get("data"))

            if action_name == "take_note":
                try:
                    content = data.get("content")
                    mode = data.get("mode")

                    if mode not in ("append", "overwrite"):
                        await client.send_action_result(
                            action_id,
                            False,
                            'error: "mode" 只支持 "append" 或 "overwrite"'
                        )
                    else:
                        content_length = len(content.encode("utf-8") + self.read()) if mode == "append" else len(content.encode("utf-8"))
                        if content_length > self.max_char:
                            await client.send_action_result(
                                action_id,
                                False,
                                f"error: 内容超过最大长度限制 ({self.max_char} 字符, 尝试减少字数或者选择覆盖)"
                            )
                            return

                        with open(
                                os.path.join(self.file_path, "note.txt"),
                                "a" if mode == "append" else "w",
                                encoding="utf-8"
                        ) as f:
                            f.write(content + "\n")

                        await client.send_action_result(action_id, True, "write success")
                except Exception as e:
                    await client.send_action_result(action_id, False, f"error: {e}")

        client.on_action(on_action)
        await asyncio.Event().wait()
        await client.disconnect()

    def read(self):
        path = os.path.join(self.file_path, "note.txt")
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()