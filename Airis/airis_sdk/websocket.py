"""
参考了Neuro API
https://github.com/VedalAI/neuro-sdk/blob/main/API/SPECIFICATION.md
"""

from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import websockets
import json
import asyncio

@dataclass
class Action:
    """
        Airis API 中的可注册动作.

        :param name: 动作的唯一标识符. 小写字母, 单词间用下划线或短横线分隔.
        :param description: 动作的自然语言描述, 会直接呈现给 Airis.
        :param schema: 可选的 JSON Schema 对象，描述动作参数的结构.
                       必须为 {"type": "object", ...} 格式.
    """

    name: str
    description: str
    schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "description": self.description}
        if self.schema is not None:
            result["schema"] = self.schema
        return result

class Websocket:
    def __init__(self):
        self.uri = None
        self.ws = None
        self.game_name = None
        self._action_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._callback_is_async: bool = False
        self._listen_task: Optional[asyncio.Task] = None

    async def startup(self, game_name: str) -> None:
        """
        这条消息应在游戏开始时立即发送, 以告知 Airis 游戏正在运行.
        这个消息会清除游戏之前注册的所有操作并进行初始设置, 因此应该是你发送的第一条消息.

        :param game_name: 游戏名称. 这用来识别游戏. 它应该始终保持不变, 不应改变.
        你应该使用游戏的显示名称, 包括任何空格和符号(例如 "Buckshot Roulette").
        服务器不会包含这个字段.
        """

        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")
        self.game_name = game_name
        payload = {
            "command": "startup",
            "game": game_name
        }

        await self.ws.send(json.dumps(payload))

    async def send_context(self, message: str, silent: bool = False) -> None:
        """
        这条消息可以用来通知 Airis 游戏中正在发生的事情。

        :param message: 一条描述游戏中发生情况的纯文本信息.这些信息将直接由 Airis 接收.
        :param silent: 若为 True, 消息将被添加到 Airis 的上下文中, 而不会提示她对此作出回应.
                       若为 False, Airis 可能会直接回应消息, 除非她正忙于与其他人交谈或聊天.
        """

        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")
        payload = {
            "command": "context",
            "game": self.game_name,
            "data": {
                "message": message,
                "silent": silent
            }
        }

        await self.ws.send(json.dumps(payload))

    async def register_actions(self, actions: list[dict]) -> None:
        """
        此消息为 Airis 注册一个或多个操作以供使用.

        :param actions: 一组需要注册的动作. 如果你尝试注册已经注册的动作, 它会被忽略.
        """

        payload = {
            "command": "actions/register",
            "game": self.game_name,
            "data": {
                "actions": actions
            }
        }

        await self.ws.send(json.dumps(payload))

    async def unregister_actions(self, action_names: list[str]) -> None:
        """
        该消息会取消注册一个或多个动作, 阻止 Airis 继续使用它们。

        :param action_names: 要注销的操作名称. 如果你尝试注销一个未注册的操作, 不会有任何问题.
        """

        payload = {
            "command": "actions/unregister",
            "game": self.game_name,
            "data": {
                "action_names": action_names
            }
        }

        await self.ws.send(json.dumps(payload))

    async def force_actions(
            self,
            query: str,
            action_names: list[str],
            state: str = "",
            ephemeral_context: bool = False,
            priority: str = "low"
        ) -> None:
        """
        强制 Airis 立即从给定的动作列表中选择一个执行。
        服务器将构建一个临时的决策上下文, 要求 Airis 必须返回一个工具调用.

        :param query: 告诉 Airis 当前应该做什么的指令(例如 "It is your turn. Please place an O.").
        :param action_names: 限定 Airis 只能从中选择的动作名称列表。
        :param state: 可选,描述当前游戏完整状态的字符串(支持 Markdown 或 JSON).
        :param ephemeral_context: 若为 True, 此次强制请求的状态和指令在动作完成后会被遗忘;
                                  若为 False, 信息会保留在 Airis 的长期上下文中.
        :param priority: 决定 Airis 回应紧急程度的优先级。可选值:
                         - "low"：等待 Airis 说完当前的话再处理.
                         - "medium"：让 Airis 尽快结束当前话语.
                         - "high"：缩短 Airis 当前话语并立即处理.
                         - "critical"：立即打断 Airis 说话并处理 (谨慎使用).
        """
        payload = {
            "command": "actions/force",
            "game": self.game_name,
            "data": {
                "state": state,
                "query": query,
                "ephemeral_context": ephemeral_context,
                "priority": priority,
                "action_names": action_names
            }
        }

        await self.ws.send(json.dumps(payload))

    async def send_action_result(self, action_id: str, success: bool, message: str = "") -> None:
        """
        将 Airis 要求执行的动作结果返回给服务器。

        :param action_id: 从 Airis 下发的 action 命令中获取的唯一 ID。
        :param success: 动作是否执行成功。若为 False 且该动作属于一次 force 请求，
                        整个 force 流程会被立即重试。
        :param message: 可选的附加信息。成功时可提供简短上下文提示；失败时应包含错误原因。
        """

        payload = {
            "command": "action/result",
            "game": self.game_name,
            "data": {
                "id": action_id,
                "success": success,
                "message": message
            }
        }

        await self.ws.send(json.dumps(payload))

    def on_action(self, callback) -> None:
        """
        注册一个回调函数，用于处理服务器下发的 action 命令。

        回调函数应接收一个包含以下字段的字典参数：
            - id (str): 本次动作调用的唯一标识，需在 send_action_result 中原样返回。
            - name (str): AI 决定执行的动作名称。
            - data (str): 包含动作参数的 JSON 字符串，需要解析并校验。

        :param callback: 签名为 `def handle_action(action_data: dict) -> None` 的函数。
        """

        self._action_callback = callback
        self._callback_is_async = asyncio.iscoroutinefunction(callback)

    async def _listen(self) -> None:
        """
        内部监听循环, 持续接收 WebSocket 消息并处理.
        """
        try:
            async for raw_msg in self.ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                if data.get("command") == "action":
                    cb = self._action_callback
                    is_async = self._callback_is_async

                    if cb is None:
                        continue

                    payload = data["data"]
                    if is_async:
                        await cb(payload) # noqa
                    else:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, cb, payload)
        except websockets.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            pass
        finally:
            self.ws = None

    async def connect(self, uri) -> None:
        self.uri = uri
        self.ws = await websockets.connect(uri)
        self._listen_task = asyncio.create_task(self._listen())

    async def disconnect(self) -> None:
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self.ws:
            await self.ws.close()
            self.ws = None

    def is_connected(self) -> bool:
        return self.ws is not None and self.ws.open
