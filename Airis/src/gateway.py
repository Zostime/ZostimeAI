from websockets.server import WebSocketServerProtocol # noqa
from typing import Any, Dict, Optional
from collections import defaultdict
from urllib.parse import urlparse
import websockets
import threading
import asyncio
import json
import time

class WebSocketServer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.clients = defaultdict(dict)
        self.queue = asyncio.Queue()
        self.handlers = {}
        self.loop = None

    def on(self, path: str, handler):
        self.handlers[path] = handler

    async def run(self):
        self.loop = asyncio.get_running_loop()
        ws_url = self.config.get_json("system.websocket_url")
        parsed = urlparse(ws_url)
        host = parsed.hostname
        port = parsed.port
        async with websockets.serve(self._connection_handler, host, port):
            await self._sender_loop()

    async def _connection_handler(self, websocket: WebSocketServerProtocol):
        path = websocket.path.lstrip("/")
        client_id = id(websocket)
        self.clients[path][client_id] = websocket
        self.logger.debug(f"[websocket] New connection {client_id} -> {path}")

        try:
            if path in self.handlers:
                await self.handlers[path](websocket, client_id)
        except websockets.ConnectionClosed:
            self.logger.debug(f"[websocket] {client_id} -> {path} connection closed.")
        except Exception:  # noqa
            pass
        finally:
            self.clients[path].pop(client_id, None)

    async def publish_async(self, path: str, data: Any, client_id: int = None):
        await self.queue.put((path, client_id, data))

    def publish(self, path: str, data: Any, client_id: int = None):
        if self.loop is None:
            raise RuntimeError("Event loop not ready")
        asyncio.run_coroutine_threadsafe(
            self.queue.put((path, client_id, data)),
            self.loop
        )

    async def _sender_loop(self):
        while True:
            path, client_id, data = await self.queue.get()

            targets = []

            if client_id is not None:
                ws = self.clients.get(path, {}).get(client_id)
                if ws:
                    targets.append(ws)
            else:
                targets = list(self.clients.get(path, {}).values())

            msg = json.dumps(data)

            for ws in targets:
                try:
                    await ws.send(msg)
                except:  # noqa
                    pass

class ProtocolRouter:
    ws = None
    _initialized = False
    logger = None
    state = None
    event_bus = None

    @classmethod
    def setup(cls, *, config, logger, state, event_bus):
        if cls._initialized:
            return

        cls.logger = logger
        cls.state = state
        cls.event_bus = event_bus
        cls.ws = WebSocketServer(config, logger)

        cls.ws.on("runtime", cls.Runtime.handle)
        cls.ws.on("game", cls.Game.handle)
        # ...

        cls._initialized = True

        threading.Thread(target=lambda: asyncio.run(cls.ws.run()), daemon=True).start()

    class Runtime:
        @staticmethod
        async def handle(websocket: WebSocketServerProtocol, client_id: int):  # noqa
            async for msg in websocket:
                ProtocolRouter.logger.debug(f"[websocket] {client_id} -> {msg}")

        @staticmethod
        def emit(event: str, data: dict[str, Any], meta: Optional[dict[str, Any]] = {}):
            ProtocolRouter.ws.publish(
                path="runtime",
                data={
                    "event": event,
                    "data": data,
                    "meta": {
                        "timestamp": time.time()
                    } | meta
                }
            )

    class Game:
        sessions: Dict[int, 'ProtocolRouter.Game.Session'] = {}  # client_id -> Session

        class Session:
            def __init__(self, client_id: int, websocket: WebSocketServerProtocol):
                self.client_id = client_id
                self.websocket = websocket
                self.game_name: Optional[str] = None
                self.registered_actions: dict = {}
                self.pending_action_id: Optional[str] = None
                self.pending_actions: Dict[str, asyncio.Future] = {}
                self.forced_action_names: list = []
                self.force_payload: Optional[dict] = None

        @staticmethod
        async def handle(websocket: WebSocketServerProtocol, client_id: int):
            session = ProtocolRouter.Game.Session(client_id, websocket)
            ProtocolRouter.Game.sessions[client_id] = session

            try:
                async for raw_msg in websocket:
                    try:
                        data = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        ProtocolRouter.logger.warning(f"Invalid JSON: {raw_msg}")
                        continue
                    await ProtocolRouter.Game._handle_message(session, data)
            except websockets.ConnectionClosed:
                pass
            finally:
                ProtocolRouter.Game.sessions.pop(client_id, None)

        @staticmethod
        async def _handle_message(session, message):
            command = message.get("command")
            data = message.get("data")
            game = message.get("game")

            if command == "startup":
                session.game_name = game

            elif command == "context":
                message = data.get("message", "")
                silent = data.get("silent", True)
                if silent:
                    ProtocolRouter.state.agent.unread_events.append(f"来自{game}的msg:{message}")
                else:
                    ProtocolRouter.state.agent.unread_events.append(f"[应该回复]来自{game}的msg:{message}")

            elif command == "actions/register":
                for act in data.get("actions", []):
                    session.registered_actions[act["name"]] = act

            elif command == "actions/unregister":
                for name in data.get("action_names", []):
                    session.registered_actions.pop(name, None)

            elif command == "actions/force":
                query = data.get("query", "")
                action_names = data.get("action_names", [])
                state = data.get("state", "")
                priority = data.get("priority", "low")
                ephemeral_context = data.get("ephemeral_context", False)

                session.forced_action_names = action_names
                if not ephemeral_context:
                    session.force_payload = {
                        "query": query,
                        "state": state
                    }
                    ProtocolRouter.state.agent.unread_events.append(f"来自{game}的[force]: {session.force_payload}")

                ProtocolRouter.event_bus.emit(
                    event_type="input",
                    data={
                        "type": "force_action",
                        "source": game,
                        "content": query,
                        "ephemeral_context": ephemeral_context,
                    },
                    priority=priority
                )  # 触发LLM

            elif command == "action/result":
                action_id = data.get("id")
                if not action_id:
                    return
                future = session.pending_actions.pop(action_id, None)
                if future and not future.done():
                    future.set_result(data)

        @staticmethod
        def run_action(session, data):
            async def _action_worker():
                action_id = data["data"]["id"]

                loop = asyncio.get_running_loop()
                future = loop.create_future()

                session.pending_actions[action_id] = future

                await ProtocolRouter.ws.publish_async(
                    path="game",
                    client_id=session.client_id,
                    data=data
                )

                return await future

            return asyncio.run_coroutine_threadsafe(
                _action_worker(),
                ProtocolRouter.ws.loop
            ).result()
