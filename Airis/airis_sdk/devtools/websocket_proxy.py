import websockets
import asyncio
import json
import sys

# config
TARGET = "ws://localhost:8000/game"  # 真实服务端地址
PROXY_HOST = "localhost"
PROXY_PORT = 8000                    # 代理监听端口
UNICODE_DECODE = True                # 是否解码Unicode字符

def expand_json_strings(o):
    if isinstance(o, dict):
        return {k: expand_json_strings(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [expand_json_strings(i) for i in o]
    elif isinstance(o, str):
        try:
            nested = json.loads(o)
            return expand_json_strings(nested)
        except (json.JSONDecodeError, TypeError):
            return o
    else:
        return o

async def handle_client(local_ws, path): # noqa
    async with websockets.connect(TARGET) as remote_ws:
        async def forward(src, dst, tag):
            try:
                async for message in src:
                    if UNICODE_DECODE:
                        try:
                            obj = json.loads(message)
                            obj = expand_json_strings(obj)
                            print(f"[{tag}] {json.dumps(obj, ensure_ascii=False)}")
                        except json.JSONDecodeError:
                            print(f"[{tag}] {message}")
                    else:
                        print(f"[{tag}] {message}")
                    await dst.send(message)
            except websockets.ConnectionClosed:
                pass

        task1 = asyncio.ensure_future(forward(local_ws, remote_ws, "C2S"))
        task2 = asyncio.ensure_future(forward(remote_ws, local_ws, "S2C"))
        await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)

async def main():
    async with websockets.serve(handle_client, PROXY_HOST, PROXY_PORT):
        print(f"ws://{PROXY_HOST}:{PROXY_PORT} -> {TARGET}")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)