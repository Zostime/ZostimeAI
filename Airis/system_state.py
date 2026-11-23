import asyncio, fnmatch, json, re, sys, threading, time, tkinter as tk
import websockets

RUNTIME_URL = "ws://localhost:8000/runtime"
FONT = ("Consolas", 11)
MAX_LINES = 5000
HELP_TEXT = """\
commands:
    /help   :   获取命令提示
    /clear  :   清除终端文本
    /events :   查看已注册路由清单
    /check <path>   :   查看路径状态
    /mute <path> ...    :   屏蔽事件
    /unmute <path> ...  :   取消屏蔽事件
    /listen all|none|<path> ...    :   监听事件
"""


def route(*patterns):
    """把事件路径标在处理方法上，EventRouter.bind() 会自动注册"""

    def decorate(handler):
        handler.route_patterns = tuple(patterns)
        return handler

    return decorate


def match_path(pattern, path):
    """pattern / path 都是段元组: * 匹配一层, ** 匹配零到多层"""
    for index, segment in enumerate(pattern):
        if segment == "**":
            return index == len(pattern) - 1 or match_path(pattern[index + 1:], path[index:])
        if index >= len(path) or not fnmatch.fnmatchcase(path[index], segment):
            return False
    return len(path) == len(pattern)


def to_pattern(value, expand=True):
    """文本 -> 段元组. llm => llm/**; llm/* 为一层; llm/** 为所有后代"""
    text = str(value).strip().lower().replace("\\", "/")
    if "/" not in text:
        text = text.replace(".", "/")
    if text in ("", "none"):
        return ()
    segments = tuple(part for part in text.split("/") if part)
    if not segments:
        return ()
    if segments in (("*",), ("**",)) or (expand and segments[0] == "all"):
        return ("**",)
    if expand and len(segments) == 1 and not any(c in segments[0] for c in "*?["):
        return (segments[0], "**")
    return segments


def pattern_text(pattern):
    return "/".join(pattern) or "none"


def path_of(event):
    """'A/b//c' 或 ['a','b'] → ('a','b','c')"""
    if isinstance(event, (list, tuple)):
        parts = [str(part) for part in event]
    else:
        parts = str(event or "").split("/")
    return tuple(part.strip().lower() for part in parts if part and part.strip())


def specificity(pattern):
    literals = sum(1 for segment in pattern if not any(c in segment for c in "*?["))
    return literals, len(pattern)


class EventRouter:
    def __init__(self, emit):
        self.emit = emit
        self.show_unmatched = False
        self.routes = []
        self.enabled = (("**",),)      # 启动全开
        self.disabled = ()

    def register(self, pattern, handler=None):
        compiled = to_pattern(pattern)

        def attach(handler_function):
            self.routes = [item for item in self.routes if item[0] != compiled]
            self.routes.append((compiled, handler_function))
            self.routes.sort(key=lambda item: specificity(item[0]), reverse=True)
            return handler_function

        return attach if handler is None else attach(handler)

    def bind(self, owner):
        """扫描 owner 上所有 @route 标记的方法并注册"""
        for name, member in vars(type(owner)).items():
            for pattern in getattr(member, "route_patterns", ()):
                self.register(pattern, getattr(owner, name))

    def is_enabled(self, path):
        return (bool(path)
                and not any(match_path(pattern, path) for pattern in self.disabled)
                and any(match_path(pattern, path) for pattern in self.enabled))

    def dispatch(self, event, data=None, meta=None):
        path = path_of(event)
        if not self.is_enabled(path):
            return

        for pattern, handler in self.routes:
            if match_path(pattern, path):
                try:
                    handler(path, {} if data is None else data, {} if meta is None else meta)
                except Exception as exc:
                    self.emit(f"! {pattern_text(path)} @{pattern_text(pattern)} {exc!r}\n", "#f44747")
                return

        if self.show_unmatched:
            self.emit(f"? {pattern_text(path)} {data} {meta}\n", "#808080")

    def handle_command(self, line):
        body = line.strip().lstrip("/")
        command, _, remainder = body.partition(" ")
        command, arguments = command.lower(), remainder.split()

        if command in ("listen"):
            if arguments and arguments[0] in ("all"):
                self.enabled, self.disabled = (("**",),), ()
            elif arguments and arguments[0] in ("none"):
                self.enabled, self.disabled = (), ()
            elif arguments:
                self.enabled, self.disabled = tuple(to_pattern(item) for item in arguments), ()
            self.emit(f"on: {', '.join(pattern_text(p) for p in self.enabled) or '-'} | "
                      f"off: {', '.join(pattern_text(p) for p in self.disabled) or '-'}\n", "#4ec9b0")
            return True

        if command in ("mute"):
            self.disabled += tuple(to_pattern(item) for item in arguments)
            self.emit(f"off: {', '.join(pattern_text(p) for p in self.disabled) or '-'}\n", "#dcdcaa")
            return True

        if command == "unmute":
            if arguments:
                dropped = {to_pattern(item) for item in arguments}
                self.disabled = tuple(p for p in self.disabled if p not in dropped)
            else:
                self.disabled = ()
            self.emit(f"off: {', '.join(pattern_text(p) for p in self.disabled) or '-'}\n", "#dcdcaa")
            return True

        if command == "events":
            self.emit("".join(f"{pattern_text(pattern)}\n" for pattern, _ in self.routes))
            return True

        if command == "check":
            path = to_pattern(arguments[0]) if arguments else ()
            matched = next((pattern for pattern, _ in self.routes if match_path(pattern, path)), None)
            self.emit(f"{pattern_text(path)}: {'on' if self.is_enabled(path) else 'off'} -> "
                      f"{pattern_text(matched) if matched else '-'}\n")
            return True

        if command == "unknown":
            if arguments:
                self.show_unmatched = arguments[0].lower() in ("on", "1", "yes")
            else:
                self.show_unmatched = not self.show_unmatched
            self.emit(f"unknown: {'on' if self.show_unmatched else 'off'}\n")
            return True

        return False


class App:
    json_decoder = json.JSONDecoder()

    @staticmethod
    def setup_dpi():
        if sys.platform != "win32":
            return
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # Windows 8.1+
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()    # Windows 7
            except Exception:
                pass

    def __init__(self):
        self.setup_dpi()
        self.root = tk.Tk()
        self.root.title("runtime")
        self.root.geometry("960x620")
        self.root.configure(bg="#1e1e1e")
        self.configured_colors = set()

        output_frame = tk.Frame(self.root, bg="#1e1e1e")
        output_frame.pack(fill="both", expand=True, padx=6, pady=(6, 3))
        scrollbar = tk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")
        self.output_view = tk.Text(output_frame, bg="#1e1e1e", fg="#d4d4d4",
                                   insertbackground="#d4d4d4", font=FONT, wrap="word",
                                   relief="flat", yscrollcommand=scrollbar.set)
        self.output_view.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.output_view.yview)
        self.output_view.configure(state="disabled")

        input_frame = tk.Frame(self.root, bg="#252526")
        input_frame.pack(fill="x", padx=6, pady=(0, 6))
        tk.Label(input_frame, text=">", bg="#252526", fg="#4ec9b0", font=FONT).pack(side="left", padx=(8, 0))
        self.input_entry = tk.Entry(input_frame, bg="#252526", fg="#d4d4d4",
                                    insertbackground="#d4d4d4", font=FONT, relief="flat")
        self.input_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=6, pady=4)
        self.input_entry.bind("<Return>", self.on_enter)
        self.input_entry.focus_set()

        self.router = EventRouter(self.emit)
        self.router.bind(self)

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_receiver, daemon=True).start()

    def emit(self, text, color=None):
        def write():
            self.output_view.configure(state="normal")
            if color:
                if color not in self.configured_colors:
                    self.output_view.tag_configure(color, foreground=color)
                    self.configured_colors.add(color)
                self.output_view.insert("end", text, color)
            else:
                self.output_view.insert("end", text)
            line_count = int(self.output_view.index("end-1c").split(".")[0])
            if line_count > MAX_LINES:
                self.output_view.delete("1.0", f"{line_count - MAX_LINES}.0")
            self.output_view.see("end")
            self.output_view.configure(state="disabled")

        self.root.after(0, write)

    def clear_output(self):
        def clear():
            self.output_view.configure(state="normal")
            self.output_view.delete("1.0", "end")
            self.output_view.configure(state="disabled")

        self.root.after(0, clear)

    def parse_json(self, value):
        text = value.lstrip()
        if not text:
            return None, ""
        try:
            parsed, end = self.json_decoder.raw_decode(text)
            return parsed, text[end:]
        except json.JSONDecodeError as exc:
            return exc, text

    def on_enter(self, event=None):
        line = self.input_entry.get().strip()
        self.input_entry.delete(0, "end")
        if line:
            self.emit(f"> {line}\n", "#ce9178")
            self.submit(line)
        return "break"

    def submit(self, line):
        if line.startswith("/"):
            command = line[1:].split(None, 1)[0].lower()
            if command in ("clear"):
                self.clear_output()
            elif command in ("help"):
                self.emit(HELP_TEXT, "#4ec9b0")
            elif not self.router.handle_command(line):
                self.emit(f"unknown command: /{command}\n", "#f44747")
            return

        matched = re.match(r"(\S+)\s+(.*)", line, re.S)
        if matched:
            event, remainder = matched.group(1), matched.group(2)
        else:
            event, remainder = line, ""

        data, remainder = self.parse_json(remainder)
        if isinstance(data, json.JSONDecodeError):
            self.emit(f"! data json {data}\n", "#f44747")
            data, remainder = {}, ""
        meta, _ = self.parse_json(remainder)
        if isinstance(meta, json.JSONDecodeError):
            self.emit(f"! meta json {meta}\n", "#f44747")
            meta = {}

        if not self.loop or not self.loop.is_running():
            self.emit("! loop not running\n", "#f44747")
            return

        future = asyncio.run_coroutine_threadsafe(
            self.send_message(event, data or {}, meta or {}), self.loop)
        future.add_done_callback(
            lambda done: None if done.exception() is None
            else self.emit(f"! send {done.exception()!r}\n", "#f44747"))

    async def send_message(self, event, data, meta):
        async with websockets.connect(RUNTIME_URL) as websocket:
            await websocket.send(json.dumps(
                {"event": event, "data": data, "meta": {"t": time.time(), **meta}},
                ensure_ascii=False))

    async def receive_loop(self):
        while True:
            try:
                async with websockets.connect(RUNTIME_URL) as websocket:
                    self.emit("[connected]\n", "#4ec9b0")
                    async for raw in websocket:
                        try:
                            states = json.loads(raw)
                        except ValueError:
                            continue
                        if isinstance(states, dict):
                            self.router.dispatch(states.get("event"),
                                                 states.get("data"), states.get("meta"))
            except Exception as exc:
                self.emit(f"[disconnected] {exc!r}\n", "#f44747")
            await asyncio.sleep(3)

    def start_receiver(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.receive_loop())

    def run(self):
        self.root.mainloop()

    @route("llm/**")
    def on_llm(self, path, data, meta):
        sub_event = path[1] if len(path) > 1 else ""
        if sub_event == "stream":
            chunk = data.get("text") if isinstance(data, dict) else str(data)
            if chunk:
                self.emit(chunk)
        elif sub_event == "stream_start":
            self.emit(f"LLM stream started: {data}\n", "#4ec9b0")
        elif sub_event == "stream_end":
            self.emit(f"\nLLM stream ended: {data}\n", "#4ec9b0")
        elif sub_event == "tool_call":
            self.emit(f"Tool call: {data}\n", "#4ec9b0")

    @route("tts/**")
    def on_tts(self, path, data, meta):
        sub_event = path[1] if len(path) > 1 else ""
        if sub_event == "start":
            self.emit(f"TTS started: {data}\n", "#dcdcaa")
        elif sub_event == "end":
            self.emit(f"TTS ended: {data}\n", "#dcdcaa")
        elif sub_event == "interrupt":
            self.emit(f"TTS interrupted: {data}\n", "#dcdcaa")

    @route("websocket/**")
    def on_websocket(self, path, data, meta):
        sub_event = path[1] if len(path) > 1 else ""
        if sub_event == "connected":
            pass
        elif sub_event == "message":
            payload = data if isinstance(data, dict) else {}
            self.emit(f"[{payload.get('direction', '?')}] {payload.get('path')}: "
                      f"{payload.get('message')}\n", "#9cdcfe")
        elif sub_event == "disconnected":
            pass

    @route("debug/**")
    def on_debug(self, path, data, meta):
        sub_event = path[1] if len(path) > 1 else ""
        if sub_event == "info":
            self.emit(f"[DEBUG] info: {data}, t={meta}\n", "#808080")

if __name__ == "__main__":
    App().run()
