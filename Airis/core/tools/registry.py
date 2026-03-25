import importlib
import os
from typing import Dict, List, Callable, Any

class ToolRegistry:
    def __init__(self):
        self.tools: List[Dict] = []
        self.tool_map: Dict[str, Callable] = {}
        self.tools_dir = os.path.dirname(__file__)
        self._load_tools()

    def _load_tools(self):
        self.tools.clear()
        self.tool_map.clear()

        for name in os.listdir(self.tools_dir):
            tool_path = os.path.join(self.tools_dir, name)

            if not os.path.isdir(tool_path) or name.startswith("__"):
                continue

            try:
                module_path = f"core.tools.{name}.tool"
                module = importlib.import_module(module_path)

                if not hasattr(module, "TOOL"):
                    print(f"[tools] 跳过 {name}: 缺少 TOOL")
                    continue

                if not hasattr(module, "run"):
                    print(f"[tools] 跳过 {name}: 缺少 run()")
                    continue

                tool_schema = module.TOOL
                tool_func = module.run

                tool_name = tool_schema["function"]["name"]

                self.tools.append(tool_schema)
                self.tool_map[tool_name] = tool_func

                print(f"[tools] 已加载: {tool_name}")

            except Exception as e:
                print(f"[tools] 加载失败 {name}: {e}")

    def get_tools(self) -> List[Dict]:
        return self.tools

    def run_tool(self, tool_call: Dict[str, Any]) -> Any:
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})

        if name not in self.tool_map:
            return f"未知工具: {name}"

        try:
            return self.tool_map[name](**args)
        except Exception as e:
            return f"工具执行失败({name}): {e}"

tools=ToolRegistry()
