from .calculator import calculate

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式,例如:2+2"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

TOOL_MAP = {
    "calculate": calculate
}

def run_tool(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name in TOOL_MAP:
        return TOOL_MAP[name](**args)

    return f"未知工具:{name}"