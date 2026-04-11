from simpleeval import simple_eval

TOOL = {
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

def run(expression: str):
    try:
        return simple_eval(expression)
    except Exception as e:
        return f"计算错误:{e}"