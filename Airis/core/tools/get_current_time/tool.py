from datetime import datetime

TOOL = {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }

def run():
    try:
        return datetime.now()
    except Exception as e:
        return f"获取时间时发生错误:{e}"