from playsound3 import playsound
import os
from pathlib import Path

audios_dir = f"{Path(__file__).parent}/audios"

TOOL = {
    "type": "function",
    "function": {
        "name": "sound_board",
        "description": "播放指定音效",
        "parameters": {
            "type": "object",
            "properties": {
                "audio": {
                    "type": "string",
                    "description": f"音频名称,只可以是{os.listdir(audios_dir)}中的任意一个"
                }
            },
            "required": ["audio"]
        }
    }
}

def run(audio: str):
    try:
        playsound(f"{audios_dir}/{audio}")
        return "播放成功"
    except Exception as e:
        return f"播放失败: {e}"
