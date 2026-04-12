import os
from ..common.config import ConfigManager

class PromptManager:
    def __init__(self):
        self.config = ConfigManager()
        self.prompts_path = self.config.get_json("prompts.path")
        self.prompt = ""

    def render(self, context: dict = None) -> str:
        context = context or {}
        parts = []

        for prompt in self.prompts_path:
            try:
                with open(prompt, "r", encoding="utf-8") as f:
                    text = f.read()
                    text = text.format(**context)
                    filename = os.path.basename(prompt)
                    parts.append(f"# {filename}\n{text}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt {prompt} not found")

        self.prompt = "".join(parts)
        return self.prompt

    def append(self, context) -> str:
        self.prompt += context
        return self.prompt