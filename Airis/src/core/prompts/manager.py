"""
SystemPrompt: static + ChatHistoryPrompt: dynamic + RuntimeStatePrompt: dynamic -> prompt
"""
import os
from ..common.config import ConfigManager   #配置管理器

class PromptBuilder:
    def __init__(self):
        self.config = ConfigManager()
        self.prompt = ""

    def build(self, profile, context: dict = None) -> str:
        prompts_path = self.config.get_json(f"prompts.path.{profile}")
        context = context or {}
        parts = []

        for prompt in prompts_path:
            prompt_path = self.config.base_dir / prompt
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    filename = os.path.splitext(os.path.basename(prompt))[0]
                    text = f.read()
                    text = text.format(**context[f"{filename}.md"])
                    parts.append(f"[{filename}]\n{text}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt {prompt_path} not found")

            except KeyError as e:
                raise KeyError(f"You need to pass in the variables {e} required by {prompt}")

        self.prompt = "".join(parts)
        return self.prompt