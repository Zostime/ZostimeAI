from mem0 import Memory
import os

from ..common.config import ConfigManager

# ===============================
# Memory
# ===============================
class MemoryClient:
    def __init__(self):
        self.config = ConfigManager()
        self.memory=self._init_memory()

    def _init_memory(self):
        os.environ["HF_TOKEN"] = self.config.get_env("HF_TOKEN")

        mem_config = {
            "vector_store": {
                "provider": self.config.get_json("memory.vector_store.provider"),
                "config": {
                    "collection_name": self.config.get_json("memory.vector_store.config.collection_name"),
                    "path": str(self.config.get_path("memory.vector_store.config.path"))
                }
            },
            "embedder": {
                "provider": self.config.get_json("memory.embedder.provider"),
                "config": {
                    "model": self.config.get_json("memory.embedder.config.model")
                }
            },
            "llm": {
                "provider": self.config.get_json("memory.llm.provider"),
                "config": {
                    "model": self.config.get_env("MEMORY_MODEL"),
                    "api_key": self.config.get_env("MEMORY_API_KEY"),
                    "openai_base_url": self.config.get_env("MEMORY_BASE_URL"),
                    "temperature": self.config.get_json("memory.llm.config.temperature"),
                    "max_tokens": self.config.get_json("memory.llm.config.max_tokens"),
                    "top_p": self.config.get_json("memory.llm.config.top_p"),
                }
            },
            "history_db": {
                "provider": self.config.get_json("memory.history_db.provider"),
                "config": {
                    "path": str(self.config.get_path("memory.history_db.config.path"))
                }
            }
        }
        return Memory.from_config(mem_config)

    def memory_add(self,message: str, user_id: str):
        self.memory.add(message, user_id=user_id)

    def memory_search(self,message: str,
                      user_id: str,
                      limit: int,
                      ):
        relevant_memories = self.memory.search(
            query=message,
            user_id=user_id,
            limit=limit
        )
        return relevant_memories