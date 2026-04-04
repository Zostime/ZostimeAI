from mem0 import Memory
import os
import math
import time

from ..common.config import ConfigManager   #配置管理器
from ..common.logger import LogManager      #日志管理器

# ===============================
# LTM客户端
# ===============================
class LTMClient:
    def __init__(self):
        self.config = ConfigManager()
        self.memory=self._init_memory()
        self.logger = LogManager("memory").get_logger()

    def _init_memory(self):
        os.environ["HF_TOKEN"] = self.config.get_env("HF_TOKEN")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(self.config.get_path("memory.ltm.embedder.config.path"))

        mem_config = {
            "vector_store": {
                "provider": self.config.get_json("memory.ltm.vector_store.provider"),
                "config": {
                    "collection_name": self.config.get_json("memory.ltm.vector_store.config.collection_name"),
                    "path": str(self.config.get_path("memory.ltm.vector_store.config.path"))
                }
            },
            "embedder": {
                "provider": self.config.get_json("memory.ltm.embedder.provider"),
                "config": {
                    "model": self.config.get_json("memory.ltm.embedder.config.model")
                }
            },
            "llm": {
                "provider": self.config.get_json("memory.ltm.llm.provider"),
                "config": {
                    "model": self.config.get_env("MEMORY_MODEL"),
                    "api_key": self.config.get_env("MEMORY_API_KEY"),
                    "openai_base_url": self.config.get_env("MEMORY_BASE_URL"),
                    "temperature": self.config.get_json("memory.ltm.llm.config.temperature"),
                    "max_tokens": self.config.get_json("memory.ltm.llm.config.max_tokens"),
                    "top_p": self.config.get_json("memory.ltm.llm.config.top_p"),
                }
            },
            "history_db": {
                "provider": self.config.get_json("memory.ltm.history_db.provider"),
                "config": {
                    "path": str(self.config.get_path("memory.ltm.history_db.config.path"))
                }
            }
        }
        return Memory.from_config(mem_config)

    @staticmethod
    def _compute_score(mem):
        now = time.time()

        metadata = mem.get("metadata") or {}
        importance = metadata.get("importance", 0.5)
        last_access = metadata.get("last_access", now)
        access_count = metadata.get("access_count", 1)

        # 时间衰减
        time_diff = now - last_access
        recency = math.exp(-time_diff / (86400 * 7))  # 7天衰减

        # 使用频率
        access = min(access_count / 5, 1.0)

        score = (
                importance * 0.6 +
                recency * 0.3 +
                access * 0.1
        )

        return score

    def add_memory(self, message: str, user_id: str):
        metadata = {
            "importance": 0.5,
            "created_at": time.time(),
            "last_access": time.time(),
            "access_count": 1
        }

        self.memory.add(
            message,
            user_id=user_id,
            metadata=metadata
        )

    def search_memory(self, message: str, user_id: str):
        results = self.memory.search(
            query=message,
            user_id=user_id,
            limit=self.config.get_json("memory.ltm.limit"),
            threshold=self.config.get_json("memory.ltm.threshold")
        )

        scored = []

        for mem in results.get("results", []):
            score = self._compute_score(mem)

            if score > 0.3:
                metadata = mem.get("metadata") or {}
                metadata["last_access"] = time.time()
                metadata["access_count"] = metadata.get("access_count", 1) + 1

                scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        filtered = [m for m, _ in scored]

        return {"results": filtered[:self.config.get_json("memory.ltm.limit")]}