from .ltm import LTMClient
from .stm import STMClient

class MemoryManager:
    def __init__(self):
        self.LTM = LTMClient()
        self.STM = STMClient()

    def add_memory(self, message: str, user_id):
        self.STM.add_memory(message, user_id)
        self.LTM.add_memory(message, user_id)

    def search_memory(self,message: str, user_id):
        stm_result = self.STM.get_memory(user_id)
        stm_memories = stm_result.get("results", [])

        ltm_raw = self.LTM.search_memory(message, user_id)

        if isinstance(ltm_raw, dict):
            ltm_entries = ltm_raw.get("results", [])
        elif isinstance(ltm_raw, list):
            ltm_entries = ltm_raw
        else:
            ltm_entries = []

        ltm_memories = []
        for entry in ltm_entries:
            if isinstance(entry, dict):
                content = entry.get("memory")
            else:
                content = entry
            if content is not None:
                ltm_memories.append({"memory": content})

        combined = {"results": stm_memories + ltm_memories}
        return combined