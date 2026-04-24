from .ltm import LTMClient
from .stm import STMClient
from .note import NoteManager

class MemoryManager:
    def __init__(self):
        self.LTM = LTMClient()
        self.STM = STMClient()
        self.note = NoteManager()

    def add_memory(self, message: str, user_id):
        self.STM.add_memory(message, user_id)
        self.LTM.add_memory(message, user_id)

    def search_ltm(self,message: str, user_id):
        ltm_raw = self.LTM.search_memory(message, user_id)

        if isinstance(ltm_raw, dict):
            ltm_entries = ltm_raw.get("results", [])
        elif isinstance(ltm_raw, list):
            ltm_entries = ltm_raw
        else:
            ltm_entries = []

        ltm_memories = []
        for entry in ltm_entries:
            inner = entry.get("memory", {})
            memory_content = inner.get("memory")
            score_value = inner.get("score")
            if memory_content is not None:
                ltm_memories.append({"memory": memory_content, "score": score_value})
        return ltm_memories

    def search_stm(self, user_id):
        stm_result = self.STM.get_memory(user_id)
        stm_memories = stm_result.get("results", [])
        return stm_memories