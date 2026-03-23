from collections import defaultdict, deque
from ..common.config import ConfigManager
from ..common.logger import LogManager

# ===============================
# STM客户端
# ===============================
class STMClient:
    def __init__(self):
        self.config = ConfigManager()
        self.logger = LogManager("memory").get_logger()

        self.limit = self.config.get_json("memory.stm.limit")
        self.path = self.config.get_json("memory.stm.path")

        self.memory = defaultdict(lambda: deque(maxlen=self.limit))

        self.logger.info(f"STM初始化,limit={self.limit},path={self.path}")

    def add_memory(self, content, user_id):
        """
        添加一条记忆。

        Args:
            content: 记忆内容（字符串或其他可序列化对象）
            user_id: 用户标识
        """
        user_memory = self.memory[user_id]
        user_memory.append(content)
        self.logger.debug(f"Added memory for user {user_id}: {content}")

    def get_memory(self, user_id):
        """
        获取用户最近的 limit 条记忆(按添加时间倒序，最新在前)

        Args:
            user_id: 用户标识
        Returns:
            dict: 包含 "results" 键，值为记忆列表，每个元素为 {"memory": content}
                  若用户无记忆则返回 {"results": []}
        """
        user_memory = self.memory.get(user_id)
        if not user_memory:
            self.logger.debug(f"No memory found for user {user_id}")
            return {"results": []}

        # 按添加时间倒序（最新在前）
        reversed_memory = list(user_memory)[::-1]
        results = [{"memory": content} for content in reversed_memory]
        self.logger.debug(f"Retrieved {len(results)} memories for user {user_id}")
        return {"results": results}