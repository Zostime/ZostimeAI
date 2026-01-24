"""
短期记忆 (Short-Term Memory) 管理模块
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class STMEntry:
    """短期记忆条目"""
    id: str
    content: str
    timestamp: datetime
    source: str  # user, assistant, system
    importance: float = 0.5  # 重要性评分 0-1
    tokens: int = 0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'importance': self.importance,
            'tokens': self.tokens,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'STMEntry':
        return cls(
            id=data['id'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            importance=data.get('importance', 0.5),
            tokens=data.get('tokens', 0),
            metadata=data.get('metadata', {})
        )


class ShortTermMemory:
    """短期记忆管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化短期记忆

        Args:
            config: 配置字典
        """
        self.config = config
        self.capacity = config.get('capacity', 10)
        self.persistence = config.get('persistence', True)
        self.summary_enabled = config.get('summary_enabled', True)

        # 记忆缓冲区
        self.buffer: List[STMEntry] = []
        self.summary_buffer: List[str] = []

        # 统计信息
        self.stats = {
            'total_entries': 0,
            'summaries_generated': 0,
            'compression_ratio': 0.0
        }

        # 文件存储路径
        self.storage_path = Path("Files/storage/short_term")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 加载持久化数据
        if self.persistence:
            self._load_from_disk()

        logger.info(f"短期记忆初始化完成，容量: {self.capacity}")

    def add_entry(self, content: str, source: str, importance: float = 0.5,
                  metadata: Dict[str, Any] = None) -> str:
        """
        添加新的记忆条目

        Args:
            content: 内容
            source: 来源 (user/assistant/system)
            importance: 重要性评分
            metadata: 元数据

        Returns:
            str: 记忆ID
        """
        # 创建新条目
        entry_id = f"stm_{datetime.now().timestamp()}_{len(self.buffer)}"
        entry = STMEntry(
            id=entry_id,
            content=content,
            timestamp=datetime.now(),
            source=source,
            importance=importance,
            tokens=len(content.split()),
            metadata=metadata or {}
        )

        # 添加到缓冲区
        self.buffer.append(entry)

        # 维持容量限制
        if len(self.buffer) > self.capacity:
            # 移除最不重要的条目
            self.buffer.sort(key=lambda x: x.importance)
            removed = self.buffer.pop(0)
            logger.debug(f"移除不重要记忆: {removed.id}")

        # 更新统计
        self.stats['total_entries'] += 1

        # 自动生成摘要（如果需要）
        if self.summary_enabled and len(self.buffer) % 3 == 0:
            self._generate_summary()

        # 持久化
        if self.persistence:
            self._save_to_disk()

        logger.debug(f"添加短期记忆: {entry_id}, 重要性: {importance}")
        return entry_id

    def get_context(self, n: int = None, min_importance: float = 0.3) -> str:
        """
        获取上下文记忆

        Args:
            n: 获取最近N条记忆，None表示全部
            min_importance: 最小重要性阈值

        Returns:
            str: 格式化后的上下文
        """
        if not self.buffer:
            return ""

        # 筛选重要记忆
        relevant_entries = [
            entry for entry in self.buffer
            if entry.importance >= min_importance
        ]

        # 按时间排序（最近优先）
        relevant_entries.sort(key=lambda x: x.timestamp, reverse=True)

        # 限制数量
        if n is not None:
            relevant_entries = relevant_entries[:n]

        # 格式化输出
        context_lines = []
        for entry in relevant_entries:
            prefix = "用户" if entry.source == "user" else "助手"
            time_str = entry.timestamp.strftime("%H:%M")
            context_lines.append(f"[{time_str}] {prefix}: {entry.content}")

        # 添加摘要
        if self.summary_buffer:
            context_lines.append("\n[记忆摘要]:")
            context_lines.extend(self.summary_buffer[-2:])  # 最近两个摘要

        return "\n".join(context_lines)

    def search(self, query: str, top_k: int = 3) -> List[STMEntry]:
        """
        搜索相关记忆

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[STMEntry]: 相关记忆条目
        """
        # 简单关键词匹配
        query_words = set(query.lower().split())
        scored_entries = []

        for entry in self.buffer:
            content_words = set(entry.content.lower().split())
            common_words = query_words.intersection(content_words)

            if common_words:
                # 计算简单匹配分数
                score = len(common_words) / len(query_words)
                scored_entries.append((score, entry))

        # 按分数排序
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        return [entry for _, entry in scored_entries[:top_k]]

    def update_importance(self, entry_id: str, new_importance: float):
        """
        更新记忆重要性

        Args:
            entry_id: 记忆ID
            new_importance: 新的重要性评分
        """
        for entry in self.buffer:
            if entry.id == entry_id:
                entry.importance = max(0.0, min(1.0, new_importance))
                logger.debug(f"更新记忆重要性: {entry_id} -> {new_importance}")

                if self.persistence:
                    self._save_to_disk()
                break

    def clear(self):
        """清空短期记忆"""
        self.buffer.clear()
        self.summary_buffer.clear()
        logger.info("短期记忆已清空")

        if self.persistence:
            self._save_to_disk()

    def _generate_summary(self):
        """生成记忆摘要"""
        if len(self.buffer) < 3:
            return

        # 获取最近的重要记忆
        recent_important = [
            entry for entry in self.buffer[-5:]
            if entry.importance >= 0.6
        ]

        if not recent_important:
            return

        # 生成摘要文本
        summary_parts = []
        for entry in recent_important:
            prefix = "用户提到" if entry.source == "user" else "我回复了"
            summary_parts.append(f"{prefix}: {entry.content[:50]}...")

        summary = "; ".join(summary_parts)
        self.summary_buffer.append(summary)

        # 维持摘要数量
        if len(self.summary_buffer) > 5:
            self.summary_buffer.pop(0)

        self.stats['summaries_generated'] += 1
        logger.debug(f"生成记忆摘要: {summary[:100]}...")

    def _save_to_disk(self):
        """保存到磁盘"""
        try:
            data = {
                'entries': [entry.to_dict() for entry in self.buffer],
                'summaries': self.summary_buffer,
                'stats': self.stats,
                'saved_at': datetime.now().isoformat()
            }

            file_path = self.storage_path / "stm_state.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug("短期记忆已保存到磁盘")
        except Exception as e:
            logger.error(f"保存短期记忆失败: {e}")

    def _load_from_disk(self):
        """从磁盘加载"""
        try:
            file_path = self.storage_path / "stm_state.json"
            if not file_path.exists():
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载条目
            self.buffer = [STMEntry.from_dict(entry) for entry in data.get('entries', [])]

            # 加载摘要
            self.summary_buffer = data.get('summaries', [])

            # 加载统计
            self.stats = data.get('stats', self.stats)

            logger.info(f"从磁盘加载 {len(self.buffer)} 条短期记忆")
        except Exception as e:
            logger.error(f"加载短期记忆失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'current_buffer_size': len(self.buffer),
            'summary_count': len(self.summary_buffer),
            'avg_importance': sum(e.importance for e in self.buffer) / len(self.buffer) if self.buffer else 0
        }