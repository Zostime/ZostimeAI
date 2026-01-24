"""
长期记忆 (Long-Term Memory) 管理模块
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class LTMMemory:
    """长期记忆条目"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    source: str = "conversation"
    category: str = "general"
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = None
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    relationships: List[str] = None

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.relationships is None:
            self.relationships = []

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'id': self.id,
            'content': self.content,
            'source': self.source,
            'category': self.category,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'relationships': self.relationships
        }
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LTMMemory':
        embedding = None
        if 'embedding' in data:
            embedding = np.array(data['embedding'])

        return cls(
            id=data['id'],
            content=data['content'],
            embedding=embedding,
            source=data.get('source', 'conversation'),
            category=data.get('category', 'general'),
            importance=data.get('importance', 0.5),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {}),
            relationships=data.get('relationships', [])
        )


class LongTermMemory:
    """长期记忆管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化长期记忆

        Args:
            config: 配置字典
        """
        self.config = config
        self.storage_type = config.get('storage_type', 'vector')
        self.similarity_threshold = config.get('similarity_threshold', 0.75)
        self.retrieval_top_k = config.get('retrieval_top_k', 3)
        self.auto_consolidation = config.get('auto_consolidation', True)

        # 嵌入模型
        embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # 存储结构
        self.memories: Dict[str, LTMMemory] = {}
        self.categories: Dict[str, List[str]] = {}

        # 向量索引（简单实现）
        self.embeddings: List[np.ndarray] = []
        self.memory_ids: List[str] = []

        # 知识图谱
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}

        # 存储路径
        self.storage_path = Path("Files/storage/long_term")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 加载现有记忆
        self._load_memories()

        logger.info(f"长期记忆初始化完成，存储类型: {self.storage_type}")

    def store_memory(self, content: str, source: str = "conversation",
                     category: str = None, importance: float = None,
                     metadata: Dict[str, Any] = None) -> str:
        """
        存储长期记忆

        Args:
            content: 记忆内容
            source: 来源
            category: 分类
            importance: 重要性评分
            metadata: 元数据

        Returns:
            str: 记忆ID
        """
        # 生成唯一ID
        memory_id = str(uuid.uuid4())

        # 自动分类
        if category is None:
            category = self._classify_content(content)

        # 自动评估重要性
        if importance is None:
            importance = self._assess_importance(content)

        # 生成嵌入向量
        embedding = self._generate_embedding(content)

        # 创建记忆对象
        memory = LTMMemory(
            id=memory_id,
            content=content,
            embedding=embedding,
            source=source,
            category=category,
            importance=importance,
            metadata=metadata or {}
        )

        # 存储到内存
        self.memories[memory_id] = memory

        # 添加到向量索引
        if embedding is not None:
            self.embeddings.append(embedding)
            self.memory_ids.append(memory_id)

        # 更新分类索引
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(memory_id)

        # 提取知识图谱信息
        if self.config.get('knowledge_graph_enabled', False):
            self._extract_knowledge_graph(memory)

        # 自动巩固
        if self.auto_consolidation:
            self._auto_consolidate()

        # 持久化
        self._save_memories()

        logger.info(f"存储长期记忆: {memory_id}, 分类: {category}, 重要性: {importance}")
        return memory_id

    def retrieve(self, query: str, category: str = None,
                 top_k: int = None, min_similarity: float = None) -> List[LTMMemory]:
        """
        检索相关记忆

        Args:
            query: 查询文本
            category: 指定分类
            top_k: 返回数量
            min_similarity: 最小相似度阈值

        Returns:
            List[LTMMemory]: 相关记忆
        """
        if not self.memories:
            return []

        if top_k is None:
            top_k = self.retrieval_top_k

        if min_similarity is None:
            min_similarity = self.similarity_threshold

        # 生成查询嵌入
        query_embedding = self._generate_embedding(query)

        # 计算相似度
        similarities = []
        for memory_id, memory in self.memories.items():
            # 分类过滤
            if category and memory.category != category:
                continue

            if memory.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)

                if similarity >= min_similarity:
                    # 考虑重要性和访问频率
                    boost = memory.importance * 0.3 + (memory.access_count * 0.01)
                    final_score = similarity + boost
                    similarities.append((final_score, memory_id))

        # 排序
        similarities.sort(key=lambda x: x[0], reverse=True)

        # 获取记忆并更新访问统计
        results = []
        for _, memory_id in similarities[:top_k]:
            memory = self.memories[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)

        # 保存更新
        self._save_memories()

        return results

    def consolidate_memories(self, memory_ids: List[str]) -> Optional[str]:
        """
        巩固多个记忆

        Args:
            memory_ids: 要巩固的记忆ID列表

        Returns:
            Optional[str]: 新生成的综合记忆ID
        """
        if len(memory_ids) < 2:
            return None

        # 获取记忆内容
        memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]

        if len(memories) < 2:
            return None

        # 提取共同主题
        common_keywords = self._extract_common_keywords([m.content for m in memories])

        # 生成综合记忆
        consolidated_content = f"综合记忆：关于{common_keywords}。"
        for i, memory in enumerate(memories, 1):
            consolidated_content += f" 记忆{i}: {memory.content}"

        # 计算平均重要性
        avg_importance = sum(m.importance for m in memories) / len(memories)

        # 存储新记忆
        new_memory_id = self.store_memory(
            content=consolidated_content,
            source="consolidation",
            category=memories[0].category,
            importance=avg_importance * 1.1,  # 稍微提高重要性
            metadata={
                'consolidated_from': memory_ids,
                'common_keywords': common_keywords
            }
        )

        # 标记原记忆已被巩固
        for memory in memories:
            memory.metadata['consolidated_into'] = new_memory_id
            memory.importance *= 0.8  # 降低原记忆重要性

        logger.info(f"巩固 {len(memory_ids)} 条记忆为: {new_memory_id}")
        return new_memory_id

    def search_knowledge_graph(self, entity: str, relationship: str = None) -> List[Dict[str, Any]]:
        """
        在知识图谱中搜索

        Args:
            entity: 实体名称
            relationship: 关系类型

        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        results = []

        if entity in self.knowledge_graph:
            entity_info = self.knowledge_graph[entity]

            if relationship:
                if relationship in entity_info['relationships']:
                    results.append({
                        'entity': entity,
                        'relationship': relationship,
                        'targets': entity_info['relationships'][relationship],
                        'source_memories': entity_info['source_memories']
                    })
            else:
                results.append(entity_info)

        return results

    def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入向量"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return np.zeros(384)  # 默认维度

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _classify_content(self, content: str) -> str:
        """自动分类内容"""
        content_lower = content.lower()

        # 简单关键词分类
        categories = {
            'personal': ['我', '我的', '喜欢', '讨厌', '感觉', '希望'],
            'fact': ['是', '有', '在', '位于', '包含', '数据'],
            'skill': ['如何', '怎样', '步骤', '方法', '技巧'],
            'event': ['昨天', '今天', '明天', '发生', '事件', '会议'],
            'preference': ['喜欢', '不喜欢', '偏爱', '讨厌', '最爱']
        }

        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category

        return 'general'

    def _assess_importance(self, content: str) -> float:
        """评估记忆重要性"""
        importance = 0.5  # 基础重要性

        # 长度因素
        length = len(content)
        if length > 100:
            importance += 0.1
        elif length < 20:
            importance -= 0.1

        # 关键词因素
        important_keywords = ['重要', '记住', '关键', '必须', '总是', '从不']
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        # 情感因素（简单检测）
        emotional_words = ['高兴', '生气', '伤心', '兴奋', '失望']
        if any(word in content for word in emotional_words):
            importance += 0.1

        return min(1.0, max(0.1, importance))

    def _extract_knowledge_graph(self, memory: LTMMemory):
        """从记忆中提取知识图谱信息"""
        # 简单的实体提取（实际应使用NER）
        content = memory.content

        # 提取可能的实体（这里简化处理）
        entities = []
        for word in content.split():
            if len(word) > 1 and word[0].isupper():  # 简单的大写词检测
                entities.append(word)

        for entity in entities[:5]:  # 限制数量
            if entity not in self.knowledge_graph:
                self.knowledge_graph[entity] = {
                    'entity': entity,
                    'mentions': 1,
                    'source_memories': [memory.id],
                    'relationships': {}
                }
            else:
                self.knowledge_graph[entity]['mentions'] += 1
                self.knowledge_graph[entity]['source_memories'].append(memory.id)

    def _extract_common_keywords(self, texts: List[str]) -> str:
        """提取共同关键词"""
        from collections import Counter

        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words[:10])  # 只考虑前10个词

        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.items() if count >= 2]

        return "、".join(common_words[:3]) if common_words else "多个主题"

    def _auto_consolidate(self):
        """自动巩固记忆"""
        # 每存储10条记忆尝试巩固一次
        if len(self.memories) % 10 != 0:
            return

        # 按分类分组
        memories_by_category = {}
        for memory in self.memories.values():
            if memory.category not in memories_by_category:
                memories_by_category[memory.category] = []
            memories_by_category[memory.category].append(memory)

        # 对每个分类尝试巩固
        for category, memories in memories_by_category.items():
            if len(memories) >= 3:
                # 选择重要性较低的记忆进行巩固
                low_importance = [m for m in memories if m.importance < 0.4]
                if len(low_importance) >= 2:
                    memory_ids = [m.id for m in low_importance[:3]]
                    self.consolidate_memories(memory_ids)

    def _save_memories(self):
        """保存记忆到磁盘"""
        try:
            # 保存记忆数据
            memories_data = {
                'memories': [memory.to_dict() for memory in self.memories.values()],
                'categories': self.categories,
                'knowledge_graph': self.knowledge_graph,
                'saved_at': datetime.now().isoformat()
            }

            file_path = self.storage_path / "ltm_data.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"保存 {len(self.memories)} 条长期记忆到磁盘")
        except Exception as e:
            logger.error(f"保存长期记忆失败: {e}")

    def _load_memories(self):
        """从磁盘加载记忆"""
        try:
            file_path = self.storage_path / "ltm_data.json"
            if not file_path.exists():
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载记忆
            for memory_data in data.get('memories', []):
                memory = LTMMemory.from_dict(memory_data)
                self.memories[memory.id] = memory

            # 加载分类
            self.categories = data.get('categories', {})

            # 加载知识图谱
            self.knowledge_graph = data.get('knowledge_graph', {})

            # 重建向量索引
            for memory_id, memory in self.memories.items():
                if memory.embedding is not None:
                    self.embeddings.append(memory.embedding)
                    self.memory_ids.append(memory_id)

            logger.info(f"从磁盘加载 {len(self.memories)} 条长期记忆")
        except Exception as e:
            logger.error(f"加载长期记忆失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_importance = sum(m.importance for m in self.memories.values())
        total_access = sum(m.access_count for m in self.memories.values())

        return {
            'total_memories': len(self.memories),
            'categories_count': len(self.categories),
            'knowledge_graph_entities': len(self.knowledge_graph),
            'avg_importance': total_importance / len(self.memories) if self.memories else 0,
            'avg_access_count': total_access / len(self.memories) if self.memories else 0,
            'recent_memories': len([m for m in self.memories.values()
                                    if (datetime.now() - m.created_at).days < 7])
        }