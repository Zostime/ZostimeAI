"""
记忆管理器 - 协调STM和LTM
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from .stm import ShortTermMemory
from .ltm import LongTermMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器"""

    def __init__(self, config_path: str = None):
        """
        初始化记忆管理器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)

        # 初始化子模块
        self.stm = ShortTermMemory(self.config.get('stm', {}))
        self.ltm = LongTermMemory(self.config.get('ltm', {}))

        # 记忆桥接配置
        self.bridge_config = {
            'stm_to_ltm_threshold': 0.7,
            'consolidation_enabled': True,
            'auto_extract_key_facts': True
        }

        # 会话状态
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history = []

        logger.info("记忆管理器初始化完成")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'stm': {
                'capacity': 10,
                'persistence': True,
                'summary_enabled': True
            },
            'ltm': {
                'storage_type': 'vector',
                'similarity_threshold': 0.75,
                'auto_consolidation': True
            }
        }

        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
            except Exception as e:
                logger.error(f"加载记忆配置失败: {e}")

        return default_config

    def process_conversation(self, user_input: str, assistant_response: str) -> Dict[str, Any]:
        """
        处理对话，更新记忆系统

        Args:
            user_input: 用户输入
            assistant_response: 助手回复

        Returns:
            Dict[str, Any]: 处理结果
        """
        result = {
            'stm_updated': False,
            'ltm_updated': False,
            'memories_retrieved': [],
            'session_id': self.session_id
        }

        # 添加到会话历史
        self.conversation_history.append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat()
        })

        # 1. 添加到短期记忆
        user_memory_id = self.stm.add_entry(
            content=user_input,
            source="user",
            importance=self._assess_stm_importance(user_input)
        )

        assistant_memory_id = self.stm.add_entry(
            content=assistant_response,
            source="assistant",
            importance=self._assess_stm_importance(assistant_response)
        )

        result['stm_updated'] = True

        # 2. 评估是否需要存储到长期记忆
        user_importance = self._assess_ltm_importance(user_input)
        assistant_importance = self._assess_ltm_importance(assistant_response)

        if user_importance >= self.bridge_config['stm_to_ltm_threshold']:
            ltm_id = self.ltm.store_memory(
                content=user_input,
                source="user_conversation",
                importance=user_importance,
                metadata={
                    'stm_id': user_memory_id,
                    'session_id': self.session_id,
                    'type': 'user_input'
                }
            )
            result['ltm_updated'] = True
            result['memories_retrieved'].append({
                'type': 'ltm_stored',
                'id': ltm_id,
                'content': user_input[:100] + '...'
            })

        if assistant_importance >= self.bridge_config['stm_to_ltm_threshold']:
            ltm_id = self.ltm.store_memory(
                content=assistant_response,
                source="assistant_conversation",
                importance=assistant_importance,
                metadata={
                    'stm_id': assistant_memory_id,
                    'session_id': self.session_id,
                    'type': 'assistant_response'
                }
            )
            result['ltm_updated'] = True
            result['memories_retrieved'].append({
                'type': 'ltm_stored',
                'id': ltm_id,
                'content': assistant_response[:100] + '...'
            })

        # 3. 自动提取关键事实
        if self.bridge_config['auto_extract_key_facts']:
            self._extract_key_facts(user_input, assistant_response)

        logger.debug(f"对话处理完成: STM={result['stm_updated']}, LTM={result['ltm_updated']}")
        return result

    def get_context(self, query: str = None, include_stm: bool = True,
                    include_ltm: bool = True, top_k: int = 3) -> str:
        """
        获取上下文信息

        Args:
            query: 查询文本
            include_stm: 是否包含短期记忆
            include_ltm: 是否包含长期记忆
            top_k: 返回LTM记忆数量

        Returns:
            str: 格式化上下文
        """
        context_parts = []

        # 1. 添加短期记忆上下文
        if include_stm:
            stm_context = self.stm.get_context(n=5, min_importance=0.4)
            if stm_context:
                context_parts.append("【短期记忆上下文】")
                context_parts.append(stm_context)

        # 2. 添加长期记忆上下文
        if include_ltm and query:
            ltm_memories = self.ltm.retrieve(query, top_k=top_k)
            if ltm_memories:
                context_parts.append("\n【长期记忆参考】")
                for i, memory in enumerate(ltm_memories, 1):
                    memory_preview = memory.content
                    if len(memory_preview) > 100:
                        memory_preview = memory_preview[:100] + "..."

                    context_parts.append(f"{i}. [{memory.category}] {memory_preview}")
                    context_parts.append(f"   重要性: {memory.importance:.2f}, 访问次数: {memory.access_count}")

        # 3. 添加知识图谱信息
        if query and hasattr(self.ltm, 'knowledge_graph'):
            entities = self._extract_entities(query)
            for entity in entities[:2]:  # 限制数量
                kg_results = self.ltm.search_knowledge_graph(entity)
                if kg_results:
                    context_parts.append(f"\n【知识图谱: {entity}】")
                    for result in kg_results[:2]:
                        context_parts.append(f"  提及次数: {result.get('mentions', 0)}")

        return "\n".join(context_parts) if context_parts else ""

    def consolidate_session(self):
        """巩固会话记忆"""
        # 1. 获取当前会话的重要记忆
        recent_stm = [entry for entry in self.stm.buffer[-10:] if entry.importance >= 0.6]

        if len(recent_stm) >= 3:
            # 2. 提取主题
            themes = self._extract_session_themes([entry.content for entry in recent_stm])

            # 3. 生成会话摘要
            summary = self._generate_session_summary(recent_stm, themes)

            # 4. 存储到长期记忆
            ltm_id = self.ltm.store_memory(
                content=summary,
                source="session_summary",
                category="session",
                importance=0.8,
                metadata={
                    'session_id': self.session_id,
                    'themes': themes,
                    'entry_count': len(recent_stm)
                }
            )

            logger.info(f"会话记忆已巩固: {ltm_id}")
            return ltm_id

        return None

    def search_memories(self, query: str, search_type: str = "hybrid") -> Dict[str, Any]:
        """
        搜索记忆

        Args:
            query: 查询文本
            search_type: 搜索类型 (stm, ltm, hybrid)

        Returns:
            Dict[str, Any]: 搜索结果
        """
        results = {
            'query': query,
            'search_type': search_type,
            'stm_results': [],
            'ltm_results': [],
            'knowledge_graph_results': []
        }

        # STM搜索
        if search_type in ["stm", "hybrid"]:
            stm_results = self.stm.search(query, top_k=3)
            results['stm_results'] = [
                {
                    'id': entry.id,
                    'content': entry.content[:150] + ('...' if len(entry.content) > 150 else ''),
                    'importance': entry.importance,
                    'source': entry.source,
                    'timestamp': entry.timestamp.isoformat()
                }
                for entry in stm_results
            ]

        # LTM搜索
        if search_type in ["ltm", "hybrid"]:
            ltm_results = self.ltm.retrieve(query, top_k=5)
            results['ltm_results'] = [
                {
                    'id': memory.id,
                    'content': memory.content[:150] + ('...' if len(memory.content) > 150 else ''),
                    'category': memory.category,
                    'importance': memory.importance,
                    'access_count': memory.access_count,
                    'created_at': memory.created_at.isoformat()
                }
                for memory in ltm_results
            ]

        # 知识图谱搜索
        entities = self._extract_entities(query)
        for entity in entities[:3]:
            kg_results = self.ltm.search_knowledge_graph(entity)
            if kg_results:
                results['knowledge_graph_results'].extend(kg_results)

        return results

    def _assess_stm_importance(self, text: str) -> float:
        """评估短期记忆重要性"""
        # 基础重要性
        importance = 0.5

        # 长度因素
        if len(text) > 50:
            importance += 0.1
        elif len(text) < 10:
            importance -= 0.1

        # 问题类型
        question_words = ['吗', '？', '?', '如何', '为什么', '什么']
        if any(word in text for word in question_words):
            importance += 0.15

        # 情感词
        emotional_words = ['开心', '难过', '生气', '兴奋', '失望']
        if any(word in text for word in emotional_words):
            importance += 0.1

        return min(1.0, max(0.1, importance))

    def _assess_ltm_importance(self, text: str) -> float:
        """评估长期记忆重要性"""
        # 比STM更高的门槛
        importance = self._assess_stm_importance(text)

        # 增加长期重要性评估因素
        long_term_indicators = [
            '记住', '重要', '关键', '总是', '从不',
            '喜欢', '讨厌', '偏好', '习惯'
        ]

        if any(indicator in text for indicator in long_term_indicators):
            importance += 0.25

        # 个人信息
        personal_indicators = ['我', '我的', '自己', '个人']
        if any(indicator in text for indicator in personal_indicators):
            importance += 0.15

        return min(1.0, importance)

    def _extract_key_facts(self, user_input: str, assistant_response: str):
        """提取关键事实"""
        # 这里可以集成NER和关系提取
        # 简化实现：提取看起来像事实的陈述
        facts = []

        # 检查用户输入中的事实
        fact_patterns = ['是', '有', '在', '位于', '包含']
        for pattern in fact_patterns:
            if pattern in user_input:
                # 简单提取包含该词的句子
                sentences = user_input.split('。')
                for sentence in sentences:
                    if pattern in sentence and len(sentence) > 5:
                        facts.append(sentence.strip())

        # 存储提取的事实
        for fact in facts[:2]:  # 限制数量
            self.ltm.store_memory(
                content=fact,
                source="extracted_fact",
                category="fact",
                importance=0.6,
                metadata={
                    'extraction_method': 'pattern_match',
                    'original_input': user_input[:50]
                }
            )

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体（简化版）"""
        # 实际应使用NER模型
        entities = []
        for word in text.split():
            if len(word) > 1 and word[0].isupper():  # 简单的大写词检测
                entities.append(word)
            elif len(word) > 3 and word.isalpha():  # 较长的名词
                entities.append(word)

        return entities[:5]  # 返回前5个

    def _extract_session_themes(self, texts: List[str]) -> List[str]:
        """提取会话主题"""
        from collections import Counter

        # 提取关键词
        keywords = []
        for text in texts:
            words = text.split()
            keywords.extend([w for w in words if len(w) > 1][:10])

        # 统计词频
        word_counts = Counter(keywords)
        common_words = [word for word, count in word_counts.items() if count >= 2]

        return common_words[:3] if common_words else ["综合对话"]

    def _generate_session_summary(self, memories: List, themes: List[str]) -> str:
        """生成会话摘要"""
        summary_parts = [f"会话主题: {'、'.join(themes)}。"]

        for i, memory in enumerate(memories[:5], 1):
            prefix = "用户" if memory.source == "user" else "助手"
            summary_parts.append(f"{i}. {prefix}: {memory.content[:50]}...")

        return " ".join(summary_parts)

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stm_stats = self.stm.get_stats()
        ltm_stats = self.ltm.get_stats()

        return {
            'session_id': self.session_id,
            'conversation_count': len(self.conversation_history),
            'short_term_memory': stm_stats,
            'long_term_memory': ltm_stats,
            'bridge': {
                'stm_to_ltm_threshold': self.bridge_config['stm_to_ltm_threshold'],
                'consolidation_enabled': self.bridge_config['consolidation_enabled']
            }
        }

    def clear_memory(self, memory_type: str = "stm"):
        """
        清除记忆

        Args:
            memory_type: 记忆类型 (stm, ltm, all)
        """
        if memory_type in ["stm", "all"]:
            self.stm.clear()
            logger.info("短期记忆已清除")

        if memory_type in ["ltm", "all"]:
            # 注意：清除LTM可能需要更多确认
            logger.warning("长期记忆清除功能需要额外确认")

        if memory_type == "session":
            self.conversation_history.clear()
            logger.info("会话历史已清除")