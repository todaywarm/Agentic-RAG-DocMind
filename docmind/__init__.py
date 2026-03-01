"""
DocMind - 企业级 RAG 智能问答系统

核心特性：
- 三路混合检索（HyDE + 语义重写 + BM25）
- BGE-M3 Embedding + BGE-Reranker-v2-m3
- 三级置信度过滤（抗幻觉）
- 完整评估体系（Recall/MRR/NDCG/Faithfulness）
"""

__version__ = "2.0.0"
__author__ = "DocMind Team"

from docmind.core.config import settings
