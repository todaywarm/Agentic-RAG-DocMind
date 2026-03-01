"""
混合检索器

实现三路混合检索：
1. HyDE 向量检索
2. 语义重写向量检索
3. BM25 关键词检索

以及后续的融合、重排序、过滤流程
"""

import re
from typing import List, Dict, Any, Optional, Set
import jieba

from docmind.core.config import settings
from docmind.embedding import get_embedding_model, BaseEmbedding
from docmind.reranker import get_reranker_model, BaseReranker
from .vector_store import VectorStore
from .bm25_store import BM25Store
from .query_processor import QueryProcessor


class HybridRetriever:
    """
    混合检索器
    
    实现完整的 RAG 检索流程：
    查询处理 → 三路召回 → 融合去重 → Rerank → 过滤
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        bm25_store: BM25Store = None,
        query_processor: QueryProcessor = None,
        embedding_model: BaseEmbedding = None,
        reranker: BaseReranker = None,
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储
            bm25_store: BM25 存储
            query_processor: 查询处理器
            embedding_model: Embedding 模型
            reranker: Reranker 模型
        """
        self.embedding_model = embedding_model or get_embedding_model()
        
        self.vector_store = vector_store or VectorStore(
            embedding_model=self.embedding_model
        )
        self.bm25_store = bm25_store or BM25Store()
        self.query_processor = query_processor or QueryProcessor()
        self.reranker = reranker or get_reranker_model()
        
        # 停用词（用于关键词硬过滤）
        self.stopwords: Set[str] = BM25Store.DEFAULT_STOPWORDS
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        history: List[Dict[str, str]] = None,
        threshold_low: float = None,
        threshold_high: float = None,
    ) -> Dict[str, Any]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            history: 对话历史（用于上下文重写）
            threshold_low: 低置信度阈值
            threshold_high: 高置信度阈值
            
        Returns:
            {
                "results": [{"content": str, "score": float, "confidence": str, ...}, ...],
                "query_info": {...},  # 查询处理信息
                "logs": [...],        # 执行日志
                "direct_answer": str | None,  # 元问题直接回答
            }
        """
        top_k = top_k or settings.retrieval.top_k
        threshold_low = threshold_low if threshold_low is not None else settings.retrieval.threshold_low
        threshold_high = threshold_high if threshold_high is not None else settings.retrieval.threshold_high
        
        logs = []
        logs.append(f"🔎 [Start] Query: '{query}'")
        
        # ========== Step 0: 查询处理 ==========
        # 查询处理返回的是一个结果字典 包括语义重写的query用于向量检索
        # 关键词提取用于BM25全局文本检索
        # 上下文重写，加入了历史对话信息
        # 原问题直接回答
        # 假设性回答用于增强向量检索的匹配程度
       

        query_info = self.query_processor.process(query, history)
        
        # 元问题直接回答
        if query_info.get("direct_answer"):
            logs.append("🟣 [Direct Answer] Bypassed retrieval")
            return {
                "results": [],
                "query_info": query_info,
                "logs": logs,
                "direct_answer": query_info["direct_answer"]
            }
        
        logs.append(f"🔹 [Query Processing]")
        logs.append(f"    Standalone: {query_info['standalone_query']}")
        logs.append(f"    Vector Query: {query_info['vector_query']}")
        logs.append(f"    Keywords: {query_info['keywords']}")
        logs.append(f"    HyDE: {query_info['hyde_doc'][:50]}...")
        
        # ========== Step 1: 三路召回 ==========
        candidate_k = top_k * settings.retrieval.candidate_multiplier
        logs.append(f"🔹 [Step 1] Multi-Path Retrieval (3-Way, candidate_k={candidate_k})")
        
        candidates = {}  # id -> result_dict
        
        # Path A: HyDE 向量检索
        #检索结果列表 [{"id": int, "score": float, "content": str, ...}, ...]
        hyde_results = self.vector_store.search(
            query=query_info["hyde_doc"],
            top_k=candidate_k
        )
        logs.append(f"    Path A (HyDE): {len(hyde_results)} results")
        
        for r in hyde_results:
            candidates[r["id"]] = {
                **r,
                "retrieval_source": "Vector(HyDE)",
                "vector_score": r["score"]
            }
        
        # Path B: 语义重写向量检索
        #检索结果列表 [{"id": int, "score": float, "content": str, ...}, ...]
        semantic_results = self.vector_store.search(
            query=query_info["vector_query"],
            top_k=candidate_k
        )
        logs.append(f"    Path B (Semantic): {len(semantic_results)} results")
        
        for r in semantic_results:
            if r["id"] in candidates:
                candidates[r["id"]]["vector_score"] = max(
                    candidates[r["id"]]["vector_score"], r["score"]
                )
                candidates[r["id"]]["retrieval_source"] += "+Semantic"
            else:
                candidates[r["id"]] = {
                    **r,
                    "retrieval_source": "Vector(Semantic)",
                    "vector_score": r["score"]
                }
        
        # Path C: BM25 检索
        bm25_scores = self.bm25_store.search_multi(
            queries=[query_info["standalone_query"]] + query_info["keywords"],
            top_k=candidate_k
        )
        logs.append(f"    Path C (BM25): {len(bm25_scores)} unique results")
        
        for doc_id, score in bm25_scores.items():
            if doc_id in candidates:
                candidates[doc_id]["bm25_score"] = score
                candidates[doc_id]["retrieval_source"] += "+BM25"
            else:
                content = self.bm25_store.get_document(doc_id)
                if content:
                    candidates[doc_id] = {
                        "id": doc_id,
                        "content": content,
                        "vector_score": 0.0,
                        "bm25_score": score,
                        "retrieval_source": "BM25"
                    }
        
        logs.append(f"🔹 [Step 2] Merged: {len(candidates)} unique candidates")
        
        if not candidates:
            return {
                "results": [],
                "query_info": query_info,
                "logs": logs,
                "direct_answer": None
            }
        
        # ========== Step 2: Rerank ==========
        logs.append(f"🔹 [Step 3] Reranking with BGE-Reranker")
        
        candidate_list = list(candidates.values())
        rerank_results = self.reranker.rerank(
            query=query_info["vector_query"],
            documents=[c["content"] for c in candidate_list]
        )
        
        # 更新分数
        for idx, score in rerank_results:
            candidate_list[idx]["rerank_score"] = score
        
        # 按 rerank 分数排序
        candidate_list.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # ========== Step 3: 关键词硬过滤 ==========
        logs.append(f"🔹 [Step 4] Keyword Hard Filter")
        
        keywords = self._extract_query_terms(query_info["standalone_query"])
        filtered = []
        
        for c in candidate_list:
            content = c["content"]
            rerank_score = c.get("rerank_score", 0)
            
            # 检查关键词匹配
            has_keyword = any(kw in content for kw in keywords)
            
            # 高分豁免
            if has_keyword or rerank_score > settings.retrieval.hard_filter_exempt_score:
                filtered.append(c)
            else:
                logs.append(f"    ❌ Filtered: score={rerank_score:.1f}, no keyword match")
        
        candidate_list = filtered
        
        # ========== Step 4: 三级置信度过滤 ==========
        logs.append(f"🔹 [Step 5] Confidence Filtering (low={threshold_low}, high={threshold_high})")
        
        final_results = []
        for c in candidate_list[:top_k]:
            score = c.get("rerank_score", 0)
            
            if score < threshold_low:
                logs.append(f"    ❌ Discarded: score={score:.2f} < {threshold_low}")
                continue
            elif score < threshold_high:
                c["confidence"] = "low"
                logs.append(f"    ⚠️ Low confidence: score={score:.2f}")
            else:
                c["confidence"] = "high"
            
            final_results.append(c)
        
        if final_results:
            logs.append(f"✅ [Done] Returning {len(final_results)} results")
        else:
            logs.append(f"⚠️ All results filtered out")
        
        return {
            "results": final_results,
            "query_info": query_info,
            "logs": logs,
            "direct_answer": None
        }
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """提取查询关键词（用于硬过滤）"""
        terms = []
        seen = set()
        
        # jieba 分词
        for t in jieba.cut(query):
            t = t.strip()
            if len(t) >= 2 and t not in self.stopwords and t not in seen:
                terms.append(t)
                seen.add(t)
        
        # 英文/代码 token
        for t in re.findall(r"[A-Za-z_][A-Za-z0-9_$.]{1,}", query):
            if t not in seen:
                terms.append(t)
                seen.add(t)
        
        return terms
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
    ) -> int:
        """
        添加文档到检索系统（同时更新向量和 BM25 索引）
        
        Args:
            documents: 文档内容列表
            metadatas: 元数据列表
            
        Returns:
            添加的文档数量
        """
        if not documents:
            return 0
        
        # 添加到向量存储
        count = self.vector_store.add_documents(documents, metadatas)
        
        # 添加到 BM25
        self.bm25_store.add_documents(documents)
        
        return count
    
    def reset(self):
        """重置所有索引"""
        self.vector_store.reset()
        self.bm25_store.reset()
