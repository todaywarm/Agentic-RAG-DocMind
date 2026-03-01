"""
BM25 存储

负责 BM25 索引的构建、持久化和检索
"""

import os
import pickle
from typing import List, Dict, Any, Set, Optional
import jieba
import logging
from rank_bm25 import BM25Okapi

from docmind.core.config import settings
from docmind.core.exceptions import RetrievalError

# 降低 jieba 日志级别
jieba.setLogLevel(logging.WARNING)


class BM25Store:
    """
    BM25 存储
    
    职责：
    - BM25 索引的构建和维护
    - 基于关键词的稀疏检索
    - 索引的持久化
    """
    
    # 中文停用词表
    DEFAULT_STOPWORDS: Set[str] = {
        # 疑问词
        "如何", "怎么", "怎样", "什么", "为什么", "哪些", "哪个",
        "是否", "能否", "可否", "是不是",
        # 助词
        "的", "了", "吗", "呢", "啊", "吧", "呀", "着", "过",
        # 代词
        "我", "你", "他", "它", "这", "那", "这个", "那个",
        "我们", "你们", "他们", "这里", "那里",
        # 连词
        "和", "与", "或", "以及", "还有", "但是", "而且",
        # 介词
        "在", "从", "到", "对", "把", "被", "给", "关于", "对于",
        # 副词
        "都", "也", "就", "才", "很", "太", "更", "最", "非常", "比较",
        # 动词
        "是", "有", "做", "用", "要", "会", "能", "可以",
        "想", "知道", "了解", "告诉", "说说", "介绍", "讲", "一下",
    }
    
    def __init__(
        self,
        persist_path: str = None,
        stopwords: Set[str] = None,
    ):
        """
        初始化 BM25 存储
        
        Args:
            persist_path: 持久化文件路径
            stopwords: 停用词集合
        """
        self.persist_path = persist_path or os.path.join(
            settings.retrieval.persist_dir, "bm25_index.pkl"
        )
        self.stopwords = stopwords or self.DEFAULT_STOPWORDS
        
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []  # 原始文档
        self.tokenized_corpus: List[List[str]] = []  # 分词后的文档
        
        # 尝试加载已有索引
        self._load()
    
    def _tokenize(self, text: str) -> List[str]:
        """分词并去除停用词"""
        tokens = jieba.cut(text)
        return [t for t in tokens if t.strip() and t not in self.stopwords]
    
    def add_documents(self, documents: List[str]) -> int:
        """
        添加文档到 BM25 索引
        
        Args:
            documents: 文档列表
            
        Returns:
            添加的文档数量
        """
        if not documents:
            return 0
        
        # 添加到语料库
        self.corpus.extend(documents)
        
        # 分词
        new_tokenized = [self._tokenize(doc) for doc in documents]
        self.tokenized_corpus.extend(new_tokenized)
        
        # 重建 BM25 索引
        print(f"Building BM25 index with {len(self.corpus)} documents...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 持久化
        self._save()
        
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            检索结果 [{"id": int, "score": float, "content": str}, ...]
        """
        top_k = top_k or settings.retrieval.top_k
        
        if self.bm25 is None or not self.corpus:
            return []
        
        # 分词查询
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        
        # 计算分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 Top-K
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的
                results.append({
                    "id": int(idx),
                    "score": float(scores[idx]),
                    "content": self.corpus[idx]
                })
        
        return results
    
    def search_multi(
        self,
        queries: List[str],
        top_k: int = None,
    ) -> Dict[int, float]:
        """
        多查询 BM25 检索（用于扩展关键词）
        
        Args:
            queries: 查询列表
            top_k: 每个查询的返回数量
            
        Returns:
            {doc_id: max_score} 合并后的结果
        """
        top_k = top_k or settings.retrieval.top_k
        
        merged_scores: Dict[int, float] = {}
        
        for query in queries:
            results = self.search(query, top_k=top_k)
            for r in results:
                idx = r["id"]
                score = r["score"]
                # 取最大分数
                if idx not in merged_scores or score > merged_scores[idx]:
                    merged_scores[idx] = score
        
        return merged_scores
    
    def get_document(self, doc_id: int) -> Optional[str]:
        """根据 ID 获取文档内容"""
        if 0 <= doc_id < len(self.corpus):
            return self.corpus[doc_id]
        return None
    
    def count(self) -> int:
        """返回文档数量"""
        return len(self.corpus)
    
    def reset(self):
        """重置索引"""
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
        
        # 删除持久化文件
        if os.path.exists(self.persist_path):
            os.remove(self.persist_path)
        
        print("BM25 index reset.")
    
    def _save(self):
        """持久化索引"""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        data = {
            "bm25": self.bm25,
            "corpus": self.corpus,
            "tokenized_corpus": self.tokenized_corpus
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"BM25 index saved to {self.persist_path}")
    
    def _load(self):
        """加载持久化的索引"""
        if not os.path.exists(self.persist_path):
            return
        
        try:
            print(f"Loading BM25 index from {self.persist_path}...")
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
            
            self.bm25 = data.get("bm25")
            self.corpus = data.get("corpus", [])
            self.tokenized_corpus = data.get("tokenized_corpus", [])
            
            print(f"Loaded {len(self.corpus)} documents into BM25.")
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            self.bm25 = None
            self.corpus = []
            self.tokenized_corpus = []
