"""
Reranker 基类/接口

定义 Reranker 的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseReranker(ABC):
    """Reranker 基类"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            **kwargs: 其他参数
            
        Returns:
            排序后的 (文档索引, 分数) 列表，按分数降序
        """
        pass
    
    def predict(self, pairs: List[List[str]]) -> List[float]:
        """
        计算 query-document 对的相关性分数
        
        Args:
            pairs: [[query, doc1], [query, doc2], ...] 格式
            
        Returns:
            分数列表（归一化到 0-10）
        """
        if not pairs:
            return []
        
        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        
        results = self.rerank(query, documents)
        
        # 按原始顺序返回分数
        scores = [0.0] * len(pairs)
        for idx, score in results:
            scores[idx] = score
        
        return scores
