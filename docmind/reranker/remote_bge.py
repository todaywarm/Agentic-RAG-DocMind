"""
远程 BGE Reranker (SiliconFlow API)

通过 API 调用，无需下载模型文件
"""

from typing import List, Tuple
import numpy as np
import requests

from .base import BaseReranker
from docmind.core.config import settings
from docmind.core.exceptions import RerankerError


class RemoteBGEReranker(BaseReranker):
    """
    远程 BGE-Reranker (SiliconFlow API)
    
    通过 REST API 调用
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
        timeout: int = 30,
        score_bias: float = None,
    ):
        """
        初始化远程 Reranker
        
        Args:
            api_key: API 密钥
            base_url: API 地址
            model_name: 模型名称
            timeout: 超时时间
            score_bias: Sigmoid 偏差校准参数
        """
        self.api_key = api_key or settings.reranker.api_key
        self.base_url = base_url or settings.reranker.base_url
        self.model_name = model_name or settings.reranker.model_name
        self.timeout = timeout
        self.score_bias = score_bias if score_bias is not None else settings.reranker.score_bias
        
        if not self.api_key:
            print("⚠️ Warning: RAG_API_KEY not set. Remote reranker will fail.")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        通过 API 重排序文档
        
        Returns:
            [(doc_idx, normalized_score), ...] 按分数降序
        """
        if not documents:
            return []
        
        if not self.api_key:
            # 返回原始顺序，分数为0
            return [(i, 0.0) for i in range(len(documents))]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "return_documents": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            results = response.json()['results']
            
            # API 返回: [{"index": 0, "relevance_score": 0.8}, ...]
            # 应用 Sigmoid 偏差校准
            reranked = []
            for item in results:
                idx = item['index']
                raw_score = item['relevance_score']
                
                # 归一化
                shifted = raw_score + self.score_bias
                sigmoid = 1 / (1 + np.exp(-shifted))
                norm_score = sigmoid * 10
                
                reranked.append((idx, norm_score))
            
            # 按分数降序排序
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"⚠️ Reranker API quota exceeded or invalid key")
            raise RerankerError(f"Remote reranker API error: {e}")
        except Exception as e:
            raise RerankerError(f"Remote reranking failed: {e}")


def get_reranker_model(use_remote: bool = None) -> BaseReranker:
    """
    工厂函数：获取 Reranker 模型实例
    
    Args:
        use_remote: 是否使用远程 API
        
    Returns:
        BaseReranker 实例
    """
    if use_remote is None:
        use_remote = settings.use_remote_api
    
    if use_remote:
        return RemoteBGEReranker()
    else:
        from .local_bge import LocalBGEReranker
        return LocalBGEReranker()
