"""
本地 BGE Reranker

使用 FlagEmbedding 库的 Cross-Encoder
"""

from typing import List, Tuple
import numpy as np

from .base import BaseReranker
from docmind.core.config import settings
from docmind.core.exceptions import RerankerError


class LocalBGEReranker(BaseReranker):
    """
    本地 BGE-Reranker-v2-m3
    
    使用 Cross-Encoder 进行精细化重排序
    """
    
    def __init__(
        self,
        model_name: str = None,
        use_fp16: bool = True,
        score_bias: float = None,
    ):
        """
        初始化本地 Reranker
        
        Args:
            model_name: 模型名称
            use_fp16: 是否使用 FP16
            score_bias: Sigmoid 偏差校准参数
        """
        self.model_name = model_name or settings.reranker.local_model_name
        self.use_fp16 = use_fp16
        self.score_bias = score_bias if score_bias is not None else settings.reranker.score_bias
        self._model = None
    
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                print(f"Loading local BGE Reranker: {self.model_name}...")
                self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
                print("Reranker loaded successfully.")
            except ImportError:
                raise RerankerError(
                    "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
                )
            except Exception as e:
                raise RerankerError(f"Failed to load Reranker: {e}")
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        重排序文档
        
        Returns:
            [(doc_idx, normalized_score), ...] 按分数降序
        """
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        
        try:
            raw_scores = self.model.compute_score(pairs)
            
            # 单个结果时返回 float
            if isinstance(raw_scores, float):
                raw_scores = [raw_scores]
            
            # Sigmoid 偏差校准归一化
            normalized_scores = self._normalize_scores(raw_scores)
            
            # 构建结果并排序
            results = list(enumerate(normalized_scores))
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception as e:
            raise RerankerError(f"Reranking failed: {e}")
    
    def _normalize_scores(self, raw_scores: List[float]) -> List[float]:
        """
        Sigmoid 偏差校准
        
        BGE logits 通常在 [-10, 10] 范围，很多相关文档的分数是负数
        使用偏移后的 Sigmoid 将其映射到 [0, 10]
        
        例如：raw=-2.2, bias=4.0 -> shifted=1.8 -> sigmoid(1.8)≈0.85 -> score=8.5
        """
        normalized = []
        for score in raw_scores:
            shifted = score + self.score_bias
            sigmoid = 1 / (1 + np.exp(-shifted))
            normalized.append(sigmoid * 10)
        return normalized
