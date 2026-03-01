"""
本地 BGE-M3 Embedding

使用 FlagEmbedding 库加载本地模型
"""

from typing import List, Union
import numpy as np

from .base import BaseEmbedding
from docmind.core.config import settings
from docmind.core.exceptions import EmbeddingError


class LocalBGEEmbedding(BaseEmbedding):
    """
    本地 BGE-M3 Embedding
    
    使用 BAAI/bge-m3 模型，支持中英文
    """
    
    def __init__(
        self,
        model_name: str = None,
        use_fp16: bool = True,
    ):
        """
        初始化本地 BGE Embedding
        
        Args:
            model_name: 模型名称，默认从配置读取
            use_fp16: 是否使用 FP16 加速
        """
        self.model_name = model_name or settings.embedding.local_model_name
        self.use_fp16 = use_fp16
        self._model = None
        self._embedding_dim = settings.embedding.embedding_dim
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
                print(f"Loading local BGE-M3 model: {self.model_name}...")
                self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
                print("Model loaded successfully.")
            except ImportError:
                raise EmbeddingError(
                    "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load BGE model: {e}")
        return self._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本或文本列表
            **kwargs: 传递给模型的其他参数
            
        Returns:
            numpy 数组，shape 为 (n, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            output = self.model.encode(
                texts,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                **kwargs
            )
            return output['dense_vecs']
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}")
