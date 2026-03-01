"""
Embedding 基类/接口

定义 Embedding 的统一接口，方便切换不同实现
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedding(ABC):
    """Embedding 基类"""
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """返回 Embedding 维度"""
        pass
    
    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            **kwargs: 其他参数
            
        Returns:
            numpy 数组，shape 为 (n, embedding_dim)
        """
        pass
    
    def encode_single(self, text: str, **kwargs) -> np.ndarray:
        """编码单个文本，返回一维向量"""
        result = self.encode([text], **kwargs)
        return result[0]
