"""
Embedding 模块

提供文本向量化能力，支持本地和远程两种模式
"""

from .base import BaseEmbedding
from .local_bge import LocalBGEEmbedding
from .remote_bge import RemoteBGEEmbedding, get_embedding_model

__all__ = [
    "BaseEmbedding",
    "LocalBGEEmbedding",
    "RemoteBGEEmbedding",
    "get_embedding_model",
]
