"""
Reranker 模块

提供文档重排序能力，支持本地和远程两种模式
"""

from .base import BaseReranker
from .local_bge import LocalBGEReranker
from .remote_bge import RemoteBGEReranker, get_reranker_model

__all__ = [
    "BaseReranker",
    "LocalBGEReranker",
    "RemoteBGEReranker",
    "get_reranker_model",
]
