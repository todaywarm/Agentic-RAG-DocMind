"""
向量存储 (Qdrant)

负责向量的存储、索引和检索
"""

import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

from docmind.core.config import settings
from docmind.core.exceptions import RetrievalError
from docmind.embedding import get_embedding_model, BaseEmbedding


class VectorStore:
    """
    向量存储（基于 Qdrant）
    
    职责：
    - 向量的存储和持久化
    - 基于向量的相似度检索
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_dir: str = None,
        embedding_model: BaseEmbedding = None,
    ):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            persist_dir: 持久化目录
            embedding_model: Embedding 模型实例
        """
        self.collection_name = collection_name or settings.retrieval.collection_name
        self.persist_dir = persist_dir or settings.retrieval.persist_dir
        self.embedding_model = embedding_model or get_embedding_model()
        
        # 确保目录存在
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # 初始化 Qdrant 客户端
        print(f"Initializing Qdrant at: {self.persist_dir}")
        self.client = QdrantClient(path=self.persist_dir)
        
        # 创建集合（如果不存在）
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保集合存在"""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created.")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[int] = None,
    ) -> int:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档内容列表
            metadatas: 元数据列表
            ids: 文档 ID 列表（如果不提供，自动生成）
            
        Returns:
            添加的文档数量
        """
        if not documents:
            return 0
        
        # 编码文档
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents)
        
        # 生成 ID
        if ids is None:
            # 获取当前最大 ID
            current_count = self.count()
            ids = list(range(current_count, current_count + len(documents)))
        
        # 准备元数据
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        # 构建 points
        points = []
        for i, (doc, vector, meta) in enumerate(zip(documents, embeddings, metadatas)):
            payload = {"content": doc, **meta}
            points.append(models.PointStruct(
                id=ids[i], 
                vector=vector.tolist(),
                payload=payload
            ))
        
        # 批量插入
        print(f"Upserting {len(points)} points to Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return len(points)
    
    def search(
        self,
        query: str = None,
        query_vector: np.ndarray = None,
        top_k: int = None,
        filter_conditions: Dict = None,
    ) -> List[Dict[str, Any]]:
        """
        向量检索
        
        Args:
            query: 查询文本（会自动编码）
            query_vector: 查询向量（直接使用，优先级高于 query）
            top_k: 返回数量
            filter_conditions: 过滤条件
            
        Returns:
            检索结果列表 [{"id": int, "score": float, "content": str, ...}, ...]
        """
        top_k = top_k or settings.retrieval.top_k
        
        # 获取查询向量
        if query_vector is None:
            if query is None:
                raise RetrievalError("Either query or query_vector must be provided")
            query_vector = self.embedding_model.encode_single(query)
        
        # 构建过滤条件
        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
            )
        
        # 执行搜索
        response = self.client.query_points( #修改为新的api调用方式 注意返回值已经与原始的不同需要不同的处理
            collection_name=self.collection_name,
            query=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            limit=top_k,
            query_filter=qdrant_filter
        )
        results = response.points
        # 格式化结果
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                **{k: v for k, v in hit.payload.items() if k != "content"}
            }
            for hit in results
        ]
    
    def count(self) -> int:
        """返回文档数量"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
    
    def delete_collection(self):
        """删除集合"""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted.")
    
    def reset(self):
        """重置集合（删除后重建）"""
        self.delete_collection()
        self._ensure_collection()
        print(f"Collection '{self.collection_name}' reset.")
    
    def document_exists(self, source: str) -> bool:
        """检查文档是否已存在（通过 source 字段）"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                #用于精确匹配metat data中字典key为‘source’，值为传入的source
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            ),
            limit=1
        )[0]
        return len(results) > 0
