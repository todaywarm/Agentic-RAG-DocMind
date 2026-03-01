"""
远程 BGE Embedding (SiliconFlow API)

通过 API 调用，无需下载模型文件
"""

from typing import List, Union
import numpy as np
import requests

from .base import BaseEmbedding
from docmind.core.config import settings
from docmind.core.exceptions import EmbeddingError


class RemoteBGEEmbedding(BaseEmbedding):
    """
    远程 BGE-M3 Embedding (SiliconFlow API)
    
    通过 REST API 调用，适合：
    - 不想下载大模型文件
    - 资源受限的环境
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
        timeout: int = 30,
    ):
        """
        初始化远程 BGE Embedding
        
        Args:
            api_key: API 密钥，默认从配置读取
            base_url: API 地址，默认从配置读取
            model_name: 模型名称，默认从配置读取
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or settings.embedding.api_key
        self.base_url = base_url or settings.embedding.base_url
        self.model_name = model_name or settings.embedding.model_name
        self.timeout = timeout
        self._embedding_dim = settings.embedding.embedding_dim
        
        if not self.api_key:
            print("⚠️ Warning: RAG_API_KEY not set. Remote embedding will fail.")
            print("   Get free key at: https://cloud.siliconflow.cn/")
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def encode(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        通过 API 编码文本为向量
        
        Args:
            texts: 文本或文本列表
            **kwargs: 保留参数（兼容接口）
            
        Returns:
            numpy 数组，shape 为 (n, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.api_key:
            # 返回零向量（防止崩溃，但搜索结果会很差）
            return np.zeros((len(texts), self._embedding_dim))
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # OpenAI 格式: data['data'][i]['embedding']
            embeddings = [item['embedding'] for item in data['data']]
            return np.array(embeddings)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"⚠️ API quota exceeded or invalid key")
            raise EmbeddingError(f"Remote embedding API error: {e}")
        except Exception as e:
            raise EmbeddingError(f"Remote embedding failed: {e}")


def get_embedding_model(use_remote: bool = None) -> BaseEmbedding:
    """
    工厂函数：获取 Embedding 模型实例
    
    Args:
        use_remote: 是否使用远程 API，默认从配置读取
        
    Returns:
        BaseEmbedding 实例
    """
    if use_remote is None:
        use_remote = settings.use_remote_api
    
    if use_remote:
        return RemoteBGEEmbedding()
    else:
        from .local_bge import LocalBGEEmbedding
        return LocalBGEEmbedding()
