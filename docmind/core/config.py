"""
全局配置管理

从环境变量 / .env 文件加载配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class LLMConfig:
    """LLM 配置"""
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    model_id: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ID", "deepseek-chat"))
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    api_key: str = field(default_factory=lambda: os.getenv("RAG_API_KEY") or os.getenv("SILICONFLOW_API_KEY", ""))
    base_url: str = "https://api.siliconflow.cn/v1/embeddings"
    model_name: str = "BAAI/bge-m3"
    local_model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    use_remote: bool = True


@dataclass
class RerankerConfig:
    """Reranker 配置"""
    api_key: str = field(default_factory=lambda: os.getenv("RAG_API_KEY") or os.getenv("SILICONFLOW_API_KEY", ""))
    base_url: str = "https://api.siliconflow.cn/v1/rerank"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    local_model_name: str = "BAAI/bge-reranker-v2-m3"
    use_remote: bool = True
    # Sigmoid 偏差校准参数
    score_bias: float = 4.0


@dataclass
class RetrievalConfig:
    """检索配置"""
    # Qdrant 配置
    collection_name: str = "docmind_knowledge_base"
    persist_dir: str = "./data/qdrant_db"
    
    # 检索参数
    top_k: int = 5
    candidate_multiplier: int = 3  # 候选数 = top_k * multiplier
    
    # 三级阈值过滤
    threshold_low: float = 4.0
    threshold_high: float = 6.0
    
    # 硬过滤：高分豁免阈值
    hard_filter_exempt_score: float = 8.5


@dataclass
class Settings:
    """全局配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # 是否使用远程 API（统一开关）
    use_remote_api: bool = field(default_factory=lambda: bool(
        os.getenv("RAG_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    ))
    
    def __post_init__(self):
        # 同步远程 API 开关
        self.embedding.use_remote = self.use_remote_api
        self.reranker.use_remote = self.use_remote_api


# 全局单例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置"""
    return settings


def reload_settings():
    """重新加载配置（用于测试）"""
    global settings
    load_dotenv(override=True)
    settings = Settings()
    return settings
