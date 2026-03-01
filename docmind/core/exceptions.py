"""
自定义异常类
"""


class DocMindException(Exception):
    """DocMind 基础异常"""
    pass


class ConfigurationError(DocMindException):
    """配置错误"""
    pass


class EmbeddingError(DocMindException):
    """Embedding 相关错误"""
    pass


class RerankerError(DocMindException):
    """Reranker 相关错误"""
    pass


class RetrievalError(DocMindException):
    """检索相关错误"""
    pass


class DocumentProcessingError(DocMindException):
    """文档处理错误"""
    pass


class LLMError(DocMindException):
    """LLM 调用错误"""
    pass


class GenerationError(DocMindException):
    """答案生成错误"""
    pass
