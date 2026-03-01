"""
DocMind 评估系统

包含：
- retrieval_metrics: 传统检索指标（Recall@K, MRR, NDCG）
- generation_metrics: 生成质量指标（LLM-as-Judge）
- ragas_evaluator: RAGAS 端到端评估（业界标准）
- ablation_study: 消融实验
- test_set_generator: 测试集生成器
"""

from .retrieval_metrics import RetrievalEvaluator
from .generation_metrics import GenerationEvaluator
from .ablation_study import AblationStudy

# RAGAS 是可选依赖
try:
    from .ragas_evaluator import RAGASEvaluator
    __all__ = [
        "RetrievalEvaluator",
        "GenerationEvaluator",
        "RAGASEvaluator",
        "AblationStudy",
    ]
except ImportError:
    __all__ = [
        "RetrievalEvaluator",
        "GenerationEvaluator",
        "AblationStudy",
    ]
