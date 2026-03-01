"""
传统检索指标评估器

指标说明：
- Recall@K: 在 Top-K 结果中是否命中相关文档
- MRR (Mean Reciprocal Rank): 第一个相关文档的排名倒数的均值
- NDCG@K: 归一化折损累积增益，考虑排序质量
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class RetrievalEvaluator:
    """传统检索指标评估器"""

    def __init__(self, vector_store, similarity_threshold: float = 0.65):
        """
        Args:
            vector_store: VectorStore 实例
            similarity_threshold: 判断相关性的相似度阈值
        """
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self._sim_model = None

    def _get_similarity_model(self):
        """延迟加载相似度计算模型"""
        if self._sim_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading similarity model (bge-small-zh-v1.5)...")
                self._sim_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
            except ImportError:
                print("Warning: sentence-transformers not installed, using simple matching")
                self._sim_model = "simple"
        return self._sim_model

    def evaluate(
        self,
        test_set: List[Dict],
        top_k: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        计算精确检索指标

        Args:
            test_set: 测试集，每个元素包含：
                - question: 问题
                - ground_truth_contexts: 标准答案对应的文档片段列表
                - (可选) ground_truth_answer: 标准答案
            top_k: 检索数量
            verbose: 是否显示进度条

        Returns:
            {
                "recall@1": float,
                "recall@3": float,
                "recall@5": float,
                "recall@10": float,
                "mrr": float,
                "ndcg@3": float,
                "ndcg@5": float,
                "avg_rerank_score": float,
                "avg_latency_ms": float,
                "total_samples": int,
                "hit_samples": int,
                "details": [...]  # 每个样本的详细结果
            }
        """
        recalls_at_k = {k: [] for k in [1, 3, 5, 10]}
        mrr_scores = []
        ndcg_at_3 = []
        ndcg_at_5 = []
        rerank_scores = []
        latencies = []
        details = []

        iterator = tqdm(test_set, desc="Evaluating Retrieval") if verbose else test_set

        for item in iterator:
            query = item["question"]
            ground_truth_contexts = item.get("ground_truth_contexts", [])

            # 执行检索（记录耗时）
            start = time.time()
            try:
                results = self.vector_store.search(query, top_k=top_k)
            except Exception as e:
                print(f"Search error for query '{query[:30]}...': {e}")
                continue
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            # 获取检索结果
            retrieved_results = results.get("results", [])

            # 判断相关性
            relevance_labels = []
            hit_position = None

            for idx, result in enumerate(retrieved_results):
                retrieved_content = result.get("content", "")

                # 判断是否匹配 Ground Truth
                is_relevant = self._is_relevant(
                    retrieved_content,
                    ground_truth_contexts
                )
                relevance_labels.append(1 if is_relevant else 0)

                # 记录第一次命中位置
                if is_relevant and hit_position is None:
                    hit_position = idx + 1

                # 收集 Top-1 的 Rerank 分数
                if idx == 0:
                    rerank_scores.append(result.get("rerank_score", 0))

            # 补齐 relevance_labels 到 top_k 长度
            while len(relevance_labels) < top_k:
                relevance_labels.append(0)

            # 计算 Recall@K
            for k in recalls_at_k.keys():
                hit = any(relevance_labels[:k])
                recalls_at_k[k].append(1 if hit else 0)

            # 计算 MRR
            mrr = 1.0 / hit_position if hit_position else 0.0
            mrr_scores.append(mrr)

            # 计算 NDCG
            ndcg_at_3.append(self._ndcg_at_k(relevance_labels, k=3))
            ndcg_at_5.append(self._ndcg_at_k(relevance_labels, k=5))

            # 记录详细结果
            details.append({
                "question": query,
                "hit_position": hit_position,
                "mrr": mrr,
                "relevance_labels": relevance_labels[:5],  # 只保留前5个
                "latency_ms": latency_ms,
                "top1_rerank_score": rerank_scores[-1] if rerank_scores else 0
            })

        # 汇总结果
        hit_samples = sum(1 for d in details if d["hit_position"] is not None)

        return {
            "recall@1": np.mean(recalls_at_k[1]) if recalls_at_k[1] else 0,
            "recall@3": np.mean(recalls_at_k[3]) if recalls_at_k[3] else 0,
            "recall@5": np.mean(recalls_at_k[5]) if recalls_at_k[5] else 0,
            "recall@10": np.mean(recalls_at_k[10]) if recalls_at_k[10] else 0,
            "mrr": np.mean(mrr_scores) if mrr_scores else 0,
            "ndcg@3": np.mean(ndcg_at_3) if ndcg_at_3 else 0,
            "ndcg@5": np.mean(ndcg_at_5) if ndcg_at_5 else 0,
            "avg_rerank_score": np.mean(rerank_scores) if rerank_scores else 0,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "total_samples": len(test_set),
            "hit_samples": hit_samples,
            "hit_rate": hit_samples / len(test_set) if test_set else 0,
            "details": details
        }

    def _is_relevant(
        self,
        retrieved: str,
        ground_truths: List[str],
    ) -> bool:
        """
        判断检索结果是否相关

        使用两种策略：
        1. 关键词匹配（快速）
        2. 语义相似度（精确）
        """
        if not ground_truths:
            return False

        # 策略1：关键词匹配（任一 ground_truth 的关键内容出现在 retrieved 中）
        for gt in ground_truths:
            # 提取关键短语（【强制】【推荐】后的内容）
            import re
            key_phrases = re.findall(r'【[^】]+】([^。]+)', gt)
            for phrase in key_phrases:
                # 取前20个字符作为关键匹配
                key = phrase.strip()[:20]
                if key and key in retrieved:
                    return True

            # 如果 ground_truth 较短，直接检查是否包含
            if len(gt) < 100 and gt[:30] in retrieved:
                return True

        # 策略2：语义相似度
        model = self._get_similarity_model()
        if model == "simple":
            # 简单模式：检查是否有重叠
            for gt in ground_truths:
                overlap = len(set(retrieved) & set(gt)) / max(len(set(gt)), 1)
                if overlap > 0.3:
                    return True
            return False

        try:
            from sentence_transformers import util
            retrieved_emb = model.encode(retrieved, convert_to_tensor=True)

            for gt in ground_truths:
                gt_emb = model.encode(gt, convert_to_tensor=True)
                similarity = util.cos_sim(retrieved_emb, gt_emb).item()

                if similarity >= self.similarity_threshold:
                    return True
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            # Fallback to simple matching
            for gt in ground_truths:
                if gt[:30] in retrieved or retrieved[:30] in gt:
                    return True

        return False

    def _ndcg_at_k(self, relevance_labels: List[int], k: int) -> float:
        """计算 NDCG@K"""
        relevance_labels = relevance_labels[:k]

        if not any(relevance_labels):
            return 0.0

        # DCG
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_labels)])

        # IDCG (理想情况：所有相关文档排在前面)
        ideal_labels = sorted(relevance_labels, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_labels)])

        return dcg / idcg if idcg > 0 else 0.0

    def print_summary(self, results: Dict[str, Any]):
        """打印评估结果摘要"""
        print("\n" + "=" * 60)
        print("📊 检索指标评估结果")
        print("=" * 60)

        print(f"\n📈 召回率 (Recall)")
        print(f"   Recall@1:  {results['recall@1']:.2%}")
        print(f"   Recall@3:  {results['recall@3']:.2%}")
        print(f"   Recall@5:  {results['recall@5']:.2%}")
        print(f"   Recall@10: {results['recall@10']:.2%}")

        print(f"\n📊 排序质量")
        print(f"   MRR:       {results['mrr']:.4f}")
        print(f"   NDCG@3:    {results['ndcg@3']:.4f}")
        print(f"   NDCG@5:    {results['ndcg@5']:.4f}")

        print(f"\n⚡ 性能指标")
        print(f"   平均延迟:   {results['avg_latency_ms']:.0f} ms")
        print(f"   平均Rerank: {results['avg_rerank_score']:.2f}")

        print(f"\n📋 统计")
        print(f"   测试样本:   {results['total_samples']}")
        print(f"   命中样本:   {results['hit_samples']}")
        print(f"   命中率:     {results['hit_rate']:.2%}")

        print("=" * 60)


def quick_evaluate(vector_store, test_set: List[Dict], top_k: int = 5) -> Dict:
    """快速评估函数"""
    evaluator = RetrievalEvaluator(vector_store)
    results = evaluator.evaluate(test_set, top_k=top_k)
    evaluator.print_summary(results)
    return results
