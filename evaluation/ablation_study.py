"""
消融实验模块

用于验证各个模块的增益贡献：
- Baseline: 仅 Dense Vector 检索
- + BM25: 混合检索
- + Query Rewrite: 查询重写
- + HyDE: 假设文档嵌入
- Full Pipeline: 完整流程 (+ Reranker)
"""

import os
import sys
import time
import copy
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    use_hyde: bool = True
    use_rewrite: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    use_hard_filter: bool = True


class AblationStudy:
    """消融实验：验证各个模块的增益"""

    # 预定义的实验配置
    EXPERIMENTS = [
        ExperimentConfig(
            name="baseline",
            use_hyde=False,
            use_rewrite=False,
            use_bm25=False,
            use_rerank=False,
            use_hard_filter=False
        ),
        ExperimentConfig(
            name="+ bm25",
            use_hyde=False,
            use_rewrite=False,
            use_bm25=True,
            use_rerank=False,
            use_hard_filter=False
        ),
        ExperimentConfig(
            name="+ rewrite",
            use_hyde=False,
            use_rewrite=True,
            use_bm25=True,
            use_rerank=False,
            use_hard_filter=False
        ),
        ExperimentConfig(
            name="+ hyde",
            use_hyde=True,
            use_rewrite=True,
            use_bm25=True,
            use_rerank=False,
            use_hard_filter=False
        ),
        ExperimentConfig(
            name="full",
            use_hyde=True,
            use_rewrite=True,
            use_bm25=True,
            use_rerank=True,
            use_hard_filter=True
        ),
    ]

    def __init__(self, vector_store):
        """
        Args:
            vector_store: VectorStore 实例
        """
        self.vector_store = vector_store
        self.results = {}

    def run_experiments(
        self,
        test_set: List[Dict],
        top_k: int = 5,
        experiments: Optional[List[ExperimentConfig]] = None
    ) -> Dict[str, Dict]:
        """
        运行消融实验

        Args:
            test_set: 测试集
            top_k: 检索数量
            experiments: 自定义实验配置列表，默认使用预定义配置

        Returns:
            {
                "baseline": {"recall@3": 0.72, "mrr": 0.65, ...},
                "+ bm25": {...},
                ...
            }
        """
        if experiments is None:
            experiments = self.EXPERIMENTS

        self.results = {}

        for config in experiments:
            print(f"\n{'='*60}")
            print(f"🧪 Running Experiment: [{config.name}]")
            print(f"   Config: hyde={config.use_hyde}, rewrite={config.use_rewrite}, "
                  f"bm25={config.use_bm25}, rerank={config.use_rerank}")
            print(f"{'='*60}")

            metrics = self._eval_with_config(test_set, config, top_k)
            self.results[config.name] = metrics

            print(f"   ✅ Recall@3: {metrics['recall@3']:.2%}, "
                  f"MRR: {metrics['mrr']:.4f}, "
                  f"Latency: {metrics['avg_latency_ms']:.0f}ms")

        # 打印对比表格
        self._print_comparison_table()

        return self.results

    def _eval_with_config(
        self,
        test_set: List[Dict],
        config: ExperimentConfig,
        top_k: int
    ) -> Dict[str, float]:
        """使用特定配置运行评估"""

        recalls_at_k = {k: [] for k in [1, 3, 5]}
        mrr_scores = []
        latencies = []

        for item in test_set:
            query = item["question"]
            ground_truth_contexts = item.get("ground_truth_contexts", [])

            start = time.time()

            # 根据配置执行不同的检索流程
            results = self._search_with_config(query, config, top_k)

            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            # 计算指标
            retrieved_results = results.get("results", [])
            hit_position = None

            for idx, result in enumerate(retrieved_results):
                content = result.get("content", "")
                if self._is_hit(content, ground_truth_contexts):
                    hit_position = idx + 1
                    break

            # Recall@K
            for k in recalls_at_k.keys():
                hit = hit_position is not None and hit_position <= k
                recalls_at_k[k].append(1 if hit else 0)

            # MRR
            mrr = 1.0 / hit_position if hit_position else 0.0
            mrr_scores.append(mrr)

        return {
            "recall@1": np.mean(recalls_at_k[1]),
            "recall@3": np.mean(recalls_at_k[3]),
            "recall@5": np.mean(recalls_at_k[5]),
            "mrr": np.mean(mrr_scores),
            "avg_latency_ms": np.mean(latencies),
            "config": {
                "use_hyde": config.use_hyde,
                "use_rewrite": config.use_rewrite,
                "use_bm25": config.use_bm25,
                "use_rerank": config.use_rerank,
            }
        }

    def _search_with_config(
        self,
        query: str,
        config: ExperimentConfig,
        top_k: int
    ) -> Dict[str, Any]:
        """
        根据配置执行检索

        通过临时修改 search 行为来实现不同配置
        """
        # 保存原始的 query_processor 方法
        original_hyde = self.vector_store.query_processor.generate_hyde_doc
        original_rewrite = self.vector_store.query_processor.rewrite_query

        try:
            # 根据配置禁用某些功能
            if not config.use_hyde:
                # 禁用 HyDE：返回原 query
                self.vector_store.query_processor.generate_hyde_doc = lambda q: q

            if not config.use_rewrite:
                # 禁用 Query Rewrite：返回原 query
                self.vector_store.query_processor.rewrite_query = lambda q: {
                    "vector_query": q,
                    "keywords": [q]
                }

            # 执行检索
            # 注意：BM25 和 Rerank 的控制需要在 search 内部处理
            # 这里我们通过设置阈值来间接控制
            threshold_low = 4.0 if config.use_rerank else 0.0
            threshold_high = 6.0 if config.use_rerank else 0.0

            results = self.vector_store.search(
                query,
                top_k=top_k,
                threshold_low=threshold_low,
                threshold_high=threshold_high
            )

            # 如果不使用 BM25，过滤掉 BM25 来源的结果
            if not config.use_bm25:
                filtered_results = []
                for r in results.get("results", []):
                    source = r.get("retrieval_source", "")
                    if "BM25" not in source:
                        filtered_results.append(r)
                results["results"] = filtered_results[:top_k]

            return results

        finally:
            # 恢复原始方法
            self.vector_store.query_processor.generate_hyde_doc = original_hyde
            self.vector_store.query_processor.rewrite_query = original_rewrite

    def _is_hit(self, retrieved: str, ground_truths: List[str]) -> bool:
        """简化的命中判断"""
        for gt in ground_truths:
            # 关键词匹配
            import re
            key_phrases = re.findall(r'【[^】]+】([^。]+)', gt)
            for phrase in key_phrases:
                key = phrase.strip()[:20]
                if key and key in retrieved:
                    return True

            # 短文本直接匹配
            if len(gt) < 100 and gt[:30] in retrieved:
                return True

            # 字符重叠
            overlap = len(set(retrieved) & set(gt)) / max(len(set(gt)), 1)
            if overlap > 0.4:
                return True

        return False

    def _print_comparison_table(self):
        """打印对比表格"""
        if not self.results:
            return

        print("\n" + "=" * 70)
        print("📊 消融实验结果对比")
        print("=" * 70)

        # 表头
        print(f"{'配置':<15} {'Recall@1':>10} {'Recall@3':>10} {'Recall@5':>10} {'MRR':>10} {'延迟(ms)':>10}")
        print("-" * 70)

        # 数据行
        for name, metrics in self.results.items():
            print(f"{name:<15} "
                  f"{metrics['recall@1']:>10.2%} "
                  f"{metrics['recall@3']:>10.2%} "
                  f"{metrics['recall@5']:>10.2%} "
                  f"{metrics['mrr']:>10.4f} "
                  f"{metrics['avg_latency_ms']:>10.0f}")

        print("-" * 70)

        # 增益分析
        if "baseline" in self.results and "full" in self.results:
            baseline = self.results["baseline"]
            full = self.results["full"]

            print("\n📈 关键发现:")

            # 总体增益
            if baseline["recall@3"] > 0:
                improvement = (full["recall@3"] - baseline["recall@3"]) / baseline["recall@3"] * 100
                print(f"   ✅ 完整 Pipeline vs Baseline: Recall@3 提升 {improvement:.1f}%")

            # 各模块增益
            prev_recall = baseline["recall@3"]
            for name in ["+ bm25", "+ rewrite", "+ hyde", "full"]:
                if name in self.results:
                    curr_recall = self.results[name]["recall@3"]
                    delta = (curr_recall - prev_recall) * 100
                    if delta > 0:
                        print(f"   ✅ {name} 贡献: +{delta:.1f}%")
                    prev_recall = curr_recall

        print("=" * 70)

    def export_results(self, output_path: str):
        """导出结果到 JSON 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"📄 Results exported to: {output_path}")


def run_ablation(vector_store, test_set: List[Dict], top_k: int = 5) -> Dict:
    """快速运行消融实验"""
    study = AblationStudy(vector_store)
    results = study.run_experiments(test_set, top_k=top_k)
    return results
