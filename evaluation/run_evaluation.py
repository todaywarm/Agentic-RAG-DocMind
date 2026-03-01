#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DocMind RAG 评估系统 - 主入口脚本

功能：
1. 传统检索指标评估（Recall@K, MRR, NDCG）
2. 消融实验（验证各模块增益）
3. 生成评估报告

使用方法：
    # 运行完整评估
    python -m evaluation.run_evaluation

    # 仅运行检索指标
    python -m evaluation.run_evaluation --mode retrieval

    # 仅运行消融实验
    python -m evaluation.run_evaluation --mode ablation

    # 指定测试集
    python -m evaluation.run_evaluation --test-set path/to/test_set.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from evaluation.retrieval_metrics import RetrievalEvaluator
from evaluation.generation_metrics import GenerationEvaluator
from evaluation.ablation_study import AblationStudy
from evaluation.test_set_generator import load_or_create_test_set, get_default_test_set

# RAGAS 是可选的
try:
    from evaluation.ragas_evaluator import RAGASEvaluator
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def init_vector_store():
    """初始化 VectorStore"""
    print("🚀 Initializing DocMind RAG System...")

    from docmind.memory.vector_store import VectorStore

    vector_store = VectorStore(
        collection_name="docmind_java_manual_bge",
        persist_dir="./data/qdrant_db_bge"
    )

    print("   ✅ System ready\n")
    return vector_store


def run_retrieval_evaluation(vector_store, test_set, top_k=10):
    """运行检索指标评估"""
    print("\n" + "=" * 60)
    print("🔍 Layer 1: Retrieval Metrics Evaluation")
    print("=" * 60)

    evaluator = RetrievalEvaluator(vector_store)
    results = evaluator.evaluate(test_set, top_k=top_k)
    evaluator.print_summary(results)

    return results


def run_generation_evaluation(vector_store, test_set, top_k=5):
    """运行生成质量评估（Faithfulness 等）"""
    print("\n" + "=" * 60)
    print("🎯 Layer 2: Generation Quality Evaluation (LLM-as-Judge)")
    print("=" * 60)

    evaluator = GenerationEvaluator()
    results = evaluator.evaluate(test_set, vector_store, top_k=top_k)
    evaluator.print_summary(results)

    return results


def run_ragas_evaluation(vector_store, test_set, top_k=5):
    """运行 RAGAS 评估（业界标准）"""
    print("\n" + "=" * 60)
    print("📊 Layer 3: RAGAS Evaluation (Industry Standard)")
    print("=" * 60)

    if not RAGAS_AVAILABLE:
        print("⚠️ RAGAS not installed. Install with:")
        print("   pip install ragas datasets langchain-openai langchain-community")
        print("   Skipping RAGAS evaluation...")
        return {}

    evaluator = RAGASEvaluator(vector_store)
    results = evaluator.evaluate(test_set, top_k=top_k)
    evaluator.print_summary(results)

    return results


def run_ablation_study(vector_store, test_set, top_k=5):
    """运行消融实验"""
    print("\n" + "=" * 60)
    print("🧪 Ablation Study")
    print("=" * 60)

    study = AblationStudy(vector_store)
    results = study.run_experiments(test_set, top_k=top_k)

    return results


def generate_report(
    retrieval_results: dict,
    ablation_results: dict,
    test_set_size: int,
    output_dir: str,
    generation_results: dict = None,
    ragas_results: dict = None
):
    """生成评估报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON 报告
    report = {
        "timestamp": timestamp,
        "test_set_size": test_set_size,
        "retrieval_metrics": {
            k: v for k, v in retrieval_results.items()
            if k != "details"  # 不保存详细结果到摘要
        } if retrieval_results else {},
        "generation_metrics": {
            k: v for k, v in (generation_results or {}).items()
            if k != "details"
        },
        "ragas_metrics": {
            k: v for k, v in (ragas_results or {}).items()
            if k != "details"
        },
        "ablation_study": ablation_results
    }

    json_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    md_content = generate_markdown_report(report)
    md_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n📄 Reports saved to:")
    print(f"   - JSON: {json_path}")
    print(f"   - Markdown: {md_path}")

    return report


def generate_markdown_report(report: dict) -> str:
    """生成 Markdown 格式报告"""
    rm = report.get("retrieval_metrics", {})
    gm = report.get("generation_metrics", {})
    rg = report.get("ragas_metrics", {})
    ab = report.get("ablation_study", {})

    md = f"""# DocMind RAG 评估报告

## 📋 测试概览
- **测试集大小**: {report['test_set_size']} 个问答对
- **评估时间**: {report['timestamp']}

---

## 1. 检索性能指标 (Retrieval)

### 召回率 (Recall)
| 指标 | 得分 |
|------|------|
| Recall@1 | {rm.get('recall@1', 0):.2%} |
| Recall@3 | {rm.get('recall@3', 0):.2%} |
| Recall@5 | {rm.get('recall@5', 0):.2%} |
| Recall@10 | {rm.get('recall@10', 0):.2%} |

### 排序质量
| 指标 | 得分 | 说明 |
|------|------|------|
| MRR | {rm.get('mrr', 0):.4f} | 平均倒数排名 |
| NDCG@3 | {rm.get('ndcg@3', 0):.4f} | 归一化折损累积增益 |
| NDCG@5 | {rm.get('ndcg@5', 0):.4f} | - |

### 性能指标
| 指标 | 数值 |
|------|------|
| 平均延迟 | {rm.get('avg_latency_ms', 0):.0f} ms |
| 平均 Rerank 分数 | {rm.get('avg_rerank_score', 0):.2f} |
| 命中率 | {rm.get('hit_rate', 0):.2%} |

---

## 2. 生成质量指标 (Generation)
"""

    # RAGAS 指标（如果有）
    if rg:
        md += f"""
### RAGAS 评估（业界标准）
| 指标 | 得分 | 说明 |
|------|------|------|
| Context Precision | {rg.get('context_precision', 0):.2%} | 检索结果排序质量 |
| Context Recall | {rg.get('context_recall', 0):.2%} | 信息完整性 |
| **Faithfulness** | **{rg.get('faithfulness', 0):.2%}** | 答案忠实度（抗幻觉）⭐ |
| Answer Relevancy | {rg.get('answer_relevancy', 0):.2%} | 答案相关性 |
"""

    # LLM-as-Judge 指标（如果有）
    if gm:
        md += f"""
### LLM-as-Judge 评估
| 指标 | 得分 |
|------|------|
| Faithfulness | {gm.get('faithfulness', 0):.2%} |
| Answer Relevancy | {gm.get('answer_relevancy', 0):.2%} |
| Groundedness | {gm.get('groundedness', 0):.2%} |
"""

    md += """
---

## 3. 消融实验 (Ablation Study)

各模块对 Recall@3 的贡献分析：

| 配置 | Recall@1 | Recall@3 | MRR | 延迟(ms) |
|------|----------|----------|-----|----------|
"""

    for name, metrics in ab.items():
        md += f"| {name} | {metrics.get('recall@1', 0):.2%} | {metrics.get('recall@3', 0):.2%} | {metrics.get('mrr', 0):.4f} | {metrics.get('avg_latency_ms', 0):.0f} |\n"

    # 增益分析
    baseline_r3 = 0
    full_r3 = 0
    if "baseline" in ab and "full" in ab:
        baseline_r3 = ab["baseline"].get("recall@3", 0)
        full_r3 = ab["full"].get("recall@3", 0)
        if baseline_r3 > 0:
            improvement = (full_r3 - baseline_r3) / baseline_r3 * 100
            md += f"""
### 关键发现
- ✅ **完整 Pipeline vs Baseline**: Recall@3 提升 **{improvement:.1f}%**
"""

    # 简历建议
    faithfulness = rg.get('faithfulness', 0) or gm.get('faithfulness', 0)
    imp = (full_r3 - baseline_r3) / baseline_r3 * 100 if baseline_r3 > 0 else 0

    md += f"""
---

## 4. 简历撰写建议

基于以上评估结果，建议在简历中这样描述：

> **DocMind - 企业级 RAG 智能问答系统**
>
> - 设计三路混合检索架构（HyDE + 语义重写 + BM25），**Recall@3 达 {rm.get('recall@3', 0):.0%}**
> - 引入 BGE-Reranker + 三级置信度过滤，**MRR 达 {rm.get('mrr', 0):.4f}**
> - 通过消融实验验证各模块增益，完整 Pipeline 较 Baseline **提升 {imp:.0f}%**
> - RAGAS **Faithfulness 达 {faithfulness:.2f}**（业界优秀水平 0.85+）
> - 平均检索延迟 **{rm.get('avg_latency_ms', 0):.0f}ms**，满足生产环境要求

---

*报告由 DocMind 评估系统自动生成*
"""

    return md


def main():
    parser = argparse.ArgumentParser(description="DocMind RAG Evaluation System")
    parser.add_argument(
        "--mode",
        choices=["all", "retrieval", "generation", "ragas", "ablation"],
        default="all",
        help="Evaluation mode: all, retrieval, generation (LLM-as-Judge), ragas (industry standard), ablation"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Path to test set JSON file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/evaluation",
        help="Output directory for reports"
    )

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载测试集
    print("📂 Loading test set...")
    if args.test_set and os.path.exists(args.test_set):
        test_set = load_or_create_test_set(args.test_set)
    else:
        # 使用默认测试集
        default_path = os.path.join(args.output_dir, "golden_set.json")
        test_set = load_or_create_test_set(default_path)

    print(f"   Loaded {len(test_set)} test cases\n")

    # 初始化系统
    vector_store = init_vector_store()

    # 运行评估
    retrieval_results = {}
    generation_results = {}
    ragas_results = {}
    ablation_results = {}

    if args.mode in ["all", "retrieval"]:
        retrieval_results = run_retrieval_evaluation(
            vector_store, test_set, top_k=args.top_k
        )

    if args.mode in ["all", "generation"]:
        generation_results = run_generation_evaluation(
            vector_store, test_set, top_k=args.top_k
        )

    if args.mode in ["all", "ragas"]:
        ragas_results = run_ragas_evaluation(
            vector_store, test_set, top_k=args.top_k
        )

    if args.mode in ["all", "ablation"]:
        ablation_results = run_ablation_study(
            vector_store, test_set, top_k=args.top_k
        )

    # 生成报告
    if retrieval_results or generation_results or ragas_results or ablation_results:
        generate_report(
            retrieval_results,
            ablation_results,
            len(test_set),
            args.output_dir,
            generation_results,
            ragas_results
        )

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
