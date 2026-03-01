#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速评估脚本 - 独立运行版本

使用方法：
    cd DocMind_agent
    python evaluation/quick_eval.py

这个脚本可以独立运行，不需要额外参数。
"""

import os
import sys

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv('.env')


def main():
    print("=" * 60)
    print("🚀 DocMind RAG 快速评估")
    print("=" * 60)

    # 1. 初始化 VectorStore
    print("\n📦 初始化向量存储...")
    from docmind.memory.vector_store import VectorStore

    vector_store = VectorStore(
        collection_name="docmind_java_manual_bge",
        persist_dir="./data/qdrant_db_bge"
    )

    # 2. 加载测试集
    print("\n📂 加载测试集...")
    from evaluation.test_set_generator import get_default_test_set
    test_set = get_default_test_set()
    print(f"   已加载 {len(test_set)} 个测试用例")

    # 3. 运行检索评估
    print("\n" + "=" * 60)
    print("🔍 检索指标评估")
    print("=" * 60)

    from evaluation.retrieval_metrics import RetrievalEvaluator
    evaluator = RetrievalEvaluator(vector_store)
    retrieval_results = evaluator.evaluate(test_set, top_k=5, verbose=True)
    evaluator.print_summary(retrieval_results)

    # 4. 运行消融实验（可选，耗时较长）
    run_ablation = input("\n是否运行消融实验？(y/n, 默认 n): ").strip().lower()
    if run_ablation == 'y':
        print("\n" + "=" * 60)
        print("🧪 消融实验")
        print("=" * 60)

        from evaluation.ablation_study import AblationStudy
        study = AblationStudy(vector_store)
        ablation_results = study.run_experiments(test_set, top_k=5)

    # 5. 输出简历撰写建议
    print("\n" + "=" * 60)
    print("📝 简历撰写建议")
    print("=" * 60)

    r3 = retrieval_results.get('recall@3', 0)
    mrr = retrieval_results.get('mrr', 0)
    latency = retrieval_results.get('avg_latency_ms', 0)

    print(f"""
基于评估结果，建议在简历中这样描述：

┌─────────────────────────────────────────────────────────────┐
│  DocMind - 企业级 RAG 智能问答系统                           │
├─────────────────────────────────────────────────────────────┤
│  • 设计三路混合检索架构（HyDE + 语义重写 + BM25）            │
│    ➜ Recall@3 达 {r3:.0%}                                     │
│  • 引入 BGE-Reranker + 三级置信度过滤                        │
│    ➜ MRR 达 {mrr:.4f}                                         │
│  • 平均检索延迟 {latency:.0f}ms，满足生产环境要求              │
└─────────────────────────────────────────────────────────────┘
""")

    print("✅ 评估完成！")


if __name__ == "__main__":
    main()
