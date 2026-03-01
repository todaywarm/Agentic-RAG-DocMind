"""
RAGAS 评估器

RAGAS (Retrieval Augmented Generation Assessment) 是业界标准的 RAG 评估框架。

核心指标：
- context_precision: 检索结果中相关文档的排名质量
- context_recall: 答案所需信息是否都被召回
- faithfulness: 答案是否基于检索内容（抗幻觉）
- answer_relevancy: 答案是否解决了用户问题

安装依赖：
    pip install ragas datasets langchain-openai

参考文档：https://docs.ragas.io/
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 检查 RAGAS 是否安装
RAGAS_AVAILABLE = False
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall, 
        faithfulness,
        answer_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    pass


class RAGASEvaluator:
    """RAGAS 端到端评估器"""

    def __init__(self, vector_store=None, llm_client=None):
        """
        Args:
            vector_store: VectorStore 实例
            llm_client: OpenAI 兼容的 LLM 客户端
        """
        if not RAGAS_AVAILABLE:
            print("⚠️ RAGAS not installed. Install with: pip install ragas datasets langchain-openai")
            
        self.vector_store = vector_store
        self.llm_client = llm_client or self._init_llm_client()
        
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(os.path.join(project_root, '.env'))
        
        return OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        )

    def evaluate(
        self,
        test_set: List[Dict],
        top_k: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        使用 RAGAS 评估完整 RAG 流程

        Args:
            test_set: 测试集，每个元素包含：
                - question: 问题
                - ground_truth_answer: 标准答案（用于 context_recall）
                - ground_truth_contexts: 标准上下文（可选）
            top_k: 检索数量
            verbose: 是否显示进度

        Returns:
            {
                "context_precision": float,
                "context_recall": float,
                "faithfulness": float,
                "answer_relevancy": float,
                "details": DataFrame
            }
        """
        if not RAGAS_AVAILABLE:
            print("❌ RAGAS not available. Using fallback evaluation...")
            return self._fallback_evaluate(test_set, top_k, verbose)

        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        iterator = tqdm(test_set, desc="Running RAG Pipeline") if verbose else test_set

        for item in iterator:
            question = item["question"]

            # 1. 执行检索
            search_results = self.vector_store.search(question, top_k=top_k)
            retrieved_contexts = [r.get("content", "") for r in search_results.get("results", [])]

            # 2. 生成答案
            answer = self._generate_answer(question, retrieved_contexts)

            # 3. 收集数据
            questions.append(question)
            answers.append(answer)
            contexts_list.append(retrieved_contexts if retrieved_contexts else [""])
            ground_truths.append(item.get("ground_truth_answer", ""))

        # 转换为 RAGAS Dataset 格式
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        })

        # 配置 RAGAS 使用的 LLM
        ragas_llm, ragas_embeddings = self._get_ragas_models()

        # 执行 RAGAS 评估
        print("\n🔄 Running RAGAS evaluation...")
        try:
            result = evaluate(
                ragas_dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                ],
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )

            result_df = result.to_pandas()

            return {
                "context_precision": float(result_df["context_precision"].mean()),
                "context_recall": float(result_df["context_recall"].mean()),
                "faithfulness": float(result_df["faithfulness"].mean()),
                "answer_relevancy": float(result_df["answer_relevancy"].mean()),
                "total_samples": len(test_set),
                "details": result_df.to_dict('records')
            }
        except Exception as e:
            print(f"❌ RAGAS evaluation failed: {e}")
            print("   Falling back to custom evaluation...")
            return self._fallback_evaluate(test_set, top_k, verbose)

    def _fallback_evaluate(
        self,
        test_set: List[Dict],
        top_k: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        当 RAGAS 不可用时的回退评估方案（LLM-as-Judge）
        """
        from evaluation.generation_metrics import GenerationEvaluator
        
        gen_evaluator = GenerationEvaluator(self.llm_client)
        results = gen_evaluator.evaluate(test_set, self.vector_store, top_k, verbose)
        
        # 映射到 RAGAS 风格的指标名
        return {
            "context_precision": 0.0,  # 回退方案无法计算
            "context_recall": 0.0,
            "faithfulness": results.get("faithfulness", 0),
            "answer_relevancy": results.get("answer_relevancy", 0),
            "total_samples": results.get("total_samples", 0),
            "details": results.get("details", []),
            "note": "Using fallback LLM-as-Judge (RAGAS not available)"
        }

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """生成答案"""
        if not contexts:
            return "抱歉，没有找到相关信息。"

        context_str = "\n\n".join([f"【参考{i+1}】{ctx}" for i, ctx in enumerate(contexts[:5])])

        prompt = f"""请根据以下参考资料回答问题。

参考资料：
{context_str}

问题：{question}

要求：
1. 仅根据参考资料回答，不要编造
2. 如果资料不足以回答，请明确说明
3. 回答简洁清晰

答案："""

        try:
            response = self.llm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL_ID", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"生成答案时出错: {e}"

    def _get_ragas_models(self):
        """获取 RAGAS 需要的 LLM 和 Embeddings"""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            
            # 使用 DeepSeek 作为评估 LLM
            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL_ID", "deepseek-chat"),
                openai_api_key=os.getenv("LLM_API_KEY"),
                openai_api_base=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
                temperature=0
            )
            
            # Embeddings - 尝试使用 SiliconFlow 或本地模型
            rag_api_key = os.getenv("RAG_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
            if rag_api_key:
                embeddings = OpenAIEmbeddings(
                    model="BAAI/bge-m3",
                    openai_api_key=rag_api_key,
                    openai_api_base="https://api.siliconflow.cn/v1"
                )
            else:
                # 回退到 HuggingFace 本地模型
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-zh-v1.5"
                )
                
            return llm, embeddings
            
        except ImportError as e:
            print(f"⚠️ Failed to initialize RAGAS models: {e}")
            print("   Install: pip install langchain-openai langchain-community")
            return None, None

    def print_summary(self, results: Dict[str, Any]):
        """打印评估结果摘要"""
        print("\n" + "=" * 60)
        print("📊 RAGAS 评估结果")
        print("=" * 60)

        print(f"\n🔍 检索质量")
        print(f"   Context Precision:  {results.get('context_precision', 0):.2%}")
        print(f"   Context Recall:     {results.get('context_recall', 0):.2%}")

        print(f"\n🎯 生成质量")
        print(f"   Faithfulness:       {results.get('faithfulness', 0):.2%}")
        print(f"   Answer Relevancy:   {results.get('answer_relevancy', 0):.2%}")

        print(f"\n📋 统计")
        print(f"   测试样本: {results.get('total_samples', 0)}")
        
        if results.get("note"):
            print(f"\n⚠️ 注意: {results['note']}")

        print("=" * 60)

        # 简历建议
        faith = results.get('faithfulness', 0)
        relevancy = results.get('answer_relevancy', 0)
        
        print(f"\n📝 简历写法建议:")
        print(f"   「RAGAS Faithfulness 达 {faith:.2f}，Answer Relevancy 达 {relevancy:.2f}」")


def quick_ragas_eval(vector_store, test_set: List[Dict], top_k: int = 5) -> Dict:
    """快速 RAGAS 评估"""
    evaluator = RAGASEvaluator(vector_store)
    results = evaluator.evaluate(test_set, top_k=top_k)
    evaluator.print_summary(results)
    return results


# ============ 独立运行 ============
if __name__ == "__main__":
    print("检查 RAGAS 安装状态...")
    
    if RAGAS_AVAILABLE:
        print("✅ RAGAS 已安装")
    else:
        print("❌ RAGAS 未安装")
        print("\n安装命令:")
        print("  pip install ragas datasets langchain-openai langchain-community")
