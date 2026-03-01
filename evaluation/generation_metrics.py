"""
生成质量评估器

包含：
- Faithfulness（忠实度）: 答案是否基于检索内容，不编造
- Answer Relevancy（答案相关性）: 答案是否解决了用户问题
- Groundedness（溯源性）: 答案中的每个声明是否都有依据

这些指标使用 LLM-as-Judge 方式评估
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class GenerationEvaluator:
    """生成质量评估器（LLM-as-Judge）"""

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM 客户端，如果不提供则自动初始化
        """
        self.llm = llm or self._init_llm()

    def _init_llm(self):
        """初始化 LLM 客户端"""
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(os.path.join(project_root, '.env'))

        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

        if not api_key:
            raise ValueError("LLM_API_KEY not found in .env")

        return OpenAI(api_key=api_key, base_url=base_url)

    def evaluate(
        self,
        test_set: List[Dict],
        vector_store,
        top_k: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整的生成质量评估

        Args:
            test_set: 测试集
            vector_store: VectorStore 实例
            top_k: 检索数量
            verbose: 是否显示进度

        Returns:
            {
                "faithfulness": float,      # 忠实度均值
                "answer_relevancy": float,  # 答案相关性均值
                "groundedness": float,      # 溯源性均值
                "details": [...]            # 每个样本的详细结果
            }
        """
        faithfulness_scores = []
        relevancy_scores = []
        groundedness_scores = []
        details = []

        iterator = tqdm(test_set, desc="Evaluating Generation") if verbose else test_set

        for item in iterator:
            question = item["question"]

            # 1. 执行 RAG 检索
            search_results = vector_store.search(question, top_k=top_k)
            contexts = [r.get("content", "") for r in search_results.get("results", [])]

            # 2. 生成答案
            answer = self._generate_answer(question, contexts)

            # 3. 评估忠实度
            faith_score, faith_reason = self.evaluate_faithfulness(answer, contexts)
            faithfulness_scores.append(faith_score)

            # 4. 评估答案相关性
            relevancy_score, relevancy_reason = self.evaluate_answer_relevancy(
                question, answer
            )
            relevancy_scores.append(relevancy_score)

            # 5. 评估溯源性
            ground_score, ground_details = self.evaluate_groundedness(answer, contexts)
            groundedness_scores.append(ground_score)

            details.append({
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "num_contexts": len(contexts),
                "faithfulness": faith_score,
                "faithfulness_reason": faith_reason,
                "answer_relevancy": relevancy_score,
                "relevancy_reason": relevancy_reason,
                "groundedness": ground_score,
                "groundedness_details": ground_details
            })

        return {
            "faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0,
            "answer_relevancy": np.mean(relevancy_scores) if relevancy_scores else 0,
            "groundedness": np.mean(groundedness_scores) if groundedness_scores else 0,
            "total_samples": len(test_set),
            "details": details
        }

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> tuple[float, str]:
        """
        评估忠实度：答案是否完全基于提供的上下文

        Returns:
            (score: 0-1, reason: str)
        """
        if not contexts:
            return 0.0, "No context provided"

        context_str = "\n\n".join([f"[文档{i+1}] {ctx[:500]}" for i, ctx in enumerate(contexts[:5])])

        prompt = f"""你是一个严格的评估专家。请判断【回答】是否完全基于【参考文档】的内容。

【参考文档】
{context_str}

【回答】
{answer}

评估标准：
- 1.0 分：回答完全基于参考文档，没有任何编造或超出文档范围的内容
- 0.8 分：回答主要基于文档，但有少量合理推断
- 0.5 分：回答部分基于文档，部分是常识性补充
- 0.2 分：回答大部分内容在文档中找不到依据
- 0.0 分：回答完全是编造的，与文档无关

请严格按以下 JSON 格式输出（不要有其他内容）：
{{
    "score": 0.0-1.0,
    "reason": "简要说明评分理由",
    "unsupported_claims": ["列出无法从文档中找到依据的说法"]
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            result_text = response.choices[0].message.content.strip()

            # 解析 JSON
            result_text = self._extract_json(result_text)
            result = json.loads(result_text)

            return float(result.get("score", 0)), result.get("reason", "")
        except Exception as e:
            print(f"Faithfulness evaluation error: {e}")
            return 0.5, f"Evaluation failed: {str(e)}"

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> tuple[float, str]:
        """
        评估答案相关性：答案是否解决了用户的问题

        Returns:
            (score: 0-1, reason: str)
        """
        prompt = f"""你是一个评估专家。请判断【回答】是否有效解决了【用户问题】。

【用户问题】
{question}

【回答】
{answer}

评估标准：
- 1.0 分：完全解决问题，答案准确、完整、切题
- 0.8 分：基本解决问题，但可能缺少一些细节
- 0.5 分：部分解决问题，或者答非所问
- 0.2 分：基本没有解决问题
- 0.0 分：完全没有回答问题，或者答案是"不知道"

请严格按以下 JSON 格式输出：
{{
    "score": 0.0-1.0,
    "reason": "简要说明评分理由"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            result_text = response.choices[0].message.content.strip()
            result_text = self._extract_json(result_text)
            result = json.loads(result_text)

            return float(result.get("score", 0)), result.get("reason", "")
        except Exception as e:
            print(f"Relevancy evaluation error: {e}")
            return 0.5, f"Evaluation failed: {str(e)}"

    def evaluate_groundedness(
        self,
        answer: str,
        contexts: List[str]
    ) -> tuple[float, List[Dict]]:
        """
        评估溯源性：答案中的每个关键声明是否都能找到依据

        Returns:
            (score: 0-1, claim_details: List[Dict])
        """
        if not contexts:
            return 0.0, []

        context_str = "\n\n".join([f"[文档{i+1}] {ctx[:400]}" for i, ctx in enumerate(contexts[:5])])

        prompt = f"""你是一个事实核查专家。请分析【回答】中的每个关键声明，判断是否能在【参考文档】中找到依据。

【参考文档】
{context_str}

【回答】
{answer}

请执行以下步骤：
1. 从回答中提取 3-5 个关键声明（Key Claims）
2. 对每个声明，判断是否能在文档中找到依据
3. 计算总体溯源得分

请严格按以下 JSON 格式输出：
{{
    "claims": [
        {{"claim": "声明内容", "supported": true/false, "evidence": "文档中的依据或'无'"}},
        ...
    ],
    "overall_score": 0.0-1.0,
    "summary": "总体评价"
}}"""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            result_text = response.choices[0].message.content.strip()
            result_text = self._extract_json(result_text)
            result = json.loads(result_text)

            claims = result.get("claims", [])
            score = float(result.get("overall_score", 0))

            return score, claims
        except Exception as e:
            print(f"Groundedness evaluation error: {e}")
            return 0.5, []

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """调用 RAG 生成答案"""
        if not contexts:
            return "抱歉，没有找到相关信息。"

        context_str = "\n\n".join([f"【参考{i+1}】{ctx}" for i, ctx in enumerate(contexts[:5])])

        prompt = f"""请根据以下参考资料回答问题。

参考资料：
{context_str}

问题：{question}

要求：
1. 仅根据参考资料回答，不要编造
2. 如果资料不足，明确说明
3. 回答简洁清晰

答案："""

        try:
            response = self.llm.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"生成答案时出错: {e}"

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON"""
        # 处理 markdown 代码块
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # 尝试找到 JSON 对象
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group()

        return text

    def print_summary(self, results: Dict[str, Any]):
        """打印评估结果摘要"""
        print("\n" + "=" * 60)
        print("📊 生成质量评估结果")
        print("=" * 60)

        print(f"\n🎯 核心指标")
        print(f"   Faithfulness（忠实度）:    {results['faithfulness']:.2%}")
        print(f"   Answer Relevancy（相关性）: {results['answer_relevancy']:.2%}")
        print(f"   Groundedness（溯源性）:    {results['groundedness']:.2%}")

        print(f"\n📋 统计")
        print(f"   测试样本: {results['total_samples']}")

        # 显示低分样本
        low_faith = [d for d in results['details'] if d['faithfulness'] < 0.5]
        if low_faith:
            print(f"\n⚠️ 低忠实度样本 ({len(low_faith)} 个):")
            for d in low_faith[:3]:
                print(f"   Q: {d['question'][:40]}...")
                print(f"      原因: {d['faithfulness_reason']}")

        print("=" * 60)


def quick_generation_eval(vector_store, test_set: List[Dict], top_k: int = 5) -> Dict:
    """快速生成质量评估"""
    evaluator = GenerationEvaluator()
    results = evaluator.evaluate(test_set, vector_store, top_k=top_k)
    evaluator.print_summary(results)
    return results
