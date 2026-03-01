"""
答案生成器

负责基于检索结果生成最终答案
"""

from typing import List, Dict, Any, Optional

from docmind.llm import LLMClient, get_llm_client
from docmind.llm.prompts import (
    ANSWER_GENERATION_PROMPT,
    NO_CONTEXT_RESPONSE,
    GENERATION_ERROR_RESPONSE,
)
from docmind.core.exceptions import GenerationError


class AnswerGenerator:
    """
    答案生成器
    
    职责：
    - 基于检索上下文生成答案
    - 格式化参考资料
    - 处理生成异常
    """
    
    def __init__(
        self,
        llm: LLMClient = None,
        max_contexts: int = 6,
        max_context_length: int = 500,
    ):
        """
        初始化答案生成器
        
        Args:
            llm: LLM 客户端
            max_contexts: 最大上下文数量
            max_context_length: 单个上下文最大长度
        """
        self.llm = llm or get_llm_client()
        self.max_contexts = max_contexts
        self.max_context_length = max_context_length
    
    def generate(
        self,
        query: str,
        contexts: List[str],
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """
        生成答案
        
        Args:
            query: 用户问题
            contexts: 检索到的上下文列表（已按相关性排序）
            temperature: 温度参数
            **kwargs: 其他 LLM 参数
            
        Returns:
            生成的答案
        """
        # 注意这里的处理方式 他与我们的场景有关系 如果知识库中没有检索到结果 
        # 说明这个问题有一定难度 模型可以选择不回答 那么不回答就是直接返回一个硬编码的字符串“抱歉，知识库没有检索到信息”
        if not contexts:
            return NO_CONTEXT_RESPONSE
        
        # 限制上下文数量
        contexts = contexts[:self.max_contexts]
        
        # 格式化上下文
        formatted_context = self._format_contexts(contexts)
        
        # 构建 prompt
        prompt = ANSWER_GENERATION_PROMPT.format(
            query=query,
            context=formatted_context
        )
        
        try:
            answer = self.llm.complete(
                prompt,
                temperature=temperature,
                max_tokens=1000,
                **kwargs
            )
            return answer
        except Exception as e:
            print(f"Answer generation error: {e}")
            return GENERATION_ERROR_RESPONSE
    
    def generate_with_retrieval_results(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        基于检索结果生成答案（返回完整信息）
        
        Args:
            query: 用户问题
            retrieval_results: 检索结果列表
            **kwargs: 其他参数
            
        Returns:
            {
                "answer": str,
                "sources": List[Dict],  # 参考来源
                "confidence": str,      # 置信度（基于检索结果）
            }
        """
        # 提取上下文
        contexts = [r.get("content", "") for r in retrieval_results if r.get("content")]
        
        # 生成答案
        answer = self.generate(query, contexts, **kwargs)
        
        # 确定置信度
        if not retrieval_results:
            confidence = "none"
        elif all(r.get("confidence") == "high" for r in retrieval_results):
            confidence = "high"
        elif any(r.get("confidence") == "low" for r in retrieval_results):
            confidence = "low"
        else:
            confidence = "medium"
        
        # 格式化参考来源
        sources = []
        for i, r in enumerate(retrieval_results[:3]):  # 只返回前3个
            sources.append({
                "index": i + 1,
                "content": r.get("raw_content", r.get("content", ""))[:200],
                "source": r.get("source", "unknown"),
                "score": r.get("rerank_score", r.get("score", 0)),
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }
    
    def _format_contexts(self, contexts: List[str]) -> str:
        """格式化上下文为 prompt 片段"""
        formatted = []
        for i, ctx in enumerate(contexts):
            # 截断过长的上下文
            if len(ctx) > self.max_context_length:
                ctx = ctx[:self.max_context_length] + "..."
            
            formatted.append(f"【参考资料 {i+1}】:\n{ctx}")
        
        return "\n\n".join(formatted)
    
    def generate_streaming(
        self,
        query: str,
        contexts: List[str],
        temperature: float = 0.3,
        **kwargs
    ):
        """
        流式生成答案
        
        Yields:
            答案片段
        """
        if not contexts:
            yield NO_CONTEXT_RESPONSE
            return
        
        contexts = contexts[:self.max_contexts]
        formatted_context = self._format_contexts(contexts)
        
        prompt = ANSWER_GENERATION_PROMPT.format(
            query=query,
            context=formatted_context
        )
        
        try:
            for chunk in self.llm.chat(
                [{"role": "user", "content": prompt}],
                stream=True,
                temperature=temperature,
                **kwargs
            ):
                yield chunk
        except Exception as e:
            print(f"Streaming generation error: {e}")
            yield GENERATION_ERROR_RESPONSE


# 便捷函数
def generate_answer(query: str, contexts: List[str], **kwargs) -> str:
    """生成答案的便捷函数"""
    return AnswerGenerator().generate(query, contexts, **kwargs)
