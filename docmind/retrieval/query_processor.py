"""
查询处理器

负责查询的预处理，包括：
- 查询重写（双轨制）
- 上下文重写（多轮对话）
- HyDE 生成
- 元问题识别
"""

import re
from typing import Dict, List, Optional, Any

from docmind.llm import LLMClient, get_llm_client
from docmind.llm.prompts import (
    QUERY_REWRITE_PROMPT,
    CONTEXT_REWRITE_PROMPT,
    HYDE_PROMPT,
    IDENTITY_RESPONSE,
)
from docmind.core.exceptions import LLMError


class QueryProcessor:
    """
    查询处理器
    
    职责：
    - 查询理解和预处理
    - 为检索优化查询
    """
    
    def __init__(self, llm: LLMClient = None):
        """
        初始化查询处理器
        
        Args:
            llm: LLM 客户端实例
        """
        self.llm = llm or get_llm_client()
    
    def process(
        self,
        query: str,
        history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        完整的查询处理流程
        
        Args:
            query: 原始查询
            history: 对话历史
            
        Returns:
            {
                "original_query": str,
                "standalone_query": str,  # 上下文重写后
                "vector_query": str,      # 语义优化后（用于向量检索）
                "keywords": List[str],    # 扩展关键词（用于 BM25）
                "hyde_doc": str,          # 假设文档
                "direct_answer": str | None,  # 元问题直接回答
            }
        """
        result = {
            "original_query": query,
            "standalone_query": query,
            "vector_query": query,
            "keywords": [query],
            "hyde_doc": query,
            "direct_answer": None,
        }
        
        # 1. 检查元问题（直接回答，不走检索）
        direct_answer = self.get_direct_answer(query, history)
        if direct_answer:
            result["direct_answer"] = direct_answer
            return result
        
        # 2. 上下文重写（处理多轮对话中的指代）
        if history:
            standalone = self.rewrite_with_context(query, history)
            result["standalone_query"] = standalone
            query = standalone  # 后续处理使用重写后的查询
        
        # 3. 双轨查询重写
        rewrite_result = self.rewrite_query(query)
        result["vector_query"] = rewrite_result.get("vector_query", query)
        result["keywords"] = rewrite_result.get("keywords", [query])
        
        # 4. HyDE 生成
        result["hyde_doc"] = self.generate_hyde_doc(query)
        
        return result
    
    def get_direct_answer(
        self,
        query: str,
        history: List[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        识别并回答元问题（不需要检索的问题）
        
        支持的元问题类型：
        - 身份询问：你是谁/你叫什么
        - 历史询问：我刚刚问了什么/你刚才说了什么
        """
        q = (query or "").strip()
        if not q:
            return None
        
        # 身份询问
        identity_patterns = ["你是谁", "你是啥", "你是什么", "你叫什么", "介绍一下你", "你是哪位"]
        if any(p in q for p in identity_patterns):
            return IDENTITY_RESPONSE
        
        # 查询历史中的最后一条消息
        def last_message(role: str) -> Optional[str]:
            if not history:
                return None
            for turn in reversed(history):
                if turn.get("role") == role:
                    content = (turn.get("content") or "").strip()# 注意python的语法限制情形 这里返回的不是布尔值 而是返回第一个为真的值
                    if content:
                        return content
            return None
        
        # 询问上一个问题
        if ("我" in q and any(x in q for x in ["刚刚", "刚才", "上次", "之前"]) and "问" in q) or "上一个问题" in q:
            last_user = last_message("user")
            if last_user:
                return f"你刚刚问的是：{last_user}"
            return "当前会话还没有上一条用户问题。"
        
        # 询问上一个回答
        if re.search(r"(你|您).*(刚刚|刚才|上次).*(说|回答).*(什么|啥)", q):
            last_assistant = last_message("assistant")
            if last_assistant:
                return f"我刚刚的回答是：{last_assistant}"
            return "当前会话还没有上一条助手回复。"
        
        return None
    
    def rewrite_with_context(
        self,
        query: str,
        history: List[Dict[str, str]]
    ) -> str:
        """
        基于对话历史重写查询，使其独立完整
        
        Args:
            query: 当前查询
            history: 对话历史
            
        Returns:
            重写后的独立查询
        """
        if not history:
            return query
        
        # 构建历史字符串（只用最近3轮）
        history_str = ""
        for turn in history[-3:]:
            role = "User" if turn["role"] == "user" else "AI"
            history_str += f"{role}: {turn['content']}\n"
        
        prompt = CONTEXT_REWRITE_PROMPT.format( #构建重写查询的提示送入大模型进行生成
            history=history_str,
            query=query
        )
        
        try:
            rewritten = self.llm.complete(prompt, temperature=0.3, max_tokens=100)
            return rewritten.strip() or query
        except Exception as e:
            print(f"Context rewrite error: {e}")
            return query
    
    def rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        双轨查询重写
        
        Returns:
            {
                "vector_query": str,  # 语义优化（用于向量检索）
                "keywords": List[str] # 扩展关键词（用于 BM25）
            }
        """
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        
        try:
            content = self.llm.complete(prompt, temperature=0.5, max_tokens=150)
            
            result = {
                "vector_query": query,
                "keywords": [query]
            }
            
            # 解析输出
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("[Vector]"):
                    result["vector_query"] = line.replace("[Vector]", "").strip()
                elif line.startswith("[Keywords]"):
                    kws = line.replace("[Keywords]", "").strip()
                    kws = kws.replace("，", ",")
                    result["keywords"] = [k.strip() for k in kws.split(',') if k.strip()]
            
            return result
            
        except Exception as e:
            print(f"Query rewrite error: {e}")
            return {"vector_query": query, "keywords": [query]}
    
    def generate_hyde_doc(self, query: str) -> str:
        """
        生成假设性回答（HyDE）
        
        用于向量检索，提高语义匹配度
        """
        prompt = HYDE_PROMPT.format(query=query)
        
        try:
            hyde_doc = self.llm.complete(prompt, temperature=0.7, max_tokens=200)
            return hyde_doc.strip() or query
        except Exception as e:
            print(f"HyDE generation error: {e}")
            return query
