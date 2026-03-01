"""
RAG Agent

整合检索、生成能力的智能体
"""

from typing import List, Dict, Any, Optional, Iterator

from docmind.retrieval import HybridRetriever
from docmind.generation import AnswerGenerator
from docmind.document import DocumentProcessor
from docmind.core.config import settings


class RAGAgent:
    """
    RAG 智能体
    
    整合完整的 RAG 流程：
    文档处理 → 索引 → 检索 → 生成
    """
    
    def __init__(
        self,
        retriever: HybridRetriever = None,
        generator: AnswerGenerator = None,
        doc_processor: DocumentProcessor = None,
    ):
        """
        初始化 RAG Agent
        
        Args:
            retriever: 混合检索器
            generator: 答案生成器
            doc_processor: 文档处理器
        """
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or AnswerGenerator()
        self.doc_processor = doc_processor or DocumentProcessor()
        
        # 对话历史
        self.history: List[Dict[str, str]] = []#eg:[{"role":"user","content":"hi"},{{"role":"AI","content":"hello, i am the rag assistant"}}]
        self.max_history: int = 10#最多存储十轮对话值
    
    def chat(
        self,
        query: str,
        use_history: bool = True,
        top_k: int = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        问答接口
        
        Args:
            query: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索数量
            stream: 是否流式返回
            
        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "confidence": str,
                "retrieval_info": Dict,
            }
        """
        # 获取对话历史
        history = self.history if use_history else None
        
        # 检索
        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            history=history,
        )
        # print("检索结果是",retrieval_result)
        # 如果是直接回答（元问题）
        if retrieval_result.get("direct_answer"):
            answer = retrieval_result["direct_answer"]
            result = {
                "answer": answer,
                "sources": [],
                "confidence": "direct",
                "retrieval_info": {
                    "type": "direct_answer",
                    "logs": retrieval_result.get("logs", [])
                }
            }
        else:
            # 生成答案
            gen_result = self.generator.generate_with_retrieval_results(
                query=query,
                retrieval_results=retrieval_result["results"]
            )
            
            result = {
                "answer": gen_result["answer"],
                "sources": gen_result["sources"],
                "confidence": gen_result["confidence"],
                "retrieval_info": {
                    "type": "rag",
                    "query_info": retrieval_result.get("query_info"),
                    "num_results": len(retrieval_result["results"]),
                    "logs": retrieval_result.get("logs", [])
                }
            }
        
        # 更新历史
        self._update_history(query, result["answer"])
        
        return result
    
    def chat_stream(
        self,
        query: str,
        use_history: bool = True,
        top_k: int = None,
    ) -> Iterator[str]:
        """
        流式问答接口
        
        Yields:
            答案片段
        """
        history = self.history if use_history else None
        
        # 检索
        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            history=history,
        )
       
        # 直接回答
        if retrieval_result.get("direct_answer"):
            yield retrieval_result["direct_answer"]
            self._update_history(query, retrieval_result["direct_answer"])
            return
        
        # 流式生成
        contexts = [r["content"] for r in retrieval_result["results"]]
        full_answer = ""
        
        for chunk in self.generator.generate_streaming(query, contexts):
            full_answer += chunk
            yield chunk
        
        self._update_history(query, full_answer)
    
    def add_document(
        self,
        file_path: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        添加文档到知识库
        
        Args:
            file_path: 文件路径
            force: 是否强制重新索引
            
        Returns:
            {"status": str, "chunks": int, "message": str}
        """
        import os
        filename = os.path.basename(file_path)
        
        # 检查是否已存在
        if not force and self.retriever.vector_store.document_exists(filename):
            return {
                "status": "skipped",
                "chunks": 0,
                "message": f"Document '{filename}' already indexed. Use force=True to re-index."
            }
        
        # 处理文档
        chunks = self.doc_processor.process(file_path)#将文件切块处理后返回
        """[
             {
                "content": str,          # 增强后的内容（包含标题路径）
                "metadata": {
                    "source": str,       # 源文件名
                    "h1": str,           # 一级标题
                    "h2": str,           # 二级标题
                    "h3": str,           # 三级标题
                    "raw_content": str,  # 原始内容
                }
            }
            ]
        """
        
        if not chunks:
            return {
                "status": "error",
                "chunks": 0,
                "message": "No chunks extracted from document."
            }
        
        # 添加到索引
        documents = [c["content"] for c in chunks]
        metadatas = [{"source": filename, **c["metadata"]} for c in chunks]
        
        count = self.retriever.add_documents(documents, metadatas)#将chunks用embedding模型嵌入后构建向量数据点 初始化payload包含content以及metatdata
        
        return {
            "status": "indexed",
            "chunks": count,
            "message": f"Successfully indexed {count} chunks from '{filename}'."
        }
    
    def reset_knowledge_base(self):
        """重置知识库"""
        self.retriever.reset()
        return {"status": "reset", "message": "Knowledge base has been reset."}
    
    def clear_history(self):
        """清除对话历史"""
        self.history = []
        return {"status": "cleared", "message": "Conversation history cleared."}
    
    def _update_history(self, query: str, answer: str):
        """更新对话历史"""
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        
        # 限制历史长度
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]


# 便捷函数
def create_agent(**kwargs) -> RAGAgent:
    """创建 RAG Agent"""
    return RAGAgent(**kwargs)
