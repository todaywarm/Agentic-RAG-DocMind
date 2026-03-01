"""
LLM 客户端

基于 OpenAI API 协议，支持多种 LLM 服务：
- DeepSeek
- OpenAI
- 其他兼容服务
"""

import os
from typing import List, Dict, Any, Optional, Iterator, Union
from openai import OpenAI

from docmind.core.config import settings
from docmind.core.exceptions import LLMError


class LLMClient:
    """
    统一的 LLM 客户端
    
    支持流式和非流式调用
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None,
    ):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: API 密钥，默认从配置读取
            base_url: API 地址，默认从配置读取
            model: 模型名称，默认从配置读取
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 超时时间（秒）
        """
        self.api_key = api_key or settings.llm.api_key
        self.base_url = base_url or settings.llm.base_url
        self.model = model or settings.llm.model_id
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_tokens = max_tokens or settings.llm.max_tokens
        self.timeout = timeout or settings.llm.timeout
        
        if not self.api_key:
            raise LLMError("LLM API key not configured. Please set LLM_API_KEY in .env")
        
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=2
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            stream: 是否流式返回
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大 token 数（覆盖默认值）
            **kwargs: 其他参数传递给 API
            
        Returns:
            如果 stream=False，返回完整响应字符串
            如果 stream=True，返回字符串迭代器
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens or self.max_tokens:
            params["max_tokens"] = max_tokens or self.max_tokens
        
        try:
            response = self._client.chat.completions.create(**params)
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            raise LLMError(f"LLM API call failed: {e}")
    
    def _stream_response(self, response) -> Iterator[str]:
        """处理流式响应"""
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        简化的补全接口
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            **kwargs: 其他参数
            
        Returns:
            响应字符串
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, stream=False, **kwargs)


# 便捷函数
def get_llm_client(**kwargs) -> LLMClient:
    """获取 LLM 客户端实例"""
    return LLMClient(**kwargs)
