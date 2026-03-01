"""
文档处理器

负责文档的预处理和切块：
- 清理（去除目录、附录、噪声）
- 结构化切块
- 元数据提取
"""

import os
import re
from typing import List, Dict, Any, Tuple
from enum import Enum

from docmind.core.exceptions import DocumentProcessingError
from .loader import DocumentLoader, load_document


class DocumentProcessor:
    """
    文档处理器
    
    职责：
    - 文档内容清理
    - 结构化切块
    - 元数据提取
    """
    
    def __init__(self, loader: DocumentLoader = None):
        """     
        初始化文档处理器
        
        Args:
            loader: 文档加载器
        """
        self.loader = loader or DocumentLoader()
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        处理文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            切块列表，每个切块包含：
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
        """
        # 加载文档
        raw_content = self.loader.load(file_path)
        
        # 根据文件类型选择处理方式
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        if ext == '.pdf':
            # PDF 使用结构化切块
            chunks = self._process_structured(raw_content, filename)
        else:
            # 其他格式使用简单切块
            chunks = self._process_simple(raw_content, filename)
        
        print(f"Processed {filename}: {len(chunks)} chunks")
        return chunks
    
    def _process_structured(
        self,
        content: str,
        source: str
    ) -> List[Dict[str, Any]]:
        """
        结构化处理（适用于有明确章节结构的文档）
        
        针对《Java开发手册》等规范类文档优化
        """
        # 1. 清理目录和附录
        content = self._remove_toc_and_appendix(content)
        
        # 2. 按行处理
        lines = content.split('\n')
        chunks = []
        
        current_h1 = ""
        current_h2 = ""
        current_h3 = ""
        buffer = []
        
        def flush_buffer():
            if not buffer:
                return
            
            text = "\n".join(buffer).strip()# 这里是将\n加到buffer的每个元素后再将字符串拼接起来 去掉首尾的空字符"elem1\n elem2\n elem3\n"
            if not text:
                buffer.clear()
                return
            
            # 构建标题路径
            title_path = []
            if current_h1:
                title_path.append(current_h1)
            if current_h2:
                title_path.append(current_h2)
            
            full_title = " > ".join(title_path)
            
            # 增强内容（加上标题路径）
            enhanced_content = f"{full_title}\n{text}" if full_title else text
            
            chunks.append({
                "content": enhanced_content,
                "metadata": {
                    "source": source,
                    "h1": current_h1,
                    "h2": current_h2,
                    "h3": current_h3,
                    "raw_content": text
                }
            })
            buffer.clear()#清空列表
        
        for line in lines:
            line = line.rstrip()
            
            # 跳过噪声行 如-----，版本号 页号
            if self._should_skip_line(line):
                continue
            
            # 清理链接
            line = self._clean_links(line)
            
            # 增强标题标记
            line, is_header = self._enhance_headers(line)
            
            # 识别标题层级
            if line.startswith("# "):
                flush_buffer()
                current_h1 = line.lstrip("# ").strip()
                current_h2 = ""
                current_h3 = ""
                continue
            
            if line.startswith("## "):
                flush_buffer()
                current_h2 = line.lstrip("# ").strip()
                current_h3 = ""
                continue
            
            if line.startswith("### "):
                flush_buffer()
                current_h3 = line.lstrip("# ").strip()
                buffer.append(current_h3)
                continue
            
            # 识别规约项（如 "1.【强制】..."）
            if re.match(r'^\d+\.\s*【', line.strip()):
                flush_buffer()
                current_h3 = line.strip()
                buffer.append(current_h3)
                continue
            
            # 普通内容
            buffer.append(line)
        
        # 处理最后的 buffer
        flush_buffer()
        
        return chunks
    
    #定义私有函数
    def _process_simple(
        self,
        content: str,
        source: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        简单切块（按字符数）
        
        适用于没有明确结构的文档
        """
        chunks = []
        
        # 按段落分割
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": {
                            "source": source,
                            "raw_content": current_chunk.strip()
                        }
                    })
                current_chunk = para + "\n\n"
        
        # 添加最后一个 chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    "source": source,
                    "raw_content": current_chunk.strip()
                }
            })
        
        return chunks
    
    def _remove_toc_and_appendix(self, text: str) -> str:
        """删除目录和附录"""
        lines = text.split('\n')
        cleaned = []
        skip_toc = False
        
        # 目录特征：标题后跟大量点和页码
        toc_pattern = r'^.*\.{3,}\s*\d+\s*$'
        
        for line in lines:
            # 检测目录开始
            if line.strip() in ["# 目录", "目录"]:
                skip_toc = True
                continue
            
            # 检测目录项
            if skip_toc and re.match(toc_pattern, line.strip()):
                continue
            
            # 检测目录结束（遇到正文标题）
            if skip_toc and re.match(r'^#*\s*[一二三四五六七八九十]+、', line.strip()):
                skip_toc = False
            
            # 检测附录开始
            if re.match(r'^\s*#*\s*附\s*\d+\s*[:：]', line.strip()):
                break
            if re.match(r'^\s*#*\s*附录\s*\d*\s*[:：]?', line.strip()):
                break
            
            if not skip_toc:
                cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    def _should_skip_line(self, line: str) -> bool:
        """判断是否应跳过该行"""
        line = line.strip()
        
        if not line:#空行不是噪声
            return False
        
        # 页码
        if re.match(r'^\d+/\d+$', line):
            return True
        
        # 文档标题
        if "Java 开发手册" in line or "Java开发手册" in line:
            return True
        
        # 版本名
        if line in ["黄山版", "嵩山版", "泰山版", "华山版"]:
            return True
        
        # 分隔线
        if re.match(r'^[-=_. ]{3,}$', line):
            return True
        
        return False
    
    def _clean_links(self, line: str) -> str:
        """清理 Markdown 链接，保留文字"""
        return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
    
    def _enhance_headers(self, line: str) -> Tuple[str, bool]:
        """增强标题结构识别"""
        line_stripped = line.strip()
        
        # 一级标题：一、编程规约
        if re.match(r'^#*\s*[一二三四五六七八九十]+、', line_stripped):
            clean_text = line_stripped.lstrip('#').strip()
            return f"# {clean_text}", True
        
        # 二级标题：(一) 命名风格
        if re.match(r'^#*\s*\([一二三四五六七八九十]+\)', line_stripped):
            clean_text = line_stripped.lstrip('#').strip()
            return f"## {clean_text}", True
        
        # 三级标题：1.【强制】...
        if re.match(r'^\d+\.\s*【', line_stripped):
            return f"### {line_stripped}", True
        
        return line, line_stripped.startswith('#')#返回处理后的标题 并且标识是否是标题 是则为true


# 便捷函数
def process_document(file_path: str) -> List[Dict[str, Any]]:
    """处理文档的便捷函数"""
    return DocumentProcessor().process(file_path)
