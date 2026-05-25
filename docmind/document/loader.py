import os
import time
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from docmind.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """文档加载器基类"""
    
    @abstractmethod
    def load(self, file_path: str) -> str:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档内容（Markdown 格式）
        """
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名"""
        pass


class PDFLoader(BaseLoader):
    """PDF 文档加载器"""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]
    
    def load(self, file_path: str) -> str:
        """使用 pymupdf4llm 将 PDF 转换为 Markdown"""
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(file_path)
        except ImportError:
            raise DocumentProcessingError(
                "pymupdf4llm not installed. Install with: pip install pymupdf4llm"
            )
        except Exception as e:
            raise DocumentProcessingError(f"Failed to load PDF: {e}")


class DocxLoader(BaseLoader):
    """DOCX 文档加载器"""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".docx", ".doc"]
    
    def load(self, file_path: str) -> str:
        """使用 python-docx 加载 DOCX"""
        try:
            import docx
            doc = docx.Document(file_path)
            
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    if para.style.name.startswith('Heading'):
                        level = int(para.style.name[-1]) if para.style.name[-1].isdigit() else 1
                        paragraphs.append(f"{'#' * level} {text}")
                    else:
                        paragraphs.append(text)
            
            return "\n\n".join(paragraphs)
            
        except ImportError:
            raise DocumentProcessingError(
                "python-docx not installed. Install with: pip install python-docx"
            )
        except Exception as e:
            raise DocumentProcessingError(f"Failed to load DOCX: {e}")


class MarkdownLoader(BaseLoader):
    """Markdown 文档加载器"""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]
    
    def load(self, file_path: str) -> str:
        """直接读取 Markdown 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise DocumentProcessingError(f"Failed to load Markdown: {e}")


class TextLoader(BaseLoader):
    """纯文本加载器"""
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".txt", ".text"]
    
    def load(self, file_path: str) -> str:
        """读取纯文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise DocumentProcessingError(f"Failed to load text file: {e}")


class DocumentLoader:
    """
    统一文档加载器
    
    自动根据文件扩展名选择合适的加载器
    """
    
    def __init__(self):
        """初始化加载器注册表"""
        self.loaders: Dict[str, BaseLoader] = {}
        
        # 注册默认加载器
        for loader in [PDFLoader(), DocxLoader(), MarkdownLoader(), TextLoader()]:
            for ext in loader.supported_extensions:
                self.loaders[ext.lower()] = loader
    
    def load(self, file_path: str) -> str:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档内容（Markdown 格式）
        """
        start_time = time.time()
        logger.info(f"[DocumentLoader] Start loading: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"[DocumentLoader] File not found: {file_path}")
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.loaders:
            logger.error(
                f"[DocumentLoader] Unsupported file type: {ext}. "
                f"Supported: {list(self.loaders.keys())}"
            )
            raise DocumentProcessingError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(self.loaders.keys())}"
            )
        
        result = self.loaders[ext].load(file_path)
        elapsed = time.time() - start_time
        logger.info(
            f"[DocumentLoader] Finished loading: {file_path} (elapsed: {elapsed:.2f}s)"
        )
        return result
    
    def register_loader(self, loader: BaseLoader):
        """注册自定义加载器"""
        for ext in loader.supported_extensions:
            self.loaders[ext.lower()] = loader


# 便捷函数
def load_document(file_path: str) -> str:
    """加载文档的便捷函数"""
    return DocumentLoader().load(file_path)
