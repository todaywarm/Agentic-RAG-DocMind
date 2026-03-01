"""
Generation 模块

提供答案生成能力
"""

from .answer_generator import AnswerGenerator, generate_answer

__all__ = [
    "AnswerGenerator",
    "generate_answer",
]
