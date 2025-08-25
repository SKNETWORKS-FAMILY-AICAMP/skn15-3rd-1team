"""
Lecture-RAG 패키지
강의록 기반 검색 증강 생성(RAG) 시스템
"""

__version__ = "1.0.0"

from .config import Config
from .utils import read_text, detect_code_blocks, extract_allowed_tokens, find_unknown_tokens
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_handler import LLMHandler

__all__ = [
    "Config",
    "read_text",
    "detect_code_blocks", 
    "extract_allowed_tokens",
    "find_unknown_tokens",
    "DocumentProcessor",
    "VectorStore", 
    "LLMHandler"
]