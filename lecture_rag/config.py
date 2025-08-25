"""
설정 관리 모듈
환경변수 및 기본 설정값들을 중앙화해서 관리
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


@dataclass
class Config:
    """애플리케이션 설정"""
    
    # LLM 설정
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    
    # 문서 처리 설정
    code_chunk_size: int = 600
    code_chunk_overlap: int = 80
    text_chunk_size: int = 800
    text_chunk_overlap: int = 120
    
    # 벡터 스토어 설정
    default_store_dir: str = ".lecture_index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 검색 설정
    default_top_k: int = 8
    max_top_k: int = 15
    min_top_k: int = 2
    
    # 프롬프트 설정
    system_prompt: str = (
        "당신은 수업용 코치입니다.\n"
        "제공된 컨텍스트(강의록 조각들)에서 근거를 찾아 답변하십시오.\n"
        "컨텍스트에 관련 내용이 있으면 그것을 기반으로 설명하고 코드를 제공하세요.\n"
        "컨텍스트에 없는 라이브러리나 함수는 사용하지 마세요.\n"
        "질문이 프로그래밍과 전혀 무관하거나(예: 요리, 여행, 운동 등) 컨텍스트와 완전히 무관한 경우에만 '죄송합니다. 해당 내용은 강의록에서 다루지 않은 주제입니다.'라고 거절하세요.\n"
        "프로그래밍 관련 질문이라면 컨텍스트에서 유사한 내용을 찾아 최대한 도움을 주려고 노력하세요.\n"
        "가능하면 강의록의 변수명/함수명/스타일을 모방하세요.\n"
    )
    
    answer_guide: str = (
        "출력 형식:\n"
        "1) 간단한 설명\n"
        "2) 코드(필요 시)\n"
        "3) 사용한 근거 스니펫들(강의일 기준으로 표시)\n"
        "주의: 코드 블록은 반드시 ```python 으로 시작하세요."
    )
    
    @classmethod
    def from_env(cls) -> "Config":
        """환경변수에서 설정을 로드"""
        return cls(
            model_name=os.getenv("LECTURE_RAG_MODEL", cls.model_name),
            temperature=float(os.getenv("LECTURE_RAG_TEMPERATURE", str(cls.temperature)))
        )
    
    def to_env(self):
        """현재 설정을 환경변수로 설정"""
        os.environ["LECTURE_RAG_MODEL"] = self.model_name
        os.environ["LECTURE_RAG_TEMPERATURE"] = str(self.temperature)