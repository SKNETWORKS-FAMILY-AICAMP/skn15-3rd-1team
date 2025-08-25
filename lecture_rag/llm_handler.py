"""
LLM 프롬프팅 및 호출 로직
OpenAI GPT를 사용한 질의응답 및 코드 검증 기능
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .utils import find_unknown_tokens
from .config import Config


class LLMHandler:
    """LLM 호출 및 응답 처리 클래스"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self.config = config
        
    def _create_llm(self) -> ChatOpenAI:
        """LLM 인스턴스 생성"""
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature
        )
    
    def _make_context_block(self, docs: List[Document]) -> str:
        """검색된 문서들을 컨텍스트 블록으로 구성"""
        parts = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            location_info = f"라인 {meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
            
            # 날짜 기반 청킹인 경우 날짜 표시
            if meta.get('kind') == 'lecture_date':
                date = meta.get('date', 'Unknown')
                chunk_info = f"[강의일: {date} | {location_info}]"
            else:
                chunk_info = f"[Chunk {i} | {meta.get('kind')} | {location_info} | {meta.get('chunk_id')}]"
            
            parts.append(f"{chunk_info}\n{d.page_content}")
        return "\n\n".join(parts)
    
    def _create_style_hint(self, allowed: Dict[str, Any]) -> str:
        """허용된 토큰들을 스타일 힌트로 구성"""
        return (
            "허용 모듈: " + ", ".join(allowed.get("modules", [])) + "\n"
            "허용 심볼: " + ", ".join(allowed.get("symbols", [])) + "\n"
        )
    
    def generate_answer(
        self, 
        query: str, 
        docs: List[Document], 
        allowed: Dict[str, Any]
    ) -> str:
        """
        질문에 대한 답변 생성 (토큰 검증 포함)
        
        Args:
            query: 사용자 질문
            docs: 검색된 컨텍스트 문서들
            allowed: 허용된 토큰들
            
        Returns:
            str: 생성된 답변
        """
        llm = self._create_llm()
        
        # 컨텍스트 구성
        context = self._make_context_block(docs)
        
        style_hint = self._create_style_hint(allowed)
        
        # 1차 답변 생성
        messages = [
            ("system", self.config.system_prompt),
            ("user", f"질문: {query}\n\n{self.config.answer_guide}\n\n[컨텍스트 시작]\n{context}\n[컨텍스트 끝]\n\n{style_hint}\n")
        ]
        
        resp = llm.invoke(messages)
        answer = resp.content
        
        # 미허용 토큰 검사 및 재생성
        unknown_tokens = self._check_unknown_tokens(answer, allowed)
        
        if unknown_tokens:
            # 미허용 토큰이 발견되면 제약조건 추가하여 재생성
            retry_prompt = (
                query + 
                "\n(주의: 아래 미허용 목록을 절대 사용하지 마세요: " + 
                ", ".join(sorted(set(unknown_tokens))) + 
                ")"
            )
            
            retry_messages = [
                ("system", self.config.system_prompt),
                ("user", f"질문: {retry_prompt}\n\n{self.config.answer_guide}\n\n[컨텍스트 시작]\n{context}\n[컨텍스트 끝]\n\n{style_hint}\n")
            ]
            
            retry_resp = llm.invoke(retry_messages)
            answer = retry_resp.content
        
        return answer
    
    def _check_unknown_tokens(self, answer: str, allowed: Dict[str, Any]) -> List[str]:
        """답변에서 미허용 토큰들을 검사"""
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
        unknown_total: List[str] = []
        
        for cb in code_blocks:
            unknown_total.extend(find_unknown_tokens(cb, allowed))
        
        return unknown_total