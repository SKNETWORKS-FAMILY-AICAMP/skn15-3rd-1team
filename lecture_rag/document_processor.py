"""
문서 처리 및 청킹 로직
강의록 텍스트를 벡터화하기 위한 청크로 분할하는 기능
"""
from __future__ import annotations
from typing import List
from pathlib import Path
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .utils import read_text, detect_code_blocks


class DocumentProcessor:
    """문서를 청크 단위로 분할하는 프로세서"""
    
    def __init__(
        self, 
        code_chunk_size: int = 600,
        code_chunk_overlap: int = 80,
        text_chunk_size: int = 800,
        text_chunk_overlap: int = 120
    ):
        """
        Args:
            code_chunk_size: 코드 청크 크기
            code_chunk_overlap: 코드 청크 오버랩
            text_chunk_size: 텍스트 청크 크기  
            text_chunk_overlap: 텍스트 청크 오버랩
        """
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=code_chunk_size, 
            chunk_overlap=code_chunk_overlap, 
            separators=["\n\n", "\n", ")\n", ":\n", ",\n"]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_chunk_size, 
            chunk_overlap=text_chunk_overlap, 
            separators=["\n\n", "\n", ". "]
        )
    
    def process_file(self, file_path: Path) -> List[Document]:
        """
        파일을 읽어서 Document 청크로 변환
        
        Args:
            file_path: 처리할 파일 경로
            
        Returns:
            List[Document]: 청크된 Document 리스트
        """
        text = read_text(file_path)
        return self.chunk_documents(text, source=str(file_path))
    
    def chunk_documents(self, text: str, source: str) -> List[Document]:
        """
        텍스트를 날짜 기반으로 청킹 (YYYY.MM.DD 패턴으로 분할)
        
        Args:
            text: 원본 텍스트
            source: 소스 파일명/경로
            
        Returns:
            List[Document]: 날짜별로 청크된 Document 리스트
        """
        # 날짜 패턴으로 분할 (YYYY.MM.DD)
        date_pattern = r'^(\d{4}\.\d{2}\.\d{2})\s*'
        
        # 전체 텍스트를 줄 단위로 분할
        lines = text.splitlines()
        docs: List[Document] = []
        
        current_chunk_lines = []
        current_date = None
        current_start_line = 1
        chunk_idx = 0
        
        for line_idx, line in enumerate(lines, 1):
            date_match = re.match(date_pattern, line.strip())
            
            if date_match:
                # 이전 청크가 있다면 저장
                if current_chunk_lines and current_date:
                    chunk_content = '\n'.join(current_chunk_lines)
                    if chunk_content.strip():  # 빈 청크 제외
                        docs.append(self._create_date_chunk_document(
                            chunk_content, source, current_date, current_start_line, 
                            line_idx - 1, chunk_idx
                        ))
                        chunk_idx += 1
                
                # 새 청크 시작
                current_date = date_match.group(1)
                current_start_line = line_idx
                current_chunk_lines = [line]
            else:
                # 현재 청크에 라인 추가
                current_chunk_lines.append(line)
        
        # 마지막 청크 처리
        if current_chunk_lines and current_date:
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_content.strip():
                docs.append(self._create_date_chunk_document(
                    chunk_content, source, current_date, current_start_line, 
                    len(lines), chunk_idx
                ))
        
        return docs
    
    def _create_date_chunk_document(
        self, content: str, source: str, date: str, 
        start_line: int, end_line: int, chunk_idx: int
    ) -> Document:
        """날짜 기반 청크용 Document 생성"""
        lines = content.splitlines()
        first_line = lines[0] if lines else ""
        last_line = lines[-1] if lines else ""
        
        # 첫 줄에서 날짜 부분 제거한 미리보기
        first_line_preview = re.sub(r'^\d{4}\.\d{2}\.\d{2}\s*', '', first_line)
        first_line_preview = first_line_preview[:50] + "..." if len(first_line_preview) > 50 else first_line_preview
        
        return Document(
            page_content=content,
            metadata={
                "source": source,
                "kind": "lecture_date",
                "date": date,
                "chunk_id": f"{source}:date_{chunk_idx}",
                "start_line": start_line,
                "end_line": end_line,
                "line_count": len(lines),
                "first_line_preview": first_line_preview,
                "last_line_preview": last_line[:50] + "..." if len(last_line) > 50 else last_line
            }
        )