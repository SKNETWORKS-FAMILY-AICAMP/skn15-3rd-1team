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
        텍스트를 청킹 (파일에 따라 다른 전략 사용)
        
        Args:
            text: 원본 텍스트
            source: 소스 파일명/경로
            
        Returns:
            List[Document]: 청크된 Document 리스트
        """
        from pathlib import Path
        
        # 파일 확인
        file_path = Path(source)
        
        # 강의록.txt 파일만 날짜 기반 청킹
        if file_path.name == "강의록.txt":
            return self._chunk_by_date(text, source)
        else:
            # 그 외 파일은 내용 기반 청킹
            return self._chunk_by_content(text, source)
    
    def _chunk_by_date(self, text: str, source: str) -> List[Document]:
        """날짜 기반 청킹 (강의록.txt 전용)"""
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
    
    def _chunk_by_content(self, text: str, source: str) -> List[Document]:
        """내용 기반 청킹 (.py, .md 등 일반 파일용)"""
        from pathlib import Path
        
        # 코드 블록 감지
        blocks = detect_code_blocks(text)
        docs: List[Document] = []
        chunk_idx = 0
        
        for block_type, content in blocks:
            if not content.strip():
                continue
                
            # 적절한 스플리터 선택
            if block_type == "code":
                chunks = self.code_splitter.split_text(content)
            else:
                chunks = self.text_splitter.split_text(content)
            
            # 각 청크를 Document로 변환
            for chunk in chunks:
                if chunk.strip():
                    # 첫 줄과 마지막 줄 추출
                    lines = chunk.splitlines()
                    first_line = lines[0] if lines else ""
                    last_line = lines[-1] if lines else ""
                    
                    # 미리보기 생성
                    first_line_preview = first_line[:50] + "..." if len(first_line) > 50 else first_line
                    
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "kind": block_type,
                            "chunk_id": f"{source}:{block_type}_{chunk_idx}",
                            "start_line": 1,  # 추정값
                            "end_line": len(lines),  # 추정값
                            "line_count": len(lines),
                            "first_line_preview": first_line_preview,
                            "last_line_preview": last_line[:50] + "..." if len(last_line) > 50 else last_line
                        }
                    ))
                    chunk_idx += 1
        
        return docs