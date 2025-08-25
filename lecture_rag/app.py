"""
Streamlit 기반 Lecture-RAG 웹 애플리케이션
강의록을 인덱싱하고 질의응답을 제공하는 웹 인터페이스
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import streamlit as st

from .config import Config
from .vector_store import VectorStore
from .llm_handler import LLMHandler
from .google_drive import GoogleDriveClient, is_google_drive_available


class LectureRAGApp:
    """Lecture-RAG"""
    
    def __init__(self):
        self.config = Config.from_env()
        self._setup_page()
    
    def _setup_page(self):
        """페이지 기본 설정"""
        st.set_page_config(
            page_title="Lecture-RAG", 
            page_icon="📚", 
            layout="wide"
        )
        st.title("Lecture-RAG")
    
    def _render_sidebar(self) -> tuple[VectorStore, str, float]:
        """사이드바 렌더링"""
        with st.sidebar:
            st.header("설정")
            
            # 벡터 스토어 디렉토리 설정
            store_dir_str = st.text_input(
                "FAISS 저장 디렉터리", 
                value=self.config.default_store_dir
            )
            store_dir = Path(store_dir_str)
            vector_store = VectorStore(store_dir)
            
            # LLM 모델 설정
            available_models = [
                "gpt-3.5-turbo",
                "gpt-4o-mini",
                "gpt-4o",
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]
            
            # 현재 설정된 모델이 목록에 없으면 추가
            if self.config.model_name not in available_models:
                available_models.insert(0, self.config.model_name)
            
            model = st.selectbox(
                "LLM 모델", 
                options=available_models,
                index=available_models.index(self.config.model_name)
            )
            temp = st.slider(
                "Temperature", 
                0.0, 1.0, 
                self.config.temperature, 
                0.05
            )
            
            st.caption("※ OpenAI 사용 시 환경변수 OPENAI_API_KEY가 필요합니다.")
            
            # 인덱싱 섹션
            self._render_indexing_section(vector_store, model, temp)
            
            return vector_store, model, temp
    
    def _render_indexing_section(
        self, 
        vector_store: VectorStore, 
        model: str, 
        temp: float
    ):
        """인덱싱 섹션 렌더링"""
        st.divider()
        st.subheader("인덱싱")
        
        # Google Drive에서 가져오기 버튼 (상단에 배치)
        if is_google_drive_available():
            if st.button("📥 구글드라이브에서 강의록 가져오기", 
                        help="SKN15 폴더의 강의록.txt를 자동으로 다운로드합니다",
                        use_container_width=True):
                self._handle_google_drive_download()
            st.divider()  # 구분선 추가
        else:
            st.info("💡 Google Drive 연동을 위해 `pip install google-api-python-client google-auth-oauthlib` 설치")
            st.divider()
        
        # 파일 업로드 및 경로 입력
        st.write("**또는 직접 업로드:**")
        upload = st.file_uploader(
            "강의록 파일 업로드(.txt, .md 등)", 
            type=["txt", "md", "py", "mdx"], 
            accept_multiple_files=False
        )
        
        st.write("**또는 로컬 파일 경로:**")
        manual_path = st.text_input("파일 경로를 입력하세요", value="강의록.txt")

        if st.button("인덱싱 실행", type="primary"):
            self._handle_indexing(vector_store, upload, manual_path, model, temp)
    
    def _handle_indexing(
        self, 
        vector_store: VectorStore, 
        upload: Optional[st.runtime.uploaded_file_manager.UploadedFile], 
        manual_path: str, 
        model: str, 
        temp: float
    ):
        """인덱싱 처리"""
        # 파일 경로 결정
        if upload is not None:
            tmp_path = vector_store.store_dir / "uploaded_lecture.txt"
            tmp_path.write_bytes(upload.read())
            path_to_index = tmp_path
        else:
            path_to_index = Path(manual_path)

        if not path_to_index.exists():
            st.error(f"강의록 파일을 찾을 수 없습니다: {path_to_index}")
            return
        
        # 환경 설정 업데이트
        config = Config(model_name=model, temperature=temp)
        config.to_env()
        
        # 인덱싱 실행
        with st.spinner("인덱싱 중..."):
            n_docs, allowed = vector_store.index_document(path_to_index)
        
        st.success(f"인덱싱 완료! 문서 조각 {n_docs}개 생성")
        with st.expander("허용 토큰(모듈/심볼) 보기"):
            st.json(allowed)
    
    def _handle_google_drive_download(self):
        """Google Drive에서 강의록 다운로드 처리"""
        try:
            with st.spinner("Google Drive 연결 중..."):
                drive_client = GoogleDriveClient()
                
                # 인증 확인
                auth_result = drive_client.authenticate()
                
                if auth_result is True:
                    # 이미 인증됨 - 바로 다운로드
                    with st.spinner("SKN15/강의록.txt 다운로드 중..."):
                        success = drive_client.download_lecture_file(Path("강의록.txt"))
                        
                    if success:
                        st.success("✅ 구글드라이브에서 강의록.txt를 성공적으로 다운로드했습니다!")
                        st.info("이제 '인덱싱 실행' 버튼을 클릭하여 인덱싱을 진행하세요.")
                    else:
                        st.error("❌ 파일 다운로드에 실패했습니다.")
                        
                elif isinstance(auth_result, tuple):
                    # 인증 필요
                    auth_success, auth_url = auth_result
                    
                    st.warning("🔐 Google Drive 인증이 필요합니다.")
                    st.markdown(f"**1단계:** [여기를 클릭하여 인증하세요]({auth_url})")
                    
                    with st.expander("📋 인증 과정 안내", expanded=True):
                        st.write("1. 위 링크 클릭 → Google 로그인")
                        st.write("2. 앱 권한 승인")
                        st.write("3. ✨ **브라우저에 표시되는 코드를 복사**")
                        st.write("4. 아래 입력란에 붙여넣기")
                        st.code("예시: 4/0AZEOvhX-abcd1234efgh5678...")
                    
                    # 인증 코드 입력란
                    auth_code = st.text_input(
                        "**2단계:** 인증 코드를 여기에 붙여넣으세요:",
                        help="브라우저에 'Please copy this code...' 라고 나오는 긴 코드를 복사해서 붙여넣으세요",
                        placeholder="4/0AZEOvhX-..."
                    )
                    
                    if auth_code and st.button("인증 완료"):
                        try:
                            if drive_client.complete_auth(auth_code):
                                st.success("✅ 인증 완료! 다시 다운로드 버튼을 클릭하세요.")
                                st.rerun()
                            else:
                                st.error("❌ 인증에 실패했습니다.")
                        except Exception as e:
                            st.error(f"❌ 인증 처리 중 오류: {str(e)}")
                            
        except Exception as e:
            st.error(f"❌ Google Drive 연결 실패: {str(e)}")
            st.info("💡 클라이언트 ID가 올바르게 설정되었는지 확인하세요.")
    
    def _render_qa_section(
        self, 
        vector_store: VectorStore, 
        model: str, 
        temp: float
    ):
        """질의응답 섹션 렌더링"""
        st.divider()
        
        # 질문 입력 및 옵션 설정
        colQ, colOpt = st.columns([2, 1], vertical_alignment="top")
        
        with colQ:
            st.subheader("질의응답")
            query = st.text_input(
                "질문을 입력하세요", 
                placeholder="예) RAG에 대해 알려줘"
            )
        
        with colOpt:
            topk = st.slider(
                "Top-K 문서", 
                min_value=self.config.min_top_k, 
                max_value=self.config.max_top_k, 
                value=self.config.default_top_k, 
                step=1
            )
        
        ask = st.button("답변 생성", type="primary")
        
        if ask:
            self._handle_qa(vector_store, query, topk, model, temp)
    
    def _handle_qa(
        self, 
        vector_store: VectorStore, 
        query: str, 
        topk: int, 
        model: str, 
        temp: float
    ):
        """질의응답 처리"""
        if not query.strip():
            st.warning("질문을 입력하세요.")
            return
        
        # 환경 설정 업데이트
        config = Config(model_name=model, temperature=temp)
        config.to_env()
        
        # 문서 검색
        with st.spinner("검색 중..."):
            try:
                docs, allowed = vector_store.search(query, k=topk)
            except Exception as e:
                st.error(f"인덱스를 로드할 수 없습니다. 먼저 인덱싱을 수행하세요. 상세: {e}")
                return

        if not docs:
            st.error("검색 결과가 없습니다. 사이드바에서 인덱싱을 먼저 수행했는지 확인하세요.")
            return
        
        
        # LLM 답변 생성
        llm_handler = LLMHandler(config)
        with st.spinner("LLM 응답 생성 중..."):
            answer = llm_handler.generate_answer(query, docs, allowed)
        
        # 미허용 토큰 로그 (디버깅용)
        unknown_tokens = llm_handler._check_unknown_tokens(answer, allowed)
        if unknown_tokens:
            with st.expander("미허용 토큰 감지 로그", expanded=False):
                st.write(", ".join(sorted(set(unknown_tokens))))
        
        # 결과 표시
        st.subheader("답변")
        st.markdown(answer)
        
        # 거절 응답인지 확인하여 스니펫 표시 여부 결정
        rejection_keywords = [
            "강의록에서 다루지 않은 주제",
            "강의록에 없는 내용",
            "제공된 컨텍스트에서 찾을 수 없",
            "강의록에서 관련 내용을 찾을 수 없"
        ]
        
        is_rejection = any(keyword in answer for keyword in rejection_keywords)
        
        if not is_rejection:
            self._render_evidence_snippets(docs)
        else:
            st.info("💡 관련 내용이 강의록에 없어 근거 스니펫을 표시하지 않습니다.")
    
    def _render_evidence_snippets(self, docs):
        """근거 스니펫들을 렌더링"""
        st.subheader("근거 스니펫")
        
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            location_info = f"라인 {meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
            first_line = meta.get('first_line_preview', '')
            
            # 날짜 기반 청킹인 경우 날짜 표시
            if meta.get('kind') == 'lecture_date':
                date = meta.get('date', 'Unknown')
                title = f"📅 {date} | {location_info} | {first_line}"
            else:
                title = f"Chunk {i} | {meta.get('kind')} | {location_info} | 시작: {first_line}"
            
            with st.expander(title, expanded=False):
                # 위치 정보 표시
                if meta.get('kind') == 'lecture_date':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📅 강의일: {meta.get('date', 'Unknown')}")
                    with col2:
                        st.caption(f"📍 위치: {location_info}")
                    with col3:
                        st.caption(f"📏 줄 수: {meta.get('line_count', '?')}줄")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📍 위치: {location_info}")
                    with col2:
                        st.caption(f"📏 줄 수: {meta.get('line_count', '?')}줄")
                    with col3:
                        st.caption(f"🏷️ 타입: {meta.get('kind')}")
                
                # 내용 표시
                snippet = d.page_content
                language = "python" if meta.get('kind') == 'code' else "text"
                st.code(snippet, language=language)
                
                # 원본 파일에서 찾는 방법 안내
                st.info(f"💡 원본에서 찾기: '{meta.get('first_line_preview', '')}' 검색하여 {location_info} 확인")
    
    def run(self):
        """애플리케이션 실행"""
        # URL 파라미터에서 Google OAuth 코드 자동 처리
        self._auto_handle_oauth_callback()
        
        # 사이드바 렌더링
        vector_store, model, temp = self._render_sidebar()
        
        # 질의응답 섹션 렌더링
        self._render_qa_section(vector_store, model, temp)
    
    def _auto_handle_oauth_callback(self):
        """OAuth 콜백을 자동으로 처리"""
        query_params = st.query_params
        auth_code = query_params.get("code")
        
        if auth_code:
            # 인증 코드가 URL에 있으면 자동 처리
            st.info("🔄 Google Drive 인증을 처리하고 있습니다...")
            
            try:
                drive_client = GoogleDriveClient()
                if drive_client.complete_auth(auth_code):
                    st.success("✅ Google Drive 인증이 완료되었습니다!")
                    st.info("이제 '📥 구글드라이브에서 강의록 가져오기' 버튼을 사용할 수 있습니다.")
                    
                    # URL 파라미터 제거하고 새로고침
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error("❌ 인증 처리에 실패했습니다.")
                    
            except Exception as e:
                st.error(f"❌ 인증 처리 중 오류 발생: {str(e)}")
                
            # URL 파라미터 제거
            st.query_params.clear()


def main():
    """메인 함수"""
    app = LectureRAGApp()
    app.run()


if __name__ == "__main__":
    main()