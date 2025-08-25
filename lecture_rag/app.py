"""
Streamlit 기반 Lecture-RAG 웹 애플리케이션
강의록을 인덱싱하고 질의응답을 제공하는 웹 인터페이스
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

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
        self._init_session_state()
    
    def _setup_page(self):
        """페이지 기본 설정"""
        st.set_page_config(
            page_title="Lecture-RAG", 
            page_icon="📚", 
            layout="wide"
        )
        st.title("Lecture-RAG")
        
        # 채팅 스타일 CSS 추가
        st.markdown("""
        <style>
        .chat-container {
            max-width: 800px;
            margin: 0;
            padding: 20px 0;
        }
        
        /* 사용자 메시지 (오른쪽) */
        .user-message-container {
            display: flex;
            justify-content: flex-end;
            margin: 15px 0;
        }
        .user-message {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
            box-shadow: 0 2px 12px rgba(79, 70, 229, 0.3);
            margin-left: 20px;
        }
        .user-label {
            font-size: 0.75em;
            opacity: 0.8;
            margin-bottom: 4px;
            text-align: right;
        }
        .user-content {
            font-size: 15px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        /* AI 메시지 (왼쪽) */
        .ai-message-container {
            display: flex;
            justify-content: flex-start;
            margin: 15px 0;
            padding-left: 0;
        }
        .ai-avatar {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
            margin-right: 12px;
            flex-shrink: 0;
        }
        .ai-message {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 14px 18px;
            border-radius: 18px 18px 18px 4px;
            max-width: 70%;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            position: relative;
        }
        .ai-label {
            font-size: 0.75em;
            color: #10b981;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .ai-content {
            font-size: 15px;
            line-height: 1.5;
            color: #374151;
            word-wrap: break-word;
        }
        .click-hint {
            font-size: 0.7em;
            color: #9ca3af;
            margin-top: 8px;
            font-style: italic;
        }
        
        /* 버튼 스타일 개선 */
        .stButton > button {
            width: 100% !important;
            background: transparent !important;
            border: 2px solid #000000 !important;
            border-radius: 18px !important;
            padding: 0 !important;
            margin: 0 !important;
            text-align: left !important;
            color: #000000 !important;
        }
        .stButton > button:hover {
            background: transparent !important;
            border-color: #000000 !important;
        }
        .ai-message {
            background: #ffffff;
            border: 2px solid #000000;
            padding: 14px 18px;
            border-radius: 18px 18px 18px 4px;
            max-width: 70%;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            position: relative;
        }
        .ai-message:hover {
            border-color: #000000;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .ai-content {
            font-size: 15px;
            line-height: 1.5;
            color: #000000;
            word-wrap: break-word;
        }
        
        /* 입력창 스타일 */
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid #e5e7eb;
            padding: 12px 20px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
    
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
    
    def _init_session_state(self):
        """세션 상태 초기화"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "detailed_view" not in st.session_state:
            st.session_state.detailed_view = None  # 상세 보기할 메시지 ID
        if "message_counter" not in st.session_state:
            st.session_state.message_counter = 0
    
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
        
        # 상세 보기가 활성화된 경우 상세 화면 렌더링
        if st.session_state.detailed_view is not None:
            self._render_detailed_view()
            return
        
        # 채팅형 인터페이스 렌더링
        self._render_chat_interface(vector_store, model, temp)
    
    def _render_chat_interface(
        self, 
        vector_store: VectorStore, 
        model: str, 
        temp: float
    ):
        """채팅형 인터페이스 렌더링"""
        st.subheader("💬 채팅")
        
        # 옵션 설정
        with st.expander("⚙️ 옵션 설정", expanded=False):
            topk = st.slider(
                "Top-K 문서", 
                min_value=self.config.min_top_k, 
                max_value=self.config.max_top_k, 
                value=self.config.default_top_k, 
                step=1
            )
            st.session_state.topk = topk
        
        # 메시지 히스토리 표시
        if st.session_state.messages:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                self._render_message(message)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '''
                <div style="text-align: center; padding: 40px 0; color: #9ca3af;">
                    <h3>안녕하세요! AI웅이에요 🤖</h3>
                    <p>강의록에 대해 궁금한 것을 물어보세요!</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
        
        # 질문 입력
        query = st.text_input(
            "", 
            placeholder="질문을 입력하세요",
            key=f"chat_input_{st.session_state.message_counter}"
        )
        
        # 전송 버튼
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            send_button = st.button("📤 전송", type="primary")
        with col2:
            clear_button = st.button("🗑️ 초기화")
        
        if clear_button:
            st.session_state.messages = []
            st.session_state.message_counter = 0
            st.rerun()
        
        if send_button and query.strip():
            self._handle_chat_qa(vector_store, query, model, temp)
    
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
    
    def _render_message(self, message: Dict[str, Any]):
        """개별 메시지 렌더링"""
        if message["role"] == "user":
            # 사용자 메시지 (오른쪽)
            st.markdown(
                f'''
                <div class="user-message-container">
                    <div class="user-message">
                        <div class="user-label">You</div>
                        <div class="user-content">{message["content"]}</div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:  # assistant
            # 요약 답변 비공인지 확인
            has_summary = "summary" in message and message["summary"].strip()
            
            if has_summary:
                # AI 메시지 (왼쪽) - 클릭 가능한 버튼으로 표시
                col1, col2 = st.columns([0.5, 11])
                
                with col1:
                    st.markdown(
                        '<div class="ai-avatar">웅</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    if st.button(
                        f"**AI웅**\n\n{message['summary']}\n\n*📋 클릭하여 상세 답변 및 근거 스니펫 보기*",
                        key=f"message_{message['id']}",
                        help="클릭하여 상세 답변 보기",
                        use_container_width=True
                    ):
                        st.session_state.detailed_view = message["id"]
                        st.rerun()
            else:
                # 요약이 없으면 전체 답변 표시
                col1, col2 = st.columns([0.5, 11])
                
                with col1:
                    st.markdown(
                        '<div class="ai-avatar">웅</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f'''
                        <div class="ai-message">
                            <div class="ai-label">AI웅</div>
                            <div class="ai-content">{message["content"]}</div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
    
    
    def _handle_chat_qa(
        self, 
        vector_store: VectorStore, 
        query: str, 
        model: str, 
        temp: float
    ):
        """채팅형 질의응답 처리"""
        # 사용자 메시지 추가
        user_message = {
            "role": "user",
            "content": query,
            "id": st.session_state.message_counter
        }
        st.session_state.messages.append(user_message)
        st.session_state.message_counter += 1
        
        # 입력 필드는 자동으로 초기화됨 (직접 수정하지 않음)
        
        # 답변 생성
        topk = st.session_state.get('topk', self.config.default_top_k)
        
        # 로딩 메시지 임시 표시
        with st.empty():
            col1, col2 = st.columns([0.5, 11])
            with col1:
                st.markdown('<div class="ai-avatar">웅</div>', unsafe_allow_html=True)
            with col2:
                with st.spinner("AI웅이 답변을 생성하고 있어요..."):
                    try:
                        # 기존 QA 로직 사용
                        config = Config(model_name=model, temperature=temp)
                        config.to_env()
                        
                        # 문서 검색
                        docs, allowed = vector_store.search(query, k=topk)
                        
                        if not docs:
                            assistant_message = {
                                "role": "assistant",
                                "content": "검색 결과가 없습니다. 인덱싱을 먼저 수행해주세요.",
                                "id": st.session_state.message_counter,
                                "summary": ""
                            }
                        else:
                            # LLM 답변 생성
                            llm_handler = LLMHandler(config)
                            full_answer = llm_handler.generate_answer(query, docs, allowed)
                            
                            # 요약 답변 생성
                            summary_answer = self._generate_summary_answer(llm_handler, query, full_answer)
                            
                            assistant_message = {
                                "role": "assistant",
                                "content": full_answer,
                                "summary": summary_answer,
                                "id": st.session_state.message_counter,
                                "query": query,
                                "docs": docs,
                                "allowed": allowed,
                                "unknown_tokens": llm_handler._check_unknown_tokens(full_answer, allowed)
                            }
                        
                        # 답변 메시지 추가
                        st.session_state.messages.append(assistant_message)
                        st.session_state.message_counter += 1
                        
                    except Exception as e:
                        error_message = {
                            "role": "assistant",
                            "content": f"오류가 발생했습니다: {str(e)}",
                            "id": st.session_state.message_counter,
                            "summary": ""
                        }
                        st.session_state.messages.append(error_message)
                        st.session_state.message_counter += 1
        
        st.rerun()
    
    def _generate_summary_answer(self, llm_handler: LLMHandler, query: str, full_answer: str) -> str:
        """요약 답변 생성"""
        try:
            llm = llm_handler._create_llm()
            summary_prompt = f"""
            다음 질문과 답변을 바탕으로 2-3문장으로 간단히 요약해주세요.
            
            질문: {query}
            
            원본 답변:
            {full_answer}
            
            요약 (간단하고 핵심로운 2-3문장):
            """
            
            response = llm.invoke([("user", summary_prompt)])
            return response.content.strip()
        except:
            # 요약 생성 실패 시 기본 메시지
            return "답변이 준비되었습니다. 클릭하여 상세 답변을 확인하세요."
    
    def _render_detailed_view(self):
        """상세 보기 화면 렌더링"""
        message_id = st.session_state.detailed_view
        
        # 해당 메시지 찾기
        target_message = None
        for msg in st.session_state.messages:
            if msg.get("id") == message_id:
                target_message = msg
                break
        
        if not target_message:
            st.error("메시지를 찾을 수 없습니다.")
            st.session_state.detailed_view = None
            st.rerun()
            return
        
        # 뒤로가기 버튼
        if st.button("⬅️ 채팅으로 돌아가기"):
            st.session_state.detailed_view = None
            st.rerun()
            return
        
        st.divider()
        
        # 질문 표시
        st.subheader("🙋‍♂️ 질문")
        st.markdown(f"**{target_message.get('query', '질문 정보 없음')}**")
        
        # 상세 답변 표시
        st.subheader("🤖 상세 답변")
        st.markdown(target_message.get("content", "답변 정보 없음"))
        
        # 미허용 토큰 로그
        unknown_tokens = target_message.get("unknown_tokens", [])
        if unknown_tokens:
            with st.expander("미허용 토큰 감지 로그", expanded=False):
                st.write(", ".join(sorted(set(unknown_tokens))))
        
        # 근거 스니펫 표시
        docs = target_message.get("docs", [])
        if docs:
            # 거절 응답인지 확인
            content = target_message.get("content", "")
            rejection_keywords = [
                "강의록에서 다루지 않은 주제",
                "강의록에 없는 내용",
                "제공된 컨텍스트에서 찾을 수 없",
                "강의록에서 관련 내용을 찾을 수 없"
            ]
            
            is_rejection = any(keyword in content for keyword in rejection_keywords)
            
            if not is_rejection:
                self._render_evidence_snippets(docs)
            else:
                st.info("💡 관련 내용이 강의록에 없어 근거 스니펫을 표시하지 않습니다.")
    
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