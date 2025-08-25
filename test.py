# lecture_streamlit.py
# Streamlit + LangChain 기반 Lecture-RAG 웹앱
# ================================================
# 
# [기술 스택]
# - Frontend: Streamlit (웹 UI 프레임워크)
# - LLM: OpenAI GPT (langchain-openai)
# - Vector DB: FAISS (Facebook AI Similarity Search)
# - Embeddings: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
# - Text Processing: LangChain RecursiveCharacterTextSplitter
# - Language: Python 3.8+
# 
# [주요 기능]
# - 강의록 문서 인덱싱 (벡터 임베딩 생성 및 저장)
# - 의미적 유사도 검색 (질문과 유사한 문서 조각 검색)
# - 컨텍스트 기반 LLM 답변 생성
# - 허용된 모듈/심볼만 사용하도록 코드 생성 제한
# - 미허용 토큰 감지 시 자동 재생성
# - 근거 문서 조각 제공 (explainability)
# 
# [설치 및 실행]
# 의존성 설치:
#   pip install -U streamlit "langchain>=0.2" langchain-community faiss-cpu sentence-transformers langchain-openai
# 실행:
#   streamlit run lecture_streamlit.py
# 환경변수:
#   export OPENAI_API_KEY=your_api_key_here  # 필수
#   export LECTURE_RAG_MODEL=gpt-4o-mini     # 선택 (기본값)
#   export LECTURE_RAG_TEMPERATURE=0.2       # 선택 (기본값)
# 
# [시스템 프로세스 플로우]
# 1. 인덱싱 단계:
#    강의록 파일 → 텍스트/코드 블록 분리 → 청킹 → 임베딩 생성 → FAISS 저장
#    허용 토큰(모듈/심볼) 추출 → JSON 저장
# 
# 2. 질의응답 단계:
#    사용자 질문 → 임베딩 변환 → FAISS 유사도 검색 → 상위 K개 문서 조각 검색
#    → 컨텍스트 구성 → LLM 프롬프트 생성 → GPT 답변 생성
#    → 생성된 코드에서 미허용 토큰 검사 → (필요시) 재생성 → 최종 답변 반환
# 
# 3. 보안 검증:
#    생성된 코드 → 정규식으로 import/함수 추출 → 허용 토큰과 비교
#    → 미허용 토큰 발견시 제약조건 추가하여 재요청

from __future__ import annotations
import os
import re
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ---------------------------
# Utilities (원본 로직 최대 유지)
# ---------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def detect_code_blocks(text: str) -> List[Tuple[str, str]]:
    """``` fenced code ``` 우선, 없으면 간이 코드 감지."""
    blocks: List[Tuple[str, str]] = []
    fence_pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)
    pos = 0

    # 먼저 fenced를 전부 찾아서 code 처리하고 사이 사이를 text로
    matches = list(fence_pattern.finditer(text))
    if matches:
        for m in matches:
            before = text[pos:m.start()].strip()
            if before:
                blocks.append(("text", before))
            code = m.group(1)
            blocks.append(("code", code))
            pos = m.end()
        tail = text[pos:].strip()
        if tail:
            blocks.append(("text", tail))
        return blocks

    # fallback: 간이 코드 감지
    lines = text.splitlines()
    buf: List[str] = []
    mode = "text"

    def flush():
        nonlocal buf, mode
        if buf:
            blocks.append((mode, "\n".join(buf).strip()))
            buf = []

    codey = re.compile(r"^(\s*(def |class |import |from |if __name__ == '__main__':|for |while |try:|except |with ))")
    for ln in lines:
        if codey.search(ln):
            if mode != "code":
                flush(); mode = "code"
            buf.append(ln)
        else:
            if mode != "text":
                flush(); mode = "text"
            buf.append(ln)
    flush()
    return blocks

def chunk_documents(text: str, source: str) -> List[Document]:
    blocks = detect_code_blocks(text)
    docs: List[Document] = []
    
    # 전체 텍스트의 라인 번호 매핑을 위한 준비
    all_lines = text.splitlines()
    total_lines = len(all_lines)

    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80, separators=["\n\n", "\n", ")\n", ":\n", ",\n"]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". "]
    )
    
    idx = 0
    current_line = 1
    
    for kind, content in blocks:
        block_lines = content.splitlines()
        block_start_line = current_line
        
        splitter = code_splitter if kind == "code" else text_splitter
        chunks = splitter.split_text(content)
        
        for chunk in chunks:
            chunk_lines = chunk.splitlines()
            chunk_start_line = current_line
            chunk_end_line = current_line + len(chunk_lines) - 1
            
            # 청크의 첫 번째와 마지막 줄의 일부 내용을 미리보기로 저장
            first_line_preview = chunk_lines[0][:50] + "..." if len(chunk_lines[0]) > 50 else chunk_lines[0]
            last_line_preview = chunk_lines[-1][:50] + "..." if len(chunk_lines[-1]) > 50 else chunk_lines[-1]
            
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source, 
                        "kind": kind, 
                        "chunk_id": f"{source}:{idx}",
                        "start_line": chunk_start_line,
                        "end_line": chunk_end_line,
                        "total_lines": total_lines,
                        "line_count": len(chunk_lines),
                        "first_line_preview": first_line_preview,
                        "last_line_preview": last_line_preview
                    },
                )
            )
            current_line += len(chunk_lines)
            idx += 1
            
        # 블록 간 줄바꿈 고려
        if current_line <= total_lines:
            current_line += 1
            
    return docs

def extract_allowed_tokens(text: str) -> Dict[str, Any]:
    imports = re.findall(r"^(?:from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)|import\s+([\w\.,\s]+))", text, re.M)
    modules = set()
    names = set()
    for a, b, c in imports:
        if a:
            modules.add(a)
            for nm in re.split(r"\s*,\s*", b.strip()):
                if nm:
                    names.add(nm)
        elif c:
            for m in re.split(r"\s*,\s*", c.strip()):
                if m:
                    modules.add(m)

    funcs = re.findall(r"^\s*def\s+([a-zA-Z_][\w]*)\(", text, re.M)
    klass = re.findall(r"^\s*class\s+([A-Z][\w]*)\(", text, re.M)
    consts = re.findall(r"^([A-Z_]{3,})\s*=\s*", text, re.M)

    return {
        "modules": sorted(modules),
        "symbols": sorted(set(funcs) | set(klass) | set(consts)),
    }

def find_unknown_tokens(code: str, allowed: Dict[str, Any]) -> List[str]:
    unknown = []
    for m in re.finditer(r"^(?:from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)|import\s+([\w\.,\s]+))", code, re.M):
        mod_from, names, mods = m.groups()
        if mod_from and mod_from not in allowed["modules"]:
            unknown.append(f"module:{mod_from}")
        if names:
            for nm in re.split(r"\s*,\s*", names.strip()):
                nm = nm.strip()
                if nm and nm not in allowed["symbols"]:
                    unknown.append(f"symbol:{nm}")
        if mods:
            for md in re.split(r"\s*,\s*", mods.strip()):
                md = md.strip()
                if md and md not in allowed["modules"]:
                    unknown.append(f"module:{md}")
    return sorted(set(unknown))

# ---------------------------
# Indexing & Retrieval (with cache)
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_or_load_store(store_dir: Path):
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir

def do_index(lecture_path: Path, store_dir: Path):
    text = read_text(lecture_path)
    docs = chunk_documents(text, source=str(lecture_path))

    embeddings = get_embeddings()
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(str(store_dir))

    allowed = extract_allowed_tokens(text)
    (store_dir / "allowed_tokens.json").write_text(
        json.dumps(allowed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(docs), allowed

@st.cache_resource(show_spinner=False)
def load_vectorstore(store_dir_str: str):
    embeddings = get_embeddings()
    return FAISS.load_local(store_dir_str, embeddings, allow_dangerous_deserialization=True)

def retrieve(query: str, store_dir: Path, k: int = 8) -> Tuple[List[Document], Dict[str, Any]]:
    vs = load_vectorstore(str(store_dir))
    docs = vs.similarity_search(query, k=k)
    allowed_path = store_dir / "allowed_tokens.json"
    allowed = {"modules": [], "symbols": []}
    if allowed_path.exists():
        allowed = json.loads(allowed_path.read_text(encoding="utf-8"))
    return docs, allowed

# ---------------------------
# LLM Prompting
# ---------------------------

SYSTEM_PROMPT = (
    "당신은 수업용 코치입니다.\n"
    "반드시 제공된 컨텍스트(강의록 조각들)에서만 근거를 사용하여 답하십시오.\n"
    "컨텍스트에 없는 라이브러리, 프레임워크, 함수 사용은 금지합니다.\n"
    "근거가 부족하면 정중히 거절하고, 강의록에서 가까운 대안 절차를 제안하세요.\n"
    "가능하면 강의록의 변수명/함수명/스타일을 모방하세요.\n"
)
ANSWER_GUIDE = (
    "출력 형식:\n"
    "1) 간단한 설명\n"
    "2) 코드(필요 시)\n"
    "3) 사용한 근거 스니펫들(식별자/요약)\n"
    "주의: 코드 블록은 반드시 ```python 으로 시작하세요."
)

def make_context_block(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        location_info = f"라인 {meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        chunk_info = f"[Chunk {i} | {meta.get('kind')} | {location_info} | {meta.get('chunk_id')}]"
        parts.append(f"{chunk_info}\n{d.page_content}")
    return "\n\n".join(parts)

def call_llm(query: str, docs: List[Document], allowed: Dict[str, Any], extra_file_text: Optional[str] = None) -> str:
    temp = float(os.getenv("LECTURE_RAG_TEMPERATURE", "0.2"))
    model_name = os.getenv("LECTURE_RAG_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=temp)

    style_hint = (
        "허용 모듈: " + ", ".join(allowed.get("modules", [])) + "\n"
        "허용 심볼: " + ", ".join(allowed.get("symbols", [])) + "\n"
    )

    context = make_context_block(docs)
    if extra_file_text:
        context = context + "\n\n[UserFile]\n" + extra_file_text

    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", f"질문: {query}\n\n{ANSWER_GUIDE}\n\n[컨텍스트 시작]\n{context}\n[컨텍스트 끝]\n\n{style_hint}\n")
    ]
    resp = llm.invoke(messages)
    return resp.content

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Lecture-RAG (Streamlit)", page_icon="📚", layout="wide")
st.title("Lecture-RAG - 강의록 스타일 강제 RAG (Streamlit)")

with st.sidebar:
    st.header("설정")
    default_store = ".lecture_index"
    store_dir_str = st.text_input("FAISS 저장 디렉터리", value=default_store)
    store_dir = build_or_load_store(Path(store_dir_str))

    model = st.text_input("LLM 모델 (env LECTURE_RAG_MODEL 우선)", value=os.getenv("LECTURE_RAG_MODEL", "gpt-4o-mini"))
    temp = st.slider("Temperature", 0.0, 1.0, float(os.getenv("LECTURE_RAG_TEMPERATURE", "0.2")), 0.05)
    st.caption("※ OpenAI 사용 시 환경변수 OPENAI_API_KEY가 필요합니다.")

    st.divider()
    st.subheader("인덱싱")
    upload = st.file_uploader("강의록 파일 업로드(.txt, .md 등)", type=["txt", "md", "py", "mdx"], accept_multiple_files=False)
    manual_path = st.text_input("또는 로컬 강의록 경로 입력", value="강의록.txt")

    if st.button("인덱싱 실행", type="primary"):
        if upload is not None:
            tmp_path = Path(store_dir) / "uploaded_lecture.txt"
            tmp_path.write_bytes(upload.read())
            path_to_index = tmp_path
        else:
            path_to_index = Path(manual_path)

        if not path_to_index.exists():
            st.error(f"강의록 파일을 찾을 수 없습니다: {path_to_index}")
        else:
            # 환경 값 주입(선택)
            os.environ["LECTURE_RAG_MODEL"] = model
            os.environ["LECTURE_RAG_TEMPERATURE"] = str(temp)
            with st.spinner("인덱싱 중..."):
                n_docs, allowed = do_index(path_to_index, store_dir)
            st.success(f"인덱싱 완료! 문서 조각 {n_docs}개 생성")
            with st.expander("허용 토큰(모듈/심볼) 보기"):
                st.json(allowed)

st.divider()
colQ, colOpt = st.columns([2, 1], vertical_alignment="top")
with colQ:
    st.subheader("질의응답")
    query = st.text_input("질문을 입력하세요", placeholder="예) 리스트를 역순으로 정렬하는 함수 만들어줘")
with colOpt:
    topk = st.slider("Top-K 문서", min_value=2, max_value=15, value=8, step=1)
    ref_file = st.file_uploader("추가 컨텍스트로 내 코드 파일 첨부(선택)", type=["py","txt","md"])

ask = st.button("답변 생성", type="primary")

if ask:
    if not query.strip():
        st.warning("질문을 입력하세요.")
    else:
        # 환경 값 주입(선택)
        os.environ["LECTURE_RAG_MODEL"] = model
        os.environ["LECTURE_RAG_TEMPERATURE"] = str(temp)

        # 검색
        with st.spinner("검색 중..."):
            try:
                docs, allowed = retrieve(query, store_dir, k=topk)
            except Exception as e:
                st.error(f"인덱스를 로드할 수 없습니다. 먼저 인덱싱을 수행하세요. 상세: {e}")
                docs, allowed = [], {"modules": [], "symbols": []}

        if not docs:
            st.error("검색 결과가 없습니다. 사이드바에서 인덱싱을 먼저 수행했는지 확인하세요.")
        else:
            extra_text = None
            if ref_file is not None:
                try:
                    extra_text = ref_file.read().decode("utf-8", errors="ignore")
                except Exception:
                    extra_text = None

            # 1차 생성
            with st.spinner("LLM 응답 생성 중..."):
                answer = call_llm(query, docs, allowed, extra_file_text=extra_text)

            # 미허용 심볼 검사 → 1회 재시도
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
            unknown_total: List[str] = []
            for cb in code_blocks:
                unknown_total.extend(find_unknown_tokens(cb, allowed))

            if unknown_total:
                with st.expander("미허용 토큰 감지 (자동 재생성 전 로그)", expanded=False):
                    st.write(", ".join(sorted(set(unknown_total))))
                retry_prompt = query + "\n(주의: 아래 미허용 목록을 절대 사용하지 마세요: " + ", ".join(sorted(set(unknown_total))) + ")"
                with st.spinner("미허용 토큰 제거하여 재생성 중..."):
                    answer = call_llm(retry_prompt, docs, allowed, extra_file_text=extra_text)

            st.subheader("답변")
            st.markdown(answer)

            st.subheader("근거 스니펫")
            for i, d in enumerate(docs, 1):
                meta = d.metadata
                location_info = f"라인 {meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
                first_line = meta.get('first_line_preview', '')
                
                # 더 자세한 제목 표시
                title = f"Chunk {i} | {meta.get('kind')} | {location_info} | 시작: {first_line}"
                
                with st.expander(title, expanded=False):
                    # 위치 정보 표시
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
