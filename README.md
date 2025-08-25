# skn15-3rd-1team


# 1. 팀 소개

<div align="center">

## 햄토리GO 🐹



| 조태민 | 박진우 | 서혜선 | 임가은 | 임경원 | 홍민식 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/f4e37d90-54e7-412f-9eb0-6c94ffd08170" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/6ec5c5be-b7dc-4b77-84f8-73eae0735138" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/98f8c5b4-eaf1-44f1-ac6f-c90be49f40fb" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/48f3f3e0-5118-4c93-b7c1-4302fd0c6803" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/b5ad3ea4-cdde-4ad8-bde3-8237cdd6cae0" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/84179981-6f18-4ad5-adab-9a7216a254c5" /> |
| [@o2mandoo](https://github.com/o2mandoo) | [@pjw876](https://github.com/pjw876) | [@hyeseon](https://github.com/hyeseon7135) | [@mars7421](https://github.com/mars7421) | [@KYUNGWON-99](https://github.com/KYUNGWON-99) | [@minnnsik](https://github.com/minnnsik) |



</div>

---
# 2. 프로젝트 기간
	- 2025.08.22 ~ 2025.08.25 (총 2일)
---
# 3. 프로젝트 개요

## 📕 프로젝트명
### Lecture-RAG: 강의록 기반 AI 학습 도우미
---

## ✅ 프로젝트 배경 및 목적
### - 부트캠프에서 다루는 방대한 강의 내용을 효과적으로 학습할 수 있도록 LLM을 활용한 “맞춤형 질문/답변 시스템” 필요

### - 단순 GPT 질의응답이 아닌, 사용자가 제공한 강의록/문서 기반으로 답변을 생성하여 맥락적 신뢰성 확보

### - 불필요한 외부 라이브러리 호출을 방지하고, 강의록에서 제공된 함수·코드 스타일을 모방하여 코드 예제 제시
---
## 🖐️ 프로젝트 소개
### - 업로드한 강의록(.txt, .py, .md 등) 파일을 자동으로 청킹하고 임베딩하여 FAISS VectorStore에 저장

### - 사용자가 입력한 질문을 기반으로 강의록에서 의미적으로 유사한 조각을 검색

### - OpenAI GPT 모델을 통해 컨텍스트 RAG 기반 답변 생성

### - 미허용된 토큰(import, 함수) 사용 시 자동으로 감지하여 재생성 기능 수행

### - Streamlit UI를 통해 손쉽게 문서 업로드, 인덱싱, 질의응답 가능
---
## ❤️ 기대효과
### - 강의 내용 복습 및 개인화된 학습 도우미 활용 가능

### - 코딩 학습 시 강의 자료 기반 맞춤 코드 예제를 제공 → 실습 효율성 향상

### - 불필요하거나 잘못된 답변 최소화
---
## 👤 대상 사용자
### - 부트캠프 수강생 및 기타 강의 학습자

### - 내부 매뉴얼/문서를 기반으로 효율적 학습이 필요한 개발자 및 연구원
---
## 📁 프로젝트 폴더 구조
```
SKN15-3rd-1Team/

├── 📁 lecture_rag/                     # 강의록 기반 RAG 패키지
│   ├── __init__.py                     # 패키지 초기화/공개 API
│   ├── app.py                          # Streamlit 앱 엔트리(UI, 인덱싱/QA)
│   ├── config.py                       # dataclass 설정(LLM/청킹/검색/프롬프트)
│   ├── document_processor.py           # 날짜(YYYY.MM.DD) 기반 청킹 및 메타 생성
│   ├── google_drive.py                 # (신규) 구글 드라이브 연계 모듈(파일 로드/저장)
│   ├── llm_handler.py                  # ChatOpenAI 래퍼, 답변 생성/재시도/토큰검증
│   ├── utils.py                        # 텍스트 I/O, 코드블록 감지, 허용 토큰 유틸
│   └── vector_store.py                 # FAISS 임베딩/인덱싱/검색, allowed_tokens.json 관리
├── .gitignore                          # Git 제외 규칙
├── langgraph_flow.py                   # 랭그래프 파이프라인 정의/다이어그램 생성 스크립트
├── main.py                             # (선택) 앱 실행 진입점
├── rag_flow.png                        # 파이프라인 이미지(README 삽입용)
├── rag_flow_mermaid.md                 # Mermaid 다이어그램 정의(문서화용)
├── README.md                           # 프로젝트 문서
└── requirements.txt                    # 의존성 목록
  🚫 로컬에만 있는 파일들 (.gitignore로 제외)
#.env - OpenAI API 키 등
# .lecture_index/ - FAISS 인덱스 (토큰 포함)
# pycache/ - Python 캐시
# .claude/ - Claude Code 설정
```
---
## ♒ LangGraph Flow
<img width="250" height="600" alt="image" src="https://github.com/user-attachments/assets/9834c9d1-7bf2-4400-868e-90cbff2487c5" />

- preprocess_query(불용어 제거/정규화) → vector_search(Top‑K 스니펫 수집) → load_allowed_tokens(화이트리스트 로드) → build_context(근거 블록 생성) → generate_answer(LLM 호출) → check_tokens(코드블록 검증) → retry_with_constraints(제약 강화 재시도)
- check_tokens 단계에서 허용되지 않은 모듈/심볼이 검출되면 retry_with_constraints로 분기해 화이트리스트를 강제한 프롬프트로 한 번 더 답변을 생성하고, 문제가 없으면 end로 종료한다.
---
## 🖥️ 시스템 아키텍쳐
<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/7831f314-49d2-42f5-b674-aae1204dd047" />


---
## 🚦 전체 파이프라인
| 단계 | 설명 요약 |
| --- | --- |
| 🧹 전처리 | 입력 질의를 정규화해 공백 정리·불용어 최소화 등으로 검색 친화적 형태로 변환한다. |
| 🔎 벡터 검색 | 인덱싱된 강의록에서 질의 임베딩과 유사한 문서 조각 Top‑K를 수집한다. |
| ✅ 토큰 로드 | 인덱싱 시 저장된 화이트리스트(modules, symbols)를 읽어 제약 조건의 기준으로 사용한다. |
| 🧩 컨텍스트 구성 | 스니펫 본문과 메타데이터(강의일, 시작/끝 라인, chunk_id)를 묶어 LLM에 전달할 컨텍스트 블록을 만든다.|
| 🧠 1차 생성 | 시스템 프롬프트·출력 가이드·컨텍스트·스타일 힌트를 포함해 LLM을 호출해 답변을 생성한다. |
| 🔒 토큰 검증 | 답변의 코드 블록만 추출해 화이트리스트에 없는 모듈/심볼 사용 여부를 검사한다. |
| ♻️ 재시도 | 위반 항목이 있으면 “금지 목록을 절대 사용하지 말라” 제약을 추가해 한 번 더 재생성한다. |
| ✅ 종료/표시 | 최종 답변을 출력하고, 거절 응답이 아니면 근거 스니펫(강의일/라인)을 함께 노출한다. |
---
## 🔌 단계별 입·출력
| 단계 | 설명 요약 |
| --- | --- |
| 🧹 전처리 입력/출력 | 입력: 원문 질의 → 출력: 정규화 질의.  |
| 🔎 벡터 검색 입력/출력 | 입력: 정규화 질의 → 출력: 문서 조각 리스트(docs). |
| ✅ 토큰 로드 입력/출력 | 입력: 인덱스 경로 → 출력: allowed={modules, symbols}. |
| 🧩 컨텍스트 구성 입력/출력 | 입력: docs → 출력: 컨텍스트 문자열(context).  |
| 🧠 답변 생성 입력/출력 | 입력: query, context, allowed, 프롬프트 → 출력: answer(text/markdown).  |
| 🔒 토큰 검증 입력/출력 | 입력: answer, allowed → 출력: unknown_tokens 리스트. |
| ♻️ 재시도 입력/출력 | 입력: query+금지목록, context, 프롬프트 → 출력: 교정된 answer. |
---
## 프롬프트
### Systeam Prompt
```
 system_prompt: str = (
        "당신은 수업용 코치입니다.\n"
        "제공된 컨텍스트(강의록 조각들)에서 근거를 찾아 답변하십시오.\n"
        "컨텍스트에 관련 내용이 있으면 그것을 기반으로 설명하고 코드를 제공하세요.\n"
        "컨텍스트에 없는 라이브러리나 함수는 사용하지 마세요.\n"
        "질문이 프로그래밍과 전혀 무관하거나(예: 요리, 여행, 운동 등) 컨텍스트와 완전히 무관한 경우에만 '죄송합니다. 해당 내용은 강의록에서 다루지 않은 주제입니다.'라고 거절하세요.\n"
        "프로그래밍 관련 질문이라면 컨텍스트에서 유사한 내용을 찾아 최대한 도움을 주려고 노력하세요.\n"
        "가능하면 강의록의 변수명/함수명/스타일을 모방하세요.\n"
    )
```
---
### Answer Prompt 
```
answer_guide: str = (
        "출력 형식:\n"
        "1) 간단한 설명\n"
        "2) 코드(필요 시)\n"
        "3) 사용한 근거 스니펫들(강의일 기준으로 표시)\n"
        "주의: 코드 블록은 반드시 ```python 으로 시작하세요."
    )
```
### 프롬프트 의도
- **역할**: 본 프롬프트는 “수업용 코치 + 코딩 어시스턴트”로서, 강의록에서 추출한 스니펫을 근거로만 답하도록 설계되었다.

- **근거 원칙**: 외부 지식이나 강의록에 없는 라이브러리/함수 사용을 금지해 일관성과 재현성을 보장한다.

- **실패 처리**: 컨텍스트에 근거가 없을 때는 정중히 거절하고, 인접 주제/검색어 제안을 통해 탐색을 유도한다.

- **스타일**: 한국어로 간결·정확하게 답하며, 가능하면 강의록의 변수명·함수명·코딩 스타일을 모방한다.

- **출력 포맷**: “간단한 설명 → 코드(있을 때) → 근거 스니펫(강의일/라인)” 순서를 엄수하고, 코드블록은 반드시 ```python 으로 시작한다.

- **화이트리스트**: 허용되지 않은 모듈/심볼은 사용하지 않으며, 필요한 경우 대체 표준 라이브러리를 설명 수준에서만 제안한다.

- **품질 관리**: 유사·중복 스니펫은 통합하고 Top-K 내에서 핵심만 사용하여 장황한 인용을 피한다.
---
## 데이터 전처리
### 1️⃣ 입력: 파일 읽기
```python
def process_file(self, file_path: Path) -> List[Document]:
    """
    파일을 읽어서 Document 청크로 변환
    """
    text = read_text(file_path)  # <- 여기서 전체 텍스트 읽음
    return self.chunk_documents(text, source=str(file_path))
```
- read_text(file_path) → utils.py의 함수

- 전체 텍스트를 불러와서 chunk_documents로 전달

2️⃣ 날짜 기반 청크 분할
```python
def chunk_documents(self, text: str, source: str) -> List[Document]:
    """
    텍스트를 날짜 기반으로 청킹 (YYYY.MM.DD 패턴)
    """
    date_pattern = r'^(\d{4}\.\d{2}\.\d{2})\s*'  # <- 날짜 패턴 정의
    lines = text.splitlines()
    docs: List[Document] = []
    current_chunk_lines = []
    current_date = None
    current_start_line = 1
    chunk_idx = 0

    for line_idx, line in enumerate(lines, 1):
        date_match = re.match(date_pattern, line.strip())
        
        if date_match:
            if current_chunk_lines and current_date:
                # 이전 청크 저장
                chunk_content = '\n'.join(current_chunk_lines)
                if chunk_content.strip():
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
```

- 핵심: 날짜가 나오면 새 청크 시작, 이전 청크는 _create_date_chunk_document로 Document 생성

- 줄 단위 순회하면서 문맥 유지

3️⃣ Document 객체 생성
```python
def _create_date_chunk_document(
    self, content: str, source: str, date: str, 
    start_line: int, end_line: int, chunk_idx: int
) -> Document:
    """날짜 기반 청크용 Document 생성"""
    lines = content.splitlines()
    first_line = lines[0] if lines else ""
    last_line = lines[-1] if lines else ""
    
    # 첫 줄에서 날짜 제거 후 미리보기
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
```

- 날짜 기반 청크를 Document 객체로 변환

- 메타데이터 포함 → RAG에서 근거 스니펫 표시 가능


---
## 환경설정
### 1️⃣ 패키지 설치
```
pip install -r requirements.txt
```
### 2️⃣ 환경 변수 (.env) 설정
```
# Google Drive API (구글드라이브 연동용)
GOOGLE_CLIENT_ID = "your_key"
GOOGLE_CLIENT_SECRET = "your_key"
OPENAI_API_KEY = "your_key"
```
### 3️⃣ 서비스 실행
```
python -m streamlit run lecture_rag/app.py
```
### 4️⃣ 강의록 등 .txt 파일 다운로드 (Google api key 없을 시 수동으로 다운로드) 
```
# 파일 다운로드 후, streamlit 웹에 동적으로 업로드 후 이용가능
```
---
# 4. 기술 스택


| Field	| Tool |
|----|---|
| Frontend	| <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> |
| LLM	| <img src="https://img.shields.io/badge/OpenAI%20GPT-412991?style=for-the-badge&logo=openai&logoColor=white"> <img src="https://img.shields.io/badge/langchain--openai-0B3B5B?style=for-the-badge&logo=openai&logoColor=white"> |
| Vector DB	| <img src="https://img.shields.io/badge/FAISS-20232A?style=for-the-badge&logo=facebook&logoColor=white"> |
| Embedding	| <img src="https://img.shields.io/badge/HuggingFace%20all--MiniLM--L6--v2-FFAE00?style=for-the-badge&logo=huggingface&logoColor=white"> |
| Framework	| <img src="https://img.shields.io/badge/LangChain-0B3B5B?style=for-the-badge&logo=chainlink&logoColor=white"> |
| Language	| <img src="https://img.shields.io/badge/Python%203.8%2B-3776AB?style=for-the-badge&logo=Python&logoColor=white">| 

---
# 5. 수행결과
## 🎥 시연 화면 (예시)
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/60f44115-c70d-4ebf-b8a6-3390a60f5c0e" />
****
---
### 🏠 0) 홈 화면(Overview)
**설정 패널(좌측)**
- FAISS 저장 디렉터리: 벡터 인덱스와 허용 토큰 파일이 저장될 로컬 경로를 지정한다(예: `.lecture_index`).  
- LLM 모델 선택: 기본값은 환경변수(예: `LECTURE_RAG_MODEL`)를 따르며, 드롭다운에서 실행 중에 변경 가능하다.  
- Temperature 슬라이더: 생성 다양성을 제어한다(낮을수록 보수적, 높을수록 창의적).  

**질문/옵션 영역(중앙)**
- 질의 입력창: 분석·설명·코드 요청 등 자연어 질문을 입력한다.  
- Top‑K 문서: 검색에서 가져올 스니펫 개수를 조절하여 컨텍스트의 폭을 설정한다.  
- 답변 생성 버튼: 현재 설정과 인덱스를 기반으로 컨텍스트를 구성해 LLM 답변을 생성한다.  

**인덱싱(좌측 하단)**
- 구글 드라이브에서 강의록 가져오기: 드라이브에 업로드된 텍스트/마크다운/코드 파일을 불러와 인덱싱할 수 있다.  
- 로컬 파일 업로드(대안): 드라이브 사용이 어려울 때 로컬 파일을 직접 선택해 인덱싱을 수행한다.  

**사용 흐름 요약**
1) 좌측에서 모델·인덱스 경로·온도를 설정한다.  
2) 인덱싱 섹션에서 강의록을 업로드(또는 드라이브에서 가져오기)하여 벡터 인덱스를 생성한다.  
3) 중앙 입력창에 질문을 입력하고 Top‑K를 조절한 뒤 “답변 생성”을 누르면, 근거 스니펫을 바탕으로 답변과 코드가 출력된다.  

**안내**
- OpenAI 계열 모델을 사용할 때는 사전에 API 키(예: `OPENAI_API_KEY`)가 환경에 설정되어 있어야 정상 동작한다.  
- 프로젝트별 문서로 인덱싱을 분리하려면 저장 디렉터리를 서로 다른 경로로 지정해 독립적인 컨텍스트를 유지할 수 있다.  
---
### 📌 1) 답변 섹션(개요 + RAG 코드 샘플)
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/4f9553b6-979b-4427-bf19-4383ca24c5ee" />
- 상단에 “답변” 제목과 함께 RAG의 개념 요약이 먼저 노출되고, 그 아래에는 기본 RAG 체인을 구성하는 코드 예시가 포함된다.  
- 이 화면은 사용자가 모델이 어떤 방식으로 답을 생성하는지 이해하도록 돕는 “학습/가이드 영역” 역할을 한다.  
---
### 📌 2) 근거 스니펫(세부 확인 패널)
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/fea657c8-a0a9-4dc6-9f60-90692c2d184a" />


- 검색된 문서 조각들이 Chunk 단위로 나열되며, 각 항목을 펼치면 해당 스니펫의 본문과 라인 범위, 타입(text/code) 같은 메타데이터를 확인할 수 있다.  
- “원본에서 찾기” 힌트를 제공해 실제 파일의 위치(라인 번호 기준)를 빠르게 탐색하도록 돕는 검증용 UI다. 
---
### 📌 3) 질의/옵션 입력 영역(실행 패널)
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/a6382b77-f3c5-4caf-a2c9-3a5dd8f061d5" />

- 중앙 상단에 질문 입력창과 Top‑K 슬라이더가 배치되어 검색 범위를 조정할 수 있고, 우측 드롭존에는 추가 컨텍스트 파일을 일시 첨부할 수 있다.  
- 좌측 사이드바에는 인덱스 경로, LLM 모델, Temperature 등의 런타임 설정이 모여 있어 “실행 전 준비/세팅”을 담당한다.  
---
### 📌 4) 최종 답변 + 근거 요약(결과 패널)
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/78c73580-808e-4363-9147-1468486f5414" />

- 생성된 답변이 문단과 코드 블록으로 제시되며, 하단에는 “사용한 근거 스니펫들” 목록이 요약되어 근거 기반 응답임을 명확히 한다.  
- 이 화면은 사용자가 즉시 실행 가능한 코드와 함께, 어떤 스니펫이 답변을 뒷받침했는지를 한 눈에 확인하는 “결과/검증” 단계다.  
---
### 📌 5) 답변 방식 비교
### 1) 기술 개념 질문 화면: “rag가 뭐야?”
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/63067634-177f-4e40-8643-2c134c147fcf" />

- 목적: 강의 컨텍스트에 존재하는 개념(RAG)을 질의했을 때의 정상 동작 예시를 보여준다.  
- 특징:
  - 상단 입력창과 Top‑K 조절 슬라이더로 검색 폭을 설정하고, 우측 드롭존에 추가 컨텍스트를 첨부할 수 있다.  
  - 하단 “답변” 섹션에서 RAG 정의와 동작 원리를 서술하고, 이어서 RAG 체인을 구성하는 코드 예시를 제공한다.  
  - 근거 스니펫 패널로 어떤 강의 일자/라인에서 정보를 인용했는지 검증 가능하다.  

### 2) 비컨텍스트 일반지식 질문 화면: “호날두가 누구야?”
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/2639cf92-4692-4912-92ee-2ae39238a519" />

- 목적: 강의록 범위를 벗어난 일반 지식 질문에 대해 시스템이 어떻게 안전하게 응답하는지 보여준다.  
- 특징:
  - 출력 포맷은 동일하게 유지하되, 코드 섹션은 “필요 시” 조건에 따라 생략된다.  
  - “사용한 근거 스니펫들”에 컨텍스트 부재 안내가 포함되어, 강의 자료에 근거가 없음을 명확히 알린다.  
  - 환각 방지를 위해 외부 지식으로 채우지 않고, 정중한 제한 응답 정책을 따른다.
 
### 📖 답변 방식 요약
#### ➡️ 첫 화면은 강의 컨텍스트 기반의 “설명+코드+근거” 풀 세트를 보여 주고, 두 번째 화면은 컨텍스트 부재 시 “정중한 제한 응답”으로 전환하는 정책을 시각적으로 확인
---
# 6. 한 줄 회고

|조태민|박진우|서혜선|
|----|---|---|
| LLM을 활용해서, 데이터의 저장, 청크, 임베딩, 라우터, 래그 등 랭채인을 이용한 파이프라인을 처음부터 끝까지 다뤄볼 수 있었습니다. 의도대로 기능을 구현하고 논리를 설계하면서 흐름을 더 잘 이해할 수 있는 기회를 얻어 좋았습니다. |  | 업로드된 문서 기반으로 LLM이 질의응답하는 구조를 구현하며, 코드 중심 접근의 장점을 체감할 수 있었습니다. 이해안됐던 부분들도 이해할 수 있었고 팀원들도 다 열심히 해줘서 좋은 결과로 마무리한 것 같아 좋습니다. 다들 고생하셨습니다. |



|임가은|임경원|홍민식|
|----|---|---|
| LLM에게 내 문서를 전달하여 그를 바탕으로 정보를 얻을 수 있다는 점이 흥미로웠다. gpt처럼 마냥 새로운 것을 주는 게 아니라, 내 문서를 기반으로 대답한다는 점이 좋았고, 다양한 문서 형식을 처리하기 위해 전처리 과정과 청킹 전략이 얼마나 중요한지 깨달았다. LLM 관련으로 좋은 경험이 된 것 같다.  | 이번 프로젝트는 단순히 LLM을 활용하는 기술을 익히는 것을 넘어, 데이터의 흐름을 처음부터 끝까지 직접 설계하고 제어하는 경험이었습니다. 각 단계가 유기적으로 연결되어야만 의도한 대로 기능이 구현되는 것을 보며 시스템 전반의 논리적인 구조를 설계하는 능력을 기를 수 있었습니다. 앞으로 더 복잡하고 정교한 LLM 애플리케이션을 개발하는 데 도움이 되었습니다. |  |



