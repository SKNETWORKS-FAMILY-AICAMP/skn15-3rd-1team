# skn15-3rd-1team


# 1. 팀 소개

<div align="center">

## 햄토리GO 🐹



| 조태민 | 박진우 | 서혜선 | 임가은 | 임경원 | 홍민식 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/f4e37d90-54e7-412f-9eb0-6c94ffd08170" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/6ec5c5be-b7dc-4b77-84f8-73eae0735138" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/98f8c5b4-eaf1-44f1-ac6f-c90be49f40fb" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/48f3f3e0-5118-4c93-b7c1-4302fd0c6803" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/b5ad3ea4-cdde-4ad8-bde3-8237cdd6cae0" /> | <img width="110" height="120" alt="Image" src="https://github.com/user-attachments/assets/84179981-6f18-4ad5-adab-9a7216a254c5" /> |
| [@o2mandoo](https://github.com/o2mandoo) | [@pjw876](https://github.com/pjw876) | [@hyeseon](https://github.com/hyeseon7135) | [@mars7421](https://github.com/mars7421) | [@KYUNGWON-99](https://github.com/KYUNGWON-99) | [@minnnsik](https://github.com/minnnsik) |



</div>


# 2. 프로젝트 기간
	- 2025.08.22 ~ 2025.08.25 (총 2일)

# 3. 프로젝트 개요

## 📕 프로젝트명
### Lecture-RAG: 강의록 기반 AI 학습 도우미


## ✅ 프로젝트 배경 및 목적
### - 부트캠프에서 다루는 방대한 강의 내용을 효과적으로 학습할 수 있도록 LLM을 활용한 “맞춤형 질문/답변 시스템” 필요

### - 단순 GPT 질의응답이 아닌, 사용자가 제공한 강의록/문서 기반으로 답변을 생성하여 맥락적 신뢰성 확보

### - 불필요한 외부 라이브러리 호출을 방지하고, 강의록에서 제공된 함수·코드 스타일을 모방하여 코드 예제 제시

## 🖐️ 프로젝트 소개
### - 업로드한 강의록(.txt, .py, .md 등) 파일을 자동으로 청킹하고 임베딩하여 FAISS VectorStore에 저장

### - 사용자가 입력한 질문을 기반으로 강의록에서 의미적으로 유사한 조각을 검색

### - OpenAI GPT 모델을 통해 컨텍스트 RAG 기반 답변 생성

### - 미허용된 토큰(import, 함수) 사용 시 자동으로 감지하여 재생성 기능 수행

### - Streamlit UI를 통해 손쉽게 문서 업로드, 인덱싱, 질의응답 가능
  
## ❤️ 기대효과
### - 강의 내용 복습 및 개인화된 학습 도우미 활용 가능

### - 코딩 학습 시 강의 자료 기반 맞춤 코드 예제를 제공 → 실습 효율성 향상

### - 불필요하거나 잘못된 답변 최소화

## 👤 대상 사용자
### - 부트캠프 수강생 및 기타 강의 학습자

### - 내부 매뉴얼/문서를 기반으로 효율적 학습이 필요한 개발자 및 연구원

## 📁 프로젝트 폴더 구조
```
SKN15-3rd-1Team/

├── 📁 lecture_rag/                  # 강의록 기반 RAG 패키지(인덱싱·검색·LLM 호출)
│   ├── __init__.py                  # 공개 API와 버전, 핵심 클래스/함수 내보내기
│   ├── app.py                       # Streamlit 웹앱(UI, 인덱싱/질의응답 흐름)
│   ├── config.py                    # dataclass 기반 설정(LLM/청킹/탐색/LFS 등)
│   ├── document_processor.py        # 날짜 패턴 기반(YYYY.MM.DD) 청킹 로직
│   ├── llm_handler.py               # LangChain ChatOpenAI 래퍼, 답변 생성/검증
│   ├── utils.py                     # 텍스트 입출력, 코드블록 감지, 토큰 추출/검증
│   └── vector_store.py              # FAISS 벡터스토어 관리(인덱싱/저장/검색)
├── .gitignore                       # 불필요 파일 제외 규칙
├── main.py                          # 앱 실행 진입점(lecture_rag.app.main 호출 가정)
├── README.md                        # 프로젝트 문서
└── requirements.txt                 # 의존성 목록
                       
```
# 4. 기술 스택


| Field	| Tool |
|----|---|
| Frontend	| <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> |
| LLM	| <img src="https://img.shields.io/badge/OpenAI%20GPT-412991?style=for-the-badge&logo=openai&logoColor=white"> <img src="https://img.shields.io/badge/langchain--openai-0B3B5B?style=for-the-badge&logo=openai&logoColor=white"> |
| Vector DB	| <img src="https://img.shields.io/badge/FAISS-20232A?style=for-the-badge&logo=facebook&logoColor=white"> |
| Embedding	| <img src="https://img.shields.io/badge/HuggingFace%20all--MiniLM--L6--v2-FFAE00?style=for-the-badge&logo=huggingface&logoColor=white"> |
| Framework	| <img src="https://img.shields.io/badge/LangChain-0B3B5B?style=for-the-badge&logo=chainlink&logoColor=white"> |
| Language	| <img src="https://img.shields.io/badge/Python%203.8%2B-3776AB?style=for-the-badge&logo=Python&logoColor=white">| 


# 5. 수행결과
🎥 시연 화면 (예시)
### - 메인 UI: 강의록 업로드 & 인덱싱

### - 질의응답 데모: 질문 입력 → 강의록 기반 답변 생성

### - 출력 결과:

* 간단한 설명

* 필요 시 코드

* 근거가 된 강의록 스니펫

## 📂 **주요 파일 구조**

# 6. 한 줄 회고

|조태민|박진우|서혜선|
|----|---|---|
| LLM을 활용해서, 데이터의 저장, 청크, 임베딩, 라우터, 래그 등 랭채인을 이용한 파이프라인을 처음부터 끝까지 다뤄볼 수 있었습니다. 의도대로 기능을 구현하고 논리를 설계하면서 흐름을 더 잘 이해할 수 있는 기회를 얻어 좋았습니다. |  | 업로드된 문서 기반으로 LLM이 질의응답하는 구조를 구현하며, 코드 중심 접근의 장점을 체감할 수 있었습니다. 이해안됐던 부분들도 이해할 수 있었고 팀원들도 다 열심히 해줘서 좋은 결과로 마무리한 것 같아 좋습니다. 다들 고생하셨습니다. |



|임가은|임경원|홍민식|
|----|---|---|
| LLM에게 내 문서를 전달하여 그를 바탕으로 정보를 얻을 수 있다는 점이 흥미로웠다. gpt처럼 마냥 새로운 것을 주는 게 아니라, 내 문서를 기반으로 대답한다는 점이 좋았고, 다양한 문서 형식을 처리하기 위해 전처리 과정과 청킹 전략이 얼마나 중요한지 깨달았다. LLM 관련으로 좋은 경험이 된 것 같다.  | 이번 프로젝트는 단순히 LLM을 활용하는 기술을 익히는 것을 넘어, 데이터의 흐름을 처음부터 끝까지 직접 설계하고 제어하는 경험이었습니다. 각 단계가 유기적으로 연결되어야만 의도한 대로 기능이 구현되는 것을 보며 시스템 전반의 논리적인 구조를 설계하는 능력을 기를 수 있었습니다. 앞으로 더 복잡하고 정교한 LLM 애플리케이션을 개발하는 데 도움이 되었습니다. |  |



