"""
LangGraph를 사용한 Lecture-RAG 플로우 시각화
"""
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
import json

class RAGState(TypedDict):
    """RAG 처리 상태를 관리하는 상태 클래스"""
    query: str
    documents: List[Document]
    allowed_tokens: Dict[str, Any]
    context: str
    answer: str
    unknown_tokens: List[str]
    needs_retry: bool

def preprocess_query(state: RAGState) -> RAGState:
    """사용자 질문 전처리"""
    query = state["query"].strip()
    print(f"📝 Query preprocessing: {query}")
    # query는 수정하지 않고 그대로 유지
    return state

def vector_search(state: RAGState) -> RAGState:
    """벡터 검색 수행"""
    query = state["query"]
    print(f"🔍 Vector search for: {query}")
    
    # 실제로는 VectorStore.search() 호출
    # 여기서는 시뮬레이션
    mock_docs = [
        Document(
            page_content="def reverse_list(lst): return lst[::-1]",
            metadata={"kind": "code", "start_line": 10, "end_line": 12}
        )
    ]
    
    # documents만 업데이트
    return {**state, "documents": mock_docs}  # 안전한 업데이트

def load_allowed_tokens(state: RAGState) -> RAGState:
    """허용된 토큰 로드"""
    print("🔐 Loading allowed tokens...")
    
    # 실제로는 allowed_tokens.json에서 로드
    mock_tokens = {
        "modules": ["numpy", "pandas", "matplotlib"],
        "symbols": ["list", "dict", "str", "int"]
    }
    
    print(f"✅ Loaded {len(mock_tokens['modules'])} modules, {len(mock_tokens['symbols'])} symbols")
    return {**state, "allowed_tokens": mock_tokens}

def build_context(state: RAGState) -> RAGState:
    """검색된 문서들로 컨텍스트 구성"""
    docs = state["documents"]
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        location = f"라인 {meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
        context_parts.append(f"[Chunk {i} | {meta.get('kind')} | {location}]\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    print(f"📄 Built context with {len(docs)} documents")
    return {**state, "context": context}

def generate_answer(state: RAGState) -> RAGState:
    """LLM을 사용한 답변 생성"""
    query = state["query"]
    context = state["context"]
    
    print(f"🤖 Generating answer for: {query}")
    
    # 실제로는 LLM 호출
    # 여기서는 시뮬레이션
    mock_answer = """
리스트를 역순으로 정렬하는 함수는 다음과 같습니다:

```python
def reverse_list(lst):
    return lst[::-1]
```

이 함수는 슬라이싱을 사용하여 리스트를 역순으로 반환합니다.
"""
    
    print("✅ Answer generated")
    return {**state, "answer": mock_answer.strip(), "needs_retry": False}

def check_tokens(state: RAGState) -> RAGState:
    """답변에서 미허용 토큰 검사"""
    answer = state["answer"]
    allowed = state["allowed_tokens"]
    
    print("🔍 Checking for unknown tokens...")
    
    # 실제로는 utils.find_unknown_tokens() 호출
    # 여기서는 시뮬레이션
    unknown_tokens = []  # 빈 리스트 = 모든 토큰이 허용됨
    
    if unknown_tokens:
        print(f"⚠️ Found {len(unknown_tokens)} unknown tokens: {unknown_tokens}")
    else:
        print("✅ All tokens are allowed")
    
    return {**state, "unknown_tokens": unknown_tokens, "needs_retry": len(unknown_tokens) > 0}

def retry_with_constraints(state: RAGState) -> RAGState:
    """제약조건과 함께 답변 재생성"""
    query = state["query"]
    unknown_tokens = state["unknown_tokens"]
    
    print(f"🔄 Retrying with constraints: avoid {unknown_tokens}")
    
    # 실제로는 제약조건이 포함된 프롬프트로 LLM 재호출
    constrained_answer = state["answer"] + "\n\n(재생성됨: 미허용 토큰 제거)"
    
    print("✅ Answer regenerated with constraints")
    return {**state, "answer": constrained_answer, "unknown_tokens": [], "needs_retry": False}

def should_retry(state: RAGState) -> str:
    """재시도가 필요한지 판단"""
    if state["needs_retry"]:
        return "retry"
    return "end"

def create_rag_graph():
    """RAG 플로우 그래프 생성"""
    
    # 그래프 생성
    workflow = StateGraph(RAGState)
    
    # 노드 추가
    workflow.add_node("preprocess_query", preprocess_query)
    workflow.add_node("vector_search", vector_search)
    workflow.add_node("load_allowed_tokens", load_allowed_tokens)
    workflow.add_node("build_context", build_context)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("check_tokens", check_tokens)
    workflow.add_node("retry_with_constraints", retry_with_constraints)
    
    # 엣지 추가 (플로우 정의) - 순차적으로 변경
    workflow.set_entry_point("preprocess_query")
    
    # 순차 처리로 변경 (동시 업데이트 문제 해결)
    workflow.add_edge("preprocess_query", "vector_search")
    workflow.add_edge("vector_search", "load_allowed_tokens")
    workflow.add_edge("load_allowed_tokens", "build_context")
    workflow.add_edge("build_context", "generate_answer")
    workflow.add_edge("generate_answer", "check_tokens")
    
    # 조건부 라우팅
    workflow.add_conditional_edges(
        "check_tokens",
        should_retry,
        {
            "retry": "retry_with_constraints",
            "end": END
        }
    )
    
    # 재시도 후 다시 토큰 검사
    workflow.add_edge("retry_with_constraints", "check_tokens")
    
    return workflow.compile()

def visualize_graph():
    """그래프 시각화"""
    print("🎨 Creating graph visualizations...")
    
    # 그래프 생성
    app = create_rag_graph()
    
    try:
        # 1. Mermaid 다이어그램으로 출력
        mermaid_code = app.get_graph().draw_mermaid()
        with open("rag_flow_mermaid.md", "w", encoding="utf-8") as f:
            f.write("# RAG Flow Diagram\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_code)
            f.write("\n```")
        print("✅ Mermaid diagram saved to: rag_flow_mermaid.md")
        
        # 2. PNG 이미지로 출력 (graphviz 필요)
        try:
            img = app.get_graph().draw_mermaid_png()
            with open("rag_flow.png", "wb") as f:
                f.write(img)
            print("✅ PNG diagram saved to: rag_flow.png")
        except Exception as e:
            print(f"⚠️ PNG generation failed (install graphviz): {e}")
            
        # 3. ASCII 다이어그램 출력
        print("\n📊 Graph Structure (ASCII):")
        print(app.get_graph().draw_ascii())
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        print("Install required packages: pip install 'langgraph[draw]'")

def run_rag_demo():
    """RAG 데모 실행"""
    print("🚀 Starting Lecture-RAG LangGraph Demo")
    print("=" * 50)
    
    # 그래프 생성
    app = create_rag_graph()
    
    # 시각화
    visualize_graph()
    print("\n" + "=" * 50)
    
    # 초기 상태
    initial_state = {
        "query": "리스트를 역순으로 정렬하는 함수 만들어줘",
        "documents": [],
        "allowed_tokens": {},
        "context": "",
        "answer": "",
        "unknown_tokens": [],
        "needs_retry": False
    }
    
    # 실행
    print("🎯 Processing query:", initial_state["query"])
    print("-" * 50)
    
    result = app.invoke(initial_state)
    
    print("-" * 50)
    print("🎉 Final Result:")
    print(result["answer"])
    
    return result

if __name__ == "__main__":
    # 패키지 설치 안내
    try:
        from langgraph.graph import StateGraph, END
        run_rag_demo()
    except ImportError:
        print("❌ LangGraph가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("pip install langgraph")