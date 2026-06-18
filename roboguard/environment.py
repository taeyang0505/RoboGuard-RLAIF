"""
environment.py — RAG 검색 환경 (Chroma DB 연동)
=================================================
[Self-RAG §2 "Retrieval-Augmented Generation Environment" 개념 적용]

Self-RAG 논문의 핵심 아이디어 (p.3):
  검색(Retrieval)은 단순한 전처리가 아닌, 에이전트가 능동적으로 활용하는
  '환경(Environment)'의 일부이다.
  모델은 언제 검색할지(Retrieve token)를 스스로 결정하며,
  검색 결과의 관련성(IsREL)도 자체 평가한다.

RL 표준 인터페이스인 gym.Env의 step() 패턴을 모방하여
RetrievalEnvironment.step(question) → context 형태로 구현합니다.
이를 통해 향후 다른 Vector DB(Pinecone, Weaviate 등)로의
교체가 용이한 인터페이스를 제공합니다.

Phase 1 — Source Citation:
  step_with_citations()가 컨텍스트 문자열과 함께 참조 페이지 번호 목록을
  별도로 반환합니다. Chroma 메타데이터의 'page' 키를 사용합니다.
"""
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CONFIG

load_dotenv()


@dataclass
class RetrievalResult:
    """
    Vector DB 검색 결과 컨테이너.

    context     : 검색된 문서 조각들을 연결한 컨텍스트 문자열
    source_pages: 중복 제거된 오름차순 페이지 번호 목록 (1-indexed)
    """
    context: str
    source_pages: list[int] = field(default_factory=list)


class RetrievalEnvironment:
    """
    Chroma Vector DB 기반 RAG 검색 환경.

    [Self-RAG §2 "Retrieval Model p(d|x)" 구현]
    질문 x를 받아 관련 문서 집합 D를 반환하는 확률 모델 p(d|x)를
    임베딩 기반 유사도 검색으로 근사합니다.

    [RL Environment 인터페이스]
    - reset(): 환경 초기화 (향후 멀티턴/에피소드 지원 대비)
    - step(question): 컨텍스트 문자열만 반환 (하위 호환 유지)
    - step_with_citations(question): 컨텍스트 + 페이지 목록 반환 (Phase 1)
    """

    def __init__(self) -> None:
        """
        Chroma DB and embedding model initialization.

        Configuration (hardcoded via CONFIG):
        - CHROMA_DB_PATH: "./chroma_db"
        - EMBEDDING_MODEL: "models/gemini-embedding-001"
        - TOP_K_DOCS: 5
        """
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG.model.EMBEDDING_MODEL
        )
        self._vectorstore = Chroma(
            persist_directory=CONFIG.model.CHROMA_DB_PATH,
            embedding_function=self._embeddings
        )
        self._retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": CONFIG.rl.TOP_K_DOCS}
        )

    def step_with_citations(self, question: str) -> RetrievalResult:
        """
        질문을 받아 컨텍스트 문자열과 참조 페이지 번호 목록을 함께 반환합니다.

        [Phase 1 — Source Citation]
        Chroma는 PDF 청킹 시 각 문서 조각에 {'page': N} 메타데이터를 저장합니다.
        이 메서드는 해당 메타데이터를 추출하여 중복 제거 후 정렬된 페이지 번호
        목록을 반환합니다.

        Args:
            question: 작업자의 자연어 질문
        Returns:
            RetrievalResult(context=..., source_pages=[...])
        """
        docs = self._retriever.invoke(question)

        # 컨텍스트 문자열 조합 (기존 동작 유지)
        context = "\n\n".join(doc.page_content for doc in docs)

        # 메타데이터에서 페이지 번호 추출 — 중복 제거 후 오름차순 정렬
        # PyPDFLoader는 page 키를 0-indexed로 저장하므로 +1 하여 1-indexed로 변환
        pages: list[int] = []
        for doc in docs:
            raw_page = doc.metadata.get("page")
            if raw_page is not None:
                try:
                    pages.append(int(raw_page) + 1)
                except (ValueError, TypeError):
                    pass
        source_pages = sorted(set(pages))

        return RetrievalResult(context=context, source_pages=source_pages)

    def step(self, question: str) -> str:
        """
        하위 호환용 인터페이스 — 컨텍스트 문자열만 반환합니다.

        [Self-RAG §3 "Retrieve" 토큰 단계 대응]

        Args:
            question: 작업자의 자연어 질문
        Returns:
            context: 검색된 문서 조각들을 이중 개행(\\n\\n)으로 연결한 문자열
        """
        return self.step_with_citations(question).context

    def reset(self) -> None:
        """
        환경 리셋 (향후 멀티턴 에피소드 지원 대비).
        현재는 Chroma DB가 무상태(stateless)이므로 구현 불필요.
        """
        pass
