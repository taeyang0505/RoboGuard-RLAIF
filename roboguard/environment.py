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
"""
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CONFIG

load_dotenv()


class RetrievalEnvironment:
    """
    Chroma Vector DB 기반 RAG 검색 환경.

    [Self-RAG §2 "Retrieval Model p(d|x)" 구현]
    질문 x를 받아 관련 문서 집합 D를 반환하는 확률 모델 p(d|x)를
    임베딩 기반 유사도 검색으로 근사합니다.

    [RL Environment 인터페이스]
    - reset(): 환경 초기화 (향후 멀티턴/에피소드 지원 대비)
    - step(question): 질문을 받아 컨텍스트를 반환 (gym.Env.step 패턴)
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

    def step(self, question: str) -> str:
        """
        질문을 받아 관련 매뉴얼 컨텍스트를 반환합니다.

        [Self-RAG §3 "Retrieve" 토큰 단계 대응]
        논문에서 [Retrieve] 토큰이 발생하면 검색기를 호출하듯,
        이 메서드가 호출되면 Vector DB에서 관련 문서를 가져옵니다.

        Args:
            question: 작업자의 자연어 질문
        Returns:
            context: 검색된 문서 조각들을 이중 개행(\\n\\n)으로 연결한 문자열
        """
        docs = self._retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)

    def reset(self) -> None:
        """
        환경 리셋 (향후 멀티턴 에피소드 지원 대비).
        현재는 Chroma DB가 무상태(stateless)이므로 구현 불필요.
        """
        pass
