"""
environment.py — Chroma vector store retrieval layer.

Wraps the Chroma vector store in a simple step()-style interface, inspired
by the gym.Env pattern, so it can be swapped for a different backend
(Pinecone, Weaviate, etc.) by modifying only this file.

step_with_citations() returns the context string along with the source
page numbers extracted from Chroma's chunk metadata.
"""
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import CONFIG

load_dotenv()


@dataclass
class RetrievalResult:
    """Container for a vector store query result.

    context     : Retrieved document chunks joined by double newlines.
    source_pages: Deduplicated, sorted 1-indexed page numbers.
    """
    context: str
    source_pages: list[int] = field(default_factory=list)


class RetrievalEnvironment:
    """Chroma-backed RAG retrieval environment.

    Embeds the input question and retrieves the top-k most similar manual
    chunks from the persistent Chroma collection.

    Methods:
      reset()                : No-op reset for multi-turn episode support.
      step(question)         : Returns context string only (backward-compat).
      step_with_citations()  : Returns context + source page numbers.
    """

    def __init__(self) -> None:
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
        """Retrieve relevant chunks and return context with source pages.

        PyPDFLoader stores page indices as 0-based in chunk metadata, so
        each value is incremented by 1 to produce 1-based page numbers.

        Args:
            question: User's natural-language query.
        Returns:
            RetrievalResult(context=..., source_pages=[...])
        """
        docs = self._retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in docs)

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
        """Return only the context string (backward-compatible wrapper).

        Args:
            question: User's natural-language query.
        Returns:
            Retrieved chunks joined by double newlines.
        """
        return self.step_with_citations(question).context

    def reset(self) -> None:
        """No-op reset kept for multi-turn episode compatibility.

        Chroma is stateless, so no action is needed here.
        """
        pass
