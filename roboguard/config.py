"""
config.py — Centralized configuration for the RoboGuard pipeline.

All hyperparameters, file paths, and model names are defined here as frozen
dataclasses (immutable after creation) so that modifying this single file
propagates changes across the entire pipeline.

LangSmith tracing:
  After load_dotenv(), LangChain environment variables are explicitly bound
  to os.environ so that LangGraph automatically forwards traces to LangSmith.
"""
import logging
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


load_dotenv()

_logger = logging.getLogger(__name__)


# Bind LangSmith tracing variables from .env into os.environ so LangGraph
# picks them up automatically without requiring manual export in the shell.
_LANGSMITH_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
)
for _var in _LANGSMITH_VARS:
    _val = os.getenv(_var, "")
    if _val:
        os.environ.setdefault(_var, _val)

_langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
if not _langsmith_key:
    _logger.warning(
        "[RoboGuard] LANGCHAIN_API_KEY is not set. "
        "LangSmith tracing will be disabled. "
        "Add LANGCHAIN_API_KEY to your .env file to enable it."
    )
else:
    _logger.info(
        "[RoboGuard] LangSmith tracing enabled — project: %s",
        os.getenv("LANGCHAIN_PROJECT", "RoboGuard-RLAIF"),
    )


@dataclass(frozen=True)
class RLConfig:
    """Hyperparameters controlling the RLAIF self-correction loop.

    MAX_RETRIES  : Maximum number of revision attempts after a FAIL verdict.
    API_SLEEP_SEC: Seconds to wait between retries to respect API rate limits.
    TOP_K_DOCS   : Number of document chunks retrieved per query.
    """
    MAX_RETRIES: int = 3
    API_SLEEP_SEC: float = 2.0
    TOP_K_DOCS: int = 5


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and vector store path configuration.

    CHROMA_DB_PATH  : Persistent directory for the Chroma vector store.
    EMBEDDING_MODEL : Embedding model used to index and query chunks.
    LLM_MODEL       : Gemini model used for both generation and evaluation.
    LLM_TEMPERATURE : Fixed at 0.0 for deterministic, reproducible output.
    """
    CHROMA_DB_PATH: str = "./chroma_db"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    REPORT_PATH: str = "eval_report_v2.csv"


# Global singleton — imported by all modules.
CONFIG = AppConfig()
