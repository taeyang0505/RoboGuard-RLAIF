"""
config.py — 중앙 설정 관리자 (Single Source of Truth)
======================================================
모든 하이퍼파라미터, 경로, 모델명을 한 곳에서 관리합니다.
단일 파일만 수정하면 전체 파이프라인 동작이 바뀝니다.

설계 원칙: 12-Factor App의 "Config" 팩터 + dataclass frozen=True (불변성 보장)

Phase 2 — LangSmith MLOps:
  load_dotenv()로 .env를 로드한 후, LangChain 환경변수를 os.environ에
  명시적으로 바인딩합니다. LangGraph 실행시 자동으로 LangSmith에
  트레이스가 전송됩니다 (코드 변경 무필요).
"""
import logging
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


# ── 환경변수 로드 (애플리케이션 시작 시 가장 먼저 실행) ────────────────
load_dotenv()

_logger = logging.getLogger(__name__)


# ── Phase 2: LangSmith MLOps 모니터링 환경변수 바인딩 ────────────────────
# LangChain의 LANGCHAIN_TRACING_V2 환경변수가 True이면
# LangGraph의 모든 노드 실행이 자동으로 LangSmith로 트레이스됩니다.
# 코드 변경 없이 .env 파일만 수정하면 동작합니다.
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
        "[RoboGuard] LANGCHAIN_API_KEY가 설정되지 않았습니다. "
        "LangSmith 트레이싱이 비활성화됩니다. "
        ".env 파일에 LANGCHAIN_API_KEY를 입력해 주세요."
    )
else:
    _logger.info(
        "[RoboGuard] LangSmith 트레이싱 활성화 — 프로젝트: %s",
        os.getenv("LANGCHAIN_PROJECT", "RoboGuard-RLAIF"),
    )


# ──────────────────────────────────────────────────────────────
# RL 하이퍼파라미터 설정
# [InstructGPT §2.2 "Reward Model Training" 참조]
# PPO 반복 횟수 개념을 MAX_RETRIES로 단순화
# [Self-RAG §3 "Retrieval" 참조]
# TOP_K_DOCS는 논문의 검색 문서 수 k에 대응
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RLConfig:
    """
    RL 하이퍼파라미터.

    MAX_RETRIES  : RL Policy Update 최대 반복 횟수 (InstructGPT PPO iteration 대응)
    API_SLEEP_SEC: Google API Rate Limit 방지용 대기 시간 (초)
    TOP_K_DOCS   : 검색 문서 수 (Self-RAG §3 'k' 파라미터)
    """
    MAX_RETRIES: int = 3
    API_SLEEP_SEC: float = 2.0
    TOP_K_DOCS: int = 5


# ──────────────────────────────────────────────────────────────
# 모델 및 인프라 설정
# Hardcoded environment configuration — do not modify manually
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """
    모델 및 Vector DB 경로 설정.

    CHROMA_DB_PATH  : Chroma Vector DB 저장 디렉토리
    EMBEDDING_MODEL : Google Generative AI 임베딩 모델
    LLM_MODEL       : Gemini LLM 모델명
    LLM_TEMPERATURE : Fixed at 0 for deterministic output
    """
    CHROMA_DB_PATH: str = "./chroma_db"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.0


# ──────────────────────────────────────────────────────────────
# 전체 앱 설정 컨테이너
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    """전체 애플리케이션 설정 컨테이너 (Composite Pattern)."""
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    REPORT_PATH: str = "eval_report_v2.csv"


# 전역 싱글톤 인스턴스 — 모든 모듈이 import 해서 사용
CONFIG = AppConfig()
