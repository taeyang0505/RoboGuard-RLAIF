"""
config.py — 중앙 설정 관리자 (Single Source of Truth)
======================================================
모든 하이퍼파라미터, 경로, 모델명을 한 곳에서 관리합니다.
단일 파일만 수정하면 전체 파이프라인 동작이 바뀝니다.

설계 원칙: 12-Factor App의 "Config" 팩터 + dataclass frozen=True (불변성 보장)
"""
from dataclasses import dataclass, field


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
