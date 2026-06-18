"""
state.py — LangGraph AgentState 정의 (RL 실패 기억 메모리 포함)
================================================================
[Reflexion §3.2 "Episodic Memory Buffer" 핵심 구현]

Reflexion 논문의 핵심 주장 (p.4):
  "에이전트는 과거 실패 궤적(trajectory)을 에피소딕 메모리에 저장하고,
   이를 다음 시도의 컨텍스트로 주입함으로써 명시적 가중치 업데이트 없이
   정책(Policy)을 언어적으로 업데이트할 수 있다."

AgentState의 trajectory_log 필드가 이 에피소딕 메모리 역할을 합니다.
"""
from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


class TrajectoryEntry(TypedDict):
    """
    단일 에피소드(시도)의 완전한 기록.

    [Reflexion §3.3 "Self-Reflection" 참조]
    각 에피소드의 (행동, 피드백, 결과)를 저장하여
    다음 Actor 호출 시 반성의 재료로 활용됩니다.

    answer   : 해당 시도에서 Actor가 생성한 답변
    feedback : Critic(LLM-as-a-judge)의 언어적 피드백 원문
    pass_fail: 보상 신호 ("PASS" = +1.0 | "FAIL" = -1.0)
    """
    answer: str
    feedback: str
    pass_fail: str


class AgentState(TypedDict):
    """
    LangGraph 전체 파이프라인의 공유 상태 (에이전트의 '메모장').

    논문별 필드 매핑:
    ┌─────────────────┬──────────────────────────────────────────────────┐
    │ 필드            │ 논문 개념                                         │
    ├─────────────────┼──────────────────────────────────────────────────┤
    │ question        │ InstructGPT §2.1 — SFT 입력 프롬프트             │
    │ context         │ Self-RAG §2 — Retrieval 결과 passage D            │
    │ answer          │ InstructGPT §2.1 — SFT 출력 응답 y               │
    │ source_pages    │ Source Citation                                  │
    │ feedback        │ Reflexion §3.3 — Self-Reflection 언어적 피드백   │
    │ pass_fail       │ Self-RAG §3 — Critique Token ([PASS]/[FAIL])     │
    │ retry_count     │ InstructGPT §2.2 — RL iteration 카운터          │
    │ trajectory_log  │ Reflexion §3.2 — Episodic Memory Buffer          │
    │ image_b64       │ Phase 3 — Multimodal Vision RAG 이미지 입력      │
    └─────────────────┴──────────────────────────────────────────────────┘
    """
    # ── 기본 RAG 입출력 ──────────────────────────────────────────────────────
    question: str        # 작업자의 자연어 질문
    context: str         # Vector DB에서 검색된 매뉴얼 컨텍스트 (Self-RAG passage D)
    answer: str          # Actor가 생성한 현재 답변
    source_pages: list   # 검색된 청크의 원본 페이지 번호 목록 (Source Citation)

    # ── RL 제어 상태 ──────────────────────────────────────────────────────────
    feedback: str        # Critic의 언어적 피드백 (Reflexion Verbal Reinforcement)
    pass_fail: str       # 보상 신호: "PASS" 또는 "FAIL" (Self-RAG Critique Token)
    retry_count: int     # 현재 재시도 횟수 (InstructGPT RL iteration 카운터)

    # ── Reflexion 에피소딕 메모리 ─────────────────────────────────────────
    trajectory_log: list  # TrajectoryEntry 리스트 — 누적 실패 궤적 저장소

    # ── Phase 3: Multimodal Vision RAG ───────────────────────────────────
    image_b64: Optional[str]  # Base64 인코딩 이미지 (없으면 None — 텍스트 전용 모드)
