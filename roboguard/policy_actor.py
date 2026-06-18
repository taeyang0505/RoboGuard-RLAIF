"""
policy_actor.py — RL Actor (정책 네트워크)
==========================================
[Reflexion §3.1 "Actor" + §3.3 "Self-Reflection" 핵심 구현]

Reflexion 논문의 핵심 주장 (p.4):
  "Actor는 두 가지 모드로 동작한다:
   1) 기본 정책(Base Policy): 환경 관찰만으로 행동 생성
   2) 자기 반성 정책(Reflective Policy): 에피소딕 메모리 전체를 읽고
      이전 실패를 반성한 뒤 행동을 재생성

   이 두 번째 모드가 명시적 가중치 업데이트 없이
   Verbal Policy Update를 달성한다."

generate_initial()   → 기본 정책 (InstructGPT SFT 단계 대응)
reflect_and_refine() → 자기 반성 정책 (Reflexion Verbal RL 핵심)

Phase 1 — Source Citation:
  source_pages 파라미터를 받아 프롬프트에 출처 명시 지시를 포함합니다.
  LLM은 참조 페이지를 기반으로 답변 말미에 출처를 명시합니다.
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .config import CONFIG

load_dotenv()


# ── 출처 포맷 헬퍼 ────────────────────────────────────────────────────────

def _format_citation(source_pages: list[int]) -> str:
    """
    페이지 번호 목록을 출처 문자열로 변환합니다.

    Args:
        source_pages: 정렬된 1-indexed 페이지 번호 목록
    Returns:
        "[Reference: UR10e Manual, p. 12, 15, 23]" 형태의 문자열,
        또는 페이지 정보가 없을 경우 빈 문자열
    """
    if not source_pages:
        return ""
    pages_str = ", ".join(f"p. {p}" for p in source_pages)
    return f"[Reference: UR10e User Manual, {pages_str}]"


# ── 프롬프트 템플릿 ───────────────────────────────────────────────────────

_INITIAL_PROMPT_TEMPLATE = """\
Role: Technical documentation assistant for UR10e industrial robot systems.

Task: Answer the question below using only the information provided in the reference document.

Constraints:
- Do not use any knowledge outside the reference document.
- Do not infer, estimate, or extrapolate values not explicitly stated.
- If the document does not contain relevant information, respond with: \
"The requested information is not available in the provided documentation."
- When citing numerical values (current, voltage, weight, distance, etc.), \
use only the values explicitly stated in the document.
- If an image is attached by the user, perform a precise visual analysis of the image \
(error codes, component conditions, connector states, LED indicators, warning labels, etc.) \
and cross-reference your visual findings against the reference document context before answering.
- At the end of your response, append the following citation line exactly as provided, \
on a new line preceded by a blank line:
  {citation}

Reference Document:
{context}

Question: {question}
Answer:"""


# [Reflexion §3.3 "Self-Reflection Prompt"]
# trajectory_log 전체(누적 실패 기억)를 컨텍스트에 주입하는 것이 핵심
_REFLECTION_PROMPT_TEMPLATE = """\
Role: Technical documentation assistant for UR10e industrial robot systems.

Context: Previous response attempts were flagged as containing unverified information.
Review the prior attempt log below and produce a corrected response.

Prior Attempt Log:
{trajectory_summary}

Revision Instructions:
- Address all issues identified in the prior attempt log.
- Use only information explicitly stated in the reference document below.
- Do not introduce any knowledge from outside the reference document.
- If the document does not contain relevant information, respond with: \
"The requested information is not available in the provided documentation."
- When citing numerical values (current, voltage, weight, distance, etc.), \
use only the values explicitly stated in the document.
- If an image is attached by the user, perform a precise visual analysis of the image \
(error codes, component conditions, connector states, LED indicators, warning labels, etc.) \
and cross-reference your visual findings against the reference document context before answering.
- At the end of your response, append the following citation line exactly as provided, \
on a new line preceded by a blank line:
  {citation}

Reference Document:
{context}

Question: {question}
Revised Answer:"""


def _format_trajectory(trajectory_log: list) -> str:
    """
    Formats the episodic memory buffer into a structured text block for LLM consumption.

    [Reflexion §3.2 "Episodic Memory Buffer"]
    Each attempt entry (answer, feedback) is numbered and truncated to stay within token limits.

    Args:
        trajectory_log: List of TrajectoryEntry dicts
    Returns:
        Formatted string of prior attempt records
    """
    lines: list[str] = []
    for i, entry in enumerate(trajectory_log, start=1):
        answer_preview = str(entry.get("answer", ""))[:250]
        feedback_preview = str(entry.get("feedback", ""))[:400]
        lines.append(f"--- [Attempt {i}] ---")
        lines.append(f"  Response preview: {answer_preview}...")
        lines.append(f"  Evaluator feedback: {feedback_preview}...")
        lines.append("")
    return "\n".join(lines)



class PolicyActor:
    """
    RL Actor — 답변 생성 정책 네트워크.

    [Reflexion §3.1 "Actor"]
    현재 환경 상태(context, question)를 관찰하고
    행동(answer)을 생성하는 정책 π를 구현합니다.

    두 가지 동작 모드:
    1. generate_initial()   : 기본 정책 π_base (InstructGPT SFT 대응)
    2. reflect_and_refine() : 반성 정책 π_reflex (Reflexion Verbal RL 핵심)

    [InstructGPT §2.1 "SFT Model"]
    generate_initial()은 지도 학습(SFT)으로 초기화된 기본 정책에 대응합니다.
    """

    def __init__(self) -> None:
        """Actor LLM 초기화."""
        self._llm = ChatGoogleGenerativeAI(
            model=CONFIG.model.LLM_MODEL,
            temperature=CONFIG.model.LLM_TEMPERATURE
        )

    def generate_initial(
        self,
        context: str,
        question: str,
        source_pages: list[int] | None = None,
        image_b64: str | None = None,
    ) -> str:
        """
        첫 번째 답변 생성 — 기본 정책 실행 (π_base).

        [InstructGPT §2.1 "SFT Policy"]
        피드백 없이 매뉴얼(context)과 질문(question)만으로 답변을 생성합니다.

        Args:
            context      : 검색된 매뉴얼 컨텍스트
            question     : 작업자 질문
            source_pages : 참조 페이지 번호 목록 (Source Citation)
            image_b64    : Base64 인코딩 이미지 문자열 (Phase 3 Vision RAG, 없으면 None)
        Returns:
            생성된 답변 문자열 (말미에 출처 표기 포함)
        """
        citation = _format_citation(source_pages or [])
        prompt_text = _INITIAL_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            citation=citation,
        )
        # ── Phase 3: 멀티모달 / 텍스트 전용 분기 ────────────────────────────
        # 항상 list[HumanMessage] 형태로 전달 → Sequence[MessageLikeRepresentation] 타입 보장
        if image_b64:
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])
        else:
            message = HumanMessage(content=prompt_text)
        return str(self._llm.invoke([message]).content)

    def reflect_and_refine(
        self,
        context: str,
        question: str,
        trajectory_log: list,
        source_pages: list[int] | None = None,
        image_b64: str | None = None,
    ) -> str:
        """
        실패 궤적을 읽고 자기 반성 후 답변을 재작성 — 반성 정책 (π_reflex).

        [Reflexion §3.3 "Self-Reflection" 핵심 구현]
        trajectory_log 전체(누적 실패 기억)를 LLM 컨텍스트에 주입하여
        명시적 그래디언트(가중치 업데이트) 없이 Verbal Policy Update를 달성합니다.

        Args:
            context        : 검색된 매뉴얼 컨텍스트
            question       : 작업자 질문
            trajectory_log : 과거 모든 시도의 (answer, feedback, pass_fail) 리스트
            source_pages   : 참조 페이지 번호 목록 (Source Citation)
            image_b64      : Base64 인코딩 이미지 문자열 (Phase 3 Vision RAG, 없으면 None)
        Returns:
            재작성된 답변 문자열 (말미에 출처 표기 포함)
        """
        trajectory_summary = _format_trajectory(trajectory_log)
        citation = _format_citation(source_pages or [])
        prompt_text = _REFLECTION_PROMPT_TEMPLATE.format(
            trajectory_summary=trajectory_summary,
            context=context,
            question=question,
            citation=citation,
        )
        # ── Phase 3: 멀티모달 / 텍스트 전용 분기 ────────────────────────────
        # 항상 list[HumanMessage] 형태로 전달 → Sequence[MessageLikeRepresentation] 타입 보장
        if image_b64:
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])
        else:
            message = HumanMessage(content=prompt_text)
        return str(self._llm.invoke([message]).content)
