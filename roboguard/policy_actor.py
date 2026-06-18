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
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import CONFIG

load_dotenv()


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

    def generate_initial(self, context: str, question: str) -> str:
        """
        첫 번째 답변 생성 — 기본 정책 실행 (π_base).

        [InstructGPT §2.1 "SFT Policy"]
        피드백 없이 매뉴얼(context)과 질문(question)만으로 답변을 생성합니다.
        이는 InstructGPT의 SFT 단계에서 기본 정책이 행동을 출력하는 것과 동일합니다.

        Args:
            context : 검색된 매뉴얼 컨텍스트
            question: 작업자 질문
        Returns:
            생성된 답변 문자열
        """
        prompt = _INITIAL_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        return str(self._llm.invoke(prompt).content)

    def reflect_and_refine(
        self,
        context: str,
        question: str,
        trajectory_log: list
    ) -> str:
        """
        실패 궤적을 읽고 자기 반성 후 답변을 재작성 — 반성 정책 (π_reflex).

        [Reflexion §3.3 "Self-Reflection" 핵심 구현]
        trajectory_log 전체(누적 실패 기억)를 LLM 컨텍스트에 주입하여
        명시적 그래디언트(가중치 업데이트) 없이 Verbal Policy Update를 달성합니다.

        "언어 모델 자체가 Policy Network이자 Value Function이다" — Reflexion p.5

        Args:
            context        : 검색된 매뉴얼 컨텍스트
            question       : 작업자 질문
            trajectory_log : 과거 모든 시도의 (answer, feedback, pass_fail) 리스트
        Returns:
            재작성된 답변 문자열
        """
        trajectory_summary = _format_trajectory(trajectory_log)
        prompt = _REFLECTION_PROMPT_TEMPLATE.format(
            trajectory_summary=trajectory_summary,
            context=context,
            question=question
        )
        return str(self._llm.invoke(prompt).content)
