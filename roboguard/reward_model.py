"""
reward_model.py — LLM-as-a-judge Reward Model (Fact-Verification)
=============================================================
[InstructGPT §2.2 "Reward Model Training" + Self-RAG §3 "Critique Tokens" 결합]

InstructGPT의 기여 (p.4):
  "인간 레이블러가 직접 선호도를 평가하는 대신, 학습된 Reward Model RM이
   (prompt x, response y) 쌍에 스칼라 보상 r = RM(x, y)를 자동 계산한다.
   이 보상이 PPO 업데이트의 신호로 사용된다."

Self-RAG의 기여 (p.5):
  "모델은 [IsREL], [IsSUP], [IsUSE] 크리틱 토큰을 생성하여
   검색 결과의 관련성, 사실 지지도, 유용성을 자가 평가한다."

이 모듈에서 두 아이디어를 결합합니다:
  - LLM(Gemini)이 인간 레이블러를 대체하는 RM 역할 (InstructGPT)
  - [PASS]/[FAIL] 구조화 크리틱 토큰으로 출력 파싱 (Self-RAG 단순화)
  - PASS=+1.0, FAIL=-1.0 스칼라 보상 (InstructGPT KL 패널티 개념의 이산화)

[v2.2 Robust Parsing — Bug Fix]
  판사 LLM이 대괄호 없이 "PASS The response..." 형태로 판정을 반환할 경우에도
  올바르게 PASS(+1.0)로 인식하도록 정규식 기반 유연한 파싱 로직을 적용합니다.
  탐지 우선순위: 첫 줄(first line) → 전체 텍스트 앞 50자 → 전체 텍스트
"""
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import CONFIG

load_dotenv()


# ── 스칼라 보상 상수 ──────────────────────────────────────────────────────
# [InstructGPT §2.2]: r_θ(x, y) — 보상 모델 출력값을 이산화
REWARD_PASS: float = +1.0   # 팩트 준수 → 양의 보상 (Policy 강화)
REWARD_FAIL: float = -1.0   # Unverified content detected — negative reward (penalty)


# ── Critic 프롬프트 템플릿 ────────────────────────────────────────────────
# [Self-RAG §3 "Critique Token" 구조를 자연어 프롬프트로 내재화]
_CRITIC_PROMPT_TEMPLATE = """\
Task: Evaluate whether the response below is strictly grounded in the provided reference document.

Evaluation Criteria:
- Output [PASS] on the first line if the response contains only information explicitly stated \
in the reference document.
- Output [FAIL] on the first line if the response contains any information not found in the \
reference document, including inferred, assumed, or externally sourced content.
- After the verdict, provide a concise rationale (1-2 sentences).

Reference Document:
{context}

Response:
{answer}

Evaluation:"""


@dataclass
class RewardSignal:
    """
    보상 함수의 구조화된 출력.

    [InstructGPT §2.2]: 스칼라 보상(score)을 통해 Policy를 업데이트
    [Reflexion §3.3]:   언어적 피드백(feedback)을 Episodic Memory에 저장
    두 논문의 출력을 하나의 데이터 구조로 통합합니다.

    pass_fail : "PASS" 또는 "FAIL" (Self-RAG Critique Token)
    feedback  : Critic LLM의 상세 평가문 (Reflexion Self-Reflection 메모리용)
    score     : 스칼라 보상 (+1.0 / -1.0) (InstructGPT Reward)
    """
    pass_fail: str
    feedback: str
    score: float


class RewardModel:
    """
    LLM-as-a-judge based fact-verification reward model.

    [InstructGPT §2.2 "Reward Model RM(x, y)"]
    (prompt=context+question, response=answer) 쌍을 입력받아
    스칼라 보상과 언어적 피드백을 반환합니다.

    [Self-RAG §3 "ISREL / ISSUP / ISUSE Critique Tokens"]
    [PASS]/[FAIL] 구조화 토큰으로 출력의 파싱 신뢰도를 높입니다.

    설계 결정:
    - 온도 0 고정: 판정의 결정론적 일관성 확보 (같은 답변 → 같은 판정)
    - 별도 LLM 인스턴스: Actor와 Critic을 분리하여 역할 명확화
    """

    def __init__(self) -> None:
        """Critic LLM 초기화 (온도 0 고정: 결정론적 판정 보장)."""
        self._llm = ChatGoogleGenerativeAI(
            model=CONFIG.model.LLM_MODEL,
            temperature=CONFIG.model.LLM_TEMPERATURE
        )

    def score(self, context: str, answer: str) -> RewardSignal:
        """
        (context, answer) 쌍에 보상 신호를 계산합니다.

        [InstructGPT §2.2]: r = RM(x, y) — 스칼라 보상 산출
        [Self-RAG §3]:       크리틱 토큰 [PASS]/[FAIL] 파싱

        Args:
            context : Vector DB에서 검색된 매뉴얼 텍스트 (ground truth)
            answer  : PolicyActor가 생성한 답변
        Returns:
            RewardSignal: pass_fail, feedback(언어적 피드백), score(스칼라)
        """
        prompt = _CRITIC_PROMPT_TEMPLATE.format(
            context=context,
            answer=answer
        )
        raw_output = str(self._llm.invoke(prompt).content)

        # ── Robust Critique Token 파싱 [v2.2 Bug Fix] ──────────────────
        # Self-RAG 스타일의 [PASS]/[FAIL] 구조화 토큰뿐 아니라,
        # 판사 LLM이 "PASS The response..." 처럼 대괄호 없이 반환하는 경우도
        # 정상적으로 PASS(+1.0)로 인식합니다.
        #
        # 탐지 전략 (우선순위 순):
        #   1) 첫 줄(first line)에 독립 단어 'PASS' 또는 'FAIL' 포함 여부
        #   2) 전체 텍스트 앞 50자 내 독립 단어 탐지 (fallback)
        #   3) 위 모두 해당 없을 시 → 보수적으로 FAIL 처리
        _PASS_RE = re.compile(r'\bPASS\b', re.IGNORECASE)
        _FAIL_RE = re.compile(r'\bFAIL\b', re.IGNORECASE)

        first_line = raw_output.strip().splitlines()[0] if raw_output.strip() else ""
        prefix_50  = raw_output.strip()[:50]

        if _PASS_RE.search(first_line) and not _FAIL_RE.search(first_line):
            pass_fail = "PASS"
        elif _FAIL_RE.search(first_line) and not _PASS_RE.search(first_line):
            pass_fail = "FAIL"
        elif _PASS_RE.search(prefix_50):
            pass_fail = "PASS"
        elif _FAIL_RE.search(prefix_50):
            pass_fail = "FAIL"
        else:
            # 최후 fallback: 전체 텍스트에서 탐지 (기존 동작 유지)
            pass_fail = "PASS" if _PASS_RE.search(raw_output) else "FAIL"

        scalar_score = REWARD_PASS if pass_fail == "PASS" else REWARD_FAIL

        return RewardSignal(
            pass_fail=pass_fail,
            feedback=raw_output,
            score=scalar_score
        )
