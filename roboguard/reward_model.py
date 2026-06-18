"""
reward_model.py — LLM-as-a-judge faithfulness verifier.

Evaluates whether a generated answer is strictly grounded in the retrieved
manual context. Returns a binary verdict (PASS / FAIL), a scalar reward
(+1.0 / -1.0), and the judge's raw feedback for use in the next retry.

When an image is provided, it is forwarded to the judge model as well.
This prevents the judge from misclassifying image-derived visual descriptions
as hallucinations — those observations are grounded in the user's image,
not in the reference document text.

Verdict parsing is regex-based and handles cases where the model omits
the surrounding brackets (e.g., "PASS The response..." instead of "[PASS]").
"""
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import CONFIG

load_dotenv()


REWARD_PASS: float = +1.0
REWARD_FAIL: float = -1.0


_CRITIC_PROMPT_TEMPLATE = """\
Task: Evaluate whether the response below is strictly grounded in the provided reference document.

Evaluation Criteria:
- Output [PASS] on the first line if the response contains only information explicitly stated \
in the reference document.
- Output [FAIL] on the first line if the response contains any information not found in the \
reference document, including inferred, assumed, or externally sourced content.
- After the verdict, provide a concise rationale (1-2 sentences).

Critical Exception — Multimodal Visual Analysis:
  If the user attached an image (the image is provided to you alongside this prompt), \
the response may contain visual descriptions derived directly from that image \
(e.g., robot color, component condition, cable routing, error codes on screen, \
LED indicator states, warning labels, visible damage).
  You MUST NOT penalize or mark as [FAIL] for such visual observations. \
Those descriptions are grounded in the user-provided image, NOT in the reference document.
  Evaluate ONLY technical specifications, numerical values, and procedural \
claims against the provided Reference Document text.

Reference Document:
{context}

Response:
{answer}

Evaluation:"""


@dataclass
class RewardSignal:
    """Output container for the reward model.

    pass_fail : "PASS" or "FAIL"
    feedback  : The judge's full evaluation text, used as reflection material.
    score     : Scalar reward (+1.0 for PASS, -1.0 for FAIL).
    """
    pass_fail: str
    feedback: str
    score: float


class RewardModel:
    """LLM-as-a-judge faithfulness verifier.

    Takes a (context, answer) pair and returns a scalar reward with verbal
    feedback. Temperature is fixed at 0 for consistent verdicts. Uses a
    separate LLM instance from the actor to avoid interference.
    """

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(
            model=CONFIG.model.LLM_MODEL,
            temperature=CONFIG.model.LLM_TEMPERATURE
        )

    def score(
        self,
        context: str,
        answer: str,
        image_b64: str | None = None,
    ) -> RewardSignal:
        """Evaluate answer faithfulness and return a reward signal.

        Forwards the image to the judge when provided, so visual observations
        made by the actor are not wrongly penalised as hallucinations.

        Args:
            context   : Manual text retrieved from the vector store (ground truth).
            answer    : Response produced by PolicyActor.
            image_b64 : Base64-encoded image, or None for text-only mode.
        Returns:
            RewardSignal with pass_fail, feedback, and scalar score.
        """
        prompt = _CRITIC_PROMPT_TEMPLATE.format(
            context=context,
            answer=answer
        )
        if image_b64:
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])
        else:
            message = HumanMessage(content=prompt)
        raw_output = str(self._llm.invoke([message]).content)

        # Parse PASS/FAIL verdict.
        # Priority: (1) first line, (2) first 50 chars, (3) full text fallback.
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
            pass_fail = "PASS" if _PASS_RE.search(raw_output) else "FAIL"

        scalar_score = REWARD_PASS if pass_fail == "PASS" else REWARD_FAIL

        return RewardSignal(
            pass_fail=pass_fail,
            feedback=raw_output,
            score=scalar_score
        )
