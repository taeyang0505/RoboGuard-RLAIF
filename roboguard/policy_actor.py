"""
policy_actor.py — Answer generation actor for the RoboGuard RLAIF pipeline.

Provides two generation modes:
  generate_initial()   : Produces the first answer given retrieved context.
  reflect_and_refine() : Revises a prior answer using the accumulated failure log.

When an image is provided (image_b64), the prompt and image are passed together
so Gemini Vision can cross-reference visual observations against the manual.
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .config import CONFIG

load_dotenv()


def _format_citation(source_pages: list[int]) -> str:
    """Return a formatted citation string for the given page numbers.

    Returns an empty string when source_pages is empty.
    """
    if not source_pages:
        return ""
    pages_str = ", ".join(f"p. {p}" for p in source_pages)
    return f"[Reference: UR10e User Manual, {pages_str}]"


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


# On retry, the full trajectory_log is injected so the model can reason about
# what specifically went wrong in each prior attempt.
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
    """Serialize the trajectory log into a text block for the reflection prompt.

    Each entry's answer is capped at 250 characters and feedback at 400
    to avoid exceeding the model's context window.
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
    """Answer generation actor for the RLAIF pipeline.

    Calls generate_initial() for the first attempt and reflect_and_refine()
    on subsequent retries. Automatically switches to multimodal mode when
    an image is provided.
    """

    def __init__(self) -> None:
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
        """Generate the first answer without any prior feedback.

        Args:
            context      : Retrieved manual context from the vector store.
            question     : User's natural-language question.
            source_pages : Page numbers from the retrieved chunks.
            image_b64    : Base64-encoded image, or None for text-only mode.
        Returns:
            Generated answer string including a trailing citation line.
        """
        citation = _format_citation(source_pages or [])
        prompt_text = _INITIAL_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            citation=citation,
        )
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
        """Revise a prior answer using the accumulated failure history.

        The full trajectory_log is injected into the prompt so the model can
        identify and correct the specific issues from each previous attempt.

        Args:
            context        : Retrieved manual context from the vector store.
            question       : User's natural-language question.
            trajectory_log : List of (answer, feedback, pass_fail) entries.
            source_pages   : Page numbers from the retrieved chunks.
            image_b64      : Base64-encoded image, or None for text-only mode.
        Returns:
            Revised answer string including a trailing citation line.
        """
        trajectory_summary = _format_trajectory(trajectory_log)
        citation = _format_citation(source_pages or [])
        prompt_text = _REFLECTION_PROMPT_TEMPLATE.format(
            trajectory_summary=trajectory_summary,
            context=context,
            question=question,
            citation=citation,
        )
        if image_b64:
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])
        else:
            message = HumanMessage(content=prompt_text)
        return str(self._llm.invoke([message]).content)
