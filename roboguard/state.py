"""
state.py — Shared state schema for the LangGraph pipeline.

All nodes read from and write to this TypedDict. The trajectory_log field
acts as episodic memory: it accumulates (answer, feedback, pass_fail) entries
across retries so the actor can reflect on its own prior mistakes.
"""
from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


class TrajectoryEntry(TypedDict):
    """A single retry attempt record.

    answer   : The answer generated in this attempt.
    feedback : Raw evaluation text from the LLM judge.
    pass_fail: Verdict token — "PASS" or "FAIL".
    """
    answer: str
    feedback: str
    pass_fail: str


class AgentState(TypedDict):
    """Shared state passed between pipeline nodes.

    Fields:
      question       : User's natural-language query.
      context        : Manual chunks retrieved from the vector store.
      answer         : Current answer produced by the actor.
      source_pages   : 1-indexed page numbers of the retrieved chunks.
      feedback       : Judge's verbal evaluation from the last attempt.
      pass_fail      : Latest verdict — "PASS" or "FAIL".
      retry_count    : Number of generation attempts so far.
      trajectory_log : Accumulated list of TrajectoryEntry dicts.
      image_b64      : Base64-encoded image for Vision RAG, or None.
    """
    question: str
    context: str
    answer: str
    source_pages: list

    feedback: str
    pass_fail: str
    retry_count: int

    trajectory_log: list

    image_b64: Optional[str]
