"""
graph_builder.py — LangGraph node and edge configuration for RoboGuard.

Connects three nodes (retrieve → generate → evaluate) and attaches a
conditional edge so a FAIL verdict loops back to generate for revision.
On retry, the retrieval step is skipped and the existing context is reused;
only the actor receives the updated trajectory_log for self-correction.
The loop terminates on PASS or when MAX_RETRIES is reached.
"""
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from .config import CONFIG
from .state import AgentState, TrajectoryEntry
from .environment import RetrievalEnvironment
from .policy_actor import PolicyActor
from .reward_model import RewardModel

load_dotenv()


class RoboGuardGraph:
    """Assembles the RoboGuard RLAIF pipeline as a LangGraph workflow.

    Components:
      Environment  → _retrieve_node  (vector store lookup)
      Actor        → _generate_node  (initial generation / reflection)
      Evaluator    → _evaluate_node  (faithfulness scoring)

    Call build() to get a compiled, invokable graph.
    """

    def __init__(self) -> None:
        self._env = RetrievalEnvironment()
        self._actor = PolicyActor()
        self._reward_model = RewardModel()

    def _retrieve_node(self, state: AgentState) -> dict:
        """Query the vector store and return relevant context.

        Skipped on retries (retry_count > 0) — the same query produces the
        same results, so reusing the existing context avoids redundant API calls.

        Reads:  question, retry_count
        Writes: context, source_pages
        """
        retry_cnt = state.get("retry_count", 0)
        if retry_cnt == 0:
            print("[INFO] Retrieve: Querying Vector DB for relevant context.")
            result = self._env.step_with_citations(state["question"])
            print(
                f"[INFO] Retrieve: {len(result.source_pages)} unique page(s) referenced "
                f"— {result.source_pages}"
            )
            return {"context": result.context, "source_pages": result.source_pages}
        return {}

    def _generate_node(self, state: AgentState) -> dict:
        """Generate or revise an answer based on current state.

        First attempt uses the base policy; subsequent attempts use the
        reflection policy with the accumulated trajectory_log.

        Reads:  retry_count, context, question, trajectory_log, image_b64
        Writes: answer, retry_count
        """
        retry_cnt = state.get("retry_count", 0)
        trajectory_log = state.get("trajectory_log", [])
        source_pages: list[int] = state.get("source_pages", [])
        image_b64: str | None = state.get("image_b64")

        if retry_cnt == 0:
            print("[INFO] Generate: Producing initial response (base policy).")
            answer = self._actor.generate_initial(
                context=state["context"],
                question=state["question"],
                source_pages=source_pages,
                image_b64=image_b64,
            )
        else:
            print(
                f"[INFO] Generate: Revising response using {retry_cnt} prior attempt(s) "
                f"(retry {retry_cnt})."
            )
            answer = self._actor.reflect_and_refine(
                context=state["context"],
                question=state["question"],
                trajectory_log=trajectory_log,
                source_pages=source_pages,
                image_b64=image_b64,
            )

        return {
            "answer": answer,
            "retry_count": retry_cnt + 1
        }

    def _evaluate_node(self, state: AgentState) -> dict:
        """Run faithfulness verification and append the result to the trajectory.

        Reads:  context, answer, trajectory_log, image_b64
        Writes: feedback, pass_fail, trajectory_log
        """
        print("[INFO] Evaluate: Running fact-verification via LLM-as-a-judge.")

        image_b64: str | None = state.get("image_b64")
        signal = self._reward_model.score(
            context=state["context"],
            answer=state["answer"],
            image_b64=image_b64,
        )

        if signal.pass_fail == "FAIL":
            print(
                f"[WARN] Evaluate: FAIL (score={signal.score:+.1f}). "
                "Unverified content detected. Requesting revision."
            )
        else:
            print(f"[INFO] Evaluate: PASS (score={signal.score:+.1f}). Response verified against source document.")

        current_entry: TrajectoryEntry = {
            "answer": state["answer"],
            "feedback": signal.feedback,
            "pass_fail": signal.pass_fail
        }
        updated_log: list = list(state.get("trajectory_log", [])) + [current_entry]

        return {
            "feedback": signal.feedback,
            "pass_fail": signal.pass_fail,
            "trajectory_log": updated_log
        }

    def _should_continue(self, state: AgentState) -> str:
        """Route to 'end' on PASS or max retries, otherwise 'retry'.

        Returns:
            "end" or "retry"
        """
        if state["pass_fail"] == "PASS":
            return "end"

        if state["retry_count"] >= CONFIG.rl.MAX_RETRIES:
            print(
                f"[WARN] Router: Maximum retry limit ({CONFIG.rl.MAX_RETRIES}) reached. "
                "Terminating with best available response."
            )
            return "end"

        time.sleep(CONFIG.rl.API_SLEEP_SEC)
        return "retry"

    def build(self):
        """Assemble and compile the LangGraph workflow.

        Graph structure:
          START → retrieve → generate → evaluate
                                 ↑           │
                           (FAIL, retry) ────┘
                                           PASS or MAX_RETRIES → END

        Returns:
            A compiled LangGraph graph ready for app.invoke(initial_state).
        """
        workflow = StateGraph(AgentState)  # type: ignore[type-var]

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate", self._evaluate_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"end": END, "retry": "generate"}
        )

        return workflow.compile()
