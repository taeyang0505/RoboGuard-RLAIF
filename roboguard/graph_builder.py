"""
graph_builder.py — LangGraph 노드/엣지 라우팅 (RL 루프 설계)
=============================================================
[Reflexion §3 "Decision Making Loop" 전체 구조 구현]

Reflexion 논문의 에이전트 루프 (p.4):
  Actor(행동 생성)
    → Evaluator(보상 계산, 환경 피드백)
    → Self-Reflector(언어적 반성, 에피소딕 메모리 업데이트)
    → Actor (반성 내용을 반영하여 재시도)
    → ... (PASS 또는 MAX_RETRIES 도달 시 종료)

이를 LangGraph의 Cyclic Directed Graph로 구현합니다:
  retrieve → generate → evaluate
                ↑              │
          (RL Loop: FAIL) ─────┘
                               PASS → END

[4_rlaif_agent.py 베이스라인의 핵심 로직을 OOP로 캡슐화]
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
    """
    RoboGuard RLAIF 파이프라인 빌더 (Builder Pattern).

    [Reflexion §3 "Language Agent" 3대 컴포넌트를 LangGraph로 조합]
    1. Environment  → _retrieve_node  (RetrievalEnvironment.step())
    2. Actor        → _generate_node  (PolicyActor.generate / reflect)
    3. Evaluator    → _evaluate_node  (RewardModel.score())

    빌더 패턴(build() 메서드)을 사용하여 그래프 조립과 실행을 분리합니다.
    이를 통해 테스트 시 개별 노드를 독립적으로 검증할 수 있습니다.
    """

    def __init__(self) -> None:
        """
        핵심 컴포넌트 의존성 주입 초기화.

        각 컴포넌트는 독립적으로 교체 가능한 인터페이스를 가집니다.
        (Dependency Inversion Principle)
        """
        self._env = RetrievalEnvironment()
        self._actor = PolicyActor()
        self._reward_model = RewardModel()

    # ─── 노드 1: 검색 환경 (Environment) ────────────────────────────────────
    def _retrieve_node(self, state: AgentState) -> dict:
        """
        [Self-RAG §2 "Retrieve" 단계 + Reflexion §3.1 "Environment" 관찰]

        Vector DB에서 관련 매뉴얼 컨텍스트와 참조 페이지 번호를 검색합니다.
        RL 루프 재시도 시에는 검색을 생략하고 기존 context/source_pages를 재사용합니다.
        (API 호출 최소화 + Self-RAG의 선택적 검색 정신 반영)

        State 읽기: question, retry_count
        State 쓰기: context, source_pages
        """
        retry_cnt = state.get("retry_count", 0)
        # 첫 시도일 때만 검색 (재시도 시 동일 질문이므로 결과 동일)
        if retry_cnt == 0:
            print("[INFO] Retrieve: Querying Vector DB for relevant context.")
            result = self._env.step_with_citations(state["question"])
            print(
                f"[INFO] Retrieve: {len(result.source_pages)} unique page(s) referenced "
                f"— {result.source_pages}"
            )
            return {"context": result.context, "source_pages": result.source_pages}
        # 재시도: 기존 context/source_pages 유지 (state 변경 없음)
        return {}

    # ─── 노드 2: Actor — 정책 네트워크 ─────────────────────────────────────
    def _generate_node(self, state: AgentState) -> dict:
        """
        [Reflexion §3.1 "Actor" + §3.3 "Self-Reflection"]

        첫 시도:  기본 정책 π_base로 초기 답변 생성
        재시도:   trajectory_log(에피소딕 메모리) 전체를 읽고
                  자기 반성 후 답변 재작성 (Verbal Policy Update)

        State 읽기: retry_count, context, question, trajectory_log
        State 쓰기: answer, retry_count
        """
        retry_cnt = state.get("retry_count", 0)
        trajectory_log = state.get("trajectory_log", [])

        source_pages: list[int] = state.get("source_pages", [])
        image_b64: str | None = state.get("image_b64")  # Phase 3: Vision RAG

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

    # ─── 노드 3: Reward Model (Evaluator / Critic) ──────────────────────────
    def _evaluate_node(self, state: AgentState) -> dict:
        """
        [InstructGPT §2.2 "Reward Model" + Self-RAG §3 "Critique Token"]

        현재 답변에 보상 신호를 계산하고 trajectory_log에 기록합니다.
        RewardSignal(pass_fail, feedback, score)를 반환받아
        State에 기록하고 Reflexion 에피소딕 메모리를 업데이트합니다.

        State 읽기: context, answer, trajectory_log
        State 쓰기: feedback, pass_fail, trajectory_log
        """
        print("[INFO] Evaluate: Running fact-verification via LLM-as-a-judge.")

        image_b64: str | None = state.get("image_b64")  # Phase 3: 판사도 동일 이미지 수신
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

        # [Reflexion §3.2] 현재 에피소드를 Episodic Memory Buffer에 추가
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

    # ─── 라우터: RL 루프 제어기 (조건부 엣지) ──────────────────────────────
    def _should_continue(self, state: AgentState) -> str:
        """
        [Reflexion §3 "Decision Making Loop" 조건부 분기 제어]

        보상 신호와 반복 횟수를 기반으로 다음 노드를 결정합니다:
        - "end"   : PASS(보상 획득) 또는 MAX_RETRIES 초과 → END로 이동
        - "retry" : FAIL(페널티) + 재시도 여유 있음 → generate 노드로 루프백

        [4_rlaif_agent.py 베이스라인의 should_continue() 함수를 메서드로 이전]

        Returns:
            "end" 또는 "retry" (add_conditional_edges 키와 일치)
        """
        if state["pass_fail"] == "PASS":
            return "end"

        if state["retry_count"] >= CONFIG.rl.MAX_RETRIES:
            print(
                f"[WARN] Router: Maximum retry limit ({CONFIG.rl.MAX_RETRIES}) reached. "
                "Terminating with best available response."
            )
            return "end"

        # FAIL + 재시도 여유 있음 → RL Loop 발동!
        time.sleep(CONFIG.rl.API_SLEEP_SEC)  # Google API Rate Limit 방지
        return "retry"

    # ─── 그래프 조립 (Builder Pattern) ──────────────────────────────────────
    def build(self):
        """
        LangGraph CompiledGraph를 조립하고 반환합니다.

        [Reflexion §3 "Decision Making Loop" 그래프 구조]
        START
          └→ retrieve (환경 관찰)
               └→ generate (Actor 행동)
                    └→ evaluate (Reward Model 평가)
                         ├─ PASS  → END
                         ├─ FAIL + retry available → generate (revision loop)
                         └─ FAIL + MAX_RETRIES → END

        Returns:
            LangGraph CompiledGraph — app.invoke(initial_state) 사용 가능
        """
        workflow = StateGraph(AgentState)  # type: ignore[type-var]

        # 노드 등록
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate", self._evaluate_node)

        # 순방향 엣지 (Directed Acyclic 구간)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")

        # 조건부 엣지 — 여기서 RL Loop(Cyclic Graph)가 완성됩니다!
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"end": END, "retry": "generate"}
        )

        return workflow.compile()
