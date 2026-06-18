"""
app.py — RoboGuard RLAIF 웹 채팅 UI
=====================================
Streamlit 기반 챗봇 인터페이스.
roboguard 패키지의 백엔드 엔진을 그대로 호출합니다.

실행 방법:
  ./.venv/bin/streamlit run app.py
"""
import time
import streamlit as st
from dotenv import load_dotenv

from roboguard.graph_builder import RoboGuardGraph

load_dotenv()

# ─── 페이지 기본 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoboGuard — UR10e Technical Support",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 커스텀 CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8f0;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
}

/* 채팅 입력창 */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #fff !important;
    border-radius: 12px !important;
}

/* 사용자 말풍선 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 4px 8px;
    margin: 6px 0;
}

/* AI 말풍선 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 16px;
    padding: 4px 8px;
    margin: 6px 0;
}

/* 메트릭 박스 */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    backdrop-filter: blur(8px);
}

/* PASS 배지 */
.badge-pass {
    display: inline-block;
    background: linear-gradient(90deg, #059669, #10b981);
    color: #fff;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 999px;
    margin-top: 8px;
    letter-spacing: 0.04em;
}

/* FAIL 배지 */
.badge-fail {
    display: inline-block;
    background: linear-gradient(90deg, #dc2626, #ef4444);
    color: #fff;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 999px;
    margin-top: 8px;
    letter-spacing: 0.04em;
}

/* RL 루프 카운터 */
.rl-badge {
    display: inline-block;
    background: rgba(139,92,246,0.25);
    border: 1px solid rgba(139,92,246,0.5);
    color: #c4b5fd;
    font-size: 0.74rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin-left: 8px;
}

/* 사이드바 논문 카드 */
.paper-card {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.82rem;
    line-height: 1.5;
}

/* 헤더 타이틀 */
.hero-title {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #34d399, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─── 사이드바 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">⚙️ RoboGuard</div>', unsafe_allow_html=True)
    st.caption("UR10e Technical Support System — Document-Grounded QA")
    st.divider()

    st.markdown("#### ▪ Inference Architecture")

    # InstructGPT
    st.markdown("""
    <div class="paper-card" style="border-color:#818cf8;">
    <b>InstructGPT (OpenAI, 2022)</b><br>
    LLM-as-a-judge deployed as a <b>Reward Model</b>.<br>
    Assigns scalar reward (+1 / -1) based on document grounding.
    </div>
    """, unsafe_allow_html=True)

    # Reflexion
    st.markdown("""
    <div class="paper-card" style="border-color:#34d399;">
    <b>Reflexion (Shinn et al., 2023)</b><br>
    Prior failed attempts stored in an <b>episodic memory buffer</b>.<br>
    Policy updated via verbal feedback injection — no gradient required.
    </div>
    """, unsafe_allow_html=True)

    # Self-RAG
    st.markdown("""
    <div class="paper-card" style="border-color:#60a5fa;">
    <b>Self-RAG (Asai et al., 2023)</b><br>
    <b>[PASS]/[FAIL] critique tokens</b> used to assess factual support.<br>
    Retrieval modeled as an active environment component.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### ⚙️ System Configuration")
    st.markdown("""
    <div class="metric-card">
    <b>LLM</b>: gemini-2.5-flash<br>
    <b>Vector DB</b>: Chroma (local)<br>
    <b>Temperature</b>: 0 (deterministic)<br>
    <b>Max Revision Cycles</b>: 3<br>
    <b>Retrieved Documents (k)</b>: 5
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = []
        st.rerun()

    st.caption("RoboGuard v2.0 — Internal Use Only")


# ─── 세션 상태 초기화 ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # 채팅 메시지 히스토리
if "stats" not in st.session_state:
    st.session_state.stats = []      # 각 응답의 RL 통계

# ─── LangGraph 앱 캐싱 (매 질문마다 재빌드 방지) ──────────────────────────
@st.cache_resource(show_spinner="Initializing inference pipeline...")
def get_app():
    """Builds and caches the RoboGuardGraph compiled pipeline."""
    return RoboGuardGraph().build()


# ─── 메인 헤더 ────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⚙️ RoboGuard — UR10e Technical Support</div>', unsafe_allow_html=True)
st.markdown(
    "Document-grounded QA system for UR10e robot operations. "
    "All responses are cross-validated against the source manual via a fact-verification pipeline."
)
st.divider()

# ─── 이전 대화 기록 렌더링 ────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚙️"):
        st.markdown(msg["content"])

        # Show verification badge on assistant messages only
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.stats):
            stat = st.session_state.stats[i // 2]
            pass_fail = stat.get("pass_fail", "PASS")
            retries   = stat.get("retries", 0)
            elapsed   = stat.get("elapsed", 0.0)

            badge_class = "badge-pass" if pass_fail == "PASS" else "badge-fail"
            badge_label = "Verified — PASS" if pass_fail == "PASS" else "Unverified — FAIL"

            st.markdown(
                f'<span class="{badge_class}">{badge_label}</span>'
                f'<span class="rl-badge">Revision cycles: {retries}</span>'
                f'<span class="rl-badge">{elapsed:.1f}s</span>',
                unsafe_allow_html=True
            )


# ─── 채팅 입력 처리 ───────────────────────────────────────────────────────
if prompt := st.chat_input("Enter a question about the UR10e robot (e.g. maximum payload capacity)"):

    # 1) User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2) Assistant response
    with st.chat_message("assistant", avatar="⚙️"):
        status_area = st.empty()

        with st.spinner(""):
            step_msgs = [
                "Querying Vector DB for relevant document sections...",
                "Generating response based on retrieved context...",
                "Running fact-verification (LLM-as-a-judge)...",
            ]
            for step in step_msgs:
                status_area.info(step)
                time.sleep(0.3)

            langgraph_app = get_app()
            t0 = time.time()
            result = langgraph_app.invoke({
                "question": prompt,
                "retry_count": 0,
                "trajectory_log": [],
                "context": "",
                "answer": "",
                "feedback": "",
                "pass_fail": ""
            })
            elapsed = time.time() - t0

        status_area.empty()

        answer    = result.get("answer", "")
        pass_fail = result.get("pass_fail", "PASS")
        retries   = max(0, result.get("retry_count", 1) - 1)
        traj_cnt  = len(result.get("trajectory_log", []))

        st.markdown(answer)

        # Verification badge
        badge_class = "badge-pass" if pass_fail == "PASS" else "badge-fail"
        badge_label = "Verified — PASS" if pass_fail == "PASS" else "Unverified — FAIL"
        st.markdown(
            f'<span class="{badge_class}">{badge_label}</span>'
            f'<span class="rl-badge">Revision cycles: {retries}</span>'
            f'<span class="rl-badge">{elapsed:.1f}s</span>',
            unsafe_allow_html=True
        )

        # Show evaluator feedback on FAIL
        if pass_fail == "FAIL" and result.get("feedback"):
            with st.expander("Evaluator Feedback — Fact-Verification Detail"):
                st.warning(result["feedback"])

        # Show revision log if retries occurred
        if retries > 0 and traj_cnt > 0:
            with st.expander(f"Revision & Inference Log ({traj_cnt} attempt(s))"):
                for i, ep in enumerate(result["trajectory_log"], 1):
                    verdict = "PASS" if ep.get("pass_fail") == "PASS" else "FAIL"
                    st.markdown(f"**Attempt {i}** — {verdict}")
                    st.caption(ep.get("answer", "")[:300] + "...")
                    st.divider()

    # 3) Save to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.stats.append({
        "pass_fail": pass_fail,
        "retries":   retries,
        "elapsed":   elapsed,
    })

