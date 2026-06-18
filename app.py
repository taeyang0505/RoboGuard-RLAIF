"""
app.py — RoboGuard Streamlit chat UI.

A Streamlit-based chat interface that calls the roboguard backend directly.

Features:
  Source citation  : Reference page numbers are surfaced inline in each response.
  Streaming UI     : st.status relays per-node pipeline progress in real time;
                     the final answer is rendered word-by-word via st.write_stream.
  Vision RAG       : A sidebar file uploader accepts PNG/JPG images, encodes them
                     in Base64, and injects them into the LangGraph initial state so
                     Gemini Vision can cross-reference visual observations against
                     the manual context.

Usage:
  ./.venv/bin/streamlit run app.py
"""
import re
import base64
import time
import threading
import queue
import streamlit as st
from dotenv import load_dotenv

from roboguard.graph_builder import RoboGuardGraph

load_dotenv()

# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoboGuard — UR10e Technical Support",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* page background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8f0;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
}

/* chat input */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #fff !important;
    border-radius: 12px !important;
}

/* user message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 4px 8px;
    margin: 6px 0;
}

/* assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 16px;
    padding: 4px 8px;
    margin: 6px 0;
}

/* metric card */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    backdrop-filter: blur(8px);
}

/* PASS badge */
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

/* FAIL badge */
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

/* RL loop counter badge */
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

/* sidebar paper card */
.paper-card {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.82rem;
    line-height: 1.5;
}

/* hero header title */
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


# Sidebar ────────────────
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

    # ── Phase 3: Vision RAG ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📸 Vision Input (Optional)")
    uploaded_file = st.file_uploader(
        "Upload an image for visual analysis",
        type=["png", "jpg", "jpeg"],
        help="Attach a photo of the robot error screen or component for AI-powered visual diagnosis.",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Attached image", width="stretch")
        st.session_state.pending_image_b64 = base64.b64encode(
            uploaded_file.read()
        ).decode("utf-8")
    else:
        st.session_state.pending_image_b64 = None

    st.caption("RoboGuard v3.0 — Vision RAG · Source Citation · Streaming UI")


# Session state initialisation ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []           # chat history
if "stats" not in st.session_state:
    st.session_state.stats = []              # per-response RL metrics
if "pending_image_b64" not in st.session_state:
    st.session_state.pending_image_b64 = None  # queued Vision RAG image


# Cache the compiled graph so it isn't rebuilt on every Streamlit rerun.────
@st.cache_resource(show_spinner="Initializing inference pipeline...")
def get_app():
    """Builds and caches the RoboGuardGraph compiled pipeline."""
    return RoboGuardGraph().build()


# HTML tag cleanup helper────────
def _clean_answer(text: str) -> str:
    """Replace <br> variants in LLM output with Markdown double newlines.

    Args:
        text: Raw answer string returned by the LLM.
    Returns:
        Answer string with HTML line-break tags replaced by \n\n.
    """
    return re.sub(r"<br\s*/?>" , "\n\n", text, flags=re.IGNORECASE)


# Word-by-word streaming generator────
def _token_stream(text: str, delay: float = 0.012):
    """Yield the text word-by-word with a short delay for a typewriter effect.

    Args:
        text : Full cleaned response string.
        delay: Seconds between words.
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)

# Main header ───────────
st.markdown('<div class="hero-title">⚙️ RoboGuard — UR10e Technical Support</div>', unsafe_allow_html=True)
st.markdown(
    "Document-grounded QA system for UR10e robot operations. "
    "All responses are cross-validated against the source manual via a fact-verification pipeline."
)
st.divider()

# Render conversation history ──────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚙️"):
        st.markdown(msg["content"])

        # Show attached image thumbnail when replaying history
        if msg["role"] == "user":
            msg_idx = i // 2
            if msg_idx < len(st.session_state.stats):
                stat_img = st.session_state.stats[msg_idx].get("image_b64")
                if stat_img:
                    st.image(
                        base64.b64decode(stat_img),
                        caption="📸 Attached image",
                        width=240,
                    )

        # Show verification badge on assistant messages only
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.stats):
            stat = st.session_state.stats[i // 2]
            pass_fail   = stat.get("pass_fail", "PASS")
            retries     = stat.get("retries", 0)
            elapsed     = stat.get("elapsed", 0.0)
            src_pages   = stat.get("source_pages", [])

            badge_class = "badge-pass" if pass_fail == "PASS" else "badge-fail"
            badge_label = "Verified — PASS" if pass_fail == "PASS" else "Unverified — FAIL"

            pages_str = (
                f"  ·  Ref. pp. {', '.join(str(p) for p in src_pages)}"
                if src_pages else ""
            )
            st.markdown(
                f'<span class="{badge_class}">{badge_label}</span>'
                f'<span class="rl-badge">Revision cycles: {retries}</span>'
                f'<span class="rl-badge">{elapsed:.1f}s{pages_str}</span>',
                unsafe_allow_html=True
            )


# Chat input handler ────────────
if prompt := st.chat_input("Enter a question about the UR10e robot (e.g. maximum payload capacity)"):

    # 1) User message
    image_b64_current: str | None = st.session_state.get("pending_image_b64")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        # Show attached image below the user's message
        if image_b64_current:
            st.image(
                base64.b64decode(image_b64_current),
                caption="📸 Attached image",
                width=240,
            )

    # 2) Assistant response
    with st.chat_message("assistant", avatar="⚙️"):

        # st.status: relay per-node pipeline progress in real time
        result: dict = {}
        elapsed: float = 0.0

        with st.status("Processing query...", expanded=True) as status_box:

            status_box.write("**[1/3]** Querying Vector DB for relevant document sections.")
            langgraph_app = get_app()
            t0 = time.time()

            # Run LangGraph in a background thread so Streamlit can update
            # the st.status panel while inference is in progress.
            result_queue: queue.Queue = queue.Queue()

            def _run_graph() -> None:
                r = langgraph_app.invoke({
                    "question": prompt,
                    "retry_count": 0,
                    "trajectory_log": [],
                    "context": "",
                    "answer": "",
                    "feedback": "",
                    "pass_fail": "",
                    "source_pages": [],
                    "image_b64": image_b64_current,  # Phase 3: Vision RAG
                })
                result_queue.put(r)

            thread = threading.Thread(target=_run_graph, daemon=True)
            thread.start()

            # Progress relay: display stage messages as wall-clock time elapses
            stage_delays = [
                (3.0,  "**[2/3]** Generating initial response from retrieved context (base policy)."),
                (6.0,  "**[3/3]** Running fact-verification via LLM-as-a-judge."),
            ]
            stage_idx = 0
            while thread.is_alive():
                if stage_idx < len(stage_delays):
                    delay, msg_text = stage_delays[stage_idx]
                    elapsed_so_far = time.time() - t0
                    if elapsed_so_far >= delay:
                        status_box.write(msg_text)
                        stage_idx += 1
                time.sleep(0.2)

            thread.join()
            result = result_queue.get()
            elapsed = time.time() - t0

            # If the pipeline ran revision cycles, surface that in the status panel
            retries = max(0, result.get("retry_count", 1) - 1)
            if retries > 0:
                status_box.write(
                    f"**[Revision]** Unverified content detected. "
                    f"Revision loop executed {retries} time(s). Re-verifying..."
                )

            status_box.update(
                label=f"Pipeline complete — {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )

        # Extract result fields and render streaming answer
        answer      = _clean_answer(result.get("answer", ""))
        pass_fail   = result.get("pass_fail", "PASS")
        traj_cnt    = len(result.get("trajectory_log", []))
        src_pages   = result.get("source_pages", [])

        st.write_stream(_token_stream(answer))

        # Verification badge and source page reference────
        badge_class = "badge-pass" if pass_fail == "PASS" else "badge-fail"
        badge_label = "Verified — PASS" if pass_fail == "PASS" else "Unverified — FAIL"
        pages_str = (
            f"  ·  Ref. pp. {', '.join(str(p) for p in src_pages)}"
            if src_pages else ""
        )
        st.markdown(
            f'<span class="{badge_class}">{badge_label}</span>'
            f'<span class="rl-badge">Revision cycles: {retries}</span>'
            f'<span class="rl-badge">{elapsed:.1f}s{pages_str}</span>',
            unsafe_allow_html=True
        )

        # Show evaluator rationale on FAIL
        if pass_fail == "FAIL" and result.get("feedback"):
            with st.expander("Evaluator Feedback — Fact-Verification Detail"):
                st.warning(result["feedback"])

        # Show revision log when at least one retry occurred
        if retries > 0 and traj_cnt > 0:
            with st.expander(f"Revision & Inference Log ({traj_cnt} attempt(s))"):
                for idx, ep in enumerate(result["trajectory_log"], 1):
                    verdict = "PASS" if ep.get("pass_fail") == "PASS" else "FAIL"
                    st.markdown(f"**Attempt {idx}** — {verdict}")
                    st.caption(ep.get("answer", "")[:300] + "...")
                    st.divider()

    # 3) Persist to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.stats.append({
        "pass_fail":   pass_fail,
        "retries":     retries,
        "elapsed":     elapsed,
        "source_pages": src_pages,
        "image_b64":   image_b64_current,  # stored for thumbnail re-rendering in history
    })
    # The sidebar uploader is managed by Streamlit's own state, so no manual
    # reset is needed to prevent the same image from attaching to the next query.
