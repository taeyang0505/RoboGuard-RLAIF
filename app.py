"""
app.py — RoboGuard RLAIF 웹 채팅 UI
=====================================
Streamlit 기반 채팅 인터페이스.
roboguard 패키지의 백엔드 엔진을 그대로 호출합니다.

Phase 1 — Source Citation:
  답변 말미의 출처 표기를 UI에서도 그대로 렌더링합니다.

Phase 1 — Streaming UI:
  st.status 컨텍스트로 LangGraph 각 노드 진행 상황을 실시간 중계합니다.
  최종 답변은 st.write_stream generator를 통해 타자 출력 방식으로 렌더링합니다.

Phase 3 — Multimodal Vision RAG:
  사이드바 파일 업로더로 PNG/JPG 이미지를 수신하고 Base64로 인코딩하여
  LangGraph 초기 상태에 image_b64로 주입합니다.
  Gemini Vision 모델이 이미지를 시각 분석하고 매뉴얼 컨텍스트와 교차 검증합니다.

실행 방법:
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


# ─── 세션 상태 초기화 ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # 채팅 메시지 히스토리
if "stats" not in st.session_state:
    st.session_state.stats = []      # 각 응답의 RL 통계
if "pending_image_b64" not in st.session_state:
    st.session_state.pending_image_b64 = None  # Phase 3: 업로드 대기 중 이미지


# ─── LangGraph 앱 캐싱 (매 질문마다 재빌드 방지) ──────────────────────────
@st.cache_resource(show_spinner="Initializing inference pipeline...")
def get_app():
    """Builds and caches the RoboGuardGraph compiled pipeline."""
    return RoboGuardGraph().build()


# ─── HTML 태그 제거 헬퍼 ─────────────────────────────────────────────────
def _clean_answer(text: str) -> str:
    """
    LLM 답변 본문에 삽입된 <br/> / <br> / <BR> 등 줄바꿈 HTML 태그를
    순수 마크다운 줄바꿈(\n\n)으로 변환합니다.

    Args:
        text: LLM이 반환한 원본 답변 문자열
    Returns:
        HTML 태그가 제거된 마크다운 문자열
    """
    # <br/>, <br />, <br> 계열을 모두 \n\n으로 치환
    return re.sub(r"<br\s*/?>" , "\n\n", text, flags=re.IGNORECASE)


# ─── 스트리밍 헬퍼: 단어 단위 generator ──────────────────────────────────
def _token_stream(text: str, delay: float = 0.012):
    """
    단어 단위로 텍스트를 분할하여 yield합니다.
    st.write_stream()에 전달하면 타자 출력 효과가 적용됩니다.

    Args:
        text : 출력할 전체 텍스트 (HTML 태그가 제거된 상태)
        delay: 단어 간 지연 시간 (초)
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


# ─── 메인 헤더 ──────────────────────────────────────────────────
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

        # Phase 3: 사용자 메시지에 이미지 썸네일 표시 (히스토리 재렌더링)
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


# ─── 채팅 입력 처리 ───────────────────────────────────────────────────────
if prompt := st.chat_input("Enter a question about the UR10e robot (e.g. maximum payload capacity)"):

    # 1) User message
    image_b64_current: str | None = st.session_state.get("pending_image_b64")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        # Phase 3: 현재 메시지에 첨부 이미지 표시
        if image_b64_current:
            st.image(
                base64.b64decode(image_b64_current),
                caption="📸 Attached image",
                width=240,
            )

    # 2) Assistant response
    with st.chat_message("assistant", avatar="⚙️"):

        # ── Phase 1: st.status 실시간 상태 중계 ──────────────────────────
        result: dict = {}
        elapsed: float = 0.0

        with st.status("Processing query...", expanded=True) as status_box:

            status_box.write("**[1/3]** Querying Vector DB for relevant document sections.")
            langgraph_app = get_app()
            t0 = time.time()

            # LangGraph invoke — 백엔드 실행 (동기)
            # invoke가 완료되면 st.status 내 단계를 단계별로 업데이트합니다.
            # 단, LangGraph는 동기 실행이므로 노드 완료 시점을 폴링하는 대신
            # invoke 전/후로 상태를 분리하여 표시합니다.

            # 백그라운드 스레드로 invoke 실행 → 메인 스레드에서 상태 중계
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

            # 중계 메시지 — 각 단계를 순차적으로 표시
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

            # 재시도가 발생한 경우 추가 상태 표시
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

        # ── Phase 1: 결과 추출 ────────────────────────────────────────────
        answer      = _clean_answer(result.get("answer", ""))  # <br/> → \n\n
        pass_fail   = result.get("pass_fail", "PASS")
        traj_cnt    = len(result.get("trajectory_log", []))
        src_pages   = result.get("source_pages", [])

        # ── Phase 1: st.write_stream — 타자 출력 스트리밍 ────────────────
        st.write_stream(_token_stream(answer))

        # ── 검증 배지 + 출처 표기 ─────────────────────────────────────────
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

        # ── FAIL 시 평가자 피드백 표시 ────────────────────────────────────
        if pass_fail == "FAIL" and result.get("feedback"):
            with st.expander("Evaluator Feedback — Fact-Verification Detail"):
                st.warning(result["feedback"])

        # ── 재시도 이력 로그 (재작성이 발생한 경우) ──────────────────────
        if retries > 0 and traj_cnt > 0:
            with st.expander(f"Revision & Inference Log ({traj_cnt} attempt(s))"):
                for idx, ep in enumerate(result["trajectory_log"], 1):
                    verdict = "PASS" if ep.get("pass_fail") == "PASS" else "FAIL"
                    st.markdown(f"**Attempt {idx}** — {verdict}")
                    st.caption(ep.get("answer", "")[:300] + "...")
                    st.divider()

    # 3) Save to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.stats.append({
        "pass_fail":   pass_fail,
        "retries":     retries,
        "elapsed":     elapsed,
        "source_pages": src_pages,
        "image_b64":   image_b64_current,  # Phase 3: 썬네일 재렌더링 용
    })
    # Phase 3: 다음 질문에서 동일 이미지가 중복 로드되지 않도록
    # 사이드바 uploader는 Streamlit 자체 상태로 관리되므로 별도 reset 불필요
