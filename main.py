"""
main.py — 단일 질의응답 실행 진입점
=====================================
RoboGuard RLAIF 에이전트를 단일 질문으로 실행합니다.

CLI 사용 예시:
  python main.py
  python main.py --question "UR10e의 최대 적재량은?"
  python main.py -q "Protective Stop 해제 절차를 알려주세요."

[4_rlaif_agent.py 베이스라인의 __main__ 블록을 CLI 모듈로 격상]
"""
import argparse
import time
from dotenv import load_dotenv

from roboguard.graph_builder import RoboGuardGraph

load_dotenv()

# Default test query
DEFAULT_QUESTION = (
    "로봇 팔을 수심 10m 물속에 푹 담가서 수중 작업을 해도 되나요? "
    "IP 방수 등급 규정을 근거로 설명해주세요."
)


def main() -> None:
    """단일 질의응답 실행 메인 함수."""
    parser = argparse.ArgumentParser(
        description="RoboGuard v2 — UR10e Query Agent with Revision Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=DEFAULT_QUESTION,
        help="로봇 매뉴얼 관련 질문 (기본값: 수중 작업 IP 등급 질문)"
    )
    args = parser.parse_args()

    print("-" * 75)
    print("[INFO] RoboGuard Technical Support System — UR10e Query Interface")
    print("       Inference pipeline: InstructGPT RM + Reflexion + Self-RAG")
    print("-" * 75)
    print(f"\n[INFO] Input query: {args.question}")

    # Graph build and execution
    app = RoboGuardGraph().build()

    start_t = time.time()
    result = app.invoke({
        "question": args.question,
        "retry_count": 0,
        "trajectory_log": [],
        "context": "",
        "answer": "",
        "feedback": "",
        "pass_fail": ""
    })
    elapsed = time.time() - start_t

    print("\n" + "-" * 75)
    print("[INFO] Final Response")
    print(result["answer"])
    print("\n[INFO] Execution Summary")
    print(f"  Elapsed time     : {elapsed:.2f}s")
    print(f"  Revision cycles  : {result['retry_count'] - 1}")
    print(f"  Verification     : {result['pass_fail']}")
    if result.get("trajectory_log"):
        print(f"  Episode records  : {len(result['trajectory_log'])}")
    print("-" * 75 + "\n")


if __name__ == "__main__":
    main()
