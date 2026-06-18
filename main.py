"""
main.py — CLI entry point for single-query execution.

Runs the RoboGuard RLAIF agent against a single question and prints
a structured summary of the response and pipeline metrics.

Usage:
  python main.py
  python main.py --question "What is the maximum payload of the UR10e?"
  python main.py -q "How do I clear a Protective Stop?"
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
    """Build the graph, run the query, and print results."""
    parser = argparse.ArgumentParser(
        description="RoboGuard v2 — UR10e Query Agent with Revision Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=DEFAULT_QUESTION,
        help="Natural-language question about the UR10e manual."
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
        "pass_fail": "",
        "source_pages": [],
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
