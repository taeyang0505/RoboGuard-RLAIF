"""
batch_eval.py — Golden dataset stress-test and CSV report generator.

Runs the RLAIF agent against a fixed set of five domain-specific questions
and saves per-query metrics (verdict, revision count, elapsed time) to a CSV.

Usage:
  python batch_eval.py
"""
import csv
import time
from dotenv import load_dotenv

from roboguard.config import CONFIG
from roboguard.graph_builder import RoboGuardGraph

load_dotenv()


# Golden dataset: five questions covering distinct failure modes.
# Designed to stress-test: fault recovery, numerical precision,
# out-of-scope boundary detection, safety specs, and core specifications.
GOLDEN_DATASET: list[dict] = [
    {
        "id": "Q1",
        "category": "Fault Recovery",
        "question": (
            "A Protective Stop has been triggered due to joint overload. "
            "Describe the step-by-step procedure to clear the condition and resume operation "
            "using the Teach Pendant."
        ),
    },
    {
        "id": "Q2",
        "category": "Numerical Specification",
        "question": (
            "What is the maximum allowable current output (in amperes) "
            "provided by the 24V power supply on the I/O terminal block "
            "inside the Control Box?"
        ),
    },
    {
        "id": "Q3",
        "category": "Out-of-Scope Boundary Test",
        "question": (
            "Is it permissible to fully submerge the robot arm to a depth of 10 meters "
            "for underwater operations? Provide justification based on the IP rating "
            "specifications in the documentation."
        ),
    },
    {
        "id": "Q4",
        "category": "Safety Specification",
        "question": (
            "During collaborative operation (Collaborative Mode) with the UR10e, "
            "what is the minimum required distance that operators must maintain from the robot arm?"
        ),
    },
    {
        "id": "Q5",
        "category": "Core Specifications",
        "question": (
            "What are the exact maximum payload capacity and maximum reach "
            "of the UR10e robot?"
        ),
    },
]


def run_batch_eval() -> None:
    """Run the full golden dataset sequentially and write results to a CSV report.

    Records per-query metrics: final response, verification verdict,
    revision cycle count, and elapsed time.
    """
    print("-" * 75)
    print("[INFO] RoboGuard — Batch Evaluation Pipeline")
    print(f"       {len(GOLDEN_DATASET)} queries | Output: {CONFIG.REPORT_PATH}")
    print("-" * 75)

    app = RoboGuardGraph().build()

    with open(CONFIG.REPORT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Query ID",
            "Category",
            "Question",
            "Final Response",
            "Verification Result",
            "Revision Cycles",
            "Elapsed Time (s)"
        ])

        pass_count = 0
        total_retries = 0

        for item in GOLDEN_DATASET:
            q_id = item["id"]
            category = item["category"]
            question = item["question"]

            print(f"\n[INFO] Processing {q_id} [{category}]")
            print(f"       {question[:90]}...")

            start_t = time.time()
            result = app.invoke({
                "question": question,
                "retry_count": 0,
                "trajectory_log": [],
                "context": "",
                "answer": "",
                "feedback": "",
                "pass_fail": "",
                "source_pages": [],
            })
            elapsed = time.time() - start_t

            retries = result["retry_count"] - 1
            pass_fail = result["pass_fail"]
            total_retries += retries
            if pass_fail == "PASS":
                pass_count += 1

            verdict_tag = "[PASS]" if pass_fail == "PASS" else "[FAIL]"
            print(f"       Response preview: {result['answer'][:120]}...")
            print(f"       {verdict_tag} | Revision cycles: {retries} | Elapsed: {elapsed:.2f}s")

            writer.writerow([
                q_id,
                category,
                question,
                result["answer"],
                pass_fail,
                retries,
                f"{elapsed:.2f}"
            ])

            # Throttle API calls to avoid rate-limit errors
            time.sleep(CONFIG.rl.API_SLEEP_SEC)

    print("\n" + "-" * 75)
    print("[INFO] Batch evaluation complete.")
    print(f"       PASS rate    : {pass_count}/{len(GOLDEN_DATASET)} "
          f"({pass_count/len(GOLDEN_DATASET)*100:.0f}%)")
    print(f"       Avg revisions: {total_retries/len(GOLDEN_DATASET):.1f} per query")
    print(f"       Report saved : '{CONFIG.REPORT_PATH}'")
    print("-" * 75)


if __name__ == "__main__":
    run_batch_eval()
