"""
batch_eval.py — 다중 질문 스트레스 테스트 및 CSV 리포트 추출
=============================================================
Golden Dataset에 정의된 5개의 고난도 실무 질문으로
RLAIF 에이전트를 배치 평가하고 결과를 CSV로 저장합니다.

[InstructGPT §3 "Experiments" 평가 프레임워크 참조]
논문이 다양한 프롬프트 카테고리로 모델을 평가했듯이,
이 스크립트는 다양한 난이도와 유형의 질문으로 에이전트의
#   - Out-of-scope boundary testing

사용법:
  python batch_eval.py
"""
import csv
import time
from dotenv import load_dotenv

from roboguard.config import CONFIG
from roboguard.graph_builder import RoboGuardGraph

load_dotenv()


# ─── Golden Dataset: 5개 고난도 실무 질문 ─────────────────────────────────
# [InstructGPT §3.4 "Prompt Dataset 구성 원칙" 적용]
# 다양한 실패 모드를 유발하는 질문을 골고루 배치합니다:
#   - 장애 복구 절차 (다단계 순서 추론)
#   - 정밀 수치 인용 (수치 강박 테스트)
#   - Out-of-scope boundary test (external knowledge detection)
#   - 안전 규정 (매뉴얼 방어 테스트)
#   - 복합 수치 (페이로드 + 도달 범위)
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
    """
    Runs the full golden dataset sequentially and saves results to a CSV report.

    [InstructGPT §3 "Experiments"]
    Records the following metrics per query:
    - Final response text
    - Fact-verification result (PASS / FAIL)
    - Number of revision cycles
    - Elapsed time (seconds)
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
                "pass_fail": ""
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
