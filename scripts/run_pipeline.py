"""
Algions AI Labs - E-Commerce Recommendation Engine
Master Orchestration Script: Executes complete end-to-end pipeline:
Data Generation -> Ingestion & Split -> Training -> Evaluation -> Test Suite.
"""

import sys
import os
import subprocess
import time

def run_step(step_name: str, command: list):
    print("\n" + "=" * 70)
    print(f"PIPELINE STEP: {step_name}")
    print("=" * 70)
    start = time.time()
    result = subprocess.run([sys.executable] + command, check=True)
    duration = time.time() - start
    print(f"[{step_name} COMPLETED in {duration:.2f}s]")


def main():
    print("=" * 70)
    print("ALGIONS AI LABS - E-COMMERCE RECOMMENDATION ENGINE PIPELINE")
    print("=" * 70)
    pipeline_start = time.time()

    # Step 1: Data Generation
    run_step("1. Data Generation", ["scripts/generate_data.py"])

    # Step 2: Training Pipeline
    run_step("2. Training Pipeline", ["scripts/train.py"])

    # Step 3: Offline Model Evaluation
    run_step("3. Model Evaluation", ["scripts/evaluate.py"])

    # Step 4: Automated Test Suite Execution
    print("\n" + "=" * 70)
    print("PIPELINE STEP: 4. Automated Acceptance & Unit Test Suite")
    print("=" * 70)
    test_start = time.time()
    test_result = subprocess.run([sys.executable, "-m", "unittest", "discover", "tests", "-v"])
    if test_result.returncode != 0:
        print("[WARNING] Some tests failed. Check logs above.")
        sys.exit(test_result.returncode)
    else:
        print(f"[4. Automated Test Suite COMPLETED in {time.time() - test_start:.2f}s]")

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("Saved Artifacts:")
    print("  - artifacts/popularity_model.joblib")
    print("  - artifacts/content_model.joblib")
    print("  - artifacts/collaborative_model.joblib")
    print("  - artifacts/hybrid_model.joblib")
    print("  - artifacts/evaluation_results.json")
    print("  - artifacts/model_metadata.json")
    print("  - artifacts/figures/interaction_distribution.png")
    print("  - artifacts/figures/funnel_distribution.png")
    print("  - artifacts/figures/model_comparison.png")
    print("\nTo start the real-time inference server and interactive dashboard:")
    print("  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload")
    print("=" * 70)


if __name__ == "__main__":
    main()

