"""Smoke test orchestrator (step 1)."""
from __future__ import annotations

from multiagent_pipeline.main import run_pipeline


def main() -> None:
    perimeters = [
        {"anno": 2024},
        {"anno": 2024, "paese_partenza": "Algeria"},
        {"anno": 2024, "aeroporto_partenza": "ZZZ"},
    ]
    print("=== Smoke Orchestrator ===")
    for i, p in enumerate(perimeters, 1):
        _, summary = run_pipeline(p, run_report=False, save_outputs=False)
        print(f"[{i}] {p} -> completed={summary['completed_stages']} failed={summary['failed_stages']}")


if __name__ == "__main__":
    main()

