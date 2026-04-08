"""Smoke test rapido per BaselineAgent.

Esecuzione:
  PYTHONPATH=. python3 multiagent_pipeline/tests/smoke_baseline_agent.py
"""
from __future__ import annotations

from multiagent_pipeline.agents.feature_agent import run_feature_agent
from multiagent_pipeline.agents.baseline_agent import run_baseline_agent


def main() -> None:
    perimeters = [
        {"anno": 2024},
        {"anno": 2024, "paese_partenza": "Algeria"},
        {"anno": 2024, "aeroporto_partenza": "ZZZ"},  # no data atteso
    ]

    print("=== BaselineAgent smoke test ===")
    for i, p in enumerate(perimeters, 1):
        s = {"perimeter": p}
        s = run_feature_agent(s)
        s = run_baseline_agent(s)

        meta = s.get("baseline_meta") or {}
        if meta.get("error"):
            print(f"[{i}] perimeter={p} -> ERROR: {meta['error']}")
            continue

        print(
            f"[{i}] perimeter={p} -> OK "
            f"shape={s['df_baseline'].shape}, "
            f"soglia_media={meta['soglia_media']}, "
            f"soglia_alta={meta['soglia_alta']}"
        )


if __name__ == "__main__":
    main()

