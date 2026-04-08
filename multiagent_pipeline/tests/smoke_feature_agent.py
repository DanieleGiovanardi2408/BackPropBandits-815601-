"""Smoke test rapido per FeatureAgent su perimetri multipli.

Esecuzione:
  PYTHONPATH=. python3 multiagent_pipeline/tests/smoke_feature_agent.py
"""

from __future__ import annotations

from multiagent_pipeline.agents.feature_agent import run_feature_agent


def main() -> None:
    perimeters = [
        {"anno": 2024},
        {"anno": 2024, "paese_partenza": "Algeria"},
        {"anno": 2024, "paese_partenza": "algeria"},  # case-insensitive
        {"anno": 2024, "aeroporto_partenza": "ZZZ"},  # no match atteso
    ]

    print("=== FeatureAgent smoke test ===")
    for i, perimeter in enumerate(perimeters, 1):
        out = run_feature_agent({"perimeter": perimeter})
        meta = out.get("feature_meta") or {}

        if meta.get("error"):
            print(f"[{i}] perimeter={perimeter} -> ERROR: {meta['error']}")
            continue

        shape = out["df_features"].shape if out.get("df_features") is not None else None
        print(
            f"[{i}] perimeter={perimeter} -> OK "
            f"shape={shape}, n_numeric={len(meta.get('feature_cols', []))}"
        )


if __name__ == "__main__":
    main()

