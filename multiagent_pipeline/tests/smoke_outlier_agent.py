"""Smoke test per catena completa fino a OutlierAgent.

Uso:
  PYTHONPATH=. python3 multiagent_pipeline/tests/smoke_outlier_agent.py
"""
from __future__ import annotations

from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.agents.feature_agent import run_feature_agent
from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
from multiagent_pipeline.agents.outlier_agent import run_outlier_agent


def run_chain(perimeter: dict) -> dict:
    state = {
        "perimeter": perimeter,
        "df_raw": None,
        "df_allarmi": None,
        "df_viaggiatori": None,
        "data_meta": None,
        "df_features": None,
        "feature_meta": None,
        "df_baseline": None,
        "baseline_meta": None,
        "df_anomalies": None,
        "anomaly_meta": None,
        "report": None,
        "report_path": None,
    }
    state = data_agent_node(state)
    state = run_feature_agent(state)
    state = run_baseline_agent(state)
    state = run_outlier_agent(state)
    return state


def main() -> None:
    perimeters = [
        {"anno": 2024},
        {"anno": 2024, "paese_partenza": "Algeria"},
        {"anno": 2024, "aeroporto_partenza": "ZZZ"},
    ]
    print("=== Smoke OutlierAgent ===")
    for i, p in enumerate(perimeters, 1):
        s = run_chain(p)
        a_meta = s.get("anomaly_meta") or {}
        if a_meta.get("error"):
            print(f"[{i}] {p} -> ERROR: {a_meta['error']}")
        else:
            print(
                f"[{i}] {p} -> OK "
                f"shape={s['df_anomalies'].shape}, "
                f"ALTA={a_meta['n_alta']} MEDIA={a_meta['n_media']} NORMALE={a_meta['n_normale']}"
            )


if __name__ == "__main__":
    main()

