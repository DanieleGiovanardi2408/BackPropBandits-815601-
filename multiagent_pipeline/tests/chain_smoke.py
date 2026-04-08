"""Smoke test end-to-end della catena multi-agent.

Esegue:
    DataAgent -> FeatureAgent -> BaselineAgent
su piu perimetri e stampa un riepilogo sintetico.

Uso:
    PYTHONPATH=. python3 multiagent_pipeline/tests/chain_smoke.py
"""
from __future__ import annotations

from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.agents.feature_agent import run_feature_agent
from multiagent_pipeline.agents.baseline_agent import run_baseline_agent


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

    s1 = data_agent_node(state)
    s2 = run_feature_agent(s1)
    s3 = run_baseline_agent(s2)

    return {
        "perimeter": perimeter,
        "data_error": (s1.get("data_meta") or {}).get("error"),
        "feature_error": (s2.get("feature_meta") or {}).get("error"),
        "baseline_error": (s3.get("baseline_meta") or {}).get("error"),
        "df_raw_shape": None if s1.get("df_raw") is None else s1["df_raw"].shape,
        "df_features_shape": None if s2.get("df_features") is None else s2["df_features"].shape,
        "df_baseline_shape": None if s3.get("df_baseline") is None else s3["df_baseline"].shape,
    }


def main() -> None:
    perimeters = [
        {"anno": 2024},
        {"anno": 2024, "paese_partenza": "Algeria"},
        {"anno": 2024, "zona": 5},
        {"anno": 2024, "aeroporto_partenza": "ZZZ"},  # no data atteso
    ]

    print("=== Chain smoke: Data -> Feature -> Baseline ===")
    for idx, p in enumerate(perimeters, 1):
        res = run_chain(p)
        print(f"\n[{idx}] perimeter={res['perimeter']}")
        print(f"  data_agent: {'OK' if not res['data_error'] else 'ERROR'}")
        print(f"  feature_agent: {'OK' if not res['feature_error'] else 'ERROR'}")
        print(f"  baseline_agent: {'OK' if not res['baseline_error'] else 'ERROR'}")
        if res["data_error"]:
            print(f"  -> data_error: {res['data_error']}")
        if res["feature_error"]:
            print(f"  -> feature_error: {res['feature_error']}")
        if res["baseline_error"]:
            print(f"  -> baseline_error: {res['baseline_error']}")
        print(f"  shapes: raw={res['df_raw_shape']} features={res['df_features_shape']} baseline={res['df_baseline_shape']}")


if __name__ == "__main__":
    main()

