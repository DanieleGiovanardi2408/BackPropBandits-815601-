"""Smoke test per catena completa fino a ReportAgent.

Uso:
  PYTHONPATH=. python3 multiagent_pipeline/tests/smoke_report_agent.py
"""
from __future__ import annotations

from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.agents.feature_agent import run_feature_agent
from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
from multiagent_pipeline.agents.outlier_agent import run_outlier_agent
from multiagent_pipeline.agents.report_agent import run_report_agent


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
    state = run_report_agent(state)
    return state


def main() -> None:
    perimeters = [
        {"anno": 2024},
        {"anno": 2024, "paese_partenza": "Algeria"},
        {"anno": 2024, "aeroporto_partenza": "ZZZ"},
    ]
    print("=== Smoke ReportAgent ===")
    for i, p in enumerate(perimeters, 1):
        s = run_chain(p)
        report = s.get("report") or {}
        if report.get("error"):
            print(f"[{i}] {p} -> ERROR: {report['error']}")
        else:
            print(
                f"[{i}] {p} -> OK "
                f"path={s.get('report_path')}, "
                f"n_rotte={report.get('n_rotte_analizzate')}, "
                f"alta={report.get('distribution', {}).get('alta')}"
            )


if __name__ == "__main__":
    main()

