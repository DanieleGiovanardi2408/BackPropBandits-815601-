"""Runner standalone per DataAgent."""
from __future__ import annotations

from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.state import AgentState


def main() -> None:
    state: AgentState = {
        "perimeter": {"anno": 2024},
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
    out = data_agent_node(state, save_artifacts=False)
    print("\n=== RISULTATO DataAgent ===")
    print("data_meta:", out["data_meta"])
    if out["df_raw"] is not None:
        print("df_raw shape:", out["df_raw"].shape)


if __name__ == "__main__":
    main()

