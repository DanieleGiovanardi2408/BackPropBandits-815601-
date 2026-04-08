"""Entry point per lanciare il DataAgent in standalone."""
from multiagent_pipeline.agents.data_agent import data_agent_node
from multiagent_pipeline.state import AgentState

if __name__ == "__main__":
    state: AgentState = {
        "perimeter"    : {"anno": 2024},
        "df_raw"       : None,
        "data_meta"    : None,
        "df_features"  : None,
        "feature_meta" : None,
        "df_baseline"  : None,
        "baseline_meta": None,
        "df_anomalies" : None,
        "anomaly_meta" : None,
        "report"       : None,
        "report_path"  : None,
    }
    out = data_agent_node(state)

    print("\n=== RISULTATO DataAgent ===")
    print("data_meta:", out["data_meta"])
    if out["df_raw"] is not None:
        print("df_raw shape:", out["df_raw"].shape)
