"""OutlierAgent — quarto nodo del grafo multi-agent.

Responsabilità:
    Combina più segnali anomaly-like in uno score ensemble e assegna
    una risk label (NORMALE/MEDIA/ALTA) per ogni rotta.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "multiagent_pipeline.agents"

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multiagent_pipeline.state import (
    AgentState,
    ENSEMBLE_WEIGHTS,
    THRESHOLD_ALTA,
    THRESHOLD_MEDIA,
)

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    s_min = float(s.min())
    s_max = float(s.max())
    if s_max <= s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)


def run_outlier_agent(
    state: AgentState,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Calcola score ensemble e risk label su `state['df_baseline']`."""
    logger.info("OutlierAgent -- Avvio")
    started_at = time.perf_counter()

    try:
        df = state.get("df_baseline")
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df_baseline mancante: esegui prima BaselineAgent.")
        if df.empty:
            raise ValueError("df_baseline vuoto: impossibile stimare outlier.")

        out = df.copy()

        # Segnali base (deterministici) per approssimare IF/LOF/Z/AE.
        out["score_if"] = _minmax(out["score_composito"]) if "score_composito" in out.columns else 0.0
        out["score_lof"] = _minmax(out["baseline_score"]) if "baseline_score" in out.columns else 0.0

        z_cols = [c for c in out.columns if c.startswith("z_")]
        z_proxy = out[z_cols].abs().mean(axis=1) if z_cols else pd.Series(np.zeros(len(out)), index=out.index)
        out["score_z"] = _minmax(z_proxy)

        # Proxy AE: combinazione IF/LOF (stabile e bounded).
        out["score_ae"] = ((out["score_if"] + out["score_lof"]) / 2).clip(0, 1)

        # Ensemble pesata con pesi condivisi nel contratto.
        out["ensemble_score"] = (
            out["score_if"] * ENSEMBLE_WEIGHTS["IF"] +
            out["score_lof"] * ENSEMBLE_WEIGHTS["LOF"] +
            out["score_z"] * ENSEMBLE_WEIGHTS["Z"] +
            out["score_ae"] * ENSEMBLE_WEIGHTS["AE"]
        ).clip(0, 1)

        out["risk_label"] = np.where(
            out["ensemble_score"] >= THRESHOLD_ALTA,
            "ALTA",
            np.where(out["ensemble_score"] >= THRESHOLD_MEDIA, "MEDIA", "NORMALE"),
        )

        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / "data" / "processed" / "anomaly_results_live.csv"
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("OutlierAgent output salvato in: %s", saved_to)

        meta = {
            "n_alta": int((out["risk_label"] == "ALTA").sum()),
            "n_media": int((out["risk_label"] == "MEDIA").sum()),
            "n_normale": int((out["risk_label"] == "NORMALE").sum()),
            "soglia_alta": float(THRESHOLD_ALTA),
            "soglia_media": float(THRESHOLD_MEDIA),
            "metodo_ensemble": "weighted_average",
            "saved_to": saved_to,
            "top_rotte": out.sort_values("ensemble_score", ascending=False)
            .head(10)[["ROTTA", "ensemble_score", "risk_label"]]
            .to_dict(orient="records"),
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }

        logger.info(
            "OutlierAgent ✓ Completato — ALTA=%d MEDIA=%d NORMALE=%d",
            meta["n_alta"], meta["n_media"], meta["n_normale"],
        )
        return {**state, "df_anomalies": out, "anomaly_meta": meta}
    except Exception as e:
        logger.error("OutlierAgent ✗ Errore: %s", e)
        return {
            **state,
            "df_anomalies": None,
            "anomaly_meta": {
                "error": str(e),
                "user_message": "Outlier detection fallita: verifica output baseline e filtri selezionati.",
                "elapsed_s": round(time.perf_counter() - started_at, 3),
            },
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent import data_agent_node
    from multiagent_pipeline.agents.feature_agent import run_feature_agent
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    s: AgentState = {"perimeter": {"anno": 2024}}
    s = data_agent_node(s)
    s = run_feature_agent(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    print("\n=== RISULTATO OutlierAgent ===")
    print(s["anomaly_meta"])
