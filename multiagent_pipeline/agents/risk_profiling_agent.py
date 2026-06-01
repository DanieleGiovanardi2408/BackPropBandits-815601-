"""RiskProfilingAgent — fifth node of the multi-agent graph.

Responsibilities (from the Reply slide):
    "Applies business rules equivalent to the classical post-processing layer
     (e.g. 'alert rate on route X exceeds 3x baseline')."

Implements the five canonical business rules shared with the inline classical
pipeline in main.ipynb (Section 6). Thresholds and logic are strictly
identical on both sides, so the comparative analysis in Section 9 reports a
true convergence certificate rather than an artefact of different rules.

Business rules (each returns 0/1):
    br_high_interpol     pct_interpol         >= 0.30
    br_high_rejection    tasso_respinti       >= 0.25
    br_low_closure       tot_allarmi_log > 3  AND tasso_chiusura < 0.10
    br_multi_source      pct_interpol >= 0.10 AND pct_sdi >= 0.10
    br_high_alarm_rate   tasso_allarme_medio  >= 0.50

Aggregates:
    br_score   = sum(br_*) / 5                              in [0, 1]
    confidence = 0.60 * ensemble_score + 0.40 * br_score    in [0, 1]
    final_risk in {CRITICAL, HIGH, MEDIUM, LOW}
        - CRITICAL : anomaly_label == HIGH    AND br_score >= 0.4
        - HIGH     : anomaly_label == HIGH    OR (MEDIUM AND br_score >= 0.4)
        - MEDIUM   : anomaly_label == MEDIUM
        - LOW      : otherwise

Also produces a `risk_drivers` list per route (textual reason codes) used
downstream by ReportAgent for richer LLM explanations.
"""
from __future__ import annotations

# ── Bootstrap for direct execution ───────────────────────────────────────────
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

from multiagent_pipeline.state import AgentState

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─── Business-rule thresholds ────────────────────────────────────────────────
# These constants MUST stay aligned with main.ipynb Section 6 (inline classical
# post-processing) and with classical_pipeline/main.py step_post_processing().
# Changing them here without updating those two sides breaks the comparative
# analysis in Section 9 of main.ipynb.
BR_THRESHOLDS = {
    "high_interpol_pct":     0.30,
    "high_rejection_rate":   0.25,
    "low_closure_volume":    3.0,    # tot_allarmi_log > 3
    "low_closure_rate":      0.10,   # AND tasso_chiusura < 0.10
    "multi_source_pct":      0.10,   # pct_interpol >= 0.10 AND pct_sdi >= 0.10
    "high_alarm_rate":       0.50,
}

# Confidence blend weights (60% ML / 40% rules) — same as classical pipeline.
CONFIDENCE_WEIGHTS = {"ml": 0.60, "rules": 0.40}

# Driver labels surfaced to the LLM ReportAgent.
_DRIVER_LABELS = {
    "br_high_interpol":   "High INTERPOL alarm share (>=30%)",
    "br_high_rejection":  "High traveller rejection rate (>=25%)",
    "br_low_closure":     "Operational backlog: high alarm volume with low closure rate",
    "br_multi_source":    "Multi-database corroboration (INTERPOL + SDI, both >=10%)",
    "br_high_alarm_rate": "High average alarm-per-traveller ratio (>=50%)",
}


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Returns the column if present, otherwise a zero-filled Series."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    logger.warning("RiskProfilingAgent: column '%s' missing — defaulting to 0", col)
    return pd.Series(np.zeros(len(df)), index=df.index)


def _classify_final(ml_label: str, br_score: float) -> str:
    """Replicates the canonical final-risk classification ladder.

    Mirrors the inline classical post-processing in main.ipynb (Section 6)
    and classical_pipeline.main.step_post_processing().
    """
    if ml_label == "HIGH" and br_score >= 0.4:
        return "CRITICAL"
    if ml_label == "HIGH" or (ml_label == "MEDIUM" and br_score >= 0.4):
        return "HIGH"
    if ml_label == "MEDIUM":
        return "MEDIUM"
    return "LOW"


def _drivers_for_row(row: pd.Series) -> list[str]:
    """Builds the risk-drivers narrative list for a single route."""
    drivers: list[str] = []
    for col, label in _DRIVER_LABELS.items():
        if int(row.get(col, 0)) == 1:
            drivers.append(label)
    return drivers


def run_risk_profiling_agent(
    state: AgentState,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Applies business rules on the OutlierAgent output.

    Reads ``state['df_anomalies']`` (must contain anomaly_label + ensemble_score
    + the BASELINE_FEATURES used by the rules), produces:
        * df_risk        : df_anomalies + br_* columns + br_score
                            + confidence + final_risk + risk_drivers
        * risk_meta      : counts per final_risk level + summary stats
    """
    logger.info("RiskProfilingAgent ── Starting")
    started_at = time.perf_counter()

    try:
        df_in = state.get("df_anomalies")
        a_meta = state.get("anomaly_meta") or {}

        if a_meta.get("error"):
            raise ValueError(f"anomaly_meta contains error: {a_meta['error']}")
        if df_in is None or not isinstance(df_in, pd.DataFrame):
            raise ValueError("df_anomalies missing: run OutlierAgent first.")
        if df_in.empty:
            raise ValueError("df_anomalies is empty: cannot apply business rules.")

        df = df_in.copy()

        # ── 1. Five binary business rules (same logic as classical) ──────────
        pct_interpol         = _safe_col(df, "pct_interpol")
        pct_sdi              = _safe_col(df, "pct_sdi")
        tasso_respinti       = _safe_col(df, "tasso_respinti")
        tot_allarmi_log      = _safe_col(df, "tot_allarmi_log")
        tasso_chiusura       = _safe_col(df, "tasso_chiusura")
        tasso_allarme_medio  = _safe_col(df, "tasso_allarme_medio")

        df["br_high_interpol"]   = (pct_interpol         >= BR_THRESHOLDS["high_interpol_pct"]).astype(int)
        df["br_high_rejection"]  = (tasso_respinti       >= BR_THRESHOLDS["high_rejection_rate"]).astype(int)
        df["br_low_closure"]     = (
            (tot_allarmi_log     >  BR_THRESHOLDS["low_closure_volume"]) &
            (tasso_chiusura      <  BR_THRESHOLDS["low_closure_rate"])
        ).astype(int)
        df["br_multi_source"]    = (
            (pct_interpol >= BR_THRESHOLDS["multi_source_pct"]) &
            (pct_sdi      >= BR_THRESHOLDS["multi_source_pct"])
        ).astype(int)
        df["br_high_alarm_rate"] = (tasso_allarme_medio  >= BR_THRESHOLDS["high_alarm_rate"]).astype(int)

        # ── 2. Aggregate score (mean of binary rules) ────────────────────────
        df["br_score"] = (
            df["br_high_interpol"]
            + df["br_high_rejection"]
            + df["br_low_closure"]
            + df["br_multi_source"]
            + df["br_high_alarm_rate"]
        ) / 5.0

        # ── 3. Confidence blend (ML + rules) ─────────────────────────────────
        ensemble = pd.to_numeric(df["ensemble_score"], errors="coerce").fillna(0.0)
        df["confidence"] = (
            CONFIDENCE_WEIGHTS["ml"] * ensemble
            + CONFIDENCE_WEIGHTS["rules"] * df["br_score"]
        ).round(4)

        # ── 4. Final risk classification (CRITICAL/HIGH/MEDIUM/LOW) ─────────
        df["final_risk"] = [
            _classify_final(label, score)
            for label, score in zip(df["anomaly_label"], df["br_score"])
        ]

        # ── 5. Per-route narrative drivers (consumed by ReportAgent) ────────
        df["risk_drivers"] = df.apply(_drivers_for_row, axis=1)

        # ── 6. Sort by confidence (most actionable first) ───────────────────
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

        # ── 7. Persist if requested ─────────────────────────────────────────
        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / "data" / "processed" / "risk_profiles_live.csv"
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # CSV cannot store list columns cleanly → join drivers
            df_to_save = df.copy()
            df_to_save["risk_drivers"] = df_to_save["risk_drivers"].apply(
                lambda lst: " | ".join(lst) if isinstance(lst, list) else ""
            )
            df_to_save.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("RiskProfilingAgent output saved to: %s", saved_to)

        # ── 8. Build meta payload ───────────────────────────────────────────
        risk_counts = df["final_risk"].value_counts().to_dict()
        rule_hits = {col: int(df[col].sum()) for col in _DRIVER_LABELS}

        risk_meta = {
            "n_routes":         int(len(df)),
            "n_critical":       int(risk_counts.get("CRITICAL", 0)),
            "n_high":           int(risk_counts.get("HIGH", 0)),
            "n_medium":         int(risk_counts.get("MEDIUM", 0)),
            "n_low":            int(risk_counts.get("LOW", 0)),
            "rule_hits":        rule_hits,
            "br_thresholds":    BR_THRESHOLDS,
            "confidence_weights": CONFIDENCE_WEIGHTS,
            "top_routes":       (
                df.head(10)[
                    ["ROTTA", "anomaly_label", "final_risk",
                     "ensemble_score", "br_score", "confidence"]
                ].to_dict(orient="records")
            ),
            "saved_to":         saved_to,
            "elapsed_s":        round(time.perf_counter() - started_at, 3),
        }

        logger.info(
            "RiskProfilingAgent ✓ Completed — CRITICAL=%d HIGH=%d MEDIUM=%d LOW=%d (%.2fs)",
            risk_meta["n_critical"], risk_meta["n_high"],
            risk_meta["n_medium"],   risk_meta["n_low"],
            risk_meta["elapsed_s"],
        )

        return {
            **state,
            "df_risk":   df,
            "risk_meta": risk_meta,
        }

    except Exception as e:
        logger.error("RiskProfilingAgent ✗ Error: %s", e)
        return {
            **state,
            "df_risk": None,
            "risk_meta": {
                "error":        str(e),
                "user_message": "Risk profiling failed: check that OutlierAgent ran "
                                "and the BASELINE_FEATURES columns are present.",
                "elapsed_s":    round(time.perf_counter() - started_at, 3),
            },
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent     import data_agent_node
    
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.agents.outlier_agent  import run_outlier_agent
    from multiagent_pipeline.tools.data_tools      import load_last_perimeter

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    _perimeter = load_last_perimeter() or {"anno": 2024}
    print(f"  Perimeter: {_perimeter}")
    s: AgentState = {"perimeter": _perimeter}
    s = data_agent_node(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    s = run_risk_profiling_agent(s)

    print("\n=== RiskProfilingAgent RESULT ===")
    rm = s["risk_meta"]
    if rm.get("error"):
        print("ERROR:", rm["error"])
    else:
        print(f"  CRITICAL={rm['n_critical']}  HIGH={rm['n_high']}  "
              f"MEDIUM={rm['n_medium']}  LOW={rm['n_low']}")
        print(f"  Rule hits: {rm['rule_hits']}")
        print(f"  Top 3 routes by confidence: {rm['top_routes'][:3]}")
