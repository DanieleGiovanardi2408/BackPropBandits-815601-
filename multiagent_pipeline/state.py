"""
state.py
────────
Shared data contract between all agents in the multi-agent system.

This file defines:
  1. AgentState  — the LangGraph state that flows between nodes
  2. Pydantic models — input/output schemas for each agent
  3. Constants   — thresholds and weights shared with the classical pipeline

RULE: no agent imports another agent directly.
      They communicate ONLY through AgentState.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

import pandas as pd
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# 1. SHARED LANGGRAPH STATE
#    Each field corresponds to the output of a specific agent.
#    Fields are Optional: an agent not yet executed leaves its field as None.
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    State that flows between nodes of the LangGraph graph.

    Flow:
        DataAgent → FeatureAgent → BaselineAgent → OutlierAgent → ReportAgent

    Each agent reads fields from its predecessors and writes to its own field.
    """

    # ── User input ─────────────────────────────────────────────────────────────
    perimeter: dict                       # user filters: year, airport, country, zone

    # ── DataAgent output ──────────────────────────────────────────────────────
    df_raw: Optional[Any]                 # pd.DataFrame — filtered merged dataset
    df_allarmi: Optional[Any]             # pd.DataFrame — cleaned alarms
    df_viaggiatori: Optional[Any]         # pd.DataFrame — cleaned traveller records
    data_meta: Optional[dict]             # dataset stats: n_rows, n_routes, columns

    # ── FeatureAgent output ───────────────────────────────────────────────────
    df_features: Optional[Any]            # pd.DataFrame — 54 features aggregated per route
    feature_meta: Optional[dict]          # n_routes, n_features, list of feature columns

    # ── BaselineAgent output ──────────────────────────────────────────────────
    df_baseline: Optional[Any]            # pd.DataFrame — features + z-scores vs baseline
    baseline_meta: Optional[dict]         # z-score threshold, n_baseline_features, source

    # ── OutlierAgent output ───────────────────────────────────────────────────
    df_anomalies: Optional[Any]           # pd.DataFrame — IF/LOF/Z/AE scores + risk label
    anomaly_meta: Optional[dict]          # n_alta, n_media, n_normale, thresholds used

    # ── RiskProfilingAgent output ─────────────────────────────────────────────
    df_risk: Optional[Any]                # pd.DataFrame — anomalies + br_* + final_risk
    risk_meta: Optional[dict]             # n_critico, n_alto, rule_hits, thresholds

    # ── ReportAgent output ────────────────────────────────────────────────────
    report: Optional[dict]                # final report with LLM explanations
    report_path: Optional[str]            # path of the JSON file saved to disk


# ══════════════════════════════════════════════════════════════════════════════
# 2. PER-AGENT INPUT / OUTPUT SCHEMAS
#    Used to validate data entering and leaving each node.
# ══════════════════════════════════════════════════════════════════════════════

class Perimeter(BaseModel):
    """
    Filter parameters passed by the user (or Streamlit UI) to DataAgent.
    All optional: if not specified, no filter is applied.
    """
    anno: Optional[int] = Field(None, description="e.g. 2024")
    aeroporto_arrivo: Optional[str] = Field(None, description="e.g. 'FCO'")
    aeroporto_partenza: Optional[str] = Field(None, description="e.g. 'ALG'")
    paese_partenza: Optional[str] = Field(None, description="e.g. 'Algeria'")
    zona: Optional[int] = Field(None, description="Geographic zone 1-9")


class DataAgentOutput(BaseModel):
    """DataAgent output — passed to FeatureAgent."""
    n_righe: int
    n_rotte_uniche: int
    colonne: list[str]
    anni_presenti: list[int]
    paesi_partenza_top5: list[str]


class FeatureAgentOutput(BaseModel):
    """FeatureAgent output — passed to BaselineAgent."""
    n_rotte: int
    n_features: int
    feature_cols: list[str]
    rotte_sample: list[str] = Field(description="First 5 routes in the dataset")


class BaselineAgentOutput(BaseModel):
    """BaselineAgent output — passed to OutlierAgent."""
    n_features_baseline: int
    z_score_threshold: float
    source: str = Field(description="'precomputed' or 'computed_live'")
    n_rotte_con_zscore: int


class OutlierAgentOutput(BaseModel):
    """OutlierAgent output — passed to RiskProfilingAgent."""
    n_alta: int
    n_media: int
    n_normale: int
    soglia_alta: float
    soglia_media: float
    metodo_ensemble: str = "weighted_average"
    top_rotte: list[dict] = Field(description="Top 10 anomalous routes with scores")


class RiskProfilingAgentOutput(BaseModel):
    """RiskProfilingAgent output — passed to ReportAgent.

    Mirrors the classical post-processing layer: business rules, blended
    confidence and final risk classification (CRITICO/ALTO/MEDIO/BASSO).
    """
    n_routes: int
    n_critico: int
    n_alto: int
    n_medio: int
    n_basso: int
    rule_hits: dict = Field(description="Hits per business rule (br_high_*)")
    top_routes: list[dict] = Field(description="Top 10 routes by confidence")


class ReportAgentOutput(BaseModel):
    """ReportAgent final output."""
    n_anomalie_spiegate: int
    report_path: str
    sommario: str = Field(description="Plain-English summary of the report")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONSTANTS SHARED WITH THE CLASSICAL PIPELINE
#    These values must be IDENTICAL to those used in the classical notebooks
#    to ensure a fair comparison between the two architectures.
# ══════════════════════════════════════════════════════════════════════════════

# Ensemble weights — same as classical pipeline (anomaly_summary.json)
ENSEMBLE_WEIGHTS = {
    "IF":  0.35,   # IsolationForest
    "LOF": 0.30,   # Local Outlier Factor
    "Z":   0.15,   # Z-score
    "AE":  0.20,   # Autoencoder
}

# Reference risk-label thresholds from the classical pipeline (anomaly_summary.json).
# The multi-agent recomputes them data-driven (p97/p90) at runtime in OutlierAgent;
# these values serve as documentation of the classical baseline only.
THRESHOLD_ALTA_CLASSICAL  = 0.3579   # classical p97
THRESHOLD_MEDIA_CLASSICAL = 0.2897   # classical p90

# Features used for z-score baseline — same as classical (baseline_stats.json)
BASELINE_FEATURES = [
    "tot_allarmi_log",
    "pct_interpol",
    "pct_sdi",
    "pct_nsis",
    "tasso_chiusura",
    "tasso_rilevanza",
    "tasso_allarme_medio",
    "tasso_inv_medio",
    "score_rischio_esiti",
    "tasso_respinti",
    "tasso_fermati",
    # Extended features computed by FeatureBuilder
    "false_positive_rate",
    "alarm_per_invest",
]

# Key columns of the merged dataset (DataAgent → FeatureAgent contract)
DATASET_MERGED_COLS = [
    "AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA", "ANNO_PARTENZA", "MESE_PARTENZA",
    "PAESE_PART", "ZONA", "TOT", "MOTIVO_ALLARME", "flag_rischio",
    "tot_entrati", "tot_allarmati", "tot_investigati",
    "tasso_allarme_volo", "tasso_inv_volo",
    "n_respinti", "n_fermati", "n_segnalati",
]

# File paths (relative to project root)
PATHS = {
    "dataset_merged":   "data/processed/dataset_merged.csv",
    "allarmi_clean":    "data/processed/allarmi_clean.csv",
    "viaggiatori_clean":"data/processed/viaggiatori_clean.csv",
    "features":         "data/processed/features_classical.csv",
    "baseline_stats":   "data/processed/baseline_stats.json",
    "feature_cols":     "data/processed/feature_cols.json",
    "anomaly_results":  "data/processed/anomaly_results.csv",
    "multiagent_report": "data/processed/multiagent_report.json",
}
