"""
state.py
────────
Contratto dei dati condiviso tra tutti gli agenti del sistema multi-agent.

Questo file definisce:
  1. AgentState  — lo stato LangGraph che fluisce tra i nodi
  2. Pydantic models — gli schemi di input/output di ogni agente
  3. Costanti      — soglie e pesi condivisi con la pipeline classica

REGOLA: nessun agente importa direttamente un altro agente.
        Comunicano SOLO attraverso AgentState.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

import pandas as pd
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATO CONDIVISO LANGGRAPH
#    Ogni campo corrisponde all'output di un agente specifico.
#    I campi sono Optional: un agente non ancora eseguito lascia None.
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Stato che fluisce tra i nodi del grafo LangGraph.

    Flusso:
        DataAgent → FeatureAgent → BaselineAgent → OutlierAgent → ReportAgent

    Ogni agente legge i campi dei suoi predecessori e scrive nel suo campo.
    """

    # ── Input utente ───────────────────────────────────────────────────────────
    perimeter: dict                       # filtri utente: anno, aeroporto, paese, zona

    # ── Output DataAgent ──────────────────────────────────────────────────────
    df_raw: Optional[Any]                 # pd.DataFrame — dataset_merged filtrato
    data_meta: Optional[dict]             # statistiche dataset: n_righe, n_rotte, colonne

    # ── Output FeatureAgent ───────────────────────────────────────────────────
    df_features: Optional[Any]            # pd.DataFrame — 63 feature aggregate per ROTTA
    feature_meta: Optional[dict]          # n_rotte, n_features, lista colonne usate

    # ── Output BaselineAgent ──────────────────────────────────────────────────
    df_baseline: Optional[Any]            # pd.DataFrame — features + z-score vs baseline
    baseline_meta: Optional[dict]         # soglia z-score, n_features baseline, source

    # ── Output OutlierAgent ───────────────────────────────────────────────────
    df_anomalies: Optional[Any]           # pd.DataFrame — scores IF/LOF/Z/AE + label
    anomaly_meta: Optional[dict]          # n_alta, n_media, n_normale, soglie usate

    # ── Output ReportAgent ────────────────────────────────────────────────────
    report: Optional[dict]                # report finale con spiegazioni LLM
    report_path: Optional[str]            # path del JSON salvato su disco


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCHEMI INPUT / OUTPUT DI OGNI AGENTE
#    Usati per validare i dati in entrata e in uscita da ogni nodo.
# ══════════════════════════════════════════════════════════════════════════════

class Perimeter(BaseModel):
    """
    Parametri di filtro che l'utente (o la UI Streamlit) passa al DataAgent.
    Tutti opzionali: se non specificati, nessun filtro viene applicato.
    """
    anno: Optional[int] = Field(None, description="Es. 2024")
    aeroporto_arrivo: Optional[str] = Field(None, description="Es. 'FCO'")
    aeroporto_partenza: Optional[str] = Field(None, description="Es. 'ALG'")
    paese_partenza: Optional[str] = Field(None, description="Es. 'Algeria'")
    zona: Optional[int] = Field(None, description="Zona geografica 1-9")


class DataAgentOutput(BaseModel):
    """Output del DataAgent — passato a FeatureAgent."""
    n_righe: int
    n_rotte_uniche: int
    colonne: list[str]
    anni_presenti: list[int]
    paesi_partenza_top5: list[str]


class FeatureAgentOutput(BaseModel):
    """Output del FeatureAgent — passato a BaselineAgent."""
    n_rotte: int
    n_features: int
    feature_cols: list[str]
    rotte_sample: list[str] = Field(description="Prime 5 rotte nel dataset")


class BaselineAgentOutput(BaseModel):
    """Output del BaselineAgent — passato a OutlierAgent."""
    n_features_baseline: int
    z_score_threshold: float
    source: str = Field(description="'precomputed' o 'computed_live'")
    n_rotte_con_zscore: int


class OutlierAgentOutput(BaseModel):
    """Output dell'OutlierAgent — passato a ReportAgent."""
    n_alta: int
    n_media: int
    n_normale: int
    soglia_alta: float
    soglia_media: float
    metodo_ensemble: str = "weighted_average"
    top_rotte: list[dict] = Field(description="Top 10 rotte anomale con score")


class ReportAgentOutput(BaseModel):
    """Output finale del ReportAgent."""
    n_anomalie_spiegate: int
    report_path: str
    sommario: str = Field(description="Sommario in linguaggio naturale del report")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COSTANTI CONDIVISE CON LA PIPELINE CLASSICA
#    Questi valori devono essere IDENTICI a quelli usati nei notebook classici
#    per garantire un confronto onesto.
# ══════════════════════════════════════════════════════════════════════════════

# Pesi ensemble — stessi del classico (anomaly_summary.json)
ENSEMBLE_WEIGHTS = {
    "IF":  0.35,   # IsolationForest
    "LOF": 0.30,   # Local Outlier Factor
    "Z":   0.15,   # Z-score
    "AE":  0.20,   # Autoencoder
}

# Soglie risk label — stesse del classico (anomaly_summary.json)
THRESHOLD_ALTA  = 0.3579   # p97
THRESHOLD_MEDIA = 0.2897   # p90

# Feature usate per z-score baseline — stesse del classico (baseline_stats.json)
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
]

# Colonne chiave del dataset merged (contratto DataAgent → FeatureAgent)
DATASET_MERGED_COLS = [
    "AREOPORTO_ARRIVO", "AREOPORTO_PARTENZA", "ANNO_PARTENZA", "MESE_PARTENZA",
    "PAESE_PART", "ZONA", "TOT", "MOTIVO_ALLARME", "flag_rischio",
    "tot_entrati", "tot_allarmati", "tot_investigati",
    "tasso_allarme_volo", "tasso_inv_volo",
    "n_respinti", "n_fermati", "n_segnalati",
]

# Percorsi file (relativi alla root del progetto)
PATHS = {
    "dataset_merged":   "data/processed/dataset_merged.csv",
    "features":         "data/processed/features_classical.csv",
    "baseline_stats":   "data/processed/baseline_stats.json",
    "feature_cols":     "data/processed/feature_cols.json",
    "anomaly_results":  "data/processed/anomaly_results.csv",
    "multiagent_report": "data/processed/multiagent_report.json",
}
