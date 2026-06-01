"""OutlierAgent — fourth node of the multi-agent graph.

Responsibilities (from the Reply slide):
    "Applies IsolationForest, LOF, or Z-score on the engineered features"

Implements the three real models on sklearn, identical to classical notebook 04:
    - IsolationForest  (contamination=0.03, random_state=42)
    - LocalOutlierFactor (n_neighbors=20, contamination=0.03)
    - Z-score          (on BASELINE_FEATURES, already computed by BaselineAgent)

Weighted ensemble with the same ENSEMBLE_WEIGHTS as the classical pipeline.
Data-driven thresholds (p97/p90) for consistency with notebook 05.
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from multiagent_pipeline.state import (
    AgentState,
    BASELINE_FEATURES,
    ENSEMBLE_WEIGHTS,
)
from shared.autoencoder import train_and_score as _ae_train_and_score

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Same hyperparameters as classical notebook 04
_CONTAMINATION  = 0.10   # classical: contamination=0.10 (10% expected anomalous routes)
_N_NEIGHBORS    = 20
_RANDOM_STATE   = 42
_N_ESTIMATORS   = 200    # classical: 200 trees for IsolationForest


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Selects the available numeric features for the ML models.

    Uses BASELINE_FEATURES if present, otherwise all numeric columns
    excluding metadata/already-computed score columns.
    """
    exclude = {"ZONA", "n_osservazioni_allarmi", "n_osservazioni_viag",
               "score_composito", "baseline_score", "baseline_flag"}
    exclude |= {c for c in df.columns if c.startswith("z_")}

    # Priority: BASELINE_FEATURES present in the df
    bl_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    if bl_cols:
        cols = bl_cols
    else:
        cols = [c for c in df.select_dtypes(include="number").columns
                if c not in exclude]

    X = df[cols].fillna(0.0)
    return X, cols


def run_outlier_agent(
    state: AgentState,
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> AgentState:
    """Applies IsolationForest, LOF and Z-score on df_baseline → ensemble score."""
    logger.info("OutlierAgent -- Starting")
    started_at = time.perf_counter()

    try:
        df = state.get("df_baseline")
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df_baseline missing: run BaselineAgent first.")
        if df.empty:
            raise ValueError("df_baseline is empty: cannot estimate outliers.")

        out = df.copy()
        X, feat_cols = _get_feature_matrix(out)
        logger.info("Features used for ML: %d columns — %s", len(feat_cols), feat_cols)

        # ── Normalization ─────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── 1. IsolationForest ────────────────────────────────────────────────
        if_model = IsolationForest(
            contamination=_CONTAMINATION,
            random_state=_RANDOM_STATE,
            n_estimators=_N_ESTIMATORS,
        )
        # decision_function: lower = more anomalous → invert and normalize
        if_raw = if_model.fit(X_scaled).decision_function(X_scaled)
        out["score_if"] = _minmax(pd.Series(-if_raw, index=out.index))
        logger.info("IsolationForest: score_if range [%.4f, %.4f]",
                    out["score_if"].min(), out["score_if"].max())

        # ── 2. LocalOutlierFactor ─────────────────────────────────────────────
        n_neighbors = min(_N_NEIGHBORS, len(out) - 1)
        if n_neighbors < _N_NEIGHBORS:
            logger.warning("LOF: n_neighbors reduced to %d (dataset has only %d rows)",
                           n_neighbors, len(out))
        lof_model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=_CONTAMINATION,
        )
        lof_model.fit(X_scaled)
        lof_scores = lof_model.negative_outlier_factor_    # more negative = more anomalous
        out["score_lof"] = _minmax(pd.Series(-lof_scores, index=out.index))
        logger.info("LOF: score_lof range [%.4f, %.4f]",
                    out["score_lof"].min(), out["score_lof"].max())

        # ── 3. Z-score (consume baseline_score directly from BaselineAgent) ──
        # The BaselineAgent already aggregates the per-feature MAD z-scores
        # into the `baseline_score` column (mean of |z|). Using it directly
        # makes the data flow visible (BaselineAgent → OutlierAgent) and
        # removes the duplicated mean-of-z computation that used to live here.
        if "baseline_score" in out.columns:
            z_signal = pd.to_numeric(out["baseline_score"], errors="coerce").fillna(0.0)
            z_source = "baseline_score (from BaselineAgent)"
        else:
            # Fallback chain — only fires if BaselineAgent did not run or its
            # output schema is unexpected. Behaviour is identical to the
            # historical inline computation.
            z_cols = [c for c in out.columns if c.startswith("z_")]
            if z_cols:
                z_signal = out[z_cols].abs().mean(axis=1)
                z_source = f"mean(|z|) over {len(z_cols)} z_ cols (fallback)"
            else:
                z_signal = pd.Series(np.abs(X_scaled).mean(axis=1), index=out.index)
                z_source = "|X_scaled| mean (fallback, BaselineAgent missing)"
        out["score_z"] = _minmax(z_signal)
        logger.info(
            "Z-score: source=%s, score_z range [%.4f, %.4f]",
            z_source, out["score_z"].min(), out["score_z"].max(),
        )

        # ── 4. Autoencoder (shared deterministic module) ─────────────────────
        # Delegates to ``shared.autoencoder.train_and_score`` which forces a
        # stable row order (sort by ROTTA) and disables early stopping so
        # both pipelines produce IDENTICAL AE scores on the same input.
        # The previous local implementation used ``early_stopping=True``
        # which introduced run-to-run variability on borderline routes;
        # that source of stochastic divergence has been eliminated.
        normal_mask_arr = (if_model.predict(X_scaled) == 1)
        row_ids = (
            out["ROTTA"].astype(str).values if "ROTTA" in out.columns
            else np.arange(len(out))
        )
        ae_result = _ae_train_and_score(
            X_scaled,
            normal_mask=normal_mask_arr,
            row_ids=row_ids,
        )
        out["score_ae"] = ae_result.score_ae
        use_autoencoder = ae_result.use_ae

        # ── Weighted ensemble ────────────────────────────────────────────────
        # If Autoencoder is active: same weights as the classical pipeline.
        # If excluded: redistribute AE weight proportionally among the other 3.
        if use_autoencoder:
            w_if  = ENSEMBLE_WEIGHTS["IF"]
            w_lof = ENSEMBLE_WEIGHTS["LOF"]
            w_z   = ENSEMBLE_WEIGHTS["Z"]
            w_ae  = ENSEMBLE_WEIGHTS["AE"]
        else:
            base = ENSEMBLE_WEIGHTS["IF"] + ENSEMBLE_WEIGHTS["LOF"] + ENSEMBLE_WEIGHTS["Z"]
            w_if  = ENSEMBLE_WEIGHTS["IF"]  / base
            w_lof = ENSEMBLE_WEIGHTS["LOF"] / base
            w_z   = ENSEMBLE_WEIGHTS["Z"]   / base
            w_ae  = 0.0
            logger.info("Redistributed weights: IF=%.2f, LOF=%.2f, Z=%.2f",
                        w_if, w_lof, w_z)

        out["ensemble_score"] = (
            out["score_if"]  * w_if  +
            out["score_lof"] * w_lof +
            out["score_z"]   * w_z   +
            out["score_ae"]  * w_ae
        ).clip(0, 1)
        logger.info("Ensemble: range [%.4f, %.4f]",
                    out["ensemble_score"].min(), out["ensemble_score"].max())

        # ── Data-driven thresholds (p97/p90) — identical to classical notebook 05 ──
        threshold_high   = float(out["ensemble_score"].quantile(0.97))
        threshold_medium = float(out["ensemble_score"].quantile(0.90))
        logger.info("Data-driven thresholds: HIGH=%.4f (p97) | MEDIUM=%.4f (p90)",
                    threshold_high, threshold_medium)

        out["anomaly_label"] = np.where(
            out["ensemble_score"] >= threshold_high, "HIGH",
            np.where(out["ensemble_score"] >= threshold_medium, "MEDIUM", "NORMAL"),
        )

        saved_to = None
        if save_output:
            default_out = _PROJECT_ROOT / "data" / "processed" / "anomaly_results_live.csv"
            out_path = Path(output_path) if output_path is not None else default_out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False)
            saved_to = str(out_path)
            logger.info("OutlierAgent output saved to: %s", saved_to)

        meta = {
            "n_high"          : int((out["anomaly_label"] == "HIGH").sum()),
            "n_medium"        : int((out["anomaly_label"] == "MEDIUM").sum()),
            "n_normal"        : int((out["anomaly_label"] == "NORMAL").sum()),
            "threshold_high"  : threshold_high,
            "threshold_medium": threshold_medium,
            "threshold_method": "data-driven (p97/p90)",
            "autoencoder_used": use_autoencoder,
            "ensemble_method" : (
                "IF + LOF + Z-score + Autoencoder (real sklearn)"
                if use_autoencoder
                else "IF + LOF + Z-score (Autoencoder excluded: dataset too small)"
            ),
            "feature_cols"    : feat_cols,
            "n_features"      : len(feat_cols),
            "saved_to"        : saved_to,
            "top_routes"      : (
                out.sort_values("ensemble_score", ascending=False)
                .head(10)[["ROTTA", "ensemble_score", "anomaly_label"]]
                .to_dict(orient="records")
            ),
            "elapsed_s": round(time.perf_counter() - started_at, 3),
        }

        logger.info(
            "OutlierAgent ✓ Completed — HIGH=%d MEDIUM=%d NORMAL=%d (%.2fs)",
            meta["n_high"], meta["n_medium"], meta["n_normal"], meta["elapsed_s"],
        )
        return {**state, "df_anomalies": out, "anomaly_meta": meta}

    except Exception as e:
        logger.error("OutlierAgent ✗ Error: %s", e)
        return {
            **state,
            "df_anomalies": None,
            "anomaly_meta": {
                "error"       : str(e),
                "user_message": "Outlier detection failed: check baseline output and filters.",
                "elapsed_s"   : round(time.perf_counter() - started_at, 3),
            },
        }


if __name__ == "__main__":
    from multiagent_pipeline.agents.data_agent import data_agent_node
    # DataAgent now produces df_features inline
    from multiagent_pipeline.agents.baseline_agent import run_baseline_agent
    from multiagent_pipeline.tools.data_tools import load_last_perimeter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _perimeter = load_last_perimeter() or {"anno": 2024}
    print(f"  Perimeter: {_perimeter}")
    s: AgentState = {"perimeter": _perimeter}
    s = data_agent_node(s)
    s = run_baseline_agent(s)
    s = run_outlier_agent(s)
    print("\n=== OutlierAgent RESULT ===")
    am = s["anomaly_meta"]
    print(f"  HIGH={am['n_high']} | MEDIUM={am['n_medium']} | NORMAL={am['n_normal']}")
    print(f"  threshold_high={am['threshold_high']:.4f} | threshold_medium={am['threshold_medium']:.4f}")
    print(f"  method: {am['ensemble_method']}")
    print(f"  features: {am['n_features']} columns")
    print(f"  elapsed: {am['elapsed_s']}s")
    print(f"  top routes: {am['top_routes'][:3]}")
