"""
Classical Pipeline — End-to-End Orchestrator
=============================================
Runs the full classical anomaly-detection pipeline in sequence:

    Step 0: Preprocessing   (cleaning + merge)
    Step 1: Feature Engineering   (54 numerical features per route)
    Step 2: Baseline Construction (z-scores + anomaly flags)
    Step 3: Anomaly Detection     (4-model ensemble)
    Step 4: Post-Processing       (business rules + risk profiles)
    Step 5: Evaluation            (stability, feature importance, SHAP)

Each step mirrors the corresponding notebook (01–06) and reuses the
same shared modules (preprocessing.py, features.py) to guarantee
identical logic with the multi-agent pipeline.

Usage:
    PYTHONPATH=. python classical_pipeline/main.py [--skip-eval] [--verbose]

Outputs (all saved to data/processed/):
    - allarmi_clean.csv, viaggiatori_clean.csv, dataset_merged.csv
    - features_classical.csv, feature_cols.json
    - features_with_baseline.csv, baseline_stats.json
    - anomaly_results.csv, anomaly_summary.json
    - final_report.csv, risk_profiles.json
    - evaluation_scorecard.json (unless --skip-eval)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── Project root and paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))

from shared.preprocessing import run_preprocessing
from multiagent_pipeline.src.features import FeatureBuilder

warnings.filterwarnings("ignore")
logger = logging.getLogger("classical_pipeline")

# ── Shared constants (must match multiagent_pipeline/state.py) ────────────────

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
    # Extended features computed by FeatureBuilder (feature_engineering notebooks)
    "false_positive_rate",
    "alarm_per_invest",
]

ENSEMBLE_WEIGHTS = {"IF": 0.35, "LOF": 0.30, "Z": 0.15, "AE": 0.20}
THRESHOLD_ALTA = 0.3579   # p97
THRESHOLD_MEDIA = 0.2897  # p90


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def step_feature_engineering(
    df_allarmi: pd.DataFrame,
    df_viaggiatori: pd.DataFrame,
) -> pd.DataFrame:
    """Build route-level features using FeatureBuilder (shared with multi-agent)."""
    logger.info("Step 1: Feature Engineering")

    builder = FeatureBuilder()
    features = builder.build(df_allarmi, df_viaggiatori)

    # Save feature list metadata
    num_cols = features.select_dtypes(include="number").columns.tolist()
    feature_meta = {
        "n_routes": len(features),
        "n_features": len(num_cols),
        "feature_cols": num_cols,
    }
    with open(PROC_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    features.to_csv(PROC_DIR / "features_classical.csv", index=False)
    logger.info(
        "  -> %d routes, %d features -> features_classical.csv",
        len(features), len(num_cols),
    )
    return features


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Baseline Construction
# ══════════════════════════════════════════════════════════════════════════════

def step_baseline_construction(features: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline statistics and z-scores for each route."""
    logger.info("Step 2: Baseline Construction")

    X = features[BASELINE_FEATURES].copy()

    # ── Per-feature baseline stats ────────────────────────────────────────────
    baseline_stats: dict = {}
    for feat in BASELINE_FEATURES:
        col = X[feat]
        q1, q3 = float(col.quantile(0.25)), float(col.quantile(0.75))
        iqr = q3 - q1
        mean, std = float(col.mean()), float(col.std())
        p95 = float(col.quantile(0.95))
        p99 = float(col.quantile(0.99))

        if iqr > 0:
            tukey_upper = q3 + 1.5 * iqr
            tukey_lower = q1 - 1.5 * iqr
        else:
            tukey_upper = p95
            tukey_lower = -np.inf

        baseline_stats[feat] = {
            "mean": mean, "std": std, "median": float(col.median()),
            "q1": q1, "q3": q3, "iqr": iqr,
            "tukey_upper": tukey_upper, "tukey_lower": tukey_lower,
            "p95": p95, "p99": p99,
            "z_upper": mean + 2.5 * std,
            "is_sparse": int(iqr == 0),
        }

    # ── Z-scores ──────────────────────────────────────────────────────────────
    stds_safe = X.std().replace(0, 1.0)
    Z = (X - X.mean()) / stds_safe
    Z.columns = [f"z_{c}" for c in Z.columns]

    # ── Anomaly flags per feature (hybrid Tukey + z-score) ────────────────────
    flag_df = pd.DataFrame(index=features.index)
    for feat in BASELINE_FEATURES:
        stats = baseline_stats[feat]
        flag_tukey = features[feat] > stats["tukey_upper"]
        flag_z = Z[f"z_{feat}"] > 2.5
        flag_df[f"flag_{feat}"] = (flag_tukey | flag_z).astype(int)

    flag_df["n_anomalie"] = flag_df.filter(like="flag_").sum(axis=1)
    flag_df["pct_anomalie"] = (flag_df["n_anomalie"] / len(BASELINE_FEATURES)).round(4)

    # ── Combine ───────────────────────────────────────────────────────────────
    features_wb = pd.concat([features, Z, flag_df], axis=1)

    # ── Save ──────────────────────────────────────────────────────────────────
    features_wb.to_csv(PROC_DIR / "features_with_baseline.csv", index=False)

    baseline_meta = {
        "anomaly_features": BASELINE_FEATURES,
        "n_features": len(BASELINE_FEATURES),
        "n_routes": len(features),
        "z_score_threshold": 2.5,
        "stats": baseline_stats,
    }
    with open(PROC_DIR / "baseline_stats.json", "w") as f:
        json.dump(baseline_meta, f, indent=2)

    n_flagged = int((flag_df["n_anomalie"] >= 1).sum())
    logger.info(
        "  -> %d routes with >= 1 anomaly flag -> features_with_baseline.csv",
        n_flagged,
    )
    return features_wb


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Anomaly Detection (4-model ensemble)
# ══════════════════════════════════════════════════════════════════════════════

def step_anomaly_detection(features_wb: pd.DataFrame) -> pd.DataFrame:
    """Run 4-model ensemble: IsolationForest, LOF, Z-score, Autoencoder."""
    logger.info("Step 3: Anomaly Detection (ensemble)")

    X_raw = features_wb[BASELINE_FEATURES].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Model 1: Isolation Forest ─────────────────────────────────────────────
    iso_forest = IsolationForest(
        n_estimators=200, contamination=0.10,
        max_samples="auto", random_state=42, n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    if_raw = iso_forest.score_samples(X_scaled)
    if_anomaly = np.clip((if_raw - if_raw.max()) / (if_raw.min() - if_raw.max()), 0, 1)

    # ── Model 2: Local Outlier Factor ─────────────────────────────────────────
    lof = LocalOutlierFactor(
        n_neighbors=20, contamination=0.10,
        metric="euclidean", n_jobs=-1,
    )
    lof.fit(X_scaled)
    lof_raw = -lof.negative_outlier_factor_
    lof_anomaly = np.clip((lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min()), 0, 1)

    # ── Model 3: Z-score baseline ─────────────────────────────────────────────
    z_anomaly = features_wb["pct_anomalie"].fillna(0).values

    # ── Model 4: Autoencoder (MLPRegressor) ───────────────────────────────────
    normal_mask = iso_forest.predict(X_scaled) == 1
    X_normal = X_scaled[normal_mask]
    ae = MLPRegressor(
        hidden_layer_sizes=(8, 4, 8),
        activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=1000,
        random_state=42, early_stopping=True,
        validation_fraction=0.10, n_iter_no_change=20,
        verbose=False,
    )
    ae.fit(X_normal, X_normal)
    X_reconstructed = ae.predict(X_scaled)
    ae_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    ae_anomaly = np.clip((ae_error - ae_error.min()) / (ae_error.max() - ae_error.min()), 0, 1)

    # ── Ensemble score ────────────────────────────────────────────────────────
    W = ENSEMBLE_WEIGHTS
    ensemble = (
        W["IF"]  * if_anomaly +
        W["LOF"] * lof_anomaly +
        W["Z"]   * z_anomaly +
        W["AE"]  * ae_anomaly
    )

    # ── Risk labels (data-driven thresholds) ──────────────────────────────────
    alta_th = max(float(np.percentile(ensemble, 97)), 0.30)
    media_th = max(float(np.percentile(ensemble, 90)), 0.20)

    def classify(score: float) -> str:
        if score >= alta_th:
            return "ALTA"
        if score >= media_th:
            return "MEDIA"
        return "NORMALE"

    labels = np.array([classify(s) for s in ensemble])

    # ── Build results DataFrame ───────────────────────────────────────────────
    results = features_wb[
        ["ROTTA", "PAESE_PART", "ZONA", "score_composito",
         "n_anomalie", "tot_allarmi_log", "pct_interpol",
         "score_rischio_esiti"]
    ].copy()
    results["anomaly_score_if"] = np.round(if_anomaly, 4)
    results["anomaly_score_lof"] = np.round(lof_anomaly, 4)
    results["anomaly_score_z"] = np.round(z_anomaly, 4)
    results["anomaly_score_ae"] = np.round(ae_anomaly, 4)
    results["anomaly_score"] = np.round(ensemble, 4)
    results["anomaly_label"] = labels
    results["n_models_flagged"] = (
        (if_anomaly >= 0.5).astype(int) +
        (lof_anomaly >= 0.5).astype(int) +
        (z_anomaly >= 0.27).astype(int) +
        (ae_anomaly >= 0.5).astype(int)
    )
    results = results.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    # ── Save ──────────────────────────────────────────────────────────────────
    results.to_csv(PROC_DIR / "anomaly_results.csv", index=False)

    summary = {
        "n_routes": int(len(results)),
        "n_alta": int((labels == "ALTA").sum()),
        "n_media": int((labels == "MEDIA").sum()),
        "n_normale": int((labels == "NORMALE").sum()),
        "alta_threshold": round(alta_th, 4),
        "media_threshold": round(media_th, 4),
        "threshold_method": "data-driven (p97/p90)",
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "top10_routes": [
            {"rank": int(r["rank"]), "rotta": r["ROTTA"],
             "label": r["anomaly_label"],
             "score": float(r["anomaly_score"]),
             "paese": r["PAESE_PART"]}
            for _, r in results.head(10).iterrows()
        ],
    }
    with open(PROC_DIR / "anomaly_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n_alta = int((labels == "ALTA").sum())
    n_media = int((labels == "MEDIA").sum())
    logger.info(
        "  -> %d ALTA, %d MEDIA, %d NORMALE -> anomaly_results.csv",
        n_alta, n_media, int((labels == "NORMALE").sum()),
    )

    # Store models and scaled data for evaluation step
    results.attrs["_iso_forest"] = iso_forest
    results.attrs["_X_scaled"] = X_scaled
    results.attrs["_scaler"] = scaler
    results.attrs["_alta_th"] = alta_th
    results.attrs["_media_th"] = media_th

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Post-Processing (business rules + risk profiles)
# ══════════════════════════════════════════════════════════════════════════════

def step_post_processing(
    results: pd.DataFrame,
    features_wb: pd.DataFrame,
) -> pd.DataFrame:
    """Apply business rules, compute confidence scores, generate risk profiles."""
    logger.info("Step 4: Post-Processing (business rules)")

    df = results.merge(
        features_wb[["ROTTA", "tasso_chiusura", "tasso_rilevanza",
                      "tasso_allarme_medio", "tasso_inv_medio",
                      "tasso_respinti", "tasso_fermati",
                      "pct_sdi", "pct_nsis"]],
        on="ROTTA", how="left", suffixes=("", "_feat"),
    )

    # ── Business Rules ────────────────────────────────────────────────────────
    df["br_high_interpol"] = (df["pct_interpol"] >= 0.30).astype(int)
    df["br_high_rejection"] = (df["tasso_respinti"] >= 0.25).astype(int)
    df["br_low_closure"] = (
        (df["tot_allarmi_log"] > 3) &
        (df["tasso_chiusura"] < 0.10)
    ).astype(int)
    df["br_multi_source"] = (
        (df["pct_interpol"] > 0) &
        (df["pct_sdi"] > 0)
    ).astype(int)
    df["br_high_alarm_rate"] = (df["tasso_allarme_medio"] >= 0.50).astype(int)

    df["br_score"] = (
        df["br_high_interpol"] +
        df["br_high_rejection"] +
        df["br_low_closure"] +
        df["br_multi_source"] +
        df["br_high_alarm_rate"]
    ) / 5.0

    # ── Confidence score (60% ML + 40% business rules) ────────────────────────
    df["confidence"] = (0.60 * df["anomaly_score"] + 0.40 * df["br_score"]).round(4)

    # ── Final risk classification ─────────────────────────────────────────────
    def final_risk(row):
        ml = row["anomaly_label"]
        br = row["br_score"]
        if ml == "ALTA" and br >= 0.4:
            return "CRITICO"
        if ml == "ALTA" or (ml == "MEDIA" and br >= 0.4):
            return "ALTO"
        if ml == "MEDIA":
            return "MEDIO"
        return "BASSO"

    df["risk_level"] = df.apply(final_risk, axis=1)
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    # ── Risk profiles (for non-BASSO routes) ──────────────────────────────────
    alert_routes = df[df["risk_level"] != "BASSO"].copy()
    profiles = []
    for _, row in alert_routes.iterrows():
        drivers = []
        if row["br_high_interpol"]:
            drivers.append("High INTERPOL alarm rate")
        if row["br_high_rejection"]:
            drivers.append("High rejection rate")
        if row["br_low_closure"]:
            drivers.append("Low alarm closure rate")
        if row["br_multi_source"]:
            drivers.append("Multi-source alarms (INTERPOL + SDI)")
        if row["br_high_alarm_rate"]:
            drivers.append("High average alarm rate")

        profiles.append({
            "rotta": row["ROTTA"],
            "paese": row["PAESE_PART"],
            "zona": row.get("ZONA"),
            "risk_level": row["risk_level"],
            "anomaly_score": float(row["anomaly_score"]),
            "confidence": float(row["confidence"]),
            "br_score": float(row["br_score"]),
            "n_models_flagged": int(row["n_models_flagged"]),
            "risk_drivers": drivers,
        })

    with open(PROC_DIR / "risk_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    df.to_csv(PROC_DIR / "final_report.csv", index=False)

    for lvl in ["CRITICO", "ALTO", "MEDIO", "BASSO"]:
        n = int((df["risk_level"] == lvl).sum())
        if n:
            logger.info("  -> %s: %d routes", lvl, n)

    logger.info("  -> final_report.csv, risk_profiles.json")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def step_evaluation(
    results: pd.DataFrame,
    features_wb: pd.DataFrame,
) -> dict:
    """Evaluate pipeline: silhouette, bootstrap stability, feature importance, SHAP."""
    logger.info("Step 5: Evaluation")

    X_raw = features_wb[BASELINE_FEATURES].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    labels_binary = (results["anomaly_label"] != "NORMALE").astype(int).values

    # ── 5a. Silhouette score ──────────────────────────────────────────────────
    try:
        sil = float(silhouette_score(X_scaled, labels_binary))
    except Exception:
        sil = None
    logger.info("  Silhouette score: %s", f"{sil:.4f}" if sil else "N/A")

    # ── 5b. Bootstrap stability (100 iterations) ─────────────────────────────
    n_boot = 100
    boot_labels = np.zeros((n_boot, len(X_scaled)), dtype=int)
    rng = np.random.RandomState(42)

    for i in range(n_boot):
        idx = rng.choice(len(X_scaled), size=int(0.80 * len(X_scaled)), replace=False)
        X_boot = X_scaled[idx]
        iso = IsolationForest(
            n_estimators=200, contamination=0.10,
            random_state=42, n_jobs=-1,
        )
        iso.fit(X_boot)
        preds = iso.predict(X_scaled)
        boot_labels[i] = (preds == -1).astype(int)

    freq = boot_labels.mean(axis=0)  # how often each route is flagged
    anomaly_mask = labels_binary == 1
    stable_anomalies = int((freq[anomaly_mask] >= 0.70).sum())
    total_anomalies = int(anomaly_mask.sum())
    logger.info(
        "  Bootstrap stability: %d/%d anomalies stable (>=70%%)",
        stable_anomalies, total_anomalies,
    )

    stability_df = pd.DataFrame({
        "ROTTA": results["ROTTA"],
        "anomaly_label": results["anomaly_label"],
        "bootstrap_frequency": np.round(freq, 4),
        "is_stable": (freq >= 0.70).astype(int),
    })
    stability_df.to_csv(PROC_DIR / "stability_scores.csv", index=False)

    # ── 5c. Feature importance (permutation on IsolationForest) ───────────────
    iso_forest = IsolationForest(
        n_estimators=200, contamination=0.10,
        random_state=42, n_jobs=-1,
    )
    iso_forest.fit(X_scaled)
    base_score = iso_forest.score_samples(X_scaled).mean()

    importances = {}
    for j, feat in enumerate(BASELINE_FEATURES):
        X_perm = X_scaled.copy()
        rng.shuffle(X_perm[:, j])
        perm_score = iso_forest.score_samples(X_perm).mean()
        importances[feat] = round(float(base_score - perm_score), 6)

    imp_df = pd.DataFrame(
        sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True),
        columns=["feature", "importance"],
    )
    imp_df.to_csv(PROC_DIR / "feature_importance.csv", index=False)
    logger.info("  Top 3 features: %s", imp_df.head(3).to_dict("records"))

    # ── 5d. SHAP via surrogate GradientBoosting ──────────────────────────────
    try:
        surrogate = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42,
        )
        surrogate.fit(X_scaled, labels_binary)
        shap_imp = dict(zip(BASELINE_FEATURES, surrogate.feature_importances_))
        shap_df = pd.DataFrame(
            sorted(shap_imp.items(), key=lambda x: x[1], reverse=True),
            columns=["feature", "shap_importance"],
        )
        shap_df.to_csv(PROC_DIR / "shap_importance.csv", index=False)
    except Exception as e:
        logger.warning("  SHAP surrogate failed: %s", e)
        shap_df = pd.DataFrame(columns=["feature", "shap_importance"])

    # ── Scorecard ─────────────────────────────────────────────────────────────
    scorecard = {
        "silhouette_score": sil,
        "bootstrap": {
            "n_iterations": n_boot,
            "sample_fraction": 0.80,
            "stable_anomalies": stable_anomalies,
            "total_anomalies": total_anomalies,
            "stability_rate": round(stable_anomalies / max(total_anomalies, 1), 4),
        },
        "feature_importance": importances,
        "shap_importance": dict(zip(shap_df["feature"], shap_df["shap_importance"]))
            if not shap_df.empty else {},
    }
    with open(PROC_DIR / "evaluation_scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2)

    logger.info("  -> evaluation_scorecard.json")
    return scorecard


# ══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_classical_pipeline(*, skip_eval: bool = False, verbose: bool = False) -> dict:
    """
    Execute the entire classical pipeline end-to-end.

    Args:
        skip_eval: If True, skip the evaluation step (faster).
        verbose:   If True, enable DEBUG-level logging.

    Returns:
        Summary dict with timing and results for each step.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"steps": {}, "errors": {}}
    t_total = time.perf_counter()

    # ── Step 0: Preprocessing ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  CLASSICAL PIPELINE — START")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    try:
        df_allarmi, df_viaggiatori, df_merged = run_preprocessing()
        summary["steps"]["preprocessing"] = {
            "ok": True, "elapsed_s": round(time.perf_counter() - t0, 3),
            "allarmi_rows": len(df_allarmi),
            "viaggiatori_rows": len(df_viaggiatori),
            "merged_rows": len(df_merged),
        }
    except Exception as e:
        summary["steps"]["preprocessing"] = {"ok": False, "error": str(e)}
        summary["errors"]["preprocessing"] = str(e)
        logger.error("Preprocessing failed: %s", e)
        return summary

    # ── Step 1: Feature Engineering ───────────────────────────────────────────
    t1 = time.perf_counter()
    try:
        features = step_feature_engineering(df_allarmi, df_viaggiatori)
        summary["steps"]["feature_engineering"] = {
            "ok": True, "elapsed_s": round(time.perf_counter() - t1, 3),
            "n_routes": len(features),
            "n_features": len(features.select_dtypes(include="number").columns),
        }
    except Exception as e:
        summary["steps"]["feature_engineering"] = {"ok": False, "error": str(e)}
        summary["errors"]["feature_engineering"] = str(e)
        logger.error("Feature engineering failed: %s", e)
        return summary

    # ── Step 2: Baseline Construction ─────────────────────────────────────────
    t2 = time.perf_counter()
    try:
        features_wb = step_baseline_construction(features)
        summary["steps"]["baseline_construction"] = {
            "ok": True, "elapsed_s": round(time.perf_counter() - t2, 3),
        }
    except Exception as e:
        summary["steps"]["baseline_construction"] = {"ok": False, "error": str(e)}
        summary["errors"]["baseline_construction"] = str(e)
        logger.error("Baseline construction failed: %s", e)
        return summary

    # ── Step 3: Anomaly Detection ─────────────────────────────────────────────
    t3 = time.perf_counter()
    try:
        results = step_anomaly_detection(features_wb)
        summary["steps"]["anomaly_detection"] = {
            "ok": True, "elapsed_s": round(time.perf_counter() - t3, 3),
            "n_alta": int((results["anomaly_label"] == "ALTA").sum()),
            "n_media": int((results["anomaly_label"] == "MEDIA").sum()),
        }
    except Exception as e:
        summary["steps"]["anomaly_detection"] = {"ok": False, "error": str(e)}
        summary["errors"]["anomaly_detection"] = str(e)
        logger.error("Anomaly detection failed: %s", e)
        return summary

    # ── Step 4: Post-Processing ───────────────────────────────────────────────
    t4 = time.perf_counter()
    try:
        final = step_post_processing(results, features_wb)
        summary["steps"]["post_processing"] = {
            "ok": True, "elapsed_s": round(time.perf_counter() - t4, 3),
        }
    except Exception as e:
        summary["steps"]["post_processing"] = {"ok": False, "error": str(e)}
        summary["errors"]["post_processing"] = str(e)
        logger.error("Post-processing failed: %s", e)

    # ── Step 5: Evaluation (optional) ─────────────────────────────────────────
    if not skip_eval:
        t5 = time.perf_counter()
        try:
            scorecard = step_evaluation(results, features_wb)
            summary["steps"]["evaluation"] = {
                "ok": True, "elapsed_s": round(time.perf_counter() - t5, 3),
                "silhouette": scorecard.get("silhouette_score"),
                "stability_rate": scorecard.get("bootstrap", {}).get("stability_rate"),
            }
        except Exception as e:
            summary["steps"]["evaluation"] = {"ok": False, "error": str(e)}
            summary["errors"]["evaluation"] = str(e)
            logger.error("Evaluation failed: %s", e)
    else:
        logger.info("Step 5: Evaluation — SKIPPED (--skip-eval)")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = round(time.perf_counter() - t_total, 3)
    summary["total_runtime_s"] = total_elapsed
    summary["completed_steps"] = [
        k for k, v in summary["steps"].items() if v.get("ok")
    ]
    summary["failed_steps"] = [
        k for k, v in summary["steps"].items() if not v.get("ok")
    ]

    logger.info("=" * 60)
    logger.info("  CLASSICAL PIPELINE — COMPLETE")
    logger.info("  Total runtime: %.1fs", total_elapsed)
    logger.info("  Completed: %s", summary["completed_steps"])
    if summary["failed_steps"]:
        logger.warning("  Failed: %s", summary["failed_steps"])
    logger.info("=" * 60)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full classical anomaly-detection pipeline.",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the evaluation step (faster execution).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    args = parser.parse_args()

    summary = run_classical_pipeline(
        skip_eval=args.skip_eval,
        verbose=args.verbose,
    )

    print("\n" + json.dumps(summary, indent=2))
