"""Ensemble ablation study.

Drops one detector at a time, retrains the remaining ensemble with the
weights proportionally renormalised, and measures how the verdict changes.
For each subset we report:

* ``n_high`` / ``n_medium`` / ``n_normal``
* ``top17_overlap``  — share of the original top-17 HIGH set that the
                       reduced ensemble still classifies as HIGH
* ``br_rank_corr``   — Spearman correlation between the ensemble score
                       and the business-rule score (proxy for how
                       operationally aligned the ensemble is)
* ``stability_top17``— bootstrap stability of the top-17 HIGH set across
                       ``n_boot`` resamples at 80 % subsample

The full ensemble (IF + LOF + Z + AE) is also evaluated as the baseline.

Usage::

    df = run_ablation(df_anomalies, n_boot=200)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DETECTORS = ("IF", "LOF", "Z", "AE")
SCORE_COL = {
    "IF":  "score_if",
    "LOF": "score_lof",
    "Z":   "score_z",
    "AE":  "score_ae",
}
DEFAULT_WEIGHTS = {"IF": 0.40, "LOF": 0.15, "Z": 0.30, "AE": 0.15}
TOP_K = 17


def _renormalise(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def _ensemble_score(df: pd.DataFrame, subset: tuple[str, ...]) -> pd.Series:
    """Compute the ensemble score on ``subset`` with renormalised weights."""
    weights = _renormalise({k: DEFAULT_WEIGHTS[k] for k in subset})
    out = pd.Series(0.0, index=df.index)
    for det in subset:
        out = out + weights[det] * df[SCORE_COL[det]].fillna(0)
    return out


def _bootstrap_stability(
    df: pd.DataFrame,
    subset: tuple[str, ...],
    *,
    n_boot: int,
    sample_frac: float,
    seed: int,
) -> float:
    """How stable the top-K HIGH set is when we resample 80 % of the routes."""
    rng = np.random.RandomState(seed)
    base = _ensemble_score(df, subset)
    base_top = set(base.nlargest(TOP_K).index)
    n_sub = int(sample_frac * len(df))
    overlaps = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.choice(len(df), size=n_sub, replace=False)
        scored = _ensemble_score(df.iloc[idx], subset)
        boot_top = set(scored.nlargest(TOP_K).index)
        overlaps[k] = len(boot_top & base_top) / TOP_K
    return float(overlaps.mean())


def run_ablation(
    df: pd.DataFrame,
    *,
    n_boot: int = 200,
    sample_frac: float = 0.80,
    seed: int = 42,
    include_singletons: bool = True,
) -> pd.DataFrame:
    """Iterates over all interesting detector subsets and reports the metrics."""
    # Subsets to evaluate
    subsets: list[tuple[str, ...]] = []
    if include_singletons:
        subsets.extend([(d,) for d in DETECTORS])
    # All triplets (drop-one studies)
    subsets.extend(combinations(DETECTORS, 3))
    # The full quartet
    subsets.append(DETECTORS)
    # De-duplicate while preserving order
    seen, deduped = set(), []
    for s in subsets:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    subsets = deduped

    # Reference top-17 HIGH set (full ensemble) for the overlap metric
    full_score = _ensemble_score(df, DETECTORS)
    full_top   = set(full_score.nlargest(TOP_K).index)

    br_score = df.get("br_score", pd.Series(0.0, index=df.index))

    rows = []
    for subset in subsets:
        score = _ensemble_score(df, subset)
        threshold_high = float(score.quantile(0.97))
        threshold_med  = float(score.quantile(0.90))
        n_high   = int((score >= threshold_high).sum())
        n_medium = int(((score >= threshold_med) & (score < threshold_high)).sum())
        n_normal = int((score < threshold_med).sum())
        top17    = set(score.nlargest(TOP_K).index)
        overlap  = len(top17 & full_top) / TOP_K
        rho, _   = spearmanr(score, br_score)
        stability = _bootstrap_stability(
            df, subset, n_boot=n_boot, sample_frac=sample_frac, seed=seed,
        )
        rows.append({
            "subset":         " + ".join(subset),
            "n_detectors":    len(subset),
            "n_high":         n_high,
            "n_medium":       n_medium,
            "n_normal":       n_normal,
            "top17_overlap":  overlap,
            "br_rank_corr":   float(rho),
            "stability_top17": stability,
        })

    return pd.DataFrame(rows).sort_values(
        ["n_detectors", "subset"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    df = pd.read_csv(here / "data" / "processed" / "anomaly_results_live.csv")
    out = run_ablation(df, n_boot=200)
    print(out.to_string(index=False))
