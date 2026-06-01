"""Ensemble weight grid search.

Samples the 4-simplex on a step-``grid_step`` grid (weights summing to 1)
and scores every weight vector by a combination of:

* ``stability``     — bootstrap stability of the top-17 HIGH set over
                      ``n_boot`` resamples at 80 % subsample, scaled to
                      [0, 1].
* ``br_rank_corr``  — Spearman correlation between the ensemble score and
                      the per-route business-rule score, mapped to [0, 1]
                      via ``(rho + 1) / 2``.

The default objective is::

    objective = 0.5 * stability + 0.5 * br_rank_corr_01

so a weight vector is rewarded both for producing a self-consistent
top-set under resampling AND for ranking the routes the way the
canonical business rules would. Both halves are needed: stability alone
favours degenerate detectors that always agree with themselves; rule
correlation alone favours mirroring the rules at the expense of the ML
signal.

Usage::

    df_grid = run_grid_search(df_anomalies, grid_step=0.05, n_boot=200)
    best    = df_grid.iloc[0]
"""
from __future__ import annotations

from itertools import product

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


def _simplex_grid(step: float) -> list[tuple[float, float, float, float]]:
    """Enumerates weight vectors ``(w_if, w_lof, w_z, w_ae)`` with each
    component on a multiple of ``step`` and summing to 1 (modulo float)."""
    n = int(round(1.0 / step))
    pts = []
    for a in range(0, n + 1):
        for b in range(0, n + 1 - a):
            for c in range(0, n + 1 - a - b):
                d = n - a - b - c
                if d < 0:
                    continue
                pts.append((a * step, b * step, c * step, d * step))
    # Filter out points that exclude a detector entirely — the production
    # ensemble must use all four (we already cover drop-one studies in the
    # ablation module).
    return [p for p in pts if all(w > 0 for w in p)]


def _ensemble_score(df: pd.DataFrame, w: tuple[float, ...]) -> pd.Series:
    return (
        w[0] * df[SCORE_COL["IF"]].fillna(0)
        + w[1] * df[SCORE_COL["LOF"]].fillna(0)
        + w[2] * df[SCORE_COL["Z"]].fillna(0)
        + w[3] * df[SCORE_COL["AE"]].fillna(0)
    )


def _bootstrap_stability(
    df: pd.DataFrame,
    weights: tuple[float, ...],
    *,
    n_boot: int,
    sample_frac: float,
    rng: np.random.RandomState,
) -> float:
    base = _ensemble_score(df, weights)
    base_top = set(base.nlargest(TOP_K).index)
    n_sub = int(sample_frac * len(df))
    overlaps = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.choice(len(df), size=n_sub, replace=False)
        scored = _ensemble_score(df.iloc[idx], weights)
        boot_top = set(scored.nlargest(TOP_K).index)
        overlaps[k] = len(boot_top & base_top) / TOP_K
    return float(overlaps.mean())


def run_grid_search(
    df: pd.DataFrame,
    *,
    grid_step: float = 0.05,
    n_boot: int = 200,
    sample_frac: float = 0.80,
    stability_weight: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluates every weight vector on the grid and returns a ranked DataFrame."""
    weights_grid = _simplex_grid(grid_step)
    rng = np.random.RandomState(seed)
    br_score = df.get("br_score", pd.Series(0.0, index=df.index))

    rows = []
    for w in weights_grid:
        score = _ensemble_score(df, w)
        rho, _ = spearmanr(score, br_score)
        rho = float(rho) if np.isfinite(rho) else 0.0
        # Map [-1, 1] -> [0, 1] for easy averaging
        rho_01 = (rho + 1.0) / 2.0
        stab = _bootstrap_stability(
            df, w, n_boot=n_boot, sample_frac=sample_frac, rng=rng,
        )
        objective = (
            stability_weight * stab
            + (1.0 - stability_weight) * rho_01
        )
        rows.append({
            "w_if":          w[0],
            "w_lof":         w[1],
            "w_z":           w[2],
            "w_ae":          w[3],
            "stability":     stab,
            "br_rank_corr":  rho,
            "objective":     objective,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("objective", ascending=False)
        .reset_index(drop=True)
    )


def summarise(df_grid: pd.DataFrame) -> dict:
    """Convenience: best weight vector, current production weights, and gap."""
    if df_grid.empty:
        return {}
    best = df_grid.iloc[0]
    current_mask = (
        (df_grid["w_if"]  == DEFAULT_WEIGHTS["IF"]) &
        (df_grid["w_lof"] == DEFAULT_WEIGHTS["LOF"]) &
        (df_grid["w_z"]   == DEFAULT_WEIGHTS["Z"]) &
        (df_grid["w_ae"]  == DEFAULT_WEIGHTS["AE"])
    )
    current_row = df_grid[current_mask]
    if not current_row.empty:
        current = current_row.iloc[0]
        gap = float(best["objective"]) - float(current["objective"])
    else:
        current = None
        gap = None
    return {
        "best": {
            "w_if":         float(best["w_if"]),
            "w_lof":        float(best["w_lof"]),
            "w_z":          float(best["w_z"]),
            "w_ae":         float(best["w_ae"]),
            "stability":    float(best["stability"]),
            "br_rank_corr": float(best["br_rank_corr"]),
            "objective":    float(best["objective"]),
        },
        "current_production": (
            None if current is None else {
                "w_if":         float(current["w_if"]),
                "w_lof":        float(current["w_lof"]),
                "w_z":          float(current["w_z"]),
                "w_ae":         float(current["w_ae"]),
                "stability":    float(current["stability"]),
                "br_rank_corr": float(current["br_rank_corr"]),
                "objective":    float(current["objective"]),
            }
        ),
        "best_minus_current": gap,
        "n_weight_vectors":   int(len(df_grid)),
    }


if __name__ == "__main__":
    from pathlib import Path
    here = Path(__file__).resolve().parents[2]
    df = pd.read_csv(here / "data" / "processed" / "risk_profiles_live.csv")
    grid = run_grid_search(df, grid_step=0.05, n_boot=200)
    print("Top 10 weight vectors:")
    print(grid.head(10).to_string(index=False))
    print()
    print("Summary:")
    import json
    print(json.dumps(summarise(grid), indent=2))
