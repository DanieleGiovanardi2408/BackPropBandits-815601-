"""Shared Autoencoder training utility.

Single source of truth for the AE component of the ensemble. The classical
pipeline (`classical_pipeline.main.step_anomaly_detection`) and the
multi-agent pipeline (`OutlierAgent`) both call ``train_and_score`` so the
AE reconstruction-error scores are *identical by construction* between the
two pipelines — eliminating the ~1.8 % residual disagreement that the
stochastic AE used to introduce on borderline MEDIUM/NORMAL routes.

Determinism guarantees:
    1. Rows of the input matrix are ALWAYS sorted by the route id before
       fitting, so different upstream sort orders cannot produce different
       train/validation splits.
    2. Early stopping is OFF: the AE trains for a fixed number of epochs
       (``max_iter``) so the convergence point is a deterministic function
       of (data, random_state) alone, not of any internal split heuristic.
    3. ``random_state=42`` is fixed and exposed.

If the input has fewer than ``min_samples`` "normal" rows (as identified by
an upstream IsolationForest mask), the AE is excluded and zero-scores are
returned along with ``use_ae=False`` — both pipelines then redistribute the
AE weight proportionally over the remaining detectors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


logger = logging.getLogger(__name__)


@dataclass
class AutoencoderResult:
    """Output of ``train_and_score``."""

    score_ae: np.ndarray            # length == n_rows of the (sorted) input
    use_ae:   bool
    n_normal_used: int
    n_total:  int
    sorted_index: pd.Index | None   # to allow callers to align the output


# ── Default hyper-parameters (identical to the original pipelines) ──────────
DEFAULT_HIDDEN_LAYERS = (8, 4, 8)
DEFAULT_MAX_ITER      = 1000
DEFAULT_RANDOM_STATE  = 42
DEFAULT_MIN_SAMPLES   = 30


def train_and_score(
    X_scaled: np.ndarray | pd.DataFrame,
    *,
    normal_mask: np.ndarray,
    row_ids: np.ndarray | pd.Series | None = None,
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> AutoencoderResult:
    """Train an MLP autoencoder on the normal subset and score every row.

    Parameters
    ----------
    X_scaled : (n_rows, n_features) feature matrix already standard-scaled.
    normal_mask : (n_rows,) boolean — True for routes treated as normal
        (typically produced by an upstream IsolationForest at the same
        contamination as the pipeline).
    row_ids : (n_rows,) array of route ids (e.g. the ``ROTTA`` column).
        Used to deterministically sort the input before fitting; pass None
        if the caller is already responsible for sort stability.

    Returns
    -------
    AutoencoderResult with ``score_ae`` aligned to the *input* row order
    (the sort is undone before returning).
    """
    X_arr = np.asarray(X_scaled, dtype=float)
    n_total = X_arr.shape[0]

    # Deterministic ordering — both pipelines must see the same row sequence.
    if row_ids is not None:
        ids = np.asarray(row_ids)
        order = np.argsort(ids, kind="stable")
        inverse = np.argsort(order, kind="stable")
        X_sorted = X_arr[order]
        mask_sorted = np.asarray(normal_mask, dtype=bool)[order]
    else:
        order = np.arange(n_total)
        inverse = order
        X_sorted = X_arr
        mask_sorted = np.asarray(normal_mask, dtype=bool)

    X_normal = X_sorted[mask_sorted]
    n_normal = int(mask_sorted.sum())

    if n_normal < min_samples:
        logger.warning(
            "Autoencoder excluded: only %d normal routes available "
            "(min_samples=%d). Returning zero-scored AE component.",
            n_normal, min_samples,
        )
        return AutoencoderResult(
            score_ae=np.zeros(n_total),
            use_ae=False,
            n_normal_used=n_normal,
            n_total=n_total,
            sorted_index=None,
        )

    # ── Fit the AE — deterministic configuration ─────────────────────────────
    # ``early_stopping=False`` (the historical default was True) so the
    # convergence point is fully determined by (data, random_state).
    # The historical AE used early stopping which produced run-to-run
    # variability on borderline routes; removing it eliminates the
    # stochastic divergence between the two pipelines.
    ae = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=False,
        shuffle=False,
        verbose=False,
    )
    ae.fit(X_normal, X_normal)

    X_reconstructed = ae.predict(X_sorted)
    ae_error_sorted = np.mean((X_sorted - X_reconstructed) ** 2, axis=1)

    # Min-max normalise to [0, 1] consistently with the other detectors.
    err_min, err_max = float(ae_error_sorted.min()), float(ae_error_sorted.max())
    if err_max - err_min > 1e-12:
        score_sorted = (ae_error_sorted - err_min) / (err_max - err_min)
    else:
        score_sorted = np.zeros_like(ae_error_sorted)

    # Undo the sort so the caller gets scores in the *input* row order.
    score_ae = score_sorted[inverse]

    logger.info(
        "Autoencoder ✓ Trained on %d normal routes / %d total — score_ae range [%.4f, %.4f]",
        n_normal, n_total, float(score_ae.min()), float(score_ae.max()),
    )

    return AutoencoderResult(
        score_ae=score_ae,
        use_ae=True,
        n_normal_used=n_normal,
        n_total=n_total,
        sorted_index=None,
    )
