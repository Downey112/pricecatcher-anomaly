"""Conformal p-values for anomaly scores.

Idea: keep a sample of scores from weeks we treat as "normal" (the training weeks). For a new
score s, the conformal p-value is

    p = (1 + number of calibration scores >= s) / (n + 1)

If the new row behaves like the calibration rows (exchangeable), P(p <= alpha) <= alpha, so
"flag when p <= alpha" flags a normal price at most alpha of the time. That turns the MAD score,
which has no natural cut-off, into a decision with a stated false-alarm rate.

Caveats (see EXPLAIN.md): the training weeks contain some real anomalies, which makes the
p-values slightly conservative; prices drift over the year, which can break exchangeability.
"""
from __future__ import annotations

import numpy as np


def calibration_sample(scores: np.ndarray, n: int = 200_000, seed: int = 42) -> np.ndarray:
    """A sorted random sample of calibration scores (sorted so p-values are a binary search)."""
    scores = np.asarray(scores, dtype="float64")
    if len(scores) > n:
        scores = np.random.default_rng(seed).choice(scores, size=n, replace=False)
    return np.sort(scores)


def p_values(scores: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Conformal p-value of each score against a sorted calibration sample."""
    calibration = np.asarray(calibration, dtype="float64")
    n_at_least = len(calibration) - np.searchsorted(calibration, np.asarray(scores, dtype="float64"), side="left")
    return (1.0 + n_at_least) / (len(calibration) + 1.0)


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of discoveries controlling the false discovery rate at q.

    Useful when thousands of prices are tested at once: alpha controls false alarms per price,
    BH controls the share of flags that are false.
    """
    p = np.asarray(p, dtype="float64")
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * np.arange(1, len(p) + 1) / len(p)
    below = np.nonzero(ranked <= thresholds)[0]
    mask = np.zeros(len(p), dtype=bool)
    if len(below):
        mask[order[: below.max() + 1]] = True
    return mask
