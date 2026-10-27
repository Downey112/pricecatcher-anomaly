"""Anomaly-detection scorers: MADBaseline, IForestScorer, ECODScorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from pcad.config import FEATURES, SEED


def _mask_high(raw: np.ndarray, offset: float, df: pd.DataFrame) -> np.ndarray:
    """Shift by the fit-time minimum (floored at a tiny positive value) and
    zero out prices at or below the reference, so only high prices can rank."""
    shifted = np.maximum(raw - offset, 0.0) + 1e-6
    high = df["log_ratio"].to_numpy(dtype="float32") > 0
    return (shifted * high).astype("float32")


class MADBaseline:
    """Trivial baseline: score = max(robust_z, 0).

    ``fit`` is a no-op; no training state is required.
    """

    def fit(self, df: pd.DataFrame) -> "MADBaseline":
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return max(robust_z, 0) as a float32 array aligned to df.index."""
        return np.maximum(df["robust_z"].to_numpy(dtype="float32"), 0.0)


class IForestScorer:
    """Isolation Forest scorer.

    Trained on up to 500 000 random rows of FEATURES (fillna 0).
    Raw score is ``-IsolationForest.score_samples``.
    Multiplied by (log_ratio > 0) so cheap anomalies score 0.
    Offset fixed at fit time (see _mask_high), so scores are comparable
    across batches.
    """

    def __init__(self) -> None:
        self._model = IsolationForest(
            n_estimators=200,
            max_samples=256,
            random_state=SEED,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "IForestScorer":
        rng = np.random.default_rng(SEED)
        X = df[FEATURES].fillna(0).to_numpy(dtype="float32")
        if len(X) > 500_000:
            idx = rng.choice(len(X), 500_000, replace=False)
            X = X[idx]
        self._model.fit(X)
        # offset fixed at fit time so scores are comparable across chunks
        self._offset = float((-self._model.score_samples(X)).min())
        self._fitted = True
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].fillna(0).to_numpy(dtype="float32")
        raw = -self._model.score_samples(X).astype("float64")
        return _mask_high(raw, self._offset, df)


class ECODScorer:
    """ECOD (Li et al., 2022) scored against the TRAINING distribution only.

    For each feature, left- and right-tail probabilities come from the
    training empirical CDF; the outlier score is the larger of the summed
    -log left tails, -log right tails, and the skewness-chosen tail per
    feature. pyod's ECOD re-fits the CDFs on train + the batch being scored,
    so a row's score depends on its batch; this version does not.
    Multiplied by (log_ratio > 0) so cheap anomalies score 0.
    """

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "ECODScorer":
        rng = np.random.default_rng(SEED)
        X = df[FEATURES].fillna(0).to_numpy(dtype="float64")
        if len(X) > 500_000:
            X = X[rng.choice(len(X), 500_000, replace=False)]
        self._sorted = np.sort(X, axis=0)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        self._right_skew = (((X - mu) ** 3).mean(axis=0) / np.maximum(sd, 1e-12) ** 3) > 0
        self._fitted = True
        self._offset = float(self._raw(X).min())
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        n = len(self._sorted)
        left = np.empty_like(X)
        right = np.empty_like(X)
        for j in range(X.shape[1]):
            col = self._sorted[:, j]
            le = np.searchsorted(col, X[:, j], side="right")      # train <= x
            ge = n - np.searchsorted(col, X[:, j], side="left")   # train >= x
            left[:, j] = -np.log((le + 1) / (n + 1))
            right[:, j] = -np.log((ge + 1) / (n + 1))
        auto = np.where(self._right_skew, right, left)
        return np.maximum.reduce([left.sum(1), right.sum(1), auto.sum(1)])

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].fillna(0).to_numpy(dtype="float64")
        return _mask_high(self._raw(X), self._offset, df)
