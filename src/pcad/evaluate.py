"""Evaluation utilities: top-k flagging, budget metrics, bootstrap recall."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pcad.config import BUDGET_K, SEED


def topk_flags(
    df: pd.DataFrame,
    score_col: str,
    k: int,
    group: tuple[str, ...] = ("district", "week"),
) -> pd.Series:
    """Return a boolean Series aligned to ``df.index`` marking the top-k rows
    by score within each group.

    Tie-breaking is stable (row order within group), so each group gets at
    most k flags even when scores tie at the boundary.

    Implementation: ``rank(method='first', ascending=False) <= k`` within group.
    """
    ranks = (
        df.groupby(list(group), sort=False)[score_col]
        .rank(method="first", ascending=False)
    )
    # a zero score means "not above the reference": never spend budget on it
    return ((df[score_col] > 0) & (ranks <= k)).rename(None)


def budget_metrics(
    df: pd.DataFrame,
    score_col: str,
    ks: tuple[int, ...] = (5, 20, 50),
) -> pd.DataFrame:
    """Compute recall, precision@k, and average precision for each (k, inj_type).

    Only rows with ``is_injected=True`` contribute to recall numerators.
    ``precision_at_k`` and ``average_precision`` are *overall* metrics
    (not per inj_type) and are repeated on every row within the same k.

    Parameters
    ----------
    df:
        Must have columns ``is_injected`` (bool), ``inj_type`` (str), and
        ``score_col``.
    score_col:
        Column containing the scorer's output.
    ks:
        Inspection budget sizes to evaluate.

    Returns
    -------
    DataFrame with columns [k, inj_type, recall, precision_at_k, average_precision].
    """
    rows = []
    inj_types = [t for t in df.loc[df["is_injected"], "inj_type"].unique()
                 if t != "none"]

    for k in ks:
        flags = topk_flags(df, score_col, k)

        # Overall metrics at this k
        n_flagged_injected = int((flags & df["is_injected"]).sum())
        n_flagged = int(flags.sum())
        p_at_k = n_flagged_injected / n_flagged if n_flagged > 0 else 0.0

        # Average precision on the full df using score as ranking signal
        try:
            ap = average_precision_score(
                df["is_injected"].astype(int),
                df[score_col].fillna(0),
            )
        except ValueError:
            ap = float("nan")

        for inj_t in inj_types:
            mask_type = df["inj_type"] == inj_t
            n_type = int(mask_type.sum())
            n_type_flagged = int((flags & mask_type).sum())
            recall = n_type_flagged / n_type if n_type > 0 else 0.0
            rows.append({
                "k": k,
                "inj_type": inj_t,
                "recall": round(recall, 6),
                "precision_at_k": round(p_at_k, 6),
                "average_precision": round(ap, 6),
            })

    return pd.DataFrame(rows)


def bootstrap_recall(
    df: pd.DataFrame,
    score_col: str,
    k: int = BUDGET_K,
    inj_type: str = "m20",
    n: int = 1000,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for recall@k for a specific injection type.

    Top-k flags are computed once.  Then ``n`` bootstrap samples of WEEKS
    (with replacement) are drawn; for each sample, recall is computed over
    the resampled rows.

    Vectorised via per-week aggregated counts (no row-level loop).

    Returns
    -------
    (mean, lo, hi): mean recall and 2.5/97.5 percentile CI.
    """
    rng = np.random.default_rng(seed)

    flags = topk_flags(df, score_col, k)
    mask_type = df["inj_type"] == inj_type

    # Per-week counts: flagged-and-type, and total of type
    df2 = df.assign(
        _flag_type=(flags & mask_type).astype("int32"),
        _is_type=mask_type.astype("int32"),
    )
    per_week = df2.groupby("week", sort=True)[["_flag_type", "_is_type"]].sum()

    # weeks array and corresponding count arrays
    weeks = per_week.index.to_numpy()
    flagged_counts = per_week["_flag_type"].to_numpy(dtype="int64")
    type_counts = per_week["_is_type"].to_numpy(dtype="int64")

    W = len(weeks)
    # Bootstrap: sample week indices with replacement
    boot_idx = rng.integers(0, W, size=(n, W))

    boot_flagged = flagged_counts[boot_idx].sum(axis=1)  # shape (n,)
    boot_total = type_counts[boot_idx].sum(axis=1)       # shape (n,)

    # Avoid division by zero
    recalls = np.where(boot_total > 0, boot_flagged / boot_total, 0.0)

    mean_recall = float(recalls.mean())
    lo = float(np.percentile(recalls, 2.5))
    hi = float(np.percentile(recalls, 97.5))

    return mean_recall, lo, hi


def paired_bootstrap_diff(
    df: pd.DataFrame,
    score_a: str,
    score_b: str,
    k: int = BUDGET_K,
    inj_type: str = "m20",
    n: int = 2000,
    seed: int = SEED,
    group: tuple[str, ...] = ("district", "week"),
) -> tuple[float, float, float]:
    """Recall(a) - recall(b) with a 95% interval, resampling (district, week)
    clusters with replacement. Both scorers see the same resampled clusters,
    so the interval is for the paired difference."""
    rng = np.random.default_rng(seed)
    is_type = (df["inj_type"] == inj_type).astype("int64")
    fa = (topk_flags(df, score_a, k, group) & (is_type == 1)).astype("int64")
    fb = (topk_flags(df, score_b, k, group) & (is_type == 1)).astype("int64")
    per = (pd.DataFrame({"a": fa, "b": fb, "t": is_type})
           .groupby([df[g] for g in group], observed=True).sum())
    per = per[per["t"] > 0]
    a, b, t = (per[c].to_numpy() for c in ("a", "b", "t"))
    idx = rng.integers(0, len(per), size=(n, len(per)))
    tt = t[idx].sum(axis=1)
    diff = a[idx].sum(axis=1) / tt - b[idx].sum(axis=1) / tt
    point = a.sum() / t.sum() - b.sum() / t.sum()
    return float(point), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
