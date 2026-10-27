"""Per-state processed files: listing and sampling without loading everything."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pcad import config


def state_files(kind: str) -> list:
    """Sorted per-state parquet files under data/processed/<kind>/."""
    return sorted((config.PROCESSED_DIR / kind).glob("*.parquet"))


def week_filter(lo, hi):
    return [("week", ">=", pd.Timestamp(lo)), ("week", "<=", pd.Timestamp(hi))]


def train_sample(n: int = 500_000, seed: int = config.SEED) -> pd.DataFrame:
    """Uniform random sample of about n TRAIN feature rows across all states."""
    cols = config.FEATURES
    files = state_files("features")
    filt = week_filter(*config.TRAIN)
    sizes = [pq.read_table(f, columns=["week"], filters=filt).num_rows for f in files]
    frac = min(1.0, n / max(sum(sizes), 1))
    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        t = pd.read_parquet(f, columns=cols, filters=filt)
        parts.append(t[rng.random(len(t)) < frac])
    return pd.concat(parts, ignore_index=True)
