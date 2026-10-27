"""Rule-based hints for an inspector: why a flagged price may stand out.

Crude and checked in order; not a classifier.
"""
import numpy as np
import pandas as pd

CAUSES = ["decimal slip?", "first observation", "pack/unit mismatch?", "persistent premium",
          "spike", "above peers"]


def likely_cause(df: pd.DataFrame) -> np.ndarray:
    ratio = np.exp(df["log_ratio"].to_numpy(dtype="float64"))
    near_multiple = (ratio >= 1.8) & (np.abs(ratio - np.round(ratio)) <= 0.10)
    return np.select(
        [ratio > 8, df["has_prev"].to_numpy() == 0, near_multiple,
         df["own_premium"].to_numpy() > 0.15, df["own_jump"].to_numpy() > 0.10],
        CAUSES[:-1], CAUSES[-1])
