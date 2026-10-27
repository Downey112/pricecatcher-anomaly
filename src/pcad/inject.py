"""Synthetic price injection for detection evaluation."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from pcad.config import MARKUPS, SEED


def inject(
    panel: pd.DataFrame,
    weeks: Iterable,
    frac: float = 0.02,
    markups: list[float] = MARKUPS,
    slip_frac: float = 0.002,
    seed: int = SEED,
    persist_frac: float = 0.0,
    persist_markup: float = 0.20,
) -> pd.DataFrame:
    """Return a copy of ``panel`` with synthetic high-price anomalies injected.

    Among rows whose ``week`` is in ``weeks``:
    - A fraction ``frac`` of rows are sampled (without replacement) and their
      price is multiplied by ``1 + m`` where ``m`` is drawn uniformly from
      ``markups``.
    - A disjoint fraction ``slip_frac`` of rows are sampled and their price is
      multiplied by 10 (decimal-place slip).

    Rows outside ``weeks`` are left untouched with ``inj_type='none'`` and
    ``is_injected=False``.

    The two samples are disjoint: slip rows are excluded from the markup pool.

    Uses ``numpy.random.Generator`` seeded with ``seed``.

    Columns added
    -------------
    is_injected : bool
    inj_type    : str in {'none', 'm10', 'm20', 'm30', 'm50', 'slip'}
    """
    rng = np.random.default_rng(seed)
    panel = panel.copy()

    panel["is_injected"] = False
    panel["inj_type"] = "none"

    weeks_set = set(pd.Timestamp(w) for w in weeks)
    in_window = panel["week"].isin(weeks_set)
    window_idx = panel.index[in_window].to_numpy()

    if len(window_idx) == 0:
        return panel

    n_slip = max(1, int(round(len(window_idx) * slip_frac)))
    n_markup = max(0, int(round(len(window_idx) * frac)))

    # Sample slip rows first
    slip_pos = rng.choice(len(window_idx), size=n_slip, replace=False)
    slip_idx = window_idx[slip_pos]

    # Remaining rows are the markup pool
    remaining_mask = np.ones(len(window_idx), dtype=bool)
    remaining_mask[slip_pos] = False
    remaining_idx = window_idx[remaining_mask]

    # Sample markup rows from remaining
    n_markup = min(n_markup, len(remaining_idx))
    if n_markup > 0:
        markup_pos = rng.choice(len(remaining_idx), size=n_markup, replace=False)
        markup_idx = remaining_idx[markup_pos]
    else:
        markup_idx = np.array([], dtype=window_idx.dtype)

    # Draw a markup multiplier for each markup row
    markup_choices = np.array(markups, dtype="float64")
    chosen_markups = rng.choice(markup_choices, size=len(markup_idx), replace=True)

    # Apply slip injection
    panel.loc[slip_idx, "price"] = (
        panel.loc[slip_idx, "price"].astype("float64") * 10.0
    ).astype("float32")
    panel.loc[slip_idx, "is_injected"] = True
    panel.loc[slip_idx, "inj_type"] = "slip"

    # Apply markup injection
    if len(markup_idx) > 0:
        prices = panel.loc[markup_idx, "price"].astype("float64").to_numpy()
        new_prices = (prices * (1.0 + chosen_markups)).astype("float32")
        panel.loc[markup_idx, "price"] = new_prices
        panel.loc[markup_idx, "is_injected"] = True

        # Assign inj_type per row based on the chosen markup
        markup_label_map = {m: f"m{int(round(m * 100))}" for m in markups}
        inj_types = np.array([markup_label_map[m] for m in chosen_markups])
        panel.loc[markup_idx, "inj_type"] = inj_types

    # Persistent markup: whole premise-item pairs raised by persist_markup in
    # every window week (inj_type 'p20'), so own_jump fires at most once and
    # the shop's own history slowly absorbs the premium. Pairs already hit by
    # a one-week change are skipped.
    if persist_frac > 0:
        pair_cols = ["premise_code", "item_code"]
        touched = panel.loc[panel["is_injected"], pair_cols].drop_duplicates()
        pairs = panel.loc[in_window, pair_cols].drop_duplicates()
        pairs = pairs[~pd.MultiIndex.from_frame(pairs).isin(pd.MultiIndex.from_frame(touched))]
        n_pairs = int(round(len(pairs) * persist_frac))
        if n_pairs > 0:
            pick = pairs.iloc[rng.choice(len(pairs), size=n_pairs, replace=False)]
            key = pd.MultiIndex.from_frame(panel[["premise_code", "item_code"]])
            hit = (key.isin(pd.MultiIndex.from_frame(pick)) & in_window.to_numpy()
                   & ~panel["is_injected"].to_numpy())
            panel.loc[hit, "price"] = (panel.loc[hit, "price"].astype("float64")
                                       * (1 + persist_markup)).astype("float32")
            panel.loc[hit, "is_injected"] = True
            panel.loc[hit, "inj_type"] = f"p{int(round(persist_markup * 100))}"

    return panel
