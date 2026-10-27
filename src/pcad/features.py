"""Weekly panel construction and feature engineering for PriceCatcher."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pcad.config import MIN_PEERS


def week_start(dates: pd.Series) -> pd.Series:
    """Monday of the week containing each date."""
    return dates.dt.to_period("W-SUN").dt.start_time.dt.normalize()


def find_partial_weeks(raw_dir) -> list[pd.Timestamp]:
    """Weeks with < 50% of the median national weekly row count.

    Computed once from the date column of every month, so a per-state build
    drops the same weeks as a whole-country build would.
    """
    from pcad.load import month_files
    counts = pd.concat([week_start(pd.to_datetime(pd.read_parquet(p, columns=["date"])["date"]))
                        for p in month_files(raw_dir)]).value_counts()
    return sorted(counts[counts < 0.5 * counts.median()].index)


def weekly_panel(df: pd.DataFrame, partial_weeks=None) -> tuple[pd.DataFrame, dict]:
    """Aggregate raw daily rows into a weekly panel.

    Week is defined as the Monday of the ISO week containing each date
    (``pd.tseries.frequencies.to_offset('W-MON')`` semantics, i.e.
    ``date - date.weekday() * Day``).  We use ``to_period('W-SUN').start_time``
    which gives the Monday.

    Aggregation: median price and row count per (premise_code, item_code, week).
    Categorical columns (premise_type, state, district, item_group) are carried
    forward via first-value within the group (they are constant per premise /
    item).

    Partial weeks — those with < 50 % of the median weekly row count — are
    dropped.

    Returns
    -------
    panel:
        Columns: premise_code, item_code, week (datetime64[ns]), price (float32),
        n_obs (int32), premise_type, state, district, item_group.
    drops:
        ``{"dropped_partial_weeks": <int>, "partial_week_dates": [...]}``
    """
    drops: dict = {}

    # ── Week label: Monday of the week ───────────────────────────────────────
    # to_period('W-SUN') labels weeks Sun→Sat; .start_time gives the Monday.
    df = df.copy()
    df["week"] = (
        df["date"]
        .dt.to_period("W-SUN")
        .dt.start_time
        .dt.normalize()           # strip time component → midnight
    )

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg = (
        df.groupby(["premise_code", "item_code", "week"], sort=False)
        .agg(
            price=("price", "median"),
            n_obs=("price", "count"),
            premise_type=("premise_type", "first"),
            state=("state", "first"),
            district=("district", "first"),
            item_group=("item_group", "first"),
        )
        .reset_index()
    )
    agg["price"] = agg["price"].astype("float32")
    agg["n_obs"] = agg["n_obs"].astype("int32")

    # ── Drop partial weeks ────────────────────────────────────────────────────
    if partial_weeks is None:
        week_counts = agg.groupby("week")["n_obs"].sum()
        partial_weeks = week_counts[week_counts < 0.5 * week_counts.median()].index
    partial_weeks = pd.DatetimeIndex(partial_weeks)

    drops["dropped_partial_weeks"] = int(len(partial_weeks))
    drops["partial_week_dates"] = [str(w.date()) for w in sorted(partial_weeks)]
    drops["dropped_partial_week_rows"] = int(agg["week"].isin(partial_weeks).sum())

    panel = agg[~agg["week"].isin(partial_weeks)].copy()
    panel = panel.reset_index(drop=True)

    return panel, drops


def add_features_by_state(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run ``add_features`` one state at a time to bound peak memory.

    Every peer cell and every premise-item history lies inside one state, so
    the result is identical to a single call on the whole panel.
    """
    parts, total = [], 0
    for _, part in panel.groupby("state", observed=True, sort=True):
        f, d = add_features(part.reset_index(drop=True))
        parts.append(f)
        total += d["dropped_insufficient_peers"]
    feat = (pd.concat(parts, ignore_index=True)
            .sort_values(["week", "premise_code", "item_code"]).reset_index(drop=True))
    for col in ("premise_type", "state", "district", "item_group", "ref_level"):
        feat[col] = feat[col].astype("category")
    return feat, {"dropped_insufficient_peers": total}


def add_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute anomaly-detection features for every row in the weekly panel.

    The function accepts extra columns (``is_injected``, ``inj_type``) and
    preserves them in the output without modification.

    Reference-price hierarchy
    -------------------------
    A = (item_code, week, premise_type, district)   → MIN_PEERS distinct premises
    B = (item_code, week, premise_type, state)
    C = (item_code, week, state)
    Rows still below MIN_PEERS after C are dropped.

    Features
    --------
    log_ratio    : log(price / ref_price)
    robust_z     : log_ratio / max(1.4826 * MAD(log_ratio within cell), 0.02)
    own_jump     : log(price / prev_week_price) for same premise-item
    own_premium  : rolling median of *past* log_ratio (shift then roll 8)
    log_peers    : log(peer_count)

    Extra columns produced
    ----------------------
    ref_price, ref_level (str), peer_count (int)
    has_prev (bool, 1 when own_jump is valid)

    Returns
    -------
    feat:
        Sorted by (week, premise_code, item_code), reset index.
    drops:
        ``{"dropped_insufficient_peers": <int>}``
    """
    drops: dict = {}

    # Preserve extra injection columns if present
    extra_cols = [c for c in ("is_injected", "inj_type") if c in panel.columns]

    panel = panel.copy()

    # ── Step 1: peer counts at each hierarchy level ───────────────────────────
    # Level A: (item_code, week, premise_type, district)
    peer_A = (
        panel.groupby(["item_code", "week", "premise_type", "state", "district"],
                      sort=False)["premise_code"]
        .transform("nunique")
        .rename("peer_A")
    )
    # Level B: (item_code, week, premise_type, state)
    peer_B = (
        panel.groupby(["item_code", "week", "premise_type", "state"],
                      sort=False)["premise_code"]
        .transform("nunique")
        .rename("peer_B")
    )
    # Level C: (item_code, week, state)
    peer_C = (
        panel.groupby(["item_code", "week", "state"],
                      sort=False)["premise_code"]
        .transform("nunique")
        .rename("peer_C")
    )

    panel = panel.assign(peer_A=peer_A, peer_B=peer_B, peer_C=peer_C)

    # ── Step 2: assign ref_level ──────────────────────────────────────────────
    conditions = [
        panel["peer_A"] >= MIN_PEERS,
        panel["peer_B"] >= MIN_PEERS,
        panel["peer_C"] >= MIN_PEERS,
    ]
    choices = ["district", "state_type", "state"]
    panel["ref_level"] = np.select(conditions, choices, default="drop")

    # ── Step 3: drop rows with no usable peer cell ────────────────────────────
    n_before = len(panel)
    panel = panel[panel["ref_level"] != "drop"].copy()
    drops["dropped_insufficient_peers"] = n_before - len(panel)

    # ── Step 4: peer_count ────────────────────────────────────────────────────
    panel["peer_count"] = np.where(
        panel["ref_level"] == "district", panel["peer_A"],
        np.where(panel["ref_level"] == "state_type", panel["peer_B"], panel["peer_C"])
    ).astype("int32")

    # ── Step 5: ref_price (median price within chosen cell) ───────────────────
    # We use a group-key column to avoid repeating complex logic.
    # Build a cell-id string series per level, then compute medians.

    # For each row pick the groupby key appropriate to its ref_level
    # We compute median for all three level simultaneously, then select.

    def _ref_median(keys: list[str]) -> pd.Series:
        return (
            panel.groupby(keys, sort=False)["price"]
            .transform("median")
        )

    ref_A = _ref_median(["item_code", "week", "premise_type", "state", "district"])
    ref_B = _ref_median(["item_code", "week", "premise_type", "state"])
    ref_C = _ref_median(["item_code", "week", "state"])

    panel["ref_price"] = np.where(
        panel["ref_level"] == "district", ref_A,
        np.where(panel["ref_level"] == "state_type", ref_B, ref_C)
    ).astype("float32")

    # ── Step 6: log_ratio ─────────────────────────────────────────────────────
    panel["log_ratio"] = np.log(
        panel["price"].astype("float64") / panel["ref_price"].astype("float64")
    ).astype("float32")

    # ── Step 7: robust_z ──────────────────────────────────────────────────────
    # MAD of log price within the chosen peer cell (vectorised: two median
    # transforms instead of a Python function per group).
    log_price = np.log(panel["price"].astype("float64"))

    def _mad_within(keys: list[str]) -> pd.Series:
        g = panel[keys]
        dev = (log_price - log_price.groupby([g[k] for k in keys], sort=False)
               .transform("median")).abs()
        return dev.groupby([g[k] for k in keys], sort=False).transform("median")

    mad_A = _mad_within(["item_code", "week", "premise_type", "state", "district"])
    mad_B = _mad_within(["item_code", "week", "premise_type", "state"])
    mad_C = _mad_within(["item_code", "week", "state"])

    mad = np.where(
        panel["ref_level"] == "district", mad_A,
        np.where(panel["ref_level"] == "state_type", mad_B, mad_C)
    )
    denom = np.maximum(1.4826 * mad, 0.02)
    panel["robust_z"] = (panel["log_ratio"].astype("float64") / denom).astype("float32")

    # ── Step 8: own_jump and has_prev ─────────────────────────────────────────
    # Sort by (premise_code, item_code, week) to enable shift.
    panel = panel.sort_values(["premise_code", "item_code", "week"])
    grp_pi = panel.groupby(["premise_code", "item_code"], sort=False)

    # Only a recent record counts as "previous": within 21 days, which allows
    # one missing or dropped partial week but not a gap of months.
    prev_price = grp_pi["price"].shift(1)
    gap = panel["week"] - grp_pi["week"].shift(1)
    recent = gap <= pd.Timedelta(days=21)
    prev_price = prev_price.where(recent)
    has_prev = prev_price.notna().astype("int8")

    panel["own_jump"] = np.where(
        has_prev,
        np.log(panel["price"].astype("float64") / prev_price.astype("float64")),
        0.0,
    ).astype("float32")
    panel["has_prev"] = has_prev

    # ── Step 9: own_premium ───────────────────────────────────────────────────
    # Shift log_ratio back one period (exclude current), then rolling median of
    # the previous 8 *recorded* weeks (not calendar weeks).
    # Within each (premise_code, item_code) group, already sorted by week.
    shifted_lr = grp_pi["log_ratio"].shift(1)  # exclude current week
    rolled = (
        shifted_lr
        .groupby([panel["premise_code"], panel["item_code"]], sort=False)
        .rolling(8, min_periods=1)
        .median()
        .reset_index(level=[0, 1], drop=True)
    )
    panel["own_premium"] = rolled.reindex(panel.index).fillna(0.0).astype("float32")

    # ── Step 10: log_peers ────────────────────────────────────────────────────
    panel["log_peers"] = np.log(panel["peer_count"].astype("float64")).astype("float32")

    # ── Step 11: cleanup intermediate columns ─────────────────────────────────
    panel = panel.drop(columns=["peer_A", "peer_B", "peer_C"])

    # ── Step 12: sort and reset index ────────────────────────────────────────
    out_cols = (
        ["week", "premise_code", "item_code", "price", "n_obs",
         "premise_type", "state", "district", "item_group",
         "ref_price", "ref_level", "peer_count",
         "log_ratio", "robust_z", "own_jump", "has_prev",
         "own_premium", "log_peers"]
        + extra_cols
    )
    panel = panel[out_cols].sort_values(["week", "premise_code", "item_code"]).reset_index(drop=True)

    return panel, drops
