"""Raw data loading and lookup joining for PriceCatcher."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_lookups(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (items_df, premises_df) from the lookup parquet files.

    Strips leading/trailing whitespace from string columns and drops the
    placeholder row (code == -1).
    """
    items = pd.read_parquet(raw_dir / "lookup_item.parquet")
    premises = pd.read_parquet(raw_dir / "lookup_premise.parquet")

    # Strip whitespace from key categoricals in premises
    for col in ("premise_type", "state", "district"):
        if col in premises.columns:
            premises[col] = premises[col].str.strip()

    # Strip whitespace in items
    for col in ("item_group", "item_category"):
        if col in items.columns:
            items[col] = items[col].str.strip()

    # Drop placeholder rows
    items = items[items["item_code"] != -1].copy()
    premises = premises[premises["premise_code"] != -1].copy()

    return items, premises


def month_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("pricecatcher_2026-*.parquet"))


def load_raw(raw_dir: Path, premise_codes=None) -> tuple[pd.DataFrame, dict]:
    """Concatenate monthly parquet files, downcast types, join lookups.

    Parameters
    ----------
    raw_dir:
        Directory containing ``pricecatcher_2026-MM.parquet`` files (01..10)
        and the two lookup parquets.

    Returns
    -------
    df:
        Columns: date (datetime64[ns]), premise_code (int32), item_code (int16),
        price (float32), premise_type (str), state (str), district (str),
        item_group (str).
    drops:
        Dict with counts of rows removed at each stage.
    """
    drops: dict = {}

    # ── 1. Concatenate monthly files ─────────────────────────────────────────
    # Optional premise filter is pushed down to the parquet reader, so a
    # single state can be loaded without reading the whole country.
    filters = None
    if premise_codes is not None:
        filters = [("premise_code", "in", [int(c) for c in premise_codes])]
    months = []
    for path in month_files(raw_dir):
        m = pd.read_parquet(path, filters=filters)
        months.append(pd.DataFrame({
            "date": pd.to_datetime(m["date"]),
            "premise_code": m["premise_code"].astype("int32"),
            "item_code": m["item_code"].astype("int16"),
            "price": m["price"].astype("float32"),
        }))
        del m
    df = pd.concat(months, ignore_index=True)
    del months
    drops["rows_read"] = len(df)

    # ── 3. Drop non-positive prices ──────────────────────────────────────────
    n_before = len(df)
    df = df[df["price"] > 0]
    drops["dropped_nonpositive_price"] = n_before - len(df)

    # ── 4. Load lookups ──────────────────────────────────────────────────────
    items, premises = load_lookups(raw_dir)

    # ── 5. Inner-join items lookup ────────────────────────────────────────────
    n_before = len(df)
    df = df.merge(
        items[["item_code", "item_group"]],
        on="item_code",
        how="inner",
    )
    drops["dropped_unmatched_item"] = n_before - len(df)

    # ── 6. Inner-join premises lookup ─────────────────────────────────────────
    n_before = len(df)
    df = df.merge(
        premises[["premise_code", "premise_type", "state", "district"]],
        on="premise_code",
        how="inner",
    )
    drops["dropped_unmatched_premise"] = n_before - len(df)

    # ── 7. Final strip of joined string columns ───────────────────────────────
    for col in ("premise_type", "state", "district", "item_group"):
        df[col] = df[col].str.strip().astype("category")

    # ── 8. Select output columns ─────────────────────────────────────────────
    df = df[["date", "premise_code", "item_code", "price",
             "premise_type", "state", "district", "item_group"]].copy()

    return df, drops
