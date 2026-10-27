"""Central configuration for the pcad package."""
from pathlib import Path

import pandas as pd

# ── directory roots ──────────────────────────────────────────────────────────
# config.py lives at  src/pcad/config.py  →  parents[2] == project root
_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR       = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
RESULTS_DIR   = _ROOT / "results"

# ── temporal split boundaries (week-start Mondays) ───────────────────────────
TRAIN = (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-06-29"))
VAL   = (pd.Timestamp("2026-07-06"), pd.Timestamp("2026-08-31"))
TEST  = (pd.Timestamp("2026-09-07"), pd.Timestamp("2099-12-31"))

# ── model / evaluation hyper-parameters ─────────────────────────────────────
MIN_PEERS = 5
BUDGET_K  = 20
SEED      = 42

# ── injection markups ────────────────────────────────────────────────────────
MARKUPS = [0.10, 0.20, 0.30, 0.50]

# ── feature names fed into ML scorers ────────────────────────────────────────
# log_peers is kept as a column but not used: peer-group size describes the
# market, not the price, and thin cells would otherwise look anomalous.
FEATURES = ["log_ratio", "robust_z", "own_jump", "own_premium"]
