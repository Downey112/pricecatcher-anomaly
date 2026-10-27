"""Load raw PriceCatcher months, build the weekly panel and its features.

Runs one state at a time (every peer cell and premise history lies inside one
state) and writes one parquet file per state, so peak memory stays well under
1 GB on the full 16.6M-row year.
"""
import json
from collections import Counter

import pyarrow.parquet as pq

import _path  # noqa: F401
from pcad import config
from pcad.features import add_features, find_partial_weeks, weekly_panel
from pcad.load import load_lookups, load_raw, month_files

_, premises = load_lookups(config.RAW_DIR)
partial = find_partial_weeks(config.RAW_DIR)
print("partial weeks dropped:", [str(w.date()) for w in partial])

for sub in ("panel", "features"):
    d = config.PROCESSED_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.parquet"):
        old.unlink()

counts = Counter()
levels = Counter()
weeks = set()
for state, codes in premises.groupby("state")["premise_code"]:
    df, drops = load_raw(config.RAW_DIR, premise_codes=codes)
    panel, pdrops = weekly_panel(df, partial_weeks=partial)
    del df
    feat, fdrops = add_features(panel)
    slug = state.replace(" ", "_").replace(".", "")
    panel.to_parquet(config.PROCESSED_DIR / "panel" / f"{slug}.parquet", index=False)
    feat.to_parquet(config.PROCESSED_DIR / "features" / f"{slug}.parquet", index=False)
    counts.update({k: v for k, v in {**drops, **pdrops, **fdrops}.items() if isinstance(v, int)})
    counts.update(panel_rows=len(panel), feature_rows=len(feat))
    levels.update(feat["ref_level"].value_counts().to_dict())
    weeks |= set(feat["week"].unique())
    print(f"{state}: panel {len(panel):,} features {len(feat):,}")
    del panel, feat

# rows whose premise code is missing from the lookup never enter a state filter
counts["raw_rows"] = sum(pq.ParquetFile(p).metadata.num_rows for p in month_files(config.RAW_DIR))
counts["dropped_unmatched_premise"] = counts["raw_rows"] - counts["rows_read"]
counts["weeks"] = len(weeks)
counts.pop("dropped_partial_weeks", None)
summary = {**dict(counts), "partial_weeks": [str(w.date()) for w in partial],
           "ref_level_share": {k: round(v / counts["feature_rows"], 4) for k, v in levels.items()}}
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(config.RESULTS_DIR / "build_counts.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
