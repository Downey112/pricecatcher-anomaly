"""Score the latest real week (no injection) and write the inspection shortlist."""
import json

import numpy as np
import pandas as pd

import _path  # noqa: F401
from pcad import config
from pcad.detect import ECODScorer, IForestScorer, MADBaseline
from pcad.evaluate import topk_flags
from pcad.io import state_files, train_sample
from pcad.labels import likely_cause
from pcad.load import load_lookups

best = json.loads((config.RESULTS_DIR / "summary.json").read_text())["chosen_on_val"]
scorer = {"mad_baseline": MADBaseline, "iforest": IForestScorer, "ecod": ECODScorer}[best]()
if best != "mad_baseline":  # the baseline needs no training
    scorer.fit(train_sample())

files = state_files("features")
latest = max(pd.read_parquet(f, columns=["week"])["week"].max() for f in files)
wk = pd.concat([pd.read_parquet(f, filters=[("week", "==", latest)]) for f in files],
               ignore_index=True)
for col in ("state", "district", "premise_type", "ref_level", "item_group"):
    wk[col] = wk[col].astype(str)
wk["score"] = scorer.score(wk)
wk["group"] = wk["state"] + " | " + wk["district"]
wk = wk[wk["score"] > 0]
wk = wk[topk_flags(wk, "score", config.BUDGET_K, group=("group", "week")).to_numpy()]

wk["likely_cause"] = likely_cause(wk)

items, premises = load_lookups(config.RAW_DIR)
out = (wk.merge(items[["item_code", "item", "unit"]], on="item_code", how="left")
         .merge(premises[["premise_code", "premise"]], on="premise_code", how="left"))
out["ratio_to_ref"] = np.exp(out["log_ratio"]).round(2)
cols = ["week", "state", "district", "premise_code", "premise", "premise_type",
        "item_code", "item", "unit", "price",
        "ref_price", "ratio_to_ref", "ref_level", "peer_count", "own_premium", "score",
        "likely_cause"]
out = out[cols].sort_values(["state", "district", "score"], ascending=[True, True, False])
out.to_csv(config.RESULTS_DIR / "shortlist_latest.csv", index=False)
print(f"scorer={best} week={latest.date()} flagged={len(out)} "
      f"districts={out['district'].nunique()}")
print(out["likely_cause"].value_counts().to_string())
print(out.sort_values("score", ascending=False).head(10)[
    ["district", "premise", "item", "price", "ref_price", "ratio_to_ref", "likely_cause"]].to_string())
