"""Export the small tables the Power BI report reads (powerbi/data/*.csv).

Run after 01-03. Everything here is derived from results/ and the processed
per-state features; no new modelling.
"""
import json

import numpy as np
import pandas as pd

import _path  # noqa: F401
from pcad import config
from pcad.detect import ECODScorer, IForestScorer, MADBaseline
from pcad.evaluate import topk_flags
from pcad.io import state_files, train_sample
from pcad.labels import CAUSES, likely_cause
from pcad.load import load_lookups

OUT = config.RESULTS_DIR.parent / "powerbi" / "data"
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((config.RESULTS_DIR / "summary.json").read_text())
build = json.loads((config.RESULTS_DIR / "build_counts.json").read_text())
items, premises = load_lookups(config.RAW_DIR)
cause_order = {c: i + 1 for i, c in enumerate(CAUSES)}

# ---------------------------------------------------------------- shortlist (latest week)
sl = pd.read_csv(config.RESULTS_DIR / "shortlist_latest.csv")
sl = sl.merge(items[["item_code", "item_category"]], on="item_code", how="left")
sl["district_key"] = sl["district"] + ", " + sl["state"]
sl["cause_order"] = sl["likely_cause"].map(cause_order)
sl["rank_in_district"] = (sl.groupby("district_key")["score"]
                          .rank(method="first", ascending=False).astype(int))
sl["premise"] = sl["premise"].str.strip()
cols = ["state", "district", "district_key", "premise", "premise_type", "item_category", "item",
        "unit", "price", "ref_price", "ratio_to_ref", "peer_count", "own_premium", "score",
        "rank_in_district", "likely_cause", "cause_order"]
sl[cols].to_csv(OUT / "shortlist.csv", index=False)

# ---------------------------------------------------------------- all weeks, same rule
best = summary["chosen_on_val"]
scorer = {"mad_baseline": MADBaseline, "iforest": IForestScorer, "ecod": ECODScorer}[best]()
if best != "mad_baseline":
    scorer.fit(train_sample())
weekly, item_rows = [], []
for f in state_files("features"):
    feat = pd.read_parquet(f)
    for col in ("state", "district"):
        feat[col] = feat[col].astype(str)
    feat["score"] = scorer.score(feat)
    item_obs = feat.groupby("item_code").size().rename("prices_checked")
    flagged = feat[topk_flags(feat, "score", config.BUDGET_K).to_numpy()].copy()
    flagged["likely_cause"] = likely_cause(flagged)
    flagged["ratio"] = np.exp(flagged["log_ratio"])
    w = (flagged.groupby(["week", "state", "likely_cause"])
         .agg(flags=("score", "size"), mean_ratio=("ratio", "mean")).reset_index())
    weekly.append(w)
    item_rows.append(pd.concat([flagged.groupby("item_code").size().rename("flags"), item_obs],
                               axis=1).fillna(0).reset_index())
    del feat, flagged
fw = pd.concat(weekly, ignore_index=True)
fw["cause_order"] = fw["likely_cause"].map(cause_order)
fw["week"] = pd.to_datetime(fw["week"]).dt.date
fw.to_csv(OUT / "flags_weekly.csv", index=False)

it = (pd.concat(item_rows).groupby("item_code")[["flags", "prices_checked"]].sum().reset_index()
      .merge(items[["item_code", "item", "unit", "item_category"]], on="item_code", how="left"))
it[["flags", "prices_checked"]] = it[["flags", "prices_checked"]].astype(int)
it["flag_rate"] = it["flags"] / it["prices_checked"]
it = it[it["prices_checked"] >= 1000].sort_values("flag_rate", ascending=False)
it[["item_category", "item", "unit", "flags", "prices_checked", "flag_rate"]].to_csv(
    OUT / "items_flagged.csv", index=False)

# ---------------------------------------------------------------- evaluation
labels = {"mad_baseline": "MAD baseline", "iforest": "Isolation Forest", "ecod": "ECOD"}
inj = {"m10": ("+10% one week", 1), "m20": ("+20% one week", 2), "m30": ("+30% one week", 3),
       "m50": ("+50% one week", 4), "p20": ("+20% held", 5), "slip": ("×10 slip", 6)}
m = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
m = m[m["k"] == config.BUDGET_K].copy()
m["scorer_label"] = m["scorer"].map(labels)
m["scorer_order"] = m["scorer"].map({"mad_baseline": 1, "iforest": 2, "ecod": 3})
m["planted"] = m["inj_type"].map(lambda t: inj[t][0])
m["planted_order"] = m["inj_type"].map(lambda t: inj[t][1])
m[["split", "scorer_label", "scorer_order", "planted", "planted_order", "recall",
   "precision_at_k", "average_precision"]].to_csv(OUT / "recall.csv", index=False)

rows = []
for key, v in summary["test_paired_recall_diff_k20"].items():
    _, _, other, t = key.split("_", 3)
    rows.append({"comparison": f"vs {labels[other]}", "planted": inj[t][0],
                 "planted_order": inj[t][1], "diff": v["diff"], "ci_lo": v["ci_lo"],
                 "ci_hi": v["ci_hi"],
                 "verdict": "baseline" if v["ci_lo"] > 0 else
                 ("model" if v["ci_hi"] < 0 else "unclear")})
pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False)

funnel = pd.DataFrame([
    ("Raw records", build["raw_rows"], 1),
    ("Known item & shop", build["rows_read"] - build["dropped_unmatched_item"], 2),
    ("Shop-item-weeks", build["panel_rows"] + build["dropped_partial_week_rows"], 3),
    ("Full weeks only", build["panel_rows"], 4),
    ("Scored (5+ peers)", build["feature_rows"], 5),
], columns=["step", "rows", "step_order"])
funnel.to_csv(OUT / "funnel.csv", index=False)

latest = pd.to_datetime(pd.read_csv(config.RESULTS_DIR / "shortlist_latest.csv")["week"].iloc[0])
kv = {
    "raw_rows": build["raw_rows"], "scored_rows": build["feature_rows"], "weeks": build["weeks"],
    "latest_week_serial": (latest - pd.Timestamp("1899-12-30")).days,
    "test_rows": summary["test_rows"], "test_planted": summary["test_injected"],
    "test_district_weeks": summary["test_district_weeks"], "budget_k": config.BUDGET_K,
}
pd.DataFrame(list(kv.items()), columns=["metric", "value"]).to_csv(OUT / "metrics.csv", index=False)
pd.DataFrame([{"chosen_scorer": labels[best], "latest_week": latest.date()}]).to_csv(
    OUT / "run_info.csv", index=False)

for p in sorted(OUT.glob("*.csv")):
    print(f"{p.name}: {sum(1 for _ in open(p, encoding='utf-8')) - 1} rows")
