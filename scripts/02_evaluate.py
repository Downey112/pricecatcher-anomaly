"""Synthetic-injection evaluation.

Scorers are fitted on a sample of real TRAIN weeks. Markups are injected into
the weekly panel for VAL and TEST weeks (separately), features are recomputed
so peer medians and own-history react, and each scorer ranks rows within
(district, week) under a fixed inspection budget. The scorer is chosen on VAL;
TEST is reported once. Runs state by state to keep memory low.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import _path  # noqa: F401
from pcad import config
from pcad.detect import ECODScorer, IForestScorer, MADBaseline
from pcad.evaluate import budget_metrics, paired_bootstrap_diff
from pcad.features import add_features
from pcad.inject import inject
from pcad.io import state_files, train_sample

train = train_sample()
scorers = {"mad_baseline": MADBaseline(), "iforest": IForestScorer(), "ecod": ECODScorer()}
for s in scorers.values():
    s.fit(train)
print(f"fitted on {len(train):,} sampled train rows")
del train

KEEP = ["week", "state", "district", "premise_code", "item_code", "log_ratio",
        "is_injected", "inj_type"]


def score_split(lo, hi, seed):
    parts = []
    for i, f in enumerate(state_files("panel")):
        panel = pd.read_parquet(f)
        weeks = panel.loc[panel["week"].between(lo, hi), "week"].unique()
        # ~0.5% one-week markups + 0.1% slips + 0.5% of pairs with a +20%
        # markup held for the whole window: about 10 planted rows per
        # district-week, so the K=20 budget is not saturated by construction.
        inj = inject(panel, weeks, frac=0.005, slip_frac=0.001, persist_frac=0.005,
                     seed=seed + i)
        feat, _ = add_features(inj)
        feat = feat[feat["week"].between(lo, hi)]
        out = feat[KEEP].copy()
        for key, s in scorers.items():
            out[key] = s.score(feat)
        parts.append(out)
    df = pd.concat(parts, ignore_index=True)
    # budget groups are (district, week); prefix the state so district names
    # that repeat across states stay separate
    df["district"] = df["state"].astype(str) + " | " + df["district"].astype(str)
    return df


def metrics_for(df, name):
    rows = []
    for key in scorers:
        m = budget_metrics(df, key)
        m.insert(0, "scorer", key)
        m.insert(0, "split", name)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


val_df = score_split(*config.VAL, seed=config.SEED)
val_m = metrics_for(val_df, "val")
del val_df
test_df = score_split(*config.TEST, seed=config.SEED + 1000)
test_m = metrics_for(test_df, "test")

metrics = pd.concat([val_m, test_m], ignore_index=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
metrics.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)

# choose on validation: recall within budget K=20 at a 20% markup
sel = val_m[(val_m["k"] == config.BUDGET_K) & (val_m["inj_type"] == "m20")]
best = sel.sort_values("recall", ascending=False)["scorer"].iloc[0]

# paired differences vs the baseline, resampling (district, week) clusters
paired = {}
for inj_type in ("m20", "p20", "m50"):
    for other in ("iforest", "ecod"):
        d, lo, hi = paired_bootstrap_diff(test_df, "mad_baseline", other, k=config.BUDGET_K,
                                          inj_type=inj_type, seed=config.SEED)
        paired[f"mad_minus_{other}_{inj_type}"] = {"diff": round(d, 4), "ci_lo": round(lo, 4),
                                                   "ci_hi": round(hi, 4)}
grp = test_df.groupby(["district", "week"]).size()
summary = {"chosen_on_val": best, "test_paired_recall_diff_k20": paired,
           "test_rows": len(test_df), "test_injected": int(test_df["is_injected"].sum()),
           "test_injected_by_type": test_df["inj_type"].value_counts().drop("none").to_dict(),
           "test_weeks": int(test_df["week"].nunique()),
           "test_district_weeks": int(len(grp)),
           "rows_per_district_week_median": float(grp.median())}
(config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(metrics[metrics["k"] == config.BUDGET_K].pivot_table(
    index=["split", "scorer"], columns="inj_type", values="recall").round(3).to_string())
print(metrics.groupby(["split", "scorer", "k"])[["precision_at_k", "average_precision"]]
      .first().round(3).to_string())

# figure: test recall at K=20 by markup level
order = ["m10", "m20", "m30", "m50", "p20", "slip"]
t = test_m[test_m["k"] == config.BUDGET_K].pivot(index="inj_type", columns="scorer", values="recall")
t = t.reindex([o for o in order if o in t.index])
ax = t.plot(kind="bar", figsize=(7, 4), rot=0)
ax.set_ylabel(f"Share caught in top {config.BUDGET_K} per district-week")
ax.set_xlabel("Planted change (m10 = +10% for one week, p20 = +20% held, slip = x10)")
ax.set_ylim(0, 1)
ax.set_title("Test weeks: recall within the inspection budget")
plt.tight_layout()
plt.savefig(config.RESULTS_DIR / "recall_by_markup.png", dpi=150)
