"""Weekly run: score the latest complete week, flag by conformal p-value, write a brief, log to MLflow.

Needs the features built for the recent months (00_download.py --months N, then 01_build.py) and
results/calibration_scores.npy (from 04_calibrate.py, committed to the repo).

    python scripts/07_weekly.py [--alpha 0.0001]

Outputs results/weekly/<week>.csv and <week>.md, and one MLflow run (./mlruns by default, or
any server in MLFLOW_TRACKING_URI, e.g. DagsHub).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import _path  # noqa: F401
from pcad import config
from pcad.brief import llm_brief, template_brief
from pcad.conformal import p_values
from pcad.detect import MADBaseline
from pcad.io import state_files
from pcad.labels import likely_cause
from pcad.load import load_lookups

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.0001)
args = parser.parse_args()
started = time.perf_counter()

calibration = np.load(config.RESULTS_DIR / "calibration_scores.npy").astype("float64")
files = state_files("features")
latest = max(pd.read_parquet(f, columns=["week"])["week"].max() for f in files)
week = pd.concat([pd.read_parquet(f, filters=[("week", "==", latest)]) for f in files], ignore_index=True)
for col in ("state", "district", "premise_type", "ref_level", "item_group"):
    week[col] = week[col].astype(str)

week["score"] = MADBaseline().score(week)
week["p_value"] = p_values(week["score"].to_numpy(), calibration)
flags = week[(week["p_value"] <= args.alpha) & (week["score"] > 0)].copy()
flags["likely_cause"] = likely_cause(flags)

items, premises = load_lookups(config.RAW_DIR)
flags = (flags.merge(items[["item_code", "item", "unit"]], on="item_code", how="left")
              .merge(premises[["premise_code", "premise"]], on="premise_code", how="left"))
flags["ratio_to_ref"] = np.exp(flags["log_ratio"]).round(2)
cols = ["week", "state", "district", "premise_code", "premise", "premise_type", "item_code", "item", "unit",
        "price", "ref_price", "ratio_to_ref", "peer_count", "score", "p_value", "likely_cause"]
flags = flags[cols].sort_values("p_value")

label = str(latest.date())
out_dir = config.RESULTS_DIR / "weekly"
out_dir.mkdir(parents=True, exist_ok=True)
flags.to_csv(out_dir / f"{label}.csv", index=False)

problems = []
if os.environ.get("GROQ_API_KEY") and len(flags):
    brief, problems = llm_brief(flags, label, args.alpha, len(week))
    brief_source = "llm" if not problems else "template (llm failed check)"
else:
    brief, brief_source = template_brief(flags, label, args.alpha, len(week)), "template"
(out_dir / f"{label}.md").write_text(brief, encoding="utf-8")

metrics = {
    "rows_scored": len(week),
    "flagged": len(flags),
    "flag_rate": len(flags) / max(len(week), 1),
    "districts_flagged": flags["district"].nunique(),
    "share_pack_unit_mismatch": float((flags["likely_cause"] == "pack/unit mismatch?").mean()) if len(flags) else 0.0,
    "seconds": round(time.perf_counter() - started, 1),
}
print(f"week {label}: {metrics['flagged']} flags of {metrics['rows_scored']:,} rows; brief: {brief_source}")
if problems:
    print("LLM brief rejected:", problems)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402  (imported late: slow, and only needed here)

# default: a local SQLite store (mlflow.db + mlartifacts/), uploaded by the CI job as an artifact
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(config._ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment("pricecatcher-weekly")
with mlflow.start_run(run_name=label):
    mlflow.log_params({"week": label, "alpha": args.alpha, "scorer": "mad_baseline",
                       "n_calibration": len(calibration), "brief": brief_source})
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(out_dir / f"{label}.csv"))
    mlflow.log_artifact(str(out_dir / f"{label}.md"))
print("logged to MLflow:", mlflow.get_tracking_uri())
