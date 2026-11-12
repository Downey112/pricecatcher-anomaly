"""Build the conformal calibration sample from training weeks and check it on later weeks.

Writes results/calibration_scores.npy (sorted sample of training-week MAD scores, used by the
weekly run so it does not need the whole year) and results/conformal.json (flag rate per alpha
on validation and test weeks, compared with the nominal alpha).
"""
import json

import numpy as np
import pandas as pd

import _path  # noqa: F401
from pcad import config
from pcad.conformal import calibration_sample, p_values
from pcad.io import state_files

ALPHAS = [0.01, 0.001, 0.0001]


def mad_scores(lo, hi) -> pd.DataFrame:
    """MAD scores (max(robust_z, 0)) for weeks in [lo, hi], all states, two columns only."""
    parts = []
    for f in state_files("features"):
        df = pd.read_parquet(f, columns=["week", "robust_z"],
                             filters=[("week", ">=", lo), ("week", "<=", hi)])
        parts.append(pd.DataFrame({"week": df["week"],
                                   "score": np.maximum(df["robust_z"].to_numpy("float64"), 0.0)}))
    return pd.concat(parts, ignore_index=True)


train = mad_scores(*config.TRAIN)
calibration = calibration_sample(train["score"].to_numpy(), n=200_000, seed=config.SEED)
np.save(config.RESULTS_DIR / "calibration_scores.npy", calibration.astype("float32"))

report = {"n_train_rows": len(train), "n_calibration": len(calibration),
          "score_quantiles": {q: round(float(np.quantile(calibration, q)), 3)
                              for q in (0.5, 0.9, 0.99, 0.999, 0.9999)}}
for name, (lo, hi) in (("validation", config.VAL), ("test", config.TEST)):
    period = mad_scores(lo, hi)
    p = p_values(period["score"].to_numpy(), calibration)
    weeks = period["week"].nunique()
    report[name] = {
        "rows": len(period),
        "weeks": int(weeks),
        "flag_rate": {str(a): round(float((p <= a).mean()), 6) for a in ALPHAS},
        "flags_per_week": {str(a): round(float((p <= a).sum() / weeks), 1) for a in ALPHAS},
    }
(config.RESULTS_DIR / "conformal.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
