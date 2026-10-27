import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcad.detect import MADBaseline  # noqa: E402
from pcad.evaluate import topk_flags  # noqa: E402
from pcad.features import add_features  # noqa: E402
from pcad.inject import inject  # noqa: E402

WEEKS = pd.date_range("2026-01-05", periods=6, freq="7D")


def make_panel(n_district_a=6, n_district_b=2, price=10.0):
    """One item; district A has enough peers, district B does not."""
    rows = []
    pid = 0
    for district, n in (("A", n_district_a), ("B", n_district_b)):
        for _ in range(n):
            pid += 1
            for w in WEEKS:
                rows.append(dict(premise_code=pid, item_code=1, week=w, price=price,
                                 n_obs=1, premise_type="Kedai Runcit", state="S1",
                                 district=district, item_group="G"))
    p = pd.DataFrame(rows)
    p["price"] = p["price"].astype("float32")
    return p


def test_reference_falls_back_when_district_is_thin():
    feat, _ = add_features(make_panel())
    levels = feat.groupby("district")["ref_level"].unique()
    assert set(levels["A"]) == {"district"}
    assert set(levels["B"]) == {"state_type"}


def test_rows_dropped_when_no_level_has_enough_peers():
    feat, drops = add_features(make_panel(n_district_a=2, n_district_b=2))
    assert len(feat) == 0
    assert sum(v for v in drops.values() if isinstance(v, (int, np.integer))) > 0


def test_own_history_features_ignore_future_weeks():
    p = make_panel()
    target = (p["premise_code"] == 1)
    base, _ = add_features(p)
    p2 = p.copy()
    last = target & (p2["week"] == WEEKS[-1])
    p2.loc[last, "price"] = 1000.0  # change only the final week
    changed, _ = add_features(p2)
    key = ["premise_code", "item_code", "week"]
    a = base.set_index(key).sort_index()
    b = changed.set_index(key).sort_index()
    earlier = a.index.get_level_values("week") < WEEKS[-1]
    for col in ("own_jump", "own_premium"):
        np.testing.assert_allclose(a.loc[earlier, col], b.loc[earlier, col])


def test_own_jump_reflects_previous_week():
    p = make_panel()
    p.loc[(p["premise_code"] == 1) & (p["week"] == WEEKS[3]), "price"] = 20.0
    feat, _ = add_features(p)
    row = feat[(feat["premise_code"] == 1) & (feat["week"] == WEEKS[3])].iloc[0]
    assert row["own_jump"] == pytest.approx(np.log(2.0), rel=1e-4)
    assert row["log_ratio"] > 0


def test_injection_only_touches_requested_weeks_and_moves_features():
    p = make_panel(n_district_a=50, n_district_b=0)
    test_weeks = WEEKS[-2:]
    inj = inject(p, test_weeks, frac=0.2, slip_frac=0.02, seed=0)
    assert not inj.loc[~inj["week"].isin(test_weeks), "is_injected"].any()
    assert inj["is_injected"].any()
    changed = inj["price"].to_numpy() != p["price"].to_numpy()
    np.testing.assert_array_equal(changed, inj["is_injected"].to_numpy())
    feat, _ = add_features(inj)
    assert (feat.loc[feat["is_injected"], "log_ratio"] > 0).all()


def test_topk_returns_at_most_k_per_group():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"district": np.repeat(["A", "B", "C"], 30),
                       "week": WEEKS[0], "s": rng.random(90)})
    df.loc[df["district"] == "C", "s"] = 0.5  # all ties
    f = topk_flags(df, "s", 5)
    counts = f.groupby(df["district"]).sum()
    assert (counts == 5).all()
    top_a = df[(df["district"] == "A")].nlargest(5, "s").index
    assert f.loc[top_a].all()


def test_mad_baseline_ignores_cheap_prices():
    df = pd.DataFrame({"robust_z": [-5.0, 0.0, 3.0]})
    np.testing.assert_array_equal(MADBaseline().fit(df).score(df), [0.0, 0.0, 3.0])


def test_by_state_matches_single_call():
    from pcad.features import add_features_by_state
    a = make_panel()
    b = make_panel()
    b["state"] = "S2"  # same district names as S1, different prices
    b["premise_code"] += 100
    b["price"] = (b["price"] * 2).astype("float32")
    p = pd.concat([a, b], ignore_index=True)
    whole, _ = add_features(p)
    split, _ = add_features_by_state(p)
    cols = ["premise_code", "week", "ref_price", "log_ratio", "robust_z", "own_jump", "own_premium"]
    pd.testing.assert_frame_equal(whole[cols].reset_index(drop=True),
                                  split[cols].reset_index(drop=True))


def test_own_jump_ignores_stale_previous_record():
    p = make_panel()
    gap = (p["premise_code"] == 1) & p["week"].isin(WEEKS[1:5])
    p = p[~gap]  # premise 1 skips four weeks: a 35-day gap
    feat, _ = add_features(p)
    row = feat[(feat["premise_code"] == 1) & (feat["week"] == WEEKS[5])].iloc[0]
    assert row["has_prev"] == 0 and row["own_jump"] == 0


def test_persistent_markup_covers_whole_pairs_in_window():
    p = make_panel(n_district_a=50, n_district_b=0)
    win = WEEKS[-3:]
    inj = inject(p, win, frac=0.0, slip_frac=0.0, persist_frac=0.1, seed=1)
    hit = inj[inj["inj_type"] == "p20"]
    assert len(hit) > 0 and hit["week"].isin(win).all()
    weeks_per_pair = hit.groupby("premise_code")["week"].nunique()
    assert (weeks_per_pair == len(win)).all()
    np.testing.assert_allclose(hit["price"], 12.0, rtol=1e-5)


def test_topk_never_flags_zero_scores():
    df = pd.DataFrame({"district": "A", "week": WEEKS[0], "s": [0.0] * 8 + [1.0, 2.0]})
    assert topk_flags(df, "s", 5).sum() == 2


def _scored_frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0, 0.1, n) for c in
                       ("log_ratio", "robust_z", "own_jump", "own_premium")})
    df["robust_z"] = df["log_ratio"] / 0.1
    return df


@pytest.mark.parametrize("cls", ["IForestScorer", "ECODScorer"])
def test_ml_scorers_rank_only_high_prices_and_extremes_first(cls):
    import pcad.detect as d
    train = _scored_frame()
    test = _scored_frame(seed=1)
    test.loc[0, ["log_ratio", "robust_z", "own_jump"]] = [0.7, 7.0, 0.7]
    s = getattr(d, cls)().fit(train).score(test)
    assert (s >= 0).all()
    assert (s[test["log_ratio"].to_numpy() <= 0] == 0).all()
    assert s[0] == s.max()
    # fit-time offset: scoring a subset gives the same values
    np.testing.assert_allclose(getattr(d, cls)().fit(train).score(test.iloc[:100]), s[:100],
                               rtol=1e-5)


def test_budget_metrics_and_paired_bootstrap():
    from pcad.evaluate import budget_metrics, paired_bootstrap_diff
    n = 40
    df = pd.DataFrame({"district": np.repeat(["A", "B"], n), "week": WEEKS[0],
                       "is_injected": False, "inj_type": "none"})
    df.loc[[0, 1, 40, 41], ["is_injected", "inj_type"]] = [True, "m20"]
    df["good"] = 0.1
    df.loc[[0, 1, 40, 41], "good"] = 5.0
    df["bad"] = np.linspace(1, 0.01, len(df))
    df.loc[[0, 40], "bad"] = 0.0
    m = budget_metrics(df, "good", ks=(2,))
    assert m.loc[0, "recall"] == 1.0 and m.loc[0, "precision_at_k"] == 1.0
    d, lo, hi = paired_bootstrap_diff(df, "good", "bad", k=2, inj_type="m20", n=200)
    assert d == pytest.approx(0.5) and lo <= d <= hi
