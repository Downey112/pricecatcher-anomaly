import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcad.conformal import benjamini_hochberg, calibration_sample, p_values


def test_p_value_formula():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    # 3.0: two calibration scores are >= 3 -> (1 + 2) / 5
    assert p_values(np.array([3.0]), cal)[0] == pytest.approx(3 / 5)
    assert p_values(np.array([10.0]), cal)[0] == pytest.approx(1 / 5)  # beyond everything
    assert p_values(np.array([0.0]), cal)[0] == pytest.approx(1.0)


def test_false_alarm_rate_is_controlled_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cal = calibration_sample(rng.standard_normal(50_000))
    new = rng.standard_normal(200_000)
    for alpha in (0.05, 0.01, 0.001):
        rate = (p_values(new, cal) <= alpha).mean()
        assert rate <= alpha * 1.15  # within sampling noise of the guarantee


def test_calibration_sample_is_sorted_and_capped():
    cal = calibration_sample(np.arange(1000)[::-1], n=100)
    assert len(cal) == 100 and np.all(np.diff(cal) >= 0)


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6])
    assert benjamini_hochberg(p, q=0.05).tolist() == [True, True, False, False, False]
    assert not benjamini_hochberg(np.array([0.5, 0.9]), q=0.05).any()
