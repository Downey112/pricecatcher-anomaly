import importlib.util
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("download", ROOT / "scripts" / "00_download.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_recent_months_within_a_year():
    assert download.recent_months(2, date(2026, 11, 12)) == ["2026-10", "2026-11"]


def test_recent_months_across_new_year():
    assert download.recent_months(3, date(2027, 1, 5)) == ["2026-11", "2026-12", "2027-01"]
