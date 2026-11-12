"""Download PriceCatcher months and lookup tables from data.gov.my (skips files already present).

    python scripts/00_download.py              # Jan-Oct 2026: the months the study was built on
    python scripts/00_download.py --months 2   # the current month and the one before (weekly run)

With --months, the current month is always re-downloaded (data.gov.my appends to it daily), and
a current month that is not published yet is skipped instead of failing.
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import date

import _path  # noqa: F401
from pcad import config

BASE = "https://storage.data.gov.my/pricecatcher/"


def recent_months(n: int, today: date) -> list[str]:
    """'YYYY-MM' for the n months ending with today's month, oldest first."""
    months, y, m = [], today.year, today.month
    for _ in range(n):
        months.append(f"{y}-{m:02d}")
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return months[::-1]


def fetch(name: str, force: bool = False, optional: bool = False) -> None:
    dest = config.RAW_DIR / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    for attempt in range(5):  # the host sometimes resets connections
        try:
            urllib.request.urlretrieve(BASE + name, dest)
            print("downloaded", name)
            return
        except urllib.error.HTTPError as e:
            if e.code == 404 and optional:
                print("not published yet:", name)
                return
            print(f"retry {attempt + 1} for {name}: {e}")
        except OSError as e:
            print(f"retry {attempt + 1} for {name}: {e}")
        time.sleep(2 ** attempt)
    sys.exit(f"failed to download {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, help="only the latest N months, ending this month")
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.months:
        months = recent_months(args.months, date.today())
        for old in config.RAW_DIR.glob("pricecatcher_*.parquet"):  # keep the build to these months only
            if old.stem.removeprefix("pricecatcher_") not in months:
                old.unlink()
    else:
        months = [f"2026-{m:02d}" for m in range(1, 11)]
    for month in months:
        is_current = args.months and month == months[-1]
        fetch(f"pricecatcher_{month}.parquet", force=bool(is_current), optional=bool(is_current))
    for name in ("lookup_item.parquet", "lookup_premise.parquet"):
        fetch(name, force=bool(args.months))
