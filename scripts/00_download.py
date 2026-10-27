"""Download PriceCatcher months and lookup tables from data.gov.my (skips files already present)."""
import sys
import time
import urllib.request

import _path  # noqa: F401
from pcad import config

BASE = "https://storage.data.gov.my/pricecatcher/"
names = [f"pricecatcher_2026-{m:02d}.parquet" for m in range(1, 11)]
names += ["lookup_item.parquet", "lookup_premise.parquet"]

config.RAW_DIR.mkdir(parents=True, exist_ok=True)
for name in names:
    dest = config.RAW_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        continue
    for attempt in range(5):  # the host sometimes resets connections
        try:
            urllib.request.urlretrieve(BASE + name, dest)
            print("downloaded", name)
            break
        except OSError as e:
            print(f"retry {attempt + 1} for {name}: {e}")
            time.sleep(2 ** attempt)
    else:
        sys.exit(f"failed to download {name}")
