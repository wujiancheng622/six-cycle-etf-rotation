"""Extend official SHFE index observations through the common data cutoff."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "guosheng_original_reconstruction_20261114"))
from fetch_shfe import daily_index


def main():
    folder = HERE / "input"
    folder.mkdir(exist_ok=True)
    output = folder / "shfe_index.csv"
    original = ROOT / "guosheng_original_reconstruction_20261114/shfe_metals_index.csv"
    rows = pd.read_csv(output if output.exists() else original)
    existing = set(rows.date)
    days = pd.read_parquet(ROOT / "guosheng_six_regime_long_20261112/data/returns_gs.parquet", columns=[]).index
    missing = [day for day in days if day not in existing]
    fetched, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (day, value) in enumerate(pool.map(daily_index, missing)):
            if isinstance(value, str):
                failures.append(dict(date=day, error=value))
            else:
                fetched.append(value)
            if (i + 1) % 250 == 0:
                print(f"Official metals overlap {i+1}/{len(missing)}, failed {len(failures)}", flush=True)
    pd.concat([rows, pd.DataFrame(fetched)], ignore_index=True).sort_values("date").to_csv(output, index=False)
    pd.DataFrame(failures).to_csv(folder / "shfe_failures.csv", index=False)
    assert not failures, failures[:4]
    print(f"Official metals complete {len(rows)+len(fetched)} rows", flush=True)


if __name__ == "__main__":
    main()
