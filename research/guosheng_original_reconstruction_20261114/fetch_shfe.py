"""Snapshot the exchange-published prelisting history of the metals ETF benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


HERE = Path(__file__).resolve().parent
HISTORY = HERE.parent / "guosheng_six_regime_long_20261112" / "data" / "returns_gs.parquet"
SOURCE = "https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{date}.dat"
LISTING = "2019-12-24"
FIELDS = ("date", "close", "previous_close", "open", "source", "sha256")


def daily_index(day):
    stamp = day.replace("-", "")
    url = SOURCE.format(date=stamp)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    for wait in (0.05, 0.5, 1.5, 3):
        time.sleep(wait)
        try:
            with urlopen(request, timeout=12) as response:
                payload = response.read()
            break
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            failure = error
    else:
        return day, str(failure)
    try:
        report = json.loads(payload.decode("utf-8-sig"))
        assert str(report["report_date"]) == stamp
        index, = report["o_curmetalindex"]
        assert str(index["TRADINGDAY"]) == stamp
        close = float(index["CLOSEPRICE"])
        previous = float(index["PRECLOSEPRICE"])
        opening = float(index["OPENPRICE"])
        assert 0 < min(close, previous, opening) <= max(close, previous, opening) < 100_000
    except (AssertionError, ValueError, KeyError, TypeError) as error:
        return day, f"{type(error).__name__}: {error}"
    return day, dict(date=day, close=close, previous_close=previous, open=opening,
                     source=url, sha256=hashlib.sha256(payload).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", nargs="+", help="Check selected YYYY-MM-DD sessions without writing")
    args = parser.parse_args()
    sessions = pd.read_parquet(HISTORY, columns=[]).index
    days = args.sample or sessions[sessions <= LISTING].tolist()
    for day in days:
        datetime.strptime(day, "%Y-%m-%d")
    output = HERE / "shfe_metals_index.csv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = {}
        if output.is_file() and not args.sample:
            with output.open(encoding="utf-8-sig", newline="") as file:
                found = {row["date"]: row for row in csv.DictReader(file)}
        misses = []
        for day, item in pool.map(daily_index, (day for day in days if day not in found)):
            if isinstance(item, str):
                misses.append((day, item))
            else:
                found[day] = item
            if (len(found) + len(misses)) % 250 == 0:
                print(f"SHFE {len(found)} confirmed, {len(misses)} missing", flush=True)
    if args.sample:
        print(json.dumps(found, ensure_ascii=False))
    else:
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(found[day] for day in days if day in found)
        temporary.replace(output)
    print(f"SHFE {len(found)} of {len(days)} confirmed. Missing: {misses}", flush=True)
    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
