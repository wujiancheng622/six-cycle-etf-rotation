"""Cache dated market and macro data through the project's read-only clients."""
from datetime import datetime, timezone
from hashlib import file_digest
from pathlib import Path
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
PRIOR = WORK / "huatai_2022_etf_reproduction_20261112"
DATA = HERE / "data/tushare"
END = "20261112"
CHUNKS = (("20140101", "20171231"), ("20180101", "20201231"),
          ("20210101", "20231231"), ("20240101", END))
CARRIERS = {
    "growth": "159915.SZ", "gold": "518880.SH", "metals": "159980.SZ",
    "fcf": "159201.SZ", "bond": "511090.SH", "dividend": "515100.SH",
}

sys.dont_write_bytecode = True
sys.path.insert(0, str(WORK / "ah_premium_20261106"))
try:
    from data_access import OfficialData
finally:
    sys.path.pop(0)


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def read_checked(path):
    meta = next((p for p in (path.with_suffix(".json"), path.with_suffix(".meta.json"))
                 if p.exists()), None)
    assert meta and json.loads(meta.read_text(encoding="utf-8"))["sha256"] == sha(path), path
    return pd.read_parquet(path)


def get(endpoint, code=None, **args):
    name = "_".join([code or "all", *map(str, args.values())])
    path = DATA / endpoint / f"{name}.parquet"
    if path.is_file():
        return read_checked(path)
    options = dict(args)
    if code is not None:
        options["ts_code"] = code
    df = OfficialData().query(endpoint, options)
    cap = {"index_daily": 6000, "fund_nav": 2000}.get(endpoint, 2000)
    assert len(df) < cap, (endpoint, options, len(df))
    if code is not None and "ts_code" in df:
        assert set(df.ts_code) <= {code}, (endpoint, options, sorted(set(df.ts_code)))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    write(path.with_suffix(".json"), dict(endpoint=endpoint, args=options, rows=len(df), sha256=sha(path),
                                         retrieved=datetime.now(timezone.utc).isoformat()))
    return df


def dated(endpoint, code, chunks=CHUNKS, date_column="trade_date"):
    table = pd.concat([get(endpoint, code, start_date=a, end_date=b) for a, b in chunks],
                      ignore_index=True)
    assert not table[date_column].duplicated().any(), (endpoint, code)
    return table.set_index(date_column).sort_index()


def main():
    for name, code, date_col in (("repo_daily", "DR007.IB", "trade_date"),
                                 ("shibor", None, "date"), ("shibor_lpr", None, "date")):
        t = dated(name, code, date_column=date_col)
        print(name, len(t), t.index.min(), t.index.max(), list(t.columns), flush=True)
    t = get("cn_pmi", start_m="201301", end_m="202610")
    print("cn_pmi", len(t), t.month.min(), t.month.max(), flush=True)
    for key, code in CARRIERS.items():
        basic = get("fund_basic", code, market="E")
        assert len(basic) == 1
        print(key, code, basic.list_date.iloc[0], flush=True)
        for name in ("fund_daily", "fund_adj"):
            q = dated(name, code)
            print(name, code, len(q), q.index.min(), q.index.max(), flush=True)
        get("fund_div", code)
        get("etf_limit", code, start_date="20240101", end_date=END)
    for code in ("980092.CNI", "980092.SZ", "930955.CSI", "H20955.CSI", "CBA21801.CS",
                 "IMCI.SHFE"):
        q = get("index_daily", code, start_date="20140101", end_date="20171231")
        print("index_daily", code, len(q), flush=True)
        if len(q):
            dated("index_daily", code)


if __name__ == "__main__":
    main()
