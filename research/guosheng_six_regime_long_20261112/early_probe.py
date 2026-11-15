"""Isolated discovery and data snapshots for the early tail."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
import json
import re
import sys

import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "guosheng_six_regime_20261112"
sys.path.insert(0, str(BASE))
import pbc_loans as pbc
import provider_data as provider
sys.path.pop(0)

DATA = HERE / "data"
MFD = "https://mfd.ru/calendar/details/?id=502"
JIN = "https://datacenter-api.jin10.com/reports/list_v2"
OLD = HERE.parent / "review_runs/20261109_v12_2007_holdout_v1/review_results/rebuilt_inputs/main_returns.csv"


def snapshot(url, path, headers=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with urlopen(Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"}), timeout=18) as r:
            data = r.read()
        assert len(data) > 100, (url, len(data))
        path.write_bytes(data)
    return path.read_bytes()


def ts(endpoint, code=None, **args):
    original = provider.DATA
    try:
        provider.DATA = DATA / "tushare"
        return provider.get(endpoint, code, **args)
    finally:
        provider.DATA = original


def quotes():
    todo = [
        ("trade_cal", None, dict(exchange="SSE", start_date="20120101", end_date="20140115")),
        ("fund_daily", "159915.SZ", dict(start_date="20111201", end_date="20140115")),
        ("fund_daily", "518880.SH", dict(start_date="20130101", end_date="20140115")),
        ("sge_daily", "Au99.99", dict(start_date="20120101", end_date="20140115")),
        ("index_daily", "980092.SZ", dict(start_date="20120101", end_date="20140116")),
        ("index_daily", "H20955.CSI", dict(start_date="20120101", end_date="20140116")),
        ("shibor", None, dict(start_date="20100101", end_date="20131231")),
        ("cn_pmi", None, dict(start_m="200901", end_m="201212")),
    ]
    for api, code, args in todo:
        try:
            t = ts(api, code, **args)
        except Exception as e:
            print("unavailable", api, code, type(e).__name__, flush=True)
            continue
        cols = [k for k in ("cal_date", "trade_date", "date", "month") if k in t]
        print(api, code, len(t), [(t[k].min(), t[k].max()) for k in cols], list(t.columns)[:9],
              flush=True)


def archives():
    def read(number):
        page = f"11871-{number}.html"
        try:
            if (pbc.ARCHIVE / "lists" / page).exists():
                raw = (pbc.ARCHIVE / "lists" / page).read_bytes()
            else:
                raw = snapshot(urljoin(pbc.ROOT, page), DATA / "pbc_lists" / page)
            rows = pbc.INDEX_PATTERN.findall(raw.decode("utf-8"))
            return [(number, urljoin(pbc.ROOT, u), pbc.clean(t), d) for u,t,d in rows]
        except (OSError, UnicodeError) as e:
            print(number, type(e).__name__, getattr(e, "code", None), flush=True)
            return []
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = [row for page in pool.map(read, range(30, 46)) for row in page]
    t = pd.DataFrame(rows, columns=["page", "url", "title", "released"])
    t["month"] = t.title.map(pbc.period)
    t.to_csv(DATA / "pbc_early_index.csv", index=False, encoding="utf-8-sig")
    print(t.groupby("page").released.agg(["min", "max", "count"]).to_string(), flush=True)
    print(t[t.month.notna() & t.month.lt("2013-01")].to_string(index=False), flush=True)


def loan_pages():
    idx = pd.read_csv(DATA / "pbc_early_index.csv", dtype=str, encoding="utf-8-sig").dropna(subset="month")
    idx = (idx[idx.month.between("2009-11", "2012-12")].sort_values(["month", "released"])
           .drop_duplicates("month"))
    pbc.ARCHIVE = DATA / "pbc"
    expected = pd.period_range("2009-11", "2012-12", freq="M").astype(str)
    assert list(idx.month) == list(expected), (set(expected) - set(idx.month))
    with ThreadPoolExecutor(max_workers=6) as pool:
        t = pd.DataFrame(pool.map(pbc.announcement, idx[["month", "released", "title", "url"]].to_dict("records")))
    t.to_csv(pbc.ARCHIVE / "raw_loans.csv", index=False, encoding="utf-8-sig")
    print(t[["month","released","basis","hint","household_reported_yi","firm_reported_yi"]].to_string(index=False),
          flush=True)
    for r in t[t.basis.isin(["unparsed", "unknown"])].itertuples():
        raw = (pbc.ARCHIVE / "releases" / f"{r.month}.html").read_text(encoding="utf-8")
        lines = re.sub(r"\s+", "", pbc.clean(raw))
        idx = lines.find("住户贷款")
        print(r.month, ascii(lines[max(0, idx-120):idx+310]), flush=True)


class Table(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside = False
        self.cells = []
        self.rows = []
        self.part = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cells = []
        if tag in ("td", "th"):
            self.part = []
            self.inside = True

    def handle_data(self, data):
        if self.inside:
            self.part.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.cells.append(" ".join(" ".join(self.part).split()))
            self.inside = False
        if tag == "tr" and self.cells:
            self.rows.append(self.cells)


def pmi():
    try:
        raw = snapshot(MFD, DATA / "mfd_hsbc_final.html")
        parser = Table()
        parser.feed(raw.decode("utf-8"))
        rows = [r for r in parser.rows if r and re.match(r"\d{2}\.\d{2}\.20\d{2}", r[0])]
        (DATA / "mfd_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("MFD", len(rows), rows[-6:], flush=True)
    except (OSError, UnicodeError) as e:
        print("MFD unavailable", type(e).__name__, getattr(e, "code", None), flush=True)
    end = "2013-01-01"
    hdr = {"User-Agent": "Mozilla/5.0", "x-app-id": "rU6QIu7JHe2gOUeR",
           "x-csrf-token": "x-csrf-token", "x-version": "1.0.0"}
    for _ in range(5):
        try:
            url = JIN+"?"+urlencode(dict(category="ec", attr_id=73, max_date=end))
            raw = snapshot(url, DATA / "jin10" / f"{end}.json", hdr)
            values = json.loads(raw)["data"]["values"]
            print("Jin10", end, len(values), values[0] if values else None, values[-1] if values else None,
                  flush=True)
            if not values or values[-1][0] >= end or values[-1][0] < "2009-01-01":
                break
            end = values[-1][0]
        except (OSError, TypeError, KeyError, ValueError) as e:
            print("Jin10 unavailable", type(e).__name__, getattr(e, "code", None), flush=True)
            break


if __name__ == "__main__":
    {"quotes": quotes, "archives": archives, "pmi": pmi, "loans": loan_pages}[sys.argv[1]]()
