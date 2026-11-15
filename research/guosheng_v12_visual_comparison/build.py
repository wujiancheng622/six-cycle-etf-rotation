"""Render a standalone comparison from matched archived accounts."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "guosheng_final_decision_20261114"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
import study as s


def load_pair(*names):
    accounts = []
    for name in names:
        path = SOURCE / "results/accounts" / name
        meta = json.loads((path / "meta.json").read_text())
        assert meta["capital"] == 500000
        accounts.append(SimpleNamespace(ledger=pd.read_parquet(path / "ledger.parquet"), base_date=meta["base_date"]))
    return accounts


def measure(book, lo, hi):
    ledger = book.ledger.loc[lo:hi]
    first = book.ledger.index.get_loc(ledger.index[0])
    start = book.base_date if first == 0 else book.ledger.index[first - 1]
    base = 500000 if first == 0 else book.ledger.nav.iloc[first - 1]
    years = (pd.Timestamp(ledger.index[-1]) - pd.Timestamp(start)).days / 365.25
    nav = np.r_[base, ledger.nav.to_numpy()]
    ret = nav[1:] / nav[:-1] - 1
    dd = float((1 - nav / np.maximum.accumulate(nav)).max())
    vol = float(ret.std(ddof=1) * np.sqrt(252))
    cagr = float((nav[-1] / base)**(1 / years) - 1)
    return dict(total=float(nav[-1] / base - 1), cagr=cagr, vol=vol, mdd=dd,
                sharpe=float(ret.mean()*252/vol) if vol else None, calmar=cagr/dd if dd else None,
                turnover=float((ledger.notional/nav[:-1]).sum()/years/2), fees=float(ledger.cost.sum()),
                cash=float((ledger.cash/ledger.nav).mean()), start=ledger.index[0], end=ledger.index[-1])


def dataset(key, label, note, accounts, calendar):
    a, b = accounts
    dates = a.ledger.index
    assert dates.equals(b.ledger.index) and a.base_date == b.base_date
    assert dates.is_unique and dates.is_monotonic_increasing
    annual = []
    for year in sorted({d[:4] for d in dates}):
        annual.append(dict(label=year, a=measure(a, year, year+"-12-31"), b=measure(b, year, year+"-12-31")))
    if dates[0][5:7] != "01":
        annual[0]["label"] += "（部分）"
    if dates[-1][5:7] != "12":
        annual[-1]["label"] += "（截至" + dates[-1][5:] + "）"
    assert np.isclose(np.prod([1+x["a"]["total"] for x in annual]), a.ledger.nav.iloc[-1]/500000)
    assert np.isclose(np.prod([1+x["b"]["total"] for x in annual]), b.ledger.nav.iloc[-1]/500000)
    trades = ["2014–2016", "2017–2019", "2020–2022", "2023–2025"]
    blocks = []
    for span in trades:
        first, last = span.split("–")
        lo, hi = first+"-01-01", last+"-12-31"
        if dates[0] <= first+"-01-07" and dates[-1] >= last+"-12-24":
            blocks.append(dict(label=span,a=measure(a,lo,hi),b=measure(b,lo,hi)))
    if dates[0] <= "2026-07-01" < dates[-1]:
        blocks.append(dict(label="2026年7月起",a=measure(a,"2026-07-01",dates[-1]),b=measure(b,"2026-07-01",dates[-1])))
    dates_out = [a.base_date,*dates]
    nav_a = np.r_[500000,a.ledger.nav.to_numpy()]/500000
    nav_b = np.r_[500000,b.ledger.nav.to_numpy()]/500000
    months = pd.DatetimeIndex(dates).to_period("M")
    ends = pd.Series(dates,index=months).groupby(level=0).last()
    month_a = a.ledger.nav.reindex(ends).to_numpy()
    month_b = b.ledger.nav.reindex(ends).to_numpy()
    monthly = [dict(date=str(p),value=float(100*(x-y))) for p,x,y in zip(ends.index,
               month_a/np.r_[500000,month_a[:-1]]-1,month_b/np.r_[500000,month_b[:-1]]-1)]
    rolling = {}
    complete = ends.iloc[:-1]  # Frozen data stop on November 12, mid-month.
    first_month = dates[0][:7]
    if key == "long" or dates[0] != next(d for d in calendar if d.startswith(first_month)):
        complete=complete.iloc[1:]
    for length in (12,36,60):
        windows = []
        for j in range(length-1,len(complete)):
            lo = complete.index[j-length+1].start_time.date().isoformat()
            hi = complete.iloc[j]
            x,y = measure(a,lo,hi), measure(b,lo,hi)
            windows.append(dict(date=hi, a=x, b=y))
        rolling[str(length)] = windows
    rel = nav_a/nav_b
    streak = longest = 0
    begin = end = None
    high = rel[0]
    for j,value in enumerate(rel):
        if value >= high:
            streak=0
            high=value
        else:
            streak+=1
        if streak>longest:
            longest=streak
            begin=dates_out[j-streak]
            end=dates_out[j]
    return dict(key=key,label=label,note=note,dates=dates_out,
                a=nav_a.round(8).tolist(),b=nav_b.round(8).tolist(),
                overview=dict(a=measure(a,dates[0],dates[-1]),b=measure(b,dates[0],dates[-1])),
                annual=annual,blocks=blocks,rolling=rolling,monthly=monthly,
                relative_drawdown=dict(days=longest,begin=begin,end=end))


def main():
    main_pair=load_pair("monthly_2014-01-02_9bp","V12_2014-01-02_9bp")
    quotes=load_pair("actual_monthly_2025-06-04_9bp_cash0","actual_V12_2025-06-04_9bp_cash0")
    listed=load_pair("eligible_GS_monthly_2014-01-02_9bp","eligible_V12_2014-01-02_9bp")
    data=s.load()
    start="2013-04-12"
    early=[s.run(data,s.Case("monthly"),start=start),s.benchmark(data,start=start)]
    entries=[
        dataset("main","2014 主回放","2014-01-02 起 · 早期含代理 · 金额模型",main_pair,main_pair[0].ledger.index),
        dataset("long","2013 最长共同历史","2013-04-12 起 · 同一现行口径独立建账 · 早期含代理",early,main_pair[0].ledger.index),
        dataset("quotes","真实 ETF 报价模拟","2025-06-04 起 · 开盘报价／整手／成交量限制",quotes,main_pair[0].ledger.index),
        dataset("listed","仅上市后 ETF 历史","2014-01-02 起 · 双方不足 63 日自身历史的资产资金位留现金",listed,main_pair[0].ledger.index),
    ]
    reference=pd.read_csv(SOURCE/"results/main_comparison.csv")
    for strategy,name in (("a","monthly_2014-01-02_9bp"),("b","V12_2014-01-02_9bp")):
        row=reference.loc[reference.case.eq(name)&reference.period.eq("full")].iloc[0]
        for key in ("cagr","vol","mdd","total","fees","mean_cash"):
            np.testing.assert_allclose(entries[0]["overview"][strategy]["cash" if key=="mean_cash" else key],row[key],atol=1e-8)
    output=HERE/"国盛与V12全历史对照.html"
    script=Path(os.environ["GS_COMPARE_PLOTLY_JS"]).read_text(encoding="utf8")
    template=(HERE/"dashboard.template.html").read_text(encoding="utf8")
    payload=json.dumps(entries,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    output.write_text(template.replace("/* @PLOTLY@ */",script).replace("/* @DATA@ */",lambda_value(payload)),encoding="utf8")
    for entry in entries:
        csv=[]
        for row in entry["annual"]:
            csv.append({"year":row["label"],**{"国盛_"+k:v for k,v in row["a"].items()},**{"V12_"+k:v for k,v in row["b"].items()}})
        pd.DataFrame(csv).to_csv(HERE/("逐年指标_"+entry["key"]+".csv"),index=False,encoding="utf-8-sig")
    print(json.dumps([dict(key=x["key"], overview=x["overview"], annual=[(y["label"],y["a"]["total"],y["b"]["total"]) for y in x["annual"]],
                           windows={k:(len(v),sum(z["a"]["cagr"]>z["b"]["cagr"] for z in v)) for k,v in x["rolling"].items()},
                           relative=x["relative_drawdown"]) for x in entries],ensure_ascii=False))


def lambda_value(text):
    return text.replace("</script","<\\/script")


if __name__=="__main__":main()
