"""Price-to-NAV diagnostics and dated long-return benchmarks."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
GS_ROOT = WORK / "guosheng_six_regime_20261112"
LONG = WORK / "guosheng_six_regime_long_20261112"
OLD_STUDY = WORK / "review_runs" / "20261111_v12_typical_return_independent_v1"
OLD_NAV = WORK / "review_runs" / "20261111_v12_normal_regimes_v1" / "input"
FOREIGN = WORK / "review_runs" / "20261111_us_equity_20y_v1" / "output" / "daily_total_return_series.csv"
EARLY = WORK / "review_runs" / "20261109_v12_2007_holdout_v1" / "review_results" / "rebuilt_inputs" / "main_returns.csv"
CLOSE = "2026-11-04"
STARTS = ("2014-01-02", "2020-07-30", "2023-01-03", "2025-11-06")
QDII = {"nasdaq": ("513100.SH","QQQ"), "sp500": ("513500.SH","SPY")}

sys.dont_write_bytecode = True
sys.path.insert(0, str(LONG))
import replay_long as prior
from verify_long import PriorToCash
sys.path.remove(str(LONG))
PROVIDER = prior.provider
CARRIERS = PROVIDER.CARRIERS


def years(start,end):
    return (pd.Timestamp(end)-pd.Timestamp(start)).days/365.25


def stamp(s):
    return pd.to_datetime(s, format="%Y%m%d").dt.strftime("%Y-%m-%d")


def cached(endpoint,code):
    return HERE / "inputs" / "tushare" / endpoint / f"{code}_20060101_20261104.parquet"


def gs_nav_data(code):
    files = sorted((HERE/"inputs/tushare/fund_nav").glob(f"{code}_*.parquet"))
    assert files, code
    raw = pd.concat([PROVIDER.read_checked(p) for p in files])
    d = raw.set_index(stamp(raw.nav_date)).sort_index()
    assert d.index.is_unique and d[["unit_nav","adj_nav"]].notna().all().all()
    assert (d[["unit_nav","adj_nav"]]>0).all().all()
    assert (d.ann_date.astype(str)>=d.nav_date.astype(str)).all()
    return d


def gs_raw_closes(key,code):
    x = PROVIDER.dated("fund_daily",code).close
    if key in ("growth","gold"):
        prefix = "20111201" if key=="growth" else "20130101"
        older = LONG / f"data/tushare/fund_daily/{code}_{prefix}_20140115.parquet"
        # Match the original splice; the two archived requests overlap in Jan 2014.
        earlier = PROVIDER.read_checked(older).set_index("trade_date").close.sort_index()
        shared = earlier.index.intersection(x.index)
        np.testing.assert_array_equal(earlier.loc[shared], x.loc[shared])
        x = pd.concat([earlier.loc[earlier.index<x.index[0]],x])
    idx = pd.to_datetime(x.index,format="%Y%m%d").strftime("%Y-%m-%d")
    assert idx.is_unique
    return x.set_axis(idx)


def gs_normal(raw, coverage):
    out = raw.copy()
    panels = {}
    for key,code in CARRIERS.items():
        d = gs_nav_data(code)
        idx = raw.index
        values = d.adj_nav.reindex(idx)
        nav_r = values.pct_change(fill_method=None)
        eligible = idx>=coverage["actual_return_from"][key]
        use = eligible & nav_r.notna()
        out.loc[use,key] = nav_r.loc[use]
        px = gs_raw_closes(key,code).reindex(idx)
        unit = d.unit_nav.reindex(idx)
        premium = px/unit-1.
        row = pd.DataFrame({"valuation_nav":unit, "close":px,"adj_nav":values,
                            "premium_close_to_nav":premium, "nav_return":nav_r,
                            "market_return":raw[key], "nav_used":use, "listed_return":eligible},
                           index=idx)
        assert row.loc[use,"nav_return"].abs().max()<.25
        panels[key] = row
    assert np.isfinite(out.to_numpy()).all()
    return out,panels


def v12_nav_repair(code,ticker,day,foreign):
    path = OLD_NAV / f"fund_nav_{code}.csv"
    raw = pd.read_csv(path,dtype={"nav_date":str,"ann_date":str})
    raw = raw.set_index(stamp(raw.nav_date)).sort_index()
    x = raw[(raw.ann_date<day.replace("-",""))&(raw.index<day)].iloc[-1]
    current = foreign.index.searchsorted(day,side="left")-1
    base = foreign.index.searchsorted(x.name,side="right")-1
    assert current>=base>=0 and float(x.adj_nav)>0
    col = ticker+"_CNY"
    result = float(x.adj_nav*foreign[col].iloc[current]/foreign[col].iloc[base])
    return result, x.name, x.ann_date


def v12_normal(raw):
    out = raw.copy()
    panels, repairs = {}, []
    foreign = pd.read_csv(FOREIGN,index_col="date")
    for key,(code,ticker) in QDII.items():
        p = pd.read_csv(OLD_STUDY/"output"/f"premium_{code}.csv",index_col="date").loc[:CLOSE]
        assert p.index.is_unique
        fair = (p.fair_estimate*p.adjustment_factor).reindex(raw.index)
        missing = fair.loc[fair.first_valid_index():].index[fair.loc[fair.first_valid_index():].isna()]
        for day in missing:
            assert day in ("2022-01-13","2022-03-29"),(key,day)
            fair.loc[day],last_nav,last_announcement = v12_nav_repair(code,ticker,day,foreign)
            repairs.append(dict(asset=key,date=day,prior_nav=last_nav,prior_ann=last_announcement,
                                fair_adj=float(fair.loc[day])))
        for day in ("2020-07-29","2026-11-04"):
            shadow, _, _ = v12_nav_repair(code,ticker,day,foreign)
            np.testing.assert_allclose(shadow,fair.loc[day],rtol=1e-12,atol=1e-12)
        nav_r = fair.pct_change(fill_method=None)
        use = nav_r.notna()
        out.loc[use,key] = nav_r.loc[use]
        p = p.reindex(raw.index)
        panels[key] = pd.DataFrame({"valuation_nav":p.fair_estimate,"close":p.close,"adj_nav":fair,
                                    "premium_close_to_nav":p.premium_known_close_estimate,
                                    "nav_return":nav_r,"market_return":raw[key],"nav_used":use,
                                    "listed_return":raw.index>=fair.first_valid_index()},index=raw.index)
    assert np.isfinite(out.to_numpy()).all()
    return out,panels,repairs


def premiums(panels,strategy):
    windows = (("listing",None),("2014 onward","2014-01-02"),("since 2020-07","2020-07-30"),
               ("since 2023","2023-01-03"),("since 2025-11","2025-11-06"),("2026 YTD","2026-01-05"))
    rows=[]
    for k,t in panels.items():
        for label,start in windows:
            z=t.loc[start:CLOSE]
            p=z.premium_close_to_nav.dropna()
            if len(p)<2:
                continue
            usable=z.loc[z.nav_used]
            assert len(usable)>0 and z.index[-1]==CLOSE
            first,last=usable.index[[0,-1]]
            prior=t.index[t.index.get_loc(first)-1]
            paired=z.loc[first:last].loc[z.loc[first:last].nav_used]
            market=np.log1p(paired.market_return).sum()
            nav=np.log1p(paired.nav_return).sum()
            days=years(prior,last)
            r=dict(strategy=strategy,asset=k,interval=label,first_return=first,last_return=last,
                   usable_nav_days=len(usable),listed_no_nav_days=int((z.listed_return&~z.nav_used).sum()),
                   paired_premium_days=len(p),premium_q25=p.quantile(.25),premium_median=p.median(),
                   premium_q75=p.quantile(.75),premium_p90=p.quantile(.9),premium_above_5pct=p.gt(.05).mean(),
                   premium_last_date=p.index[-1],premium_last=p.iloc[-1],
                   market_annual_on_matched_dates=np.expm1(market/days),
                   nav_annual_on_matched_dates=np.expm1(nav/days),annual_log_gap=(market-nav)/days)
            if prior in t.index and pd.notna(t.at[prior,"premium_close_to_nav"]):
                r["premium_at_return_base"] = float(t.at[prior,"premium_close_to_nav"])
            rows.append(r)
    return rows


def annual_from_levels(x,end):
    v=x.loc[:end].dropna()
    assert len(v)>400 and v.index.is_monotonic_increasing and v.index.is_unique
    return dict(start=v.index[0],end=v.index[-1],observations=len(v),
                annual=float(np.expm1(np.log(v.iloc[-1]/v.iloc[0])/years(v.index[0],v.index[-1]))))


def anchors(gs_no_premium):
    old=json.loads((OLD_STUDY/"output/receipt.json").read_text(encoding="utf-8"))
    growth=PROVIDER.read_checked(cached("index_daily","399606.SZ")).sort_values("trade_date")
    growth=growth.set_index(stamp(growth.trade_date)).close
    a=pd.read_csv(EARLY,index_col=0).metals
    b=gs_no_premium.metals
    a=pd.concat([a.loc[a.index<b.index[0]],b])
    assert a.index.is_unique and np.isfinite(a).all()
    metal=(1+a).cumprod()
    metal.loc["2006-01-04"]=1.
    metal=metal.sort_index()
    bond=pd.Series(json.loads((GS_ROOT/"source/chinabond_30y_wealth.json").read_text(encoding="utf-8"))["CFZS_00"],
                   dtype=float)
    bond=bond.set_axis(pd.to_datetime(pd.to_numeric(bond.index),unit="ms",utc=True)
                      .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")).sort_index()
    bond=bond.loc["2011-01-01":]
    assert bond.index.is_unique
    source={"growth":(growth,"399606.SZ official ChiNext total return"),
            "metals":(metal,"Same-contract futures proxy, then 159980 fund adjusted NAV"),
            "bond":(bond,"ChinaBond CBA21801 wealth index")}
    records=[]
    for label,end,oldkey in (("whole",CLOSE,"whole_20y_mean"),("pre2020","2019-12-31","pre2020_mean")):
        for asset,(data,origin) in source.items():
            records.append(dict(strategy="GS",asset=asset,anchor=label,source=origin,
                                **annual_from_levels(data,end)))
        for asset,annual in old["anchors"][oldkey].items():
            for name in ("V12","GS") if asset=="gold" else ("V12",):
                records.append(dict(strategy=name,asset=asset,anchor=label,
                                    source="Earlier audited V12 study, RMB spot/ETF for gold or QQQ/SPY with dividends and FX",
                                    start="2006-01",end=end,annual=annual))
    return pd.DataFrame(records)


def drift_normal(frame,start,targets,*,down_only=False):
    i=frame.index.get_loc(start)
    base=frame.index[i-1]
    days=pd.Series(pd.to_datetime(frame.index),index=frame.index).diff().dt.days.fillna(0)/365.25
    run_years=years(base,frame.index[-1])
    np.testing.assert_allclose(days.loc[start:].sum(),run_years,rtol=0,atol=1e-13)
    adjusted=frame.copy()
    rows=[]
    for asset,rate in targets.items():
        x=np.log1p(frame[asset])
        original_log=x.loc[start:].sum()/run_years
        delta=np.log1p(rate)-original_log
        if down_only:
            delta=min(delta,0.)
        adjusted[asset]=np.expm1(x+delta*days)
        verified=np.log1p(adjusted.loc[start:,asset]).sum()/run_years
        np.testing.assert_allclose(verified,original_log+delta,rtol=0,atol=1e-12)
        rows.append(dict(asset=asset,observed=np.expm1(original_log),historical_anchor=rate,
                         used_annual=np.expm1(verified),annual_log_shift=delta))
    return adjusted,rows
