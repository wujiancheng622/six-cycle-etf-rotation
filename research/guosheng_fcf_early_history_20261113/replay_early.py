"""Extend the original cash-flow exclusion test using dated pre-2013 data."""
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT/"guosheng_six_regime_long_20261112"
CUTOFF = ROOT/"guosheng_fcf_cutoff_20261113"
DEPENDENCE = ROOT/"guosheng_v12_return_dependence_20261113"
HOLDOUT = ROOT/"review_runs/20261109_v12_2007_holdout_v1"
ORIGINAL = ROOT/"guosheng_six_regime_20261112"
DATA = PARENT/"data"
sys.dont_write_bytecode = True
sys.path[:0] = [str(CUTOFF),str(PARENT)]
from replay_fcf_cutoff import AvailableRotation, run_variant, trailing
from replay_long import END, KEYS, SLIP, engine, log_attribution, stats
from diagnostics import window
from early_inputs import macro_inputs, pbc

RULE_DATE = "2024-08-15"
YEARTODATE = {3,6,9,12}
ORIGINAL_NONMAN_2009 = (55.7,41.9,55.7,59.,60.9,55.4,58.9,59.9,58.9,62.1,52.3,57.5)
PROXY_RELEASES = {"dividend":"2017-05-26","bond":"2021-07-09"}
SOURCE = {
    "legacy_gs_returns": DATA/"returns_gs.parquet",
    "legacy_v12_returns": DATA/"returns_v12.parquet",
    "legacy_phases": DATA/"macro.parquet",
    "legacy_credit": DATA/"credit_release_sources.csv",
    "legacy_credit_values": DATA/"credit_monthly.csv",
    "legacy_pmi": DATA/"pmi_as_reported.csv",
    "cn_pmi": DATA/"tushare/cn_pmi/all_200901_201212.parquet",
    "hsbc_pmi_calendar": DATA/"mfd_rows.json",
    "credit_2009": HERE/"sources/credit_2009_extraction.csv",
    "calendar": HOLDOUT/"raw/trade_cal_SSE_20060101.csv",
    "early_growth": DEPENDENCE/"inputs/tushare/index_daily/399606.SZ_20060101_20261104.parquet",
    "early_growth_etf": DATA/"tushare/fund_daily/159915.SZ_20111201_20140115.parquet",
    "early_gold": HOLDOUT/"raw/sge_daily_Au99.99_20100101.csv",
    "early_gold_audited": HOLDOUT/"review_results/rebuilt_inputs/main_returns.csv",
    "early_dividend": DEPENDENCE/"inputs/tushare/index_daily/H20955.CSI_20060101_20261104.parquet",
    "early_bond": ORIGINAL/"source/chinabond_30y_wealth.json",
}


def iso_date(series):
    return pd.to_datetime(series,format="%Y%m%d").dt.strftime("%Y-%m-%d")


def dated_close(file):
    q = pd.read_parquet(file).sort_values("trade_date")
    s = q.set_index(iso_date(q.trade_date)).close.astype(float)
    assert s.index.is_unique
    return s


def inputs():
    benchmark = pd.read_parquet(SOURCE["legacy_v12_returns"])
    historical = pd.read_parquet(SOURCE["legacy_gs_returns"])
    cal = pd.read_csv(SOURCE["calendar"],dtype={"cal_date":str})
    sessions = pd.Index(iso_date(cal.loc[cal.is_open.eq(1),"cal_date"]).sort_values())
    sessions = sessions[(sessions >= "2011-01-04") & (sessions < historical.index[0])]
    assert sessions.difference(benchmark.index).empty
    g_idx = dated_close(SOURCE["early_growth"]).pct_change(fill_method=None)
    g_etf = dated_close(SOURCE["early_growth_etf"]).pct_change(fill_method=None)
    dividend = dated_close(SOURCE["early_dividend"]).pct_change(fill_method=None)
    marks = json.loads(SOURCE["early_bond"].read_text(encoding="utf-8"))["CFZS_00"]
    bond = pd.Series(marks,dtype=float)
    bond.index = pd.to_datetime(pd.to_numeric(bond.index),unit="ms",utc=True).tz_convert(
        "Asia/Shanghai").strftime("%Y-%m-%d")
    bond = bond.sort_index().reindex(sessions,method="ffill").pct_change(fill_method=None)
    au = pd.read_csv(SOURCE["early_gold"],dtype={"trade_date":str}).sort_values("trade_date")
    gold = au.set_index(iso_date(au.trade_date)).close.astype(float)
    assert gold.index.is_unique
    gold = gold.reindex(sessions).pct_change(fill_method=None)
    audited = pd.read_csv(SOURCE["early_gold_audited"],index_col=0)
    gaps = gold.index[gold.isna()].tolist()
    gold = gold.combine_first(audited.gold).reindex(sessions)
    early = pd.DataFrame({"growth":g_etf.combine_first(g_idx),"gold":gold,
                          "metals":benchmark.metals,"fcf":0.,"bond":bond,
                          "dividend":dividend},index=sessions).loc[:,list(KEYS)]
    start = early.dropna().index[0]
    early = early.loc[start:]
    assert np.isfinite(early.to_numpy()).all()
    result = pd.concat([early,historical])
    result.attrs.update(historical.attrs)
    assert result.index.is_unique and result.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(result.loc[historical.index],historical,check_index_type=False,
                                  check_names=False)
    calendar = pd.DatetimeIndex(pd.to_datetime(sessions.union(historical.index)))
    return result,benchmark,calendar,gaps


def extended_credit():
    early = pd.read_csv(SOURCE["credit_2009"],encoding="utf-8-sig").sort_values("month")
    expected = pd.period_range("2009-01","2009-12",freq="M").astype(str).tolist()
    assert early.month.tolist()==expected and early[["household_reported_yi","firm_reported_yi"]].notna().all().all()
    early["basis"] = np.where(early.month.str[5:].astype(int).isin(YEARTODATE),"year_to_date","monthly")
    late = pd.read_csv(SOURCE["legacy_credit"],encoding="utf-8-sig")
    new = pbc.credit_series(pd.concat([early,late],ignore_index=True))
    reference = pd.read_csv(SOURCE["legacy_credit_values"],index_col="month",encoding="utf-8-sig")
    np.testing.assert_allclose(new.loc[reference.index,"middle_long_new_yi"],reference.middle_long_new_yi,
                               atol=1e-10,rtol=0)
    ix=reference.index[reference.credit_change.notna()]
    np.testing.assert_allclose(new.loc[ix,"credit_change"],reference.loc[ix,"credit_change"],
                               atol=2e-14,rtol=0)
    return new


def extended_pmi():
    government = pd.read_parquet(SOURCE["cn_pmi"]).set_index("month").sort_index()
    government.index = pd.to_datetime(government.index,format="%Y%m").to_period("M")
    idx = pd.period_range("2009-01","2009-12",freq="M")
    daily = pd.DataFrame(json.loads(SOURCE["hsbc_pmi_calendar"].read_text(encoding="utf-8")),
                         columns=["date","label","private","forecast","previous","revision"])
    daily["released"] = pd.to_datetime(daily.date.str[:10],format="%d.%m.%Y")
    daily["month"] = daily.released.dt.to_period("M")-daily.released.dt.day.le(10).astype(int)
    daily = daily.loc[daily.month.isin(idx)].set_index("month")
    assert len(daily)==12 and daily.index.is_unique
    older = pd.DataFrame({"manufacturing":government.loc[idx,"pmi010000"].astype(float),
                          "nonmanufacturing":ORIGINAL_NONMAN_2009,
                          "private":daily.private.reindex(idx).astype(float)},index=idx)
    older["available"] = daily.released.reindex(idx).clip(
        lower=(idx+1).to_timestamp()+pd.Timedelta(days=2))
    legacy = pd.read_csv(SOURCE["legacy_pmi"],index_col="month",parse_dates=["available"])
    legacy.index = pd.PeriodIndex(legacy.index,freq="M")
    columns = ["manufacturing","nonmanufacturing","private","available"]
    joined = pd.concat([older,legacy[columns]])
    values = joined[columns[:3]]
    assert values.notna().all().all() and joined.available.is_monotonic_increasing
    change = values.rolling(12).mean().pct_change(12,fill_method=None).diff(3)
    votes = pd.DataFrame(np.where(change>0,1,-1),index=joined.index)
    joined["votes"] = votes.sum(axis=1).where(change.notna().all(axis=1))
    joined["direction"] = np.sign(joined.votes)
    ix = legacy.index[legacy.direction.notna()]
    np.testing.assert_array_equal(joined.loc[ix,"direction"],legacy.loc[ix,"direction"])
    return joined


def verify_phase_inputs(phases):
    assert (phases.credit_published.dropna()<=
            phases.credit_published.dropna().index.strftime("%Y-%m-%d")).all()
    assert (phases.growth_available.dropna()<=phases.growth_available.dropna().index).all()


def first_run(returns,phases):
    first_state = phases.stage.first_valid_index().date().isoformat()
    risk_ready = returns.index[63]
    ready = max(risk_ready,returns.index[returns.index.searchsorted(first_state,side="right")])
    assert pd.notna(phases.at[pd.Timestamp(returns.index[returns.index.get_loc(ready)-1]),"stage"])
    return ready,risk_ready,first_state


def verify_case(account):
    assert account.ledger.index[-1]==END and account.ledger.cash.ge(-1e-8).all()
    assert (account.signals.known<account.signals.date).all()
    assert (account.trades.known<account.trades.date).all()
    log_attribution(account)


def brief(a):
    fields=("total","cagr","vol","mdd","peak","trough","sessions")
    info=stats(a)
    return {k:info[k] for k in fields}


def write(frame,name):
    frame.to_csv(HERE/name,index=False,encoding="utf-8-sig")


def main():
    protocol=HERE/"protocol.json"
    assert sha256(protocol.read_bytes()).hexdigest()==(HERE/"protocol.sha256").read_text(encoding="ascii").strip()
    returns,benchmark,dates,gaps=inputs()
    c=extended_credit()
    p=extended_pmi()
    phases=macro_inputs(dates,c,p)
    legacy=pd.read_parquet(SOURCE["legacy_phases"])
    for frame in (phases,legacy):
        verify_phase_inputs(frame)
    pd.testing.assert_series_equal(phases.loc["2012-06-11":,"stage"],
                                  legacy.loc["2012-06-11":,"stage"],check_index_type=False)
    start,risk,signal=first_run(returns,phases)
    last_start,_,legacy_signal=first_run(returns,legacy)
    first_fcf=returns.index[returns.index.searchsorted(RULE_DATE,side="right")]
    ranges=[("extended_first",start,phases),("legacy_first",last_start,legacy),
            ("jul2012","2012-07-02",legacy),("apr2013","2013-04-12",phases)]
    all_rows, target_rows, nav_rows = [],[],[]
    write(c.reset_index(),"credit_from_2009.csv")
    write(p.rename_axis("month").reset_index(),"pmi_from_2009.csv")
    write(phases.rename_axis("known_date").reset_index(),"phases.csv")
    returns.to_parquet(HERE/"returns_gs_extended.parquet")
    for label,day,state in ranges:
        gs=run_variant(returns,state,day,first_fcf,True)
        v12=engine.economic("V12",benchmark,None,day,SLIP)
        assert gs.ledger.index.equals(v12.ledger.index) and gs.base_date==v12.base_date
        for name,acct in (("GS",gs),("V12",v12)):
            verify_case(acct)
            all_rows.append(dict(study=label,start=day,end=END,account=name,**brief(acct)))
            for window_name,st,ed in (("added_before_2013_04_12",day,"2013-04-11"),
                                      ("through_2012",day,"2012-12-31"),
                                      ("2013_only","2013-01-01","2013-12-31")):
                if st<=ed:
                    record=window(acct,st,ed)
                    all_rows.append(dict(study=label+"_"+window_name,start=record["start"],
                                         end=record["end"],account=name,**{
                                             k:record[k] for k in ("total","cagr","vol","mdd","peak","trough","sessions")}))
        target_rows.append(gs.signals.assign(study=label))
        nav_rows.append(pd.DataFrame({"date":gs.ledger.index,"study":label,
                                     "GS":gs.ledger.nav.to_numpy(),"V12":v12.ledger.nav.to_numpy()}))
        if label=="apr2013":
            parent=pd.read_csv(CUTOFF/"comparison.csv")
            ref=parent.loc[parent.start.eq(day)&parent.case.eq("fcf_from_published_index")].iloc[0]
            for k in ("cagr","vol","mdd","total"):
                np.testing.assert_allclose(stats(gs)[k],ref[k],rtol=1e-12,atol=1e-12)
        if label=="extended_first":
            other=run_variant(returns,state,day,first_fcf,True,perturb=True)
            for k in ("ledger","pnl","weights","signals","trades"):
                pd.testing.assert_frame_equal(getattr(gs,k),getattr(other,k),check_exact=True)
    write(pd.DataFrame(all_rows),"comparison.csv")
    write(pd.concat(target_rows,ignore_index=True),"targets.csv")
    write(pd.concat(nav_rows,ignore_index=True),"nav.csv")
    metadata=dict(first_run=start,risk_ready=risk,first_stage=signal,legacy_first_stage=legacy_signal,
                  legacy_first_run=last_start,
                  first_credit_observation=c.credit_change.first_valid_index(),
                  first_credit_release=c.loc[c.credit_change.first_valid_index(),"released"],
                  first_growth_available=str(p.loc[p.direction.notna(),"available"].iloc[0].date()),
                  early_missing_direct_gold_pairs=gaps,early_fcf_perturbation_invariant=True,
                  cutoff_parent_2013_reproduced=True,
                  sources={k:sha256(v.read_bytes()).hexdigest() for k,v in SOURCE.items()})
    (HERE/"verification.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(all_rows).to_string(index=False),flush=True)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__":
    main()
