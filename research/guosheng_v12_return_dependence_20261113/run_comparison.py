"""Same-date replays for premium, drift, and historical basket timing."""
from contextlib import nullcontext
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from normalization import (CLOSE, EARLY, FOREIGN, GS_ROOT, HERE, LONG, OLD_NAV, OLD_STUDY,
                           PROVIDER, PriorToCash, QDII, STARTS, anchors, drift_normal, gs_normal,
                           premiums, prior, v12_normal, years)
from run_study import CAPITAL, KEYS, Rotation, apply_policy, engine, measures

TABLES = HERE / "tables"
MAIN = prior.MAIN


def csv(table,name):
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / (name+".csv"),index=False,encoding="utf-8-sig")


def verify_protocol():
    for name in ("protocol","additional_controls"):
        path=HERE/(name+".json")
        assert PROVIDER.sha(path)==path.with_suffix(".sha256").read_text(encoding="ascii").strip()
    prior.verify_frozen()


def run(name,rets,state,start,cash_dates=None):
    with apply_policy(lambda n,r,_: Rotation(MAIN,r,state) if cash_dates is None
                      else PriorToCash(MAIN,r,state,cash_dates),KEYS) if name=="GS" else nullcontext():
        a = engine.economic(MAIN if name=="GS" else "V12",rets,None,start,.0009)
    r,_,log = prior.log_attribution(a)
    assert a.ledger.index.equals(rets.loc[start:].index)
    assert (a.signals.known<a.signals.date).all()
    assert (a.trades.known<a.trades.date).all()
    if name=="GS":
        assert (a.signals.last_return<=a.signals.known).all()
        assert (a.signals.credit_date<=a.signals.known).all()
        assert (a.signals.growth_date<=a.signals.known).all()
    m=measures(a.ledger,a.base_date,CAPITAL)
    np.testing.assert_allclose(m["cagr"],np.expm1(np.log1p(r).sum()/years(a.base_date,r.index[-1])),
                               rtol=0,atol=1e-12)
    return m,a,r,log


def shifted(frame,session,n,with_return_flags):
    ix = frame.index.get_indexer(session)
    assert (ix>=0).all() and (np.diff(ix)==1).all()
    d = frame.loc[session]
    original_rows = np.arange(len(d))
    before_and_test = frame.index[ix[0]-70:ix[-1]+1]
    clock_mapping = (np.arange(-70,len(d))+n)%len(d)
    source_dates=d.index[clock_mapping]
    matrix=pd.DataFrame(d.to_numpy()[clock_mapping],columns=d.columns,index=before_and_test)
    assert len(matrix)==len(session)+70
    np.testing.assert_array_equal(np.sort(clock_mapping[70:]),original_rows)
    np.testing.assert_allclose(np.log1p(matrix.loc[session]).sum(),np.log1p(d).sum(),rtol=0,atol=1e-12)
    if with_return_flags:
        excluded=frame.attrs.get("excluded_risk_dates",{})
        matrix.attrs["excluded_risk_dates"] = {
            k:before_and_test[source_dates.isin(v)].tolist() for k,v in excluded.items()
        }
    return matrix


def timing_controls(raw,normal,state):
    # Keep co-movements across both strategies on the same historical session.
    # This is a calendar reassignment test, with one wrap in each test path.
    session=raw["GS"].loc["2014-01-02":"2025-12-31"].index
    rows=[]
    for source,pair in (("market",raw),("premium_neutral",normal)):
        for n in range(12):
            d={}
            for name,frame in pair.items():
                m=shifted(frame,session,252*n,name=="GS")
                d[name]=run(name,m,state,session[0])[0]
            assert d["GS"]["base_date"]==d["V12"]["base_date"] and d["GS"]["end"]==d["V12"]["end"]
            rows.append(dict(source=source,shift_trading_days=252*n,
                             gs_cagr=d["GS"]["cagr"],v12_cagr=d["V12"]["cagr"],
                             gs_vol=d["GS"]["vol"],v12_vol=d["V12"]["vol"],
                             gs_mdd=d["GS"]["mdd"],v12_mdd=d["V12"]["mdd"],
                             cagr_gap=d["GS"]["cagr"]-d["V12"]["cagr"]))
    return pd.DataFrame(rows)


def attributable(a,log,scope,name,case,start):
    g=a.pnl.sum()
    full=float(log.to_numpy().sum())
    gross=float(g.sum())
    return [dict(scope=scope,strategy=name,scenario=case,start=start,asset=k,log_pnl=float(log[k].sum()),
                 log_share=float(log[k].sum()/full) if abs(full)>1e-12 else np.nan,
                 gross_pnl=float(g[k]) if k!="fee" else -float(a.ledger.cost.sum()),
                 gross_share=float(g[k]/gross) if k!="fee" and abs(gross)>1e-12 else np.nan)
            for k in log]


def verified_source_hashes():
    sources=[LONG/"protocol.json",LONG/"supplementary_protocol.json",LONG/"coverage.json",
             LONG/"data"/"macro.parquet",LONG/"data"/"returns_gs.parquet",LONG/"data"/"returns_v12.parquet",
             EARLY,FOREIGN,GS_ROOT/"source"/"chinabond_30y_wealth.json",
             OLD_STUDY/"output"/"receipt.json",
             *(OLD_STUDY/"output"/f"premium_{code}.csv" for code,_ in QDII.values()),
             *(OLD_NAV/f"fund_nav_{code}.csv" for code,_ in QDII.values()),
             *(HERE/"inputs"/"tushare").rglob("*.parquet")]
    return {p.relative_to(HERE.parent).as_posix():PROVIDER.sha(p) for p in sources}


def actual_reference(attrib):
    rows=[]
    for start in ("2025-07-01","2025-11-06"):
        for name,case in (("GS",MAIN),("V12","V12")):
            a=prior.load_account("actual",start,case)
            idx=a.ledger.loc[:CLOSE].index
            shortened=SimpleNamespace(ledger=a.ledger.loc[idx],pnl=a.pnl.loc[idx],base_date=a.base_date)
            r,_,logs=prior.log_attribution(shortened)
            metrics=measures(shortened.ledger,a.base_date,CAPITAL)
            rows.append(dict(scope="actual_quote",strategy=name,scenario="market",**metrics))
            attrib.extend(attributable(shortened,logs,"actual_quote",name,"market",start))
    return rows


def mark_effects(name, start, account, raw, fair, observed, *, case):
    """Change only that day's mark, retaining the original prior positions and costs."""
    idx=account.ledger.index
    held=account.weights.loc[:,raw.columns].shift(1,fill_value=0.)
    prior_value=account.ledger.nav.shift(1,fill_value=CAPITAL)
    fees=account.ledger.cost/prior_value
    daily_gross=held*raw.loc[idx]
    np.testing.assert_allclose(daily_gross.sum(axis=1)-fees,observed,rtol=0,atol=3e-15)
    impacts=held*(raw.loc[idx]-fair.loc[idx])
    counterpart=observed-impacts.sum(axis=1)
    assert (counterpart>-1.).all()
    elapsed=years(account.base_date,idx[-1])
    return [dict(strategy=name,start=start,end=idx[-1],case=case,asset=k,
                 annual_arithmetic_mark_gap=impacts[k].mean()*252,
                 active_days=int((held[k].gt(1e-9)&raw.loc[idx,k].ne(fair.loc[idx,k])).sum()),
                 market_strategy_cagr=float(np.expm1(np.log1p(observed).sum()/elapsed)),
                 fixed_holdings_navmark_cagr=float(np.expm1(np.log1p(counterpart).sum()/elapsed)))
            for k in raw.columns if impacts[k].ne(0.).any()]


def premium_reversion(panel_sets):
    """Value the held positions if price premiums move to their history medians."""
    rows=[]
    for name,panels in panel_sets.items():
        account=prior.load_account("actual","2025-11-06",MAIN if name=="GS" else "V12")
        for asset,series in panels.items():
            p=series.premium_close_to_nav.dropna()
            assert p.index[-1]==CLOSE
            now,median=p.iloc[-1],p.median()
            market_return=(1+median)/(1+now)-1
            held=float(account.weights.at[CLOSE,asset])
            rows.append(dict(strategy=name,asset=asset,date=CLOSE,weight=held,premium=now,
                             median_premium=median,asset_price_change=market_return,
                             portfolio_one_time_change=held*market_return))
    return pd.DataFrame(rows)


def main():
    verify_protocol()
    hashes=verified_source_hashes()
    raw={"GS":pd.read_parquet(LONG/"data/returns_gs.parquet").loc[:CLOSE],
         "V12":pd.read_parquet(LONG/"data/returns_v12.parquet").loc[:CLOSE]}
    assert raw["GS"].index.isin(raw["V12"].index).all()
    state=pd.read_parquet(LONG/"data/macro.parquet")
    coverage=json.loads((LONG/"coverage.json").read_text(encoding="utf-8"))
    gs_fair, gs_p=gs_normal(raw["GS"],coverage)
    v12_fair,v12_p,repairs=v12_normal(raw["V12"])
    fair={"GS":gs_fair,"V12":v12_fair}
    shared=v12_fair.copy()
    for k in ("gold","metals"):
        eligible=gs_p[k].index[gs_p[k].nav_used]
        shared.loc[eligible,k]=gs_fair.loc[eligible,k]
    shared_prem={"GS":gs_fair,"V12":shared}
    source=anchors(gs_fair)
    prem=pd.DataFrame(premiums(gs_p,"GS")+premiums(v12_p,"V12"))
    for asset,z in gs_p.items():
        z.to_parquet(HERE/"inputs"/f"valuation_GS_{asset}.parquet")
    for asset,z in v12_p.items():
        z.to_parquet(HERE/"inputs"/f"valuation_V12_{asset}.parquet")
    csv(prem,"premium_and_asset_returns")
    csv(premium_reversion({"GS":gs_p,"V12":v12_p}),"premium_reversion")
    csv(source,"anchors")
    result, attribution, shifts, curves, period_returns, marks = [],[],[],[],[],[]
    for start in STARTS:
        i=raw["GS"].index.get_loc(start)
        assert raw["GS"].index[i-1]==raw["V12"].index[raw["V12"].index.get_loc(start)-1]
        for name,frame in raw.items():
            cases=[("market",frame,None),("premium_neutral",fair[name],None)]
            if name=="V12":
                cases.append(("premium_neutral_shared",shared,None))
            for label in ("whole","pre2020"):
                rates=source.loc[source.strategy.eq(name)&source.anchor.eq(label)].set_index("asset").annual.to_dict()
                drift, details=drift_normal(fair[name],start,rates)
                cases.append((f"premium_neutral_{label}",drift,None))
                shifts.extend(dict(strategy=name,scenario=f"premium_neutral_{label}",start=start,**r) for r in details)
                if label=="whole":
                    drift_lower, details=drift_normal(fair[name],start,rates,down_only=True)
                    cases.append(("only_lower_above_long_history",drift_lower,None))
                    shifts.extend(dict(strategy=name,scenario="only_lower_above_long_history",start=start,**r) for r in details)
                    if name=="V12":
                        common_rates=rates|{"metals":float(source.loc[
                            source.strategy.eq("GS")&source.asset.eq("metals")&source.anchor.eq(label),"annual"].iloc[0])}
                        drift_shared,details=drift_normal(shared,start,common_rates)
                        cases.append(("premium_neutral_whole_shared",drift_shared,None))
                        shifts.extend(dict(strategy=name,scenario="premium_neutral_whole_shared",start=start,**r) for r in details)
                    if name=="GS" and start in ("2014-01-02","2020-07-30"):
                        cases.append(("premium_neutral_whole_before_listing_cash",drift,coverage["actual_return_from"]))
            if name=="GS" and start in ("2014-01-02","2020-07-30"):
                cases.append(("before_listing_cash",frame,coverage["actual_return_from"]))
            for case,rets,cash_dates in cases:
                m,a,r,log=run(name,rets,state,start,cash_dates)
                if case=="market":
                    marks.extend(mark_effects(name,start,a,frame,fair[name],r,case="primary"))
                    if name=="V12":
                        marks.extend(mark_effects(name,start,a,frame,shared,r,case="shared"))
                if start=="2014-01-02" and case=="market":
                    parent=prior.load_account("amount",start,MAIN if name=="GS" else name)
                    pd.testing.assert_frame_equal(a.ledger,parent.ledger.loc[:CLOSE],check_exact=True)
                result.append(dict(scope="economic",strategy=name,scenario=case,**m))
                if case in ("market","premium_neutral","premium_neutral_whole","premium_neutral_pre2020",
                            "premium_neutral_whole_before_listing_cash"):
                    attribution.extend(attributable(a,log,"economic",name,case,start))
                    curves.append(pd.DataFrame({"scope":"economic","strategy":name,"scenario":case,
                                                "start":start,"date":r.index,"daily_return":r.to_numpy(),
                                                "equity":a.ledger.nav.to_numpy()/CAPITAL}))
                for y,x in r.groupby(r.index.str[:4]):
                    period_returns.append(dict(strategy=name,scenario=case,start=start,year=y,
                                               total=np.expm1(np.log1p(x).sum()),days=len(x)))
                print(start,name,case,*(round(m[k]*100,3) for k in ("cagr","vol","mdd")),flush=True)
    result.extend(actual_reference(attribution))
    csv(pd.DataFrame(result),"scenarios")
    csv(pd.DataFrame(attribution),"attribution")
    csv(pd.DataFrame(shifts),"drift_adjustments")
    csv(pd.DataFrame(period_returns),"annual")
    csv(pd.DataFrame(marks),"fixed_holdings_mark_effect")
    pd.concat(curves,ignore_index=True).to_parquet(TABLES/"curves.parquet",index=False)
    c=timing_controls(raw,fair,state)
    csv(c,"timing_controls")
    assert hashes==verified_source_hashes()
    PROVIDER.write(HERE/"verification.json",dict(verified_input_hashes=hashes,split_dates_fair_mark=repairs,
                       scenarios=len(result),timing_controls=len(c),baseline_2014_matches_frozen=True,
                       same_dates_between_strategies=True,checked_ledger_identities=True,
                       nav_changes_are_ex_post_counterfactuals=True))
    print("Verified",len(result),"scenario accounts and",len(c),"paired timing control accounts.",flush=True)


if __name__=="__main__":
    main()
