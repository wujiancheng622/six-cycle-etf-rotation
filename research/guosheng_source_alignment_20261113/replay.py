"""Replay independently documented source corrections; unknown events remain uncovered."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LONG = ROOT/"guosheng_six_regime_long_20261112"
sys.dont_write_bytecode = True
sys.path[:0] = [str(LONG),str(ROOT/"guosheng_fcf_cutoff_20261113")]
from replay_long import Inputs, MAIN, load_account, run, state_path
from replay_fcf_cutoff import run_variant

END = "2026-11-12"
REPORT_END = "2025-10-31"


def votes(dates, events):
    event_dates = pd.DatetimeIndex(pd.to_datetime(events.date))
    assert event_dates.is_monotonic_increasing and event_dates.is_unique
    position = event_dates.searchsorted(dates, side="right")-1
    valid = position>=0
    result = np.zeros(len(dates))
    elapsed = (dates[valid]-event_dates[position[valid]]).days
    result[valid] = np.where(elapsed<90,events.vote.to_numpy()[position[valid]],0)
    return result


def scenarios(inputs):
    old=inputs.macro
    group=old.copy()
    for cols,name in ((["lpr_1y","lpr_5y"],"loan"),(["rrr_B","rrr_S"],"reserve")):
        assert not group[cols].prod(axis=1).lt(0).any()
        group[name]=np.sign(group[cols].sum(axis=1))
    group["money_score"]=group.market+group.loan+group.reserve
    group["money"]=np.sign(group.money_score)
    group["stage"]=state_path(group.money,group.credit,group.growth)
    pit=group.copy()
    pmi=pd.read_csv(LONG/"data/pmi_as_reported.csv",index_col="month",parse_dates=["available"])
    assert pmi.loc["2025-10","available"]==pd.Timestamp("2025-11-10")
    assert pmi.loc["2025-10","private"]==50.6
    pmi.loc["2025-10","available"]=pd.Timestamp("2025-11-03")
    pub=pd.DatetimeIndex(pmi.available)
    assert pub.is_unique and pub.is_monotonic_increasing
    pit["growth"]=pmi.direction.set_axis(pub).reindex(pit.index,method="ffill")
    pit["growth_available"]=pd.Series(pub,index=pub).reindex(pit.index,method="ffill")
    pd.testing.assert_series_equal(pit.growth,group.growth)
    pit["stage"]=state_path(pit.money,pit.credit,pit.growth)

    public=pit.copy()
    events=pd.read_csv(HERE/"public_events.csv")
    for cat,e in events.groupby("category"):
        v=votes(public.index,e.sort_values("date"))
        if cat=="loan":
            assert not ((v!=0)&public.loan.ne(0)).any()
            public["loan"]=public.loan+v
        else:
            public[cat]=v
    public["money_score"]=public[["market","loan","reserve","reverse_repo","mlf"]].sum(axis=1,min_count=5)
    public["money"]=np.sign(public.money_score)
    public["stage"]=state_path(public.money,public.credit,public.growth)
    assert not public.stage.loc["2014":].isna().any()
    fields=["money","growth","growth_available","stage"]
    pd.concat({k:v[fields] for k,v in (("old",old),("dedup",group),("pmi",pit),("public",public))},
              axis=1).to_csv(HERE/"signal_comparison.csv",encoding="utf-8-sig")
    return {"old":old,"dedup":group,"dedup_pmi":pit,"documented_events":public}


def metrics(a,start,end):
    l=a.ledger.loc[start:end]
    j=a.ledger.index.get_loc(l.index[0])
    prev_date=a.base_date if j==0 else a.ledger.index[j-1]
    prev=1e6 if j==0 else a.ledger.nav.iloc[j-1]
    nav=np.r_[prev,l.nav.to_numpy()]
    ret=nav[1:]/nav[:-1]-1
    years=(pd.Timestamp(l.index[-1])-pd.Timestamp(prev_date)).days/365.25
    return dict(period_start=l.index[0],period_end=l.index[-1],days=len(l),total=nav[-1]/nav[0]-1,
                cagr=(nav[-1]/nav[0])**(1/years)-1,vol=np.std(ret,ddof=1)*np.sqrt(252),
                mdd=(1-nav/np.maximum.accumulate(nav)).max())


def main():
    inputs=Inputs.create()
    states=scenarios(inputs)
    rows=[]
    targets=[]
    for mode,start in (("amount","2014-01-02"),("amount","2020-01-02"),("actual","2025-11-06")):
        old=load_account(mode,start,MAIN)
        match=old.signals.set_index("date")
        cases={k:run(inputs,mode,start,MAIN,state=state) for k,state in states.items()}
        cases["documented_events_switch"] = run(inputs,mode,start,"GS_stage_switch",state=states["documented_events"])
        pd.testing.assert_frame_equal(cases["old"].ledger,old.ledger)
        pd.testing.assert_frame_equal(cases["dedup"].ledger,cases["dedup_pmi"].ledger)
        if mode=="amount":
            cutoff=inputs.hist.index[inputs.hist.index.searchsorted("2024-08-15",side="right")]
            for k in ("old","documented_events"):
                cases[k+"_fcf_cutoff"]=run_variant(inputs.hist,states[k],start,cutoff,True)
        cases["V12"]=load_account(mode,start,"V12")
        spans=[("full",start,END)]
        if mode=="amount":
            spans.append(("paper_cutoff",start,REPORT_END))
            for year in ("2015","2020","2025"):
                if year>=start[:4]:
                    spans.append((year,f"{year}-01-01",REPORT_END if year=="2025" else f"{year}-12-31"))
        else:
            spans.append(("July onward","2026-07-01",END))
        for k,a in cases.items():
            signal=a.signals.set_index("date")
            changed=None
            if k!="V12":
                if k!="documented_events_switch":
                    assert signal.index.equals(match.index)
                    changed=int(signal.selected.ne(match.selected).sum())
                targets.append(signal[["known","selected"]].assign(case=k,mode=mode,account_start=start))
            for label,lo,hi in spans:
                rows.append(dict(case=k,mode=mode,account_start=start,window=label,
                                 changed_baskets_vs_old=changed,**metrics(a,lo,hi)))
    out=pd.DataFrame(rows)
    out.to_csv(HERE/"comparison.csv",index=False,encoding="utf-8-sig")
    pd.concat(targets).to_csv(HERE/"targets.csv",encoding="utf-8-sig")
    print(out.loc[out.window.isin(["full","paper_cutoff","July onward"])].round(5).to_string(index=False))

if __name__=="__main__":
    main()
