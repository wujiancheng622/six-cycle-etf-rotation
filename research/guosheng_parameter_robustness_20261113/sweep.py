"""Replay feasible schedule and macro neighbors with the audited engine."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from study_inputs import (BASE, CAPITAL, CFG, DEFAULT, END, HERE, KEYS, PARENT, PROVIDER,
                    Inputs, apply_policy, calendar_targets, check_frozen, engine, parent, risk_parity)

OUT = HERE / "runs"
ASSETS = parent.ASSETS_BY_STAGE
BASELINE = "D1_r63"
BENCHMARK = "V12"
WINDOWS = {
    "full": (None, END),
    "2014–2016": ("2014-01-01","2016-12-31"),
    "2017–2019": ("2017-01-01","2019-12-31"),
    "2020–2022": ("2020-01-01","2022-12-31"),
    "2023–2025": ("2023-01-01","2025-12-31"),
    "2026 YTD": ("2026-01-01",END),
    "2026-07 to end": ("2026-07-01",END),
    "2026-05-07 to end": ("2026-05-07",END),
}


@dataclass(frozen=True)
class Case:
    name: str
    schedule: str = "D1"
    lookback: int = 63
    delay: int = 0
    macro: str = ""
    value: int = 0
    groups: set = field(default_factory=set,compare=False)

    def row(self):
        row = asdict(self)
        row["groups"] = ",".join(sorted(row["groups"]))
        return row


def cases():
    configs={}
    def add(case,group):
        c=configs.setdefault(case.name,case)
        assert c==case
        c.groups.add(group)
    for i in CFG["schedule"]["ordinals_from_first"]:
        add(Case(f"D{i+1}_r63",f"D{i+1}"),"timing")
    for name in ("ME","W","Q"):
        add(Case(f"{name}_r63",name),"other_frequency")
    for n in CFG["covariance_sessions"]:
        add(Case(f"D1_r{n}",lookback=n),"risk")
    for off in CFG["joint_grid"]["schedule_ordinals"]:
        for n in CFG["joint_grid"]["covariance_sessions"]:
            add(Case(f"D{off+1}_r{n}",f"D{off+1}",n),"grid")
    for n in CFG["schedule"]["delayed_execution_only"]:
        add(Case(f"delay{n}_r63",delay=n),"execution_delay")
    for macro,values in CFG["macro_one_at_a_time"].items():
        for n in values:
            if n!=DEFAULT[macro]:
                add(Case(f"{macro}_{n}",macro=macro,value=n),"macro")
    return configs


class InsufficientRiskHistory(Exception):
    pass


class Risk:
    def __init__(self, frame, fcf_cutoff=None):
        self.frame=frame
        self.cutoff=fcf_cutoff
        self.excluded=frame.attrs.get("excluded_risk_dates",{})
        self.cache={}

    def weights(self, known, assets, n):
        k=known,assets,n
        if k in self.cache:
            return self.cache[k]
        mask={d for asset in assets for d in self.excluded.get(asset,())}
        train=self.frame.loc[:known,list(assets)].drop(index=sorted(mask),errors="ignore").dropna().tail(n)
        used=assets
        cash=0.
        if "fcf" in assets and self.cutoff is not None:
            clean=train.loc[self.cutoff:]
            if len(clean)<n:
                used=tuple(k for k in assets if k!="fcf")
                cash=1/len(assets)
                mask={d for asset in used for d in self.excluded.get(asset,())}
                train=self.frame.loc[:known,list(used)].drop(index=sorted(mask),errors="ignore").dropna().tail(n)
            else:
                train=clean
        if len(train)!=n:
            raise InsufficientRiskHistory(f"{known}: {len(train)}/{n} genuine returns for {used}")
        if len(used)==1:
            fractions=np.array([1.])
        else:
            cov=train.cov().to_numpy()
            assert np.linalg.eigvalsh(cov).min()>0
            fractions=risk_parity(cov)
            rc=fractions*(cov@fractions)
            np.testing.assert_allclose(rc/rc.sum(),1/len(used),atol=1e-8)
        w=pd.Series(fractions*(1-cash),index=used).reindex(KEYS,fill_value=0.).to_numpy()
        self.cache[k]=(w,train.index[0],train.index[-1],"/".join(used),cash)
        return self.cache[k]


class ScheduledRotation:
    def __init__(self, case, risk, macro, schedule, start):
        self.case=case
        self.risk=risk
        self.macro=macro
        self.schedule=schedule
        self.start=start
        self.receipts=[]
        self.pending=None
        self.last=-2

    def targets(self, day, known, i, nav):
        if i==0:
            target, signal_known=day,known
        elif day[:7]!=self.start[:7] and day in self.schedule:
            target, signal_known=self.schedule[day]
        elif i==self.last+1:
            return None,self.pending
        else:
            return None,None
        assert signal_known<target<=day and signal_known<=known<day
        macro=self.macro.loc[pd.Timestamp(signal_known)]
        assert pd.notna(macro.stage)
        phase=int(macro.stage)
        raw=ASSETS[phase]
        w,first,last,selected,cash=self.risk.weights(signal_known,raw,self.case.lookback)
        assert first<=last<=signal_known
        assert macro.credit_published<=signal_known and macro.growth_available.date().isoformat()<=signal_known
        assert w.sum()<=1.+1e-12
        self.pending=w
        self.last=i
        self.receipts.append(dict(date=day,known=signal_known,target_date=target,
                                  prior_exec_date=known,phase=phase,requested="/".join(raw),selected=selected,
                                  first_return=first,last_return=last,credit_date=macro.credit_published,
                                  growth_date=macro.growth_available.date().isoformat(),
                                  money_direction=int(macro.money),credit_direction=int(macro.credit),
                                  growth_direction=int(macro.growth),
                                  cash_target=cash,**dict(zip(KEYS,w))))
        return w,w


def verify_account(a, mode):
    assert a.ledger.index[-1]==END and a.ledger.index.is_unique
    assert (a.signals.known<a.signals.date).all()
    assert (a.trades.known<a.trades.date).all()
    if mode=="actual":
        assert (a.trades.shares<=a.trades.adv*.01+1e-8).all()
        assert (a.trades.loc[a.trades.side.eq("buy"),"shares"]%100==0).all()
    lagged=a.ledger.nav.shift(1,fill_value=CAPITAL)
    residual=(lagged+a.pnl.sum(axis=1)-a.ledger.cost-a.ledger.nav).abs().max()
    assert residual<=1e-7
    return float(residual)


def interval_metrics(a, start=None, end=END):
    sub=a.ledger.loc[start:end]
    if len(sub)<2:
        return None
    i=a.ledger.index.get_loc(sub.index[0])
    prior=a.base_date if i==0 else a.ledger.index[i-1]
    initial=CAPITAL if i==0 else a.ledger.nav.iloc[i-1]
    nav=np.r_[initial,sub.nav.to_numpy()]
    ret=nav[1:]/nav[:-1]-1
    drawdown=1.-nav/np.maximum.accumulate(nav)
    bottom=int(drawdown.argmax())
    top=int(nav[:bottom+1].argmax())
    dates=[prior,*sub.index]
    years=(pd.Timestamp(dates[-1])-pd.Timestamp(prior)).days/365.25
    mdd=drawdown[bottom]
    cagr=(nav[-1]/initial)**(1/years)-1
    without_2025=ret[sub.index.str[:4]!="2025"]
    return dict(first=sub.index[0],end=sub.index[-1],base_date=prior,days=len(sub),years=years,
                cagr=cagr,total=nav[-1]/initial-1,vol=np.std(ret,ddof=1)*np.sqrt(252),mdd=mdd,
                peak=dates[top],trough=dates[bottom],calmar=cagr/mdd if mdd else np.nan,
                fee=float(sub.cost.sum()),turnover=float((sub.notional.to_numpy()/nav[:-1]).sum()/years),
                skipped_2025_sessions=len(ret)-len(without_2025),
                cagr_with_2025_zeroed=np.expm1(np.log1p(without_2025).sum()/years))


def replay(data,mode,start,case,slip,risk,macro,schedule):
    actual=mode=="actual"
    if case is None:
        if actual:
            q,adv,hi,lo=data.v12_quotes
            return engine.actual(BENCHMARK,q,adv,hi,lo,None,start,slip)
        return engine.economic(BENCHMARK,data.v12,None,start,slip)
    with apply_policy(lambda n,r,p:ScheduledRotation(case,risk,macro,schedule,start),KEYS):
        if actual:
            q,adv,hi,lo=data.gs_quotes
            return engine.actual(case.name,q,adv,hi,lo,None,start,slip)
        return engine.economic(case.name,data.hist,None,start,slip)


def validate_baseline(mode,start,name,a):
    archived=parent.load_account(mode,start,parent.MAIN if name==BASELINE else BENCHMARK)
    for k in ("ledger","pnl","weights","trades"):
        pd.testing.assert_frame_equal(getattr(a,k),getattr(archived,k),check_exact=False,rtol=0,atol=1e-10)


def hash_inputs():
    paths=[HERE/"protocol.json",BASE/"protocol.json",PARENT/"protocol.json",PARENT/"coverage.json",
           PARENT/"data/returns_gs.parquet",PARENT/"data/returns_v12.parquet",PARENT/"data/macro.parquet",
           PARENT/"data/pmi_as_reported.csv",PARENT/"data/credit_monthly.csv",
           Path(engine.__file__),Path(__file__),HERE/"study_inputs.py"]
    return {str(p.relative_to(HERE.parent).as_posix()):PROVIDER.sha(p) for p in paths}


def run_grid(mode,start):
    check_frozen()
    configs=cases()
    data=Inputs.load()
    reference=data.build_macro()
    cols=["market","money","money_score","credit","growth","stage","credit_published","growth_available",*["lpr_1y","lpr_5y","rrr_B","rrr_S"]]
    pd.testing.assert_frame_equal(reference[cols],data.macro[cols],check_dtype=False)
    fund=data.risk_actual if mode=="actual" else data.hist
    calendar=data.gs_quotes[0].panels["close"].index if mode=="actual" else data.hist.index
    schedules={}
    all_macro={("",0):data.macro}
    risks={False:Risk(fund),True:Risk(fund,data.fcf_cutoff)}
    dest=OUT/f"{mode}_{start}"
    dest.mkdir(parents=True,exist_ok=True)
    jobs=[(case,False,.0009) for case in (None,*configs.values())]
    if mode=="amount":
        jobs.extend((c,True,.0009) for c in configs.values() if
                    "grid" in c.groups or c.macro.startswith(("credit","pmi")))
    stress_names={BASELINE,"D2_r63","D4_r63","D5_r63","D6_r63","D10_r63","D1_r56","D1_r70",
                  *(f"{k}_{n}" for k,v in CFG["local_macro_neighbors"].items()
                    for n in v if k.startswith(("credit","pmi")))}
    jobs.extend((c,False,.002) for c in [None,*(configs[n] for n in sorted(stress_names))])
    hashes=hash_inputs()
    rows,monthly,targets,curves=[],[],[],[]
    unavailable=[]
    max_residual=0.
    checked=[]
    for case,clean,slip in jobs:
        name=case.name if case is not None else BENCHMARK
        key=dict(mode=mode,start=start,case=name,fcf_clean=clean,slip_bp=int(round(slip*10000)))
        try:
            if case is None:
                schedule=macro=None
            else:
                schedule_id=(case.schedule,case.delay)
                if schedule_id not in schedules:
                    schedules[schedule_id]=calendar_targets(calendar,case.schedule,case.delay)
                schedule=schedules[schedule_id]
                spec=case.macro,case.value
                if spec not in all_macro:
                    all_macro[spec]=data.build_macro({case.macro:case.value})
                macro=all_macro[spec]
            a=replay(data,mode,start,case,slip,risks[clean],macro,schedule)
        except InsufficientRiskHistory as ex:
            assert mode=="actual"
            unavailable.append(key|{"reason":str(ex)})
            continue
        assert a.ledger.index[0]==start
        max_residual=max(max_residual,verify_account(a,mode))
        if name in (BASELINE,BENCHMARK) and not clean and slip==.0009:
            validate_baseline(mode,start,name,a)
            checked.append(name)
        for label,(a0,b0) in WINDOWS.items():
            if a0 is not None and a0[:7]<start[:7]:
                continue
            m=interval_metrics(a,a0,b0)
            if m is not None:
                rows.append(key|{"window":label,**m})
        returns=a.ledger.nav.pct_change(fill_method=None)
        returns.iloc[0]=a.ledger.nav.iloc[0]/CAPITAL-1
        monthly.extend(key|dict(month=d,total=np.expm1(x)) for d,x in
                       np.log1p(returns).groupby(returns.index.str[:7]).sum().items())
        if case is not None:
            assert (a.signals.first_return<=a.signals.last_return).all()
            if clean:
                assert a.signals.loc[a.signals.fcf.gt(0),"first_return"].ge(data.fcf_cutoff).all()
            targets.append(a.signals.assign(**key))
        if not clean and slip==.0009 and name in {BASELINE,BENCHMARK,"D2_r63","D4_r63","D5_r63","D10_r63","ME_r63"}:
            curves.append(pd.DataFrame(key|dict(date=a.ledger.index,nav=a.ledger.nav.to_numpy()/CAPITAL)))
    assert hashes==hash_inputs()
    assert sorted(checked)==sorted([BASELINE,BENCHMARK])
    pd.DataFrame(rows).to_parquet(dest/"metrics.parquet",index=False)
    pd.DataFrame(monthly).to_parquet(dest/"monthly.parquet",index=False)
    pd.concat(targets,ignore_index=True).to_parquet(dest/"targets.parquet",index=False)
    pd.concat(curves,ignore_index=True).to_parquet(dest/"curves.parquet",index=False)
    PROVIDER.write(dest/"checks.json",dict(mode=mode,start=start,attempts=len(jobs),completed=len(jobs)-len(unavailable),
                                          unavailable=unavailable,original_matched=checked,max_book_residual=max_residual,
                                          inputs_sha256=hashes,macro_default_exact=True))
    print(mode,start,"checked",len(jobs)-len(unavailable),"of",len(jobs),flush=True)


if __name__=="__main__":
    run_grid(sys.argv[1],sys.argv[2])
