"""Evaluate rebalance frequencies and nearby settings on the frozen market paths."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "guosheng_parameter_robustness_20261113"
sys.dont_write_bytecode = True
sys.path.insert(0,str(PREVIOUS))
import sweep as archived
from study_inputs import Inputs, KEYS, PARENT, PROVIDER, apply_policy, check_frozen, engine, parent
sys.path.remove(str(PREVIOUS))

PROTOCOL = HERE/"protocol.json"
FOLLOWUP = HERE/"local_macro_followup.json"
CFG = json.loads(PROTOCOL.read_text(encoding="utf-8"))
END = CFG["end"]
NORMAL_CAPITAL = CFG["fees"]["common_capital"]
SMALL_CAPITAL = CFG["fees"]["supplementary_actual_capital"]
GROUPS = ("monthly","weekly","daily","stage_change")
BENCHMARK = "V12"
COMMISSION = CFG["fees"]["commission_rate"]
MIN_COMMISSION = CFG["fees"]["min_commission_rmb"]


@dataclass(frozen=True)
class Setting:
    frequency: str
    lookback: int = 63
    ordinal: int = 1
    macro: str = ""
    value: int = 0

    @property
    def name(self):
        name = self.frequency
        if self.ordinal != 1:
            name += f"_day{self.ordinal}"
        if self.lookback != 63:
            name += f"_risk{self.lookback}"
        if self.macro:
            name += f"_{self.macro}_{self.value}"
        return name

    def columns(self):
        return dict(case=self.name,**asdict(self))


@contextmanager
def equity_size(value):
    original = engine.CAPITAL
    try:
        engine.CAPITAL = value
        yield
    finally:
        engine.CAPITAL = original


def scheduled_days(calendar,frequency,ordinal,start):
    """Use the exchange session index to derive the first eligible session."""
    if frequency in ("monthly","stage_change"):
        return frozenset(archived.calendar_targets(calendar,"D1"))
    if frequency == "daily":
        return frozenset(calendar[calendar>=start])
    assert frequency=="weekly" and ordinal in (1,2,3)
    d = pd.Series(calendar,index=calendar)
    weeks = pd.to_datetime(calendar).to_period("W-SUN")
    first_week = pd.Period(start,freq="W-SUN")
    terminal_week = weeks[-1]
    chosen = []
    for week,g in d.groupby(weeks,sort=True):
        if week<=first_week:
            continue
        if len(g)<ordinal and week==terminal_week and g.index[-1]==END:
            continue
        chosen.append(g.iloc[min(ordinal,len(g))-1])
    return frozenset(chosen)


class FrequencyPolicy:
    def __init__(self, setting, risk, macro, scheduled):
        self.setting = setting
        self.risk = risk
        self.macro = macro
        self.scheduled = scheduled
        self.receipts = []
        self.previous_assets = None
        self.pending = None
        self.last = -2

    def targets(self,day,known,i,nav):
        tick = self.macro.loc[pd.Timestamp(known)]
        assert pd.notna(tick.stage)
        phase = int(tick.stage)
        chosen = parent.ASSETS_BY_STAGE[phase]
        switched = self.setting.frequency=="stage_change" and chosen!=self.previous_assets
        if i==0 or day in self.scheduled or switched:
            weights,first,last,assets,cash = self.risk.weights(known,chosen,self.setting.lookback)
            assert first<=last<=known<day
            assert tick.credit_published<=known and tick.growth_available.date().isoformat()<=known
            self.pending = weights
            self.last = i
            self.previous_assets = chosen
            self.receipts.append(dict(date=day,known=known,first_return=first,last_return=last,
                                      phase=phase,selected=assets,requested="/".join(chosen),cash_target=cash,
                                      **dict(zip(KEYS,weights))))
            return weights,weights
        if i==self.last+1:
            return None,self.pending
        return None,None


def replay(data,mode,start,setting,slip,capital,risk,macro,schedule):
    strategy = setting.name if setting is not None else BENCHMARK
    patch = (apply_policy(lambda *args:FrequencyPolicy(setting,risk,macro,schedule),KEYS)
             if setting is not None else nullcontext())
    with equity_size(capital),patch:
        if mode=="actual":
            quotes,adv,hi,lo = data.gs_quotes if setting is not None else data.v12_quotes
            a = engine.actual(strategy,quotes,adv,hi,lo,None,start,slip)
        else:
            a = engine.economic(strategy,data.hist if setting is not None else data.v12,None,start,slip)
    return a


def verify(a,mode,capital,slip):
    ledger,trades,signals = a.ledger,a.trades,a.signals
    assert ledger.index[-1]==END and ledger.index.is_unique and ledger.index.is_monotonic_increasing
    assert (signals.known<signals.date).all() and (trades.known<trades.date).all()
    assert (trades.notional>0).all() and (ledger.cash>=-1e-7).all()
    if mode=="actual":
        assert (trades.shares<=trades.adv*.01+1e-8).all()
        assert (trades.loc[trades.side.eq("buy"),"shares"]%100==0).all()
    if "first_return" in signals:
        assert (signals.first_return<=signals.last_return).all()
        assert (signals.last_return<=signals.known).all()
        np.testing.assert_allclose(signals[list(KEYS)].sum(axis=1)+signals.cash_target,1.,rtol=0,atol=1e-9)
    prior = ledger.nav.shift(1,fill_value=capital)
    residual = (prior+a.pnl.sum(axis=1)-ledger.cost-ledger.nav).abs().max()
    assert residual<=1e-7
    charges = trades.assign(commission=(trades.notional*COMMISSION).clip(lower=MIN_COMMISSION),
                             slippage=trades.notional*slip)
    np.testing.assert_allclose(charges.commission+charges.slippage,charges.cost,rtol=0,atol=1e-9)
    fees = (charges.groupby("date")[["commission","slippage"]].sum()
            .reindex(ledger.index,fill_value=0.))
    np.testing.assert_allclose(fees.sum(axis=1),ledger.cost,rtol=0,atol=1e-9)
    return charges,fees,float(residual)


def window_stats(a,capital,charges,fees,start,end):
    ledger = a.ledger.loc[start:end]
    if len(ledger)<2:
        return None
    prev = a.ledger.nav.shift(1,fill_value=capital).loc[ledger.index]
    base_date = a.base_date if ledger.index[0]==a.ledger.index[0] else a.ledger.index[
        a.ledger.index.get_loc(ledger.index[0])-1]
    years = (pd.Timestamp(ledger.index[-1])-pd.Timestamp(base_date)).days/365.25
    nav = np.r_[prev.iloc[0],ledger.nav.to_numpy()]
    returns = nav[1:]/nav[:-1]-1.
    orders = charges.loc[charges.date.between(ledger.index[0],ledger.index[-1])]
    signals = a.signals.loc[a.signals.date.between(ledger.index[0],ledger.index[-1])]
    fee_rate = fees.loc[ledger.index].div(prev,axis=0).sum()/years
    vol = returns.std(ddof=1)*np.sqrt(252.)
    mdd = (1-nav/np.maximum.accumulate(nav)).max()
    total = nav[-1]/nav[0]-1.
    cagr = np.expm1(np.log1p(returns).sum()/years)
    signed = orders.groupby("side").notional.sum()
    stage_changes = None
    if "requested" in signals:
        all_baskets = a.signals.requested.ne(a.signals.requested.shift())
        all_baskets.iloc[0] = False
        stage_changes = int(all_baskets.loc[signals.index].sum())
    return dict(first=ledger.index[0],last=ledger.index[-1],base_date=base_date,days=len(returns),
                years=years,cagr=cagr,total=total,vol=vol,mdd=mdd,calmar=cagr/mdd if mdd else None,
                target_days=len(signals),basket_changes=stage_changes,fill_days=int(ledger.trade_count.gt(0).sum()),
                orders=len(orders),fee_floor_orders=int(orders.notional.lt(MIN_COMMISSION/COMMISSION).sum()),
                traded_notional=float(orders.notional.sum()),buy_notional=float(signed.get("buy",0)),
                sell_notional=float(signed.get("sell",0)),median_order_notional=float(orders.notional.median()),
                annual_turnover=float(ledger.notional.div(prev).sum()/years),
                commission_rmb=float(fees.loc[ledger.index,"commission"].sum()),
                slippage_rmb=float(fees.loc[ledger.index,"slippage"].sum()),
                annual_commission=fee_rate.commission,annual_slippage=fee_rate.slippage,
                annual_fee=fee_rate.sum(),cash_mean=float(a.weights.loc[ledger.index,"cash"].mean()))


def run_cases(mode, close_macro=None):
    if close_macro is not None:
        return [(Setting(f,macro=k,value=v),9,NORMAL_CAPITAL,False)
                for f in GROUPS for k,vs in close_macro.items() for v in vs]
    core = [Setting(f) for f in GROUPS]
    jobs = [(s,bp,NORMAL_CAPITAL,False) for s in [None,*core] for bp in CFG["fees"]["slip_bps"]]
    jobs += [(Setting(f,lookback=n),9,NORMAL_CAPITAL,False)
             for f in GROUPS for n in CFG["risk_sessions"] if n!=63]
    jobs += [(Setting(f,macro=k,value=v),9,NORMAL_CAPITAL,False)
             for f in GROUPS for k,vs in CFG["macro_neighbors"].items() for v in vs]
    jobs += [(Setting("weekly",ordinal=n),9,NORMAL_CAPITAL,False)
             for n in CFG["weekly_adjacent_ordinals"]]
    if mode=="amount":
        jobs += [(s,9,NORMAL_CAPITAL,True) for s in core]
    else:
        jobs += [(s,9,SMALL_CAPITAL,False) for s in [None,*core]]
    return jobs


def write_case_logs(a,path,case):
    for name in ("ledger","weights","pnl","trades","signals"):
        getattr(a,name).to_parquet(path/f"{case}_{name}.parquet")


def main(mode,start,study="core"):
    digest = PROVIDER.sha(PROTOCOL)
    assert digest == PROTOCOL.with_suffix(".sha256").read_text(encoding="ascii").strip()
    check_frozen()
    assert start in CFG["starts"][mode]
    assert study in ("core","close_macro")
    follow_digest = None
    close_macro = None
    if study=="close_macro":
        follow_digest = PROVIDER.sha(FOLLOWUP)
        assert follow_digest==FOLLOWUP.with_suffix(".sha256").read_text(encoding="ascii").strip()
        close_macro = json.loads(FOLLOWUP.read_text(encoding="utf-8"))["neighbors"]
    assert engine.CONFIG.flat["minimum_commission"]==MIN_COMMISSION
    data = Inputs.load()
    macro_base = data.build_macro()
    pd.testing.assert_series_equal(macro_base.stage,data.macro.stage,check_dtype=False)
    calendar = data.hist.index if mode=="amount" else data.gs_quotes[0].panels["close"].index
    history = data.hist if mode=="amount" else data.risk_actual
    risks = {False:archived.Risk(history),True:archived.Risk(history,data.fcf_cutoff)}
    schedules = {}
    macros = {("",0):data.macro}
    dest = HERE/("runs" if close_macro is None else "close_macro_runs")/f"{mode}_{start}"
    dest.mkdir(parents=True,exist_ok=True)
    rows = []
    all_monthly = []
    maximum_residual = 0.
    check = []
    failures = []
    jobs = run_cases(mode,close_macro)
    windows = {k:archived.WINDOWS[k] for k in CFG["segments"]}
    for setting,bp,capital,clean in jobs:
        key = (setting.columns() if setting is not None else
               dict(case=BENCHMARK,frequency=BENCHMARK,lookback=63,ordinal=0,macro="",value=0))
        key |= dict(mode=mode,start=start,slip_bps=bp,capital=capital,fcf_clean=clean)
        if setting is None:
            m,s = None,None
        else:
            mkey = setting.macro,setting.value
            if mkey not in macros:
                macros[mkey] = data.build_macro({setting.macro:setting.value})
            m = macros[mkey]
            skey = setting.frequency,setting.ordinal
            if skey not in schedules:
                schedules[skey] = scheduled_days(calendar,*skey,start)
            s = schedules[skey]
        try:
            a = replay(data,mode,start,setting,bp/1e4,capital,risks[clean],m,s)
        except archived.InsufficientRiskHistory as e:
            failures.append(key|dict(reason=str(e)))
            continue
        assert a.ledger.index[0]==start
        charges,fees,residual = verify(a,mode,capital,bp/1e4)
        maximum_residual=max(maximum_residual,residual)
        is_core = setting is None or setting==Setting(setting.frequency)
        if bp==9 and capital==NORMAL_CAPITAL and not clean:
            if key["case"] in ("monthly",BENCHMARK):
                archived.validate_baseline(mode,start,archived.BASELINE if setting is not None else BENCHMARK,a)
                check.append(key["case"])
            if key["case"]=="stage_change":
                old = PARENT/"accounts"/mode/start/"GS_stage_switch_9bp"
                if old.is_dir():
                    for k in ("ledger","weights","pnl","trades"):
                        pd.testing.assert_frame_equal(getattr(a,k),pd.read_parquet(old/f"{k}.parquet"),
                                                      check_exact=False,rtol=0,atol=1e-10)
                    check.append(key["case"])
            if is_core:
                write_case_logs(a,dest,key["case"])
                returns = a.ledger.nav/a.ledger.nav.shift(1,fill_value=capital)-1.
                mon = np.log1p(returns).groupby(returns.index.str[:7]).sum()
                all_monthly.extend(key|dict(month=k,total=np.expm1(v)) for k,v in mon.items())
        if clean:
            assert a.signals.loc[a.signals.fcf.gt(0),"first_return"].ge(data.fcf_cutoff).all()
        full_cagr = None
        for label,(left,right) in windows.items():
            if left is not None and left[:7]<start[:7]:
                continue
            row = window_stats(a,capital,charges,fees,left,right)
            if row is not None:
                rows.append(key|dict(window=label,**row))
                if label=="full":
                    full_cagr = row["cagr"]
        if is_core and not clean and capital==NORMAL_CAPITAL:
            print(mode,start,key["case"],bp,round(full_cagr*100,3),flush=True)
    if close_macro is None:
        assert {"monthly",BENCHMARK}<=set(check)
        pd.DataFrame(all_monthly).to_parquet(dest/"monthly_returns.parquet",index=False)
    pd.DataFrame(rows).to_parquet(dest/"metrics.parquet",index=False)
    PROVIDER.write(dest/"checks.json",dict(mode=mode,start=start,attempted=len(jobs),completed=len(jobs)-len(failures),
                                            archive_reconciled=check,max_account_error=maximum_residual,
                                            unavailable=failures,protocol_sha256=digest,followup_sha256=follow_digest))
    print(mode,start,"checked",len(jobs)-len(failures),"of",len(jobs),flush=True)


if __name__=="__main__":
    main(*sys.argv[1:])
