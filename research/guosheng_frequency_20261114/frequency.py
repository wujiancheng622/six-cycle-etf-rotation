import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guosheng_final_decision_20261114"))
import study as s

PROTO = HERE / "protocol.json"
CFG = json.loads(PROTO.read_text(encoding="utf8"))
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


class Frequency(s.Policy):
    def __init__(self, returns, state, name):
        base = "month_and_switch" if name in ("month_and_switch","week_and_switch") else "monthly"
        super().__init__(returns, returns, state, s.Case(name, schedule=base))
        self.name = name
        self.initial = None
        if name in ("weekly","week_and_switch","daily"):
            days = returns.index
            weeks = pd.DatetimeIndex(days).to_period("W-SUN")
            choose = np.ones(len(days), dtype=bool) if name=="daily" else ~weeks.duplicated()
            self.scheduled = {days[i]: (days[i], days[i-1]) for i in range(1,len(days)) if choose[i]}

    def targets(self, day, known, i, nav):
        if self.initial is None:
            self.initial = day
        if self.name in ("weekly","week_and_switch","daily"):
            # Parent's monthly guard suppresses its initial calendar month. For
            # the alternative clocks suppress only the already handled first day.
            self.first_month = day[:7] if day == self.initial else ""
        return super().targets(day, known, i, nav)


def play(data, mode, start, name, slip, state, reuse):
    if name == "V12":
        return s.benchmark(data,start=start,bp=slip,mode=mode,same_cash=reuse)
    factory = lambda _, frame, prices: Frequency(frame,state,name)
    with s.capital(s.CAPITAL), s.prior.apply_policy(factory,s.KEYS):
        if mode=="actual":
            q,adv,up,down = data.views.gs_quotes
            return s.cloned_actual(reuse)(name,q,adv,up,down,None,start,slip/10000)
        # Use the already audited amount variant: a dust sale whose minimum
        # commission would overdraw cash is deferred.
        frame=data.views.hist
        return s.economic(frame, factory(name,frame,None),start,slip/10000)


def summarize(a, slip, case, lo, hi):
    m = s.metrics(a,lo,hi)
    if not m:
        return None
    l = a.ledger.loc[m["start"]:m["end"]]
    prev = a.ledger.nav.shift(1,fill_value=s.CAPITAL).reindex(l.index)
    trades = a.trades.loc[a.trades.date.between(l.index[0],l.index[-1])]
    sp = trades.notional*slip/10000
    commission = np.maximum(5.,trades.notional*.00015)
    np.testing.assert_allclose(trades.cost,commission+sp,atol=1e-8)
    charge_rate = trades.date.map(1/prev)
    sig = a.signals.loc[a.signals.date.between(l.index[0],l.index[-1])]
    changes = int(sig.requested.ne(sig.requested.shift()).sum()-1) if "requested" in sig and len(sig) else None
    return dict(**m, orders=len(trades), order_days=int(l.trade_count.gt(0).sum()),
                basket_changes=changes, commission=commission.sum(), slippage=sp.sum(),
                annual_commission_drag=(commission*charge_rate).sum()/m["years"],
                annual_slippage_drag=(sp*charge_rate).sum()/m["years"],
                annual_min_fee_increment=((commission-trades.notional*.00015)*charge_rate).sum()/m["years"],
                min_fee_orders=int((trades.notional*.00015 < 5).sum()),
                name=case)


def main():
    digest=hashlib.sha256(PROTO.read_bytes()).hexdigest()
    stamp=HERE/"protocol.sha256"
    if stamp.exists():
        assert stamp.read_text().strip()==digest
    else:
        stamp.write_text(digest+"\n")
    data=s.load()
    s.OUT=OUT
    rows=[]
    portfolios=CFG["schedules"]+["V12"]
    for mode,starts in CFG["starts"].items():
        for start in starts:
            plans=[(k,bp,0,False) for k in portfolios for bp in CFG["slippage_each_side_bp"]]
            if start in ("2014-01-02","2025-11-06"):
                plans.extend((k,9,lag,False) for k in CFG["schedules"] for lag in CFG["extra_macro_delay_calendar_days"])
            if mode=="actual":
                plans.extend((k,9,0,True) for k in portfolios)
            states={0:data.state}
            states.update({lag:s.macro(data,"all_macro_extra_calendar_lag",lag) for lag in CFG["extra_macro_delay_calendar_days"]})
            for k,bp,lag,reuse in plans:
                a=play(data,mode,start,k,bp,states[lag],reuse)
                case=f"{mode}_{start}_{k}_{bp}bp_{lag}d_cash{int(reuse)}"
                assert a.ledger.cash.min()>-1e-6,(case,a.ledger.cash.min())
                assert a.trades.known.lt(a.trades.date).all()
                assert a.signals.known.lt(a.signals.date).all()
                if "last_return" in a.signals:
                    assert a.signals.last_return.le(a.signals.known).all()
                if k=="monthly" and bp==9 and lag==0 and not reuse:
                    stem=f"monthly_{start}_9bp" if mode=="amount" else f"actual_monthly_{start}_9bp_cash0"
                    path=HERE.parent/"guosheng_final_decision_20261114/results/accounts"/stem/"ledger.parquet"
                    if path.exists():
                        pd.testing.assert_frame_equal(a.ledger,pd.read_parquet(path),check_exact=False,rtol=1e-11,atol=1e-7)
                s.record(a,case,[],{},save=True)
                periods={"full":(start,s.END)} | {k:v for k,v in s.PERIODS.items() if k in ("2014-2016","2017-2019","2020-2022","2023-2025","2026","2026_H2")}
                for period,(lo,hi) in periods.items():
                    m=summarize(a,bp,case,lo,hi)
                    if m:
                        rows.append(dict(mode=mode,account_start=start,schedule=k,bp=bp,lag=lag,reuse_cash=reuse,period=period,**m))
            print(mode,start,len(plans),"accounts",flush=True)
            s.write(pd.DataFrame(rows),"comparison.csv")


if __name__=="__main__":
    main()
