"""Holdings and macro timing diagnostics for an unchanged monthly strategy."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "guosheng_final_decision_20261114"
OUT = HERE / "results"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
import study as s

PERIODS = {
    "full": ("2013-04-12", "2026-11-12"),
    "2014–2016": ("2014-01-01", "2016-12-31"),
    "2017–2019": ("2017-01-01", "2019-12-31"),
    "2020–2022": ("2020-01-01", "2022-12-31"),
    "2023–2025": ("2023-01-01", "2025-12-31"),
    "after_report_months": ("2025-12-01", "2026-11-12"),
    "2026H2": ("2026-07-01", "2026-11-12"),
}


def saved(name):
    folder = SOURCE / "results/accounts" / name
    return SimpleNamespace(
        **json.loads((folder / "meta.json").read_text()),
        **{key: pd.read_parquet(folder / (key + ".parquet"))
           for key in ("ledger", "weights", "pnl", "signals", "trades")},
    )


def write(rows, name):
    pd.DataFrame(rows).to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")


def contribution(book, label, returns, amount):
    old = book.ledger.nav.shift(1, fill_value=s.CAPITAL)
    ret = book.ledger.nav / old - 1
    raw = book.pnl.div(old, axis=0).assign(fees=-book.ledger.cost / old)
    scale = np.divide(np.log1p(ret), ret, out=np.ones(len(ret)), where=ret.ne(0))
    linked = raw.mul(scale, axis=0)
    np.testing.assert_allclose(linked.sum(axis=1), np.log1p(ret), atol=3e-15)
    assets = book.pnl.columns
    exposure = book.weights[assets].shift(1, fill_value=0)
    returns = returns.loc[book.ledger.index, assets]
    if amount:
        np.testing.assert_allclose(raw[assets], exposure * returns, atol=3e-15)
    periods = PERIODS | {str(y): (f"{y}-01-01", f"{y}-12-31") for y in range(2013,2027)}
    rows = []
    cycles = []
    for period, (lo, hi) in periods.items():
        part = linked.loc[lo:hi]
        if len(part) < 5:
            continue
        dates = part.index
        base_pos = book.ledger.index.get_loc(dates[0])
        base = book.base_date if base_pos == 0 else book.ledger.index[base_pos-1]
        years = (pd.Timestamp(dates[-1])-pd.Timestamp(base)).days/365.25
        total = part.to_numpy().sum()
        for key in part:
            cost_or_pnl = part[key].sum()
            w = exposure.loc[dates, key] if key in assets else None
            r = returns.loc[dates, key] if key in assets else None
            drift = float(w.mean() * r.sum()/years) if amount and w is not None else np.nan
            arithmetic = float(raw.loc[dates,key].sum()/years)
            rows.append(dict(run=label,period=period,start=dates[0],end=dates[-1],asset=key,
                             equity_return=np.expm1(total),log_contribution=cost_or_pnl,
                             net_log_share=cost_or_pnl/total if abs(total)>1e-10 else np.nan,
                             average_begin_weight=w.mean() if w is not None else np.nan,
                             own_buy_hold_cagr=np.expm1(np.log1p(r).sum()/years) if r is not None else np.nan,
                             annual_arithmetic_contrib=arithmetic,average_exposure_contrib=drift,
                             dynamic_covariance_contrib=arithmetic-drift))
    signal = book.signals.set_index("date")
    reg = signal.reindex(book.ledger.index,method="ffill")
    for phase, days in reg.groupby("phase").groups.items():
        pnl = linked.loc[days]
        cycles.append(dict(run=label,phase=int(phase),days=len(days),
                           dates_with_target=int(signal.phase.eq(phase).sum()),
                           log_contribution=float(pnl.to_numpy().sum())))
    return rows, cycles


def compound(returns):
    return np.expm1(np.log1p(returns).sum())


def build_months(book, label, returns, state, partial_first):
    assets = list(book.pnl.columns)
    returns = returns[assets]
    signals = book.signals.reset_index(drop=True)
    assert signals.date.is_unique and signals.date.is_monotonic_increasing
    rows, switches, month_stats, routes = [], [], [], []
    for i in range(len(signals)-1):
        row, following = signals.iloc[i], signals.iloc[i+1]
        if i == 0 and partial_first:
            continue
        assert row.last_return <= row.known < row.date < following.date
        known_pos = returns.index.get_loc(row.known)
        first, end = (returns.index.get_loc(date) for date in (row.date,following.date))
        assert known_pos < first < end and known_pos >= 20
        lagged = compound(returns.iloc[known_pos-20:known_pos+1])
        forward = compound(returns.iloc[first+1:end+1])
        assert lagged.notna().all() and forward.notna().all()
        weights = row[assets].astype(float)
        active = weights.gt(0)
        assert set(weights.index[active]) == set(row.selected.split("/"))
        prior = (signals.iloc[i-1][assets].astype(float).gt(0) if i else None)
        entered = active & ~prior if i else pd.Series(False,index=assets)
        exited = ~active & prior if i else pd.Series(False,index=assets)
        # Exclude the unused FCF series in the old-value replacement universe.
        universe = list(set(assets)-({"fcf","old_dividend"} if label=="old_value" else set()))
        rank = lagged[universe].sort_index().sort_values(ascending=False,kind="stable")
        selected_return = forward[active].mean()
        omitted_return = forward[[x for x in universe if not active.at[x]]].mean()
        momentum_return = forward[rank.index[:int(active.sum())]].mean()
        inputs = state.loc[pd.Timestamp(row.known)]
        direct = (int(inputs.money),int(inputs.credit),int(inputs.growth)) in s.state_path.__globals__["STAGE_MAP"]
        month_stats.append(dict(run=label,date=row.date,end=following.date,known=row.known,phase=int(row.phase),
                                direct=direct,basket=row.selected,selected_return=selected_return,omitted_return=omitted_return,
                                gap=selected_return-omitted_return,momentum_return=momentum_return,
                                equal_all=forward[universe].mean(),weighted_return=float(weights @ forward)))
        for key in universe:
            rows.append(dict(run=label,date=row.date,end=following.date,known=row.known,phase=int(row.phase),asset=key,
                             selected=active.at[key],start_weight=weights.at[key],prev_21=lagged.at[key],forward=forward.at[key],
                             action="enter" if entered.at[key] else "exit" if exited.at[key] else "held" if active.at[key] else "out",
                             money=int(inputs.money),credit=int(inputs.credit),growth=int(inputs.growth),direct=direct))
        if entered.any() and exited.any():
            switches.append(dict(run=label,date=row.date,end=following.date,known=row.known,phase=int(row.phase),
                                 old=signals.iloc[i-1].selected,new=row.selected,
                                 entered="/".join(weights.index[entered]),exited="/".join(weights.index[exited]),
                                 entered_prev=lagged[entered].mean(),exited_prev=lagged[exited].mean(),
                                 entered_after=forward[entered].mean(),exited_after=forward[exited].mean(),
                                 switch_gap=forward[entered].mean()-forward[exited].mean()))
        if i and (entered.any() or exited.any()):
            old = signals.iloc[i-1]
            old_weights = old[assets].astype(float)
            union = list(weights.index[active | prior])
            excluded = returns.attrs.get("excluded_risk_dates", {})
            omit = {d for key in union for d in excluded.get(key, ())}
            risk_input = returns.loc[:row.known,union].drop(index=sorted(omit),errors="ignore").dropna().tail(63)
            assert len(risk_input)==63 and risk_input.index[-1]<=row.known
            cov = risk_input.cov().to_numpy() * 252
            new_weight, old_weight = weights[union].to_numpy(), old_weights[union].to_numpy()
            to_risk = lambda w: float(np.sqrt(w @ cov @ w))
            old_portfolio = float(old_weights @ forward)
            new_portfolio = float(weights @ forward)
            routes.append(dict(run=label,date=row.date,end=following.date,known=row.known,old_phase=int(old.phase),
                               new_phase=int(row.phase),money=int(inputs.money),credit=int(inputs.credit),
                               growth=int(inputs.growth),old=old.selected,new=row.selected,
                               added="/".join(weights.index[entered]),removed="/".join(weights.index[exited]),
                               old_weight_vol=to_risk(old_weight),new_weight_vol=to_risk(new_weight),
                               old_weight_forward=old_portfolio,new_weight_forward=new_portfolio,
                               portfolio_forward_gap=new_portfolio-old_portfolio,
                               old_growth_weight=old_weights.get("growth",0),new_growth_weight=weights.get("growth",0),
                               growth_forward=forward.get("growth",np.nan)))
    return rows, switches, month_stats, routes


def aggregate(assets, switches, monthly):
    outcomes, events, stats, bootstrap = [], [], [], []
    windows = PERIODS | {"2014–2022":("2014-01-01","2022-12-31"),"2023起":("2023-01-01",s.END)}
    for run, group in monthly.groupby("run"):
        A = assets.loc[assets.run.eq(run)]
        S = switches.loc[switches.run.eq(run)]
        for period, (lo,hi) in windows.items():
            m=group.loc[group.date.between(lo,hi) & group.end.le(hi)]
            if m.empty:
                continue
            a=A.loc[A.date.isin(m.date)]
            e=S.loc[S.date.isin(m.date)]
            for name, vals in a.groupby("asset"):
                holds=vals.loc[vals.selected]
                dropped=vals.loc[~vals.selected]
                up=vals.loc[vals.forward.gt(0)]
                down=vals.loc[vals.forward.lt(0)]
                outcomes.append(dict(run=run,period=period,asset=name,months=len(vals),selected=len(holds),
                                     unselected=len(dropped),selected_prev=holds.prev_21.mean(),
                                     selected_next=holds.forward.mean(),unselected_next=dropped.forward.mean(),
                                     selected_win_rate=holds.forward.gt(0).mean(),
                                     up_months=len(up),down_months=len(down),
                                     held_during_up=up.selected.mean(),avoided_down=(~down.selected).mean(),
                                     mean_selected_forward_gap=holds.forward.mean()-dropped.forward.mean()))
                for action in ("enter","exit"):
                    rows=vals.loc[vals.action.eq(action)]
                    events.append(dict(run=run,period=period,asset=name,action=action,n=len(rows),
                                       prev_mean=rows.prev_21.mean(),prev_up=rows.prev_21.gt(0).mean(),
                                       next_mean=rows.forward.mean(),next_up=rows.forward.gt(0).mean(),
                                       prev_median=rows.prev_21.median(),next_median=rows.forward.median()))
            rets=a.pivot(index="date",columns="asset",values="forward")
            weights=a.pivot(index="date",columns="asset",values="start_weight")
            weighted_dynamic=(weights*rets).sum(axis=1)
            weighted_static=(weights.mean()*rets).sum(axis=1)
            np.testing.assert_allclose(weighted_dynamic,m.set_index("date").loc[weights.index,"weighted_return"],atol=1e-14)
            centered=rets-rets.mean()
            membership=weights.gt(0)
            mask_gap=centered.where(membership).mean(axis=1)-centered.where(~membership).mean(axis=1)
            stats.append(dict(run=run,period=period,months=len(m),selected_return=m.selected_return.mean(),
                              omitted_return=m.omitted_return.mean(),selected_minus_omitted=m.gap.mean(),
                              selected_win=m.gap.gt(0).mean(),selected_minus_momentum=(m.selected_return-m.momentum_return).mean(),
                              selected_minus_all=m.selected_return.mean()-m.equal_all.mean(),
                              after_asset_mean_removal=mask_gap.mean(),target_minus_mean_weights=(weighted_dynamic-weighted_static).mean(),
                              switch_n=len(e),switch_mean=e.switch_gap.mean(),switch_win=e.switch_gap.gt(0).mean(),
                              entrant_vs_exit_prev_gap=(e.entered_prev-e.exited_prev).mean()))
            if period=="full" and len(m)>=36:
                rng=np.random.default_rng(20261115)
                width=12
                source=m.set_index("date")[["gap"]].join(e.set_index("date")["switch_gap"])
                source["demeaned_gap"]=mask_gap
                n=len(source)
                seeds=rng.integers(0,n,size=(3000,int(np.ceil(n/width))))
                ids=((seeds[:,:,None]+np.arange(width))%n).reshape(3000,-1)[:,:n]
                for key in source:
                    arr=source[key].to_numpy()
                    sample=np.nanmean(arr[ids],axis=1)
                    bootstrap.append(dict(run=run,metric=key,estimate=np.nanmean(arr),
                                          q05=np.nanquantile(sample,.05),q95=np.nanquantile(sample,.95),
                                          blocks=width,paths=3000))
    return outcomes, events, stats, bootstrap


def parameter_audit(basic):
    metrics=pd.read_csv(SOURCE/"results/neighbor_comparison.csv")
    rows=[]
    for case in metrics.case.unique():
        base_case=case.split("_",1)[0]+"_base"
        book=saved(case)
        var=book.signals.copy()
        base=saved(base_case).signals.copy()
        for signals in (var,base):
            signals["month"]=signals.date.str[:7]
            assert not signals.month.duplicated().any()
        joined=base.merge(var,on="month",suffixes=("_base","_var"),validate="one_to_one")
        full=metrics.loc[metrics.case.eq(case)&metrics.period.eq("full")].iloc[0]
        same_date = joined.date_base.eq(joined.date_var).all()
        timing = np.nan
        growth_gap = np.nan
        sample_size = 0
        if same_date:
            template=basic[basic.run.eq("quote" if case.startswith("actual") else "base")].copy()
            # Exact common holding blocks and returns. For the shorter real-quote
            # restart, the earlier June-to-November targets are simply absent.
            target=var.melt(id_vars="date",value_vars=list(s.KEYS),var_name="asset",value_name="candidate")
            sample=template.merge(target,on=["date","asset"],validate="one_to_one")
            sample["use"]=sample.candidate.gt(0)
            contest=sample.groupby(["date","use"]).forward.mean().unstack()
            timing = float((contest[True]-contest[False]).mean())
            sample_size = len(contest)
            growth=sample.loc[sample.asset.eq("growth")]
            growth_gap = float(growth.loc[growth.use,"forward"].mean()-growth.loc[~growth.use,"forward"].mean())
        rows.append(dict(case=case,matched=len(joined),changed_date=joined.date_base.ne(joined.date_var).sum(),
                         same_dates_for_timing=same_date,comparison_blocks=sample_size,selection_gap=timing,
                         growth_gap=growth_gap,changed_phase=joined.phase_base.ne(joined.phase_var).sum(),
                         changed_basket=joined.requested_base.ne(joined.requested_var).sum(),
                         cagr=full.cagr,mdd=full.mdd))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    data=s.load()
    extended=s.run(data,s.Case("monthly"),start="2013-04-12")
    extended.capital=s.CAPITAL
    books={
        "base":(saved("monthly_2014-01-02_9bp"),data.views.hist,True,False),
        "long":(extended,data.views.hist,True,True),
        "quote":(saved("actual_monthly_2025-06-04_9bp_cash0"),data.views.hist,False,True),
        "old_value":(saved("value_permanent_2014-01-02_9bp"),data.full,True,False),
    }
    contrib,cycles,assets,switches,months,routes=[],[],[],[],[],[]
    for name,(book,returns,amount,partial) in books.items():
        a,b=contribution(book,name,returns,amount)
        contrib.extend(a)
        cycles.extend(b)
        a,b,d,f=build_months(book,name,returns,data.state,partial)
        assets.extend(a)
        switches.extend(b)
        months.extend(d)
        routes.extend(f)
    asset_months=pd.DataFrame(assets)
    outcome,events,aggregate_result,uncertainty=aggregate(asset_months,pd.DataFrame(switches),pd.DataFrame(months))
    frames={"contribution":contrib,"phase_contribution":cycles,"asset_month_detail":assets,
            "switch_detail":switches,"basket_transitions":routes,"month_detail":months,"asset_hold_vs_out":outcome,
            "entry_exit":events,"selection_summary":aggregate_result,"conditional_blocks":uncertainty,
            "parameter_audit":parameter_audit(asset_months)}
    for name,rows in frames.items():
        write(rows,name)
    print(json.dumps({name:len(rows) for name,rows in frames.items()},ensure_ascii=False))


if __name__=="__main__":
    main()
