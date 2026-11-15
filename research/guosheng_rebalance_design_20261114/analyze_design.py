"""Mechanism attribution, bounded-neighbor checks and simultaneous diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import rebalance_study as r


def save(df,name):df.to_csv(r.OUT/name,index=False,encoding='utf-8-sig')


def returns_for(case):
    path=r.OUT/'accounts'/case
    l=pd.read_parquet(path/'ledger.parquet')
    m=json.loads((path/'meta.json').read_text())
    ret=l.nav/l.nav.shift(1,fill_value=m['capital'])-1
    return ret


def rolling_and_concentration(table):
    base=table.loc[table.period.eq('full') & table.role.isin(['main','factorial']) & table['mode'].eq('amount') &
                   table.account_start.eq('2014-01-02') & table.bp.eq(9)].set_index('name')
    daily=pd.DataFrame({name:returns_for(row.case) for name,row in base.iterrows()})
    logs=np.log1p(daily)
    logs.index=pd.to_datetime(logs.index)
    monthly=logs.groupby(logs.index.to_period('M')).sum().loc[:'2026-10']
    rolling=[]
    rolling_rows=[]
    leave=[]
    for name in monthly:
        for months in (36,60):
            annual=np.expm1(monthly.rolling(months).sum()*12/months)
            diff=(annual[name]-annual.MM).dropna()
            rolling.append(dict(name=name,months=months,windows=len(diff),positive_fraction=float(diff.gt(0).mean()),
                                worst_cagr_gap=float(diff.min()),median_cagr_gap=float(diff.median()),best_cagr_gap=float(diff.max())))
            rolling_rows.extend(dict(name=name,months=months,end=str(date),cagr_gap=value) for date,value in diff.items())
        yearly=(logs[name]-logs.MM).groupby(logs.index.year).sum()
        full=yearly.sum()
        for year,contribution in yearly.items():
            remaining=full-contribution
            span=(logs.index[-1]-pd.Timestamp('2013-12-31')).days/365.25
            removed=(min(logs.index[-1],pd.Timestamp(f'{year}-12-31'))-max(pd.Timestamp('2013-12-31'),pd.Timestamp(f'{year-1}-12-31'))).days/365.25
            leave.append(dict(name=name,year=int(year),year_log_excess=float(contribution),
                              whole_log_excess=float(full),share_of_whole_log_excess=float(contribution/full) if abs(full)>1e-12 else np.nan,
                              remaining_annual_log_excess=float(remaining/(span-removed))))
    save(pd.DataFrame(rolling),'rolling_summary.csv')
    save(pd.DataFrame(rolling_rows),'rolling_windows.csv')
    save(pd.DataFrame(leave),'year_concentration.csv')
    # Joint centered paired-block bootstrap: max error over the full declared
    # architecture family, including failures. It conditions on this history.
    differences=monthly.subtract(monthly.MM,axis=0)
    names=list(differences.columns)
    x=differences.to_numpy()
    centered=x-x.mean(axis=0)
    rng=np.random.default_rng(20261114)
    n=len(x)
    errors=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/12)))
        ix=((starts[:,None]+np.arange(12))%n).ravel()[:n]
        errors.append(12*centered[ix].mean(axis=0))
    errors=np.asarray(errors)
    threshold=float(np.quantile(errors.max(axis=1),.95))
    means=12*x.mean(axis=0)
    rows=[dict(name=name,annual_log_excess=means[j],individual_lower95=means[j]-np.quantile(errors[:,j],.95),
               simultaneous_lower95=0. if name=='MM' else means[j]-threshold,joint_error95=threshold,months=n,block_months=12,paths=2000)
          for j,name in enumerate(names)]
    save(pd.DataFrame(rows),'paired_block_uncertainty.csv')


def neighbors(table):
    full=table.loc[table.period.eq('full')]
    main=full.loc[full.role.eq('main') & full.bp.eq(9)]
    rows=[]
    for (name,mode),group in full.loc[full.role.str.endswith('neighbor')].groupby(['name','mode']):
        reference=main.loc[main.name.eq(name)&main['mode'].eq(mode)&main.account_start.eq(group.account_start.iloc[0])].iloc[0]
        for role,pair in group.groupby('role'):
            rows.append(dict(name=name,mode=mode,role=role,center_cagr=reference.cagr,
                             min_cagr=min(reference.cagr,pair.cagr.min()),max_cagr=max(reference.cagr,pair.cagr.max()),
                             center_above_both_neighbors=reference.cagr-pair.cagr.max(),
                             min_mdd=min(reference.mdd,pair.mdd.min()),max_mdd=max(reference.mdd,pair.mdd.max()),
                             tests=len(pair)+1))
    save(pd.DataFrame(rows),'neighbor_summary.csv')


def cost_and_transfer(table):
    full=table.loc[table.period.eq('full')]
    rows=[]
    for row in full.loc[full.role.isin(['main','factorial']) & full.bp.eq(9)].itertuples():
        zero=full.loc[full.role.eq('gross') & full.name.eq(row.name) & full['mode'].eq(row.mode) & full.account_start.eq(row.account_start)]
        if len(zero):
            z=zero.iloc[0]
            rows.append(dict(name=row.name,mode=row.mode,start=row.account_start,gross_cagr=z.cagr,net_cagr=row.cagr,
                             cagr_cost_difference=z.cagr-row.cagr,annual_commission=row.annual_commission,
                             annual_slippage=row.annual_slippage,orders=row.orders,targets=row.targets,turnover=row.one_way_turnover))
    save(pd.DataFrame(rows),'gross_net_costs.csv')
    pairs=[]
    for row in full.loc[full.role.isin(['main','transfer'])].itertuples():
        b=full.loc[full.role.eq(row.role)&full.name.eq('MM')&full['mode'].eq(row.mode)&full.account_start.eq(row.account_start)&full.frame.eq(row.frame)&full.bp.eq(row.bp)]
        if len(b):
            b=b.iloc[0]
            pairs.append(dict(name=row.name,mode=row.mode,start=row.account_start,frame=row.frame,bp=row.bp,
                              cagr_gap=row.cagr-b.cagr,vol_gap=row.vol-b.vol,mdd_gap=row.mdd-b.mdd,calmar_gap=row.calmar-b.calmar))
    save(pd.DataFrame(pairs),'matched_differences.csv')


def timing(data,table):
    state=data.state.loc['2014-01-01':].copy()
    state['basket']=state.stage.map(lambda x:'/'.join(r.s.ASSETS[int(x)]))
    changes=state.loc[state.basket.ne(state.basket.shift())].iloc[1:]
    trading=data.views.hist.index
    observations=[]
    episodes=[]
    for j,(date,event) in enumerate(changes.iterrows()):
        day=date.strftime('%Y-%m-%d')
        next_date=changes.index[j+1] if j+1<len(changes) else pd.Timestamp('2026-11-13')
        duration=int(np.sum((trading>=day)&(trading<next_date.strftime('%Y-%m-%d'))))
        previous=state.loc[:date].iloc[-2]
        cause='/'.join(k for k in ('money','credit','growth') if previous[k]!=event[k])
        episodes.append(dict(known=day,next_change=next_date.date().isoformat(),basket=event.basket,
                             holding_sessions=duration,cause=cause,
                             round_trip_10_sessions=(j+1<len(changes) and duration<=10 and changes.iloc[j+1].basket==previous.basket)))
    base=table.loc[table.period.eq('full') & table.role.isin(['main','factorial']) & table['mode'].eq('amount') &
                   table.account_start.eq('2014-01-02') & table.bp.eq(9)]
    for run in base.itertuples():
        signals=pd.read_parquet(r.OUT/'accounts'/run.case/'signals.parquet')
        for event in episodes:
            first=int(trading.searchsorted(event['known'],side='right'))
            eligible=signals.loc[signals.date.gt(event['known']) & signals.known.lt(event['next_change']) & signals.requested.eq(event['basket'])]
            fill=eligible.date.iloc[0] if len(eligible) else None
            wait=int(trading.get_loc(fill)-first) if fill is not None else None
            observations.append(dict(name=run.name,**event,first_target=fill,wait_sessions=wait,missed=fill is None))
    save(pd.DataFrame(episodes),'macro_basket_episodes.csv')
    obs=pd.DataFrame(observations)
    save(obs,'target_response_detail.csv')
    summary=obs.groupby('name').agg(changes=('known','size'),missed=('missed','sum'),median_wait=('wait_sessions','median'),max_wait=('wait_sessions','max')).reset_index()
    save(summary,'target_response_summary.csv')


def risk_drift(data,table):
    rows=[]
    cache={}
    full=table.loc[table.period.eq('full') & table.role.isin(['main','factorial']) & table['mode'].eq('amount') &
                   table.account_start.eq('2014-01-02') & table.bp.eq(9)]
    exclusions=data.views.hist.attrs.get('excluded_risk_dates',{})
    for row in full.itertuples():
        if row.name not in ('MM','WW','DD','DM','WM','DD_band','DM_band','DN'):continue
        weights=pd.read_parquet(r.OUT/'accounts'/row.case/'weights.parquet').reindex(columns=r.s.KEYS).shift(1,fill_value=0)
        samples=[]
        for day,w in weights.iterrows():
            assets=tuple(w.index[w>1e-8])
            if len(assets)<2 or w.sum()<.9:continue
            known=data.views.hist.index[data.views.hist.index.get_loc(day)-1]
            key=(known,assets)
            if key not in cache:
                omit={date for asset in assets for date in exclusions.get(asset,())}
                x=data.views.hist.loc[:known,list(assets)].drop(index=sorted(omit),errors='ignore').dropna().tail(63)
                cache[key]=x.cov().to_numpy()
            cov=cache[key]
            a=w.loc[list(assets)].to_numpy()
            rc=a*(cov@a)/(a@cov@a)
            samples.append(float(np.max(abs(rc-1/len(a)))))
        rows.append(dict(name=row.name,observations=len(samples),median_max_risk_share_deviation=np.median(samples),
                         p95_max_risk_share_deviation=np.quantile(samples,.95),max_risk_share_deviation=max(samples)))
    save(pd.DataFrame(rows),'risk_share_drift.csv')


def main():
    table=pd.read_csv(r.OUT/'comparison.csv')
    assert table.case.nunique()==len(r.plans())
    cost_and_transfer(table)
    neighbors(table)
    rolling_and_concentration(table)
    data=r.s.load()
    timing(data,table)
    risk_drift(data,table)
    print('Analysis complete',flush=True)


if __name__=='__main__':main()
