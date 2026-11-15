from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'guosheng_final_decision_20261114'))
import study as s
d=s.load();CAP=500000.
comparison=pd.read_csv(OUT/'comparison.csv');verify=[];latencies=[];intervals=[];change_metrics=[]

def stats(L,base_date,start,end):
    z=L.loc[start:end];j=L.index.get_loc(z.index[0]);b=CAP if j==0 else L.nav.iloc[j-1];bd=base_date if j==0 else L.index[j-1]
    nav=np.r_[b,z.nav.to_numpy()];r=nav[1:]/nav[:-1]-1;years=(pd.Timestamp(end)-pd.Timestamp(bd)).days/365.25
    return [(nav[-1]/b)**(1/years)-1, r.std(ddof=1)*252**.5,(1-nav/np.maximum.accumulate(nav)).max(),nav[-1]/b-1]

for folder in sorted((OUT/'accounts').iterdir()):
    meta=json.loads((folder/'meta.json').read_text());L=pd.read_parquet(folder/'ledger.parquet');T=pd.read_parquet(folder/'trades.parquet');W=pd.read_parquet(folder/'weights.parquet');P=pd.read_parquet(folder/'pnl.parquet');S=pd.read_parquet(folder/'signals.parquet')
    if meta['bp']:
        fees=np.maximum(5,T.notional*.00015)+T.notional*meta['bp']/10000
    else:fees=np.zeros(len(T))
    fee_error=float(np.max(np.abs(T.cost-fees))) if len(T) else 0.
    identity=float((L.nav.shift(fill_value=CAP)+P.sum(axis=1)-L.cost-L.nav).abs().max())
    nums=[]
    for x in comparison.loc[comparison.case.eq(folder.name)].itertuples():nums.extend(abs(a-b) for a,b in zip(stats(L,meta['base_date'],x.start,x.end),[x.cagr,x.vol,x.mdd,x.total]))
    nav_error=None;holdings_error=None
    if meta['mode']=='amount':
        keys=list(W.columns[:-1]);R=d.views.benchmark if meta['kind']=='V12' else d.views.hist
        R=R.reindex(index=L.index,columns=keys);pos=np.zeros(len(keys));cash=CAP;nav_error=0.;holdings_error=0.
        groups={day:z for day,z in T.groupby('date')}
        for day,rr in R.iterrows():
            pos*=1+rr.to_numpy();cash_pre=cash;buys=0.
            for order in groups.get(day,T.iloc[:0]).itertuples():
                j=keys.index(order.asset)
                if order.side=='sell':pos[j]-=order.notional;cash+=order.notional-order.cost
                else:pos[j]+=order.notional;cash-=order.notional+order.cost;buys+=order.notional+order.cost
            assert buys<=cash_pre+1e-6 and cash>=-1e-6 and pos.min()>=-1e-6
            nav_error=max(nav_error,abs(pos.sum()+cash-L.at[day,'nav']))
            holdings_error=max(holdings_error,float(np.max(abs(pos-L.at[day,'nav']*W.loc[day,keys].to_numpy()))))
    assert fee_error<1e-6 and identity<1e-6 and max(nums)<1e-12
    if nav_error is not None:assert nav_error<1e-5 and holdings_error<1e-5
    verify.append(dict(case=folder.name,days=len(L),orders=len(T),metric_error=max(nums),fee_error=fee_error,identity_error=identity,independent_nav_error=nav_error,holdings_error=holdings_error))
    if meta['bp']==9 and meta['kind']!='V12' and not meta['strict']:
        cal=d.views.hist.index;known=[cal[cal.get_loc(day)-1] for day in L.index]
        ideal=pd.Series(['/'.join(s.ASSETS[int(d.state.at[pd.Timestamp(k),'stage'])]) for k in known],index=L.index)
        active=S.set_index('date').requested.reindex(L.index).ffill()
        grouping=ideal.ne(ideal.shift()).cumsum();delays=[];missed=0
        for no,span in ideal.groupby(grouping):
            match=active.loc[span.index].eq(span.iloc[0]);delay=int(np.flatnonzero(match.to_numpy())[0]) if match.any() else None
            if no!=1:
                if delay is None:missed+=1
                else:delays.append(delay)
            intervals.append(dict(case=folder.name,spell=int(no),start=span.index[0],end=span.index[-1],days=len(span),basket=span.iloc[0],delay_sessions=delay,missed=delay is None))
        latencies.append(dict(case=folder.name,mode=meta['mode'],start=meta['start'],strategy=meta['kind'],days=len(L),state_spells=int(grouping.max()),missed_spells_ex_initial=missed,
            stale_target_days=int(active.ne(ideal).sum()),stale_target_fraction=float(active.ne(ideal).mean()),mean_matched_delay_sessions=float(np.mean(delays)) if delays else 0,max_matched_delay_sessions=int(max(delays)) if delays else 0))
pd.DataFrame(verify).to_csv(OUT/'independent_validation.csv',index=False)
pd.DataFrame(latencies).to_csv(OUT/'signal_latency_verified.csv',index=False)
pd.DataFrame(intervals).to_csv(OUT/'state_interval_alignment.csv',index=False)

# Do not attribute overlapping or selected dates to out-of-sample success.
month=pd.read_csv(OUT/'monthly.csv')
concentration=[]
for mode,start,strict in [('amount','2014-01-02',False),('amount','2020-01-02',False),('actual','2025-11-06',False),('actual','2025-06-04',False)]:
    q=month.loc[month['mode'].eq(mode)&month.account_start.eq(start)&month.strictFCF.eq(strict)]
    pivot=q.pivot(index='month',columns='strategy',values='total');logs=np.log1p(pivot)
    for kind in ['weekly','daily','month_and_switch']:
        diff=logs[kind]-logs.monthly;jan_jul=diff.loc[diff.index.isin(['2026-01','2026-07'])].sum()
        concentration.append(dict(mode=mode,start=start,strategy=kind,months=len(diff),positive_months=int(diff.gt(1e-12).sum()),net_log_excess=float(diff.sum()),jan_jul_2026=float(jan_jul),other_months=float(diff.sum()-jan_jul),top_two_contribution=float(diff.nlargest(2).sum()),worst_month=diff.idxmin(),worst_month_log_excess=float(diff.min())))
pd.DataFrame(concentration).to_csv(OUT/'frequency_concentration.csv',index=False)

zero=comparison.loc[comparison.bp.eq(0)&comparison.period.eq('full')]
costdrag=[]
for z in zero.itertuples():
    q=comparison.loc[comparison['mode'].eq(z.mode)&comparison.account_start.eq(z.account_start)&comparison.strategy.eq(z.strategy)&comparison.period.eq('full')&~comparison.strictFCF&comparison.bp.gt(0)]
    for x in q.itertuples():costdrag.append(dict(mode=z.mode,start=z.account_start,strategy=z.strategy,bp=x.bp,zero_cost_cagr=z.cagr,net_cagr=x.cagr,cagr_drag=z.cagr-x.cagr,fee_per_nav_year=x.fee_per_nav_year))
pd.DataFrame(costdrag).to_csv(OUT/'cost_drag.csv',index=False)
summary=dict(accounts=len(verify),account_days=sum(v['days'] for v in verify),orders=sum(v['orders'] for v in verify),amount_accounts=sum(v['independent_nav_error'] is not None for v in verify),max_nav_error=max(v['independent_nav_error'] or 0 for v in verify),max_metric_error=max(v['metric_error'] for v in verify),scope='amount accounts independently reconstructed from zero positions and saved orders; actual accounts P&L/fees/metrics checked, not independent reconstruction from raw market and corporate actions')
(OUT/'independent_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2));print('LATENCY',pd.DataFrame(latencies).to_json(orient='records'));print('CONCENTRATION',pd.DataFrame(concentration).to_json(orient='records'))
