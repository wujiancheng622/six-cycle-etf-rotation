from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import sys,json,hashlib,inspect
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'guosheng_final_decision_20261114'))
import study as s
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
ORIGINAL=s.Policy
CONF=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
END=CONF['end'];CAPITAL=CONF['capital']

class FrequencyPolicy(ORIGINAL):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.case.name not in ('weekly','daily'):return
        cal=self.returns.index;idx=pd.to_datetime(cal)
        if self.case.name=='daily':chosen=list(range(1,len(cal)))
        else:
            period=idx.to_period('W-SUN');chosen=[i for i in range(1,len(cal)) if period[i]!=period[i-1]]
        self.scheduled={cal[i]:(cal[i],cal[i-1]) for i in chosen}
        self.first_month='periodic_no_month_exclusion'

@contextmanager
def selected_policy(zero=False):
    old=s.Policy;fee=s.ENGINE.amount_fee
    s.Policy=FrequencyPolicy
    if zero:s.ENGINE.amount_fee=lambda value,slip:np.zeros_like(value,dtype=float)
    try:yield
    finally:s.Policy=old;s.ENGINE.amount_fee=fee

def get_account(data,kind,start,bp,mode,strict=False):
    zero=bp==0
    with selected_policy(zero):
        if kind=='V12':
            if mode=='amount':
                with s.capital(CAPITAL),s.prior.apply_policy(s.ORIGINAL_V12_POLICY,tuple(s.V12_KEYS)):
                    return s.economic(data.views.benchmark,s.ORIGINAL_V12_POLICY('V12',data.views.benchmark,None),start,bp/1e4,None,False,True)
            return s.benchmark(data,start,bp,mode)
        case=s.Case(kind,schedule='month_and_switch' if kind=='month_and_switch' else 'monthly',cutoff='fcf' if strict else '')
        # Infinite capacities for original legs, same safe amount engine as audited source.
        return s.run(data,case,start=start,bp=bp,mode=mode,limit=mode=='amount')

def metrics(a,lo,hi,bp):
    l=a.ledger.loc[lo:hi];j=a.ledger.index.get_loc(l.index[0]);capital=CAPITAL if j==0 else a.ledger.nav.iloc[j-1]
    base=a.base_date if j==0 else a.ledger.index[j-1]
    path=np.r_[capital,l.nav.to_numpy()];r=np.diff(path)/path[:-1];years=(pd.Timestamp(l.index[-1])-pd.Timestamp(base)).days/365.25
    trades=a.trades.loc[a.trades.date.between(l.index[0],l.index[-1])].copy()
    commissions=np.maximum(5,trades.notional.to_numpy()*.00015) if bp else np.zeros(len(trades))
    slips=trades.notional.to_numpy()*bp/10000
    assert np.allclose(trades.cost,commissions+slips,atol=1e-7,rtol=0)
    pn=pd.Series(path[:-1],index=l.index);tn=pn.reindex(trades.date).to_numpy()
    targets=a.signals.loc[a.signals.date.between(l.index[0],l.index[-1])]
    return dict(start=l.index[0],end=l.index[-1],base_date=base,days=len(l),years=years,cagr=(path[-1]/capital)**(1/years)-1,total=path[-1]/capital-1,
        vol=r.std(ddof=1)*252**.5,mdd=(1-path/np.maximum.accumulate(path)).max(),end_nav=path[-1],
        mean_cash=float(((l.cash+l.receivable)/l.nav).mean()),trades=len(trades),targets=len(targets),
        one_way_turnover=float((l.notional.to_numpy()/path[:-1]).sum()/2/years),notional=float(trades.notional.sum()),
        commission_yuan=float(commissions.sum()),slippage_yuan=float(slips.sum()),fees_yuan=float(l.cost.sum()),
        commission_per_nav_year=float((commissions/tn).sum()/years),slippage_per_nav_year=float((slips/tn).sum()/years),
        fee_per_nav_year=float((l.cost.to_numpy()/path[:-1]).sum()/years),min_commission_trades=int((trades.notional*.00015<=5+1e-10).sum()))

def main():
    inputs=[HERE/'protocol.json',ROOT/'guosheng_final_decision_20261114/study.py',ROOT/'guosheng_final_decision_20261114/protocol.json']
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/'frozen_manifest.json').write_text(json.dumps(before,indent=2),encoding='utf8')
    d=s.load()
    data_quality=dict(state_start=str(d.state.index.min()),state_end=str(d.state.index.max()),policy_events=len(d.events),
        categories=d.events.groupby('category').agg(rows=('vote','size'),last_usable=('usable_date','max')).reset_index().to_dict('records'),
        source_status=d.events.evidence_status.value_counts(dropna=False).to_dict(),timestamp_missing=int(d.events.announced_at.isna().sum()),
        return_missing=d.views.hist.isna().sum().to_dict(),return_duplicate_dates=int(d.views.hist.index.duplicated().sum()),
        pmi_last=d.pmi.tail(3).to_dict('records'),state_changes=int(d.state.loc[s.START:,'stage'].diff().ne(0).sum()))
    (OUT/'data_profile.json').write_text(json.dumps(data_quality,ensure_ascii=False,default=str,indent=2),encoding='utf8')
    kinds=['monthly','weekly','daily','month_and_switch','V12'];rows=[];years=[];months=[];verify=[];latencies=[]
    specs=[]
    for mode,starts in [('amount',CONF['amount_starts']),('actual',CONF['actual_starts'])]:
        for start in starts:
            for bp in CONF['slippage_bp']+([0] if mode=='amount' else []):
                for kind in kinds:specs.append((mode,start,bp,kind,False))
    for bp in [9,20]:
        for kind in kinds[:-1]:specs.append(('amount','2014-01-02',bp,kind,True))
    for mode,start,bp,kind,strict in specs:
        name=f'{mode}_{start}_{bp}bp_{kind}'+('_strictFCF' if strict else '')
        a=get_account(d,kind,start,bp,mode,strict)
        folder=OUT/'accounts'/name;folder.mkdir(parents=True,exist_ok=True)
        for f in ['ledger','weights','pnl','signals','trades']:getattr(a,f).to_parquet(folder/f'{f}.parquet')
        (folder/'meta.json').write_text(json.dumps(dict(capital=CAPITAL,base_date=a.base_date,mode=mode,start=start,bp=bp,kind=kind,strict=strict)),encoding='utf8')
        prev=a.ledger.nav.shift(1,fill_value=CAPITAL);err=float((prev+a.pnl.sum(axis=1)-a.ledger.cost-a.ledger.nav).abs().max())
        assert err<1e-6 and a.ledger.cash.min()>-1e-6 and a.signals.known.lt(a.signals.date).all()
        baseline=None
        if not strict and kind in ['monthly','month_and_switch','V12'] and bp in [9,20]:
            base_name=(f'actual_{kind}_{start}_{bp}bp_cash0' if mode=='actual' else f'{kind}_{start}_{bp}bp')
            p=ROOT/'guosheng_final_decision_20261114/results/accounts'/base_name/'ledger.parquet'
            if p.exists():baseline=float((a.ledger.nav-pd.read_parquet(p).nav).abs().max());assert baseline<1e-5,(name,baseline)
        verify.append(dict(case=name,days=len(a.ledger),trades=len(a.trades),identity_error=err,baseline_error=baseline))
        meta=dict(case=name,mode=mode,account_start=start,bp=bp,strategy=kind,strictFCF=strict)
        spans=[('full',start,END)]
        if start<'2025-10-31':spans.append(('paper',start,'2025-10-31'))
        if start<'2026-07-01':spans.append(('2026_H2','2026-07-01',END))
        if start<'2023-01-01':spans.append(('2023-2025','2023-01-01','2025-12-31'))
        for period,lo,hi in spans:rows.append(dict(**meta,period=period,**metrics(a,lo,hi,bp)))
        for y in range(int(start[:4]),2027):
            lo=max(start,f'{y}-01-01');hi=min(END,f'{y}-12-31')
            if lo<=hi:years.append(dict(**meta,year=y,**metrics(a,lo,hi,bp)))
        if bp==9:
            for month,l in a.ledger.groupby(a.ledger.index.str[:7]):months.append(dict(**meta,month=month,**metrics(a,l.index[0],l.index[-1],bp)))
        if kind!='V12' and bp==9 and not strict:
            targets=a.signals.copy();dates=a.ledger.index;active=None;delays=[];mismatch=0;changes=0
            for i,day in enumerate(dates):
                known=d.views.hist.index[d.views.hist.index.get_loc(day)-1]
                ideal='/'.join(s.ASSETS[int(d.state.at[pd.Timestamp(known),'stage'])])
                today=targets.loc[targets.date.eq(day)]
                if len(today):active=today.requested.iloc[-1]
                mismatch+=active!=ideal
                if i and ideal!=last_ideal:
                    changes+=1
                    future=targets.loc[targets.date.ge(day)&targets.requested.eq(ideal)]
                    if len(future):delays.append(int(dates.searchsorted(future.date.iloc[0])-i))
                last_ideal=ideal
            latencies.append(dict(case=name,days=len(dates),macro_basket_changes=changes,stale_target_days=mismatch,matched_transitions=len(delays),mean_sessions_to_matching_target=np.mean(delays) if delays else None,max_sessions_to_matching_target=max(delays) if delays else None))
        print('DONE',name,metrics(a,start,END,bp)['cagr'],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(years).to_csv(OUT/'annual.csv',index=False)
    pd.DataFrame(months).to_csv(OUT/'monthly.csv',index=False)
    pd.DataFrame(verify).to_csv(OUT/'validation.csv',index=False)
    pd.DataFrame(latencies).to_csv(OUT/'signal_latency.csv',index=False)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in before.items())
    print('COMPLETED',len(verify),'accounts',sum(x['days'] for x in verify),'account_days',flush=True)

if __name__=='__main__':main()
