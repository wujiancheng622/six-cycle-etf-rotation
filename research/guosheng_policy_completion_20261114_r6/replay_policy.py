"""Policy-only matched replay. Parent inputs are read-only, outputs isolated."""
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, sys, socket
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'guosheng_parameter_robustness_20261113'),str(ROOT/'guosheng_fcf_cutoff_20261113'),str(ROOT/'guosheng_six_regime_long_20261112')]
import study_inputs as si
from replay_long import run, engine, MAIN, KEYS, ASSETS_BY_STAGE, state_path, apply_policy, Rotation
from replay_fcf_cutoff import AvailableRotation, assert_clean

CATS=['reverse_repo','slf','mlf','reserve','reserve_interest','loan','deposit']

def scores(events,dates,lag=False):
    out=pd.DataFrame(0,index=dates,columns=CATS,dtype=float)
    for cat,g in events.groupby('category'):
        z=g.groupby('usable_date').vote.agg(['min','max'])
        direction=z['min'].where(z['min'].eq(z['max']),0.)
        if lag:
            d=dates.searchsorted(direction.index,side='right')
            ok=d<len(dates)
            direction=direction.iloc[np.where(ok)[0]].set_axis(dates[d[ok]])
            direction=direction.groupby(level=0).last()
        last_date=pd.Series(direction.index,index=direction.index).reindex(dates,method='ffill')
        v=direction.reindex(dates,method='ffill')
        out[cat]=v.where((dates-last_date)<pd.Timedelta(days=90),0.).fillna(0.)
    return out

def mkstate(base,events,lag=False):
    f=base.copy()
    v=scores(events,f.index,lag)
    for col in CATS:f[col]=v[col]
    f['money_score']=(v.sum(axis=1)+f.market)/8
    f['money']=np.sign(f.money_score)
    f['stage']=state_path(f.money,f.credit,f.growth)
    assert not f.loc['2014':,'stage'].isna().any()
    return f

def metrics(a,lo,hi,capital):
    l=a.ledger.loc[lo:hi]
    if l.empty:return None
    j=a.ledger.index.get_loc(l.index[0]);d0=a.base_date if j==0 else a.ledger.index[j-1]
    n0=capital if j==0 else a.ledger.nav.iloc[j-1]
    nav=np.r_[n0,l.nav.to_numpy()];r=nav[1:]/nav[:-1]-1
    yrs=(pd.Timestamp(l.index[-1])-pd.Timestamp(d0)).days/365.25
    cagr=(nav[-1]/nav[0])**(1/yrs)-1;dd=(1-nav/np.maximum.accumulate(nav)).max()
    return dict(start=l.index[0],end=l.index[-1],base_date=d0,days=len(l),total=nav[-1]/nav[0]-1,cagr=cagr,vol=np.std(r,ddof=1)*252**.5,mdd=dd,calmar=cagr/dd,mean_cash=float((l.cash/l.nav).mean()),cost=float(l.cost.sum()),final_nav=nav[-1])

def main():
    # Audit hooks guard against accidental writes in imported frozen projects.
    def audit(event,args):
        if event=='open':
            path,mode,flags=args
            if isinstance(path,(str,bytes)) and (flags & (0x1|0x2|0x40|0x200|0x400)):
                p=Path(path).resolve()
                if not p.is_relative_to(HERE):raise PermissionError('Write outside isolated output: '+str(p))
        if event in ('socket.connect','socket.getaddrinfo'):raise PermissionError('Offline replay: network disabled')
    sys.addaudithook(audit)
    tag=sys.argv[1] if len(sys.argv)>1 else 'v1'
    out=HERE/tag;out.mkdir(exist_ok=True)
    data=ROOT/'guosheng_six_regime_long_20261112/data'
    hist=pd.read_parquet(data/'returns_gs.parquet');bm=pd.read_parquet(data/'returns_v12.parquet')
    original=pd.read_parquet(data/'macro.parquet')
    old=original.copy()
    aligned=pd.read_csv(ROOT/'guosheng_source_alignment_20261113/signal_comparison.csv',header=[0,1],index_col=0)
    aligned.index=pd.to_datetime(aligned.index)
    for c in ['money','growth','stage']:
        old[c]=pd.to_numeric(aligned['public'][c],errors='coerce')
    old['growth_available']=pd.to_datetime(aligned['public']['growth_available'])
    old['money_score']=old.money # scale-only in metadata; does not change old targets
    rawfile=ROOT/'guosheng_original_reconstruction_20261114/results/policy_events_used.csv'
    events=pd.read_csv(rawfile,parse_dates=['effective_date','known_date','usable_date'])
    # These existing events are inherited evidence, not claimed as newly discovered.
    events['evidence_status']='inherited_pending_source_check'
    addon=HERE/'additional_verified_events.csv'
    if tag!='v1' and addon.exists():
        extra=pd.read_csv(addon,parse_dates=['effective_date','known_date','usable_date'])
        for r in extra.itertuples():
            # A verified category/date supersedes the old evidence for that event.
            match=events.category.eq(r.category)&events.effective_date.eq(r.effective_date)
            events=events.loc[~match]
        events=pd.concat([events,extra],ignore_index=True)
    events=events.loc[events.usable_date.le('2026-11-12')].sort_values(['usable_date','category']).copy()
    events.to_csv(out/'events_used.csv',index=False,encoding='utf-8-sig')
    expanded=mkstate(old,events)
    lagged=mkstate(old,events,True)
    states={'previous_documented':old,'expanded_policy_only':expanded,'expanded_policy_lag1':lagged}
    for n,f in states.items():f.to_parquet(out/(n+'_signals.parquet'))
    for d in ['2015-01-01','2020-03-01','2024-09-01']:
        e=events.loc[events.usable_date.le(d)]
        pd.testing.assert_frame_equal(scores(e,original.loc[:d].index),scores(events,original.loc[:d].index))
    for f in [expanded,lagged]:
        pd.testing.assert_series_equal(f.credit,old.credit)
        pd.testing.assert_series_equal(f.growth,old.growth)
    receipts=[];rows=[];checks=[];navs=[]
    inputs=SimpleNamespace(hist=hist,benchmark=bm,macro=old)
    first=hist.index[hist.index.searchsorted('2024-08-15',side='right')]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [data/'returns_gs.parquet',data/'returns_v12.parquet',data/'macro.parquet',rawfile,HERE/'protocol.json',Path(__file__)]}
    if addon.exists():manifest[str(addon.relative_to(ROOT))]=hashlib.sha256(addon.read_bytes()).hexdigest()
    (out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for capital in [1000000.,500000.]:
        engine.CAPITAL=capital
        for start in ['2014-01-02','2020-01-02']:
            for bp in [9,20]:
                cases=[('V12',None,False)]
                for n,f in states.items():cases.append((n,f,False))
                cases.extend([(n+'_strictFCF',states[n],True) for n in ['previous_documented','expanded_policy_only']])
                for name,f,strict in cases:
                    if strict:
                        with apply_policy(lambda n,r,p:AvailableRotation(r,f,first,True),KEYS):
                            a=engine.economic(MAIN,hist,None,start,bp/10000)
                        assert_clean(a,first)
                    else:a=run(inputs,'amount',start,MAIN if name!='V12' else 'V12',bp/10000,state=f)
                    runid=f'{int(capital)}_{start}_{bp}bp_{name}'
                    folder=out/'accounts'/runid;folder.mkdir(parents=True,exist_ok=True)
                    for field in ['ledger','weights','pnl','signals','trades']:
                        getattr(a,field).to_parquet(folder/(field+'.parquet'))
                    (folder/'meta.json').write_text(json.dumps({'capital':capital,'base_date':a.base_date,'case':name}),encoding='utf-8')
                    prev=a.ledger.nav.shift(1,fill_value=capital)
                    residual=float((prev+a.pnl.sum(axis=1)-a.ledger.cost-a.ledger.nav).abs().max())
                    assert residual<1e-6
                    assert (a.signals.known<a.signals.date).all()
                    checks.append(dict(runid=runid,residual=residual,days=len(a.ledger),signals=len(a.signals),trades=len(a.trades)))
                    for label,lo,hi in [('full',start,'2026-11-12'),('paper',start,'2025-10-31'),('2015','2015-01-01','2015-12-31'),('2020','2020-01-01','2020-12-31'),('recent','2025-11-06','2026-11-12')]:
                        if hi<start:continue
                        row=metrics(a,lo,hi,capital)
                        rows.append(dict(runid=runid,case=name,capital=capital,account_start=start,bp=bp,window=label,**row))
                    navs.append(a.ledger.nav.rename(runid))
                    print(runid,[(x['window'],round(x['cagr']*100,3),round(x['mdd']*100,3)) for x in rows[-5:] if x['runid']==runid],flush=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(out/'ledger_checks.csv',index=False)
    pd.concat(navs,axis=1).to_csv(out/'daily_nav.csv.gz',compression='gzip')
    changes=[]
    for n,f in states.items():
        ix=f.loc['2014':].index
        bucket=lambda x:'/'.join(ASSETS_BY_STAGE[int(x)]) if pd.notna(x) else ''
        changes.append(dict(case=n,money_days=int(f.loc[ix].money.ne(old.loc[ix].money).sum()),stage_days=int(f.loc[ix].stage.ne(old.loc[ix].stage).sum()),basket_days=int(f.loc[ix].stage.map(bucket).ne(old.loc[ix].stage.map(bucket)).sum())))
    pd.DataFrame(changes).to_csv(out/'signal_changes.csv',index=False)
    print('DONE',tag,'accounts',len(checks),'events',len(events),flush=True)

if __name__=='__main__':main()
