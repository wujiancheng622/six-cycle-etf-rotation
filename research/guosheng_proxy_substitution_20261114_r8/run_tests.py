"""Fixed, isolated tests of historical substitutes and proxy measurement error."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'guosheng_parameter_robustness_20261113'),str(ROOT/'guosheng_fcf_cutoff_20261113'),str(ROOT/'guosheng_six_regime_long_20261112')]
from replay_long import engine,apply_policy,Rotation,KEYS,ASSETS_BY_STAGE
from replay_fcf_cutoff import AvailableRotation,trailing
from run_study import risk_parity
VPolicy=engine.Policy
GKEYS=list(KEYS); VKEYS=list(engine.KEYS)
engine.CAPITAL=500000.
DATA=ROOT/'guosheng_six_regime_long_20261112/data'
BASE=ROOT/'guosheng_six_regime_20261112'
INP=HERE/'input'; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
H=pd.read_parquet(DATA/'returns_gs.parquet')
V=pd.read_parquet(DATA/'returns_v12.parquet')
S=pd.read_parquet(ROOT/'guosheng_policy_completion_20261114_r6/v2/expanded_policy_only_signals.parquet')
COVER=json.loads((ROOT/'guosheng_six_regime_long_20261112/coverage.json').read_text(encoding='utf8'))
EX=H.attrs.get('excluded_risk_dates',{})
PROXY_V=['csi300','csi1000','divlow','bond10','metals','energy','soy']
PROXY_G=['metals','fcf','bond','dividend']
END='2026-11-12'; PEND='2026-11-04'; CAPITAL=500000.
rows=[];checks=[];annual=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stats(a,lo,hi):
    z=a.ledger.loc[lo:hi]
    j=a.ledger.index.get_loc(z.index[0]); b=CAPITAL if j==0 else a.ledger.nav.iloc[j-1]
    bd=a.base_date if j==0 else a.ledger.index[j-1]
    nav=np.r_[b,z.nav.to_numpy()]; rr=nav[1:]/nav[:-1]-1
    yrs=(pd.Timestamp(z.index[-1])-pd.Timestamp(bd)).days/365.25
    dd=1-nav/np.maximum.accumulate(nav); k=dd.argmax()
    return dict(start=z.index[0],end=z.index[-1],base_date=bd,days=len(z),total=nav[-1]/b-1,cagr=(nav[-1]/b)**(1/yrs)-1,
                vol=rr.std(ddof=1)*np.sqrt(252),mdd=dd.max(),calmar=((nav[-1]/b)**(1/yrs)-1)/dd.max(),
                mean_cash=float((z.cash/z.nav).mean()),cost=float(z.cost.sum()),trough=([bd]+list(z.index))[k])

def save(a,label,group,bp,start):
    folder=OUT/'accounts'/f'{group}_{start}_{bp}bp_{label}';folder.mkdir(parents=True,exist_ok=True)
    for col in ['ledger','signals','trades','pnl','weights']:getattr(a,col).to_parquet(folder/f'{col}.parquet')
    (folder/'meta.json').write_text(json.dumps(dict(base_date=a.base_date,capital=CAPITAL)),encoding='utf8')
    err=float((a.ledger.nav.shift(1,fill_value=CAPITAL)+a.pnl.sum(axis=1)-a.ledger.cost-a.ledger.nav).abs().max())
    assert err<1e-6 and a.ledger.cash.min()>-1e-7
    checks.append(dict(label=label,group=group,start=start,bp=bp,days=len(a.ledger),trades=len(a.trades),nav_identity_error=err))
    spans=[('full',start,a.ledger.index[-1])]
    if start<'2025-10-31':spans.append(('paper',start,'2025-10-31'))
    for window,lo,hi in spans:rows.append(dict(label=label,group=group,bp=bp,account_start=start,**stats(a,lo,hi),window=window))
    for year in range(int(start[:4]),2027):
        lo=max(start,f'{year}-01-01');hi=min(a.ledger.index[-1],f'{year}-12-31')
        if lo<=hi:annual.append(dict(label=label,group=group,bp=bp,account_start=start,year=year,**stats(a,lo,hi)))
    print(group,start,bp,label,{k:round(v*100,4) for k,v in stats(a,start,a.ledger.index[-1]).items() if k in ['cagr','vol','mdd']},flush=True)

def dated_raw(api,code):
    parts=[]
    for root in [BASE/'data/tushare',DATA/'tushare']:
        for p in (root/api).glob(code+'_*.parquet'):
            f=pd.read_parquet(p)
            if 'trade_date' in f:parts.append(f)
    assert parts,(api,code)
    f=pd.concat(parts).drop_duplicates('trade_date').set_index('trade_date').sort_index()
    f.index=pd.to_datetime(f.index).strftime('%Y-%m-%d')
    return f

def make_fallbacks():
    out={};excluded={};review=[]
    for key,code in [('old_dividend','510880.SH'),('old_value','510030.SH')]:
        f=pd.read_parquet(INP/f'fund_daily_{code}.parquet').set_index('trade_date').sort_index()
        f.index=pd.to_datetime(f.index).strftime('%Y-%m-%d')
        dates=H.index
        q=f.reindex(dates);px=q.close.ffill(); valid=q.vol.gt(0)
        d=pd.read_parquet(INP/f'fund_div_{code}.parquet')
        cash=pd.Series(0.,index=dates)
        if len(d):
            d=d.loc[d.div_proc.eq('实施')].drop_duplicates(['ex_date','div_cash'])
            for e in d.itertuples():
                dt=pd.Timestamp(e.ex_date).date().isoformat()
                if dt in cash.index:cash.loc[dt]+=float(e.div_cash)
        splits=pd.Series(1.,index=dates)
        if code=='510030.SH':splits.loc['2021-06-21']=8.0 # first quoted day after documented 2021-06-18 1:8 split
        ret=(px*splits+cash)/px.shift()-1
        # First study row is not a trade day; recover its true predecessor from raw history.
        first=dates[0];prior=f.index[f.index<first][-1]
        ret.iloc[0]=(px.iloc[0]+cash.iloc[0])/f.at[prior,'close']-1
        assert ret.notna().all() and ret.abs().max()<.25,(code,ret.abs().nlargest(3))
        out[key]=ret
        excluded[key]=list(dates[~(valid&valid.shift(1,fill_value=True))])
        review.append(dict(code=code,dividend_events=int(cash.ne(0).sum()),suspension_days=int((~valid).sum()),split_date='2021-06-18' if code=='510030.SH' else None,first_market_reaction='2021-06-21' if code=='510030.SH' else None,max_abs_return=float(ret.abs().max())))
    z=pd.DataFrame(out);z.to_parquet(INP/'fallback_returns.parquet')
    (OUT/'fallback_actions.json').write_text(json.dumps(review,indent=2),encoding='utf8')
    return z,excluded

class SubstitutionPolicy:
    def __init__(self,risk,states,keys,fallback,first_fcf,permanent=False):
        self.risk=risk;self.states=states;self.keys=keys;self.fallback=fallback;self.first_fcf=first_fcf;self.permanent=permanent
        self.receipts=[];self.pending=None;self.last=-2
        self.excluded={**EX,**FEX}
    def targets(self,day,known,i,nav):
        if self.pending is None or day[:7]!=known[:7]:
            state=self.states.loc[pd.Timestamp(known)]; requested=ASSETS_BY_STAGE[int(state.stage)]
            available=list(requested);missing=False
            fcf_train=self.risk.loc[self.first_fcf:known,'fcf'].dropna()
            allowed=len(fcf_train.drop(index=self.excluded.get('fcf',()),errors='ignore'))>=63 and not self.permanent
            if 'fcf' in available and not allowed:
                j=available.index('fcf');available.pop(j)
                if self.fallback is None:missing=True
                else:available.insert(j,self.fallback)
            exclusions={d for k in available for d in self.excluded.get(k,[])}
            train=self.risk.loc[:known,available].drop(index=sorted(exclusions),errors='ignore').dropna().tail(63)
            if 'fcf' in available:assert train.index[0]>=self.first_fcf
            assert len(train)==63
            w=np.ones(1) if len(available)==1 else risk_parity(train.cov().to_numpy())
            ct=1/len(requested) if missing else 0.
            weights=pd.Series(w*(1-ct),index=available).reindex(self.keys,fill_value=0).to_numpy()
            self.pending=weights;self.last=i
            self.receipts.append(dict(date=day,known=known,phase=int(state.stage),requested='/'.join(requested),selected='/'.join(available),first_return=train.index[0],last_return=train.index[-1],cash_target=ct,fcf_eligible=allowed,**dict(zip(self.keys,weights))))
            return weights,weights
        return (None,self.pending) if i==self.last+1 else (None,None)

def gs(R,start,bp=9,risk=None,alternative=None):
    risk=R if risk is None else risk
    keys=list(R.columns)
    if alternative is None:factory=lambda n,r,p:Rotation('GS_monthly',risk,S)
    else:
        fallback,cut,perm=alternative
        factory=lambda n,r,p:SubstitutionPolicy(risk,S,keys,fallback,cut,perm)
    with apply_policy(factory,keys):return engine.economic('GS_monthly',R,None,start,bp/1e4)

def v12(R,start,bp=9,risk=None):
    risk=R if risk is None else risk
    with apply_policy(lambda n,r,p:VPolicy('V12',risk,None),VKEYS):return engine.economic('V12',R,None,start,bp/1e4)

def proxy_inputs():
    Q=pd.read_csv(ROOT/'p75_risk15_codex_20261106/input/economic_returns_all_proxy.csv',index_col=0)
    P=H.copy().loc[:PEND];P['metals']=Q.metals.reindex(P.index)
    for key,code in [('fcf','980092.SZ'),('dividend','H20955.CSI')]:
        f=dated_raw('index_daily',code).close.astype(float)
        P[key]=f.pct_change(fill_method=None).reindex(P.index)
    obj=json.loads((BASE/'source/chinabond_30y_wealth.json').read_text(encoding='utf8'))
    b=pd.Series(obj['CFZS_00'],dtype=float)
    b.index=pd.to_datetime(pd.to_numeric(b.index),unit='ms',utc=True).tz_convert('Asia/Shanghai').tz_localize(None).strftime('%Y-%m-%d')
    b=b.sort_index();P['bond']=b.reindex(P.index,method='ffill').pct_change(fill_method=None)
    A=H.copy().loc[:PEND]
    for key,dt in COVER['actual_return_from'].items():A.loc[A.index<dt,key]=np.nan
    Vactual=V.loc['2020-01-20':PEND].copy()
    Vproxy=Vactual.copy();Vproxy[PROXY_V]=Q.loc[Vactual.index,PROXY_V]
    # Index starts may have missing first rows; no invented observations in overlap.
    for name,x in [('gs_proxy',P),('gs_actual',A),('v12_proxy',Vproxy),('v12_actual',Vactual)]:x.to_parquet(INP/(name+'.parquet'))
    return A,P,Vactual,Vproxy

def asset_metrics(A,P,keys,label):
    data=[]
    for k in keys:
        pair=pd.concat([A[k],P[k]],axis=1).dropna();pair.columns=['a','p']
        e=pair.a-pair.p; lr=np.log1p(pair.a)-np.log1p(pair.p)
        years=(pd.Timestamp(pair.index[-1])-pd.Timestamp(pair.index[0])).days/365.25
        data.append(dict(strategy=label,asset=k,start=pair.index[0],end=pair.index[-1],days=len(pair),correlation=pair.a.corr(pair.p),tracking_error=e.std(ddof=1)*252**.5,
                         annual_log_drift=lr.mean()*252,actual_cagr=np.expm1(np.log1p(pair.a).sum()/years),proxy_cagr=np.expm1(np.log1p(pair.p).sum()/years),max_daily_error=e.abs().max()))
    return data

def overlap_suite(A,P,which):
    joint=A.notna().all(axis=1)&P.notna().all(axis=1);dates=A.index[joint]
    A=A.loc[dates].copy();P=P.loc[dates].copy()
    start=dates[63];fun=gs if which=='GS' else v12
    results={}
    for label,risk,R in [('actual_actual',A,A),('proxy_actual',P,A),('actual_proxy',A,P),('proxy_proxy',P,P)]:
        a=fun(R,start,9,risk=risk);save(a,label,which+'_overlap',9,start);results[label]=a
    a=results['actual_actual'];b=results['proxy_proxy']
    sa=a.signals.set_index('date');sb=b.signals.set_index('date');ks=list(A.columns)
    target_error=(sa[ks]-sb[ks]).abs()
    diag=dict(strategy=which,start=start,end=A.index[-1],portfolio_corr=a.ledger.nav.pct_change().corr(b.ledger.nav.pct_change()),
              mean_target_abs_sum=float(target_error.sum(axis=1).mean()),p95_target_abs_sum=float(target_error.sum(axis=1).quantile(.95)),max_target_abs_sum=float(target_error.sum(axis=1).max()))
    return diag

def main():
    global FEX
    (OUT/'protocol.sha256').write_text(sha(HERE/'protocol.json'),encoding='ascii')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'returns_gs.parquet',DATA/'returns_v12.parquet',ROOT/'guosheng_policy_completion_20261114_r6/v2/expanded_policy_only_signals.parquet',HERE/'run_tests.py',HERE/'protocol.json']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    F,FEX=make_fallbacks();ext=pd.concat([H,F],axis=1);ext.attrs.update(H.attrs)
    rule='2024-08-16';actual=COVER['actual_return_from']['fcf']
    for start in ['2014-01-02','2020-01-02']:
        for bp in [9,20]:
            a=gs(H,start,bp);save(a,'original_backfill','substitute',bp,start)
            oldpath=ROOT/f'guosheng_policy_completion_20261114_r6/v2/accounts/500000_{start}_{bp}bp_expanded_policy_only/ledger.parquet'
            pd.testing.assert_frame_equal(a.ledger,pd.read_parquet(oldpath))
            save(v12(V,start,bp),'V12','substitute',bp,start)
            for label,fallback,cut,perm in [('cash_rule',None,rule,False),('cash_ETF',None,actual,False),('dividend_to_FCF','old_dividend',actual,False),('value_to_FCF','old_value',actual,False),('dividend_permanent','old_dividend',actual,True),('value_permanent','old_value',actual,True)]:
                a=gs(ext,start,bp,alternative=(fallback,cut,perm));save(a,label,'substitute',bp,start)
                if label=='cash_rule':
                    p=ROOT/f'guosheng_policy_completion_20261114_r6/v2/accounts/500000_{start}_{bp}bp_expanded_policy_only_strictFCF/ledger.parquet'
                    np.testing.assert_allclose(a.ledger.nav,pd.read_parquet(p).nav,rtol=0,atol=1e-6)
    A,P,VA,VP=proxy_inputs()
    ass=asset_metrics(A,P,PROXY_G,'GS')+asset_metrics(VA,VP,PROXY_V,'V12')
    pd.DataFrame(ass).to_csv(OUT/'asset_overlap.csv',index=False)
    diags=[overlap_suite(A,P,'GS'),overlap_suite(VA,VP,'V12')]
    # Same recent window for the two strategies as an additional apples-to-apples diagnostic.
    common=A.dropna().index[63]
    for label,risk,R in [('actual_actual',VA,VA),('proxy_proxy',VP,VP)]:
        save(v12(R,common,9,risk=risk),label,'V12_common_recent',9,common)
    pd.DataFrame(diags).to_csv(OUT/'overlap_target_changes.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'checks.csv',index=False)
    print('FINISHED',len(checks),'accounts',flush=True)

if __name__=='__main__':main()
