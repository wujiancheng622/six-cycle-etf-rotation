"""Fixed factorial attribution tests. No parameter selection; parents read-only."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import hashlib, importlib.util, itertools, json, os, sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'protocol.json').read_text(encoding='utf8'));CAP=CFG['capital']
sp=importlib.util.spec_from_file_location('r9_source',ROOT/'guosheng_cross_review_20261114/compare_evidence.py')
ce=importlib.util.module_from_spec(sp);sys.modules[sp.name]=ce;sp.loader.exec_module(ce)
from run_original import risk_parity, trailing
SKILLS=[Path('C:/Users/Administrator/.codex/skills/bayesian-quant-thinking/SKILL.md'),Path('C:/Users/Administrator/.agents/skills/quant-strategy-research/SKILL.md'),Path('C:/Users/Administrator/.agent-skills/quant-strategy-code-audit/references/video-derived-audit-rules.md'),Path('C:/Users/Administrator/.agent-skills/quant-strategy-code-audit/references/audit-rules-extension.md')]
KEYS=list(ce.KEYS);PERIODS={'paper':('2014-01-02','2025-10-31'),'full':('2014-01-02','2026-11-12'),'2014_2019':('2014-01-02','2019-12-31'),'2020_2022':('2020-01-01','2022-12-31'),'2023_2025':('2023-01-01','2025-12-31'),'2026':('2026-01-01','2026-11-12'),'2026_H2':('2026-07-01','2026-11-12')}
ALL=[];ATTR=[];YEAR=[];REGIME=[];CHECKS=[];DETAIL=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):raise PermissionError('offline test')
    if event=='open' and isinstance(args[0],(str,bytes)) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
        if not Path(args[0]).resolve().is_relative_to(HERE):raise PermissionError('parent write blocked '+str(args[0]))
sys.addaudithook(audit)

def metrics(a,lo,hi):
    z=a.ledger.loc[lo:hi]
    if z.empty:return None
    i=a.ledger.index.get_loc(z.index[0]);base=CAP if i==0 else a.ledger.nav.iloc[i-1];bd=a.base_date if i==0 else a.ledger.index[i-1]
    nav=np.r_[base,z.nav];rr=nav[1:]/nav[:-1]-1;years=(pd.Timestamp(z.index[-1])-pd.Timestamp(bd)).days/365.25
    dd=1-nav/np.maximum.accumulate(nav);k=int(dd.argmax());pk=int(nav[:k+1].argmax());d=pd.to_datetime([bd,*z.index]);dur=0;peakday=d[0];best=nav[0]
    for dt,n in zip(d,nav):
        if n>=best-1e-8:best=n;peakday=dt
        dur=max(dur,(dt-peakday).days)
    result=dict(start=z.index[0],end=z.index[-1],base_date=bd,days=len(z),total=nav[-1]/base-1,cagr=(nav[-1]/base)**(1/years)-1,vol=rr.std(ddof=1)*np.sqrt(252),mdd=float(dd.max()),log_growth=np.log(nav[-1]/base)/years,calmar=((nav[-1]/base)**(1/years)-1)/dd.max(),mean_cash=float((z.cash/z.nav).mean()),annual_turnover=float((z.notional/a.ledger.nav.shift(1,fill_value=CAP).loc[z.index]).sum()/years/2),cost=float(z.cost.sum()),cost_per_year=float((z.cost/a.ledger.nav.shift(1,fill_value=CAP).loc[z.index]).sum()/years),underwater_calendar_days=int(dur),peak=str(d[pk].date()),trough=str(d[k].date()))
    for yr in (3,5):
        vals=[]
        for j,dt in enumerate(d):
            lim=dt-pd.DateOffset(years=yr);idx=d.searchsorted(lim,side='right')-1
            if idx>=0 and idx<j:
                span=(dt-d[idx]).days/365.25;vals.append((nav[j]/nav[idx])**(1/span)-1)
        result[f'worst_{yr}calendar_year_cagr']=min(vals) if vals else None
    return result

class FactorPolicy:
    def __init__(self,R,state,T,W,E,allowed=None,allweather=False,fixed=None):
        self.R=R;self.state=state;self.T=T;self.W=W;self.E=E;self.allowed=allowed or {};self.allweather=allweather;self.fixed=fixed
        self.training=R.copy();self.ex=R.attrs.get('excluded_risk_dates',{})
        for k,day in self.allowed.items():self.training.loc[self.training.index<day,k]=np.nan
        self.receipts=[];self.pending=None;self.last=-2;self.prior_basket=None
    def allocation(self,requested,known):
        eligible=[k for k in requested if k not in self.allowed or len(trailing(self.training,self.ex,known,(k,)))==63]
        weights=pd.Series(0.,index=KEYS)
        if not eligible:return weights,None,None
        train=trailing(self.training,self.ex,known,eligible)
        assert len(train)==63 and np.isfinite(train.to_numpy()).all()
        for k in eligible:
            if k in self.allowed:assert train.index[0]>=self.allowed[k]
        w=np.ones(len(eligible))/len(eligible)
        if self.W and len(eligible)>1:
            cov=train.cov().to_numpy();w=risk_parity(cov);rc=w*(cov@w)
            np.testing.assert_allclose(rc/rc.sum(),1/len(eligible),atol=1e-8,rtol=0)
        cash=(len(requested)-len(eligible))/len(requested)
        weights.loc[eligible]=w*(1-cash)
        return weights,train.index[0],train.index[-1]
    def targets(self,day,known,i,nav):
        macro=self.state.loc[pd.Timestamp(known)];phase=int(macro.stage);basket=tuple(ce.ASSETS_BY_STAGE[phase]);month=day[:7]!=known[:7]
        # Same event calendar for T=0 and T=1: only allocation selection differs.
        trigger=self.pending is None or month or (self.E and basket!=self.prior_basket)
        self.prior_basket=basket
        if trigger:
            firsts=[];lasts=[]
            if self.fixed is not None:w=pd.Series(self.fixed,index=KEYS)
            elif self.allweather:
                terms=[]
                for phase0 in range(1,7):
                    q,a,b=self.allocation(ce.ASSETS_BY_STAGE[phase0],known);terms.append(q)
                    if a:firsts.append(a);lasts.append(b)
                w=sum(terms)/6
            else:
                w,a,b=self.allocation(basket if self.T else KEYS,known)
                if a:firsts.append(a);lasts.append(b)
            self.pending=w.to_numpy();self.last=i
            self.receipts.append(dict(date=day,known=known,phase=phase,selected='/'.join(w.index[w>0]),first_return=min(firsts) if firsts else known,last_return=max(lasts) if lasts else known,cash_target=1-float(w.sum()),month_open=month,**w.to_dict()))
            return self.pending,self.pending
        if i==self.last+1:return None,self.pending
        return None,None

@contextmanager
def zero_fees(flag):
    olda,oldf=ce.engine.amount_fee,ce.engine.fee
    if flag:ce.engine.amount_fee=lambda v,s:np.zeros_like(v);ce.engine.fee=lambda *args:0.
    try:yield
    finally:ce.engine.amount_fee,ce.engine.fee=olda,oldf

def run(mode,start,T,W,E,availability,bp,*,allweather=False,fixed=None,zero=False):
    allowed=ALLOWED[availability]
    fac=lambda n,r,p:FactorPolicy(r,STATE,T,W,E,allowed,allweather,fixed)
    with zero_fees(zero),ce.apply_policy(fac,KEYS):
        if mode=='amount':return ce.engine.economic('R10',VIEWS.hist,None,start,bp/1e4)
        q,adv,up,down=VIEWS.gs_quotes
        return ce.engine.actual('R10',q,adv,up,down,None,start,bp/1e4)

def save(a,mode,start,availability,bp,label,T=None,W=None,E=None,zero=False):
    ident=f'{mode}_{availability}_{start}_{bp}bp_{label}'+('_gross' if zero else '')
    folder=OUT/'accounts'/ident;folder.mkdir(parents=True,exist_ok=True)
    meta=dict(id=ident,mode=mode,account_start=start,availability=availability,bp=bp,label=label,T=T,W=W,E=E,zero_fees=zero,capital=CAP,base_date=a.base_date)
    (folder/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    for f in ('ledger','weights','pnl','trades','signals'):getattr(a,f).to_parquet(folder/f'{f}.parquet')
    prior=a.ledger.nav.shift(1,fill_value=CAP);ret=a.ledger.nav/prior-1
    raw=a.pnl.div(prior,axis=0).assign(fee=-a.ledger.cost/prior)
    np.testing.assert_allclose(raw.sum(axis=1),ret,rtol=0,atol=5e-15)
    scale=np.divide(np.log1p(ret),ret,out=np.ones(len(ret)),where=np.abs(ret)>1e-14)
    linked=raw.mul(scale,axis=0);err=abs(linked.sum(axis=1).sum()-np.log(a.ledger.nav.iloc[-1]/CAP));assert err<1e-10
    linked.to_parquet(folder/'linked_log_attribution.parquet')
    phase=a.signals.set_index('date').phase.reindex(a.ledger.index,method='ffill') if 'phase' in a.signals else pd.Series(0,index=a.ledger.index)
    CHECKS.append(dict(id=ident,days=len(a.ledger),trades=len(a.trades),log_identity_error=err,nav_identity_error=float((prior+a.pnl.sum(axis=1)-a.ledger.cost-a.ledger.nav).abs().max())))
    spans=PERIODS if mode=='amount' else {'actual_full':(start,'2026-11-12'),'2026_H2':('2026-07-01','2026-11-12')}
    for window,(lo,hi) in spans.items():
        m=metrics(a,lo,hi)
        if m is None:continue
        ALL.append(meta|dict(window=window)|m);z=linked.loc[lo:hi];den=z.sum().sum()
        for k,v in z.sum().items():ATTR.append(dict(id=ident,window=window,asset=k,log_contribution=v,share_net_log=v/den,money_pnl=float(a.pnl.loc[lo:hi,k].sum()) if k!='fee' else -float(a.ledger.loc[lo:hi,'cost'].sum())))
        for ph,g in z.groupby(phase.loc[z.index]):REGIME.append(dict(id=ident,window=window,phase=int(ph),days=len(g),net_log=float(g.to_numpy().sum()),share_net_log=float(g.to_numpy().sum()/den)))
    for yr in range(int(start[:4]),2027):
        m=metrics(a,f'{yr}-01-01',f'{yr}-12-31')
        if m:YEAR.append(meta|dict(year=yr)|m)
    if mode=='amount' and T==1 and W==1 and bp==9 and start=='2014-01-02':
        beginning=a.weights.drop(columns='cash').shift(1,fill_value=0)
        for window,(lo,hi) in PERIODS.items():
            ix=a.ledger.loc[lo:hi].index;avg=beginning.loc[ix].mean();r=VIEWS.hist.loc[ix,KEYS]
            exposure=r.mul(avg).mul(scale.loc[ix],axis=0).sum();varying=r.mul(beginning.loc[ix]-avg).mul(scale.loc[ix],axis=0).sum()
            np.testing.assert_allclose(exposure+varying,linked.loc[ix,KEYS].sum(),atol=1e-11,rtol=0)
            for k in KEYS:DETAIL.append(dict(id=ident,window=window,asset=k,mean_begin_weight=avg[k],exposure_log=exposure[k],varying_weight_log=varying[k]))
    print('DONE',ident,{k:round(v*100,3) for k,v in metrics(a,start,'2026-11-12').items() if k in ('cagr','vol','mdd')},flush=True)
    return ident

def main():
    global STATE,VIEWS,ALLOWED
    skills={str(p):sha(p) for p in SKILLS};(OUT/'skill_sources.json').write_text(json.dumps(skills,indent=2),encoding='utf8')
    local,VIEWS=ce.latest_local();_,_,STATE,events=ce.r6_states(local);ce.engine.CAPITAL=CAP
    ALLOWED={'proxy_marked':{},'three_index_date_control':{k:VIEWS.hist.index[VIEWS.hist.index.searchsorted(d,side='right')] for k,d in ce.CFG['index_publication_control']['dates'].items()},'listed_ETF_history_control':VIEWS.first_actual_return}
    sources=[HERE/'protocol.json',HERE/'run_attribution.py',Path(ce.engine.__file__),ROOT/'guosheng_original_reconstruction_20261114/run_original.py',ROOT/'guosheng_policy_completion_20261114_r6/v2/events_used.csv',ROOT/'guosheng_six_regime_long_20261112/data/returns_gs.parquet',ROOT/'guosheng_six_regime_long_20261112/data/returns_v12.parquet',ROOT/'guosheng_original_reconstruction_20261114/results/daily_signal.csv',ROOT/'guosheng_original_reconstruction_20261114/shfe_metals_index.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources};(OUT/'frozen_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    STATE.to_parquet(OUT/'frozen_state.parquet');VIEWS.hist.to_parquet(OUT/'gs_returns.parquet');VIEWS.benchmark.to_parquet(OUT/'v12_returns.parquet')
    # Exact 1m R9 reference before interpreting new 500k diagnostics.
    ce.engine.CAPITAL=1000000.;reference=pd.read_csv(ROOT/'guosheng_cross_review_20261114/evidence/matched_replay.csv')
    for E in [0,1]:
        a=run('amount','2014-01-02',1,1,E,'proxy_marked',9)
        b=ce.run(VIEWS,'amount','2014-01-02','GS_stage_switch' if E else ce.MAIN,.0009,state=STATE)
        pd.testing.assert_frame_equal(a.ledger,b.ledger,check_exact=True)
    ce.engine.CAPITAL=CAP
    # Factorial: same event calendar, instrument input, fees and cash settlement in each block.
    for start in CFG['starts']:
        for bp in CFG['slippage_bps']:
            v=ce.run(VIEWS,'amount',start,'V12',bp/1e4);save(v,'amount',start,'V12',bp,'V12')
            for availability in ALLOWED:
                for T,W,E in itertools.product((0,1),repeat=3):
                    a=run('amount',start,T,W,E,availability,bp);label=f'T{T}W{W}E{E}';save(a,'amount',start,availability,bp,label,T,W,E)
                if bp==9:
                    a=run('amount',start,0,1,0,availability,bp,allweather=True);save(a,'amount',start,availability,bp,'six_baskets_no_timing')
    for bp in [9,20]:
        v=ce.run(VIEWS,'actual',CFG['actual_start'],'V12',bp/1e4);save(v,'actual',CFG['actual_start'],'V12',bp,'V12')
        for T,W,E in itertools.product((0,1),repeat=3):
            a=run('actual',CFG['actual_start'],T,W,E,'proxy_marked',bp);save(a,'actual',CFG['actual_start'],'actual_quotes',bp,f'T{T}W{W}E{E}',T,W,E)
    # Cost and mean-exposure controls; no fitting a profitable new portfolio.
    for availability in ALLOWED:
        for E in [0,1]:
            a=run('amount','2014-01-02',1,1,E,availability,9,zero=True);save(a,'amount','2014-01-02',availability,9,f'T1W1E{E}',1,1,E,zero=True)
            folder=OUT/'accounts'/f'amount_{availability}_2014-01-02_9bp_T1W1E{E}'
            w=pd.read_parquet(folder/'weights.parquet').drop(columns='cash').shift(1,fill_value=0).loc[:'2025-10-31'].mean().reindex(KEYS)
            a=run('amount','2014-01-02',0,0,0,'proxy_marked',9,fixed=w)
            save(a,'amount','2014-01-02',availability,9,f'expost_mean_exposure_E{E}')
    with zero_fees(True):
        v=ce.run(VIEWS,'amount','2014-01-02','V12',.0009);save(v,'amount','2014-01-02','V12',9,'V12',zero=True)
    for name,records in [('comparison',ALL),('asset_attribution',ATTR),('yearly',YEAR),('regime_attribution',REGIME),('checks',CHECKS),('average_exposure_decomposition',DETAIL)]:pd.DataFrame(records).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    unchanged={p:sha(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    (OUT/'completion.json').write_text(json.dumps(dict(accounts=len(CHECKS),account_days=sum(x['days'] for x in CHECKS),unchanged=unchanged,reference_exact_match=True),indent=2),encoding='utf8')
    print('COMPLETE',len(CHECKS),flush=True)

if __name__=='__main__':main()
