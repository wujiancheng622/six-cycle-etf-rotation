"""Transport observed proxy errors to early rows; sensitivities, not inferred truth."""
import json,hashlib
import numpy as np
import pandas as pd
import run_tests as t
from types import SimpleNamespace

OUT=t.HERE/'stress';OUT.mkdir(exist_ok=True)
F,t.FEX=t.make_fallbacks()
A=pd.read_parquet(t.INP/'gs_actual.parquet');P=pd.read_parquet(t.INP/'gs_proxy.parquet')
VA=pd.read_parquet(t.INP/'v12_actual.parquet');VP=pd.read_parquet(t.INP/'v12_proxy.parquet')
START='2014-01-02';SEED=20261114;N=100;BLOCK=20
results=[]
execution_receipts=[]
def safe_economic(name,R,prices,start,slip=.0009):
    # An observed stress failed by -2.8067 RMB, not roundoff. Minimum commission
    # can exceed proceeds of a tiny fractional sale. Defer such dust only when
    # current cash plus all sale proceeds cannot pay fees. Do not change fees,
    # discard a scenario, permit borrowing, or edit the parent engine.
    cal=R.index;dates=cal[cal>=start];keys=list(t.engine.KEYS)
    policy=t.engine.Policy(name,R,prices);pos=np.zeros(len(keys));cash=t.CAPITAL
    ledger=[];weights=[];pnls=[];trades=[]
    for i,day in enumerate(dates):
        j=cal.get_loc(day);known=cal[j-1];nav=cash+pos.sum()
        sw,bw=policy.targets(day,known,i,nav)
        def difference(w):
            if w is None:return np.zeros(len(keys))
            c=w*nav-pos
            return np.where(np.abs(c)>.05*pos,c,0.) if name=='V12' else c
        r=R.iloc[j].to_numpy();pnl=pos*r
        sr=-np.minimum(difference(sw),0)*(1+r)
        br=np.where(sr==0,np.maximum(difference(bw),0)*(1+r),0.)
        pos+=pnl;sales=np.minimum(sr,pos);sc=t.engine.amount_fee(sales,slip)
        if cash+float((sales-sc).sum()) < -1e-8:
            bad=(sales-sc)<0
            receipt=dict(date=day,requested_notional=float(sales[bad].sum()),fees=float(sc[bad].sum()),cash_before=float(cash),strategy=name)
            execution_receipts.append(receipt);print('DEFER_UNAFFORDABLE_DUST_SALE',receipt,flush=True)
            sales[bad]=0.;sc=t.engine.amount_fee(sales,slip)
        buys=t.engine.affordable_amount(br,min(cash,cash+float((sales-sc).sum())),slip)
        bc=t.engine.amount_fee(buys,slip);cost=float(sc.sum()+bc.sum())
        cash+=float(sales.sum()-buys.sum()-cost);pos+=buys-sales;nav=cash+pos.sum();count=0
        for side,values,fees in [('sell',sales,sc),('buy',buys,bc)]:
            for k in np.flatnonzero(values>1e-9):
                trades.append(dict(date=day,known=known,asset=keys[k],side=side,notional=values[k],cost=fees[k]));count+=1
        ledger.append(dict(date=day,nav=nav,cash=cash,receivable=0.,cost=cost,notional=float(sales.sum()+buys.sum()),trade_count=count))
        weights.append(np.r_[pos/nav,cash/nav]);pnls.append(pnl)
    return t.engine.pack(ledger,weights,pnls,trades,policy,cal[cal.get_loc(start)-1])
t.engine.economic=safe_economic

def load_completed_nav(path,tag):
    f=pd.read_csv(path,index_col=0)
    for label in ['GS_original','GS_value_permanent','V12']:
        nav=f[f'{label}_{tag}'];out=dict(strategy=label,scenario=str(tag))
        for prefix,end in [('',t.END),('paper_','2025-10-31')]:
            v=np.r_[t.CAPITAL,nav.loc[:end].to_numpy()];r=v[1:]/v[:-1]-1
            years=(pd.Timestamp(nav.loc[:end].index[-1])-pd.Timestamp('2013-12-31')).days/365.25
            out.update({prefix+'cagr':(v[-1]/v[0])**(1/years)-1,prefix+'vol':r.std(ddof=1)*252**.5,prefix+'mdd':(1-v/np.maximum.accumulate(v)).max()})
        results.append(out)

def observed_residuals(A,P,keys):
    e=np.log1p(A[keys])-np.log1p(P[keys])
    means=e.mean()
    common=e.dropna()
    # Keep contemporaneous cross-asset shape, use each asset's longest overlap mean.
    joint=common-common.mean()+means
    return joint,means

def changed(R,keys,cutoffs,residual,mode,rng):
    out=R.copy();n=len(R)
    if mode=='mean':noise=np.tile(residual.mean().to_numpy(),(n,1))
    else:
        starts=rng.integers(0,len(residual),size=(n+BLOCK-1)//BLOCK)
        ii=np.concatenate([(i+np.arange(BLOCK))%len(residual) for i in starts])[:n]
        noise=residual.to_numpy()[ii]
    for j,k in enumerate(keys):
        mask=R.index<cutoffs[k]
        out.loc[mask,k]=np.expm1(np.log1p(R.loc[mask,k])+noise[mask,j])
    return out

def row(a,label,scenario):
    x=dict(strategy=label,scenario=scenario,**t.stats(a,START,t.END))
    x.update({'paper_'+k:v for k,v in t.stats(a,START,'2025-10-31').items() if k in ('cagr','vol','mdd')})
    results.append(x)

def main():
    protocol=dict(seed=SEED,n=N,block=BLOCK,capital=t.CAPITAL,start=START,end=t.END,commission=.00015,slip=.0009,
                  note='Mean and joint residual scenarios use observed later ETF-minus-proxy errors; no probability/confidence interval claim; no fitted correction adopted; original rules remain fixed.',
                  GS_FCF_substitute='Old value ETF permanent; all other parameters held')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    eg,mg=observed_residuals(A,P,t.PROXY_G)
    ev,mv=observed_residuals(VA,VP,t.PROXY_V)
    eg.to_csv(OUT/'gs_joint_residuals.csv');ev.to_csv(OUT/'v12_joint_residuals.csv')
    meta=dict(gs_mean_log_error=mg.to_dict(),v12_mean_log_error=mv.to_dict(),gs_joint_days=len(eg),v12_joint_days=len(ev),gs_first=eg.index[0],v12_first=ev.index[0])
    (OUT/'residual_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    rng_g=np.random.default_rng(SEED);rng_v=np.random.default_rng(SEED)
    gcut={k:t.COVER['actual_return_from'][k] for k in t.PROXY_G};vcut={k:'2020-01-20' for k in t.PROXY_V}
    for tag in ['base','mean',*range(N)]:
        g=t.H if tag=='base' else changed(t.H,t.PROXY_G,gcut,eg,'mean' if tag=='mean' else 'block',rng_g)
        v=t.V if tag=='base' else changed(t.V,t.PROXY_V,vcut,ev,'mean' if tag=='mean' else 'block',rng_v)
        completed=OUT/f'nav_{tag}.csv.gz'
        if completed.exists():
            load_completed_nav(completed,tag)
            continue
        a=t.gs(g,START);row(a,'GS_original',str(tag))
        ext=pd.concat([g,F],axis=1);ext.attrs.update(g.attrs)
        a2=t.gs(ext,START,alternative=('old_value',t.COVER['actual_return_from']['fcf'],True));row(a2,'GS_value_permanent',str(tag))
        b=t.v12(v,START);row(b,'V12',str(tag))
        if tag in ['base','mean']:
            for label,x in [('GS_original',a),('GS_value_permanent',a2),('V12',b)]:
                x.ledger.to_csv(OUT/f'{label}_{tag}_ledger.csv')
        # Store every NAV, not only favorable scenarios.
        pd.DataFrame({f'GS_original_{tag}':a.ledger.nav,f'GS_value_permanent_{tag}':a2.ledger.nav,f'V12_{tag}':b.ledger.nav}).to_csv(OUT/f'nav_{tag}.csv.gz',compression='gzip')
        if isinstance(tag,int) and (tag+1)%10==0:print('STRESS',tag+1,flush=True)
    f=pd.DataFrame(results);f.to_csv(OUT/'all_scenarios.csv',index=False)
    (OUT/'execution_guard_receipts.json').write_text(json.dumps(execution_receipts,indent=2),encoding='utf8')
    cols=['cagr','vol','mdd','paper_cagr','paper_vol','paper_mdd']
    summary=[]
    for strategy,g in f.groupby('strategy'):
        z=g.loc[~g.scenario.isin(['base','mean'])];base=g.loc[g.scenario.eq('base')].iloc[0];mean=g.loc[g.scenario.eq('mean')].iloc[0]
        for k in cols:summary.append(dict(strategy=strategy,metric=k,base=base[k],mean_only=mean[k],minimum=z[k].min(),p05=z[k].quantile(.05),median=z[k].median(),p95=z[k].quantile(.95),maximum=z[k].max()))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':main()
