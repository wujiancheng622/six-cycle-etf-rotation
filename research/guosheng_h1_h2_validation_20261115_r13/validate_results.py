"""Separate checks of feature construction, original outcomes, regression arithmetic and source integrity."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results'
f=pd.read_csv(O/'monthly_auxiliary_grid.csv');base=f.loc[f.run.eq('base')].copy()
checks=[]
def check(name,condition,detail):
    assert condition,(name,detail)
    checks.append(dict(check=name,passed=True,detail=detail))
check('154 original completed intervals',len(base)==154 and base.date.is_unique,'All original monthly targets except unfinished final interval')
check('known before target before completed label',bool((base.known<base.date).all() and (base.date<base.end).all()),'No target-date return included in future label')
check('H2 positive-current-growth selection',int(base.growth.gt(0).sum())==77,'Includes carried stage6, not a stage-name filter')

# Recompute labels and prior-price controls directly from frozen daily asset returns.
ret=pd.read_parquet(R/'guosheng_six_regime_long_20261112/data/returns_gs.parquet')
metal=pd.read_csv(R/'guosheng_original_reconstruction_20261114/shfe_metals_index.csv',index_col='date')
ret.loc[metal.index,'metals']=metal.close/metal.previous_close-1
keys=list(ret.columns);maxlabel=0.;maxprice=0.
for row in base.itertuples():
    selected=row.basket.split('/');out=[k for k in keys if k not in selected]
    start=ret.index.get_loc(row.date);end=ret.index.get_loc(row.end);known=ret.index.get_loc(row.known)
    future=(1+ret.iloc[start+1:end+1]).prod()-1
    past=(1+ret.iloc[known-20:known+1]).prod()-1
    y1=future.metals-future.bond;y2=future[selected].mean()-future[out].mean()
    p1=past.metals-past.bond;p2=past[selected].mean()-past[out].mean()
    maxlabel=max(maxlabel,abs(y1-row.y1),abs(y2-row.y2));maxprice=max(maxprice,abs(p1-row.price1),abs(p2-row.price2))
check('direct daily compounding agrees with labels',maxlabel<1e-12,maxlabel)
check('direct 21-session compounding agrees with controls',maxprice<1e-12,maxprice)

official=pd.read_csv(O/'official_pmi_monthly.csv',index_col='month')
pp=pd.read_csv(O/'official_ppi_monthly.csv',index_col='month')
maxx1=0.;x2count=0;monthstested=0
for row in base.itertuples():
    m=row.ppi_month_lag0;before=str(pd.Period(m,freq='M')-3)
    x=pp.at[m,'value']-pp.at[before,'value'];maxx1=max(maxx1,abs(x-row.x1_official_d3_lag0))
    check('PPI_time_'+row.date,pp.at[m,'released']<=row.known and pp.at[before,'released']<=row.known,m)
    m=row.original_pmi_month;months=list(pd.period_range(pd.Period(m)-26,m,freq='M').astype(str))
    if pd.notna(row.x2_official_sync_d3_lag0):
        v=official.loc[months,'value'].to_numpy()
        u_now=np.mean(v[-12:])/np.mean(v[-24:-12])-1
        u_before=np.mean(v[-15:-3])/np.mean(v[:12])-1
        sign=1 if u_now-u_before>0 else -1
        assert float(sign<0)==row.x2_official_sync_d3_lag0
        assert official.loc[months,'released'].le(row.known).all()
        x2count+=1;monthstested+=len(months)
check('H1 independent arithmetic',maxx1<1e-12,maxx1)
check('H2 direct-window formula and availability chain',x2count>0,dict(targets=x2count,input_month_checks=monthstested))
from prepare_inputs import transform
const=pd.Series([50.]*27);delta,sign=transform(const)
check('H2 warmup and zero rule',delta.iloc[:26].isna().all() and delta.iloc[26]==0 and sign[-1]==-1,'27 observations; zero is negative vote as frozen original')
miss=const.copy();miss.iloc[5]=np.nan;delta,_=transform(miss)
check('missing not silently neutral/negative',pd.isna(delta.iloc[-1]),'Missing raw month blocks complete feature')

# Baseline performance is a read-only reference, not a new adjusted strategy.
ref=[]
for name in ['monthly','V12']:
    folder=R/f'guosheng_final_decision_20261114/results/accounts/{name}_2014-01-02_9bp'
    L=pd.read_parquet(folder/'ledger.parquet');meta=json.loads((folder/'meta.json').read_text())
    p=np.r_[meta['capital'],L.nav.to_numpy()];daily=np.diff(p)/p[:-1];years=(pd.Timestamp(L.index[-1])-pd.Timestamp(meta['base_date'])).days/365.25
    ref.append(dict(strategy=name,start=L.index[0],end=L.index[-1],capital=meta['capital'],cagr=(p[-1]/p[0])**(1/years)-1,vol=daily.std(ddof=1)*252**.5,mdd=(1-p/np.maximum.accumulate(p)).max(),new_trade_rule=False))
pd.DataFrame(ref).to_csv(O/'unchanged_baseline_reference.csv',index=False)
if (O/'hypothesis_results.csv').exists():
    import ast
    results=pd.read_csv(O/'hypothesis_results.csv');maxfit=0.;recomputed=[]
    for r in results.itertuples():
        if r.status!='estimated':continue
        q=base.loc[base[r.variable].notna()&(base.growth.gt(0) if r.hypothesis=='H2' else True)].copy()
        cols=ast.literal_eval(r.controls);columns=[]
        for c in cols:
            if c=='intercept':column=np.ones(len(q))
            elif c.startswith('phase_'):column=q.phase.eq(int(c[6:])).to_numpy().astype(float)
            elif c=='prior_21_price':column=q['price1' if r.hypothesis=='H1' else 'price2'].to_numpy()
            else:column=q[c].to_numpy(dtype=float)
            columns.append(column)
        Z=np.column_stack(columns);X=q[r.variable].to_numpy();Y=q['y1' if r.hypothesis=='H1' else 'y2'].to_numpy()
        design=np.column_stack([Z,X]);norm=np.linalg.norm(design,axis=0)
        # QR on scaled full design instead of the analysis's partial-regression calculation.
        Q,U=np.linalg.qr(design/norm,mode='reduced');coef=np.linalg.solve(U,Q.T@Y)/norm
        err=abs(coef[-1]-r.effect);maxfit=max(maxfit,err)
        paths=pd.read_csv(O/f'bootstrap_{r.hypothesis}_{r.evidence}_{r.layer}.csv')
        valid=paths.loc[paths.status.eq('estimated'),'effect'];lo,hi=np.quantile(valid,[.025,.975])*100
        assert len(paths)==3000 and len(valid)==r.bootstrap_valid and abs(lo-r.lower_pp)<1e-10 and abs(hi-r.upper_pp)<1e-10
        recomputed.append(dict(hypothesis=r.hypothesis,evidence=r.evidence,layer=r.layer,coefficient_error=err,paths=len(paths),valid=len(valid)))
    check('independent QR regression and bootstrap quantiles',maxfit<1e-10,dict(max_effect_error=maxfit,models=len(recomputed)))
    pd.DataFrame(recomputed).to_csv(O/'independent_model_verification.csv',index=False)
frozen=json.loads((H/'freeze_receipt.json').read_text(encoding='utf8'))
same=all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in frozen['hashes'].items())
check('original source/strategy/account hashes unchanged',same,len(frozen['hashes']))
pd.DataFrame(checks).to_csv(O/'verification_checks.csv',index=False)
(O/'verification_summary.json').write_text(json.dumps(dict(checks=len(checks),all_passed=True,label_error=maxlabel,price_error=maxprice,H2_manual_formula_targets=x2count,baseline_accounts_modified=False,new_strategy_backtests=0),indent=2),encoding='utf8')
print(json.dumps(dict(checks=len(checks),all_passed=True,label_error=maxlabel,price_error=maxprice,baselines=ref),ensure_ascii=False),flush=True)
