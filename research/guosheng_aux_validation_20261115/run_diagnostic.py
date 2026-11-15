from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent; R=P.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'ah_premium_20261106'))
from data_access import OfficialData
f=P/'frozen_protocol.md'
assert hashlib.sha256(f.read_bytes()).hexdigest().upper()==json.loads((P/'protocol_hash.json').read_text(encoding='utf-8-sig'))['Hash']
api=OfficialData()
for endpoint,fields in [('cn_ppi','month,ppi_mp_p_yoy'),('cn_pmi','month,pmi010500')]:
    path=P/(endpoint+'.csv')
    if not path.exists():
        df=api.query(endpoint,dict(start_m='201001',end_m='202610'),fields)
        df.to_csv(path,index=False)
        (P/(endpoint+'_receipt.json')).write_text(json.dumps(dict(endpoint=endpoint,fields=fields,rows=len(df),retrieved=datetime.datetime.now(datetime.timezone.utc).isoformat(),quality='current vintage, historical release versions NOT certified',sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    print(endpoint,len(pd.read_csv(path)),flush=True)
a=pd.read_csv(R/'guosheng_timing_attribution_20261115/results/asset_month_detail.csv')
a=a[a.run.eq('base')]
m=a[a.asset.eq('growth')][['date','known','phase','money','credit','growth','direct']].set_index('date')
fw=a.pivot(index='date',columns='asset',values='forward'); lag=a.pivot(index='date',columns='asset',values='prev_21'); sel=a.pivot(index='date',columns='asset',values='selected')
m['y1']=fw.metals-fw.bond; m['p1']=lag.metals-lag.bond
m['y2']=fw.where(sel).mean(axis=1)-fw.where(~sel).mean(axis=1)
m['p2']=lag.where(sel).mean(axis=1)-lag.where(~sel).mean(axis=1)
p=pd.read_csv(R/'guosheng_original_reconstruction_20261114/results/pmi_dates_used.csv')
p['available']=pd.to_datetime(p.available); p['period']=pd.PeriodIndex(p.month,freq='M')
orders=pd.read_csv(P/'cn_pmi.csv',dtype={'month':str});orders.index=pd.PeriodIndex(orders.month,freq='M');orders=orders.sort_index().pmi010500.astype(float)
orders=orders.reindex(pd.period_range(orders.index.min(),orders.index.max(),freq='M'))
smooth=orders.rolling(12).mean(); pulse=smooth/smooth.shift(12)-1
for w in [3,2,4]:
    change=pulse-pulse.shift(w)
    p['order'+str(w)]=p.period.map(pd.Series(np.where(change.gt(0),1,-1),index=change.index).where(change.notna()))
for key in ['manufacturing','nonmanufacturing','private']:
    p[key+'_sign']=np.where(p[key+'_change'].gt(0),1,-1)
q=pd.merge_asof(m.reset_index().assign(known=lambda x:pd.to_datetime(x.known)).sort_values('known'),p.sort_values('available'),left_on='known',right_on='available',suffixes=('','_source')).set_index('date')
assert q.available.le(q.known).all()
q['x2']=(q.order3.eq(-1)).astype(float).where(q.order3.notna())
q['carry']=(~q.direct).astype(float)
# H1 needs independently verified monthly release dates. No synthetic calendar.
q.to_csv(P/'h2_current_vintage_panel.csv')
rows=[]
def fit(df,x,layer):
    controls=pd.get_dummies(df.phase.astype(str),prefix='phase',drop_first=True,dtype=float)
    controls.insert(0,'constant',1.)
    if layer>=1:
        for c in ['money','credit','growth','carry','manufacturing_sign','nonmanufacturing_sign','private_sign']: controls[c]=df[c].astype(float)
    if layer>=2: controls['price']=df.p2
    z=controls.to_numpy(float);v=df[x].to_numpy(float); y=df.y2.to_numpy(float)
    rank=np.linalg.matrix_rank(z); rx=v-z@np.linalg.lstsq(z,v,rcond=None)[0]; ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    if rx@rx<1e-12:return np.nan,rank,float(rx@rx)
    return float(rx@ry/(rx@rx)),rank,float(rx@rx)
eligible=q.growth.gt(0)&q.order3.notna()
for w in [3,2,4]:
    q['x'+str(w)]=q['order'+str(w)].eq(-1).astype(float).where(q['order'+str(w)].notna())
    take=q[eligible & q['x'+str(w)].notna()].copy()
    for layer in [0,1,2]:
        beta,rank,var=fit(take,'x'+str(w),layer)
        rows.append(dict(window=w,layer=layer,n=len(take),weak=int(take['x'+str(w)].sum()),beta_pp=100*beta,control_rank=rank,residual_x_ss=var))
    if w==3:
        rng=np.random.default_rng(20261115); vals=[];N=len(q)
        for _ in range(3000):
            starts=rng.integers(0,N-12+1,size=int(np.ceil(N/12)))
            ids=(starts[:,None]+np.arange(12)).reshape(-1)[:N]
            b=q.iloc[ids];b=b[b.growth.gt(0)&b.order3.notna()]
            vals.append(fit(b,'x3',2)[0] if len(b)>2 else np.nan)
        good=np.asarray(vals);good=good[np.isfinite(good)]
        (P/'h2_bootstrap.json').write_text(json.dumps(dict(n=3000,valid=len(good),percentile_pp=(np.quantile(good,[.025,.975])*100).tolist(),quality='exploratory current-vintage only'),indent=2))
pd.DataFrame(rows).to_csv(P/'h2_exploratory_results.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print((P/'h2_bootstrap.json').read_text())
