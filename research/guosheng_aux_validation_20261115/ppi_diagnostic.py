from pathlib import Path
import json,re,html
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent
rows=[]
for r in json.loads((P/'ppi_release_catalog.json').read_text(encoding='utf-8')):
    f=P/'nbs'/r['url'].split('/')[-1]
    if not f.exists():continue
    raw=f.read_text(encoding='utf-8')
    published=re.findall(r'(20\d{2})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2})',raw)
    if not published: continue
    r=r|{'released':''.join(published[0][:3]),'release_basis':'visible article timestamp, not URL migration date'}
    for tr in re.findall(r'<tr.*?</tr>',raw,re.S):
        cells=[html.unescape(re.sub('<[^>]+>','',c)).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',tr,re.S)]
        if cells and cells[0].strip()=='加工':
            try: rows.append(r|{'ppi':float(cells[2])})
            except (ValueError,IndexError):pass
p=pd.DataFrame(rows).drop_duplicates('month').sort_values('month');p['period']=pd.PeriodIndex(p.month,freq='M');p=p.set_index('period')
ts=pd.read_csv(P/'cn_ppi.csv',dtype={'month':str});ts.index=pd.PeriodIndex(ts.month,freq='M')
p['current_value']=ts.ppi_mp_p_yoy.reindex(p.index);p['difference']=p.ppi-p.current_value
p.to_csv(P/'ppi_official_values.csv')
q=pd.read_csv(P/'h2_current_vintage_panel.csv');q['known']=pd.to_datetime(q.known)
p['available']=pd.to_datetime(p.released,format='%Y%m%d')
for w in [3,2,4]:p['ppi_x'+str(w)]=p.ppi-p.ppi.reindex(p.index-w).to_numpy()
panel=pd.merge_asof(q.sort_values('known'),p.reset_index(drop=True).sort_values('available'),left_on='known',right_on='available',suffixes=('','_ppi'))
panel.to_csv(P/'h1_official_panel.csv',index=False)
def fit(d,x,layer):
    z=pd.get_dummies(d.phase.astype(str),drop_first=True,dtype=float);z.insert(0,'const',1.)
    if layer>=1:
        for k in ['money','credit','growth','carry']:z[k]=d[k].astype(float)
    if layer>=2:z['price']=d.p1
    z=z.to_numpy(float);v=d[x].to_numpy(float);y=d.y1.to_numpy(float)
    rx=v-z@np.linalg.lstsq(z,v,rcond=None)[0];ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    return (float(rx@ry/(rx@rx)) if rx@rx>1e-12 else np.nan),np.linalg.matrix_rank(z)
out=[]
for w in [3,2,4]:
    d=panel.dropna(subset=['ppi_x'+str(w),'y1','p1'])
    for level in [0,1,2]:
        beta,rank=fit(d,'ppi_x'+str(w),level);out.append(dict(window=w,layer=level,n=len(d),beta_pp=beta*100,rank=rank))
    if w==3:
        rng=np.random.default_rng(20261115); vals=[];N=len(panel)
        for _ in range(3000):
            starts=rng.integers(0,N-11,size=int(np.ceil(N/12)));ids=(starts[:,None]+np.arange(12)).reshape(-1)[:N]
            d=panel.iloc[ids].dropna(subset=['ppi_x3','y1','p1']);vals.append(fit(d,'ppi_x3',2)[0] if len(d)>2 else np.nan)
        good=np.array(vals);good=good[np.isfinite(good)]
        (P/'h1_bootstrap.json').write_text(json.dumps(dict(valid=len(good),percentile_pp=(np.quantile(good,[.025,.975])*100).tolist())))
pd.DataFrame(out).to_csv(P/'h1_results.csv',index=False)
print('official',len(p),'differences',int(p.difference.abs().gt(1e-8).sum()));print(pd.DataFrame(out).to_string(index=False));print((P/'h1_bootstrap.json').read_text())

