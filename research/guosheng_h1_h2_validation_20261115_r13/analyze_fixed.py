"""Fixed H1/H2 diagnostic only. Never converts fitted coefficients into orders."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results'
CFG=json.loads((H/'protocol.json').read_text(encoding='utf8'))
TOL=1e-10

def rank(a):
    if not a.size:return 0
    norms=np.linalg.norm(a,axis=0);scaled=a/np.where(norms>0,norms,1)
    s=np.linalg.svd(scaled,compute_uv=False)
    return int((s>TOL*(s[0] if len(s) else 1)).sum())

def controls(df,h,layer):
    z=pd.DataFrame({'intercept':np.ones(len(df))},index=df.index)
    for phase in range(2,7):z[f'phase_{phase}']=df.phase.eq(phase).astype(float)
    if layer in ['macro','price']:
        for c in ['money','credit','growth','carried']:z[c]=df[c].astype(float)
        if h=='H2':
            for c in ['manufacturing_vote','nonmanufacturing_vote','private_vote']:z[c]=df[c].astype(float)
    if layer=='price':z['prior_21_price']=df['price1' if h=='H1' else 'price2']
    return z

def fit(df,h,x,layer,fixed_names=None):
    yname='y1' if h=='H1' else 'y2'
    eligible=df[x].notna() & (df.growth.gt(0) if h=='H2' else True)
    z=controls(df,h,layer);eligible &= z.notna().all(axis=1)&df[yname].notna()
    z=z.loc[eligible];xx=df.loc[eligible,x].to_numpy(dtype=float);y=df.loc[eligible,yname].to_numpy(dtype=float)
    result=dict(n=len(y),hypothesis=h,variable=x,layer=layer,status='unidentified')
    keep=[];dropped=[]
    if fixed_names is None:
        for k in z:
            if rank(z[keep+[k]].to_numpy())>len(keep):keep.append(k)
            else:dropped.append(k)
    else:keep=fixed_names
    Z=z[keep].to_numpy(dtype=float);result.update(controls=keep,dropped=dropped,rank_z=rank(Z),df=len(y)-len(keep)-1)
    if not len(y) or rank(Z)!=len(keep):result['reason']='control_rank_failure';return result,None
    norm=np.linalg.norm(Z,axis=0);Zs=Z/np.where(norm>0,norm,1)
    rx=xx-Zs@np.linalg.lstsq(Zs,xx,rcond=1e-12)[0]
    ry=y-Zs@np.linalg.lstsq(Zs,y,rcond=1e-12)[0]
    result.update(x_residual_std=float(np.std(rx)),x_raw_std=float(np.std(xx)),x_residual_energy=float(rx@rx),x_residual_fraction=float(rx@rx/np.sum((xx-xx.mean())**2)) if np.std(xx)>0 else 0)
    if rank(np.column_stack([Z,xx]))!=len(keep)+1 or result['df']<1:
        result['reason']='no_X_incremental_variation_or_residual_df';return result,None
    beta=float(rx@ry/(rx@rx));pred=Zs@np.linalg.lstsq(Zs,y-beta*xx,rcond=1e-12)[0]+beta*xx
    full=np.column_stack([Zs,xx]);direct=np.linalg.lstsq(full,y,rcond=1e-12)[0][-1]
    assert abs(beta-direct)<1e-9
    ss=np.sum((y-y.mean())**2)
    result.update(status='estimated',effect=beta,effect_pp=beta*100,fwl_check_error=abs(beta-direct),condition=float(np.linalg.cond(full/np.linalg.norm(full,axis=0))),
        delta_r2=float((ry@ry-np.sum((y-pred)**2))/ss) if ss else None,
        y_mean=float(y.mean()),mean_y_x0=float(y[xx==0].mean()) if np.any(xx==0) else None,mean_y_x1=float(y[xx==1].mean()) if np.any(xx==1) else None,
        n_x0=int((xx==0).sum()),n_x1=int((xx==1).sum()),date_start=str(df.loc[eligible,'date'].min()),date_end=str(df.loc[eligible,'end'].max()))
    parts=pd.DataFrame({'date':df.loc[eligible,'date'].to_numpy(),'rx':rx,'ry':ry,'numerator':rx*ry,'beta_contribution':rx*ry/(rx@rx)})
    return result,parts

def bootstrap(grid,h,x,layer,names):
    rng=np.random.default_rng(CFG['bootstrap']['seed']);N=len(grid);width=12
    draws=rng.integers(0,N-width+1,size=(3000,int(np.ceil(N/width))))
    betas=[];fail={};paths=[]
    for i,seeds in enumerate(draws):
        ix=np.concatenate([np.arange(s,s+width) for s in seeds])[:N]
        d=grid.iloc[ix].reset_index(drop=True)
        rr,_=fit(d,h,x,layer,names)
        if rr['status']=='estimated':betas.append(rr['effect']);paths.append(dict(path=i,status='estimated',effect=rr['effect']))
        else:
            reason=rr['reason'];fail[reason]=fail.get(reason,0)+1;paths.append(dict(path=i,status=reason,effect=np.nan))
    q=np.quantile(betas,[.025,.975]) if betas else [np.nan,np.nan]
    return dict(bootstrap_attempted=3000,bootstrap_valid=len(betas),bootstrap_failures=fail,lower_pp=float(q[0]*100),upper_pp=float(q[1]*100)),paths

def main():
    frozen=json.loads((H/'freeze_receipt.json').read_text(encoding='utf8'))
    assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in frozen['hashes'].items())
    f=pd.read_csv(O/'monthly_auxiliary_grid.csv');g=f.loc[f.run.eq('base')].sort_values('date').reset_index(drop=True)
    assert len(g)==154 and g.date.str[:7].nunique()==154
    results=[];aux=[];segments=[];events=[];supports=[];parts_all=[]
    for scope in ['official','vendor']:
        for h,x in [('H1',f'x1_{scope}_d3_lag0'),('H2',f'x2_{scope}_sync_d3_lag0')]:
            # Same complete-case sample across every nested model of each hypothesis.
            required=controls(g,h,'price').notna().all(axis=1)&g[x].notna()
            grid=g.copy();grid.loc[~required,x]=np.nan
            for layer in ['stage','macro','price']:
                fitres,parts=fit(grid,h,x,layer);fitres['evidence']=scope
                if fitres['status']=='estimated':
                    boot,paths=bootstrap(grid,h,x,layer,fitres['controls']);fitres.update(boot)
                    pd.DataFrame(paths).to_csv(O/f'bootstrap_{h}_{scope}_{layer}.csv',index=False)
                    if parts is not None:parts_all.append(parts.assign(hypothesis=h,evidence=scope,layer=layer))
                results.append(fitres);print('MAIN',json.dumps(fitres,ensure_ascii=False),flush=True)
            final=results[-1]
            eligible=grid[x].notna()&(grid.growth.gt(0) if h=='H2' else True)
            for key,z in grid.loc[eligible].groupby(['phase','money','credit','growth','carried']):
                supports.append(dict(hypothesis=h,evidence=scope,phase=key[0],money=key[1],credit=key[2],growth=key[3],carried=key[4],n=len(z),x_min=z[x].min(),x_max=z[x].max(),n0=int(z[x].eq(0).sum()),n1=int(z[x].eq(1).sum()),within_cell_variation=z[x].nunique()>1))
            for label,lo,hi in CFG['segments']+[[str(y),f'{y}-01-01',f'{y}-12-31'] for y in range(2014,2027)]:
                z=grid.loc[grid.date.between(lo,hi)];r,_=fit(z,h,x,'price');r.update(evidence=scope,segment=label);segments.append(r)
            # Event definitions were recorded before taking in new values.
            category=np.sign(grid[x]) if h=='H1' else pd.Series(np.where(grid.growth.gt(0)&grid[x].eq(1),1,0),index=grid.index)
            category=pd.Series(category,index=grid.index).where(grid[x].notna())
            group=(category.ne(category.shift())|category.isna()).cumsum()
            for event,indices in grid.groupby(group).groups.items():
                if category.loc[indices].isna().all() or (h=='H2' and category.loc[indices].iloc[0]!=1):continue
                sub=grid.drop(index=indices);r,_=fit(sub,h,x,'price',final.get('controls'))
                events.append(dict(hypothesis=h,evidence=scope,event=int(event),first=grid.loc[indices,'date'].iloc[0],last=grid.loc[indices,'end'].iloc[-1],months=len(indices),event_state=category.loc[indices].iloc[0],status=r['status'],effect_pp=r.get('effect_pp'),reason=r.get('reason')))
            variants=[]
            for n in [2,4]:variants.append((f'neighbor_{n}',f'x1_{scope}_d{n}_lag0' if h=='H1' else f'x2_{scope}_sync_d{n}_lag0'))
            for n in [1,3]:variants.append((f'delay_{n}',f'x1_{scope}_d3_lag{n}' if h=='H1' else f'x2_{scope}_sync_d3_lag{n}'))
            if h=='H2':variants.append(('latest_month',f'x2_{scope}_latest_d3_lag0'))
            for label,xx in variants:
                same=grid.copy();ok=same[x].notna()&same[xx].notna();same.loc[~ok,xx]=np.nan;same.loc[~ok,x]=np.nan
                br,_=fit(same,h,x,'price');ar,_=fit(same,h,xx,'price');ar.update(evidence=scope,variant=label,baseline_same_dates_pp=br.get('effect_pp'),baseline_n=br['n']);aux.append(ar)
    pd.DataFrame(results).to_csv(O/'hypothesis_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(aux).to_csv(O/'registered_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(segments).to_csv(O/'segments_and_years.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(events).to_csv(O/'leave_event_out.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(supports).to_csv(O/'support_cells.csv',index=False,encoding='utf-8-sig')
    if parts_all:pd.concat(parts_all).to_csv(O/'partial_regression_contributions.csv',index=False)
    # Matched-date source-vintage comparison, never treat provider copies as independent economic observations.
    comparisons=[]
    for h,of,vf in [('H1','x1_official_d3_lag0','x1_vendor_d3_lag0'),('H2','x2_official_sync_d3_lag0','x2_vendor_sync_d3_lag0')]:
        q=g.copy();ok=q[of].notna()&q[vf].notna();q.loc[~ok,[of,vf]]=np.nan
        for scope,x in [('official',of),('vendor',vf)]:
            r,_=fit(q,h,x,'price');r.update(evidence=scope,scope='same_dates_vintage_check');comparisons.append(r)
    pd.DataFrame(comparisons).to_csv(O/'same_date_source_check.csv',index=False)
    assert all(hashlib.sha256((R/k).read_bytes()).hexdigest()==v for k,v in frozen['hashes'].items())
    (O/'run_summary.json').write_text(json.dumps(dict(frozen_utc=frozen['frozen_utc'],original_sources_unchanged=True,registered_hypotheses=2,fitted_main_layers=len(results),neighbors_per_hypothesis=2,trade_rule_candidates_run=0,new_account_backtests=0,definition_changed_after_results=False,results=results),ensure_ascii=False,indent=2),encoding='utf8')
    print('FINISHED_FIXED_HYPOTHESES',flush=True)
if __name__=='__main__':main()
