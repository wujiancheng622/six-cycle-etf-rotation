"""Independent saved-ledger checks, factorial effects, matched-risk and attribution summary."""
from pathlib import Path
import itertools,json,math
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;O=HERE/'results'
C=pd.read_csv(O/'comparison.csv');Y=pd.read_csv(O/'yearly.csv')
PANEL={'GS':pd.read_parquet(O/'gs_returns.parquet'),'V12':pd.read_parquet(O/'v12_returns.parquet')}
CAP=500000.

def metrics(nav,base,bd):
    n=np.r_[base,nav.to_numpy()];r=n[1:]/n[:-1]-1;y=(pd.Timestamp(nav.index[-1])-pd.Timestamp(bd)).days/365.25
    return dict(cagr=(n[-1]/base)**(1/y)-1,vol=np.std(r,ddof=1)*252**.5,mdd=float((1-n/np.maximum.accumulate(n)).max()),total=n[-1]/base-1)

def run():
    errors=[];ledger_checks=[];data={}
    for p in (O/'accounts').iterdir():
        m=json.loads((p/'meta.json').read_text());L=pd.read_parquet(p/'ledger.parquet');T=pd.read_parquet(p/'trades.parquet');W=pd.read_parquet(p/'weights.parquet');A=pd.read_parquet(p/'linked_log_attribution.parquet')
        prev=L.nav.shift(1,fill_value=CAP);r=L.nav/prev-1
        np.testing.assert_allclose(A.sum(axis=1),np.log1p(r),atol=3e-15,rtol=0)
        rows=C.loc[C.id.eq(m['id'])]
        for z in rows.itertuples():
            s=L.loc[z.start:z.end].nav;i=L.index.get_loc(s.index[0]);base=CAP if i==0 else L.nav.iloc[i-1];bd=m['base_date'] if i==0 else L.index[i-1]
            q=metrics(s,base,bd);errors.extend(abs(q[k]-getattr(z,k)) for k in q)
        err=0.
        if m['mode']=='amount':
            keys=list(W.drop(columns='cash'));R=PANEL['V12' if m['label']=='V12' else 'GS'].loc[L.index,keys]
            cash=CAP;pos=np.zeros(len(keys));groups={k:g for k,g in T.groupby('date')}
            for day,rr in R.iterrows():
                pos*=1+rr.to_numpy();before=cash;spend=0.
                for t in groups.get(day,T.iloc[:0]).itertuples():
                    k=keys.index(t.asset);fee=0. if m['zero_fees'] else max(5.,t.notional*.00015)+t.notional*m['bp']/10000
                    assert abs(fee-t.cost)<1e-7
                    if t.side=='buy':pos[k]+=t.notional;cash-=t.notional+fee;spend+=t.notional+fee
                    else:pos[k]-=t.notional;cash+=t.notional-fee
                assert spend<=before+1e-6 and cash>=-1e-6 and pos.min()>=-1e-6
                err=max(err,abs(cash+pos.sum()-L.at[day,'nav']))
            assert err<1e-5,(m['id'],err)
        ledger_checks.append(dict(id=m['id'],mode=m['mode'],days=len(L),trades=len(T),cash_recursion_max_error=err if m['mode']=='amount' else None))
        data[m['id']]=(m,L,A,W)
    pd.DataFrame(ledger_checks).to_csv(O/'independent_validation.csv',index=False)
    # Symmetric factorial decomposition; includes all component orderings.
    factorial=[];contrasts=[]
    base=C.loc[C.label.str.fullmatch(r'T[01]W[01]E[01]')&~C.zero_fees]
    for key,g in base.groupby(['mode','account_start','availability','bp','window']):
        if len(g)!=8:continue
        table={tuple(int(c) for c in z.label[1::2]):z for z in g.itertuples()}
        meta=dict(zip(['mode','account_start','availability','bp','window'],key))
        for metric in ['cagr','log_growth','mdd','vol']:
            f=lambda bits:getattr(table[tuple(bits)],metric)
            contributions=np.zeros(3)
            for order in itertools.permutations(range(3)):
                bits=[0,0,0]
                for j in order:
                    before=f(bits);bits[j]=1;contributions[j]+=(f(bits)-before)/6
            np.testing.assert_allclose(contributions.sum(),f([1,1,1])-f([0,0,0]),rtol=0,atol=1e-12)
            for j,feature in enumerate(['stage_selection','ERC_weighting','intramonth_switch']):factorial.append(dict(**meta,metric=metric,feature=feature,effect=contributions[j],baseline=f([0,0,0]),full=f([1,1,1])))
        for a,b,name in [('T0W0E0','T1W0E0','timing_with_equal_weight'),('T0W1E0','T1W1E0','timing_with_ERC_monthly'),('T0W1E1','T1W1E1','timing_with_ERC_switch'),('T1W0E0','T1W1E0','ERC_with_timing_monthly'),('T1W0E1','T1W1E1','ERC_with_timing_switch'),('T1W1E0','T1W1E1','switch_with_timing_ERC')]:
            x=g.loc[g.label.eq(a)].iloc[0];z=g.loc[g.label.eq(b)].iloc[0]
            contrasts.append(dict(**meta,contrast=name,cagr_diff=z.cagr-x.cagr,vol_diff=z.vol-x.vol,mdd_diff=z.mdd-x.mdd))
    pd.DataFrame(factorial).to_csv(O/'factorial_shapley.csv',index=False)
    pd.DataFrame(contrasts).to_csv(O/'factorial_contrasts.csv',index=False)
    # Ex-post shrink-to-V12-risk diagnostic: no leverage, no new claimed strategy.
    risk=[];cost=[];mean=[]
    for row in C.loc[(C.label.isin(['T1W1E0','T1W1E1','T1W0E0','T1W0E1','T0W0E0','T0W1E0']))&~C.zero_fees].itertuples():
        v=C.loc[C.label.eq('V12')&C['mode'].eq(row.mode)&C.account_start.eq(row.account_start)&C.bp.eq(row.bp)&C.window.eq(row.window)&~C.zero_fees]
        if len(v)!=1:continue
        v=v.iloc[0];m,L,*_=data[row.id];sub=L.nav.loc[row.start:row.end];i=L.index.get_loc(sub.index[0]);initial=CAP if i==0 else L.nav.iloc[i-1]
        rr=sub/sub.shift(1,fill_value=initial)-1;scale=min(1.,v.vol/row.vol)
        path=(1+rr*scale).cumprod()*initial;q=metrics(path,initial,row.base_date)
        risk.append(dict(id=row.id,window=row.window,scale=scale,**q,v12_cagr=v.cagr,v12_vol=v.vol,v12_mdd=v.mdd))
    pd.DataFrame(risk).to_csv(O/'expost_volatility_match.csv',index=False)
    paired=[]
    for row in C.loc[C.label.isin(['T1W1E0','T1W1E1'])&~C.zero_fees].itertuples():
        g=C.loc[C['mode'].eq(row.mode)&C.account_start.eq(row.account_start)&C.availability.eq(row.availability)&C.bp.eq(row.bp)&C.window.eq(row.window)&~C.zero_fees]
        for comparator in ['T0W0E0','T0W1E0','six_baskets_no_timing',f'expost_mean_exposure_E{int(row.E)}']:
            b=g.loc[g.label.eq(comparator)]
            if len(b)!=1:continue
            b=b.iloc[0];target=min(row.vol,b.vol);qs=[]
            for rr in [row,b]:
                m,L,*_=data[rr.id];z=L.nav.loc[rr.start:rr.end];i=L.index.get_loc(z.index[0]);initial=CAP if i==0 else L.nav.iloc[i-1];ret=z/z.shift(1,fill_value=initial)-1
                q=metrics((1+ret*(target/rr.vol)).cumprod()*initial,initial,rr.base_date);qs.append(q)
            paired.append(dict(id=row.id,window=row.window,comparator=comparator,common_vol=target,gs_scale=target/row.vol,comparator_scale=target/b.vol,gs_cagr=qs[0]['cagr'],comparator_cagr=qs[1]['cagr'],gs_mdd=qs[0]['mdd'],comparator_mdd=qs[1]['mdd']))
    pd.DataFrame(paired).to_csv(O/'expost_pairwise_risk_match.csv',index=False)
    # Net-log accounting by broad exposure; these labels do not imply identical securities.
    groups={'growth':'domestic_equity','fcf':'domestic_equity','dividend':'domestic_equity','csi300':'domestic_equity','csi1000':'domestic_equity','divlow':'domestic_equity','gold':'gold','metals':'commodity','energy':'commodity','soy':'commodity','bond':'treasury','bond10':'treasury','nasdaq':'US_equity','sp500':'US_equity','fee':'fee'}
    attr=pd.read_csv(O/'asset_attribution.csv');attr['exposure_group']=attr.asset.map(groups)
    assert attr.exposure_group.notna().all()
    attr.groupby(['id','window','exposure_group'],as_index=False)[['log_contribution','share_net_log','money_pnl']].sum().to_csv(O/'exposure_group_attribution.csv',index=False)
    drawdowns=[]
    for row in C.loc[C.label.isin(['T1W1E0','T1W1E1','V12'])&~C.zero_fees].itertuples():
        m,L,A,W=data[row.id];z=A.loc[(A.index>row.peak)&(A.index<=row.trough)].sum()
        for asset,val in z.items():drawdowns.append(dict(id=row.id,window=row.window,peak=row.peak,trough=row.trough,asset=asset,drawdown_log_contribution=val))
    pd.DataFrame(drawdowns).to_csv(O/'drawdown_attribution.csv',index=False)
    for row in C.loc[~C.zero_fees].itertuples():
        gross=C.loc[C.id.eq(row.id+'_gross')&C.window.eq(row.window)]
        if len(gross)==1:cost.append(dict(id=row.id,window=row.window,net_cagr=row.cagr,gross_cagr=gross.iloc[0].cagr,cagr_drag=gross.iloc[0].cagr-row.cagr,direct_fee_log=float(data[row.id][2].loc[row.start:row.end,'fee'].sum())))
    pd.DataFrame(cost).to_csv(O/'transaction_cost_effect.csv',index=False)
    summary=dict(accounts=len(data),amount_accounts=sum(x['mode']=='amount' for x in ledger_checks),actual_accounts=sum(x['mode']=='actual' for x in ledger_checks),account_days=sum(x['days'] for x in ledger_checks),orders=sum(x['trades'] for x in ledger_checks),metric_checks=len(errors),max_metric_error=max(errors),amount_nav_max_error=max(x['cash_recursion_max_error'] or 0 for x in ledger_checks),actual_scope='Saved daily P&L and log-link identity only; no new independent intraday quote rebuild')
    (O/'independent_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    view=C.loc[C.account_start.eq('2014-01-02')&C.bp.eq(9)&C.window.eq('paper')&~C.zero_fees]
    print('MAIN\n'+view[['availability','label','cagr','vol','mdd','mean_cash']].to_string(index=False),flush=True)
    print('VALIDATION',json.dumps(summary),flush=True)

if __name__=='__main__':run()
