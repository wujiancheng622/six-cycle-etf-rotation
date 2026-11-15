"""Independent saved-NAV and cash/holding recursion; inspect substitute capacity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=HERE/'results';INP=HERE/'input'
H=pd.read_parquet(ROOT/'guosheng_six_regime_long_20261112/data/returns_gs.parquet')
V=pd.read_parquet(ROOT/'guosheng_six_regime_long_20261112/data/returns_v12.parquet')
F=pd.read_parquet(INP/'fallback_returns.parquet')
C=pd.read_csv(OUT/'comparison.csv')
if (OUT/'capacity_comparison.csv').exists():C=pd.concat([C,pd.read_csv(OUT/'capacity_comparison.csv')],ignore_index=True)
if (OUT/'combined_comparison.csv').exists():C=pd.concat([C,pd.read_csv(OUT/'combined_comparison.csv')],ignore_index=True)
capacity=[];verified=[];merrors=[]
for folder in (OUT/'accounts').iterdir():
    meta=json.loads((folder/'meta.json').read_text());capital=meta['capital']
    L=pd.read_parquet(folder/'ledger.parquet');T=pd.read_parquet(folder/'trades.parquet')
    W=pd.read_parquet(folder/'weights.parquet');keys=list(W.drop(columns='cash').columns)
    code=folder.name;start=L.index[0]
    if code.startswith('combined'):
        cov=json.loads((ROOT/'guosheng_six_regime_long_20261112/coverage.json').read_text())
        drift=pd.read_csv(OUT/'asset_overlap.csv')
        is_v='V12_favorable' in code;R=V.copy() if is_v else H.copy()
        for x in drift.loc[drift.strategy.eq('V12' if is_v else 'GS')].itertuples():
            mask=R.index<('2020-01-20' if is_v else cov['actual_return_from'][x.asset])
            R.loc[mask,x.asset]=np.expm1(np.log1p(R.loc[mask,x.asset])+(1 if is_v else -1)*abs(x.annual_log_drift)/252)
        if not is_v:R=pd.concat([R,F],axis=1).loc[:,keys]
    elif code.startswith('substitute') or code.startswith('capacity'):
        R=V if code.endswith('_V12') else pd.concat([H,F],axis=1).loc[:,keys]
    else:
        strategy='gs' if code.startswith('GS_') else 'v12'
        version='proxy' if code.endswith('_proxy_proxy') or code.endswith('_actual_proxy') else 'actual'
        R=pd.read_parquet(INP/f'{strategy}_{version}.parquet')
    pos=np.zeros(len(keys));cash=float(capital);maxerr=0.;maxfee=0.
    orders={d:g for d,g in T.groupby('date')}
    for day,prices in R.loc[L.index,keys].iterrows():
        pos*=1+prices.to_numpy();before=cash
        dayorders=orders.get(day,T.iloc[:0]);buyspend=0.
        bp=float(code.split('bp_')[0].split('_')[-1])
        for order in dayorders.itertuples():
            j=keys.index(order.asset);fee=max(5.,order.notional*.00015)+order.notional*bp/10000
            maxfee=max(maxfee,abs(fee-order.cost))
            if order.side=='buy':pos[j]+=order.notional;cash-=order.notional+fee;buyspend+=order.notional+fee
            else:pos[j]-=order.notional;cash+=order.notional-fee
        assert buyspend<=before+1e-6 and cash>=-1e-6 and pos.min()>=-1e-6
        maxerr=max(maxerr,abs(cash+pos.sum()-L.at[day,'nav']))
    assert maxerr<1e-5 and maxfee<1e-6,(code,maxerr,maxfee)
    verified.append(dict(account=code,days=len(L),orders=len(T),max_nav_error=maxerr,max_fee_error=maxfee))
    # Capacity is a diagnostic, not included in the main fractional replay.
    for key,fund in [('old_dividend','510880.SH'),('old_value','510030.SH')]:
        z=T.loc[T.asset.eq(key)]
        if z.empty:continue
        q=pd.read_parquet(INP/f'fund_daily_{fund}.parquet').set_index('trade_date').sort_index()
        q.index=pd.to_datetime(q.index).strftime('%Y-%m-%d')
        amounts=q.amount.reindex(H.index).fillna(0)*1000
        adv=amounts.rolling(20,min_periods=20).mean().shift(1)
        p=z.notional.to_numpy()/adv.reindex(z.date).to_numpy()
        if code.startswith('capacity') or code.startswith('combined'):
            assert np.max(p)<=.01+1e-12 and q.vol.reindex(z.date).fillna(0).gt(0).all(),code
        capacity.append(dict(account=code,asset=key,trades=len(z),max_adv_fraction=float(np.max(p)),p95_adv_fraction=float(np.quantile(p,.95)),over1pct=int((p>.01).sum()),no_volume_orders=int(q.vol.reindex(z.date).fillna(0).le(0).sum())))
    # Recompute each stored window without importing the replay's metric helper.
    for x in C.loc[(C.group+'_'+C.account_start+'_'+C.bp.astype(str)+'bp_'+C.label).eq(code)].itertuples():
        z=L.loc[x.start:x.end];idx=L.index.get_loc(z.index[0]);p0=capital if idx==0 else L.nav.iloc[idx-1]
        dt=meta['base_date'] if idx==0 else L.index[idx-1]
        path=np.concatenate(([p0],z.nav.to_numpy()));ret=np.diff(path)/path[:-1]
        span=(pd.Timestamp(z.index[-1])-pd.Timestamp(dt)).days/365.25
        vals=[(path[-1]/p0)**(1/span)-1,np.std(ret,ddof=1)*np.sqrt(252),np.max(1-path/np.maximum.accumulate(path)),path[-1]/p0-1]
        merrors.extend([abs(a-b) for a,b in zip(vals,[x.cagr,x.vol,x.mdd,x.total])])
pd.DataFrame(verified).to_csv(OUT/'independent_cash_validation.csv',index=False)
pd.DataFrame(capacity).to_csv(OUT/'fallback_capacity.csv',index=False)
summary=dict(accounts=len(verified),account_days=sum(x['days'] for x in verified),orders=sum(x['orders'] for x in verified),max_nav_error=max(x['max_nav_error'] for x in verified),metric_checks=len(merrors),max_metric_error=max(merrors))
(OUT/'independent_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2));print(pd.DataFrame(capacity).to_string(index=False))
