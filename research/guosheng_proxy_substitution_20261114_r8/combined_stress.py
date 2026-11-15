"""Predefined adverse drift plus substitute volume-cap joint scenarios."""
import json
import numpy as np
import pandas as pd
import run_tests as t
from capacity_replay import economic_limited

def main():
    F,t.FEX=t.make_fallbacks()
    asset=pd.read_csv(t.OUT/'asset_overlap.csv')
    G=t.H.copy();V=t.V.copy()
    for r in asset.itertuples():
        if r.strategy=='GS':
            mask=G.index<t.COVER['actual_return_from'][r.asset]
            G.loc[mask,r.asset]=np.expm1(np.log1p(G.loc[mask,r.asset])-abs(r.annual_log_drift)/252)
        else:
            mask=V.index<'2020-01-20'
            V.loc[mask,r.asset]=np.expm1(np.log1p(V.loc[mask,r.asset])+abs(r.annual_log_drift)/252)
    protocol=dict(scope='Adversarial sensitivity, not estimated true returns',GS='Subtract absolute observed actual-minus-proxy mean log drift on early proxy rows',V12='Add absolute observed mean log drift on early proxy rows',substitution='Two separately reported old ETF alternatives; use actual FCF ETF only after own 63 observations',capacity='1% prior 20-day turnover for substitute legs; zero on suspension; original two attempts',capital=500000,start='2014-01-02',end=t.END)
    (t.OUT/'combined_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    R=pd.concat([G,F],axis=1);R.attrs.update(G.attrs);keys=list(R)
    caps=pd.DataFrame(np.inf,index=R.index,columns=keys)
    for key,code in [('old_dividend','510880.SH'),('old_value','510030.SH')]:
        q=pd.read_parquet(t.INP/f'fund_daily_{code}.parquet').set_index('trade_date').sort_index()
        q.index=pd.to_datetime(q.index).strftime('%Y-%m-%d');q=q.reindex(R.index)
        caps[key]=(.01*q.amount.fillna(0).rolling(20).mean().shift()*1000).where(q.vol.gt(0),0).fillna(0)
    for bp in [9,20]:
        t.save(t.v12(V,'2014-01-02',bp),'V12_favorable_drift','combined',bp,'2014-01-02')
        for label,key in [('dividend_adverse_drift','old_dividend'),('value_adverse_drift','old_value')]:
            policy=t.SubstitutionPolicy(R,t.S,keys,key,t.COVER['actual_return_from']['fcf'],False)
            a,reject=economic_limited(R,policy,'2014-01-02',bp/10000,caps)
            t.save(a,label,'combined',bp,'2014-01-02')
            reject.to_csv(t.OUT/f'combined_rejected_{bp}_{label}.csv',index=False)
    pd.DataFrame(t.rows).to_csv(t.OUT/'combined_comparison.csv',index=False)
    pd.DataFrame(t.annual).to_csv(t.OUT/'combined_annual.csv',index=False)

if __name__=='__main__':main()
