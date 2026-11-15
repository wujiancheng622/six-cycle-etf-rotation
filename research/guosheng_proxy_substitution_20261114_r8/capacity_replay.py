"""Diagnose newly introduced old ETF capacity; never force fills on suspended days."""
import numpy as np
import pandas as pd
import json
import run_tests as t

def economic_limited(R,policy,start,slip,caps):
    dates=R.index[R.index>=start];keys=list(R.columns)
    pos=np.zeros(len(keys));cash=t.CAPITAL
    ledger=[];weights=[];pnlrows=[];trades=[];reject=[]
    for i,day in enumerate(dates):
        j=R.index.get_loc(day);known=R.index[j-1];nav=cash+pos.sum()
        sw,bw=policy.targets(day,known,i,nav)
        diff=lambda w:np.zeros(len(keys)) if w is None else w*nav-pos
        r=R.iloc[j].to_numpy();pnl=pos*r
        sr=-np.minimum(diff(sw),0)*(1+r)
        br=np.where(sr==0,np.maximum(diff(bw),0)*(1+r),0.)
        pos+=pnl
        cap=caps.loc[day].to_numpy()
        sales=np.minimum(np.minimum(sr,pos),cap)
        buys_req=np.minimum(br,cap)
        for ix in np.flatnonzero((sr>cap+1e-9)|(br>cap+1e-9)):
            reject.append(dict(date=day,asset=keys[ix],sell_requested=sr[ix],buy_requested=br[ix],capacity=cap[ix]))
        sc=t.engine.amount_fee(sales,slip)
        buys=t.engine.affordable_amount(buys_req,min(cash,cash+float((sales-sc).sum())),slip)
        bc=t.engine.amount_fee(buys,slip);cost=float(sc.sum()+bc.sum())
        cash+=float(sales.sum()-buys.sum()-cost);pos+=buys-sales;nav=cash+pos.sum()
        count=0
        for side,values,fees in [('sell',sales,sc),('buy',buys,bc)]:
            for k in np.flatnonzero(values>1e-9):
                trades.append(dict(date=day,known=known,asset=keys[k],side=side,notional=values[k],cost=fees[k]));count+=1
        ledger.append(dict(date=day,nav=nav,cash=cash,receivable=0.,cost=cost,notional=float(sales.sum()+buys.sum()),trade_count=count))
        weights.append(np.r_[pos/nav,cash/nav]);pnlrows.append(pnl)
    with t.apply_policy(lambda *args:policy,keys):a=t.engine.pack(ledger,weights,pnlrows,trades,policy,R.index[R.index.get_loc(start)-1])
    return a,pd.DataFrame(reject)

def main():
    F,t.FEX=t.make_fallbacks();R=pd.concat([t.H,F],axis=1);R.attrs.update(t.H.attrs);keys=list(R.columns)
    caps=pd.DataFrame(np.inf,index=R.index,columns=keys)
    for key,code in [('old_dividend','510880.SH'),('old_value','510030.SH')]:
        q=pd.read_parquet(t.INP/f'fund_daily_{code}.parquet').set_index('trade_date').sort_index()
        q.index=pd.to_datetime(q.index).strftime('%Y-%m-%d');q=q.reindex(R.index)
        caps[key]=(.01*q.amount.fillna(0).rolling(20).mean().shift(1)*1000).where(q.vol.gt(0),0).fillna(0)
    cut=t.COVER['actual_return_from']['fcf']
    (t.OUT/'capacity_protocol.json').write_text(json.dumps(dict(cap=.01,turnover_unit='Tushare amount * 1000 RMB',signal='unchanged monthly',execution='same original two-day attempt; no intramonth reoptimization; unfilled remains until next scheduled rebalance',scope='only substitute ETF legs; no claim to authenticate early other proxies'),indent=2),encoding='utf8')
    rejected=[]
    for start in ['2014-01-02','2020-01-02']:
        for bp in [9,20]:
            for label,key in [('dividend_to_FCF','old_dividend'),('value_to_FCF','old_value')]:
                make=lambda:t.SubstitutionPolicy(R,t.S,keys,key,cut,False)
                if start=='2014-01-02' and bp==9:
                    a,_=economic_limited(R,make(),start,bp/10000,pd.DataFrame(np.inf,index=R.index,columns=keys))
                    old=pd.read_parquet(t.OUT/f'accounts/substitute_{start}_{bp}bp_{label}/ledger.parquet')
                    pd.testing.assert_frame_equal(a.ledger,old)
                a,z=economic_limited(R,make(),start,bp/10000,caps)
                t.save(a,label,'capacity',bp,start)
                rejected.append(z.assign(label=label,start=start,bp=bp))
    pd.DataFrame(t.rows).to_csv(t.OUT/'capacity_comparison.csv',index=False)
    pd.DataFrame(t.annual).to_csv(t.OUT/'capacity_annual.csv',index=False)
    pd.concat(rejected,ignore_index=True).to_csv(t.OUT/'capacity_rejections.csv',index=False)

if __name__=='__main__':main()
