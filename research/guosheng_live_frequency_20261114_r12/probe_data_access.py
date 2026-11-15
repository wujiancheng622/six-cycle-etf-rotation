"""Read-only entitlement/freshness probe; never change the frozen backtest inputs."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'ah_premium_20261106'))
from data_access import OfficialData
OUT=HERE/'data_probe';OUT.mkdir(exist_ok=True)
client=OfficialData();checks=[]
queries=[('cn_pmi',{'start_m':'202608','end_m':'202610'},'month,pmi010000,pmi020100'),
         ('shibor_lpr',{'start_date':'20260901','end_date':'20261113'},'date,1y,5y'),
         ('shibor',{'start_date':'20261109','end_date':'20261113'},'date,on,1w')]
for code in ['159915.SZ','518880.SH','159980.SZ','159201.SZ','511090.SH','515100.SH']:
    queries.append(('fund_daily',{'ts_code':code,'start_date':'20261109','end_date':'20261113'},'ts_code,trade_date,open,high,low,close,pre_close,vol,amount'))
for i,(api,params,fields) in enumerate(queries):
    before=time.monotonic();row=dict(api=api,params=params,fields=fields,received_utc=datetime.now(timezone.utc).isoformat())
    try:
        f=client.query(api,params,fields)
        p=OUT/f'{i:02d}_{api}.csv';f.to_csv(p,index=False)
        row.update(status='success',rows=len(f),columns=list(f),nulls=f.isna().sum().to_dict(),samples=f.to_dict('records'),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if api=='fund_daily' and len(f):row['ohlc_valid']=bool((f.high>=f[['open','close','low']].max(axis=1)).all() and (f.low<=f[['open','close','high']].min(axis=1)).all() and (f.close>0).all())
    except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e)[:300])
    row['elapsed_seconds']=time.monotonic()-before;checks.append(row)
    print(json.dumps(row,ensure_ascii=False,default=str),flush=True)
(OUT/'access_receipts.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,default=str),encoding='utf8')
