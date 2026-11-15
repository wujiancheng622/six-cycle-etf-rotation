from pathlib import Path
import sys, json, hashlib
from datetime import datetime, timezone
import pandas as pd
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'ah_premium_20261106'))
from data_access import OfficialData
OUT=HERE/'input'; OUT.mkdir(exist_ok=True)
client=OfficialData()
receipt=[]
for code in ('510880.SH','510030.SH'):
    for api in ('fund_daily','fund_adj','fund_div'):
        chunks=[('20120101','20161231'),('20170101','20211231'),('20220101','20261112')] if api!='fund_div' else [(None,None)]
        parts=[]
        for lo,hi in chunks:
            p=OUT/f'{api}_{code}_{lo or "all"}_{hi or "all"}.parquet'
            args={'ts_code':code}
            if lo: args.update(start_date=lo,end_date=hi)
            if p.exists(): f=pd.read_parquet(p)
            else:
                f=client.query(api,args)
                assert len(f)<2000,(api,len(f))
                f.to_parquet(p,index=False)
            parts.append(f)
            receipt.append({'file':p.name,'api':api,'params':args,'rows':len(f),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        full=pd.concat(parts,ignore_index=True)
        if api!='fund_div': assert not full.trade_date.duplicated().any()
        full.to_parquet(OUT/f'{api}_{code}.parquet',index=False)
        print(api,code,len(full),flush=True)
(OUT/'receipt.json').write_text(json.dumps({'fetched_at':datetime.now(timezone.utc).isoformat(),'files':receipt},indent=2),encoding='utf8')
