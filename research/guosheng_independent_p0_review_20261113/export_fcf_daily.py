"""Read-only export of original archived daily FCF P&L and reported attribution."""
from pathlib import Path
import pandas as pd
import json,hashlib,base64,io,zipfile,argparse
p=argparse.ArgumentParser();p.add_argument("--chunk",type=int,default=-1);a=p.parse_args()
R=Path(__file__).resolve().parents[1]
B=R/"guosheng_six_regime_long_20261112"
P=B/"accounts/amount/2013-04-12/GS_monthly_9bp"
files={k:P/(k+".parquet") for k in ("ledger","pnl","weights")}
files["daily"]=B/"tables/daily_amount_GS_monthly.parquet"
data={k:pd.read_parquet(v) for k,v in files.items()}
l,pl,w,d=(data[k] for k in ("ledger","pnl","weights","daily"))
assert all(l.index.equals(t.index) for t in (pl,w,d))
f=pd.DataFrame({"nav":l.nav,"cost":l.cost,"pnl_fcf":pl.fcf,"pnl_total":pl.sum(axis=1),"fcf_start_weight":w.fcf.shift(1,fill_value=0.),"reported_log_fcf":d.log_fcf,"reported_log_fee":d.log_fee})
f.index.name="date"
m={"capital":1000000,"base_date":"2013-04-11","rows":len(f),"source":"original archive, not a new strategy run","files":{str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in files.values()}}
zbuf=io.BytesIO()
with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for n,b in [("fcf_daily_raw.csv",f.to_csv().encode()),("fcf_daily_raw_receipt.json",json.dumps(m,indent=2).encode())]:
  i=zipfile.ZipInfo(n,(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b,compresslevel=9)
b=zbuf.getvalue();e=base64.b64encode(b).decode();N=120000
v={"bytes":len(b),"chunks":(len(e)+N-1)//N,"sha256":hashlib.sha256(b).hexdigest()} if a.chunk<0 else {"chunk":a.chunk,"data":e[a.chunk*N:(a.chunk+1)*N]}
print("PACKET:"+json.dumps(v,separators=(",",":")))

