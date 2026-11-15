"""Read-only deterministic export of frozen replays and independent-review inputs."""
from pathlib import Path
import io,json,zipfile,hashlib,base64,argparse,sys
sys.dont_write_bytecode=True
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/"results";PRIOR=ROOT/"guosheng_independent_round2_20261113/pit/results"
NAMES=["GS_original","GS_THREE_INDEX_eligible_equal","GS_ALL_ETF_eligible_equal","V12_ETF_available"]
def add(z,name,data):
 info=zipfile.ZipInfo(name,date_time=(2026,11,13,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
def make(kind):
 b=io.BytesIO()
 with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  files=["protocol.json","freeze.json","run_matrix.py","inspect_coverage.py","export_bundle.py","checks.json","failures.jsonl","macro_lag_sidecar.csv","results/summary.csv","results/daily_ledger.csv"]
  if kind=="review":files=["protocol.json","freeze.json","macro_lag_sidecar.csv"]
  for name in files:
   p=HERE/name
   if p.is_file():add(z,name,p.read_bytes())
  if kind in ["full","results"]:
   for p in sorted((OUT/"accounts").glob("*/signals.csv"))+sorted((OUT/"accounts").glob("*/meta.json")):
    add(z,str(p.relative_to(HERE)).replace("\\","/"),p.read_bytes())
  for name in (NAMES if kind!="results" else []):
   folder=OUT/"accounts"/(name+"__2013-04-12__500000__9")
   if not (folder/"meta.json").is_file():
    if kind=="review":raise RuntimeError("Review baseline not finished: "+name)
    continue
   for p in sorted(folder.glob("*.parquet")):
    f=pd.read_parquet(p)
    add(z,str(p.with_suffix(".csv").relative_to(HERE)).replace("\\","/"),f.to_csv(index=True).encode())
   if kind=="review":add(z,str((folder/"meta.json").relative_to(HERE)).replace("\\","/"),(folder/"meta.json").read_bytes())
  sources={"returns_gs":ROOT/"guosheng_six_regime_long_20261112/data/returns_gs.parquet",
           "returns_v12_actual":PRIOR/"returns_v12_earlier_actual.parquet"}
  if kind=="results":sources={}
  manifest={}
  for name,p in sources.items():
   data=p.read_bytes();f=pd.read_parquet(p);add(z,"inputs/"+name+".csv",f.to_csv(index=True).encode())
   add(z,"inputs/"+name+"_attrs.json",json.dumps(f.attrs,ensure_ascii=False,indent=2,default=str).encode())
   manifest[name]={"path":str(p),"sha256":hashlib.sha256(data).hexdigest(),"rows":len(f),"first":f.index[0],"last":f.index[-1],"columns":list(f.columns)}
  if manifest:add(z,"inputs/source_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2).encode())
 return b.getvalue()
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["info","chunk"]);p.add_argument("--kind",choices=["full","review","results"],default="full");p.add_argument("--offset",type=int,default=0);p.add_argument("--size",type=int,default=48000);a=p.parse_args()
 b=make(a.kind)
 print(json.dumps({"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}) if a.mode=="info" else base64.b64encode(b[a.offset:a.offset+a.size]).decode())

