"""Read-only deterministic export of round-two PIT result evidence."""
from pathlib import Path
import argparse,base64,hashlib,io,json,zipfile
import pandas as pd
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument("--chunk",type=int,default=-1);ap.add_argument("--size",type=int,default=40000);a=ap.parse_args()
b=io.BytesIO();checks=[]
with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 def put(n,data):
  i=zipfile.ZipInfo(n,(1980,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,data,compresslevel=9)
 for p in sorted(HERE.iterdir()):
  if p.is_file() and p.suffix in (".py",".json",".sha256",".md"):
   put("pit/"+p.name,p.read_bytes())
 for p in sorted((HERE/"results").iterdir()):
  if p.is_file() and p.suffix in (".json",".csv"):
   put("pit/results/"+p.name,p.read_bytes())
 for p in sorted((HERE/"results"/"accounts").glob("*/ledger.parquet")):
  f=pd.read_parquet(p);rel=p.relative_to(HERE)
  put("pit/"+str(rel.with_suffix(".csv")).replace(chr(92),"/"),f.to_csv(index=True).encode("utf-8"))
  m=p.with_name("meta.json");put("pit/"+str(m.relative_to(HERE)).replace(chr(92),"/"),m.read_bytes())
  checks.append({"path":str(rel),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"rows":len(f)})
 put("pit/exported_ledger_manifest.json",json.dumps(checks,indent=2).encode())
raw=b.getvalue();e=base64.b64encode(raw).decode()
v={"bytes":len(raw),"base64_length":len(e),"chunks":(len(e)+a.size-1)//a.size,"sha256":hashlib.sha256(raw).hexdigest(),"ledgers":len(checks)} if a.chunk<0 else {"chunk":a.chunk,"data":e[a.chunk*a.size:(a.chunk+1)*a.size]}
print("PACKET:"+json.dumps(v,separators=(",",":")))

