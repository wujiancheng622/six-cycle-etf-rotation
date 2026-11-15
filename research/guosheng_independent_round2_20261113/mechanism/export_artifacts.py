"""Read-only in-memory export for cloud audit handoff."""
from pathlib import Path
import argparse, base64, hashlib, io, json, zipfile
import pandas as pd
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--chunk",type=int,default=-1)
parser.add_argument("--size",type=int,default=40000)
args=parser.parse_args()
buf=io.BytesIO()
with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    def put(name,data):
        info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,data,compresslevel=9)
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in (".json",".csv",".py",".md"):
            put("mechanism/"+p.name,p.read_bytes())
    checks=[]
    for p in sorted((HERE/"accounts").glob("*/*/ledger.parquet")):
        rel=p.relative_to(HERE)
        f=pd.read_parquet(p)
        put("mechanism/"+str(rel.with_suffix(".csv")).replace("\\","/"),f.to_csv(index=True).encode("utf-8-sig"))
        meta=p.with_name("meta.json")
        put("mechanism/"+str(meta.relative_to(HERE)).replace("\\","/"),meta.read_bytes())
        checks.append({"original_file":str(p),"original_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"rows":len(f),"column_names":f.columns.tolist()})
    put("mechanism/exported_ledger_manifest.json",json.dumps(checks,ensure_ascii=False,indent=2).encode())
raw=buf.getvalue()
encoded=base64.b64encode(raw).decode()
if args.chunk<0:
    result={"bytes":len(raw),"base64_length":len(encoded),"chunks":(len(encoded)+args.size-1)//args.size,"sha256":hashlib.sha256(raw).hexdigest(),"ledgers":len(checks)}
else:
    result={"chunk":args.chunk,"data":encoded[args.chunk*args.size:(args.chunk+1)*args.size]}
print("PACKET:"+json.dumps(result,separators=(",",":")))

