"""Recover missing 13.90% daily path from archived panels without network."""
import sys,socket,json,gzip,base64
from pathlib import Path
sys.dont_write_bytecode=True
def offline(*a,**k): raise RuntimeError("Offline audit: network disabled")
socket.socket.connect=offline
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/"guosheng_parameter_robustness_20261113"))
from study_inputs import DATA,PARENT,parent,engine,apply_policy,KEYS,PROVIDER
sys.path.insert(0,str(PARENT))
from verify_long import PriorToCash
import pandas as pd
hist=pd.read_parquet(DATA/"returns_gs.parquet")
macro=pd.read_parquet(DATA/"macro.parquet")
cutoff=json.loads((PARENT/"coverage.json").read_text(encoding="utf-8"))["actual_return_from"]
with apply_policy(lambda n,r,_:PriorToCash("GS_monthly",r,macro,cutoff),KEYS):
 a=engine.economic("all_prelisting_returns_cash",hist,None,"2013-04-12",.0009)
out={"scope":"same-engine rerun from archived returns and macro panels, not independent raw-vintage reconstruction","metrics":parent.stats(a),"base_date":a.base_date,"nav":a.ledger.nav.to_dict(),"input_sha256":{p.name:PROVIDER.sha(p) for p in (DATA/"returns_gs.parquet",DATA/"macro.parquet",PARENT/"coverage.json")}}
print("AUDIT_PAYLOAD:"+base64.b64encode(gzip.compress(json.dumps(out,default=str).encode())).decode())

