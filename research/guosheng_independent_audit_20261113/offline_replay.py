"""Read archived local snapshots and replay audited recent accounts; no network."""
import sys,socket,json,gzip,base64,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
def offline(*a,**k): raise RuntimeError("Independent audit offline: network disabled")
socket.socket.connect=offline
socket.create_connection=offline
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/"guosheng_parameter_robustness_20261113"))
from study_inputs import Inputs,parent
data=Inputs.load()
out={"scope":"archived local snapshots, same engine replay; independent price audit exported separately","network_disabled":True,"quotes":{},"replays":{}}
for name,bundle in [("GS",data.gs_quotes),("V12",data.v12_quotes)]:
 q,adv,up,down=bundle
 out["quotes"][name]={"panels":{field:json.loads(q.panels[field].loc["2025-11-05":"2026-11-12"].to_json(orient="split")) for field in ("open","close")},"dividends":q.dividends,"splits":q.splits}
for name in ("GS_monthly","V12","GS_stage_switch"):
 a=parent.run(data,"actual","2025-11-06",name)
 stored=parent.load_account("actual","2025-11-06",name) if name!="GS_stage_switch" else None
 result={"metrics":parent.stats(a)}
 if stored is not None:
  result["max_nav_difference_vs_archive"]=float((a.ledger.nav-stored.ledger.nav).abs().max())
 result["first_signals"]=a.signals.head(1).to_dict(orient="records")
 out["replays"][name]=result
print("AUDIT_PAYLOAD:"+base64.b64encode(gzip.compress(json.dumps(out,default=str,ensure_ascii=True).encode())).decode())

