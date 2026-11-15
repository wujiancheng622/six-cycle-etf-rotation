"""Read-only input coverage feasibility, no strategy execution."""
from pathlib import Path
import sys,importlib.util,json,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location("pit_prior",ROOT/"guosheng_independent_round2_20261113/pit/run_pit.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
vr=p.pd.read_parquet(p.OUT/"returns_v12_earlier_actual.parquet")
out={"panels":{},"starts":{},"v12_available":p.pd.read_csv(p.OUT/"availability_v12.csv")[["key","first_actual_return"]].to_dict(orient="records")}
for name,r in [("GS",p.HIST),("V12_original",p.VOLD),("V12_earlier_actual",vr)]:
 out["panels"][name]={"first":r.index[0],"last":r.index[-1],"rows":len(r),"nan_count":int(r.isna().sum().sum()),"columns":list(r.columns),"attrs":list(r.attrs)}
for start in ["2013-04-12","2014-07-16","2025-06-04","2025-07-01"]:
 j=p.HIST.index.get_loc(start);known=p.HIST.index[j-1]
 out["starts"][start]={"known":known,"GS_complete_risk_rows":len(p.train(p.HIST,known,p.KEYS)),"V12_original_risk_rows":len(p.train(p.VOLD,known,p.VKEYS)),"macro_stage":int(p.MACRO.loc[p.pd.Timestamp(known),"stage"])}
print(json.dumps(out,ensure_ascii=False,indent=2))

