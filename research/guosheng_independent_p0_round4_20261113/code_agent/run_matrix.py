"""Round 4: fixed 50w same-calendar P0-channel amount replays. No network or source edits."""
from pathlib import Path
from datetime import datetime,timezone
import sys,importlib.util,json,hashlib,copy,traceback,io,zipfile,base64,argparse
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PROTOCOL=HERE/"protocol.json";OUT=HERE/"results"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location("prior_round2_pit",ROOT/"guosheng_independent_round2_20261113/pit/run_pit.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
np,pd=p.np,p.pd
vrpath=p.OUT/"returns_v12_earlier_actual.parquet"
vr=pd.read_parquet(vrpath)
vtab=pd.read_csv(p.OUT/"availability_v12.csv")
vcuts=dict(zip(vtab.key,vtab.first_actual_return))
gcuts=p.COVER["actual_return_from"]
release={"fcf":"2024-08-15","dividend":"2017-05-26","bond":"2021-07-09"}
def after(d):return p.HIST.index[p.HIST.index.searchsorted(d,side="right")]
groups={"FCF":{"fcf":after(release["fcf"])},
        "THREE_INDEX":{k:after(d) for k,d in release.items()},"ALL_ETF":gcuts}
rawcuts={"FCF":{"fcf":release["fcf"]},"THREE_INDEX":release,"ALL_ETF":gcuts}
macro0=p.MACRO.reindex(pd.to_datetime(p.HIST.index))
assert macro0.index.is_unique and macro0.index.is_monotonic_increasing
macro1=macro0.shift(1).iloc[1:].copy()
pd.testing.assert_frame_equal(macro1.reset_index(drop=True),macro0.iloc[:-1].reset_index(drop=True))
CACHE={}
class MemoPolicy:
    def __init__(self,cache,build):
        self.cache=cache;self.inner=None;self.build=build;self.receipts=[]
    def targets(self,day,known,i,nav):
        key=(day,known,i)
        if key not in self.cache:
            if self.inner is None:self.inner=self.build()
            n=len(self.inner.receipts)
            target=self.inner.targets(day,known,i,nav)
            self.cache[key]=(copy.deepcopy(target),copy.deepcopy(self.inner.receipts[n:]))
        target,new=self.cache[key]
        self.receipts.extend(copy.deepcopy(new))
        return copy.deepcopy(target)
def case_list():
    out=[("GS_original",p.HIST,p.KEYS,lambda r:p.Rotation("GS_monthly",r,p.MACRO),"GS"),
         ("V12_original",p.VOLD,p.VKEYS,lambda r:p.orig_policy("V12",r,None),"V12"),
         ("V12_earlier_actual",vr,p.VKEYS,lambda r:p.AlignedV12(r),"V12")]
    for group,cuts in groups.items():
        old=rawcuts[group]
        out.append(("GS_"+group+"_legacy_hold_only",p.HIST,p.KEYS,
                    lambda r,old=old:p.PriorToCash("GS_monthly",r,p.MACRO,old),"GS"))
        for risk in ["proxy","eligible"]:
            for cash in ["reference","equal"]:
                out.append(("GS_"+group+"_"+risk+"_"+cash,p.HIST,p.KEYS,
                    lambda r,cuts=cuts,risk=risk,cash=cash:p.BoundaryGS(r,p.MACRO,cuts,risk,cash),"GS"))
    out.append(("V12_ETF_available",vr,p.VKEYS,lambda r:p.BoundaryV12(r,vcuts),"V12"))
    out.append(("GS_ALL_ETF_eligible_equal_macro_lag1",p.HIST,p.KEYS,
                lambda r:p.BoundaryGS(r,macro1,gcuts),"GS"))
    assert len(out)==20
    return out
def source_manifest():
    files={PROTOCOL,Path(__file__).resolve(),p.DATA/"returns_gs.parquet",p.DATA/"returns_v12.parquet",
           p.DATA/"macro.parquet",p.PARENT/"coverage.json",vrpath,p.OUT/"availability_v12.csv"}
    for mod in list(sys.modules.values()):
        fp=getattr(mod,"__file__",None)
        if fp:
            path=Path(fp).resolve()
            if path.is_relative_to(ROOT) and path.is_file():files.add(path)
    return {str(x.relative_to(ROOT)):sha(x) for x in sorted(files)}
def freeze():
    f=HERE/"freeze.json"
    if f.exists():raise RuntimeError("Freeze exists; refuse overwrite")
    content={"utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":sha(PROTOCOL),
             "script_sha256":sha(__file__),"input_and_code_sha256":source_manifest(),
             "new_account_results_seen_before_freeze":False,
             "coverage":{"GS_start":p.HIST.index[0],"V12_start":p.VOLD.index[0],
                         "start":"2013-04-12","GS_risk_rows":len(p.train(p.HIST,"2013-04-11",p.KEYS)),
                         "V12_risk_rows":len(p.train(p.VOLD,"2013-04-11",p.VKEYS))}}
    f.write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"frozen":str(f),"utc":content["utc"],"protocol_sha256":content["protocol_sha256"],"script_sha256":content["script_sha256"]}),flush=True)
def lag_sidecar():
    z=pd.DataFrame(index=macro1.index)
    z["known_date"]=z.index.strftime("%Y-%m-%d")
    z["baseline_information_date"]=z.known_date
    z["lagged_information_date"]=p.HIST.index[:-1]
    for k in macro0.columns:
        z["baseline_"+k]=macro0.loc[z.index,k]
        z["lagged_"+k]=macro1[k]
    def basket(v):return "/".join(p.ASSETS_BY_STAGE[int(v)]) if pd.notna(v) else ""
    z["baseline_basket"]=z.baseline_stage.map(basket);z["lagged_basket"]=z.lagged_stage.map(basket)
    z.to_csv(HERE/"macro_lag_sidecar.csv",index=False)
    macro1.to_parquet(HERE/"macro_lag_one_session.parquet")
def run_one(case,start,bps):
    name,r,keys,build,engine_name=case
    ident=name+"__"+start+"__500000__"+str(bps)
    folder=OUT/"accounts"/ident
    if (folder/"meta.json").exists():return
    folder.mkdir(parents=True,exist_ok=True)
    cache=CACHE.setdefault((name,start),{})
    factory=lambda n,x,_:MemoPolicy(cache,lambda:build(x))
    capital=p.engine.CAPITAL;p.engine.CAPITAL=500000
    try:
        with p.apply_policy(factory,keys):
            a=p.economic_masked(engine_name,r,None,start,bps/10000)
    finally:p.engine.CAPITAL=capital
    assert a.ledger.index[0]==start and a.ledger.index[-1]=="2026-11-12"
    assert (a.signals.known<a.signals.date).all()
    expected=p.HIST.index[p.HIST.index>=start]
    assert a.ledger.index.equals(expected)
    metric=p.measures(a.ledger,a.base_date,500000)
    row={"account_id":ident,"case":name,"capital":500000,"slip_bps":bps,
         **metric,"mean_cash":float(a.weights.cash.mean()),"model_cost_total":float(a.ledger.cost.sum())}
    for key in ["ledger","weights","pnl","signals","trades"]:
        getattr(a,key).to_parquet(folder/(key+".parquet"))
    a.ledger.to_csv(folder/"ledger.csv")
    a.signals.to_csv(folder/"signals.csv",index=False)
    (folder/"meta.json").write_text(json.dumps(dict(base_date=a.base_date,capital=500000,summary=row),default=str,indent=2),encoding="utf-8")
    sf=OUT/"summary.csv";pd.DataFrame([row]).to_csv(sf,mode="a",header=not sf.exists(),index=False)
    nav=a.ledger.copy().reset_index()
    nav.insert(0,"account_id",ident);nav.insert(1,"case",name);nav["capital"]=500000;nav["base_date"]=a.base_date;nav["start"]=start;nav["slip_bps"]=bps
    nf=OUT/"daily_ledger.csv";nav.to_csv(nf,mode="a",header=not nf.exists(),index=False)
    print(json.dumps({k:row[k] for k in ["case","start","slip_bps","cagr","vol","mdd","mean_cash"]}),flush=True)
def run_all():
    frozen=json.loads((HERE/"freeze.json").read_text(encoding="utf-8"))
    assert frozen["protocol_sha256"]==sha(PROTOCOL) and frozen["script_sha256"]==sha(__file__)
    for path,digest in frozen["input_and_code_sha256"].items():assert sha(ROOT/path)==digest,path
    OUT.mkdir(exist_ok=True);lag_sidecar()
    cases=case_list()
    cold=[x for x in cases if x[0] in ["GS_ALL_ETF_eligible_equal","V12_ETF_available"]]
    planned=[(c,"2013-04-12",bps) for bps in [9,20] for c in cases]
    # Explicit order: all 9bp main+cold before any 20bp.
    planned=[]
    for bps in [9,20]:
        planned.extend((c,"2013-04-12",bps) for c in cases)
        planned.extend((c,start,bps) for start in ["2025-06-04","2025-07-01"] for c in cold)
    assert len(planned)==48
    for c,start,bps in planned:
        try:run_one(c,start,bps)
        except Exception:
            failure={"case":c[0],"start":start,"bps":bps,"utc":datetime.now(timezone.utc).isoformat(),"traceback":traceback.format_exc()}
            with (HERE/"failures.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(failure,ensure_ascii=False)+"\n")
            print(json.dumps(failure,ensure_ascii=False),flush=True)
    changed=[path for path,digest in frozen["input_and_code_sha256"].items() if sha(ROOT/path)!=digest]
    s=pd.read_csv(OUT/"summary.csv")
    assert s.account_id.is_unique
    checks={"completed_accounts":len(s),"expected_accounts":48,"changed_inputs":changed,
            "protocol_sha256":sha(PROTOCOL),"script_sha256":sha(__file__),
            "target_cache_policy":"case+start scoped; immutable targets reused only across costs; execution recalculated",
            "macro_lag_columns":list(macro0.columns),"macro_shift_exact":True,"new_vintage_claim":False}
    (HERE/"checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    assert not changed
    print("ROUND4_COMPLETE "+json.dumps(checks,ensure_ascii=False),flush=True)
def portable_bytes():
    names=["protocol.json","freeze.json","run_matrix.py","inspect_coverage.py","checks.json","failures.jsonl","macro_lag_sidecar.csv",
           "results/summary.csv","results/daily_ledger.csv"]
    # Include all small target/meta receipts; complete parquet tables remain remote.
    files=[HERE/x for x in names if (HERE/x).is_file()]
    files+=sorted((OUT/"accounts").glob("*/signals.csv"))+sorted((OUT/"accounts").glob("*/meta.json"))
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(files):
            info=zipfile.ZipInfo(str(f.relative_to(HERE)).replace("\\","/"),date_time=(2026,11,13,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,f.read_bytes())
    return b.getvalue()
if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("mode",choices=["freeze","run","export_info","export_chunk"]);arg.add_argument("--offset",type=int,default=0);arg.add_argument("--size",type=int,default=48000);a=arg.parse_args()
    if a.mode=="freeze":freeze()
    elif a.mode=="run":run_all()
    else:
        b=portable_bytes()
        if a.mode=="export_info":print(json.dumps({"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}))
        else:print(base64.b64encode(b[a.offset:a.offset+a.size]).decode())

