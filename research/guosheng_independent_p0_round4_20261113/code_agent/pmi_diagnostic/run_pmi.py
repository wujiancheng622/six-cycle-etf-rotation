"""Post-main fixed PMI-version diagnostic: no selection among versions, no full vintage claim."""
from pathlib import Path
from datetime import datetime,timezone
import sys,importlib.util,json,hashlib,io,zipfile,base64,argparse,traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("round4_main",HERE.parent/"run_matrix.py")
rm=importlib.util.module_from_spec(spec);sys.modules[spec.name]=rm;spec.loader.exec_module(rm)
p,pd,np=rm.p,rm.pd,rm.np
SOURCES=[p.DATA/"pmi_as_reported.csv",p.DATA/"pmi_tushare_vintage.csv",p.DATA/"macro.parquet",p.DATA/"returns_gs.parquet",rm.ROOT/"guosheng_independent_round2_20261113/pit/run_pit.py",HERE.parent/"run_matrix.py",HERE/"protocol.json",Path(__file__)]
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def run():
    frozen=HERE/"freeze.json"
    assert not frozen.exists(),"Refuse overwrite post-main freeze"
    manifest={str(f.relative_to(rm.ROOT)):digest(f) for f in SOURCES}
    frozen.write_text(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),"post_main_diagnostic":True,"main_48_results_known":True,"diagnostic_nav_results_seen_before_freeze":False,"sha256":manifest},ensure_ascii=False,indent=2),encoding="utf-8")
    print("PMI_DIAGNOSTIC_FROZEN",digest(frozen),flush=True)
    original=pd.read_csv(SOURCES[0]);alternate=pd.read_csv(SOURCES[1])
    macro=p.MACRO.copy()
    def mapped(frame,col):
        z=frame.set_index(pd.to_datetime(frame.available))[col]
        assert z.index.is_unique and z.index.is_monotonic_increasing
        return z.reindex(macro.index,method="ffill")
    mapping={(1,1,-1):1,(1,1,1):2,(-1,1,1):3,(-1,-1,1):4,(-1,-1,-1):5,(1,-1,-1):6}
    def stages(money,growth):
        last=float("nan");out=[]
        for m,c,g in zip(money,macro.credit,growth):
            candidate=mapping.get((m,c,g))
            if candidate is not None:last=candidate
            out.append(last)
        return pd.Series(out,index=macro.index)
    old_growth=mapped(original,"direction")
    pd.testing.assert_series_equal(old_growth,macro.growth,check_names=False)
    pd.testing.assert_series_equal(stages(macro.money,old_growth),macro.stage,check_names=False)
    pd.testing.assert_series_equal(stages(macro.market,old_growth),macro.stage_market,check_names=False)
    new=macro.copy()
    new["growth"]=mapped(alternate,"direction")
    new["growth_votes"]=mapped(alternate,"votes")
    new["growth_available"]=mapped(alternate,"available")
    new["stage"]=stages(macro.money,new.growth)
    new["stage_market"]=stages(macro.market,new.growth)
    fixed=[c for c in macro.columns if c not in ["growth","growth_votes","growth_available","stage","stage_market"]]
    pd.testing.assert_frame_equal(new[fixed],macro[fixed])
    new.to_parquet(HERE/"macro_early_pmi_alternative.parquet")
    row=[]
    dates=p.HIST.index
    for i in range(1,len(dates)):
        day,known=dates[i],dates[i-1]
        if day<"2013-04-12":continue
        if day!="2013-04-12" and day[:7]==known[:7]:continue
        x,y=macro.loc[pd.Timestamp(known)],new.loc[pd.Timestamp(known)]
        row.append(dict(date=day,known=known,stage_original=int(x.stage),stage_alternative=int(y.stage),
                        original_basket="/".join(p.ASSETS_BY_STAGE[int(x.stage)]),
                        alternative_basket="/".join(p.ASSETS_BY_STAGE[int(y.stage)]),
                        stage_changed=bool(x.stage!=y.stage),basket_changed=p.ASSETS_BY_STAGE[int(x.stage)]!=p.ASSETS_BY_STAGE[int(y.stage)]))
    decisions=pd.DataFrame(row);decisions.to_csv(HERE/"monthly_decision_changes.csv",index=False)
    rm.OUT=HERE/"results";rm.OUT.mkdir(exist_ok=True);rm.CACHE={}
    case=("GS_ALL_ETF_eligible_equal_pmi_alternative",p.HIST,p.KEYS,lambda r:p.BoundaryGS(r,new,rm.gcuts),"GS")
    failures=[]
    for bps in [9,20]:
        try:rm.run_one(case,"2013-04-12",bps)
        except Exception:
            err={"bps":bps,"traceback":traceback.format_exc()};failures.append(err)
            with (HERE/"failures.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(err)+"\n")
            print(err,flush=True)
    changed=[str(f) for f in SOURCES if digest(f)!=manifest[str(f.relative_to(rm.ROOT))]]
    checks={"completed_accounts":len(pd.read_csv(rm.OUT/"summary.csv")),"failures":failures,
            "changed_inputs":changed,"unchanged_non_growth_columns":fixed,
            "monthly_decisions":len(decisions),"monthly_stage_changes":int(decisions.stage_changed.sum()),
            "monthly_basket_changes":int(decisions.basket_changed.sum()),
            "first_vintage_repair_claim":False,"main_48_unchanged":True}
    (HERE/"checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    assert not changed and not failures
    print("PMI_DIAGNOSTIC_DONE",json.dumps(checks),flush=True)
def portable():
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        names=["run_pmi.py","protocol.json","freeze.json","checks.json","monthly_decision_changes.csv","results/summary.csv","results/daily_ledger.csv","failures.jsonl"]
        for name in names:
            f=HERE/name
            if not f.exists():continue
            info=zipfile.ZipInfo(name,date_time=(2026,11,13,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,f.read_bytes())
    return b.getvalue()
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=["run","info","chunk"]);ap.add_argument("--offset",type=int,default=0);ap.add_argument("--size",type=int,default=48000);a=ap.parse_args()
    if a.mode=="run":run()
    else:
        b=portable()
        print(json.dumps({"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}) if a.mode=="info" else base64.b64encode(b[a.offset:a.offset+a.size]).decode())

