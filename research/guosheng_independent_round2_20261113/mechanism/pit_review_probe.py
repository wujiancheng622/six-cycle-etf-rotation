"""Read-only audit of PIT boundary classes against saved full-history accounts."""
from pathlib import Path
import sys, importlib.util, json, hashlib, traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SCRIPT=HERE.parent/"pit/run_pit.py"
import numpy as np,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    before=sha(SCRIPT)
    spec=importlib.util.spec_from_file_location("pit_under_review",SCRIPT)
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    start="2014-07-16";capital=500000;slip=.0009
    rows=[]
    vr=pd.read_parquet(p.OUT/"returns_v12_earlier_actual.parquet")
    vtab=pd.read_csv(p.OUT/"availability_v12.csv")
    vcuts=dict(zip(vtab.key,vtab.first_actual_return))
    gcuts=p.COVER["actual_return_from"]
    for name,r,cuts,keys in [("GS_ETF_available",p.HIST,gcuts,p.KEYS),("V12_ETF_available",vr,vcuts,p.VKEYS)]:
        folder=p.OUT/"accounts"/f"{name}__{start}__{capital}__9"
        assert (folder/"meta.json").exists(), "Baseline not yet complete: "+str(folder)
        base={k:pd.read_parquet(folder/(k+".parquet")) for k in ("ledger","weights","pnl","signals","trades")}
        altered=r.copy()
        for k,d in cuts.items():
            use=altered.index<d
            altered.loc[use,k]=np.resize([.30,-.30],int(use.sum()))
        factory=(lambda n,x,_:p.BoundaryGS(x,p.MACRO,gcuts)) if name.startswith("GS") else (lambda n,x,_:p.BoundaryV12(x,vcuts))
        rerun=p.run(name+"_source_perturbation",altered,start,capital,slip,factory,keys,
                    engine_name=None if name.startswith("GS") else "V12",save=False)
        deltas={}
        for key in base:
            pd.testing.assert_frame_equal(getattr(rerun,key),base[key],check_exact=True)
            if key=="ledger":deltas["max_nav_error"]=float((rerun.ledger.nav-base[key].nav).abs().max())
        extra_cash=[];selected_bad=[];eligibility_zero=[]
        # Count common valid dates independently rather than calling the reviewed train().
        clean=r.copy()
        for k,d in cuts.items():clean.loc[clean.index<d,k]=np.nan
        exclusions=clean.attrs.get("excluded_risk_dates",{})
        for row in base["signals"].itertuples():
            original=row.requested.split("/") if name.startswith("GS") else list(p.VKEYS)
            past=clean.loc[:row.known,original]
            individually=[]
            for k in original:
                mask=past[k].notna() & ~past.index.isin(exclusions.get(k,[]))
                if int(mask.sum())>=63:individually.append(k)
            joint_ex={day for k in individually for day in exclusions.get(k,[])}
            count=int((past[individually].notna().all(axis=1)&~past.index.isin(joint_ex)).sum()) if individually else 0
            chosen=row.selected.split("/") if isinstance(row.selected,str) and row.selected else []
            if individually and count<63:
                extra_cash.append(dict(date=row.date,known=row.known,individual=individually,joint_count=count,selected=chosen))
            if not individually:eligibility_zero.append(row.date)
            if chosen:
                joint_ex={day for k in chosen for day in exclusions.get(k,[])}
                valid=past[chosen].notna().all(axis=1)&~past.index.isin(joint_ex)
                if int(valid.sum())<63:selected_bad.append(row.date)
        assert not selected_bad
        rows.append(dict(case=name,signals=len(base["signals"]),source_perturbation_five_tables_exact=True,
            **deltas,joint_insufficient_fallback_count=len(extra_cash),joint_insufficient_fallbacks=extra_cash,
            no_individually_eligible_dates=eligibility_zero,invalid_selected_windows=selected_bad))
    assert sha(SCRIPT)==before,"Reviewed script changed during QA"
    output=dict(reviewed_script=str(SCRIPT),reviewed_script_sha256=before,results=rows,
        scope="Actual full archived inputs; all pre-ETF availability values set to alternating plus/minus30%; 500k, 2014-07-16 start, 9bp. No original output was modified.")
    (HERE/"pit_review_probe.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=="__main__":
    try:main()
    except Exception:
        (HERE/"pit_review_probe_failure.json").write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2),encoding="utf-8")
        raise

