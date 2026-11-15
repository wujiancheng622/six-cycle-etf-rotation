"""Read-only cash-budget and ETF readiness summary of completed PIT accounts."""
import sys,json,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("pit_review_cash_module",HERE.parent/"pit/run_pit.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
rows=[];readiness=[];boundary=[];comparison=[]
for case,r,keys,cuts in [
    ("GS_ETF_available",p.HIST,p.KEYS,p.COVER["actual_return_from"]),
    ("V12_ETF_available",pd.read_parquet(p.OUT/"returns_v12_earlier_actual.parquet"),p.VKEYS,
        dict(pd.read_csv(p.OUT/"availability_v12.csv")[["key","first_actual_return"]].to_numpy()))]:
    ex=r.attrs.get("excluded_risk_dates",{})
    ready={}
    for k in keys:
        usable=r.index[(r.index>=cuts[k])&r[k].notna()&~r.index.isin(ex.get(k,[]))]
        known=usable[62]
        day=r.index[r.index.get_loc(known)+1]
        ready[k]=day
        readiness.append(dict(case=case,asset=k,first_actual_return=cuts[k],risk63_known=known,first_eligible_execution_day=day))
    folder=p.OUT/"accounts"/f"{case}__2014-07-16__500000__9"
    w=pd.read_parquet(folder/"weights.parquet")
    sig=pd.read_parquet(folder/"signals.parquet")
    if "date" not in sig.columns:sig=sig.reset_index()
    ct=sig.set_index("date").cash_target.reindex(w.index).ffill()
    for year in sorted(set(w.index.str[:4])):
        mask=w.index.str.startswith(year)
        rows.append(dict(case=case,year=year,days=int(mask.sum()),realized_cash_mean=float(w.loc[mask,"cash"].mean()),
            scheduled_cash_mean=float(ct.loc[mask].mean()),cash_max=float(w.loc[mask,"cash"].max())))
    for label,start,end in [("before_2020","2014-07-16","2019-12-31"),("2020_pre_complete","2020-01-01","2020-04-24"),
        ("2020_after_complete","2020-04-27","2020-12-31"),("2021_2024","2021-01-01","2024-12-31"),("2025_onward","2025-01-01","2026-11-12"),
        ("all_assets_after_ready",max(ready.values()),"2026-11-12"),("after_GS_monthly_next_signal","2025-07-01","2026-11-12")]:
        sub=w.loc[start:end,"cash"];st=ct.loc[start:end]
        rows.append(dict(case=case,year=label,days=len(sub),realized_cash_mean=float(sub.mean()),
            scheduled_cash_mean=float(st.mean()),cash_max=float(sub.max())))
    if case.startswith("GS"):
        boundary=sig[(sig.date>="2019-10-01")&(sig.date<="2020-07-31")].to_dict(orient="records")
# Verify all V12 variants started together after full availability converge by complete ledger, not just summary.
for bps in [9,20]:
    base=p.OUT/"accounts"/f"V12_original__2020-04-27__500000__{bps}"
    for other in ["V12_earlier_actual","V12_ETF_available"]:
        folder=p.OUT/"accounts"/f"{other}__2020-04-27__500000__{bps}"
        checked=[]
        for table in ["ledger","weights","pnl","trades"]:
            a=pd.read_parquet(base/(table+".parquet"));b=pd.read_parquet(folder/(table+".parquet"))
            pd.testing.assert_frame_equal(a,b,check_exact=True);checked.append(table)
        comparison.append(dict(start="2020-04-27",bps=bps,other=other,exact_tables=checked))
out=dict(readiness=readiness,cash_statistics=rows,gs_2019_2020_signals=boundary,v12_common_start_exact=comparison,
    note="Scheduled cash is signal cash target forward-filled over trading dates; realized cash is close ledger fraction. Ready dates are within the archived panel, which begins in 2013 for GS and therefore cannot establish the earliest growth ETF readiness in 2012. GS 2025-06-04 FCF readiness still waits for 2025-07-01 monthly rebalance. Eligibility alone does not establish old macro vintage or asset-selection validity.")
(HERE/"pit_cash_convergence.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(json.dumps(dict(readiness=readiness,cash_statistics=[x for x in rows if not x["year"].isdigit()],gs_2019_2020_signals=boundary,v12_common_start_exact=comparison),ensure_ascii=False,indent=2,default=str))

