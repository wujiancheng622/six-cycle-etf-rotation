"""Summarize frozen mechanism runs without changing any strategy or account."""
from pathlib import Path
import json, numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent
rows=[]
for meta in sorted((HERE/"accounts").glob("*/*/meta.json")):
    m=json.loads(meta.read_text(encoding="utf-8"))
    l=pd.read_parquet(meta.parent/"ledger.parquet")
    pnl=pd.read_parquet(meta.parent/"pnl.parquet")
    prev=l.nav.shift(1,fill_value=m["capital"])
    gross=pnl.sum(axis=1).div(prev)
    net=l.nav.div(prev)-1
    lg=float(np.log1p(gross).sum())
    ln=float(np.log1p(net).sum())
    rows.append(dict(scope=meta.parent.parent.name,cell=m["cell"],log_net=ln,log_gross_same_realized_positions=lg,
                     exact_log_fee_drag=ln-lg,fee_cash_sum=float(l.cost.sum())))
frame=pd.DataFrame(rows)
frame.to_csv(HERE/"fee_decomposition.csv",index=False,encoding="utf-8-sig")
rec=[]
for p in sorted(HERE.glob("*_monthly_contrasts.csv")):
    x=pd.read_csv(p); selected=x.month.isin(["2026-01","2026-07"])
    for field in ["B10_vs_M00","W01_vs_M00","BW11_vs_M00","interaction"]:
        total=float(x[field].sum()); head=float(x.loc[selected,field].sum())
        rec.append(dict(scope=p.name.removesuffix("_monthly_contrasts.csv"),contrast=field,total_log=total,
                        Jan_July_log=head,other_months_log=total-head,Jan_July_share=head/total if total else None))
pd.DataFrame(rec).to_csv(HERE/"concentration.csv",index=False,encoding="utf-8-sig")
print(frame.loc[frame.scope.eq("actual_2025-11-06_500000_9bp")].to_string(index=False))
print(pd.DataFrame(rec).query("scope == 'actual_2025-11-06_500000_9bp'").to_string(index=False))

