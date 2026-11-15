"""Isolate the effect of de-duplicating the two available policy categories."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LONG = HERE.parent / "guosheng_six_regime_long_20261112"
sys.path.insert(0,str(LONG))
from replay_long import ACT_START, MAIN, Inputs, load_account, run, state_path, stats

D = HERE/"evidence"
D.mkdir(exist_ok=True)
m = pd.read_parquet(LONG/"data/macro.parquet")
valid = m.stage.notna()
df = m.copy()
conflicts = {}
for name,a,b in (("lpr","lpr_1y","lpr_5y"),("rrr","rrr_B","rrr_S")):
    same = df[a].eq(df[b])
    conflict = (df[a]*df[b]).lt(0)
    conflicts[name] = int((valid&conflict).sum())
    df[name+"_group"] = np.sign(df[a]+df[b])
    assert df.loc[same,name+"_group"].eq(df.loc[same,a]).all()
df["money_score"] = df.market + df.lpr_group + df.rrr_group
df["money"] = np.sign(df.money_score)
df["stage"] = state_path(df.money,df.credit,df.growth)
changes = df.loc[valid].assign(original_stage=m.stage,original_money=m.money,
                               original_money_score=m.money_score)
changes["stage_changed"] = changes.stage.ne(changes.original_stage).fillna(False)
changes["money_changed"] = changes.money.ne(changes.original_money)
changes.index.name = "known_date"
changes.to_csv(D/"money_group_daily.csv",encoding="utf-8-sig")
print("days",len(changes),"money",int(changes.money_changed.sum()),"stage",int(changes.stage_changed.sum()),"conflicts",conflicts)
print("overweight example:",changes.loc["2025-06-13",["original_money_score","money_score","stage"]].to_dict())
inputs = Inputs.create()
results=[]
for mode,start in (("amount","2014-01-02"),("amount","2020-01-02"),("actual",ACT_START)):
    old = load_account(mode,start,MAIN)
    a = run(inputs,mode,start,MAIN,state=df)
    ix = old.signals.set_index("date")
    jx = a.signals.set_index("date")
    assert ix.index.equals(jx.index)
    for label,b in (("local_original",old),("two_categories_deduplicated",a)):
        results.append(dict(mode=mode,case=label,stage_changes=int(ix.phase.ne(jx.phase).sum()),
                            basket_changes=int(ix.selected.ne(jx.selected).sum()),targets=len(ix),**stats(b)))
    ix[["known","phase","selected"]].join(jx[["phase","selected"]],rsuffix="_dedup").to_csv(
        D/f"targets_{mode}_{start}.csv",encoding="utf-8-sig")
out=pd.DataFrame(results)
out.to_csv(D/"money_group_replay.csv",index=False,encoding="utf-8-sig")
print(out[["mode","start","case","stage_changes","basket_changes","targets","cagr","vol","mdd"]].to_string(index=False))

# Compare the matching report calendar years, including only January–October 2025.
old = load_account("amount","2014-01-02",MAIN)
v12 = load_account("amount","2014-01-02","V12")
paper = {2014:(12.9,8.8,9.5),2015:(56.5,14.3,6.9),2016:(28,14.5,7.5),
         2017:(13.2,7.9,4.3),2018:(7,4.6,4.6),2019:(25.7,9.8,4.9),
         2020:(36.4,19,12),2021:(30.9,9.6,4),2022:(8.2,10,7.2),
         2023:(7.1,11.4,9.1),2024:(21.2,6.8,3.2),2025:(23.2,11,4.4)}

def metrics(a,start,end):
    nav=a.ledger.nav.loc[start:end]
    prior = 1e6 if nav.index[0]==a.ledger.index[0] else a.ledger.nav.iloc[a.ledger.index.get_loc(nav.index[0])-1]
    x=np.r_[prior,nav]
    return (x[-1]/x[0]-1)*100,np.std(x[1:]/x[:-1]-1,ddof=1)*np.sqrt(252)*100,(1-x/np.maximum.accumulate(x)).max()*100

out=[]
for y,(ret,vol,mdd) in paper.items():
    a=f"{y}-01-01"
    b=f"{y}-10-31" if y==2025 else f"{y}-12-31"
    gr,gv,gd=metrics(old,a,b)
    vr,vv,vd=metrics(v12,a,b)
    out.append(dict(year=y,cutoff=b,paper_return_pct=ret,paper_vol_pct=vol,paper_mdd_pct=mdd,
                    local_return_pct=gr,local_vol_pct=gv,local_mdd_pct=gd,
                    v12_return_pct=vr,v12_vol_pct=vv,v12_mdd_pct=vd,local_minus_paper_return_pp=gr-ret))
pd.DataFrame(out).to_csv(D/"report_vs_local_years.csv",index=False,encoding="utf-8-sig")
print(pd.DataFrame(out).round(2).to_string(index=False))

# Reference cut at the last date of the paper's stated 2025 observation period.
common=[]
for case,account in (("local_original",old),("V12",v12)):
    l=account.ledger.loc[:"2025-10-31"]
    prev=l.nav.shift(1,fill_value=1e6)
    years=(pd.Timestamp(l.index[-1])-pd.Timestamp(account.base_date)).days/365.25
    ret,vol,mdd=metrics(account,l.index[0],l.index[-1])
    common.append(dict(case=case,start=l.index[0],end=l.index[-1],days=len(l),
                       cagr=(1+ret/100)**(1/years)-1,vol=vol/100,mdd=mdd/100,
                       annual_one_way_turnover=float((l.notional/prev).sum()/years/2)))
pd.DataFrame(common).to_csv(D/"report_cutoff_local.csv",index=False,encoding="utf-8-sig")
print(pd.DataFrame(common).to_string(index=False))
