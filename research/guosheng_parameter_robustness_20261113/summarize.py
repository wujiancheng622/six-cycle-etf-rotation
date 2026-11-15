"""Summarize the frozen neighborhoods and locate sensitive calendar decisions."""
import json

import numpy as np
import pandas as pd

from study_inputs import CFG, DEFAULT, END, HERE, KEYS, PROVIDER
from sweep import BASELINE, BENCHMARK, OUT, cases, hash_inputs

TABLES=HERE/"tables"


def save(df,name):
    TABLES.mkdir(exist_ok=True)
    df.to_csv(TABLES/(name+".csv"),index=False,encoding="utf-8-sig")


def lookup(metrics):
    base=metrics.loc[metrics.case.eq(BASELINE)].rename(columns={c:"base_"+c for c in
        ("cagr","total","vol","mdd","cagr_with_2025_zeroed")})
    v12=metrics.loc[metrics.case.eq(BENCHMARK)].rename(columns={c:"v12_"+c for c in
        ("cagr","total","vol","mdd","cagr_with_2025_zeroed")})
    i=["mode","start","slip_bp","window"]
    b=base[i+["fcf_clean"]+[c for c in base if c.startswith("base_")]]
    v=v12[i+[c for c in v12 if c.startswith("v12_")]]
    result=(metrics.merge(b,how="left",on=i+["fcf_clean"],validate="many_to_one")
            .merge(v,how="left",on=i,validate="many_to_one"))
    assert result[["base_cagr","v12_cagr"]].notna().all().all()
    for c in ("cagr","total","vol","mdd","cagr_with_2025_zeroed"):
        result[c+"_vs_base"]=result[c]-result["base_"+c]
        result[c+"_vs_v12"]=result[c]-result["v12_"+c]
    result["better_return_dd5"]=result.cagr_vs_v12.gt(0)&result.mdd_vs_v12.le(.05)
    result["return3_dd5"]=result.cagr_vs_v12.ge(.03)&result.mdd_vs_v12.le(.05)
    return result


def summary(metrics,config):
    base = metrics.loc[metrics.slip_bp.eq(9)].merge(config,on="case",how="left",validate="many_to_one")
    base["local_grid"] = base.schedule.isin(("D1","D2","D3","D4","D5"))&base.lookback.isin((56,63,70)) & (
        base.groups.fillna("").str.contains("grid"))
    base["local_grid"] |= (base.schedule.eq("D1")&base.lookback.isin((56,63,70)) &base.groups.fillna("").str.contains("risk"))
    base["close_risk"] = (base.groups.fillna("").str.split(",").apply(lambda x:"risk" in x)
                          &base.lookback.between(56,70))
    is_group=lambda col:base.groups.fillna("").str.split(",").apply(lambda x:col in x)
    families = {
        "local_timing_risk_grid": base.local_grid,
        "joint_grid": is_group("grid"),
        "monthly_dates": is_group("timing"),
        "month_start_risk_56_to_70": base.close_risk,
        "local_macro": base.apply(lambda x: x.macro in CFG["local_macro_neighbors"] and
                        x.value in CFG["local_macro_neighbors"][x.macro],axis=1),
        "fixed_target_delays": is_group("execution_delay"),
        "all_experiments": base.case.ne(BENCHMARK),
    }
    rows=[]
    for family,mask in families.items():
        for keys,x in base.loc[mask].groupby(["mode","start","fcf_clean","window"],sort=False):
            a,b,c,d=keys
            if family=="local_timing_risk_grid":
                assert len(x)==15,keys
            rows.append(dict(family=family,mode=a,start=b,fcf_clean=c,window=d,n=len(x),
                             cagr_min=x.cagr.min(),cagr_p25=x.cagr.quantile(.25),cagr_median=x.cagr.median(),
                             cagr_p75=x.cagr.quantile(.75),cagr_max=x.cagr.max(),
                             cagr_max_neighbor_loss=float(-x.cagr_vs_base.min()),
                             total_min=x.total.min(),total_median=x.total.median(),total_max=x.total.max(),
                             vol_min=x.vol.min(),vol_median=x.vol.median(),vol_max=x.vol.max(),
                             mdd_min=x.mdd.min(),mdd_median=x.mdd.median(),mdd_max=x.mdd.max(),
                             beat_v12=int(x.cagr_vs_v12.gt(0).sum()),
                             beat_and_mdd_within_5=int(x.better_return_dd5.sum()),
                             beat_3_and_mdd_within_5=int(x.return3_dd5.sum()),
                             v12_cagr=x.v12_cagr.iloc[0],v12_total=x.v12_total.iloc[0],
                             v12_mdd=x.v12_mdd.iloc[0],
                             median_cagr_without_2025=x.cagr_with_2025_zeroed.median(),
                             median_gap_without_2025=x.cagr_with_2025_zeroed_vs_v12.median()))
    return pd.DataFrame(rows)


def peak_checks(metrics,config):
    f=metrics.loc[metrics.slip_bp.eq(9)&metrics.case.ne(BENCHMARK)].merge(config,on="case",validate="many_to_one")
    variants={
        "monthly_boundary":(["ME_r63"],["D2_r63","D3_r63"]),
        "covariance_sessions_63":(["D1_r56","D1_r60"],["D1_r66","D1_r70"]),
    }
    for k,(low,high) in CFG["local_macro_neighbors"].items():
        variants[k]=([f"{k}_{low}"],[f"{k}_{high}"])
    rows=[]
    for (mode,start,clean,window),x in f.groupby(["mode","start","fcf_clean","window"],sort=False):
        s=x.set_index("case")
        for axis,(before,after) in variants.items():
            if not all(k in s.index for k in before+after):
                continue
            left=s.loc[before,"cagr_vs_base"]
            right=s.loc[after,"cagr_vs_base"]
            both=pd.concat([left,right])
            rows.append(dict(mode=mode,start=start,fcf_clean=clean,window=window,axis=axis,
                             base_cagr=s.loc[BASELINE,"cagr"],left_cases="/".join(before),right_cases="/".join(after),
                             left_deltas="/".join(f"{x:.8f}" for x in left),
                             right_deltas="/".join(f"{x:.8f}" for x in right),
                             nearest_left_delta=float(left.iloc[-1] if len(left)>1 else left.iloc[0]),
                             nearest_right_delta=float(right.iloc[0]),largest_loss=float(-both.min()),
                             all_neighbors_below_base=bool((both<0).all()),
                             all_neighbors_lose_3pp=bool((both<-.03).all())))
    return pd.DataFrame(rows)


def timing_details(monthly,targets):
    focus=["D2_r63","D4_r63","D5_r63","D6_r63","D8_r63","D10_r63","delay1_r63","delay3_r63","delay5_r63","ME_r63"]
    rows=[]
    for (mode,start), x in monthly.loc[~monthly.fcf_clean&monthly.slip_bp.eq(9)].groupby(["mode","start"]):
        p=x.pivot(index="month",columns="case",values="total")
        y=targets.loc[targets["mode"].eq(mode)&targets.start.eq(start)&~targets.fcf_clean&targets.slip_bp.eq(9)]
        y=y.assign(month=y.target_date.str[:7])
        y=y[y.case.isin([BASELINE,*focus])]
        assert not y.duplicated(["month","case"]).any()
        q={case:g.set_index("month") for case,g in y.groupby("case")}
        for k in focus:
            if k not in q or k not in p:
                continue
            common=p.index.intersection(q[BASELINE].index).intersection(q[k].index)
            for m in common:
                r=q[BASELINE].loc[m]; v=q[k].loc[m]
                seen_credit=r.credit_date!=v.credit_date
                seen_pmi=r.growth_date!=v.growth_date
                delta=np.log1p(p.at[m,k])-np.log1p(p.at[m,BASELINE])
                rows.append(dict(mode=mode,start=start,case=k,month=m,base_trade_date=r.date,
                                 variant_trade_date=v.date,base_known=r.known,variant_known=v.known,
                                 base_basket=r.selected,variant_basket=v.selected,
                                 same_basket=r.selected==v.selected,
                                 new_credit_release=seen_credit,new_pmi_release=seen_pmi,
                                 base_credit_date=r.credit_date,variant_credit_date=v.credit_date,
                                 base_growth_date=r.growth_date,variant_growth_date=v.growth_date,
                                 base_money_direction=r.money_direction,variant_money_direction=v.money_direction,
                                 base_credit_direction=r.credit_direction,variant_credit_direction=v.credit_direction,
                                 base_growth_direction=r.growth_direction,variant_growth_direction=v.growth_direction,
                                 base_month_return=float(p.at[m,BASELINE]),variant_month_return=float(p.at[m,k]),
                                 delta_month_log_return=float(delta)))
    return pd.DataFrame(rows)


def main():
    stores=[]
    records=[]
    for mode,starts in CFG["starts"].items():
        for start in starts:
            folder=OUT/f"{mode}_{start}"
            stores.append(folder)
            records.append(json.loads((folder/"checks.json").read_text(encoding="utf-8")))
    for r in records:
        assert r["inputs_sha256"]==hash_inputs()
        assert sorted(r["original_matched"])==sorted((BASELINE,BENCHMARK))
    fetch=lambda f:pd.concat([pd.read_parquet(p/(f+".parquet")) for p in stores],ignore_index=True)
    metrics=lookup(fetch("metrics"))
    monthly=fetch("monthly")
    targets=fetch("targets")
    config=pd.DataFrame(c.row() for c in cases().values()).rename(columns={"name":"case"})
    assert len(config)==len(cases())
    periods=summary(metrics,config)
    peaks=peak_checks(metrics,config)
    changes=timing_details(monthly,targets)
    for name,frame in (("cases",config),("all_metrics",metrics),("neighborhood",periods),
                       ("peak_checks",peaks),("timing_monthly_effect",changes),
                       ("unavailable",pd.DataFrame([u for r in records for u in r["unavailable"]],
                                                   columns=["mode","start","case","fcf_clean","slip_bp","reason"]))):
        save(frame,name)
    curves=fetch("curves")
    curves.to_parquet(TABLES/"selected_nav.parquet",index=False)
    PROVIDER.write(HERE/"verification.json",dict(scopes=records,unique_gs_specs=len(config),
                   replayed_accounts=sum(r["completed"] for r in records),
                   attempted_accounts=sum(r["attempts"] for r in records),
                   nominal_trials_per_2014_history_year=float(len(config)/metrics.loc[
                        metrics.start.eq("2014-01-02")&metrics.window.eq("full"),"years"].iloc[0])))
    print("Scenarios",len(config),"accounts",sum(r["completed"] for r in records),
          "paired metrics",len(metrics),"dated timing comparisons",len(changes),flush=True)


if __name__=="__main__":
    main()
