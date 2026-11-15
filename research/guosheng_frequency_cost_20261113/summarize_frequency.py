"""Build frequency, fee, and parameter-neighborhood tables from the saved runs."""
import json

import numpy as np
import pandas as pd

from frequency_replay import (BENCHMARK, CFG, FOLLOWUP, GROUPS, HERE, NORMAL_CAPITAL,
                              PREVIOUS, PROTOCOL, PROVIDER)

TABLES = HERE/"tables"
JOIN = ["mode","start","slip_bps","capital","fcf_clean","window"]


def save(df,name):
    TABLES.mkdir(exist_ok=True)
    df.to_csv(TABLES/f"{name}.csv",index=False,encoding="utf-8-sig")


def neighborhoods(raw):
    r = raw.loc[raw.slip_bps.eq(9)&raw.capital.eq(NORMAL_CAPITAL)&~raw.fcf_clean
                &raw.window.isin(("full","2026-07 to end"))&raw.frequency.isin(GROUPS)]
    output = []
    for keys,g in r.groupby(["mode","start","window","frequency"],sort=False):
        mode,start,window,frequency = keys
        base = g.loc[g.case.eq(frequency)].iloc[0]
        samples = {"covariance":g[g.macro.eq("")&g.ordinal.eq(1)&g.lookback.ne(63)]}
        samples.update({name:x for name,x in g[g.macro.ne("")].groupby("macro")})
        if frequency=="weekly":
            samples["weekly_ordinal"] = g[g.ordinal.ne(1)]
        for axis,d in samples.items():
            x = pd.concat([d,base.to_frame().T])
            assert d.shape[0]>=2
            column = "cagr" if window=="full" else "total"
            change = d[column].to_numpy()-base[column]
            output.append(dict(mode=mode,start=start,window=window,frequency=frequency,axis=axis,
                               metric=column,base_return=base[column],n_neighbors=len(d),
                               min_return=x[column].min(),max_return=x[column].max(),
                               min_mdd=x.mdd.min(),max_mdd=x.mdd.max(),
                               min_fee=x.annual_fee.min(),max_fee=x.annual_fee.max(),
                               cases="/".join(d.case),
                               deltas="/".join(f"{v:.8f}" for v in change),
                               all_neighbors_below_base=bool(np.all(change<0.)),
                               max_neighbor_loss=float(max(0.,-np.min(change)))))
    return pd.DataFrame(output)


def main():
    sources=[]
    metrics=[]
    monthly=[]
    follow_receipts=[]
    local_macro=[]
    digest=PROVIDER.sha(PROTOCOL)
    follow_digest=PROVIDER.sha(FOLLOWUP)
    for mode,starts in CFG["starts"].items():
        for start in starts:
            p=HERE/"runs"/f"{mode}_{start}"
            record=json.loads((p/"checks.json").read_text(encoding="utf-8"))
            assert record["protocol_sha256"]==digest and record["completed"]==record["attempted"]
            assert {"monthly",BENCHMARK}<=set(record["archive_reconciled"])
            sources.append(record)
            metrics.append(pd.read_parquet(p/"metrics.parquet"))
            monthly.append(pd.read_parquet(p/"monthly_returns.parquet"))
            p=HERE/"close_macro_runs"/f"{mode}_{start}"
            close=json.loads((p/"checks.json").read_text(encoding="utf-8"))
            assert close["protocol_sha256"]==digest and close["followup_sha256"]==follow_digest
            assert close["completed"]==close["attempted"]
            follow_receipts.append(close)
            local_macro.append(pd.read_parquet(p/"metrics.parquet"))
    raw = pd.concat(metrics,ignore_index=True)
    old = pd.read_parquet(PREVIOUS/"runs"/"amount_2014-01-02"/"metrics.parquet")
    old = old.loc[old.case.eq("W_r63")&old.slip_bp.eq(9)&old.window.eq("full")]
    new = raw.loc[raw.start.eq("2014-01-02")&raw.case.eq("weekly")&raw.slip_bps.eq(9)&raw.window.eq("full")&~raw.fcf_clean]
    np.testing.assert_allclose(new.iloc[0][["cagr","mdd"]].to_numpy(float),
                               old.iloc[0][["cagr","mdd"]].to_numpy(float),atol=.0001,rtol=0)
    keys = JOIN+["case"]
    assert not raw.duplicated(keys).any()
    for _,g in raw.groupby(JOIN,dropna=False):
        assert g.first.nunique() == g["last"].nunique() == g.days.nunique() == 1
    compare = raw.loc[raw.case.eq(BENCHMARK),JOIN+["cagr","mdd","vol","total","annual_fee"]]
    raw = raw.merge(compare.rename(columns=lambda c:c if c in JOIN else "v12_"+c),
                    how="left",on=JOIN,validate="many_to_one")
    for c in ("cagr","mdd","vol","total","annual_fee"):
        raw[c+"_vs_v12"]=raw[c]-raw["v12_"+c]
    save(raw,"all_metrics")
    primary = raw.loc[raw.case.isin((*GROUPS,BENCHMARK))]
    save(primary,"frequency_and_cost")
    save(neighborhoods(raw),"nearby_parameters")
    save(pd.concat(monthly,ignore_index=True),"core_monthly")
    near=pd.concat(local_macro,ignore_index=True)
    base=raw.loc[raw.case.eq(raw.frequency)&raw.frequency.isin(GROUPS)&raw.slip_bps.eq(9)&
                 raw.capital.eq(NORMAL_CAPITAL)&~raw.fcf_clean,JOIN+["frequency","cagr","mdd","total"]]
    near=near.merge(base.rename(columns={"cagr":"base_cagr","mdd":"base_mdd","total":"base_total"}),
                    on=JOIN+["frequency"],validate="many_to_one")
    for col in ("cagr","mdd","total"):
        near[col+"_vs_base"]=near[col]-near["base_"+col]
    save(near,"immediate_macro_neighbors")
    check=dict(accounts=sum(r["completed"] for r in sources),scopes=sources,results_rows=len(raw),
               followup_accounts=sum(r["completed"] for r in follow_receipts),followup_scopes=follow_receipts,
               core_cost_identity_checked=True,prior_monthly_and_v12_reconciled=True)
    PROVIDER.write(HERE/"verification.json",check)
    print("Verified",check["accounts"],"core accounts,",check["followup_accounts"],
          "immediate-neighbor accounts, and",len(raw)+len(near),"metric rows.",flush=True)


if __name__=="__main__":
    main()
