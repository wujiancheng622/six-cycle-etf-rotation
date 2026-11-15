"""Assemble a portable, bounded evidence package from the two studies."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LONG = ROOT/"guosheng_six_regime_long_20261112"
EARLY = ROOT/"guosheng_fcf_early_history_20261113"
CUT = ROOT/"guosheng_fcf_cutoff_20261113"
FREQ = ROOT/"guosheng_frequency_cost_20261113"
sys.dont_write_bytecode = True


def early_path():
    sys.path.insert(0,str(EARLY))
    import publication_control as p
    returns, _, _, _ = p.inputs()
    phases = pd.read_csv(EARLY/"phases.csv",parse_dates=["known_date"]).set_index("known_date")
    phases.stage = phases.stage.astype("Int64")
    phases.growth_available = pd.to_datetime(phases.growth_available)
    releases = p.PROXY_RELEASES | {"fcf":p.RULE_DATE}
    allowed = {k:returns.index[returns.index.searchsorted(d,side="right")] for k,d in releases.items()}
    with p.apply_policy(lambda n,r,s:p.PublishedIndexRotation(r,phases,allowed["fcf"],allowed),p.KEYS):
        a = p.engine.economic(p.MAIN,returns,None,"2011-04-15",p.SLIP)
    p.verify_case(a)
    expected=pd.read_csv(EARLY/"publication_control.csv").query(
        "case=='also_remove_backfilled_bond_and_dividend' and window=='full'").iloc[0]
    np.testing.assert_allclose([p.brief(a)[k] for k in ("cagr","vol","mdd")],
                               [expected[k] for k in ("cagr","vol","mdd")],rtol=0,atol=1e-12)
    return a.ledger.nav


def main():
    select={
      "guosheng_six_regime_20261112": ["README.md","protocol.json","regimes.py","pbc_loans.py","run_study.py","market.py"],
      LONG.name: ["研究报告.md","过拟合审计.md","audit_checks.json","coverage.json","protocol.json","early_inputs.py",
                  "verify_long.py","tables/replays.csv","tables/years.csv","tables/blocks.csv","tables/proxy_stress.csv",
                  "tables/concentration.csv","tables/rolling_summary.csv","tables/asset_attribution.csv","tables/cost_curve.csv"],
      CUT.name: ["复核说明.md","comparison.csv","periods.csv","verification.json","replay_fcf_cutoff.py"],
      EARLY.name: ["前推与P0复核.md","comparison.csv","publication_control.csv","credit_from_2009.csv",
                   "pmi_from_2009.csv","verification.json","publication_control.py","publication_control_targets.csv","replay_early.py"],
      "guosheng_v12_return_dependence_20261113": ["对比复核.md","tables/scenarios.csv","tables/anchors.csv",
                    "tables/timing_controls.csv","tables/premium_and_asset_returns.csv"],
      "guosheng_parameter_robustness_20261113": ["参数稳健性复核.md","protocol.json","tables/peak_checks.csv","tables/neighborhood.csv"],
      FREQ.name: ["频率与参数复核.md","protocol.json","verification.json","tables/frequency_and_cost.csv",
                  "tables/immediate_macro_neighbors.csv","frequency_replay.py"],
      "review_runs/20261111_v12_vs_three_etf": ["replay.py","analyze.py"],
    }
    zip_path=HERE/"Pro复核证据包.zip"
    initial=1_000_000.
    rows=[]
    def add(case,nav,base):
        rows.append(pd.DataFrame({"case":case,"date":[base,*nav.index.tolist()],
                                  "nav":[initial,*nav.to_list()]}))
    hist=pd.read_csv(EARLY/"nav.csv").query("study=='extended_first'").set_index("date")
    add("2011_fcf_cutoff",hist.GS,"2011-04-14")
    add("2011_v12",hist.V12,"2011-04-14")
    add("2011_three_index_cutoff",early_path(),"2011-04-14")
    cut=pd.read_csv(CUT/"nav.csv",index_col="date")
    for k in ("original_proxy","V12","fcf_from_published_index","old_mask_after_risk"):
        add("2013_"+k,cut[k],"2013-04-11")
    for name in ("monthly","weekly","daily","stage_change","V12"):
        f=FREQ/"runs/actual_2025-11-06"
        a=pd.read_parquet(f/f"{name}_ledger.parquet")
        add("2025_quote_"+name,a.nav,"2025-11-05")
    with ZipFile(zip_path,"w",ZIP_DEFLATED) as z:
        for name in ("README.md","国盛六周期综合分析.md","发给网页端Pro的提示词.txt","recalculate.py"):
            z.write(HERE/name,name)
        z.writestr("daily_nav.csv",pd.concat(rows,ignore_index=True).to_csv(index=False).encode("utf-8-sig"))
        for folder,paths in select.items():
            for path in paths:
                z.write(ROOT/folder/path,"evidence/"+folder+"/"+path)
        for name in ("monthly","V12","stage_change"):
            f=FREQ/"runs/actual_2025-11-06"
            for k in ("trades","signals","ledger"):
                a=pd.read_parquet(f/f"{name}_{k}.parquet")
                z.writestr(f"quote_ledger/{name}_{k}.csv",a.to_csv(index=k=="ledger").encode("utf-8-sig"))
    print(zip_path,zip_path.stat().st_size)


if __name__=="__main__":
    main()
