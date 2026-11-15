"""Check signal timing, frozen targets, and independent source receipts."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from study_inputs import BASE, CFG, END, HERE, KEYS, PARENT, PROVIDER
from sweep import BASELINE, OUT

TABLES=HERE/"tables"


def main():
    all_targets=pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("*/targets.parquet"))],ignore_index=True)
    assert (all_targets.known<all_targets.date).all()
    assert (all_targets.first_return<=all_targets.last_return).all()
    assert (all_targets.last_return<=all_targets.known).all()
    assert (all_targets.credit_date<=all_targets.known).all()
    assert (all_targets.growth_date<=all_targets.known).all()
    assert (all_targets.target_date<=all_targets.date).all()
    assert (all_targets.prior_exec_date<all_targets.date).all()
    np.testing.assert_allclose(all_targets[list(KEYS)].sum(axis=1)+all_targets.cash_target,1.,rtol=0,atol=1e-10)
    sums=[]
    for (mode,start),a in all_targets.loc[~all_targets.fcf_clean&all_targets.slip_bp.eq(9)].groupby(["mode","start"]):
        left=a[a.case.eq(BASELINE)].set_index("target_date")
        ix=(pd.read_parquet(PARENT/"data/returns_gs.parquet") if mode=="amount" else
            pd.read_parquet(PARENT/"accounts"/"actual"/start/"GS_monthly_9bp"/"ledger.parquet")).index
        for d in CFG["schedule"]["delayed_execution_only"]:
            right=a[a.case.eq(f"delay{d}_r63")].set_index("target_date")
            pd.testing.assert_frame_equal(left.loc[right.index,["known","phase","selected",*KEYS]],
                                          right[["known","phase","selected",*KEYS]],check_exact=True)
            # The shared cold-start target executes immediately.
            assert right.loc[start,"date"]==start
            r=right.drop(index=start)
            np.testing.assert_array_equal(ix.get_indexer(r.date)-ix.get_indexer(r.index),d)
            sums.append(dict(mode=mode,start=start,delay=d,targets_verified=len(right)))
    source_count=0
    for root in (BASE,PARENT):
        f=json.loads((root/"input_hashes.json").read_text(encoding="utf-8"))
        for rel,digest in f.items():
            assert PROVIDER.sha(HERE.parent/rel)==digest,rel
            source_count+=1
    x=pd.read_csv(TABLES/"all_metrics.csv")
    assert x.end.max()==END and not x.duplicated(["mode","start","case","slip_bp","fcf_clean","window"]).any()
    for g in x.groupby(["mode","start","slip_bp","window"]):
        _, t=g
        assert t.first.nunique()==1 and t.end.nunique()==1 and t.days.nunique()==1
    PROVIDER.write(HERE/"target_checks.json",dict(targets=len(all_targets),source_fingerprints=source_count,
                                                held_target_delays=sums,all_dates_and_comparisons_aligned=True))
    print("Checked",len(all_targets),"targets and",source_count,"source fingerprints.",flush=True)


if __name__=="__main__":
    main()
