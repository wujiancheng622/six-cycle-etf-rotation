"""Leave unpublished index exposures in cash using the same phase allocations."""
import json
import numpy as np
import pandas as pd

from replay_early import (END, HERE, KEYS, PROXY_RELEASES, RULE_DATE, SLIP, brief,
                          engine, inputs, trailing, verify_case, window, write)
from replay_fcf_cutoff import AvailableRotation
from replay_long import ASSETS_BY_STAGE, MAIN, apply_policy
from run_study import risk_parity


class PublishedIndexRotation(AvailableRotation):
    """Apply the existing FCF rule to the other explicitly dated index proxies too."""
    def __init__(self,returns,phases,first_return,allowed):
        super().__init__(returns,phases,first_return,True)
        self.allowed = allowed
        for key,date in allowed.items():
            self.training.loc[self.training.index<date,key] = np.nan

    def targets(self,day,known,i,nav):
        macro = self.states.loc[pd.Timestamp(known)]
        if pd.isna(macro.stage):
            return None,None
        if self.pending is not None and day[:7]==known[:7]:
            if i==self.last_rebalance+1:
                return None,self.pending
            return None,None
        original = ASSETS_BY_STAGE[int(macro.stage)]
        available = [k for k in original if k not in self.allowed or
                     len(trailing(self.training,self.excluded_risk_dates,known,(k,)))==63]
        train = trailing(self.training,self.excluded_risk_dates,known,available)
        assert len(train)==63
        for key in set(available).intersection(self.allowed):
            assert train.index[0]>=self.allowed[key]
        cov=train.cov().to_numpy()
        assert np.linalg.eigvalsh(cov).min()>0
        fractions=risk_parity(cov) if len(available)>1 else np.ones(1)
        funded=len(available)/len(original)
        weights=pd.Series(funded*fractions,index=available).reindex(KEYS,fill_value=0.).to_numpy()
        self.pending=weights
        self.last_rebalance=i
        self.receipts.append(dict(date=day,known=known,phase=int(macro.stage),
                                  requested="/".join(original),selected="/".join(available),
                                  first_return=train.index[0],last_return=train.index[-1],
                                  cash_target=1-funded,**dict(zip(KEYS,weights))))
        return weights,weights


def main():
    returns,benchmark,_,_=inputs()
    phases=pd.read_csv(HERE/"phases.csv",parse_dates=["known_date"]).set_index("known_date")
    phases.stage=phases.stage.astype("Int64")
    phases.growth_available=pd.to_datetime(phases.growth_available)
    start=json.loads((HERE/"verification.json").read_text(encoding="utf-8"))["first_run"]
    releases=PROXY_RELEASES | {"fcf":RULE_DATE}
    allowed={k:returns.index[returns.index.searchsorted(d,side="right")] for k,d in releases.items()}
    def account():
        return engine.economic(MAIN,returns,None,start,SLIP)

    with apply_policy(lambda n,r,p:PublishedIndexRotation(r,phases,allowed["fcf"],allowed),KEYS):
        cleaned=account()
    verify_case(cleaned)
    for key,date in allowed.items():
        assert cleaned.weights.loc[cleaned.weights.index<date,key].eq(0).all()
        assert cleaned.pnl.loc[cleaned.pnl.index<date,key].eq(0).all()
        bought=cleaned.trades.loc[cleaned.trades.asset.eq(key)&cleaned.trades.side.eq("buy")]
        assert (bought.date>date).all()
    original_allowed={"fcf":allowed["fcf"]}
    with apply_policy(lambda n,r,p:PublishedIndexRotation(r,phases,allowed["fcf"],original_allowed),KEYS):
        same=account()
    saved=pd.read_csv(HERE/"nav.csv").query("study == 'extended_first'").GS
    np.testing.assert_allclose(saved,same.ledger.nav.to_numpy(),rtol=1e-12,atol=1e-7)
    v=engine.economic("V12",benchmark,None,start,SLIP)
    rows=[]
    for case,a in (("fcf_only",same),("also_remove_backfilled_bond_and_dividend",cleaned),("V12",v)):
        rows.append(dict(case=case,start=start,end=END,window="full",**brief(a)))
        p=window(a,start,"2013-04-11")
        rows.append(dict(case=case,start=start,end=p["end"],window="extension",
                         **{k:p[k] for k in brief(a)}))
    write(pd.DataFrame(rows),"publication_control.csv")
    write(cleaned.signals,"publication_control_targets.csv")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
