"""Replay the FCF cutoff without letting unavailable data set portfolio weights."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "guosheng_six_regime_long_20261112"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT))
from diagnostics import window
from replay_long import ASSETS_BY_STAGE, CAPITAL, END, KEYS, MAIN, Rotation, SLIP
from replay_long import apply_policy, engine, load_account, log_attribution, stats
from run_study import risk_parity
from verify_long import PriorToCash

RULE_DATE = "2024-08-15"
STARTS = ("2013-04-12", "2014-01-02")
ASSET = "fcf"
LOOKBACK = 63


class AvailableRotation(Rotation):
    def __init__(self, returns, phases, first_return, reserve_cash):
        super().__init__(MAIN, returns, phases)
        self.first_return = first_return
        self.reserve_cash = reserve_cash
        self.training = returns.copy()
        if first_return is None:
            self.training[ASSET] = np.nan
        else:
            self.training.loc[self.training.index < first_return, ASSET] = np.nan

    def targets(self, day, known, i, nav):
        macro = self.states.loc[pd.Timestamp(known)]
        if pd.isna(macro.stage):
            return None, None
        if self.pending is None or day[:7] != known[:7]:
            phase = int(macro.stage)
            original = ASSETS_BY_STAGE[phase]
            available = list(original)
            if ASSET in original and self.first_return is not None:
                valid = trailing(self.training, self.excluded_risk_dates, known, original)
                if len(valid) < LOOKBACK:
                    available.remove(ASSET)
            elif ASSET in original:
                available.remove(ASSET)

            if available == list(original):
                sell, buy = super().targets(day, known, i, nav)
                assert sell is not None and buy is not None
                self.receipts[-1].update(requested="/".join(original), fcf_eligible=ASSET in available,
                                         cash_target=0.)
                return sell, buy

            train = trailing(self.training, self.excluded_risk_dates, known, available)
            assert len(train) == LOOKBACK
            if len(available) == 1:
                fractions = np.ones(1)
            else:
                covariance = train.cov().to_numpy()
                assert np.linalg.eigvalsh(covariance).min() > 0.
                fractions = risk_parity(covariance)
                components = fractions * (covariance @ fractions)
                np.testing.assert_allclose(components/components.sum(), 1./len(fractions), atol=1e-8)
            cash_target = 1/len(original) if self.reserve_cash else 0.
            weights = (pd.Series(fractions*(1-cash_target), index=available)
                       .reindex(KEYS, fill_value=0.).to_numpy())
            self.previous_assets = tuple(available)
            self.last_rebalance = i
            self.pending = weights
            self.receipts.append(dict(date=day, known=known, phase=phase, month_end=True,
                                      selected="/".join(available), requested="/".join(original),
                                      first_return=train.index[0], last_return=train.index[-1],
                                      credit_date=macro.credit_published,
                                      growth_date=macro.growth_available.date().isoformat(),
                                      fcf_eligible=False, cash_target=cash_target,
                                      **dict(zip(KEYS,weights))))
            return weights, weights
        if i == self.last_rebalance+1:
            return None, self.pending
        return None, None


def trailing(history, exclusions, known, assets):
    skipped = {day for key in assets for day in exclusions.get(key,())}
    return (history.loc[:known,list(assets)].drop(index=sorted(skipped),errors="ignore")
            .dropna().tail(LOOKBACK))


def assert_clean(account, first_return):
    signals = account.signals
    picked = signals[ASSET].gt(0)
    blocked = signals.requested.str.split("/").map(lambda x:ASSET in x)&~picked
    assert signals.loc[blocked,ASSET].eq(0).all()
    assert signals.loc[blocked,"selected"].str.split("/").map(lambda x:ASSET not in x).all()
    if first_return is None:
        assert not picked.any() and account.weights[ASSET].eq(0).all()
        assert account.pnl[ASSET].eq(0).all()
    else:
        assert (signals.loc[picked,"first_return"] >= first_return).all()
        assert account.weights.loc[account.weights.index < first_return,ASSET].eq(0).all()
        assert account.pnl.loc[account.pnl.index < first_return,ASSET].eq(0).all()
    assert (signals.known < signals.date).all()
    assert (signals.last_return <= signals.known).all()
    assert (signals.credit_date <= signals.known).all()
    assert (signals.growth_date <= signals.known).all()
    np.testing.assert_allclose(signals[list(KEYS)].sum(axis=1)+signals.cash_target,1.,atol=1e-10)
    return dict(targets=len(signals),frozen_out_targets=int(blocked.sum()),
                first_fcf_target=signals.loc[picked,"date"].min() if picked.any() else None)


def run_variant(returns, phases, start, first_return, reserve_cash, perturb=False):
    prices = returns.copy()
    prehistory = prices.index < first_return if first_return is not None else np.ones(len(prices),dtype=bool)
    if perturb:
        # Large changes to FCF data outside the permitted history must be inert.
        prices.loc[prehistory,ASSET] = np.resize([.30,-.30],int(np.sum(prehistory)))
    with apply_policy(lambda n,r,p:AvailableRotation(r,phases,first_return,reserve_cash),KEYS):
        account = engine.economic("GS_available_FCF",prices,None,start,SLIP)
    assert_clean(account, first_return)
    log_attribution(account)
    return account


def save(frame,name):
    frame.to_csv(HERE/name,index=False,encoding="utf-8-sig")


def main():
    protocol = HERE/"protocol.json"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        HERE/"protocol.sha256").read_text(encoding="ascii").strip()
    path = PARENT/"data/returns_gs.parquet"
    returns = pd.read_parquet(path)
    phases = pd.read_parquet(PARENT/"data/macro.parquet")
    coverage = json.loads((PARENT/"coverage.json").read_text(encoding="utf-8"))
    first_index = returns.index[returns.index.searchsorted(RULE_DATE,side="right")]
    first_etf = coverage["actual_return_from"][ASSET]
    versions = {
        "fcf_from_published_index": (first_index,True),
        "fcf_from_published_index_reallocate": (first_index,False),
        "fcf_from_etf": (first_etf,True),
        "fcf_excluded_throughout": (None,True),
    }
    summary, segments, checks, daily, signals = [], [], {}, [], []
    for start in STARTS:
        original = load_account("amount",start,MAIN)
        v12 = load_account("amount",start,"V12")
        cases = {"original_proxy":original, "V12":v12}
        for name,(first,keep_cash) in versions.items():
            case = run_variant(returns,phases,start,first,keep_cash)
            cases[name] = case
            if start == STARTS[0]:
                recheck = run_variant(returns,phases,start,first,keep_cash,perturb=True)
                for key in ("ledger","weights","pnl","trades","signals"):
                    pd.testing.assert_frame_equal(getattr(case,key),getattr(recheck,key),check_exact=True)
                checks[name] = assert_clean(case,first) | {
                    "earliest_allowed_rebalance_day_for_fcf":(
                        returns.index[returns.index.get_loc(first)+LOOKBACK] if first else None),
                    "passed_early_fcf_perturbation":True,
                }
        with apply_policy(lambda n,r,p:PriorToCash(MAIN,r,phases,{ASSET:RULE_DATE}),KEYS):
            cases["old_mask_after_risk"] = engine.economic(MAIN,returns,None,start,SLIP)
        if start == STARTS[0]:
            expected = pd.read_csv(PARENT/"tables/proxy_stress.csv").iloc[0]
            for metric in ("cagr","vol","mdd","final_equity"):
                np.testing.assert_allclose(stats(cases["old_mask_after_risk"])[metric],expected[metric],rtol=1e-12)
        for name,acct in cases.items():
            assert original.ledger.index.equals(acct.ledger.index) and acct.base_date == original.base_date
            summary.append(dict(case=name,**stats(acct)))
            for label,a,b in (
                ("before_index",start,"2024-08-14"),
                ("since_index",RULE_DATE,END),
                ("2014-2016","2014-01-01","2016-12-31"),
                ("2017-2019","2017-01-01","2019-12-31"),
                ("2020-2022","2020-01-01","2022-12-31"),
                ("2023-2024_pre_index","2023-01-01","2024-08-14"),
            ):
                segments.append(dict(case=name,account_start=start,window=label,**window(acct,a,b)))
            if start == STARTS[0]:
                daily.append(acct.ledger.nav.rename(name))
                if name in versions:
                    signals.append(acct.signals.assign(case=name))
        print(start,[(n,tuple(round(100*stats(a)[k],3) for k in ("cagr","vol","mdd"))) for n,a in cases.items()],flush=True)
    save(pd.DataFrame(summary),"comparison.csv")
    save(pd.DataFrame(segments),"periods.csv")
    save(pd.concat(daily,axis=1).rename_axis("date").reset_index(),"nav.csv")
    save(pd.concat(signals,ignore_index=True),"targets.csv")
    (HERE/"verification.json").write_text(json.dumps(dict(
        index_clean_start=first_index, etf_clean_start=first_etf,
        old_mask_reproduced=True, controls=checks,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        returns_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
