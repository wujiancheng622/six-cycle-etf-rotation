"""Executable checks behind the manual 22-rule audit."""
import json

import numpy as np
import pandas as pd

from early_inputs import loan_data, macro_inputs, pmi_data, ts
from regimes import STAGE_MAP, state_path
from replay_long import (ACT_START, ASSETS_BY_STAGE, BASE, CAPITAL, DATA, END, HERE, Inputs,
                         KEYS, MAIN, OUT, Rotation, apply_policy, engine, load_account,
                         log_attribution, provider, run, save_csv, stats, verify_account, verify_frozen)


def hash_checks():
    files = {}
    for root in (BASE, HERE):
        f = json.loads((root/"input_hashes.json").read_text(encoding="utf-8"))
        for path, digest in f.items():
            target = HERE.parent/path
            assert provider.sha(target)==digest, path
            files[path] = digest
    return len(files)


def pbc_pmi_suffix(inputs, dates):
    check = []
    credit = loan_data()
    pmi = pmi_data()
    hist = inputs.hist
    for day in dates:
        known = hist.index[hist.index.get_loc(day)-1]
        suffix_ = inputs.macro.copy()
        suffix_.loc[pd.Timestamp(day):,"stage"] = 1
        quotes_ = hist.copy()
        quotes_.loc[day:] = 0.55
        before = Rotation(MAIN,hist,inputs.macro).targets(day,known,0,CAPITAL)[0]
        after = Rotation(MAIN,quotes_,suffix_).targets(day,known,0,CAPITAL)[0]
        np.testing.assert_array_equal(before,after)
        c, p = credit.copy(),pmi.copy()
        c.loc[c.released.ge(day),"credit_change"] = -c.loc[c.released.ge(day),"credit_change"]
        p.loc[p.available.ge(day),"direction"] = -p.loc[p.available.ge(day),"direction"]
        rebuilt = macro_inputs(inputs.macro.index,c,p)
        pd.testing.assert_frame_equal(rebuilt.loc[:known],inputs.macro.loc[:known])
        check.append(dict(day=day, known=known, available_only=True))
    return check


def exact_risk(a, hist):
    excluded = hist.attrs["excluded_risk_dates"]
    errors, theory, forecast = [], [], []
    for r in a.signals.itertuples():
        keys = ASSETS_BY_STAGE[r.phase]
        exc = sorted({d for k in keys for d in excluded[k]})
        trailing = hist.loc[:r.known,list(keys)].drop(index=exc,errors="ignore").tail(63)
        assert len(trailing)==63 and trailing.index[0]==r.first_return and trailing.index[-1]==r.last_return
        cov = trailing.cov().to_numpy()
        w = np.array([getattr(r,k) for k in keys])
        rc = w*(cov@w)
        errors.append(float(np.max(np.abs(rc/rc.sum()-1/len(keys)))))
        forecast.append(float(np.sqrt(rc.sum()*252)))
        if len(keys)==2:
            v = np.sqrt(np.diag(cov))
            theory.append(float(np.max(np.abs(w-(1/v)/(1/v).sum()))))
    assert max(errors) < 1e-8 and max(theory) < 1e-8
    return dict(targets=len(errors), max_equal_risk_share_error=max(errors),
                two_asset_closed_form_checks=len(theory), max_closed_form_error=max(theory),
                ex_ante_portfolio_vol_min=min(forecast),ex_ante_portfolio_vol_max=max(forecast))


def early_raw_quotes(hist):
    checked = {}
    for code, start, since, key in (("159915.SZ","20111201","2013-01-04","growth"),
                                    ("518880.SH","20130101","2013-07-30","gold")):
        f = ts("fund_daily",code,start_date=start,end_date="20140115").sort_values("trade_date")
        assert (f.vol>0).all()
        np.testing.assert_allclose(f.pre_close.iloc[1:],f.close.iloc[:-1],atol=.0011,rtol=0)
        f.index = pd.to_datetime(f.trade_date,format="%Y%m%d").dt.strftime("%Y-%m-%d")
        x = f.close.pct_change(fill_method=None).loc[since:]
        assert hist.loc[since:x.index[-1]].index.equals(x.index)
        np.testing.assert_allclose(x,hist.loc[x.index,key],atol=0,rtol=0)
        checked[code] = len(x)
    return checked


def actual_capacity(inputs, a, start):
    q, adv, up, down = inputs.gs_quotes
    trades = a.trades
    ratio = trades.shares/trades.adv
    for t in trades.itertuples():
        assert t.price==q.panels["open"].at[t.date,t.asset]
        assert q.panels["volume"].at[t.date,t.asset]>0
        assert (t.price<up.at[t.date,t.asset] if t.side=="buy" else t.price>down.at[t.date,t.asset])
    n = trades.groupby(["date","asset"]).side.nunique()
    assert n.max()==1
    return dict(start=start,trades=len(trades),maximum_adv_fraction=float(ratio.max()),
                count_at_least_99pct_of_cap=int(ratio.ge(.0099).sum()),
                largest_trade_cny=float(trades.notional.max()),
                minimum_cash_cny=float(a.ledger.cash.min()),
                total_cost_cny=float(a.ledger.cost.sum()),
                max_end_day_asset_weight=float(a.weights.drop(columns="cash").max().max()))


class PriorToCash(Rotation):
    """Audit sensitivity: leave the affected sleeve uninvested before a date."""
    def __init__(self, name, returns, state, dated_exclusions):
        super().__init__(name, returns, state)
        self.dated_exclusions = dated_exclusions

    def targets(self, day, known, i, nav):
        a = super().targets(day, known, i, nav)
        unavailable = [KEYS.index(k) for k,d in self.dated_exclusions.items() if day < d]
        return tuple(None if w is None else np.where(np.isin(np.arange(len(w)), unavailable),0.,w)
                     for w in a)


def proxy_stresses(inputs, start):
    out = []
    for name, cutoff in (
        ("fcf_before_methodology_cash", {"fcf":"2024-08-15"}),
        ("all_prelisting_returns_cash",inputs.coverage["actual_return_from"]),
    ):
        with apply_policy(lambda n,r,_:PriorToCash(MAIN,r,inputs.macro,cutoff),KEYS):
            a = engine.economic(name,inputs.hist,None,start,.0009)
        verify_account(a,name,"amount")
        out.append(dict(test=name, **stats(a)))
    save_csv(pd.DataFrame(out),"proxy_stress.csv")
    return {x["test"]:{k:x[k] for k in ("cagr","vol","mdd")} for x in out}


def checks():
    verify_frozen()
    hashes = hash_checks()
    inputs = Inputs.create()
    early = inputs.earliest()
    assert early == "2013-04-12"
    probes = pbc_pmi_suffix(inputs,[early,"2014-01-02","2016-01-04","2020-03-02","2021-03-01",
                                     "2023-05-04",ACT_START,"2026-11-02"])
    combos = [(1,1,-1),(1,1,1),(-1,1,1),(-1,-1,1),(1,-1,1),(-1,-1,-1),(1,-1,-1),(-1,1,-1),(0,1,1)]
    c = [pd.Series(x) for x in zip(*combos)]
    assert state_path(*c).tolist() == [1,2,3,4,4,5,6,6,6] and len(STAGE_MAP)==6
    replay_hashes, total_accounts = {},0
    for d in sorted(OUT.glob("*/*/*/meta.json")):
        mode, start, name_bp = d.relative_to(OUT).parts[:3]
        name, bps = name_bp.rsplit("_",1)
        a = load_account(mode,start,name,int(bps.removesuffix("bp")))
        m = json.loads(d.read_text(encoding="utf-8"))
        assert m["ledger_sha256"]==provider.sha(d.parent/"ledger.parquet")
        assert m["verification"]==verify_account(a,name,mode)
        log_attribution(a)
        total_accounts += 1
    for mode,start in (("amount",early),("actual",ACT_START)):
        for name in (MAIN,"V12"):
            original = load_account(mode,start,name)
            rerun = run(inputs,mode,start,name)
            for k in ("ledger","trades","signals","weights","pnl"):
                pd.testing.assert_frame_equal(getattr(original,k),getattr(rerun,k),check_exact=True)
                repfile = HERE / "tables" / f"_verify_{mode}_{name}_{k}.parquet"
                getattr(rerun,k).to_parquet(repfile)
                assert provider.sha(repfile)==provider.sha(OUT/mode/start/f"{name}_9bp"/f"{k}.parquet")
                repfile.unlink()
            replay_hashes[f"{mode}/{name}"] = provider.sha(OUT/mode/start/f"{name}_9bp/ledger.parquet")
    g = load_account("amount",early,MAIN)
    ag = load_account("actual",ACT_START,MAIN)
    af = inputs.gs_quotes[0].returns
    qmt = json.loads((BASE/"verification/checks.json").read_text(encoding="utf-8"))["qmt_unadjusted"]
    assert all(x["matched"]==248 and x["only_qmt"]==x["only_tushare"]==0 and
               max(x["max_difference"].values())<2e-15 for x in qmt.values())
    proxy = {}
    for k, day in inputs.coverage["actual_return_from"].items():
        proxy[k] = dict(first_actual_return=day,
                        first_historical_model_buy=g.trades[g.trades.asset.eq(k)&g.trades.side.eq("buy")].date.min(),
                        first_actual_model_buy=ag.trades[ag.trades.asset.eq(k)&ag.trades.side.eq("buy")].date.min())
        assert proxy[k]["first_actual_model_buy"]>=day
    later = run(inputs,"actual","2026-05-07",MAIN,0.)
    later_v = load_account("actual","2026-05-07","V12")
    stage = inputs.macro.loc[g.ledger.index[0]:END]
    valid = [tuple(a) in STAGE_MAP for a in stage[["money","credit","growth"]].itertuples(index=False,name=None)]
    provider.write(HERE/"audit_checks.json",
                   dict(hashes_verified=hashes,accounts_reconciled=total_accounts,
                        repeated_main_four_sha256=replay_hashes, future_perturbation=probes,
                        state_truth_table_cases=len(combos),quote_crosscheck_from_base=qmt,
                        early_etf_session_reconciliation=early_raw_quotes(inputs.hist),
                        frozen_covariance_amount=exact_risk(g,inputs.hist),
                        frozen_covariance_actual=exact_risk(ag,af),listing_check=proxy,
                        proxy_stress_after_source_audit=proxy_stresses(inputs,early),
                        share_days_noncanonical=float(1-np.mean(valid)),
                        actual_execution=actual_capacity(inputs,ag,ACT_START),
                        later_actual_GS_at_zero_slippage_return=stats(later)["total"],
                        later_actual_V12_at_9bp_return=stats(later_v)["total"]))
    print("audit: checked",total_accounts,"accounts,",hashes,"input hashes,",len(probes),"cutoffs,",
          len(replay_hashes),"deterministic account hashes",flush=True)


if __name__=="__main__":
    checks()
