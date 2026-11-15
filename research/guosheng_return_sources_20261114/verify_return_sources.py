"""Recalculate saved accounts and exact attribution without rerunning selection."""

import ast
import hashlib
import json

import numpy as np
import pandas as pd

import return_sources as a


def main():
    results = pd.read_csv(a.OUT / "comparison.csv")
    assert results.case.is_unique
    rows, metrics = [], 0
    for case in results.itertuples():
        folder = a.OUT / "accounts" / case.case
        meta = json.loads((folder / "meta.json").read_text())
        ledger = pd.read_parquet(folder / "ledger.parquet")
        weights = pd.read_parquet(folder / "weights.parquet")
        pnl = pd.read_parquet(folder / "pnl.parquet")
        trades = pd.read_parquet(folder / "trades.parquet")
        signals = pd.read_parquet(folder / "signals.parquet")
        old = ledger.nav.shift(1, fill_value=meta["capital"])
        error = (old + pnl.sum(axis=1) - ledger.cost - ledger.nav).abs().max()
        assert error < 1e-6 and ledger.cash.min() >= -1e-7
        assert trades.known.lt(trades.date).all() and signals.known.lt(signals.date).all()
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-10)
        np.testing.assert_allclose(trades.groupby("date").cost.sum().reindex(ledger.index, fill_value=0), ledger.cost, atol=1e-8)
        path = np.r_[meta["capital"], ledger.nav]
        ret = np.diff(path)/path[:-1]
        years = (pd.Timestamp(ledger.index[-1])-pd.Timestamp(meta["base_date"])).days/365.25
        actual = [(path[-1]/path[0])**(1/years)-1, np.std(ret, ddof=1)*np.sqrt(252),
                  np.max(1-path/np.maximum.accumulate(path)), path[-1]/path[0]-1,
                  ledger.cost.sum(), (ledger.cash/ledger.nav).mean()]
        expected = [case.cagr, case.vol, case.mdd, case.total, case.fees, case.mean_cash]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)
        rows.append(dict(case=case.case, max_nav_error=float(error), metric_error=float(np.max(np.abs(np.asarray(actual)-expected)))))
        metrics += len(actual)
    d = pd.read_csv(a.OUT / "exact_decomposition.csv")
    residual = d.mean_exposure_arithmetic+d.dynamic_allocation_arithmetic-d.cost_arithmetic-d.arithmetic_to_log_penalty-d.annual_log_net
    assert residual.abs().max() < 1e-12
    np.testing.assert_allclose(np.expm1(d.annual_log_net), d.cagr, atol=1e-12)
    adj = pd.read_csv(a.OUT / "mean_adjustments.csv")
    assert (adj.achieved <= adj.original_cagr+1e-11).all()
    np.testing.assert_allclose(adj.achieved, np.minimum(adj.original_cagr, adj.target), atol=2e-12)
    checks = pd.read_csv(a.OUT / "timing_preservation_checks.csv")
    assert checks.target_weight_max_difference.max() < 1e-10 and checks.full_covariance_max_difference.max() < 1e-14
    fixed = pd.read_csv(a.OUT / "fixed_weight_controls.csv")
    assert fixed.timing_covariance_sum_max_difference.max() < 1e-12
    old = pd.read_csv(a.ROOT / "guosheng_final_decision_20261114/results/premium_drift_comparison.csv")
    for start in ("2014-01-02", "2020-07-30"):
        for strategy in ("GS", "V12"):
            for current, prior in (("market", "market"), ("nav_original", "nav")):
                x = results.loc[results.case.eq(f"{strategy}_{current}_{start}"), ["cagr", "vol", "mdd"]]
                y = old.loc[old.case.eq(f"{strategy}_{prior}_{start}"), ["cagr", "vol", "mdd"]]
                np.testing.assert_array_equal(x.to_numpy(), y.to_numpy())
    for path in a.HERE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    a.write(pd.DataFrame(rows), "account_validation.csv")
    inputs = [a.HERE / "protocol.json", *a.HERE.glob("*.py"), *a.OUT.glob("*.parquet"),
              a.ROOT / "guosheng_final_decision_20261114/study.py", a.ROOT / "guosheng_v12_return_dependence_20261113/normalization.py"]
    receipt = dict(accounts=len(results), independent_metrics=metrics, attribution_rows=len(d),
                   maximum_attribution_error=float(residual.abs().max()),
                   maximum_target_weight_change=float(checks.target_weight_max_difference.max()),
                   maximum_frozen_timing_change=float(fixed.timing_covariance_sum_max_difference.max()),
                   historical_baseline_reproduction="8 original metrics rows match exactly",
                   synthetic_scope="NAV and mean-adjusted worlds are retrospective attribution, not independent future evidence or historically executable price alternatives",
                   sha256={str(p.relative_to(a.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (a.OUT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!='sha256'}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
