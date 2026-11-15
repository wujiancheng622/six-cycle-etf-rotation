"""Paired historical uncertainty, persistence and contribution checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import study as s


def load_account(name):
    folder = s.OUT / "accounts" / name
    meta = json.loads((folder / "meta.json").read_text())
    return SimpleNamespace(**meta, **{key: pd.read_parquet(folder / f"{key}.parquet")
                                     for key in ("ledger", "weights", "pnl", "signals", "trades")})


def monthly_logs(account):
    nav = account.ledger.nav
    daily = np.log(nav / nav.shift(1, fill_value=account.capital))
    daily.index = pd.to_datetime(daily.index)
    return daily.groupby(daily.index.to_period("M")).sum().loc[:"2026-10"]


def main():
    base = load_account("V12_2014-01-02_9bp")
    benchmarks = monthly_logs(base)
    rolling, boot, contributions, behavior = [], [], [], []
    candidates = ["monthly_2014-01-02_9bp", "month_and_switch_2014-01-02_9bp",
                  "value_substitute_2014-01-02_9bp", "dividend_substitute_2014-01-02_9bp",
                  "all_publications_cash_2014-01-02_9bp", "all_listing_cash_2014-01-02_9bp"]
    for name in candidates:
        a = load_account(name)
        m = monthly_logs(a)
        assert m.index.equals(benchmarks.index)
        for width in (12, 36, 60):
            gs = np.expm1(m.rolling(width).sum().dropna() * 12 / width)
            v12 = np.expm1(benchmarks.rolling(width).sum().dropna() * 12 / width)
            for date in gs.index:
                rolling.append(dict(case=name, months=width, end=str(date), cagr=gs.at[date], v12_cagr=v12.at[date],
                                    excess=gs.at[date] - v12.at[date]))
        active = (m - benchmarks).cumsum()
        underwater = active < active.cummax().clip(lower=0)
        longest, streak = 0, 0
        for down in underwater:
            streak = streak + 1 if down else 0
            longest = max(longest, streak)
        signals = a.signals
        switches = signals.selected.ne(signals.selected.shift()).sum()
        behavior.append(dict(case=name, monthly_observations=len(m), target_decisions=len(signals),
                             basket_changes=int(switches - 1), longest_relative_underwater_months=longest,
                             negative_months=int(m.lt(0).sum()), beat_v12_months=int(m.gt(benchmarks).sum())))
        paired = np.column_stack([m, benchmarks])
        for block in (6, 12):
            rng = np.random.default_rng(20261114 + block)
            starting = rng.integers(0, len(m), size=(2000, int(np.ceil(len(m) / block))))
            index = (starting[:, :, None] + np.arange(block)[None, None, :]) % len(m)
            samples = paired[index.reshape(2000, -1)[:, :len(m)]]
            annual = np.expm1(samples.sum(axis=1) * 12 / len(m))
            gap = annual[:, 0] - annual[:, 1]
            boot.append(dict(case=name, block_months=block, paths=2000, first_month=str(m.index[0]),
                             last_month=str(m.index[-1]), gap_q05=np.quantile(gap, .05), gap_median=np.median(gap),
                             gap_q95=np.quantile(gap, .95), nonpositive_gap_fraction=np.mean(gap <= 0)))
        before = a.ledger.nav.shift(1, fill_value=a.capital)
        ret = a.ledger.nav / before - 1
        raw = a.pnl.div(before, axis=0).assign(fee=-a.ledger.cost / before)
        factor = np.divide(np.log1p(ret), ret, out=np.ones(len(ret)), where=ret.ne(0))
        linked = raw.mul(factor, axis=0)
        np.testing.assert_allclose(linked.sum(axis=1), np.log1p(ret), atol=3e-15)
        for period, (lo, hi) in s.PERIODS.items():
            subset = linked.loc[lo:hi]
            for asset, amount in subset.sum().items():
                contributions.append(dict(case=name, period=period, asset=asset, log_contribution=amount,
                                          log_share=amount / subset.to_numpy().sum()))
    s.write(pd.DataFrame(rolling), "rolling_comparison.csv")
    s.write(pd.DataFrame(boot), "paired_block_bootstrap.csv")
    s.write(pd.DataFrame(contributions), "contribution_checks.csv")
    s.write(pd.DataFrame(behavior), "decision_and_underperformance.csv")
    print(pd.DataFrame(behavior).to_string(index=False), flush=True)
    print(pd.DataFrame(boot).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
