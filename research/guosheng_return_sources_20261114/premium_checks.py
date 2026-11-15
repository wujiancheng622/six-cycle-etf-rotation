"""Hold historical exposures fixed when changing only the asset valuation basis."""

from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

import return_sources as a


def account(name):
    folder = a.OUT / "accounts" / name
    meta = json.loads((folder / "meta.json").read_text())
    return SimpleNamespace(**{k: pd.read_parquet(folder / f"{k}.parquet")
                              for k in ("ledger", "weights", "pnl", "signals", "trades")}, **meta)


def main():
    rows, pieces, shocks, asset_rows = [], [], [], []
    premiums = pd.read_csv(a.OUT / "premium_observations.csv")
    for strategy in ("GS", "V12"):
        market = pd.read_parquet(a.OUT / f"{strategy}_market.parquet")
        nav = pd.read_parquet(a.OUT / ("GS_nav.parquet" if strategy == "GS" else "V12_nav_shared.parquet"))
        for start in a.CONF["starts"]:
            acc = account(f"{strategy}_market_{start}")
            dates = acc.ledger.index
            w = acc.weights[list(market.columns)].shift(1, fill_value=0)
            prior_nav = acc.ledger.nav.shift(1, fill_value=a.s.CAPITAL)
            costs = acc.ledger.cost / prior_nav
            original_r = (w * market.loc[dates]).sum(axis=1) - costs
            nav_r = (w * nav.loc[dates]).sum(axis=1) - costs
            np.testing.assert_allclose(original_r, acc.ledger.nav/prior_nav - 1, atol=2e-14)
            years = a.n.years(acc.base_date, dates[-1])
            for mode, ret in (("market", original_r), ("nav_fixed_historical_weights", nav_r)):
                path = np.r_[1., np.exp(np.log1p(ret).cumsum())]
                rows.append(dict(strategy=strategy, start=start, end=dates[-1], mode=mode,
                                 cagr=path[-1]**(1/years)-1, vol=ret.std(ddof=1)*np.sqrt(252),
                                 mdd=(1-path/np.maximum.accumulate(path)).max()))
            difference = original_r - nav_r
            multiplier = pd.Series(1/(1+nav_r), index=dates)
            unequal = difference.abs() > 1e-13
            multiplier.loc[unequal] = ((np.log1p(original_r)-np.log1p(nav_r))/difference).loc[unequal]
            contributions = (w * (market.loc[dates]-nav.loc[dates])).mul(multiplier, axis=0).sum()/years
            np.testing.assert_allclose(contributions.sum(), (np.log1p(original_r)-np.log1p(nav_r)).sum()/years, atol=1e-13)
            for key in market:
                pieces.append(dict(strategy=strategy, start=start, asset=key,
                                   annual_log_premium_mark_contribution=contributions[key]))
                r = nav.loc[dates, key]
                asset_rows.append(dict(strategy=strategy, start=start, end=dates[-1], asset=key,
                                       buy_hold_nav_cagr=np.expm1(np.log1p(r).sum()/years)))
            if start == "2014-01-02":
                observations = premiums.loc[premiums.strategy.eq(strategy) & premiums.interval.eq("2014 onward")]
                for ob in observations.itertuples():
                    weight = float(acc.weights[ob.asset].iloc[-1])
                    for target_label, target in (("zero", 0.), ("historical_median", ob.premium_median)):
                        change = (1+target)/(1+ob.premium_last)-1
                        shocks.append(dict(strategy=strategy, asset=ob.asset, asof=ob.premium_last_date,
                                           weight=weight, premium=ob.premium_last, target_premium=target, target_label=target_label,
                                           asset_change=change, portfolio_once_change=weight*change))
    a.write(pd.DataFrame(rows), "fixed_holdings_premium.csv")
    a.write(pd.DataFrame(pieces), "premium_log_attribution.csv")
    a.write(pd.DataFrame(shocks), "premium_once_shocks.csv")
    a.write(pd.DataFrame(asset_rows), "asset_buy_hold_returns.csv")
    print(pd.DataFrame(rows).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
