"""Measure the false-positive range of the explicitly enumerated 120 basket maps."""

from itertools import permutations
import json

import numpy as np
import pandas as pd

import study as s


ORDERS = np.array(list(permutations(range(5))), dtype=int)
SEED = 20261114
PATHS = 60
BLOCK = 20


def fee(values, slip):
    return np.where(values > 1e-9, np.maximum(5., values * .00015) + values * slip, 0.)


def all_maps(history, state, start=s.START, slip=.0009):
    frame = history.loc[start:]
    indices = history.index.get_indexer(frame.index)
    keys = list(history.columns)
    slices = [[keys.index(k) for k in s.ASSETS[p]] for p in range(1, 6)]
    exclusions = history.attrs.get("excluded_risk_dates", {})
    values = np.zeros((120, len(keys)))
    cash = np.full(120, float(s.CAPITAL))
    pending = np.zeros_like(values)
    nav = np.empty((len(frame), 120))
    last_trade = -2
    for i, j in enumerate(indices):
        day, known = history.index[j], history.index[j - 1]
        refresh = i == 0 or day[:7] != known[:7]
        before = cash + values.sum(axis=1)
        if refresh:
            weights = np.zeros((5, len(keys)))
            for basket, cols in enumerate(slices):
                assets = [keys[k] for k in cols]
                removed = {d for k in assets for d in exclusions.get(k, ())}
                train = history.iloc[:j][assets].drop(index=sorted(removed), errors="ignore").tail(63)
                assert len(train) == 63
                weights[basket, cols] = s.risk_parity(train.cov().to_numpy())
            phase = min(int(state.at[pd.Timestamp(known), "stage"]), 5) - 1
            pending = weights[ORDERS[:, phase]]
            last_trade = i
        delta = pending * before[:, None] - values
        sells = -np.minimum(delta, 0) if refresh else np.zeros_like(values)
        buys = np.maximum(delta, 0) if refresh or i == last_trade + 1 else np.zeros_like(values)
        returns = history.iloc[j].to_numpy()
        sales = sells * (1 + returns)
        requested = np.where(sales == 0, buys * (1 + returns), 0.)
        values *= 1 + returns
        sales = np.minimum(sales, values)
        sell_fee = fee(sales, slip)
        insufficient = cash + (sales - sell_fee).sum(axis=1) < -1e-8
        if insufficient.any():
            dust = insufficient[:, None] & (sales < sell_fee)
            sales[dust] = 0.
            sell_fee = fee(sales, slip)
        budget = np.minimum(cash, cash + (sales - sell_fee).sum(axis=1))
        available = requested.sum(axis=1) + fee(requested, slip).sum(axis=1) <= budget
        left, right = np.zeros(120), np.ones(120)
        if not available.all():
            for _ in range(44):
                middle = (left + right) / 2
                proposal = requested * middle[:, None]
                permitted = proposal.sum(axis=1) + fee(proposal, slip).sum(axis=1) <= budget
                left = np.where(permitted, middle, left)
                right = np.where(permitted, right, middle)
        bought = requested * np.where(available, 1., left)[:, None]
        bought[bought <= 1e-9] = 0.
        buy_fee = fee(bought, slip)
        cash += (sales - bought - sell_fee - buy_fee).sum(axis=1)
        values += bought - sales
        assert cash.min() > -1e-6
        nav[i] = cash + values.sum(axis=1)
    years = (pd.Timestamp(frame.index[-1]) - pd.Timestamp(history.index[indices[0] - 1])).days / 365.25
    returns = np.r_[np.full((1, 120), s.CAPITAL), nav]
    dd = (1 - returns / np.maximum.accumulate(returns, axis=0)).max(axis=0)
    return (nav[-1] / s.CAPITAL) ** (1 / years) - 1, dd, nav


def main():
    data = s.load()
    real, drawdown, nav = all_maps(data.views.hist, data.state)
    saved = pd.read_csv(s.OUT / "mapping_negative_controls.csv").query("period == 'full'").set_index("case")
    labels = ["mapping_" + "".join(str(i + 1) for i in order) for order in ORDERS]
    np.testing.assert_allclose(real, saved.loc[labels, "cagr"], atol=5e-12, rtol=0)
    np.testing.assert_allclose(drawdown, saved.loc[labels, "mdd"], atol=5e-12, rtol=0)
    rows = [dict(path="observed", best_cagr=float(real.max()), original_map_cagr=float(real[0]),
                 median_cagr=float(np.median(real)))]
    rng = np.random.default_rng(SEED)
    matrix = data.views.hist.to_numpy()
    for path in range(PATHS):
        starts = rng.integers(0, len(matrix), size=int(np.ceil(len(matrix) / BLOCK)))
        ix = ((starts[:, None] + np.arange(BLOCK)) % len(matrix)).reshape(-1)[:len(matrix)]
        synthetic = pd.DataFrame(matrix[ix], index=data.views.hist.index, columns=data.views.hist.columns)
        synthetic.attrs.update(data.views.hist.attrs)
        cagr, _, _ = all_maps(synthetic, data.state)
        rows.append(dict(path=str(path), best_cagr=float(cagr.max()), original_map_cagr=float(cagr[0]),
                         median_cagr=float(np.median(cagr))))
        if (path + 1) % 10 == 0:
            print(f"Synthetic full 120-map search {path + 1}/{PATHS}", flush=True)
    table = pd.DataFrame(rows)
    s.write(table, "local_search_noise.csv")
    null = table.iloc[1:]
    summary = dict(seed=SEED, paths=PATHS, maps_per_path=120, source_block_days=BLOCK,
                   original_cagr=float(real[0]), real_best_cagr=float(real.max()),
                   noise_best_median=float(null.best_cagr.median()), noise_best_q95=float(null.best_cagr.quantile(.95)),
                   noise_best_max=float(null.best_cagr.max()),
                   best_exceeds_observed_fraction=float(null.best_cagr.ge(real[0]).mean()),
                   scope="Only this declared 120-basket-map search, preserving sampled joint returns and recomputing risk, costs and fills. Author-wide selection remains unknown.")
    (s.OUT / "local_search_noise_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
