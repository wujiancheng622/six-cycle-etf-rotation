"""Symmetric history eligibility and long-run return-source comparisons."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

import study as s


class EligibleV12:
    def __init__(self, actual):
        self.actual = actual
        self.receipts = []

    def targets(self, day, known, i, nav):
        eligible = [k for k in s.V12_KEYS if self.actual.loc[:known, k].notna().sum() >= 63]
        training = self.actual.loc[:known, eligible].dropna().tail(63)
        assert len(training) == 63 and eligible
        base = pd.Series(s.V12_BASE, index=s.V12_KEYS).loc[eligible].to_numpy()
        covariance = training.cov().to_numpy()
        covariance = .75 * covariance + .25 * np.diag(np.diag(covariance))
        volatility = np.sqrt(252 * base @ covariance @ base)
        scale = min(1., .12 / volatility)
        weights = pd.Series(base * scale, index=eligible).reindex(s.V12_KEYS, fill_value=0.)
        self.receipts.append(dict(date=day, known=known, first_return=training.index[0],
                                  last_return=training.index[-1], selected="/".join(eligible),
                                  cash_target=1 - weights.sum(), **weights.to_dict()))
        return weights.to_numpy(), weights.to_numpy()


def native_v12(data):
    path = s.ROOT / "joinquant_rebuild_20261110_01/data"
    dates = data.views.hist.index
    actual = pd.DataFrame(np.nan, index=dates, columns=s.V12_KEYS)
    coverage = []
    for key, code in s.ENGINE.CODES.items():
        source = list((path / "fund_daily").glob(code + "*.parquet"))
        if source:
            raw = pd.concat([pd.read_parquet(p) for p in source]).set_index("trade_date").sort_index()
            raw.index = pd.to_datetime(raw.index).strftime("%Y-%m-%d")
        else:
            folder = s.ROOT / "review_runs/20261105_p75_long_horizon_v1"
            if key == "divlow":
                raw = pd.read_csv(folder / "divlow_daily.csv", index_col="date").rename(columns={"preClose": "pre_close", "volume": "vol"})
            else:
                assert key == "nasdaq"
                raw = pd.read_csv(s.ROOT / "p75_risk15_codex_20261106/input/global/nasdaq_processed.csv", index_col=0)
                raw = raw.rename(columns={"preClose": "pre_close", "volume": "vol"}).dropna(subset=["close"])
            source = [str(folder / "divlow_daily.csv") if key == "divlow" else "global/nasdaq_processed.csv"]
        raw = raw.loc[:"2020-01-20"]
        raw = raw.astype({"close": float, "pre_close": float, "vol": float})
        assert raw.index.is_unique and raw.index.is_monotonic_increasing
        cash = pd.Series(0., index=raw.index)
        divs = list((path / "fund_div").glob(code + "*.parquet"))
        if divs:
            dividends = pd.concat([pd.read_parquet(p) for p in divs])
            dividends = dividends.loc[dividends.div_proc.eq("实施")].drop_duplicates(["ex_date", "div_cash"])
            for event in dividends.itertuples():
                day = pd.Timestamp(event.ex_date).date().isoformat()
                if day in cash.index:
                    cash.loc[day] += float(event.div_cash)
        previous = raw.close.shift()
        gap = previous - raw.pre_close - cash
        unexplained = gap.loc[gap.abs() > .0012]
        # Isolated QMT rounding/reference anomalies are bounded and listed for review.
        coverage.append(dict(asset=key, code=code, first_quote=raw.index[0], observations=len(raw),
                             dividends=int(cash.ne(0).sum()), max_reference_gap=float(gap.abs().max()),
                             unexplained_dates="/".join(unexplained.index), sources="|".join(map(str, source))))
        assert gap.abs().max() < .011, (key, unexplained)
        returns = (raw.close + cash) / previous - 1
        actual[key] = returns.reindex(actual.index).to_numpy(dtype=float)
    # The exact frozen return and corporate-action process is authoritative on the common real-ETF span.
    actual.loc["2020-01-20":] = data.views.benchmark.loc["2020-01-20":]
    s.write(pd.DataFrame(coverage), "v12_native_history_coverage.csv")
    actual.to_parquet(s.OUT / "v12_native_history.parquet")
    return actual


def comparisons(data):
    actual = native_v12(data)
    rows = []
    for start in (s.START, "2020-01-02"):
        for bp in (9, 20):
            with s.capital(s.CAPITAL), s.prior.apply_policy(lambda n, r, p: EligibleV12(actual), s.V12_KEYS):
                a = s.ENGINE.economic("V12", actual.fillna(0), None, start, bp / 10000)
            s.record(a, f"eligible_V12_{start}_{bp}bp", rows)
            for schedule in ("monthly", "month_and_switch"):
                case = s.Case(schedule, schedule, cutoff="listed")
                s.record(s.run(data, case, start=start, bp=bp), f"eligible_GS_{schedule}_{start}_{bp}bp", rows)
    s.write(pd.DataFrame(rows), "symmetric_listing_comparison.csv")


def premium_drift(data):
    sys.path.insert(0, str(s.ROOT / "guosheng_v12_return_dependence_20261113"))
    import normalization as normal
    end = "2026-11-04"
    gs = data.views.hist.loc[:end].copy()
    v12 = data.views.benchmark.loc[:end].copy()
    coverage = json.loads((s.ROOT / "guosheng_six_regime_long_20261112/coverage.json").read_text(encoding="utf-8"))
    gn, _ = normal.gs_normal(gs, coverage)
    vn, _, _ = normal.v12_normal(v12)
    anchors = pd.read_csv(normal.HERE / "tables/anchors.csv")
    rows = []
    adjust = []
    for start in (s.START, "2020-07-30"):
        pairs = {"market": (gs, v12), "nav": (gn, vn)}
        for anchor in ("whole", "pre2020"):
            rates = anchors.loc[anchors.anchor.eq(anchor)]
            gr = rates.loc[rates.strategy.eq("GS")].set_index("asset").annual.to_dict()
            vr = rates.loc[rates.strategy.eq("V12")].set_index("asset").annual.to_dict()
            g, evidence = normal.drift_normal(gn, start, gr)
            adjust.extend(dict(strategy="GS", anchor=anchor, start=start, **r) for r in evidence)
            v, evidence = normal.drift_normal(vn, start, vr)
            adjust.extend(dict(strategy="V12", anchor=anchor, start=start, **r) for r in evidence)
            pairs[anchor] = (g, v)
        for name, (g, v) in pairs.items():
            a = s.run(data, s.Case("monthly"), start=start, returns=g)
            b = s.benchmark(data, start=start, returns=v)
            spans = {"common": (start, end)}
            s.record(a, f"GS_{name}_{start}", rows, spans)
            s.record(b, f"V12_{name}_{start}", rows, spans)
    s.write(pd.DataFrame(rows), "premium_drift_comparison.csv")
    s.write(pd.DataFrame(adjust), "drift_adjustments.csv")


def main():
    data = s.load()
    comparisons(data)
    print("Symmetric listing-history comparison complete", flush=True)
    premium_drift(data)
    print("Premium and historical return-anchor controls complete", flush=True)


if __name__ == "__main__":
    main()
