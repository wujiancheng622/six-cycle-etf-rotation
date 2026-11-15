"""Decompose the existing credit signal using its archived release dates."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def main():
    # Same 12-month total and three-month pulse difference as the original rule.
    # Shares of the aggregate lagged denominator avoid dividing by sectors with
    # negative or near-zero net new lending. The two terms sum to the original.
    credit = pd.read_csv(ROOT / "guosheng_six_regime_long_20261112/data/credit_monthly.csv")
    periods = pd.PeriodIndex(credit.month, freq="M")
    assert periods.equals(pd.period_range(periods[0], periods[-1], freq="M"))
    credit["released"] = pd.to_datetime(credit.released)
    assert credit.released.is_monotonic_increasing and credit.released.is_unique
    ttm = credit[["household_new_yi", "firm_new_yi"]].rolling(12).sum()
    denom = ttm.shift(12).sum(axis=1, min_count=2)
    assert denom.dropna().gt(0).all()
    pulse = (ttm - ttm.shift(12)).div(denom, axis=0)
    delta = pulse.diff(3)
    np.testing.assert_allclose(delta.sum(axis=1, min_count=2), credit.credit_change, atol=1e-12)
    credit["household_delta"] = delta.household_new_yi
    credit["enterprise_delta"] = delta.firm_new_yi
    credit["directions_disagree"] = credit.household_delta.mul(credit.enterprise_delta).lt(0)
    columns = ["month", "released", "credit_change", "household_delta", "enterprise_delta", "directions_disagree"]
    credit.loc[credit.credit_change.notna(), columns].to_csv(OUT / "credit_decomposition.csv", index=False, encoding="utf-8-sig")

    monthly = pd.read_csv(ROOT / "guosheng_timing_attribution_20261115/results/month_detail.csv")
    base = monthly.loc[monthly.run.eq("base")].copy()
    base["known"] = pd.to_datetime(base.known)
    detail = pd.merge_asof(base.sort_values("known"), credit[columns], left_on="known", right_on="released")
    assert detail.released.le(detail.known).all()
    assert detail.credit_change.notna().all()
    detail["credit_is_improving"] = detail.credit_change.gt(0)
    detail.to_csv(OUT / "decision_date_credit.csv", index=False, encoding="utf-8-sig")
    print("Complete decision periods:", len(detail))
    print(detail.groupby("directions_disagree").gap.agg(["size", "mean"]))
    print(detail.groupby(["credit_is_improving", "directions_disagree"]).gap.agg(["size", "mean"]))
    print(detail.loc[detail.date.ge("2026-01-01"), ["date", "known", "month", "credit_change",
                                                  "household_delta", "enterprise_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
