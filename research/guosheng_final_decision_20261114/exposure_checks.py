"""Describe fixed-asset exposure, covariance sensitivity and decision count."""

import json
import numpy as np
import pandas as pd

import study as s


def main():
    data = s.load()
    name = "monthly_2014-01-02_9bp"
    folder = s.OUT / "accounts" / name
    ledger = pd.read_parquet(folder / "ledger.parquet")
    signals = pd.read_parquet(folder / "signals.parquet")
    daily = ledger.nav / ledger.nav.shift(1, fill_value=s.CAPITAL) - 1
    daily.index = pd.to_datetime(daily.index)
    target = np.expm1(np.log1p(daily).groupby(daily.index.to_period("M")).sum()).loc[:"2026-10"]
    asset = data.views.hist.loc[s.START:].copy()
    asset.index = pd.to_datetime(asset.index)
    monthly = np.expm1(np.log1p(asset).groupby(asset.index.to_period("M")).sum()).loc[:"2026-10"]
    rows = []
    for lo, hi in (("2014-01", "2026-10"), ("2014-01", "2019-12"), ("2020-01", "2022-12"), ("2023-01", "2026-10")):
        x = monthly.loc[lo:hi].to_numpy()
        y = target.loc[lo:hi].to_numpy()
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ beta
        rows.append(dict(start=lo, end=hi, months=len(x), annual_intercept=12*beta[0],
                         r_squared=1 - np.sum(residual**2) / np.sum((y-y.mean())**2),
                         **dict(zip(s.KEYS, beta[1:]))))
    s.write(pd.DataFrame(rows), "fixed_exposure_regression.csv")
    data.state.loc[s.START:, ["money", "credit", "growth"]].corr().to_csv(s.OUT / "macro_direction_correlation.csv")
    perturbations = []
    exclusions = data.views.hist.attrs.get("excluded_risk_dates", {})
    for row in signals.itertuples():
        assets = row.selected.split("/")
        excluded = {day for key in assets for day in exclusions.get(key, ())}
        train = data.views.hist.loc[:row.known, assets].drop(index=sorted(excluded), errors="ignore").tail(63)
        covariance = train.cov().to_numpy()
        baseline = s.risk_parity(covariance)
        perturbed = s.risk_parity(covariance + .01 * np.diag(np.diag(covariance)))
        perturbations.append(dict(date=row.date, assets=row.selected,
                                  l1_weight_change=float(abs(baseline - perturbed).sum()),
                                  covariance_condition=float(np.linalg.cond(covariance))))
    s.write(pd.DataFrame(perturbations), "risk_matrix_perturbation.csv")
    counts = data.state.loc[s.START:, "stage"].value_counts().sort_index()
    counts.to_csv(s.OUT / "stage_day_counts.csv")
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    print('Covariance 1% diagonal perturbation maximum weight L1', max(r['l1_weight_change'] for r in perturbations), flush=True)


if __name__ == "__main__":
    main()
