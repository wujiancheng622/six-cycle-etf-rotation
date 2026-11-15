"""Combine verified source corrections with fixed substitutes and measured proxy errors."""

import numpy as np
import pandas as pd

import study as s


def proxies(data):
    p = pd.read_parquet(s.R8 / "input/gs_proxy.parquet")
    q = pd.read_csv(s.HERE / "input/shfe_index.csv", index_col="date")
    assert q.index.is_unique and q.index.is_monotonic_increasing
    np.testing.assert_allclose(q.previous_close.iloc[1:], q.close.shift().iloc[1:], atol=.011, rtol=0)
    p["metals"] = (q.close / q.previous_close - 1).reindex(p.index)
    a = data.views.hist.loc[:p.index[-1]].copy()
    for key, date in data.views.first_actual_return.items():
        a.loc[a.index < date, key] = np.nan
    va = pd.read_parquet(s.R8 / "input/v12_actual.parquet")
    vp = pd.read_parquet(s.R8 / "input/v12_proxy.parquet")
    rows = []
    means = {}
    for strategy, actual, proxy, columns in (("GS", a, p, ("metals", "fcf", "bond", "dividend")),
                                            ("V12", va, vp, ("csi300", "csi1000", "divlow", "bond10", "metals", "energy", "soy"))):
        for key in columns:
            pair = pd.concat([actual[key], proxy[key]], axis=1).dropna()
            pair.columns = ["actual", "proxy"]
            error = np.log1p(pair.actual) - np.log1p(pair.proxy)
            means[(strategy, key)] = float(error.mean())
            rows.append(dict(strategy=strategy, asset=key, start=pair.index[0], end=pair.index[-1], days=len(pair),
                             annual_log_error=float(error.mean() * 252), tracking_error=float((pair.actual - pair.proxy).std() * np.sqrt(252)),
                             correlation=pair.actual.corr(pair.proxy)))
    s.write(pd.DataFrame(rows), "updated_proxy_error.csv")
    return a, p, va, vp, means


def overlap(data, a, p, va, vp):
    rows = []
    dates = a.dropna().index.intersection(p.dropna().index)
    a, p = a.loc[dates], p.loc[dates]
    start = dates[63]
    for name, values, risk in (("actual_actual", a, a), ("proxy_risk", a, p),
                               ("proxy_returns", p, a), ("all_proxy", p, p)):
        account = s.run(data, s.Case(name), start=start, returns=values, risk=risk)
        s.record(account, "updated_GS_overlap_" + name, rows, {"common": (start, values.index[-1])})
    for name, values, risk in (("actual_actual", va, va), ("proxy_risk", va, vp),
                               ("proxy_returns", vp, va), ("all_proxy", vp, vp)):
        with s.capital(s.CAPITAL), s.prior.apply_policy(lambda n, r, p: s.ORIGINAL_V12_POLICY("V12", risk, None), s.V12_KEYS):
            account = s.ENGINE.economic("V12", values, None, start, .0009)
        s.record(account, "updated_V12_overlap_" + name, rows, {"common": (start, values.index[-1])})
    s.write(pd.DataFrame(rows), "updated_proxy_overlap.csv")


def joint(data, means):
    gs, v12 = data.full.copy(), data.views.benchmark.copy()
    for (strategy, key), error in means.items():
        if strategy == "GS":
            early = gs.index < data.views.first_actual_return[key]
            gs.loc[early, key] = np.expm1(np.log1p(gs.loc[early, key]) - abs(error))
        else:
            early = v12.index < "2020-01-20"
            v12.loc[early, key] = np.expm1(np.log1p(v12.loc[early, key]) + abs(error))
    rows = []
    for bp in (9, 20, 50):
        s.record(s.benchmark(data, bp=bp, returns=v12), f"joint_V12_{bp}bp", rows)
        for fallback in ("old_dividend", "old_value"):
            for schedule in ("monthly", "month_and_switch"):
                for lag in (0, 3):
                    case = s.Case(f"joint_{fallback}_{schedule}_{lag}d", schedule=schedule, fallback=fallback)
                    state = data.state if lag == 0 else s.macro(data, "all_macro_extra_calendar_lag", lag)
                    result = s.run(data, case, state=state, bp=bp, returns=gs, limit=True)
                    s.record(result, f"{case.name}_{bp}bp", rows)
        print(f"Joint downside controls completed for {bp} bp", flush=True)
    s.write(pd.DataFrame(rows), "updated_joint_stress.csv")


def main():
    data = s.load()
    a, p, va, vp, means = proxies(data)
    overlap(data, a, p, va, vp)
    joint(data, means)


if __name__ == "__main__":
    main()
