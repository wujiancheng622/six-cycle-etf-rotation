"""Separate asset tailwinds and premium changes from preserved allocation timing."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "guosheng_final_decision_20261114"),
               str(ROOT / "guosheng_v12_return_dependence_20261113")]
import study as s
import normalization as n

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
CONF = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
END = CONF["end"]


def write(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def setup():
    digest = hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    receipt = HERE / "protocol.sha256"
    if receipt.exists():
        assert receipt.read_text().strip() == digest
    else:
        receipt.write_text(digest + "\n")
    data = s.load()
    s.OUT = OUT
    g, v = data.views.hist.loc[:END].copy(), data.views.benchmark.loc[:END].copy()
    coverage = json.loads((ROOT / "guosheng_six_regime_long_20261112/coverage.json").read_text())
    gn, gp = n.gs_normal(g, coverage)
    vn, vp, repairs = n.v12_normal(v)
    vs = vn.copy()
    for key in ("gold", "metals"):
        p = gp[key]
        use = p.index[p["nav_used"].to_numpy(dtype=bool)].intersection(vs.index)
        vs.loc[use, key] = p.loc[use, "nav_return"]
    anchors = n.anchors(gn)
    anchors.loc[anchors.asset.eq("metals"), "source"] = "2006-2012 futures proxy; 2013-2019 official SHFE index; thereafter 159980 adjusted NAV"
    write(anchors, "anchors.csv")
    write(pd.DataFrame(n.premiums(gp, "GS") + n.premiums(vp, "V12")), "premium_observations.csv")
    (OUT / "nav_repairs.json").write_text(json.dumps(repairs, indent=2) + "\n")
    for name, frame in (("GS_market", g), ("V12_market", v), ("GS_nav", gn), ("V12_nav_old", vn), ("V12_nav_shared", vs)):
        frame.to_parquet(OUT / (name + ".parquet"))
    return data, (g, v), (gn, vn), (gn, vs), (gp, vp), anchors


def normalize(frame, start, targets, method="simple"):
    result, rows = frame.copy(), []
    base = frame.index[frame.index.get_loc(start) - 1]
    years = n.years(base, frame.index[-1])
    for asset, target in targets.items():
        x = frame.loc[start:, asset].to_numpy()
        original = np.expm1(np.log1p(x).sum() / years)
        used = min(original, target)
        if method == "simple":
            delta = 0.
            if original > target + 1e-12:
                lo, hi = 0., min(1 + frame[asset].min(), 1 + x.min()) - 1e-9
                for _ in range(90):
                    middle = (lo + hi) / 2
                    actual_log = np.log1p(x - middle).sum() / years
                    if actual_log > np.log1p(target):
                        lo = middle
                    else:
                        hi = middle
                delta = (lo + hi) / 2
            result[asset] = frame[asset] - delta
            assert result[asset].min() > -1
        else:
            shifted, _ = n.drift_normal(frame, start, {asset: target}, down_only=True)
            result[asset] = shifted[asset]
            delta = np.nan
        achieved = np.expm1(np.log1p(result.loc[start:, asset]).sum() / years)
        np.testing.assert_allclose(achieved, used, atol=2e-12, rtol=0)
        rows.append(dict(asset=asset, original_cagr=original, target=target, achieved=achieved,
                         simple_daily_reduction=delta, clipped_upward_target=target > original))
    if method == "simple":
        np.testing.assert_allclose(result.cov(), frame.cov(), atol=1e-15, rtol=1e-12)
        for end in ("2014-12-31", "2020-12-31", "2025-12-31", END):
            np.testing.assert_allclose(result.loc[:end].tail(63).cov(), frame.loc[:end].tail(63).cov(),
                                       atol=1e-15, rtol=1e-12)
    return result, rows


def rates(anchors, strategy, anchor, scope):
    rows = anchors.loc[(anchors.strategy.eq(strategy)) & anchors.anchor.eq(anchor)]
    values = rows.set_index("asset").annual.to_dict()
    if scope == "gold":
        return {"gold": values["gold"]}
    if scope == "focus":
        return {key: values[key] for key in (("gold",) if strategy == "GS" else ("gold", "nasdaq", "sp500"))}
    if strategy == "V12":
        values["metals"] = float(anchors.loc[anchors.strategy.eq("GS") & anchors.anchor.eq(anchor) & anchors.asset.eq("metals"), "annual"].iloc[0])
    return values


def attribution(account, frame, strategy, scenario, start):
    ledger = account.ledger
    returns = frame.reindex(ledger.index)
    keys = list(returns.columns)
    weights = account.weights[keys].shift(1, fill_value=0)
    previous = ledger.nav.shift(1, fill_value=s.CAPITAL)
    costs = ledger.cost / previous
    path = ledger.nav / previous - 1
    np.testing.assert_allclose((weights * returns).sum(axis=1) - costs, path, atol=2e-14)
    rows, details = [], []
    for period, lo, hi in (("full", start, END), ("2014-2019", start, "2019-12-31"),
                          ("2020-2022", "2020-01-01", "2022-12-31"), ("2023-latest", "2023-01-01", END)):
        r = returns.loc[lo:hi]
        if len(r) < 200:
            continue
        w = weights.loc[r.index]
        index0 = ledger.index.get_loc(r.index[0])
        base = account.base_date if index0 == 0 else ledger.index[index0 - 1]
        years = n.years(base, r.index[-1])
        unconditional = w.mean() * r.sum() / years
        timing = ((w - w.mean()) * (r - r.mean())).sum() / years
        cost = costs.loc[r.index].sum() / years
        log_penalty = (path.loc[r.index] - np.log1p(path.loc[r.index])).sum() / years
        log_net = np.log1p(path.loc[r.index]).sum() / years
        np.testing.assert_allclose(unconditional.sum() + timing.sum() - cost - log_penalty, log_net, atol=1e-13)
        rows.append(dict(strategy=strategy, scenario=scenario, account_start=start, period=period,
                         start=r.index[0], end=r.index[-1], mean_exposure_arithmetic=unconditional.sum(),
                         dynamic_allocation_arithmetic=timing.sum(), cost_arithmetic=cost,
                         arithmetic_to_log_penalty=log_penalty, annual_log_net=log_net, cagr=np.expm1(log_net)))
        for key in keys:
            details.append(dict(strategy=strategy, scenario=scenario, account_start=start, period=period,
                                asset=key, mean_weight=w[key].mean(), asset_arithmetic_annual=r[key].sum()/years,
                                mean_exposure_arithmetic=unconditional[key],
                                dynamic_allocation_arithmetic=timing[key],
                                combined_arithmetic=(w[key] * r[key]).sum()/years))
    return rows, details


def fixed_weights(account, baseline_returns, changed_returns, name):
    index = account.ledger.index
    weights = account.weights[list(baseline_returns.columns)].shift(1, fill_value=0)
    original = baseline_returns.loc[index]
    changed = changed_returns.loc[index]
    before = ((weights - weights.mean()) * (original - original.mean())).sum()
    after = ((weights - weights.mean()) * (changed - changed.mean())).sum()
    error = float((before - after).abs().max())
    np.testing.assert_allclose(before, after, atol=2e-13, rtol=1e-12)
    previous = account.ledger.nav.shift(1, fill_value=s.CAPITAL)
    cost = account.ledger.cost / previous
    r = (weights * changed).sum(axis=1) - cost
    static = (weights.mean() * changed).sum(axis=1) - cost
    result = []
    years = n.years(account.base_date, index[-1])
    for mode, values in (("original_dynamic_weights", r), ("same_average_static_weights", static)):
        nav = np.r_[1., np.exp(np.log1p(values).cumsum())]
        result.append(dict(case=name, mode=mode, start=index[0], end=index[-1], cagr=nav[-1]**(1/years)-1,
                           vol=values.std(ddof=1)*np.sqrt(252), mdd=float((1-nav/np.maximum.accumulate(nav)).max()),
                           timing_covariance_sum_max_difference=error))
    return result


def main():
    data, market, old_nav, nav, panels, anchors = setup()
    allrows, adjusts, decomps, detail, fixed, checks = [], [], [], [], [], []
    for start in CONF["starts"]:
        cases = {"market": market, "nav_original": old_nav, "nav_shared": nav}
        for scenario in CONF["scenarios"][3:]:
            bits = scenario.split("_")
            scope, anchor = bits[0], bits[1]
            method = "logcalendar" if len(bits) == 3 else "simple"
            frames = []
            for strategy, frame in zip(("GS", "V12"), nav):
                changed, adjustment = normalize(frame, start, rates(anchors, strategy, anchor, scope), method)
                adjusts.extend(dict(strategy=strategy, scenario=scenario, start=start, **row) for row in adjustment)
                frames.append(changed)
            cases[scenario] = tuple(frames)
        baseline_accounts = {}
        for scenario, frames in cases.items():
            for strategy, frame in zip(("GS", "V12"), frames):
                account = s.run(data, s.Case("monthly"), start=start, returns=frame) if strategy == "GS" else s.benchmark(data, start=start, returns=frame)
                name = f"{strategy}_{scenario}_{start}"
                s.record(account, name, allrows, {"common": (start, END)})
                if scenario == "nav_shared":
                    baseline_accounts[strategy] = account
                if scenario in ("market", "nav_shared", "broad_whole", "broad_pre2020"):
                    a, d = attribution(account, frame, strategy, scenario, start)
                    decomps.extend(a)
                    detail.extend(d)
                if scenario in ("broad_whole", "broad_pre2020"):
                    original = nav[0 if strategy == "GS" else 1]
                    fixed.extend(fixed_weights(baseline_accounts[strategy], original, frame, name))
                    if strategy == "GS":
                        a, b = baseline_accounts[strategy].signals, account.signals
                        pd.testing.assert_frame_equal(a[["date", "known", "phase", "selected"]], b[["date", "known", "phase", "selected"]])
                        difference = float(np.abs(a[list(s.KEYS)].to_numpy()-b[list(s.KEYS)].to_numpy()).max())
                        assert difference < 1e-10
                        checks.append(dict(case=name, target_weight_max_difference=difference,
                                           full_covariance_max_difference=float((original.cov()-frame.cov()).abs().max().max())))
        if start != "2023-01-03":
            for fallback in ("old_dividend", "old_value"):
                for scenario in ("nav_shared", "broad_whole", "broad_pre2020"):
                    frame = pd.concat([cases[scenario][0], data.full[["old_dividend", "old_value"]].loc[:END]], axis=1)
                    frame.attrs["excluded_risk_dates"] = data.full.attrs["excluded_risk_dates"]
                    case = s.Case(fallback, fallback=fallback, permanent=True)
                    account = s.run(data, case, start=start, returns=frame, limit=True)
                    s.record(account, f"GS_{fallback}_{scenario}_{start}", allrows, {"common": (start, END)})
        print(f"Completed premium, mean and timing controls from {start}", flush=True)
    write(pd.DataFrame(allrows), "comparison.csv")
    write(pd.DataFrame(adjusts), "mean_adjustments.csv")
    write(pd.DataFrame(decomps), "exact_decomposition.csv")
    write(pd.DataFrame(detail), "asset_decomposition.csv")
    write(pd.DataFrame(fixed), "fixed_weight_controls.csv")
    write(pd.DataFrame(checks), "timing_preservation_checks.csv")
    print(pd.DataFrame(allrows).query("case.str.contains('2014-01-02')", engine="python")[["case", "cagr", "vol", "mdd"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
