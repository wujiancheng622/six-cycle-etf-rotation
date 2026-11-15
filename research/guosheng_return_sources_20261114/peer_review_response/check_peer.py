"""Check the peer's delivered claims against local inputs, books and identities.

No strategy engine is imported. Additional sensitivity is algebra, not a forecast.
"""

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ROOT = SOURCE.parent
OUT = SOURCE / "results"
DOWNLOADS = Path("C:/Users/Administrator/Downloads")


def save(frame, name):
    frame.to_csv(HERE / name, index=False, encoding="utf-8-sig")


def identify(case):
    strategy, rest = case.split("_", 1)
    scenario, start = rest.rsplit("_", 1)
    fallback = None
    for key in ("old_dividend", "old_value"):
        if scenario.startswith(key + "_"):
            fallback, scenario = key, scenario[len(key)+1:]
    return strategy, scenario, start, fallback


def metric(ledger, meta):
    nav = np.r_[meta["capital"], ledger.nav.to_numpy()]
    ret = np.diff(nav) / nav[:-1]
    years = (pd.Timestamp(ledger.index[-1])-pd.Timestamp(meta["base_date"])).days/365.25
    return dict(cagr=(nav[-1]/nav[0])**(1/years)-1, vol=np.std(ret, ddof=1)*np.sqrt(252),
                mdd=np.max(1-nav/np.maximum.accumulate(nav)), total=nav[-1]/nav[0]-1,
                mean_cash=(ledger.cash/ledger.nav).mean(), fees=ledger.cost.sum()), ret, years


def main():
    peer_receipt = json.loads((DOWNLOADS / "本轮验证范围与误差.json").read_text(encoding="utf-8-sig"))
    path_map = {name: SOURCE / name for name in ("return_sources.py", "premium_checks.py", "protocol.json")}
    path_map.update({name: OUT / name for name in ("comparison.csv", "exact_decomposition.csv", "fixed_holdings_premium.csv")})
    hashes = []
    for name, expected in peer_receipt["source_hashes"].items():
        actual = hashlib.sha256(path_map[name].read_bytes()).hexdigest()
        assert actual == expected, name
        hashes.append(dict(file=str(path_map[name]), expected=expected, actual=actual, matches=True))
    report = SOURCE / "国盛策略综合评审文档.md"
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    assert digest == peer_receipt["source_report_sha256"]
    hashes.append(dict(file=str(report), expected=peer_receipt["source_report_sha256"], actual=digest, matches=True))
    save(pd.DataFrame(hashes), "hash_checks.csv")
    comparison = pd.read_csv(OUT / "comparison.csv")
    adjustments = pd.read_csv(OUT / "mean_adjustments.csv")
    bases = {key: pd.read_parquet(OUT / f"{key}.parquet") for key in
             ("GS_market", "V12_market", "GS_nav", "V12_nav_old", "V12_nav_shared")}
    fallback_returns = pd.read_parquet(ROOT / "guosheng_proxy_substitution_20261114_r8/input/fallback_returns.parquet")
    accounts, checks, decomps = {}, [], []
    for row in comparison.itertuples():
        strategy, scenario, start, fallback = identify(row.case)
        source = (strategy + "_market") if scenario == "market" else ("GS_nav" if strategy == "GS" else
                  "V12_nav_old" if scenario == "nav_original" else "V12_nav_shared")
        returns = bases[source].copy()
        changes = adjustments.loc[adjustments.strategy.eq(strategy) & adjustments.scenario.eq(scenario) & adjustments.start.eq(start)]
        for change in changes.itertuples():
            if pd.notna(change.simple_daily_reduction):
                returns[change.asset] -= change.simple_daily_reduction
            else:
                days = pd.Series(pd.to_datetime(returns.index), index=returns.index).diff().dt.days.fillna(0)/365.25
                shift = np.log1p(change.achieved) - np.log1p(change.original_cagr)
                returns[change.asset] = np.expm1(np.log1p(returns[change.asset]) + shift*days)
        if fallback:
            returns = pd.concat([returns, fallback_returns.reindex(returns.index)], axis=1)
        folder = OUT / "accounts" / row.case
        meta = json.loads((folder / "meta.json").read_text())
        ledger, weights, pnl, trades, signals = [pd.read_parquet(folder / (key + ".parquet")) for key in
                                                ("ledger", "weights", "pnl", "trades", "signals")]
        columns = list(pnl.columns)
        returns = returns.reindex(index=ledger.index, columns=columns)
        assert returns.notna().all().all()
        net_trades = trades.assign(signed=np.where(trades.side.eq("buy"), trades.notional, -trades.notional)).pivot_table(
            index="date", columns="asset", values="signed", aggfunc="sum", fill_value=0).reindex(index=ledger.index, columns=columns, fill_value=0)
        computed_cost = np.maximum(5., trades.notional*.00015)+trades.notional*.0009
        np.testing.assert_allclose(computed_cost, trades.cost, atol=1e-8, rtol=1e-12)
        day_cost = trades.groupby("date").cost.sum().reindex(ledger.index, fill_value=0)
        cash = meta["capital"] - (net_trades.sum(axis=1)+day_cost).cumsum()
        values = np.zeros(len(columns))
        nav, max_pnl = [], 0.
        for r, trade, expected_pnl, balance in zip(returns.to_numpy(), net_trades.to_numpy(), pnl.to_numpy(), cash):
            profit = values*r
            max_pnl = max(max_pnl, float(np.max(abs(profit-expected_pnl))))
            values += profit + trade
            nav.append(values.sum()+balance)
        nav_error = float(np.max(np.abs(np.asarray(nav)-ledger.nav.to_numpy())))
        assert nav_error < 2e-6 and max_pnl < 1e-6
        np.testing.assert_allclose(cash, ledger.cash, atol=2e-6, rtol=0)
        metrics, daily, years = metric(ledger, meta)
        for key, value in metrics.items():
            np.testing.assert_allclose(value, getattr(row, key), rtol=1e-12, atol=1e-8)
        assert signals.known.lt(signals.date).all() and trades.known.lt(trades.date).all()
        checks.append(dict(case=row.case, days=len(ledger), orders=len(trades), nav_error=nav_error,
                           pnl_error=max_pnl, fee_error=float(np.max(abs(computed_cost-trades.cost)))))
        accounts[row.case] = (ledger, metrics, daily, years)
        if strategy == "GS" and scenario == "broad_pre2020" and start == "2014-01-02":
            w = weights[columns].shift(1, fill_value=0)
            exposure = float((w.mean()*returns.sum()).sum()/years)
            dynamic = float(((w-w.mean())*(returns-returns.mean())).sum().sum()/years)
            fee = float((ledger.cost/ledger.nav.shift(1, fill_value=meta["capital"])).sum()/years)
            penalty = float(np.sum(daily-np.log1p(daily))/years)
            net_log = float(np.log1p(daily).sum()/years)
            np.testing.assert_allclose(exposure+dynamic-fee-penalty, net_log, atol=1e-12)
            if fallback:
                assert weights.fcf.abs().max() == 0 and pnl.fcf.abs().max() == 0
            decomps.append(dict(case=row.case, exposure=exposure, dynamic=dynamic, fee=fee, penalty=penalty, net_log=net_log))
    checks = pd.DataFrame(checks)
    assert len(checks) == 78 and checks.days.sum() == 150280 and checks.orders.sum() == 41600
    save(checks, "independent_account_checks.csv")
    main_peer = pd.read_csv(DOWNLOADS / "主要复核结果.csv")
    for row in main_peer.itertuples():
        m = accounts[row.case][1]
        for key in ("cagr", "vol", "mdd", "mean_cash"):
            np.testing.assert_allclose(getattr(row, key), m[key], atol=2e-14)
        np.testing.assert_allclose(row.calmar, m["cagr"]/m["mdd"], atol=2e-14)
    premium = pd.read_csv(DOWNLOADS / "固定历史持仓的估值贡献.csv")
    local_premium = pd.read_csv(OUT / "fixed_holdings_premium.csv")
    for row in premium.itertuples():
        pair = local_premium.loc[local_premium.strategy.eq(row.strategy) & local_premium.start.eq(row.start)].set_index("mode")
        np.testing.assert_allclose([row.market_cagr, row.nav_fixed_weights_cagr], pair.loc[["market", "nav_fixed_historical_weights"], "cagr"], atol=2e-14)
        np.testing.assert_allclose(row.change_percentage_points, 100*(row.nav_fixed_weights_cagr-row.market_cagr), atol=2e-12)
    joint = pd.read_csv(ROOT / "guosheng_final_decision_20261114/results/updated_joint_stress.csv").query("period=='full'")
    noise = pd.read_csv(ROOT / "guosheng_final_decision_20261114/results/all_stress_uncertainty.csv").query("period=='full'")
    combined = pd.concat([joint, noise]).set_index("case")
    for row in pd.read_csv(DOWNLOADS / "额外联合压力_核读范围.csv").itertuples():
        np.testing.assert_allclose([row.cagr,row.vol,row.mdd], combined.loc[row.case,["cagr","vol","mdd"]].to_numpy(dtype=float), atol=2e-14)
    benchmark = accounts["V12_market_2014-01-02"][1]
    normalized = accounts["V12_broad_pre2020_2014-01-02"][1]
    sensitivities = []
    for d in decomps:
        for bname, benchmark_cagr in (("V12_market", benchmark["cagr"]), ("V12_pre2020", normalized["cagr"])):
            sensitivities.append(dict(case=d["case"], benchmark=bname, dynamic=d["dynamic"],
                break_even_loss_fraction=(d["net_log"]-np.log1p(benchmark_cagr))/d["dynamic"],
                only_plus3pp_loss_fraction=(d["net_log"]-np.log1p(benchmark_cagr+.03))/d["dynamic"],
                cagr_if_25pct_lost=np.expm1(d["net_log"]-.25*d["dynamic"]),
                cagr_if_50pct_lost=np.expm1(d["net_log"]-.5*d["dynamic"]),
                cagr_if_60pct_lost=np.expm1(d["net_log"]-.6*d["dynamic"])))
    sensitivities = pd.DataFrame(sensitivities)
    peer_dynamic = json.loads((DOWNLOADS / "动态项敏感性口径.json").read_text(encoding="utf-8-sig"))
    first = sensitivities.query("case=='GS_broad_pre2020_2014-01-02' and benchmark=='V12_market'").iloc[0]
    np.testing.assert_allclose(first.break_even_loss_fraction, peer_dynamic["break_even_loss_fraction"], atol=1e-12)
    np.testing.assert_allclose(first.only_plus3pp_loss_fraction, peer_dynamic["loss_fraction_to_only_plus3pp"], atol=1e-12)
    save(sensitivities, "dynamic_threshold_scope.csv")
    save(pd.DataFrame(decomps), "substitution_decomposition.csv")
    riskrows = []
    for d in decomps:
        ledger, m, returns, years = accounts[d["case"]]
        for measure in ("vol", "mdd"):
            def evaluate(scale):
                r = returns*scale
                path = np.r_[1., np.exp(np.log1p(r).cumsum())]
                return dict(cagr=path[-1]**(1/years)-1, vol=np.std(r,ddof=1)*np.sqrt(252),
                            mdd=np.max(1-path/np.maximum.accumulate(path)))
            if measure == "vol":
                scale = benchmark["vol"]/m["vol"]
            else:
                lo, hi = 0., 1.
                for _ in range(70):
                    mid = (lo+hi)/2
                    if evaluate(mid)["mdd"] > benchmark["mdd"]:
                        hi = mid
                    else:
                        lo = mid
                scale = (lo+hi)/2
            result = evaluate(scale)
            riskrows.append(dict(case=d["case"], match=measure, scale=scale, **result,
                                 benchmark_cagr=benchmark["cagr"], note="Ex-post scaling of net daily returns; zero cash return, no new order replay or commission recalculation; descriptive only"))
    save(pd.DataFrame(riskrows), "risk_definition_diagnostics.csv")
    receipt = dict(source_hashes_matched=len(hashes), accounts=len(checks), days=int(checks.days.sum()), orders=int(checks.orders.sum()),
                   account_metric_checks=len(checks)*6, peer_main_rows=len(main_peer), peer_premium_rows=len(premium),
                   max_own_reconstruction_nav_error=float(checks.nav_error.max()), max_own_fee_error=float(checks.fee_error.max()),
                   peer_dynamic_threshold_reproduced=True,
                   scope="Own input-and-saved-trade balance recursion, table matching and algebra. No independent macro source reconstruction, no full signal engine rerun; peer's 8-run process is not independently authenticated by six delivered files.",
                   dynamic_scope="Fixed decomposition terms; not a realized loss probability, forecast, backtest or future drawdown bound",
                   peer_input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DOWNLOADS/name for name in (
                       "国盛综合评审_独立复核结论.md", "主要复核结果.csv", "本轮验证范围与误差.json", "额外联合压力_核读范围.csv", "动态项敏感性口径.json", "固定历史持仓的估值贡献.csv")]})
    (HERE/"verification.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!='peer_input_sha256'},ensure_ascii=False,indent=2))
    print(sensitivities.to_string(index=False))
    print(pd.DataFrame(riskrows).drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
