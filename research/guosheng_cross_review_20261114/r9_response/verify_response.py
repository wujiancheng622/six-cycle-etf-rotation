"""Check R9's measurement corrections and R8's archived stress-test evidence."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CROSS = HERE.parent
ROOT = CROSS.parent
R8 = ROOT / "guosheng_proxy_substitution_20261114_r8"
sys.dont_write_bytecode = True
sys.path.insert(0, str(CROSS))
import compare_evidence as cross


def archived_r8():
    for name, digest in json.loads((R8 / "results/manifest.json").read_text(encoding="utf-8")).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    tables = [pd.read_csv(R8 / "results" / file) for file in
              ("comparison.csv", "capacity_comparison.csv", "combined_comparison.csv")]
    table = pd.concat(tables, ignore_index=True)
    accounts = {}
    checked = []
    for row in table.itertuples():
        identifier = f"{row.group}_{row.account_start}_{int(row.bp)}bp_{row.label}"
        folder = R8 / "results/accounts" / identifier
        if identifier not in accounts:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            data = {field: pd.read_parquet(folder / f"{field}.parquet") for field in ("ledger", "pnl", "trades")}
            ledger, pnl, trades = data["ledger"], data["pnl"], data["trades"]
            previous = ledger.nav.shift(1, fill_value=meta["capital"])
            residual = (previous + pnl.sum(axis=1) - ledger.cost - ledger.nav).abs().max()
            assert residual < 1e-6 and ledger.cash.ge(-1e-7).all()
            assert trades.known.lt(trades.date).all()
            accounts[identifier] = (ledger, meta)
        ledger, meta = accounts[identifier]
        interval = ledger.loc[row.start:row.end]
        position = ledger.index.get_loc(interval.index[0])
        base = meta["capital"] if position == 0 else ledger.nav.iloc[position - 1]
        base_date = meta["base_date"] if position == 0 else ledger.index[position - 1]
        nav = np.r_[base, interval.nav.to_numpy()]
        returns = nav[1:] / nav[:-1] - 1
        years = (pd.Timestamp(interval.index[-1]) - pd.Timestamp(base_date)).days / 365.25
        result = dict(cagr=(nav[-1] / base) ** (1 / years) - 1,
                      vol=returns.std(ddof=1) * np.sqrt(252),
                      mdd=(1 - nav / np.maximum.accumulate(nav)).max(),
                      mean_cash=(interval.cash / interval.nav).mean())
        np.testing.assert_allclose(list(result.values()), [getattr(row, key) for key in result], atol=2e-12, rtol=0)
        checked.append(dict(account=identifier, period=row.window, start=row.start, end=row.end, **result))
    pd.DataFrame(checked).to_csv(HERE / "r8_metric_checks.csv", index=False, encoding="utf-8-sig")
    print(f"R8 verified {len(accounts)} archived accounts and {len(checked) * 4} numerical metrics", flush=True)


def r8_replay_combined():
    # Importing this archived study sets its own initial capital to RMB 500,000.
    sys.path.insert(0, str(R8))
    import run_tests as study
    from capacity_replay import economic_limited

    study.engine.CAPITAL = study.CAPITAL
    fallbacks = pd.read_parquet(study.INP / "fallback_returns.parquet")
    study.FEX = {}
    capacity = pd.DataFrame(np.inf, index=study.H.index, columns=[*study.H.columns, *fallbacks.columns])
    for asset, code in (("old_dividend", "510880.SH"), ("old_value", "510030.SH")):
        quotes = pd.read_parquet(study.INP / f"fund_daily_{code}.parquet").set_index("trade_date").sort_index()
        quotes.index = pd.to_datetime(quotes.index).strftime("%Y-%m-%d")
        quotes = quotes.reindex(study.H.index)
        valid = quotes.vol.gt(0)
        study.FEX[asset] = list(quotes.index[~(valid & valid.shift(1, fill_value=True))])
        capacity[asset] = (.01 * quotes.amount.fillna(0).rolling(20).mean().shift() * 1000).where(valid, 0).fillna(0)
    gs, v12 = study.H.copy(), study.V.copy()
    errors = pd.read_csv(study.OUT / "asset_overlap.csv")
    for row in errors.itertuples():
        if row.strategy == "GS":
            eligible = gs.index < study.COVER["actual_return_from"][row.asset]
            gs.loc[eligible, row.asset] = np.expm1(np.log1p(gs.loc[eligible, row.asset]) - abs(row.annual_log_drift) / 252)
        else:
            eligible = v12.index < "2020-01-20"
            v12.loc[eligible, row.asset] = np.expm1(np.log1p(v12.loc[eligible, row.asset]) + abs(row.annual_log_drift) / 252)
    returns = pd.concat([gs, fallbacks], axis=1)
    returns.attrs.update(gs.attrs)
    results = {"V12_favorable_drift": study.v12(v12, "2014-01-02", 20)}
    for label, asset in (("dividend_adverse_drift", "old_dividend"), ("value_adverse_drift", "old_value")):
        policy = study.SubstitutionPolicy(returns, study.S, list(returns.columns), asset,
                                          study.COVER["actual_return_from"]["fcf"], False)
        account, _ = economic_limited(returns, policy, "2014-01-02", .002, capacity)
        results[label] = account
    for label, account in results.items():
        folder = study.OUT / "accounts" / f"combined_2014-01-02_20bp_{label}"
        for field in ("ledger", "signals", "weights", "pnl", "trades"):
            pd.testing.assert_frame_equal(getattr(account, field), pd.read_parquet(folder / f"{field}.parquet"))
    print("R8 20 bp joint stress: all three daily ledgers, targets and trades reproduced", flush=True)


def corrected_chart_and_heads():
    local, views = cross.latest_local()
    prior, expanded, merged, events = cross.r6_states(local)
    no_old_lpr = cross.verify_policy_impact(prior, expanded, local, merged, events)
    cross.chart_comparison(prior, expanded, local, merged, no_old_lpr)
    target = cross.RESULTS / "old_lpr_source_conflict.json"
    target.write_text(json.dumps(cross.old_lpr_receipt(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dates = views.hist.index
    allowed = {asset: dates[dates.searchsorted(day, side="right")]
               for asset, day in cross.CFG["index_publication_control"]["dates"].items()}
    gs = cross.run_published(views, merged, allowed, True, True)
    v12 = cross.run(views, "amount", cross.START, "V12")
    original = cross.run(views, "amount", cross.START, "GS_stage_switch", state=merged)
    listed = cross.run_published(views, merged, views.first_actual_return, True, True)
    cash_rows = []
    for name, account in (("original", original), ("three_index_dates", gs), ("listed_etfs", listed), ("V12", v12)):
        period = account.ledger.loc[cross.START:cross.REPORT_END]
        m = cross.window(account, cross.START, cross.REPORT_END)
        cash_rows.append(dict(case=name, mean_cash=(period.cash / period.nav).mean(), cagr=m["cagr"],
                              vol=m["vol"], mdd=m["mdd"]))
    pd.DataFrame(cash_rows).to_csv(HERE / "cash_exposure_checks.csv", index=False, encoding="utf-8-sig")
    g = np.log1p(cross.log_attribution(gs)[0].loc[cross.START:cross.REPORT_END])
    v = np.log1p(cross.log_attribution(v12)[0].loc[cross.START:cross.REPORT_END])
    assert g.index.equals(v.index)
    count = int(np.ceil(len(g) * .01))
    gs_best = g.nlargest(count).index
    v12_best = v.nlargest(count).index
    years = (pd.Timestamp(cross.REPORT_END) - pd.Timestamp(gs.base_date)).days / 365.25
    rows = []
    for name, keep_g, keep_v in (("original", np.ones(len(g), dtype=bool), np.ones(len(v), dtype=bool)),
                                ("same_gs_top_days", ~g.index.isin(gs_best), ~v.index.isin(gs_best)),
                                ("each_own_top_days", ~g.index.isin(gs_best), ~v.index.isin(v12_best))):
        rows.append(dict(test=name, removed_days=0 if name == "original" else count,
                         gs_cagr=np.expm1(g[keep_g].sum() / years),
                         v12_cagr=np.expm1(v[keep_v].sum() / years)))
    share = float((g.loc[gs_best] - v.loc[gs_best]).sum() / (g - v).sum())
    receipt = dict(top_days=count, selected_gs_days=gs_best.tolist(), selected_v12_days=v12_best.tolist(),
                   net_excess_log_share=share, rules="Keep the original elapsed calendar years after zeroing selected returns")
    pd.DataFrame(rows).to_csv(HERE / "symmetric_head_day_checks.csv", index=False, encoding="utf-8-sig")
    (HERE / "head_day_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Selected days share of net log excess: {share:.6f}", flush=True)


def main():
    def guard(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("This verification uses local frozen data")
    sys.addaudithook(guard)
    corrected_chart_and_heads()
    archived_r8()
    r8_replay_combined()


if __name__ == "__main__":
    main()
