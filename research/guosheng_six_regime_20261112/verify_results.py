"""Check the replay, concentration, and cost break-even at the matched V12 hurdle."""
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from provider_data import CARRIERS, END, HERE, dated, sha, write
from regimes import ASSETS_BY_STAGE, build, state_path
from run_study import CAPITAL, KEYS, OUT, Rotation, Snapshots, apply_policy, engine, measures
from market import market_inputs

FROZEN = "GS_monthly"
LONG_START = "2016-01-04"
ACTUAL_START = "2025-11-06"
MAIN_SLIP = .0009
DIR = HERE / "verification"


def net_annualized(ledger, base):
    years = (pd.Timestamp(ledger.index[-1])-pd.Timestamp(base)).days / 365.25
    return (ledger.nav.iloc[-1]/CAPITAL)**(1/years)-1


def verify_qmt():
    compare = {}
    for key, code in CARRIERS.items():
        params = urlencode(dict(stock_code=code, period="1d", start_time="20251105", end_time=END,
                                count=-1, dividend_type="none", download="false", fill_data="false"))
        try:
            with urlopen(f"http://127.0.0.1:8002/api/v1/market_data?{params}", timeout=16) as response:
                bars = json.load(response)["bars"]
            qmt = pd.DataFrame(bars)
            if qmt.empty:
                compare[key] = dict(code=code, matched=0, error="QMT returned no unadjusted bars")
                continue
            qmt.index = pd.to_datetime(qmt["time"], unit="ms", utc=True).dt.tz_convert(
                "Asia/Shanghai").dt.strftime("%Y%m%d")
            columns = ["open", "close"]
            check = dated("fund_daily", code)[columns].loc["20251105":]
            assert qmt.index.is_unique and check.index.is_unique
            merged = qmt[columns].join(check, how="inner", lsuffix="_qmt", rsuffix="_ts")
            delta = {f: float((merged[f+"_qmt"]-merged[f+"_ts"]).abs().max()) for f in columns}
            compare[key] = dict(code=code, matched=len(merged), only_qmt=len(qmt)-len(merged),
                                only_tushare=len(check)-len(merged), max_difference=delta)
        except OSError as e:
            compare[key] = dict(code=code, error=f"{type(e).__name__}: {e}")
    return compare


def timing_check(phases, returns):
    # A truth-table example includes both non-canonical combinations and a tie.
    ix = pd.date_range("2020-01-01", periods=9)
    bits = ((1, 1, -1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
            (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (0, 1, 1))
    cols = [pd.Series(c, index=ix) for c in zip(*bits)]
    assert state_path(*cols).tolist() == [1, 2, 3, 4, 4, 5, 6, 6, 6]
    day, known = "2025-11-06", "2025-11-05"
    left = Rotation(FROZEN, returns, phases).targets(day, known, 0, CAPITAL)[0]
    shocked = returns.copy()
    shocked.loc[day:] = .4
    future_phases = phases.copy()
    future_phases.loc[pd.Timestamp(day):, "stage"] = 5
    right = Rotation(FROZEN, shocked, future_phases).targets(day, known, 0, CAPITAL)[0]
    np.testing.assert_array_equal(left, right)
    return dict(micro_state_cases=len(bits), future_data_perturbation_invariant=True,
                weights_sum=float(left.sum()))


def verify_accounts():
    receipts = {}
    for path in sorted(OUT.rglob("ledger.parquet")):
        parent = path.parent
        sig = pd.read_parquet(parent / "signals.parquet")
        trades = pd.read_parquet(parent / "trades.parquet")
        ledger = pd.read_parquet(path)
        assert ledger.index.is_unique and ledger.index.is_monotonic_increasing
        assert (sig.known < sig.date).all()
        assert (trades.known < trades.date).all()
        if "credit_date" in sig:
            assert (sig.credit_date <= sig.known).all() and (sig.growth_date <= sig.known).all()
            assert (sig.last_return <= sig.known).all()
            if "Six_asset_ERC" not in parent.name:
                for r in sig.itertuples():
                    assert r.selected.split("/") == list(ASSETS_BY_STAGE[r.phase])
        if "adv" in trades:
            assert (trades.shares <= .01*trades.adv+1e-8).all()
        pnl = pd.read_parquet(parent / "pnl.parquet").sum(axis=1)
        prev = ledger.nav.shift(1, fill_value=CAPITAL)
        np.testing.assert_allclose(prev+pnl-ledger.cost, ledger.nav, atol=1e-7, rtol=0)
        receipts[parent.relative_to(HERE).as_posix()] = dict(sessions=len(ledger), trades=len(trades),
                                                           ending=float(ledger.nav.iloc[-1]),
                                                           ledger_sha256=sha(path))
    return receipts


def outcomes(ledger, pnl, base):
    returns = ledger.nav.pct_change(fill_method=None)
    returns.iloc[0] = ledger.nav.iloc[0]/CAPITAL-1
    assert (returns > -1).all()
    top = returns.nlargest(max(1, round(len(returns)*.01)))
    keep = returns.drop(top.index)
    reduced = ((1.+keep).prod())**(365.25/(pd.Timestamp(returns.index[-1])-pd.Timestamp(base)).days)-1
    annual = (1+returns).groupby(returns.index.str[:4]).prod()-1
    monthly = (1+returns).groupby(returns.index.str[:7]).prod()-1
    asset = pnl.sum()
    return dict(cagr=net_annualized(ledger, base), annual_return=annual.to_dict(),
                monthly_return=monthly.to_dict(), pnl_by_asset=asset.to_dict(),
                top_asset=asset.idxmax(), top_asset_net_profit_share=float(asset.max()/(ledger.nav.iloc[-1]-CAPITAL)),
                top_1pct_dates=top.index.tolist(), top_1pct_removed_cagr=float(reduced),
                positive_day_share_from_top_1pct=float(top.sum()/returns.clip(lower=0).sum()))


class CashInPlaceOfAsset(Rotation):
    def __init__(self, name, returns, states, excluded):
        super().__init__(name, returns, states)
        self.excluded = KEYS.index(excluded)

    def targets(self, day, known, i, nav):
        weights = super().targets(day, known, i, nav)
        result = []
        for target in weights:
            if target is not None:
                target = target.copy()
                target[self.excluded] = 0
            result.append(target)
        return tuple(result)


def cost_curve(market, adv, hi, lo, historical, phases, mode, start, benchmark):
    rows = []

    def replay(bps):
        slip = bps/10000
        with apply_policy(lambda n, r, _: Rotation(n, r, phases), KEYS):
            if mode == "actual":
                result = engine.actual(FROZEN, market, adv, hi, lo, None, start, slip)
            else:
                result = engine.economic(FROZEN, historical, None, start, slip)
        cagr = net_annualized(result.ledger, result.base_date)
        rows.append(dict(mode=mode, start=start, bps=bps, cagr=cagr,
                         excess_vs_v12_base=cagr-benchmark, final_nav=float(result.ledger.nav.iloc[-1])))
        return cagr

    low = 9
    high = None
    for bps in (9, 20, 40, 80, 160, 320, 640, 1280):
        if replay(bps) <= benchmark:
            high = bps
            break
        low = bps
    if high is not None:
        while high-low > 1:
            middle = (low+high)//2
            if replay(middle) > benchmark:
                low = middle
            else:
                high = middle
    return rows, (low, high)


def main():
    DIR.mkdir(exist_ok=True)
    market, adv, hi, lo, historical, sessions, _ = market_inputs()
    phases, _ = build(sessions)
    timing = timing_check(phases, market.returns)
    account_checks = verify_accounts()
    metrics = pd.read_csv(HERE / "comparison.csv", encoding="utf-8-sig")
    attribution, costs = {}, []
    for mode, start in (("amount", LONG_START), ("actual", ACTUAL_START)):
        key = f"{mode}/{start}"
        folder = OUT / mode / start / f"{FROZEN}_9bp"
        ledger = pd.read_parquet(folder / "ledger.parquet")
        pnl = pd.read_parquet(folder / "pnl.parquet")
        pair = metrics[(metrics["mode"] == mode) & metrics.start.eq(start) & metrics.slip_bps.eq(9)]
        base = pair.loc[pair.strategy.eq(FROZEN), "base_date"].iloc[0]
        detail = outcomes(ledger, pnl, base)
        benchmark = float(pair.loc[pair.strategy.eq("V12"), "cagr"].iloc[0])
        top_asset = detail["top_asset"]
        with apply_policy(lambda n, r, _: CashInPlaceOfAsset(n, r, phases, top_asset), KEYS):
            if mode == "actual":
                removed = engine.actual(FROZEN, market, adv, hi, lo, None, start, MAIN_SLIP)
            else:
                removed = engine.economic(FROZEN, historical, None, start, MAIN_SLIP)
        detail["top_asset_budget_kept_as_cash_cagr"] = net_annualized(removed.ledger, removed.base_date)
        rows, bracket = cost_curve(market, adv, hi, lo, historical, phases, mode, start, benchmark)
        detail["excess_zero_slippage_bracket_bps"] = list(bracket)
        attribution[key] = detail
        costs.extend(rows)
        print(key, "original/top-day removal/top-asset cash",
              *(round(detail[k]*100, 2) for k in ("cagr", "top_1pct_removed_cagr",
                                                 "top_asset_budget_kept_as_cash_cagr")),
              "slippage bracket", bracket, flush=True)
    write(DIR / "checks.json", dict(accounts=account_checks, **timing, qmt_unadjusted=verify_qmt()))
    write(DIR / "attribution.json", attribution)
    pd.DataFrame(costs).sort_values(["mode", "bps"]).to_csv(
        DIR / "slippage_curve.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
