"""Reconcile ETF P&L and replay fixed allocation controls on the archived accounts."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIOR = ROOT / "guosheng_cross_review_20261114"
R8 = ROOT / "guosheng_proxy_substitution_20261114_r8"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))

from compare_evidence import (CAPITAL, CFG, END, KEYS, MAIN, PAPER, REAL_START, SLIP, START,
                              latest_local, mkstate, r6_states, run_published)
from replay_long import apply_policy, engine, log_attribution, run
from regimes import state_path
from run_original import LOOKBACK, trailing
from run_study import ASSETS_BY_STAGE, risk_parity


BASKET = {phase: frozenset(assets) for phase, assets in ASSETS_BY_STAGE.items()}
PAST_CUTOFFS = CFG["index_publication_control"]["dates"]
HALF_YEAR = ("2026-07-01", END)
COVID = ("2020-02-03", "2020-03-31")
AMOUNT_WINDOWS = {
    "paper": PAPER, "full": (START, END), "2014-2019": (START, "2019-12-31"),
    "2020-2022": ("2020-01-01", "2022-12-31"), "2023-2025": ("2023-01-01", "2025-12-31"),
    "2020_Covid": COVID, "2025_through_Oct": ("2025-01-01", PAPER[1]),
    "2026_YTD": ("2026-01-01", END), "2026_H1": ("2026-01-01", "2026-06-30"), "2026_H2": HALF_YEAR,
    **{str(year): (f"{year}-01-01", f"{year}-12-31") for year in range(2014, 2026)},
}
QUOTE_WINDOWS = {"recent": (REAL_START, END), "2026_YTD": AMOUNT_WINDOWS["2026_YTD"],
                 "2026_H1": AMOUNT_WINDOWS["2026_H1"], "2026_H2": HALF_YEAR}
CATEGORY = {
    "growth": "China equity", "fcf": "China equity", "dividend": "China equity",
    "old_value": "China equity", "old_dividend": "China equity", "csi300": "China equity",
    "csi1000": "China equity", "divlow": "China equity", "gold": "Gold",
    "metals": "Metals", "energy": "Other commodity", "soy": "Other commodity",
    "bond": "CGB", "bond10": "CGB", "nasdaq": "US equity", "sp500": "US equity", "fee": "Trading cost",
}
REFERENCE_CASES = {"V12": "V12", "switch": "month_and_switch", "monthly": "monthly",
                   "switch_pit": "month_and_switch_three_cut", "monthly_pit": "monthly_three_cut"}


class MonthlyControl:
    """Use the same lagged training and next-session cash refill for all four controls."""

    def __init__(self, history, state, use_basket, use_erc, allowed, matching_cash):
        self.state = state
        self.use_basket = use_basket
        self.use_erc = use_erc
        self.allowed = allowed
        self.matching_cash = matching_cash
        self.training = history.copy()
        self.exclusions = history.attrs.get("excluded_risk_dates", {})
        for asset, date in allowed.items():
            self.training.loc[self.training.index < date, asset] = np.nan
        self.receipts = []
        self.pending = None
        self.last_trade = -2

    def targets(self, day, known, i, nav):
        if self.pending is not None and day[:7] == known[:7]:
            if i == self.last_trade + 1:
                return None, self.pending
            return None, None
        phase = int(self.state.at[pd.Timestamp(known), "stage"])
        selected = ASSETS_BY_STAGE[phase]
        available = {asset for asset in KEYS if asset not in self.allowed or
                     len(trailing(self.training, self.exclusions, known, (asset,))) == LOOKBACK}
        pool = selected if self.use_basket else KEYS
        active = tuple(asset for asset in pool if asset in available)
        cash = (len(selected) - len(set(selected) & available)) / len(selected) if self.matching_cash else 0.
        train = trailing(self.training, self.exclusions, known, active)
        assert active and len(train) == LOOKBACK and train.index[-1] <= known
        for asset in self.allowed.keys() & set(active):
            assert train.index[0] >= self.allowed[asset]
        fractions = np.full(len(active), 1 / len(active))
        if self.use_erc and len(active) > 1:
            fractions = risk_parity(train.cov().to_numpy())
        self.pending = pd.Series(fractions * (1 - cash), index=active).reindex(KEYS, fill_value=0).to_numpy()
        self.last_trade = i
        self.receipts.append(dict(date=day, known=known, phase=phase, selected="/".join(active),
                                  cash_target=cash, **dict(zip(KEYS, self.pending))))
        return self.pending, self.pending


class ScheduledRiskParity(MonthlyControl):
    """Reweight the full pool on the basket strategy's already dated target calendar."""

    def __init__(self, history, state, dates):
        super().__init__(history, state, False, True, {}, False)
        self.dates = frozenset(dates)

    def targets(self, day, known, i, nav):
        if day in self.dates:
            self.pending = None
            return super().targets(day, known, i, nav)
        if i == self.last_trade + 1:
            return None, self.pending
        return None, None


def monthly_control(views, mode, state, use_basket, use_erc, allowed, matching_cash=True):
    factory = lambda name, history, prices: MonthlyControl(history, state, use_basket, use_erc, allowed, matching_cash)
    return run_control(views, mode, factory)


def run_control(views, mode, factory):
    with apply_policy(factory, KEYS):
        if mode == "actual":
            quote, adv, up, down = views.gs_quotes
            return engine.actual(MAIN, quote, adv, up, down, None, REAL_START, SLIP)
        return engine.economic(MAIN, views.hist, None, START, SLIP)


def load_accounts(views, state, events):
    allowed = {asset: views.hist.index[views.hist.index.searchsorted(day, side="right")]
               for asset, day in PAST_CUTOFFS.items()}
    market = mkstate(state, events.iloc[:0])
    cases = {}
    for mode, start in (("amount", START), ("actual", REAL_START)):
        cases[mode] = {
            "switch": run(views, mode, start, "GS_stage_switch", state=state),
            "monthly": run(views, mode, start, MAIN, state=state),
            "V12": run(views, mode, start, "V12", SLIP),
            "DR007_switch": run(views, mode, start, "GS_stage_switch", state=market),
            "pool_equal": monthly_control(views, mode, state, False, False, {}),
            "pool_erc": monthly_control(views, mode, state, False, True, {}),
            "basket_equal": monthly_control(views, mode, state, True, False, {}),
        }
        schedule = cases[mode]["switch"].signals.date
        factory = lambda name, history, prices: ScheduledRiskParity(history, state, schedule)
        cases[mode]["pool_erc_same_days"] = run_control(views, mode, factory)
        assert cases[mode]["pool_erc_same_days"].signals.date.equals(schedule)
        independent_monthly = monthly_control(views, mode, state, True, True, {})
        pd.testing.assert_frame_equal(cases[mode]["monthly"].ledger, independent_monthly.ledger, check_exact=True)
    cases["amount"].update({
        "switch_pit": run_published(views, state, allowed, True, True),
        "monthly_pit": run_published(views, state, allowed, True, False),
        "switch_listed": run_published(views, state, views.first_actual_return, True, True),
        "DR007_switch_pit": run_published(views, market, allowed, True, True),
        "pool_equal_pit": monthly_control(views, "amount", state, False, False, allowed),
        "pool_erc_pit": monthly_control(views, "amount", state, False, True, allowed),
        "pool_erc_pit_full_investment": monthly_control(views, "amount", state, False, True, allowed, matching_cash=False),
        "basket_equal_pit": monthly_control(views, "amount", state, True, False, allowed),
    })
    independent_strict = monthly_control(views, "amount", state, True, True, allowed)
    pd.testing.assert_frame_equal(cases["amount"]["monthly_pit"].ledger, independent_strict.ledger, check_exact=True)
    actual = cases["actual"]
    for account in cases.values():
        for name, a in account.items():
            assert a.ledger.index.equals(account["V12"].ledger.index), name
    for key in ("ledger", "trades"):
        pd.testing.assert_frame_equal(getattr(actual["switch"], key), getattr(actual["DR007_switch"], key), check_exact=True)
    return cases


def span(daily, dates):
    return daily.loc[slice(*dates)]


def performance(a, dates):
    ledger = span(a.ledger, dates)
    prev = a.ledger.nav.shift(1, fill_value=CAPITAL)
    position = a.ledger.index.get_loc(ledger.index[0])
    base_date = a.ledger.index[position - 1] if position else a.base_date
    years = (pd.Timestamp(ledger.index[-1]) - pd.Timestamp(base_date)).days / 365.25
    base = prev.loc[ledger.index[0]]
    path = np.r_[base, ledger.nav.to_numpy()]
    rets = path[1:] / path[:-1] - 1
    metric = dict(start=ledger.index[0], end=ledger.index[-1], base_date=base_date, days=len(ledger), years=years,
                  total=path[-1] / base - 1, cagr=(path[-1] / base) ** (1 / years) - 1,
                  log_per_year=np.log(path[-1] / base) / years, vol=rets.std(ddof=1) * np.sqrt(252),
                  mdd=(1 - path / np.maximum.accumulate(path)).max(), mean_cash=(ledger.cash / ledger.nav).mean(),
                  fees=ledger.cost.sum(), orders=ledger.trade_count.sum(),
                  one_way_turnover=(ledger.notional / prev.loc[ledger.index]).sum() / (2 * years))
    return metric


def verify_prior(cases):
    reference = pd.read_csv(PRIOR / "evidence" / "matched_replay.csv")
    for mode, accounts in cases.items():
        windows = AMOUNT_WINDOWS if mode == "amount" else QUOTE_WINDOWS
        for name, prior_name in REFERENCE_CASES.items():
            if name not in accounts:
                continue
            control = reference.loc[reference.version.eq("combined") & reference["mode"].eq(mode)
                                    & reference.case.eq(prior_name)].set_index("period")
            for period in control.index.intersection(windows):
                m = performance(accounts[name], windows[period])
                columns = ["total", "cagr", "vol", "mdd"]
                np.testing.assert_allclose([m[c] for c in columns], control.loc[period, columns].astype(float),
                                           atol=2e-12, rtol=0)
    listed = pd.read_csv(PRIOR / "evidence" / "listed_etfs_only_control.csv")
    stored = listed.loc[listed.case.eq("month_and_switch") & listed.period.eq("paper")].iloc[0]
    measured = performance(cases["amount"]["switch_listed"], PAPER)
    assert abs(stored.cagr - measured["cagr"]) < 2e-12
    legacy = pd.read_parquet(ROOT / "guosheng_original_reconstruction_20261114" / "results" / "daily_ledger.parquet")
    for name, old in (("switch", "rebuilt_month_switch"), ("monthly", "rebuilt_monthly"), ("V12", "V12")):
        ledger = legacy.loc[legacy["mode"].eq("actual") & legacy["case"].eq(old)].set_index("date")
        expected = cases["actual"][name].ledger
        pd.testing.assert_frame_equal(ledger[expected.columns], expected, check_exact=True)


def prior_target_phase(account, mode):
    s = account.signals.set_index("date")
    assigned = s.phase.reindex(account.ledger.index).ffill().shift(1, fill_value=0).astype(int)
    if mode == "actual":
        changed = s.selected.ne(s.selected.shift()).reindex(assigned.index, fill_value=False)
        assigned = assigned.mask(changed | changed.shift(1, fill_value=False), 0)
    return assigned


def asset_attribution(cases, views):
    asset_rows, daily_rows, stage_rows, exposure_rows, drawdown_rows = [], [], [], [], []
    relative_rows = []
    for mode, accounts in cases.items():
        windows = AMOUNT_WINDOWS if mode == "amount" else QUOTE_WINDOWS
        r12, _, l12 = log_attribution(accounts["V12"])
        names = ("switch", "switch_pit", "switch_listed", "monthly", "V12") if mode == "amount" else ("switch", "monthly", "V12")
        for name in names:
            a = accounts[name]
            ret, raw, log = log_attribution(a)
            previous = a.weights.shift(1, fill_value=0)
            previous.loc[previous.index[0], "cash"] = 1
            phase = prior_target_phase(a, mode) if name != "V12" else None
            pool_name = "pool_erc_pit" if name == "switch_pit" else "pool_erc"
            pool_return, _, _ = log_attribution(accounts[pool_name])
            if name == "switch_listed":
                pool_return = None
            if mode == "amount":
                prices = views.benchmark if name == "V12" else views.hist
                aligned = prices.loc[ret.index, list(a.pnl.columns)]
                np.testing.assert_allclose(previous[a.pnl.columns] * aligned, raw[a.pnl.columns], atol=4e-15, rtol=0)
            detail = log.add_prefix("log_").assign(mode=mode, case=name, nav=a.ledger.nav, daily_return=ret,
                                                    cost=a.ledger.cost, cash_weight=a.weights.cash)
            if phase is not None:
                detail["opening_target_phase"] = phase
            daily_rows.append(detail.rename_axis("date").reset_index())
            for period, dates in windows.items():
                rows = span(log, dates)
                m = performance(a, dates)
                log_total = rows.to_numpy().sum()
                np.testing.assert_allclose(log_total, np.log1p(m["total"]), atol=2e-13, rtol=0)
                annual_scale = m["cagr"] / log_total if log_total != 0 else 1 / m["years"]
                arithmetic_scale = m["total"] / log_total if log_total != 0 else 1
                for key, contribution in rows.sum().items():
                    needs_proxy_split = mode == "amount" and name != "V12" and key in views.first_actual_return
                    proxy_dates = rows.index < views.first_actual_return[key] if needs_proxy_split else np.zeros(len(rows), dtype=bool)
                    weight = previous.loc[rows.index, key].mean() if key in previous else 0
                    asset_rows.append(dict(mode=mode, case=name, period=period, asset=key, category=CATEGORY[key],
                                           annual_contribution=contribution * annual_scale,
                                           total_contribution=contribution * arithmetic_scale,
                                           log_contribution=contribution,
                                           share_net_log=contribution / log_total if abs(log_total) > 1e-10 else np.nan,
                                           proxy_log=rows.loc[proxy_dates, key].sum() if name != "V12" else np.nan,
                                           proxy_sessions=int(proxy_dates.sum()) if needs_proxy_split else 0,
                                           mean_prior_weight=weight,
                                           pnl=-a.ledger.loc[rows.index, "cost"].sum() if key == "fee"
                                           else a.pnl.loc[rows.index, key].sum()))
                if name != "V12":
                    relative_rows.extend(relative_attribution(mode, name, period, m, rows, l12.loc[rows.index],
                                                              performance(accounts["V12"], dates)))
                    stage_rows.extend(stage_summary(mode, name, period, rows, phase.loc[rows.index],
                                                    r12.loc[rows.index], pool_return, previous))
                if mode == "amount":
                    exposure_rows.extend(arithmetic_components(name, period, raw.loc[rows.index],
                                                               previous.loc[rows.index], aligned.loc[rows.index]))
            drawdown_rows.extend(drawdown_summary(mode, name, a, log, windows))
    write(pd.DataFrame(asset_rows), "asset_contributions.csv")
    write(pd.DataFrame(relative_rows), "relative_contributions.csv")
    write(pd.DataFrame(stage_rows), "stage_contributions.csv")
    write(pd.DataFrame(exposure_rows), "exposure_and_timing.csv")
    write(pd.DataFrame(drawdown_rows), "drawdown_contributions.csv")
    pd.concat(daily_rows, ignore_index=True).to_parquet(HERE / "daily_attribution.parquet", index=False)


def relative_attribution(mode, name, period, m, gs, v12, ref):
    log_groups = defaultdict(float)
    for sign, contributions in ((1, gs.sum()), (-1, v12.sum())):
        for asset, log in contributions.items():
            log_groups[CATEGORY[asset]] += sign * log
    total = sum(log_groups.values())
    assert np.isclose(total, np.log1p(m["total"]) - np.log1p(ref["total"]), rtol=0, atol=5e-13)
    scale = (m["cagr"] - ref["cagr"]) / total if abs(total) > 1e-12 else (1 + ref["cagr"]) / m["years"]
    total_scale = (m["total"] - ref["total"]) / total if abs(total) > 1e-12 else (1 + ref["total"])
    for category, log in log_groups.items():
        yield dict(mode=mode, case=name, period=period, category=category, relative_log=log,
                   cagr_gap_contribution=log * scale, cumulative_gap_contribution=log * total_scale)


def stage_summary(mode, name, period, log, phase, r12, pool, previous):
    grouped = log.groupby(phase).sum()
    assert np.isclose(grouped.to_numpy().sum(), log.to_numpy().sum(), rtol=0, atol=1e-12)
    for state, row in grouped.iterrows():
        days = phase.index[phase.eq(state)]
        baseline = np.log1p(r12.loc[days]).sum()
        static = np.log1p(pool.loc[days]).sum() if pool is not None else np.nan
        names = (row.drop("fee").abs().sort_values(ascending=False).index.tolist())[:2]
        yield dict(mode=mode, case=name, period=period, phase=state, days=len(days), net_log=row.sum(),
                   relative_log_v12=row.sum() - baseline, relative_log_pool_erc=row.sum() - static,
                   avg_prior_cash=previous.loc[days, "cash"].mean(), strongest_asset=names[0],
                   strongest_log=row[names[0]], second_asset=names[1], second_log=row[names[1]])


def arithmetic_components(name, period, raw, held, returns):
    weights = held[returns.columns]
    ordinary = weights.mean() * returns.mean()
    timing = (weights * returns).mean() - ordinary
    np.testing.assert_allclose(ordinary + timing, raw[returns.columns].mean(), atol=5e-16, rtol=0)
    for key in returns:
        yield dict(case=name, period=period, asset=key, exposure_arith=252 * ordinary[key],
                   timing_arith=252 * timing[key], fee_arith=0, net_arith=252 * raw[key].mean())
    charge = 252 * raw.fee.mean()
    yield dict(case=name, period=period, asset="fee", exposure_arith=0, timing_arith=0, fee_arith=charge, net_arith=charge)


def drawdown_summary(mode, name, account, log, windows):
    selected = {"paper", "2020_Covid", "2026_H2", "recent"}
    for period, dates in windows.items():
        if period not in selected:
            continue
        ledger = span(account.ledger, dates)
        metrics = performance(account, dates)
        base = account.ledger.nav.get(metrics["base_date"], CAPITAL)
        path = np.r_[base, ledger.nav.to_numpy()]
        trough = np.argmax(1 - path / np.maximum.accumulate(path))
        peak = path[:trough + 1].argmax()
        part = log.loc[ledger.index[peak:trough]].sum()
        np.testing.assert_allclose(part.sum(), np.log1p(-metrics["mdd"]), atol=2e-13, rtol=0)
        stages = [metrics["base_date"], *ledger.index]
        for asset, contribution in part.items():
            yield dict(mode=mode, case=name, period=period, peak=stages[peak], trough=stages[trough],
                       max_drawdown=metrics["mdd"], asset=asset, log_contribution=contribution)


def compare_controls(cases):
    results, changes = [], []
    for mode, accounts in cases.items():
        windows = AMOUNT_WINDOWS if mode == "amount" else QUOTE_WINDOWS
        for period, dates in windows.items():
            metrics = {name: performance(a, dates) for name, a in accounts.items()}
            results.extend(dict(mode=mode, period=period, case=name, **m) for name, m in metrics.items())
            for t in ("", "_pit") if mode == "amount" else ("",):
                pe, pr, be, br = [metrics[name + t]["log_per_year"] for name in
                                  ("pool_equal", "pool_erc", "basket_equal", "monthly")]
                overlap = br - be - pr + pe
                stage = ((be - pe) + (br - pr)) / 2
                weights = ((pr - pe) + (br - be)) / 2
                switch = metrics["switch" + t]["log_per_year"] - br
                policy = metrics["switch" + t]["log_per_year"] - metrics["DR007_switch" + t]["log_per_year"]
                assert abs(stage + weights + switch + pe - metrics["switch" + t]["log_per_year"]) < 1e-12
                changes.append(dict(mode=mode, period=period, history="publication_cutoffs" if t else "as_archived",
                                    base_annual_log=pe, phase_basket=stage, risk_weight=weights, in_month_switch=switch,
                                    interaction=overlap, policy_votes_vs_market_only=policy,
                                    active_annual_log=metrics["switch" + t]["log_per_year"]))
    write(pd.DataFrame(results), "account_metrics.csv")
    write(pd.DataFrame(changes), "rule_contributions.csv")


def block_intervals(cases):
    accounts = cases["amount"]
    rows = []
    for n, control in (("switch", "pool_erc"), ("switch_pit", "pool_erc_pit"), ("switch_pit", "V12")):
        left, right = [np.log1p(span(log_attribution(accounts[name])[0], PAPER)) for name in (n, control)]
        delta = (left - right).groupby(pd.to_datetime(left.index).to_period("M")).sum()
        rng = np.random.default_rng(20261114)
        for length in (6, 12):
            nblocks = int(np.ceil(len(delta) / length))
            blocks = (rng.integers(0, len(delta), size=(5000, nblocks, 1)) + np.arange(length)) % len(delta)
            draw = (delta.to_numpy()[blocks].reshape(5000, -1)[:, :len(delta)]).mean(axis=1) * 12
            rows.append(dict(case=n, control=control, months=len(delta), block_months=length,
                             observed_annual_log=delta.mean() * 12, lower=np.quantile(draw, .05),
                             median=np.median(draw), upper=np.quantile(draw, .95), paths=len(draw)))
    write(pd.DataFrame(rows), "historical_block_intervals.csv")


def signal_diagnostics(views, state, cases):
    """Check each direction signal while keeping the other two reported directions."""
    amount = cases["amount"]
    quote = cases["actual"]
    allowed = {asset: views.hist.index[views.hist.index.searchsorted(day, side="right")]
               for asset, day in PAST_CUTOFFS.items()}
    factors = ("money", "credit", "growth")
    rows = []
    for factor in factors:
        for level in (-1, 1):
            panel = state.copy()
            panel.loc[panel[factor].notna(), factor] = level
            panel["stage"] = state_path(panel.money, panel.credit, panel.growth)
            assert panel.loc["2013-12-31":, "stage"].notna().all()
            tests = (("amount", "switch", run(views, "amount", START, "GS_stage_switch", state=panel), {"paper": PAPER}),
                     ("amount", "switch_pit", run_published(views, panel, allowed, True, True), {"paper": PAPER}),
                     ("actual", "switch", run(views, "actual", REAL_START, "GS_stage_switch", state=panel), QUOTE_WINDOWS))
            for mode, base, a, windows in tests:
                original = (amount if mode == "amount" else quote)[base]
                for period, dates in windows.items():
                    m, prev = performance(a, dates), performance(original, dates)
                    rows.append(dict(mode=mode, base=base, period=period, fixed_factor=factor, fixed_to=level,
                                     cagr=m["cagr"], total=m["total"], mdd=m["mdd"],
                                     original_cagr=prev["cagr"], original_total=prev["total"],
                                     original_mdd=prev["mdd"], mean_cash=m["mean_cash"]))
    assert len(rows) == 2 * len(factors) * (2 + len(QUOTE_WINDOWS))
    write(pd.DataFrame(rows), "signal_direction_controls.csv")


def r8_attribution():
    outputs = R8 / "results"
    table = pd.concat([pd.read_csv(outputs / name) for name in ("comparison.csv", "capacity_comparison.csv",
                                                               "combined_comparison.csv")], ignore_index=True)
    rows = []
    cases = [("substitute", 9, n) for n in ("original_backfill", "cash_ETF", "dividend_to_FCF", "value_to_FCF", "V12")]
    cases += [("capacity", 9, n) for n in ("dividend_to_FCF", "value_to_FCF")]
    cases += [("combined", 20, n) for n in ("dividend_adverse_drift", "value_adverse_drift", "V12_favorable_drift")]
    for group, bp, name in cases:
        folder = outputs / "accounts" / f"{group}_{START}_{bp}bp_{name}"
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        ledger, pnl = (pd.read_parquet(folder / f"{field}.parquet") for field in ("ledger", "pnl"))
        prev = ledger.nav.shift(1, fill_value=meta["capital"])
        returns = ledger.nav / prev - 1
        pnl = pnl.assign(fee=-ledger.cost).div(prev, axis=0)
        np.testing.assert_allclose(pnl.sum(axis=1), returns, atol=3e-15, rtol=0)
        scale = np.divide(np.log1p(returns), returns, out=np.ones(len(returns)), where=returns.ne(0))
        log = pnl.mul(scale, axis=0).sum()
        duration = (pd.Timestamp(END) - pd.Timestamp(meta["base_date"])).days / 365.25
        actual = np.expm1(log.sum() / duration)
        control = table.loc[table.label.eq(name) & table.group.eq(group) & table.bp.eq(bp)
                            & table.account_start.eq(START) & table.window.eq("full"), "cagr"].item()
        assert abs(actual - control) < 2e-12
        for asset, contribution in log.items():
            rows.append(dict(group=group, case=name, bp=bp, asset=asset, yearly_contribution=contribution * actual / log.sum(),
                             log_contribution=contribution, cagr=actual, capital=meta["capital"]))
    write(pd.DataFrame(rows), "r8_replacement_contributions.csv")


def write(frame, name):
    frame.to_csv(HERE / name, index=False, encoding="utf-8-sig")


def main():
    def no_network(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Attribution only uses archived inputs")

    sys.addaudithook(no_network)
    local, views = latest_local()
    _, _, state, events = r6_states(local)
    cases = load_accounts(views, state, events)
    verify_prior(cases)
    print(f"Replayed {sum(map(len, cases.values()))} accounts and reconciled previous metrics", flush=True)
    compare_controls(cases)
    asset_attribution(cases, views)
    block_intervals(cases)
    signal_diagnostics(views, state, cases)
    r8_attribution()
    sources = [HERE / "attribute.py", PRIOR / "compare_evidence.py", PRIOR / "evidence" / "matched_replay.csv",
               ROOT / "guosheng_policy_completion_20261114_r6" / "v2" / "events_used.csv",
               ROOT / "guosheng_original_reconstruction_20261114" / "results" / "input_receipt.json",
               R8 / "results" / "comparison.csv", R8 / "results" / "manifest.json"]
    (HERE / "manifest.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in sources}, indent=2) + "\n", encoding="utf-8")
    print("Daily P&L, costs, exposure, stages, prior trials and all log contributions reconciled", flush=True)


if __name__ == "__main__":
    main()
