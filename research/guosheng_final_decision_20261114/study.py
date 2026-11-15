"""Fixed-condition Guosheng/V12 evidence audit on the latest archived inputs."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "guosheng_cross_review_20261114"),
               str(ROOT / "guosheng_parameter_robustness_20261113")]
import compare_evidence as prior
from study_inputs import Inputs, calendar_targets
from regimes import last_asof, state_path
from run_study import risk_parity
from v12.config import BASE as V12_BASE, KEYS as V12_KEYS

CONFIG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
CAPITAL = CONFIG["capital"]
START, END = CONFIG["dates"]["start"], CONFIG["dates"]["end"]
ENGINE = prior.engine
ORIGINAL_V12_POLICY = ENGINE.Policy
KEYS = prior.KEYS
ASSETS = prior.ASSETS_BY_STAGE
R8 = ROOT / "guosheng_proxy_substitution_20261114_r8"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
PERIODS = {
    "full": (START, END), "paper": (START, CONFIG["dates"]["paper_end"]),
    "2014-2016": (START, "2016-12-31"), "2017-2019": ("2017-01-01", "2019-12-31"),
    "2020-2022": ("2020-01-01", "2022-12-31"), "2023-2025": ("2023-01-01", "2025-12-31"),
    "2026": ("2026-01-01", END), "2026_H2": ("2026-07-01", END),
    **{str(y): (f"{y}-01-01", f"{y}-12-31") for y in range(2014, 2026)},
}


@dataclass(frozen=True)
class Case:
    name: str
    schedule: str = "monthly"
    risk_days: int = 63
    read_offset: int = 0
    delay: int = 0
    fallback: str = ""
    permanent: bool = False
    cutoff: str = ""
    equal: bool = False
    all_pool: bool = False


def write(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def load():
    config_hash = hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    receipt = HERE / "protocol.sha256"
    if receipt.exists():
        assert receipt.read_text().strip() == config_hash
    else:
        receipt.write_text(config_hash + "\n", encoding="ascii")
    local, views = prior.latest_local()
    _, _, state, events = prior.r6_states(local)
    inputs = Inputs.load()
    pmi = pd.read_csv(ROOT / "guosheng_original_reconstruction_20261114/results/pmi_dates_used.csv",
                      index_col="month", parse_dates=["available"])
    fallback = pd.read_parquet(R8 / "input/fallback_returns.parquet")
    exclusions, capacities = {}, {}
    for name, code in (("old_dividend", "510880.SH"), ("old_value", "510030.SH")):
        q = pd.read_parquet(R8 / "input" / f"fund_daily_{code}.parquet").set_index("trade_date").sort_index()
        q.index = pd.to_datetime(q.index).strftime("%Y-%m-%d")
        q = q.reindex(views.hist.index)
        valid = q.vol.gt(0)
        exclusions[name] = list(q.index[~(valid & valid.shift(1, fill_value=True))])
        capacities[name] = (.01 * q.amount.fillna(0).rolling(20).mean().shift() * 1000).where(valid, 0).fillna(0)
    full = pd.concat([views.hist, fallback], axis=1)
    full.attrs["excluded_risk_dates"] = views.hist.attrs.get("excluded_risk_dates", {}) | exclusions
    caps = pd.DataFrame(np.inf, index=full.index, columns=full.columns)
    for key, values in capacities.items():
        caps[key] = values
    return SimpleNamespace(local=local, views=views, state=state, events=events, inputs=inputs, pmi=pmi,
                           full=full, caps=caps)


def macro(data, key="", value=None):
    state = data.state.copy()
    dates = state.index
    if key == "drop":
        state[value] = 0
    elif key == "market_only":
        state[prior.CATS] = 0
    elif key == "extra_vote":
        eligible = dates >= pd.Timestamp("2025-08-18")
        state.loc[eligible, "reverse_repo"] = value
    elif key == "policy_expiry_days":
        for cat, ev in data.events.groupby("category"):
            daily = ev.groupby("usable_date").vote.agg(["min", "max"])
            vote = daily["min"].where(daily["min"].eq(daily["max"]), 0)
            last = last_asof(pd.Series(vote.index, index=vote.index), dates)
            state[cat] = last_asof(vote, dates).where(dates - last < pd.Timedelta(days=value), 0).fillna(0)
    elif key in ("money_smoothing_sessions", "money_change_days"):
        smooth = data.inputs.rates.rolling(value if key == "money_smoothing_sessions" else 30).mean()
        change = value if key == "money_change_days" else 90
        before = last_asof(smooth, dates - pd.Timedelta(days=change))
        now = last_asof(smooth, dates)
        state["market"] = np.sign(before.to_numpy() - now.to_numpy())
    elif key == "credit_change_months":
        c = data.inputs.credit
        state["credit"] = np.sign(last_asof(c.pulse.diff(value).set_axis(pd.DatetimeIndex(c.released)), dates))
    elif key in ("pmi_change_months", "pmi_smoothing_months"):
        p = data.pmi
        change = value if key == "pmi_change_months" else 3
        smooth = value if key == "pmi_smoothing_months" else 12
        diff = p[["manufacturing", "nonmanufacturing", "private"]].rolling(smooth).mean().pct_change(12).diff(change)
        votes = pd.DataFrame(np.where(diff > 0, 1, -1), index=p.index).sum(axis=1).where(diff.notna().all(axis=1))
        state["growth"] = np.sign(last_asof(votes.set_axis(pd.DatetimeIndex(p.available)), dates))
    elif key == "all_macro_extra_calendar_lag":
        shifted = state.copy()
        shifted.index += pd.Timedelta(days=value)
        state = shifted.reindex(dates, method="ffill")
    elif key:
        raise ValueError(key)
    state["money_score"] = (state[prior.CATS].sum(axis=1) + state.market) / 8
    state["money"] = np.sign(state.money_score)
    state["stage"] = state_path(state.money, state.credit, state.growth)
    assert state.loc[START:, "stage"].notna().all()
    return state


class Policy:
    def __init__(self, returns, risk, state, case, allowed=None, mapping=None):
        self.returns, self.risk, self.state, self.case = returns, risk, state, case
        self.keys = tuple(returns.columns)
        self.allowed = allowed or {}
        self.mapping = mapping or ASSETS
        self.excluded = risk.attrs.get("excluded_risk_dates", {})
        self.pending, self.previous_basket, self.last = None, None, -2
        self.receipts = []
        self.scheduled = calendar_targets(returns.index, f"D{case.read_offset + 1}", case.delay)
        self.first_month = None

    def targets(self, day, known, i, nav):
        if self.first_month is None:
            self.first_month = day[:7]
        macro_day = known
        phase = self.state.at[pd.Timestamp(known), "stage"]
        if pd.isna(phase):
            return None, None
        requested = tuple(self.mapping[int(phase)])
        monthly = day in self.scheduled and day[:7] != self.first_month
        switched = self.case.schedule in ("month_and_switch", "switch_only") and requested != self.previous_basket
        regular = monthly and self.case.schedule != "switch_only"
        if self.pending is None or regular or switched:
            if regular:
                _, macro_day = self.scheduled[day]
                phase = self.state.at[pd.Timestamp(macro_day), "stage"]
                requested = tuple(self.mapping[int(phase)])
            assets = list(KEYS) if self.case.all_pool else list(requested)
            if "fcf" in assets and self.case.fallback:
                permitted = self.risk.loc["2025-02-28":macro_day, "fcf"].dropna()
                if self.case.permanent or len(permitted) < self.case.risk_days:
                    assets[assets.index("fcf")] = self.case.fallback
            total = len(assets)
            selected = []
            for asset in assets:
                valid = self.risk.loc[self.allowed.get(asset, self.risk.index[0]):macro_day, asset].dropna()
                valid = valid.drop(index=self.excluded.get(asset, ()), errors="ignore")
                if len(valid) >= self.case.risk_days:
                    selected.append(asset)
            cash = (total - len(selected)) / total
            omitted = {d for k in selected for d in self.excluded.get(k, ())}
            train = self.risk.loc[:macro_day, selected].drop(index=sorted(omitted), errors="ignore")
            for asset in selected:
                if asset in self.allowed:
                    train.loc[train.index < self.allowed[asset], asset] = np.nan
            train = train.dropna().tail(self.case.risk_days)
            assert len(train) == self.case.risk_days
            weights = np.full(len(selected), 1 / len(selected)) if selected else np.array([])
            if len(selected) > 1 and not self.case.equal:
                covariance = train.cov().to_numpy()
                assert np.linalg.eigvalsh(covariance).min() > 0
                weights = risk_parity(covariance)
            weights = pd.Series(weights * (1 - cash), index=selected).reindex(self.keys, fill_value=0).to_numpy()
            self.pending, self.previous_basket, self.last = weights, requested, i
            row = self.state.loc[pd.Timestamp(macro_day)]
            assert str(row.credit_published)[:10] <= macro_day
            assert row.growth_available.date().isoformat() <= macro_day
            self.receipts.append(dict(date=day, known=macro_day, phase=int(phase), selected="/".join(selected),
                                      requested="/".join(requested), first_return=train.index[0],
                                      last_return=train.index[-1], cash_target=cash, **dict(zip(self.keys, weights))))
            return weights, weights
        if i == self.last + 1:
            return None, self.pending
        return None, None


@contextmanager
def capital(value):
    old = ENGINE.CAPITAL
    ENGINE.CAPITAL = value
    try:
        yield
    finally:
        ENGINE.CAPITAL = old


def cloned_actual(reuse_cash=False):
    source = inspect.getsource(ENGINE.actual)
    if reuse_cash:
        needle = "min(cash_before,book.cash)"
        assert source.count(needle) == 1
        source = source.replace(needle, "book.cash")
    env = dict(ENGINE.__dict__)
    exec(compile(source, str(HERE / "actual_execution_variant.py"), "exec"), env)
    return env["actual"]


def run(data, case, state=None, start=START, bp=9, mode="amount", returns=None, risk=None,
        limit=False, same_cash=False, adv_fraction=.01, mapping=None):
    state = data.state if state is None else state
    allowed = {}
    if case.cutoff == "fcf":
        allowed = {"fcf": "2024-08-16"}
    elif case.cutoff == "publications":
        allowed = {k: data.views.hist.index[data.views.hist.index.searchsorted(v, side="right")]
                   for k, v in prior.CFG["index_publication_control"]["dates"].items()}
    elif case.cutoff == "listed":
        allowed = data.views.first_actual_return
    frame = data.full if case.fallback else data.views.hist
    frame = frame if returns is None else returns
    if mode == "actual":
        q, adv, up, down = data.views.gs_quotes
        frame = q.returns if returns is None else returns
    risk = frame if risk is None else risk
    factory = lambda _n, actual_frame, _p: Policy(actual_frame, risk, state, case, allowed, mapping)
    with capital(CAPITAL), prior.apply_policy(factory, tuple(frame.columns)):
        if mode == "actual":
            fn = cloned_actual(same_cash)
            return fn(case.name, q, adv * adv_fraction / .01, up, down, None, start, bp / 10_000)
        if limit or same_cash:
            return economic(frame, factory(None, frame, None), start, bp / 10_000,
                            data.caps.reindex(columns=frame.columns) if limit else None, same_cash)
        return ENGINE.economic(case.name, frame, None, start, bp / 10_000)


def benchmark(data, start=START, bp=9, mode="amount", returns=None, same_cash=False, adv_fraction=.01):
    with capital(CAPITAL):
        if mode == "actual":
            q, adv, up, down = data.views.v12_quotes
            return cloned_actual(same_cash)("V12", q, adv * adv_fraction / .01, up, down, None, start, bp / 10_000)
        returns = data.views.benchmark if returns is None else returns
        if same_cash:
            return economic(returns, ORIGINAL_V12_POLICY("V12", returns, None), start, bp / 10_000, None, True, True)
        return ENGINE.economic("V12", returns, None, start, bp / 10_000)


def economic(returns, policy, start, slip, caps=None, reuse_cash=False, v12=False):
    """Same amount accounting, with explicit sale-funds and capacity controls."""
    calendar, keys = returns.index, tuple(returns.columns)
    dates = calendar[calendar >= start]
    values = np.zeros(len(keys))
    cash = float(CAPITAL)
    ledger, weights, pnl_rows, trades = [], [], [], []
    for i, day in enumerate(dates):
        j = calendar.get_loc(day)
        known = calendar[j - 1]
        nav = cash + values.sum()
        sell, buy = policy.targets(day, known, i, nav)
        def changes(target):
            if target is None:
                return np.zeros(len(keys))
            delta = target * nav - values
            return np.where(np.abs(delta) > .05 * values, delta, 0) if v12 else delta
        r = returns.iloc[j].to_numpy()
        pnl = values * r
        sales = -np.minimum(changes(sell), 0) * (1 + r)
        requested = np.where(sales == 0, np.maximum(changes(buy), 0) * (1 + r), 0)
        values += pnl
        sales = np.minimum(sales, values)
        if caps is not None:
            capacity = caps.loc[day].to_numpy()
            sales, requested = np.minimum(sales, capacity), np.minimum(requested, capacity)
        sale_cost = ENGINE.amount_fee(sales, slip)
        available = cash + float((sales - sale_cost).sum())
        # Very small fraction sales can cost more than proceeds under a minimum commission.
        if available < -1e-8:
            sales[sales < sale_cost] = 0
            sale_cost = ENGINE.amount_fee(sales, slip)
            available = cash + float((sales - sale_cost).sum())
        buys = ENGINE.affordable_amount(requested, available if reuse_cash else min(cash, available), slip)
        buy_cost = ENGINE.amount_fee(buys, slip)
        fees = float(sale_cost.sum() + buy_cost.sum())
        cash += float(sales.sum() - buys.sum() - fees)
        values += buys - sales
        nav = cash + values.sum()
        count = 0
        for side, amount, cost in (("sell", sales, sale_cost), ("buy", buys, buy_cost)):
            for k in np.flatnonzero(amount > 1e-9):
                trades.append(dict(date=day, known=known, asset=keys[k], side=side, notional=amount[k], cost=cost[k]))
                count += 1
        ledger.append(dict(date=day, nav=nav, cash=cash, receivable=0., cost=fees,
                           notional=float(sales.sum() + buys.sum()), trade_count=count))
        weights.append(np.r_[values / nav, cash / nav])
        pnl_rows.append(pnl)
    return ENGINE.pack(ledger, weights, pnl_rows, trades, policy, calendar[calendar.get_loc(start) - 1])


def metrics(account, lo=None, hi=END):
    l = account.ledger.loc[lo:hi]
    if l.empty:
        return None
    pos = account.ledger.index.get_loc(l.index[0])
    base = CAPITAL if pos == 0 else account.ledger.nav.iloc[pos - 1]
    date = account.base_date if pos == 0 else account.ledger.index[pos - 1]
    path = np.r_[base, l.nav.to_numpy()]
    ret = path[1:] / path[:-1] - 1
    years = (pd.Timestamp(l.index[-1]) - pd.Timestamp(date)).days / 365.25
    dd = 1 - path / np.maximum.accumulate(path)
    return dict(start=l.index[0], end=l.index[-1], base_date=date, days=len(l), years=years,
                cagr=(path[-1] / base) ** (1 / years) - 1, total=path[-1] / base - 1,
                vol=ret.std(ddof=1) * np.sqrt(252), mdd=dd.max(),
                mean_cash=(l.cash / l.nav).mean(), fees=l.cost.sum(),
                one_way_turnover=np.sum(l.notional.to_numpy() / path[:-1]) / years / 2,
                targets=len(account.signals.loc[account.signals.date.between(l.index[0], l.index[-1])]))


def record(account, name, rows, spans=PERIODS, save=True):
    if save:
        folder = OUT / "accounts" / name
        folder.mkdir(parents=True, exist_ok=True)
        for field in ("ledger", "weights", "pnl", "signals", "trades"):
            getattr(account, field).to_parquet(folder / f"{field}.parquet")
        (folder / "meta.json").write_text(json.dumps(dict(capital=CAPITAL, base_date=account.base_date)) + "\n")
    previous = account.ledger.nav.shift(1, fill_value=CAPITAL)
    assert (previous + account.pnl.sum(axis=1) - account.ledger.cost - account.ledger.nav).abs().max() < 1e-6
    assert account.signals.known.lt(account.signals.date).all()
    for period, (lo, hi) in spans.items():
        m = metrics(account, lo, hi)
        if m is not None:
            rows.append(dict(case=name, period=period, **m))


def primary_cases():
    return [Case("monthly"), Case("month_and_switch", "month_and_switch"), Case("switch_only", "switch_only"),
            Case("dividend_substitute", fallback="old_dividend"), Case("value_substitute", fallback="old_value"),
            Case("dividend_permanent", fallback="old_dividend", permanent=True),
            Case("value_permanent", fallback="old_value", permanent=True),
            Case("fcf_publication_cash", cutoff="fcf"), Case("all_publications_cash", cutoff="publications"),
            Case("all_listing_cash", cutoff="listed"), Case("equal_basket", equal=True),
            Case("all_asset_erc", all_pool=True), Case("all_asset_equal", equal=True, all_pool=True)]


def main():
    data = load()
    rows = []
    for start in (START, CONFIG["dates"]["restart"]):
        for bp in (9, 20):
            record(benchmark(data, start, bp), f"V12_{start}_{bp}bp", rows)
            for case in primary_cases():
                a = run(data, case, start=start, bp=bp, limit=bool(case.fallback))
                record(a, f"{case.name}_{start}_{bp}bp", rows)
        print(f"Primary controls finished at {start}", flush=True)
    base = run(data, Case("monthly"))
    # Original reference helper uses current engine capital.
    with capital(CAPITAL):
        reference = prior.run(data.views, "amount", START, prior.MAIN, state=data.state)
    pd.testing.assert_frame_equal(base.ledger, reference.ledger)
    for case in (Case("monthly"), Case("month_and_switch", "month_and_switch")):
        for start in (CONFIG["dates"]["common_actual_early"], CONFIG["dates"]["actual_start"]):
            for bp in (9, 20, 50):
                for cash in (False, True):
                    for label, a in ((case.name, run(data, case, start=start, bp=bp, mode="actual", same_cash=cash)),
                                     ("V12", benchmark(data, start, bp, mode="actual", same_cash=cash))):
                        name = f"actual_{label}_{start}_{bp}bp_cash{int(cash)}"
                        if any(row["case"] == name for row in rows):
                            continue
                        record(a, name, rows)
            if start == CONFIG["dates"]["actual_start"]:
                for label, a in ((case.name, run(data, case, start=start, mode="actual", adv_fraction=.001)),
                                 ("V12", benchmark(data, start, mode="actual", adv_fraction=.001))):
                    name = f"actual_{label}_{start}_9bp_adv001"
                    if not any(row["case"] == name for row in rows):
                        record(a, name, rows)
        print(f"Actual price controls finished for {case.name}", flush=True)
    write(pd.DataFrame(rows), "main_comparison.csv")
    print("Saved main comparison", flush=True)


if __name__ == "__main__":
    main()
