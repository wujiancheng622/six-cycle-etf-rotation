"""Offline monthly replay extracted from the archived Guosheng research baseline."""
from dataclasses import dataclass
from types import SimpleNamespace
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAPITAL = 500_000
START = '2014-01-02'
END = '2026-11-12'
KEYS = ('growth', 'gold', 'metals', 'fcf', 'bond', 'dividend')

@dataclass(frozen=True)
class Case:
    schedule: str = 'monthly'
    read_offset: int = 0
    delay: int = 0
    risk_days: int = 63
    all_pool: bool = False
    fallback: str = ''
    permanent: bool = False
    equal: bool = False

STAGE_MAP = {
    (1, 1, -1): 1, (1, 1, 1): 2, (-1, 1, 1): 3,
    (-1, -1, 1): 4, (-1, -1, -1): 5, (1, -1, -1): 6,
}
ASSETS_BY_STAGE = {
    1: ("growth", "gold"),
    2: ("growth", "metals"),
    3: ("fcf", "metals"),
    4: ("fcf", "bond", "metals"),
    5: ("dividend", "bond", "gold"),
    6: ("dividend", "bond", "gold"),
}
ASSETS = ASSETS_BY_STAGE


def state_path(money, credit, growth):
    states = []
    stage = None
    for directions in zip(money, credit, growth):
        if np.isfinite(directions).all():
            candidate = STAGE_MAP.get(tuple(directions))
            if candidate is not None:
                stage = candidate
        states.append(stage)
    return pd.Series(states, index=money.index, dtype="Int64")


def calendar_targets(days, schedule, delay=0):
    """Map an execution date to its signal target date and prior market date."""
    calendar = pd.Series(days,index=days)
    timestamp = pd.to_datetime(days)
    monthly = calendar.groupby(timestamp.to_period("M"),sort=True)
    if schedule.startswith("D"):
        ordinal = int(schedule[1:])
        selected = [g.iloc[ordinal-1] if len(g)>=ordinal else g.iloc[-1]
                    for _,g in monthly if len(g)>=ordinal or g.index[-1][:7]!=END[:7]]
    elif schedule=="ME":
        selected = [g.iloc[-1] for _,g in monthly if g.index[-1][:7]!=END[:7]]
    elif schedule=="W":
        selected = calendar.groupby(timestamp.to_period("W-SUN"),sort=True).first().tolist()
    elif schedule=="Q":
        selected = calendar.groupby(timestamp.to_period("Q"),sort=True).first().tolist()
    else:
        raise ValueError(schedule)
    out={}
    for d in selected:
        j=days.get_loc(d)
        if j>0 and j+delay<len(days):
            out[days[j+delay]]=(d,days[j-1])
    return out


def risk_parity(cov):
    scaled = cov / np.max(np.diag(cov))
    count = len(cov)
    budget = 1 / count
    diagonal = np.diag(scaled)
    x = np.sqrt(budget / diagonal)
    for _ in range(4000):
        for i in range(count):
            outside = scaled[i] @ x - diagonal[i] * x[i]
            root = np.sqrt(outside**2 + 4 * diagonal[i] * budget)
            x[i] = (root - outside) / (2 * diagonal[i])
        if np.max(np.abs(x * (scaled @ x) - budget)) < 1e-11:
            break
    else:
        raise RuntimeError("Risk parity did not converge")
    return x / x.sum()


class Policy:
    def __init__(self, returns, risk, state, case, allowed=None, mapping=None):
        self.returns, self.risk, self.state, self.case = returns, risk, state, case
        self.keys = tuple(returns.columns)
        self.allowed = allowed or {}
        self.mapping = mapping or ASSETS
        self.excluded = risk.attrs.get("excluded_risk_dates", {})
        self.pending, self.previous_basket, self.last = None, None, -2
        self.receipts = []
        clock = "W" if case.schedule == "weekly" else f"D{case.read_offset + 1}"
        self.scheduled = calendar_targets(returns.index, clock, case.delay)
        self.first_month = None

    def targets(self, day, known, i, nav):
        if self.first_month is None:
            self.first_month = day[:7]
        macro_day = known
        phase = self.state.at[pd.Timestamp(known), "stage"]
        if pd.isna(phase):
            return None, None
        requested = tuple(self.mapping[int(phase)])
        switched = self.case.schedule in ("month_and_switch", "switch_only") and requested != self.previous_basket
        regular = (day in self.scheduled and self.case.schedule != "switch_only"
                   and (self.case.schedule == "weekly" or day[:7] != self.first_month))
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


def pack(ledger, weights, pnl, trades, policy, base_date):
    result = SimpleNamespace(ledger=pd.DataFrame(ledger).set_index('date'),
                             weights=pd.DataFrame(weights, columns=[*KEYS,'cash']),
                             pnl=pd.DataFrame(pnl, columns=KEYS), trades=pd.DataFrame(trades),
                             signals=pd.DataFrame(policy.receipts), base_date=base_date)
    result.weights.index = result.pnl.index = result.ledger.index
    previous = np.r_[CAPITAL, result.ledger.nav.iloc[:-1]]
    gross = result.pnl.sum(axis=1).to_numpy()
    np.testing.assert_allclose(previous+gross-result.ledger.cost, result.ledger.nav, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1, rtol=0, atol=1e-9)
    assert result.ledger.cash.min() >= -1e-7 and result.weights.min().min() >= -1e-9
    assert (result.signals.known < result.signals.date).all()
    return result


def amount_fee(value, slip):
    return np.where(value>1e-9, np.maximum(5., value*.00015)+value*slip, 0.)


def affordable_amount(request, cash, slip):
    def spend(scale):
        v = request*scale
        return v.sum()+amount_fee(v,slip).sum()
    if spend(1.) <= cash:
        return request
    left, right = 0.,1.
    for _ in range(44):
        middle = (left+right)/2
        if spend(middle) <= cash:
            left = middle
        else:
            right = middle
    return np.where(request*left>1e-9,request*left,0.)


def economic(name, returns, prices, start, slip=.0009):
    calendar = returns.index
    dates = calendar[calendar >= start]
    policy = prices
    position = np.zeros(len(KEYS))
    cash = CAPITAL
    ledger, weights, pnls, trades = [], [], [], []
    for i, day in enumerate(dates):
        j = calendar.get_loc(day)
        known = calendar[j-1]
        nav = cash + position.sum()
        sell_target,buy_target = policy.targets(day,known,i,nav)
        def difference(w):
            if w is None:
                return np.zeros(len(KEYS))
            change = w*nav - position
            if name=='V12':
                change = np.where(np.abs(change)>.05*position,change,0.)
            return change
        r = returns.iloc[j].to_numpy()
        pnl = position*r
        sales = -np.minimum(difference(sell_target),0)*(1+r)
        buys_requested = np.where(sales==0,np.maximum(difference(buy_target),0)*(1+r),0.)
        position += pnl
        sales = np.minimum(sales,position)
        sale_cost = amount_fee(sales,slip)
        buys = affordable_amount(buys_requested,min(cash,cash+float((sales-sale_cost).sum())),slip)
        buy_cost = amount_fee(buys,slip)
        cost = float(sale_cost.sum()+buy_cost.sum())
        cash += float(sales.sum()-buys.sum()-cost)
        position += buys-sales
        nav = cash+position.sum()
        count = 0
        for side, values, fees in [('sell',sales,sale_cost),('buy',buys,buy_cost)]:
            for k in np.flatnonzero(values>1e-9):
                trades.append(dict(date=day,known=known,asset=KEYS[k],side=side,notional=values[k],cost=fees[k]))
                count += 1
        ledger.append(dict(date=day,nav=nav,cash=cash,receivable=0.,cost=cost,notional=float(sales.sum()+buys.sum()),trade_count=count))
        weights.append(np.r_[position/nav,cash/nav])
        pnls.append(pnl)
    return pack(ledger,weights,pnls,trades,policy,calendar[calendar.get_loc(dates[0])-1])


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


def run(input_dir, start=START, end=END, *, schedule="monthly"):
    if schedule not in ("monthly", "weekly"):
        raise ValueError("Select the frozen monthly or weekly schedule")
    returns = pd.read_parquet(input_dir / 'returns.parquet').loc[:end, list(KEYS)]
    returns.attrs['excluded_risk_dates'] = json.loads((input_dir / 'risk_exclusions.json').read_text(encoding='utf-8'))
    state = pd.read_parquet(input_dir / 'state.parquet')
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError('Returns require a unique, increasing ISO date index')
    if start not in returns.index or returns.index.get_loc(start) < 63:
        raise ValueError('Start must be an indexed session with at least 63 preceding sessions')
    if not np.isfinite(returns.loc[start:].to_numpy()).all():
        raise ValueError('Replay returns contain missing or non-finite values')
    policy = Policy(returns, returns, state, Case(schedule=schedule))
    return economic(schedule, returns, policy, start)


def main(schedule="monthly"):
    parser = argparse.ArgumentParser(description=f"Offline Guosheng {schedule} replay")
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', default=START)
    parser.add_argument('--end', default=END)
    args = parser.parse_args()
    account = run(args.inputs, args.start, args.end, schedule=schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    for field in ('ledger', 'weights', 'pnl', 'trades', 'signals'):
        getattr(account, field).to_parquet(args.output / f'{field}.parquet')
    summary = metrics(account, hi=args.end)
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
