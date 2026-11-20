"""Execute a reviewed Guosheng weekly plan on 8001, selling before buying."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, time as clock_time
from decimal import Decimal, ROUND_CEILING
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.error import URLError

from qmt_adapter import ETF_CODES, LocalGateway, Stop, Unknown, Wait, amount, quantity
from qmt_execution import (CENT, CN, ETF_PRICE_TICK, Execution, Leg, Terms, buy_reserve,
                           locked, save, validated_mark, validated_order_quote)
from weekly_prepare import RUNTIME, SEED, decision, get, read


D = Decimal
NEW_ORDER_END = clock_time(14, 57)
CLOSE = clock_time(15)
QUOTE_AGE = 45


def now():
    return datetime.now(CN)


def fee_ceiling(shares, price):
    return (max(D(5), price * shares * D('.00015')) * D('1.1')).quantize(CENT, ROUND_CEILING)


def affordable(cap, price, budget):
    left, right = 0, cap // 100
    while left < right:
        mid = (left + right + 1) // 2
        shares = mid * 100
        if buy_reserve(shares, price, fee_ceiling(shares, price)) <= budget:
            left = mid
        else:
            right = mid - 1
    return left * 100


def plan_at(path, day, calendar):
    raw = path.read_bytes()
    plan = json.loads(raw, parse_float=D)
    due = decision(calendar, day)
    if (not due['rebalance_due'] or plan.get('date') != day or plan.get('known') != due['known']
            or plan.get('status') != 'READY_FOR_REVIEW' or plan.get('order_submission_allowed') is not False
            or plan.get('strategy_id') != 'guosheng' or plan.get('simulation_port') != 8001
            or plan.get('frequency') != 'weekly'):
        raise Stop('Expected today’s reviewed Guosheng 8001 weekly plan')
    positions = plan['reference_positions']
    if not positions or set(positions) - ETF_CODES:
        raise Stop('Unexpected weekly ETF basket')
    weights = sum(amount(r['weight']) for r in positions.values())
    if not D(0) < weights <= D('1.000000000001'):
        raise Stop('Invalid portfolio weights')
    for row in positions.values():
        if not amount(row['known_close']):
            raise Stop('A prior close is missing')
        quantity(row['held_shares'])
    return plan, sha256(raw).hexdigest()


def in_continuous_session(at):
    clock = at.astimezone(CN).time()
    return clock_time(9, 30) <= clock < clock_time(11, 30) or clock_time(13) <= clock < NEW_ORDER_END


def live_mark(code, reference, current):
    tick = get(8002, '/api/v1/tick', stock_code=code)
    return validated_mark(tick, reference, QUOTE_AGE, at=current())[0]


def live_order_price(code, side, reference, current):
    detail = get(8002, '/api/v1/instrument', stock_code=code)
    if (type(detail.get('VolumeMultiple')) is not int or detail['VolumeMultiple'] != 1
            or amount(detail.get('PriceTick')) != ETF_PRICE_TICK):
        raise Stop(f'{code}: unexpected ETF contract or price tick')
    tick = get(8002, '/api/v1/tick', stock_code=code)
    return validated_order_quote(detail, tick, reference, QUOTE_AGE, side, at=current())


def restore_leg(row):
    values = dict(row)
    for field in ('starts_at', 'deadline'):
        values[field] = datetime.fromisoformat(row[field])
    return Leg(**values)


def holdings(account):
    return {code: p.quantity for code, p in account.positions.items() if p.quantity}


def initial_state(plan, digest, account, mark):
    day = plan['date']
    starts = datetime.fromisoformat(day + 'T09:30:00').replace(tzinfo=CN)
    ends = datetime.combine(starts.date(), NEW_ORDER_END, CN)
    positions = plan['reference_positions']
    initial = {code: r['held_shares'] for code, r in positions.items() if r['held_shares']}
    if holdings(account) != initial:
        raise Stop('Current Guosheng holdings differ from the reviewed plan')
    prices = {c: mark(c, amount(r['known_close'])) for c, r in positions.items()}
    nav = account.cash + account.dividend_receivable + sum(n * prices[c] for c, n in initial.items())
    goals = {c: nav * amount(r['weight']) for c, r in positions.items()}
    legs = {'SELL': [], 'BUY': []}
    for code, value in goals.items():
        owned = initial.get(code, 0)
        delta = -owned if not value else int((value / prices[code] - owned) / 100) * 100
        if delta:
            side = 'BUY' if delta > 0 else 'SELL'
            leg = Leg(f'weekly-{day}', f'{side}-{code}', code, side, owned, abs(delta), starts, ends, None)
            legs[side].append(leg.binding())
    return {'date': day, 'digest': digest, 'cash_floor': str(max(D(0), nav - sum(goals.values()))),
            'goals': {code: str(v) for code, v in goals.items()}, 'initial': initial, 'legs': legs,
            'phase': 'SELL'}


class WeeklyExecution:
    def __init__(self, gate, orders, plan, state_path, *, current=now, pricing=live_order_price):
        self.gate = gate
        self.orders = orders
        self.plan = plan
        self.state_path = state_path
        self.current = current
        self.pricing = pricing

    def terms(self, state, leg, account, cap):
        code = leg.security
        reference = amount(self.plan['reference_positions'][code]['known_close'])
        price, expires = self.pricing(code, leg.side, reference, self.current)
        if leg.side == 'SELL':
            return Terms(cap, price, fee_ceiling(cap, price), expires)
        position = account.positions.get(code)
        owned = position.quantity if position else 0
        cash = max(D(0), account.available_cash - amount(state['cash_floor']))
        remaining = max(D(0), amount(state['goals'][code]) - owned * price)
        budget = min(cash, remaining)
        volume = affordable(cap, price, budget)
        if not volume:
            raise Wait('Remaining buy budget cannot fund one lot at the current ask')
        return Terms(volume, price, fee_ceiling(volume, price), expires, budget)

    def funded_buys(self, state, account):
        cash = max(D(0), account.available_cash - amount(state['cash_floor']))
        legs = []
        for row in state['legs']['BUY']:
            leg = restore_leg(row)
            if not cash:
                break
            # The buy legs are sized just before the buy phase, using confirmed proceeds.
            reference = amount(self.plan['reference_positions'][leg.security]['known_close'])
            price, _ = self.pricing(leg.security, 'BUY', reference, self.current)
            remaining = max(D(0), amount(state['goals'][leg.security]) - leg.opening_quantity * price)
            cap = int(remaining / price / 100) * 100
            volume = affordable(cap, price, min(remaining, cash))
            if volume:
                legs.append(replace(leg, volume=volume).binding())
                cash -= buy_reserve(volume, price, fee_ceiling(volume, price))
        return legs

    def step(self, state):
        phase = state['phase']
        if phase == 'DONE':
            return {'phase': 'DONE', 'legs': []}
        in_session = in_continuous_session(self.current())
        results = []
        for row in state['legs'][phase]:
            leg = restore_leg(row)

            def price_order(account, cap):
                return self.terms(state, leg, account, cap)

            # Refresh the original intent even outside the submission window.
            progress = self.orders.advance(leg, price_order if in_session else None)
            results.append({'security': leg.security, **asdict(progress)})
            if progress.status == 'MANUAL':
                raise Stop(f'{leg.security}: {progress.reason}')
            if progress.status == 'UNKNOWN':
                return {'phase': phase, 'legs': results}
        if all(row['status'] == 'DONE' for row in results):
            if self.gate.intents():
                return {'phase': phase, 'legs': results, 'waiting': 'Strategy intents still pending'}
            if phase == 'SELL':
                account = self.gate.account()
                expected = dict(state['initial'])
                for leg in state['legs']['SELL']:
                    expected[leg['security']] -= leg['volume']
                if holdings(account) != {c: n for c, n in expected.items() if n}:
                    raise Stop('Sell fills and Guosheng holdings differ')
                if account.unsettled_sale_proceeds or account.reserved_sale_fees:
                    return {'phase': phase, 'legs': results, 'waiting': 'Sell proceeds or fees still unsettled'}
                if state['legs']['BUY'] and not in_session:
                    return {'phase': phase, 'legs': results, 'waiting': 'Waiting for the next continuous session'}
                state['legs']['BUY'] = self.funded_buys(state, account)
            state['phase'] = 'BUY' if phase == 'SELL' else 'DONE'
            save(self.state_path, state)
        return {'phase': state['phase'], 'legs': results}


def connect(write):
    metadata = get(8001, '/api/v1/ops/overview')['meta']
    if metadata.get('real') is not False or metadata.get('port') != 8001:
        raise Stop('8001 simulation account could not be verified')
    private = read(SEED / 'private.json')
    if private.get('port') != 8001 or private.get('strategy_id') != 'guosheng':
        raise Stop('Expected the dedicated Guosheng simulation credential')
    return LocalGateway(8001, 'guosheng', private['token'], write=write)


def run(plan, digest, gate, execute):
    path = RUNTIME / f"execution_{plan['date']}.json"
    journal = Execution(gate, RUNTIME / 'order_journal.json', now=now)
    with locked(RUNTIME / 'weekly_executor.lock'):
        if execute and path.exists():
            state = read(path)
            if state['date'] != plan['date'] or state['digest'] != digest:
                raise Stop('Weekly journal refers to a different reviewed plan')
        else:
            if execute:
                recovered = journal.recover_pending()
                if any(p.status not in ('DONE', 'READY') for p in recovered.values()):
                    raise Stop('An earlier weekly request still needs reconciliation')
            if gate.intents():
                raise Stop('Reconcile existing Guosheng intents before starting a new weekly plan')
            while now().time() < NEW_ORDER_END:
                try:
                    if not in_continuous_session(now()):
                        raise Wait('Waiting for continuous trading')
                    state = initial_state(plan, digest, gate.account(), lambda c, p: live_mark(c, p, now))
                    break
                except (Wait, Unknown, URLError):
                    if not execute:
                        raise
                    time.sleep(2)
            else:
                raise Stop('No eligible intraday quote was obtained for the weekly plan')
            if not execute:
                return {'phase': 'PREVIEW', 'legs': state['legs']}
            save(path, state)

        runner = WeeklyExecution(gate, journal, plan, path)
        result = {'phase': state['phase'], 'legs': []}
        while now().date().isoformat() == plan['date'] and now().time() < CLOSE:
            try:
                result = runner.step(state)
            except (Wait, Unknown, URLError) as exc:
                result = {'phase': state['phase'], 'waiting': str(exc)}
            if state['phase'] == 'DONE':
                break
            time.sleep(2)
        # Same identity after close. No pricing callback past the new-order cutoff.
        if state['phase'] != 'DONE' and now().date().isoformat() == plan['date']:
            try:
                result = runner.step(state)
            except (Wait, Unknown, URLError) as exc:
                result = {**result, 'waiting': str(exc)}
        return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('plan', type=Path)
    cli.add_argument('--execute', action='store_true', help='Submit in the 8001 simulation account')
    args = cli.parse_args()
    try:
        plan, digest = plan_at(args.plan, now().date().isoformat(), read(RUNTIME / 'calendar.json'))
        result = run(plan, digest, connect(args.execute), args.execute)
        if args.execute:
            save(RUNTIME / f"execution_result_{plan['date']}.json", result)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        cli.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
