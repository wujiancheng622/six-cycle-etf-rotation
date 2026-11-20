"""Check the Guosheng weekly decision date and prepare a read-only plan."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from decimal import Decimal
from itertools import product
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

from qmt_adapter import LocalGateway, Stop
from qmt_execution import CN, locked, save


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime/weekly'
SEED = ROOT / 'runtime/first_20261118'
CODES = ('159915.SZ', '518880.SH', '159980.SZ', '159201.SZ', '511090.SH', '515100.SH')
POLICY_NAMES = {'reverse_repo', 'slf', 'mlf', 'reserve', 'reserve_interest', 'loan', 'deposit'}
FROZEN_END = '2026-11-12'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def decision(calendar, day):
    sessions = calendar['sessions']
    if (not calendar['valid_from'] <= day <= calendar['valid_through']
            or not sessions or sessions != sorted(set(sessions))
            or any(date.fromisoformat(d).isoformat() != d for d in sessions)):
        raise Stop('Trading calendar missing, malformed or out of date')
    week = date.fromisoformat(day).isocalendar()[:2]
    first = next((s for s in sessions if date.fromisoformat(s).isocalendar()[:2] == week), None)
    result = {'date': day, 'week_first_session': first, 'rebalance_due': day == first}
    if result['rebalance_due']:
        index = sessions.index(day)
        if not index:
            raise Stop('The preceding trading session is missing')
        result['known'] = sessions[index - 1]
    return result


def get(port, path, **params):
    suffix = '?' + urlencode(params) if params else ''
    with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}{path}{suffix}', timeout=12) as response:
        return json.load(response)


def native_closes(sessions, known, risk_floor):
    expected = [s for s in sessions if risk_floor <= s <= known]
    if not expected or expected[0] != risk_floor or expected[-1] != known:
        raise Stop('Missing source sessions from the verified calendar')

    def fetch(code):
        result = get(8002, '/api/v1/market_data', stock_code=code, period='1d', count=-1,
                     start_time=risk_floor.replace('-', ''), end_time=known.replace('-', ''),
                     dividend_type='none', fill_data='false', download='false', subscribe='false')
        rows = result['bars']
        days = [datetime.fromtimestamp(r['time'] / 1000, CN).date().isoformat() for r in rows]
        prices = [float(r['close']) for r in rows]
        if days != expected or any(p <= 0 or not float(r['amount']) > 0 for p, r in zip(prices, rows)):
            raise Stop(f'{code}: incomplete or invalid closing quotes')
        return code, prices

    with ThreadPoolExecutor(max_workers=3) as pool:
        return expected, dict(pool.map(fetch, CODES))


def verified_stage(review, known, sessions, market_votes, stage_map, seed):
    if review.get('known') != known or known <= seed['source_cutoff']:
        raise Stop(f'Policy review must match source cutoff {known}')
    from_day = seed['source_cutoff']
    expected = [s for s in sessions if from_day < s <= known]
    reviewed = review.get('days', {})
    if set(reviewed) != set(expected):
        raise Stop('Reviewed macro days do not match every trading session after the seed')
    phases = {seed['phase']}
    for day in expected:
        row = reviewed[day]
        votes = row['policy_votes']
        if (set(votes) != POLICY_NAMES or any(v is not None and (type(v) is not int or v not in (-1, 0, 1))
                                             for v in votes.values())
                or any(type(row[name]) is not int or row[name] not in (-1, 0, 1) for name in ('credit', 'growth'))
                or any(date.fromisoformat(row[name]).isoformat() > day for name in ('credit_published', 'growth_available'))
                or not row.get('sources') or any(not url.startswith('https://') for url in row['sources'])):
            raise Stop(f'{day}: incomplete or forward-looking macro evidence')
        total = sum(v for v in votes.values() if v is not None) + market_votes[day]
        unknown = sum(v is None for v in votes.values())
        money = {int(n > 0) - int(n < 0) for n in range(total - unknown, total + unknown + 1)}
        phases = {stage_map.get((m, row['credit'], row['growth']), p) for m, p in product(money, phases)}
    if len(phases) != 1:
        raise Stop(f'Policy uncertainty permits several stages: {sorted(phases)}')
    return next(iter(phases)), reviewed[known]


def prepare(day, known, calendar):
    # Reuse the verified archive, risk estimator and original dated market sources.
    import numpy as np
    import pandas as pd
    import monthly

    sys.path[:0] = [str(ROOT.parent / name) for name in ('guosheng_final_decision_20261114', 'ah_premium_20261106')]
    import study
    from data_access import OfficialData

    review_path = RUNTIME / f'macro_{known}.json'
    if not review_path.is_file():
        raise Stop(f'Reviewed macro data required at {review_path}')
    review = read(review_path)
    seed = read(SEED / 'opening_plan.json')
    if read(SEED / 'opening_result.json')['status'] != 'OPENING_RECONCILED':
        raise Stop('The first reconciled Guosheng account is missing')
    data = study.load()
    sessions = calendar['sessions']
    days, closes = native_closes(sessions, known, FROZEN_END)
    close = pd.DataFrame({k: closes[c] for k, c in zip(monthly.KEYS, CODES)}, index=days)
    frozen_market = data.views.gs_quotes[0]
    np.testing.assert_allclose(close.loc[FROZEN_END], frozen_market.panels['close'].loc[FROZEN_END, list(monthly.KEYS)],
                               atol=1e-8, rtol=0)
    client = OfficialData()
    end = known.replace('-', '')
    start = FROZEN_END.replace('-', '')
    for code in CODES:
        factors = client.query('fund_adj', {'ts_code': code, 'start_date': start, 'end_date': end},
                               'ts_code,trade_date,adj_factor')
        if (len(factors) != len(days) or set(factors.trade_date) != {d.replace('-', '') for d in days}
                or factors.adj_factor.isna().any() or factors.adj_factor.nunique() != 1):
            raise Stop(f'{code}: adjustment data missing or changed; reconcile corporate actions before planning')
    risk = pd.concat([frozen_market.returns, close.pct_change(fill_method=None).iloc[1:]])
    risk.attrs.update(frozen_market.returns.attrs)
    rates = client.query('repo_daily', {'ts_code': 'DR007.IB', 'start_date': start, 'end_date': end},
                         'ts_code,trade_date,weight')
    if rates.trade_date.duplicated().any() or rates.weight.isna().any():
        raise Stop('Invalid DR007 observations')
    dated = dict(zip(rates.trade_date, rates.weight))
    appended = days[1:]
    if any(d.replace('-', '') not in dated for d in appended):
        raise Stop('DR007 is unavailable for a required trading day')
    series = data.inputs.rates.copy()
    for trading_day in appended:
        series.loc[pd.Timestamp(trading_day)] = dated[trading_day.replace('-', '')]
    smooth = series.sort_index().rolling(30).mean()
    market_votes = {d: int(np.sign(smooth.asof(pd.Timestamp(d) - pd.Timedelta(days=90)) - smooth.asof(pd.Timestamp(d))))
                    for d in appended}
    stage, latest = verified_stage(review, known, sessions, market_votes, monthly.STAGE_MAP, seed)
    state = pd.DataFrame([{'stage': stage, 'credit_published': latest['credit_published'],
                           'growth_available': pd.Timestamp(latest['growth_available'])}], index=[pd.Timestamp(known)])
    policy = monthly.Policy(risk.reindex([*risk.index, day]), risk, state, monthly.Case('weekly'))
    policy.targets(day, known, 0, monthly.CAPITAL)
    target = policy.receipts[-1]
    if target['last_return'] != known or target['selected'] != target['requested']:
        raise Stop('The required prior-session data or ETF risk history is insufficient')

    private = read(SEED / 'private.json')
    if private['port'] != 8001 or private['strategy_id'] != 'guosheng':
        raise Stop('Expected the dedicated 8001 Guosheng account')
    acct_gate = LocalGateway(8001, 'guosheng', private['token'])
    view = acct_gate.account()
    if acct_gate.intents() or any(p.blocked for p in view.positions.values()):
        raise Stop('Guosheng has pending orders or blocked positions; reconcile before preparing a plan')
    weights = {c: float(target[k]) for k, c in zip(monthly.KEYS, CODES)}
    closed = {c: Decimal(str(close.loc[known, k])) for k, c in zip(monthly.KEYS, CODES)}
    held = {c: view.positions[c].quantity if c in view.positions else 0 for c in CODES}
    nav = view.cash + view.dividend_receivable + sum(held[c] * closed[c] for c in CODES)
    detail = {}
    for code in CODES:
        desired = nav * Decimal(str(weights[code])) / closed[code]
        delta = desired - held[code]
        shares = -held[code] if weights[code] == 0 else int(delta / 100) * 100
        if held[code] or weights[code]:
            detail[code] = {'weight': weights[code], 'known_close': str(closed[code]),
                            'held_shares': held[code], 'reference_delta_shares': shares}
    return {'status': 'READY_FOR_REVIEW', 'date': day, 'known': known, 'frequency': 'weekly', 'stage': stage,
            'simulation_port': 8001, 'strategy_id': 'guosheng', 'source_review': str(review_path),
            'market_vote': market_votes[known], 'risk_first_session': target['first_return'],
            'risk_last_session': target['last_return'], 'cash_yuan': str(view.available_cash),
            'nav_at_prior_close_yuan': str(nav), 'reference_positions': detail, 'order_submission_allowed': False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('check', 'prepare'))
    cli.add_argument('--day', help='Date override for calendar checks only')
    args = cli.parse_args()
    today = datetime.now(CN).date().isoformat()
    day = args.day or today
    if args.phase == 'prepare' and day != today:
        cli.error('Plans can be prepared only on the current China session date')
    try:
        calendar = read(RUNTIME / 'calendar.json')
        check = decision(calendar, day)
        if args.phase == 'check' or not check['rebalance_due']:
            result = {'status': 'DUE' if check['rebalance_due'] else 'SKIP', **check}
        else:
            path = RUNTIME / f'plan_{day}.json'
            with locked(path.with_suffix('.lock')):
                if path.exists():
                    result = read(path)
                    if (result.get('date') != day or result.get('known') != check['known']
                            or result.get('order_submission_allowed') is not False):
                        raise Stop('Existing weekly plan does not match the review-only contract')
                else:
                    result = prepare(day, check['known'], calendar)
                    save(path, result)
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError, TypeError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'date': day, 'reason': str(exc)}, ensure_ascii=False))
        cli.exit(1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
