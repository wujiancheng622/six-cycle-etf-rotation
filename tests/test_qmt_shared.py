"""Run with the qmt_gate checkout's test environment and tests on PYTHONPATH."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal as D
from functools import partial
from io import BytesIO
import json
from threading import Barrier
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

from qmt_adapter import LocalGateway, Stop
from qmt_execution import CN, Execution, Leg, Terms, Wait, validated_order_quote
from weekly_execute import WeeklyExecution, initial_state

gateway_tests = pytest.importorskip('test_shared_multi_strategy_http')
from qmt_bridge.app.routers import api
from qmt_bridge.app.services.shared_execution import SharedOrderCoordinator


GOLD = '518880.SH'
METALS = '159980.SZ'
TOKENS = {'peer': 'offline-peer-token', 'guosheng': 'offline-guosheng-token'}
TODAY = datetime.now(CN).replace(hour=10, minute=0, second=0, microsecond=0)
TODAY -= timedelta(days=max(0, TODAY.weekday() - 4))


class Wire:
    """Run urllib requests through the actual FastAPI endpoints in-process."""

    def __init__(self):
        self.before = None
        self.after = None
        self.posts = []

    def open(self, request, timeout):
        path = urlsplit(request.full_url)
        target = path.path + ('?' + path.query if path.query else '')
        is_submit = request.method == 'POST' and path.path.endswith('/orders')
        if is_submit:
            self.posts.append(json.loads(request.data))
            if self.before:
                self.before()
        with gateway_tests.gateway() as app:
            res = app.request(request.method, target, headers=dict(request.header_items()), content=request.data)
        body = res.content
        if is_submit and self.after:
            body = self.after(res)
        if res.status_code >= 400:
            raise HTTPError(request.full_url, res.status_code, '', None, BytesIO(body))
        stream = BytesIO(body)
        stream.status = res.status_code
        return stream


@pytest.fixture
def shared(tmp_path, monkeypatch):
    # Reuse the gateway test's temp ledger and broker substitute, with Guosheng's real ETF identifiers.
    monkeypatch.setattr(gateway_tests, 'ETF', GOLD)
    monkeypatch.setattr(gateway_tests, 'OTHER_ETF', METALS)
    book, counter = gateway_tests.build_shared_http(tmp_path, monkeypatch, TOKENS)

    def stamp_last(method, rows, field):
        def stamped(*args, **kwargs):
            before = len(rows)
            try:
                return method(*args, **kwargs)
            finally:
                for row in rows[before:]:
                    row[field] = int(TODAY.timestamp())

        return stamped

    # Keep the mock broker's execution timestamps on the same market day, including weekends.
    monkeypatch.setattr(counter, 'submit_order', stamp_last(counter.submit_order, counter.orders, 'order_time'))
    monkeypatch.setattr(counter, 'fill', stamp_last(counter.fill, counter.trades, 'traded_time'))
    monkeypatch.setattr(counter, 'partial_sale', stamp_last(counter.partial_sale, counter.trades, 'traded_time'))
    monkeypatch.setattr(api, 'SharedOrderCoordinator', partial(
        SharedOrderCoordinator, require_market_session=False, fee_rule=True, market_clock=lambda: TODAY,
    ))
    wire = Wire()
    peer = LocalGateway(8001, 'peer', TOKENS['peer'], write=True, opener=wire.open)
    guosheng = LocalGateway(8001, 'guosheng', TOKENS['guosheng'], write=True, opener=wire.open)
    return book, counter, wire, peer, guosheng


def make_leg(side='SELL', security=GOLD, owned=1000, volume=800):
    return Leg('monthly-plan', side + '-' + security, security, side, owned, volume,
               TODAY - timedelta(minutes=1), TODAY + timedelta(minutes=10), 4)


def test_weekly_live_bid_and_ask_with_no_opening_reference():
    at = TODAY.replace(hour=14, minute=35)
    detail = {'InstrumentStatus': 0, 'DownStopPrice': 3.60, 'UpStopPrice': 4.40}
    tick = {'stockStatus': 3, 'openInt': 13, 'time': int(at.timestamp() * 1000), 'lastClose': 4,
            'lastPrice': 4.15, 'bidPrice': [4.149], 'askPrice': [4.151], 'askVol': [100], 'bidVol': [100]}
    assert validated_order_quote(detail, tick, D(4), 45, 'SELL', at=at)[0] == D('4.149')
    assert validated_order_quote(detail, tick, D(4), 45, 'BUY', at=at)[0] == D('4.151')
    with pytest.raises(Wait, match='stale'):
        validated_order_quote(detail, tick, D(4), 45, 'SELL', at=at + timedelta(seconds=46))
    with pytest.raises(Wait, match='price limit'):
        validated_order_quote(detail, {**tick, 'bidPrice': [3.60], 'lastPrice': 3.60}, D(4), 45, 'SELL', at=at)


def price_order(account, cap):
    return Terms(cap, D('4.00'), D('30'), TODAY + timedelta(seconds=20), D('9000'))


def open_guosheng(account, counter):
    initial = account.submit(gateway_tests.request('open-guosheng', 1000, side=23, code=GOLD, price='4.10'))
    counter.fill(500)
    assert account.refresh(initial['intent_id'])['state'] == 'DONE'


def test_weekly_waits_across_lunch_and_buys_only_after_confirmed_sale(shared, tmp_path):
    _, counter, wire, peer, gate = shared
    open_guosheng(gate, counter)
    counter.available[GOLD] = 1000
    first_cash = gate.account().available_cash
    clock = [TODAY]
    current = lambda: clock[0]
    day = TODAY.date().isoformat()
    plan = {'date': day, 'reference_positions': {GOLD: {'weight': '0', 'known_close': '4', 'held_shares': 1000},
                                                METALS: {'weight': '1', 'known_close': '4', 'held_shares': 0}}}
    state = initial_state(plan, 'test', gate.account(), lambda c, ref: D(4))
    journal = Execution(gate, tmp_path / 'orders.json', now=current)
    path = tmp_path / 'weekly.json'
    price = lambda c, side, ref, current: (D('4.00') if side == 'SELL' else D('3.80'),
                                          current() + timedelta(seconds=30))
    runner = WeeklyExecution(gate, journal, plan, path, current=current, pricing=price)

    assert runner.step(state)['phase'] == 'SELL'
    assert wire.posts[-1]['direction'] == 24 and wire.posts[-1]['limit_price'] == '4.00'
    clock[0] = TODAY.replace(hour=12, minute=30)
    assert runner.step(state)['phase'] == 'SELL'
    clock[0] = TODAY.replace(hour=14, minute=30)
    assert runner.step(state)['phase'] == 'SELL'
    assert len(counter.calls) == 2 and not counter.cancels
    counter.fill(500)
    counter.available[GOLD] = 1000
    assert runner.step(state)['phase'] == 'BUY'
    assert gate.account().available_cash > first_cash
    assert len(counter.calls) == 2
    assert runner.step(state)['phase'] == 'BUY'
    assert wire.posts[-1]['direction'] == 23 and wire.posts[-1]['limit_price'] == '3.80'
    assert D(wire.posts[-1]['limit_price']) * wire.posts[-1]['volume'] > first_cash
    counter.fill(500)
    assert runner.step(state)['phase'] == 'DONE' and not gate.intents()
    assert peer.account().positions[GOLD].quantity == 1000


def test_weekly_late_sale_can_reconcile_without_opening_a_buy_at_close(shared, tmp_path):
    _, counter, wire, _, gate = shared
    open_guosheng(gate, counter)
    counter.available[GOLD] = 1000
    clock = [TODAY.replace(hour=14, minute=55)]
    current = lambda: clock[0]
    plan = {'date': TODAY.date().isoformat(),
            'reference_positions': {GOLD: {'weight': '0', 'known_close': '4', 'held_shares': 1000},
                                    METALS: {'weight': '1', 'known_close': '4', 'held_shares': 0}}}
    state = initial_state(plan, 'test', gate.account(), lambda c, ref: D(4))
    runner = WeeklyExecution(gate, Execution(gate, tmp_path / 'orders.json', now=current), plan,
                             tmp_path / 'weekly.json', current=current,
                             pricing=lambda c, side, ref, current: (D(4), current() + timedelta(seconds=30)))
    assert runner.step(state)['phase'] == 'SELL'
    counter.fill(500)
    counter.available[GOLD] = 1000
    clock[0] = TODAY.replace(hour=14, minute=58)
    stopped = runner.step(state)
    assert stopped['phase'] == 'SELL' and stopped['legs'][0]['status'] == 'DONE'
    assert len(wire.posts) == 2 and not counter.cancels


def test_guosheng_partial_cancel_and_public_sellable(shared, tmp_path):
    book, counter, wire, peer, gate = shared
    open_guosheng(gate, counter)
    counter.available[GOLD] = 1000
    start = gate.account().positions[GOLD]
    assert (start.quantity, start.cost_amount, start.account_quantity, start.broker_quantity) == (1000, D(4100), 2000, 2000)
    runner = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    leg = make_leg()
    sale = runner.advance(leg, price_order)
    assert sale.status == 'WORKING'
    assert peer.account().positions[GOLD].symbol_busy
    assert peer.account().positions[GOLD].sellable == 0
    counter.partial_sale(400)
    counter.available[GOLD] = 600
    pending = runner.advance(leg, cancel=True)
    assert pending.status == 'WORKING' and pending.filled == 400 and pending.intent_id == sale.intent_id
    assert len(counter.cancels) == 1
    counter.orders[-1]['order_status'] = 53
    counter.position_override = [{'account_id': gateway_tests.ACCOUNT, 'stock_code': GOLD, 'volume': 2000,
                                  'can_use_volume': 600, 'market_value': 8000}]
    reserved = book.view('peer')['positions'][0]['account_sellable']
    assert reserved == 200
    recovered = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    still_pending = recovered.advance(leg, price_order)
    assert still_pending.filled == 400 and still_pending.status == 'WORKING'
    assert book.view('peer')['positions'][0]['account_sellable'] == reserved
    assert len(wire.posts) == 2
    counter.position_override = None
    released = recovered.advance(leg, price_order)
    assert released.status == 'READY' and released.filled == 400
    assert book.view('peer')['positions'][0]['account_sellable'] == 600
    assert gate.refresh(sale.intent_id)['state'] == 'DONE'
    assert book.view('peer')['positions'][0]['account_sellable'] == 600
    assert recovered.advance(leg, price_order).request_id != sale.request_id
    assert wire.posts[-1]['volume'] == 400
    counter.fill(500)
    counter.available[GOLD] = 200
    assert recovered.advance(leg, price_order).status == 'DONE'
    g, p = gate.account(), peer.account()
    assert (g.positions[GOLD].quantity, g.positions[GOLD].cost_amount, g.cash) == (200, D(820), D(9085))
    assert (p.positions[GOLD].quantity, p.positions[GOLD].cost_amount, p.cash) == (1000, D(3900), D(10000))
    assert g.positions[GOLD].broker_quantity == g.positions[GOLD].account_quantity == 1200
    assert not peer.orders() and not peer.trades()
    assert len(gate.trades()) == 3
    assert sum(D(str(row['commission'])) for row in gate.trades()) == D(15)


def test_two_strategies_race_and_same_request_proof(shared, tmp_path):
    _, counter, wire, peer, gate = shared
    peer_runner = Execution(peer, tmp_path / 'peer.json', now=lambda: TODAY)
    gs_runner = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    legs = [make_leg('BUY', METALS, owned=0, volume=100)] * 2
    barrier = Barrier(2, timeout=5)
    wire.before = barrier.wait
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(r.advance, leg, price_order) for r, leg in zip((peer_runner, gs_runner), legs)]
        results = [item.result(timeout=15) for item in futures]
    wire.before = None
    assert sorted(r.status for r in results) == ['WAIT', 'WORKING']
    assert len(counter.calls) == 1
    winner = 0 if results[0].status == 'WORKING' else 1
    blocked = results[1 - winner]
    assert blocked.reason == 'SHARED_SYMBOL_BUSY' and blocked.intent_id
    runners = (peer_runner, gs_runner)
    counter.fill(500)
    assert runners[winner].advance(legs[winner]).status == 'DONE'
    second = runners[1 - winner].advance(legs[1 - winner], price_order)
    assert second.status == 'WORKING' and second.request_id != blocked.request_id
    counter.fill(500)
    assert runners[1 - winner].advance(legs[1 - winner]).status == 'DONE'
    assert all(a.account().positions[METALS].quantity == 100 for a in (gate, peer))
    assert len(gate.trades()) == len(peer.trades()) == 1


@pytest.mark.parametrize('reached_gateway', [True, False])
def test_transport_unknown_restores_the_original_request(shared, tmp_path, reached_gateway):
    _, counter, wire, _, gate = shared
    path = tmp_path / 'gs.json'
    leg = make_leg('BUY', METALS, owned=0, volume=100)

    def lost(*args):
        raise TimeoutError('lost order acknowledgment')

    if reached_gateway:
        wire.after = lost
        counter.lose_next_ack = True
    else:
        wire.before = lost
    first = Execution(gate, path, now=lambda: TODAY).advance(leg, price_order)
    assert first.status == 'UNKNOWN' and first.request_id and first.intent_id is None
    wire.before = wire.after = None
    if reached_gateway:
        counter.fill(500)
    found = Execution(gate, path, now=lambda: TODAY).recover_pending()
    assert len(found) == 1
    result = next(iter(found.values()))
    assert result.request_id == first.request_id and len(wire.posts) == 1
    assert result.status == ('DONE' if reached_gateway else 'UNKNOWN')
    assert len(counter.calls) == int(reached_gateway)


def test_shared_refusal_without_same_request_proof_uses_scoped_recovery(shared, tmp_path, monkeypatch):
    _, counter, wire, peer, gate = shared
    open_guosheng(gate, counter)
    counter.available[GOLD] = 1000
    stale = gate.account()
    other = peer.submit(gateway_tests.request('peer-sell', 100, code=GOLD))
    with monkeypatch.context() as patch:
        patch.setattr(gate, 'account', lambda: stale)

        def wrong_request(reply):
            result = reply.json()
            result['detail']['request_id'] = 'different-request'
            return json.dumps(result).encode()

        wire.after = wrong_request
        path = tmp_path / 'gs.json'
        first = Execution(gate, path, now=lambda: TODAY).advance(make_leg(), price_order)
        assert first.status == 'UNKNOWN'
    wire.after = None
    assert len(counter.calls) == 2
    recovered = Execution(gate, path, now=lambda: TODAY).advance(make_leg(), price_order)
    assert recovered.status == 'WAIT' and recovered.request_id == first.request_id and recovered.intent_id
    counter.fill(500)
    counter.available[GOLD] = 900
    assert peer.refresh(other['intent_id'])['state'] == 'DONE'
    expired = Execution(gate, path, now=lambda: TODAY + timedelta(minutes=20)).advance(make_leg(), price_order)
    assert expired.status == 'INCOMPLETE' and len(wire.posts) == 3


def test_unregistered_and_confirmed_discrepancy_stop(shared, tmp_path):
    book, counter, wire, _, gate = shared
    runner = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    leg = make_leg('BUY', '159915.SZ', owned=0, volume=100)
    refused = runner.advance(leg, price_order)
    assert refused.status == 'MANUAL' and refused.reason == 'SHARED_SECURITY_NOT_REGISTERED'
    assert runner.advance(leg, price_order).status == 'MANUAL'
    assert len(wire.posts) == 1 and not counter.calls
    book.add_strategy('exclusive-fixture', 'ETF', 100, 'exclusive test strategy')
    book.register_security('511090.SH', 'ETF', 'exclusive-fixture')
    conflict = runner.advance(make_leg('BUY', '511090.SH', owned=0, volume=100), price_order)
    assert conflict.status == 'MANUAL' and conflict.reason == 'POSITION_MODE_CONFLICT'
    assert len(wire.posts) == 2 and not counter.calls
    open_guosheng(gate, counter)
    counter.pos[GOLD] += 100
    with pytest.raises(Stop):
        runner.advance(make_leg('BUY', METALS, owned=0, volume=100), price_order)
    assert len(wire.posts) == 3 and len(counter.calls) == 1


def test_scope_holding_and_frozen_leg_are_bound_to_one_journal(shared, tmp_path):
    _, counter, _, _, gate = shared
    open_guosheng(gate, counter)
    runner = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    with pytest.raises(Stop, match='Own shares'):
        runner.advance(make_leg(owned=2000), price_order)
    leg = make_leg()
    runner.advance(leg, price_order)
    with pytest.raises(Stop, match='Persisted leg'):
        runner.advance(replace(leg, volume=900), price_order)
    readonly = LocalGateway(8001, 'guosheng', TOKENS['guosheng'], opener=lambda *args, **kwargs: None)
    with pytest.raises(Stop, match='Read-only'):
        readonly.submit(gateway_tests.request('no-write', 100, code=GOLD))


def test_capped_full_fill_leaves_only_the_unfilled_remainder(shared, tmp_path):
    _, counter, wire, _, gate = shared
    open_guosheng(gate, counter)
    counter.available[GOLD] = 400
    runner = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    leg = make_leg()
    first = runner.advance(leg, price_order)
    assert first.status == 'WORKING' and wire.posts[-1]['volume'] == 400
    counter.fill(500)
    counter.available[GOLD] = 0
    assert runner.advance(leg, price_order).status == 'READY'
    assert runner.advance(leg, price_order).reason == 'SHARED_SELLABLE_INSUFFICIENT'
    assert len(wire.posts) == 2
    counter.available[GOLD] = 500
    retry = runner.advance(leg, price_order)
    assert retry.request_id != first.request_id and wire.posts[-1]['volume'] == 400
    counter.fill(500)
    counter.available[GOLD] = 100
    assert runner.advance(leg, price_order).status == 'DONE'


def test_fee_pending_is_labeled_and_kept_on_original_intent(shared, tmp_path, monkeypatch):
    _, counter, wire, peer, gate = shared
    open_guosheng(gate, counter)
    monkeypatch.setattr(api, 'SharedOrderCoordinator', partial(
        SharedOrderCoordinator, require_market_session=False, fee_rule=False, market_clock=lambda: TODAY,
    ))
    runner = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    leg = make_leg(volume=100)
    first = runner.advance(leg, price_order)
    counter.fill(0)
    pending = runner.advance(leg, price_order, cancel=True)
    assert pending.status == 'FEE_PENDING' and pending.filled == 100 and pending.intent_id == first.intent_id
    assert runner.advance(leg, price_order).status == 'FEE_PENDING'
    assert (len(counter.calls), len(wire.posts), len(counter.cancels)) == (2, 2, 0)
    assert gate.account().reserved_sale_fees > 0 and peer.account().reserved_sale_fees == 0
    assert not peer.account().positions[GOLD].symbol_busy


def test_buy_cannot_spend_funds_excluded_by_the_plan(shared, tmp_path):
    _, counter, wire, _, gate = shared
    runner = Execution(gate, tmp_path / 'gs.json', now=lambda: TODAY)
    leg = make_leg('BUY', METALS, owned=0, volume=100)
    assert gate.account().available_cash > 400
    result = runner.advance(leg, lambda account, cap: replace(price_order(account, cap), buy_budget=D(0)))
    assert result.status == 'WAIT' and result.reason == 'BUY_EXCEEDS_OWN_AVAILABLE_BUDGET'
    assert not wire.posts and not counter.calls
