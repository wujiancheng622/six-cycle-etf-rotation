"""Guosheng's scoped qmt_gate client and shared ETF account contract."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import argparse
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


ETF_CODES = frozenset({'159915.SZ', '518880.SH', '159980.SZ', '159201.SZ', '511090.SH', '515100.SH'})
RETRY_REASONS = ('SHARED_SYMBOL_BUSY', 'SHARED_SELLABLE_INSUFFICIENT')
MANUAL_REASONS = ('POSITION_MODE_CONFLICT', 'SHARED_SECURITY_NOT_REGISTERED',
                  'broker/strategy owned shares differ; manual reconciliation needed')
PENDING_REASONS = (
    'pending snapshot quantity requires execution reconciliation before valuation',
    'local account changed during broker position observation; retry query',
    'broker reports no positions while the ledger holds shares; retry query',
)
ID = re.compile(r'[0-9A-Za-z_.:-]{1,120}\Z')


class Stop(RuntimeError):
    """A contract or account problem needs review before creating another order."""


class Wait(RuntimeError):
    """No new order is possible at this observation; recheck in the existing window."""


class Unknown(RuntimeError):
    """A request or observation has no conclusive response."""


def identity(value):
    return isinstance(value, str) and ID.fullmatch(value) is not None


def amount(value):
    try:
        value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise Stop('Invalid gateway or plan amount') from None
    if not value.is_finite() or value < 0:
        raise Stop('Invalid gateway or plan amount')
    return value


def quantity(value):
    if type(value) is not int or value < 0:
        raise Stop('Invalid share quantity')
    return value


@dataclass(frozen=True)
class Position:
    quantity: int
    sellable: int
    cost_amount: Decimal
    account_quantity: int
    broker_quantity: int
    account_sellable: int
    symbol_busy: bool
    blocked: bool


@dataclass(frozen=True)
class Account:
    cash: Decimal
    available_cash: Decimal
    reserved_cash: Decimal
    reserved_sale_fees: Decimal
    reserved_buy_fees: Decimal
    unsettled_sale_proceeds: Decimal
    dividend_receivable: Decimal
    market_value: Decimal
    trading_enabled: bool
    positions: dict[str, Position]

    @classmethod
    def parse(cls, raw, strategy_id):
        if (raw.get('strategy_id') != strategy_id or raw.get('asset_type') != 'ETF'
                or raw.get('position_mode') != 'shared' or raw.get('blocked') is not False
                or raw.get('broker_account_verified') is not True
                or type(raw.get('trading_enabled')) is not bool
                or not isinstance(raw.get('positions'), list)):
            raise Stop('Expected the verified, unblocked shared ETF strategy account')
        cash, available, reserve, sale_fees, buy_fees, unsettled, dividends, market, equity = (
            amount(raw.get(key)) for key in (
                'accounting_cash', 'available_cash', 'reserved_cash', 'reserved_sale_fees',
                'reserved_buy_fees', 'unsettled_sale_proceeds', 'dividend_receivable',
                'market_value', 'economic_equity',
            )
        )
        if cash + market != equity or reserve < sale_fees + buy_fees or available != max(0, cash - reserve - unsettled):
            raise Stop('Strategy cash, reservations or equity do not reconcile')
        positions = {}
        for row in raw['positions']:
            if not isinstance(row, dict):
                raise Stop('Invalid shared position row')
            code = row.get('security')
            if (code not in ETF_CODES or code in positions or row.get('position_mode') != 'shared'
                    or type(row.get('symbol_busy')) is not bool or type(row.get('blocked')) is not bool):
                raise Stop('Unexpected security or position mode in Guosheng account')
            owned, sellable, total, broker, pool = (quantity(row.get(key)) for key in (
                'quantity', 'sellable', 'account_quantity', 'broker_quantity', 'account_sellable'))
            busy, blocked = row['symbol_busy'], row['blocked']
            cost = amount(row.get('cost_amount'))
            if (owned > total or (total != broker and not blocked) or sellable > min(owned, pool)
                    or (busy and sellable != 0) or (owned == 0 and cost != 0)):
                raise Stop(f'{code}: inconsistent shared position, cost or sellable quantity')
            positions[code] = Position(owned, sellable, cost, total, broker, pool, busy, blocked)
        return cls(cash, available, reserve, sale_fees, buy_fees, unsettled, dividends, market,
                   raw['trading_enabled'], positions)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class LocalGateway:
    """Use the V12 scoped REST pattern. Writes require an explicitly enabled client."""

    def __init__(self, port, strategy_id, token, *, write=False, opener=None):
        if (port not in (8001, 8003) or not identity(strategy_id)
                or not re.fullmatch(r'[A-Za-z0-9_-]+', strategy_id)
                or strategy_id.lower() in {'v12', 'policy', 'operator', 'unassigned', 'etf_r5'}):
            raise ValueError('Select a local trading port and a dedicated Guosheng strategy identity')
        if not isinstance(token, str) or not token or token.strip() != token or '\r' in token or '\n' in token:
            raise ValueError('Guosheng strategy token required')
        self.port, self.strategy_id, self.write = port, strategy_id, write
        self._token = token
        self._open = opener or build_opener(ProxyHandler({}), NoRedirect()).open

    def _call(self, method, path='', body=None):
        if method != 'GET' and not self.write:
            raise Stop('Read-only strategy client')
        req = Request(f'http://127.0.0.1:{self.port}/api/v1/strategy-accounts/{self.strategy_id}{path}',
                      data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                      headers={'X-Strategy-Token': self._token, 'Content-Type': 'application/json'}, method=method)
        try:
            try:
                res = self._open(req, timeout=15)
            except HTTPError as exc:
                res = exc
            with res:
                status, data = res.code if isinstance(res, HTTPError) else res.status, res.read()
        except (URLError, OSError, TimeoutError) as exc:
            raise Unknown(f'{method} gateway response unavailable ({type(exc).__name__})') from None
        try:
            result = json.loads(data)
        except (UnicodeError, ValueError):
            raise Unknown(f'{method} gateway response is not JSON') from None
        if not isinstance(result, dict) or status >= 500:
            raise Unknown(f'{method} gateway response not conclusive (HTTP {status})')
        if 200 <= status < 300:
            return result
        detail = result.get('detail')
        reason = detail.get('reason', '') if isinstance(detail, dict) else str(detail or '')
        if status == 409 and method == 'GET' and path == '' and reason in PENDING_REASONS:
            raise Wait('ACCOUNT_RECONCILIATION_PENDING')
        if method == 'POST' and path == '/orders' and status == 409:
            if isinstance(detail, dict) and detail.get('submission_outcome') == 'NOT_SUBMITTED':
                return detail
            if any(reason.startswith(prefix) for prefix in MANUAL_REASONS):
                raise Stop('Gateway requires manual reconciliation or shared registration')
            # A 409 without the same-request proof cannot release the persisted request.
            raise Unknown('Gateway did not prove the request was not submitted')
        raise Stop(f'{method} strategy gateway refused the operation (HTTP {status})')

    def account(self):
        return Account.parse(self._call('GET'), self.strategy_id)

    def intents(self):
        return self._rows(self._call('GET', '/intents'), 'intents')

    def orders(self):
        return self._rows(self._call('GET', '/orders'), 'orders')

    def trades(self):
        return self._rows(self._call('GET', '/trades'), 'trades')

    @staticmethod
    def _rows(result, key):
        rows = result.get(key)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise Stop(f'Invalid scoped {key} response')
        return rows

    def events(self, day):
        result = self._call('GET', '/events?' + urlencode({'day': day}))
        if result.get('event_day') != day:
            raise Stop('Gateway did not confirm the requested event day')
        self._rows(result, 'orders')
        self._rows(result, 'trades')
        if 'local_rejections' in result:
            self._rows(result, 'local_rejections')
        return result

    def _order(self, method, intent_id, suffix=''):
        if not identity(intent_id):
            raise Stop('Invalid gateway intent identity')
        return self._call(method, f'/orders/{intent_id}{suffix}')

    def order(self, intent_id):
        return self._order('GET', intent_id)

    def submit(self, request):
        return self._call('POST', '/orders', request)

    def refresh(self, intent_id):
        return self._order('POST', intent_id, '/refresh')

    def cancel(self, intent_id):
        return self._order('POST', intent_id, '/cancel')


def main():
    cli = argparse.ArgumentParser(description='Read-only verification of the Guosheng shared account')
    cli.add_argument('--port', type=int, required=True, choices=(8001, 8003))
    cli.add_argument('--strategy-id', required=True)
    args = cli.parse_args()
    try:
        gate = LocalGateway(args.port, args.strategy_id, os.environ.get('GUOSHENG_QMT_TOKEN'))
        account = gate.account()
        print(json.dumps({'port': args.port, 'strategy_id': args.strategy_id, 'position_mode': 'shared',
                          'trading_enabled': account.trading_enabled, 'positions': len(account.positions),
                          'intents': len(gate.intents()), 'orders': len(gate.orders()),
                          'trades': len(gate.trades())}))
    except (ValueError, Stop, Wait, Unknown) as exc:
        cli.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
