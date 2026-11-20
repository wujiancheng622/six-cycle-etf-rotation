"""Execute an existing plan leg against qmt_gate without changing portfolio rules."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
import json
import os
from pathlib import Path
from uuid import uuid4

from qmt_adapter import ETF_CODES, MANUAL_REASONS, RETRY_REASONS, Stop, Unknown, Wait, amount, identity, quantity


CN = timezone(timedelta(hours=8))
CENT = Decimal('0.01')
ETF_PRICE_TICK = Decimal('.001')


@dataclass(frozen=True)
class Leg:
    plan_id: str
    leg_id: str
    security: str
    side: str
    opening_quantity: int
    volume: int
    starts_at: datetime
    deadline: datetime
    max_attempts: int | None

    def binding(self):
        if (not identity(self.plan_id) or not identity(self.leg_id) or self.security not in ETF_CODES
                or self.side not in ('BUY', 'SELL') or not quantity(self.volume)
                or (self.max_attempts is not None and not quantity(self.max_attempts))
                or self.starts_at.utcoffset() is None
                or self.deadline.utcoffset() is None or self.deadline <= self.starts_at
                or self.starts_at.astimezone(CN).date() != self.deadline.astimezone(CN).date()):
            raise Stop('Invalid plan leg or order window')
        if quantity(self.opening_quantity) < self.volume and self.side == 'SELL':
            raise Stop('Sell goal exceeds opening strategy shares')
        return {**vars(self), 'starts_at': self.starts_at.isoformat(), 'deadline': self.deadline.isoformat()}


@dataclass(frozen=True)
class Terms:
    volume: int
    limit_price: Decimal
    fee_ceiling: Decimal
    quote_expires_at: datetime
    buy_budget: Decimal | None = None


@dataclass(frozen=True)
class Progress:
    status: str
    reason: str
    filled: int
    request_id: str | None = None
    intent_id: str | None = None


@contextmanager
def locked(path):
    """Use the same process-held lock approach as the V12 execution journal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt

            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise Stop('Another process owns this execution journal') from None
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Stop('Another process owns this execution journal') from None
        yield


def save(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def buy_reserve(volume, price, fee_ceiling):
    # Same upward price and fee reservation as the gateway and V12 planner.
    gross = price.quantize(CENT, ROUND_CEILING) * volume
    rule = max(Decimal(5), (gross * Decimal('.00005')).quantize(CENT, ROUND_HALF_UP))
    fee = max(fee_ceiling, (gross * Decimal('.002')).quantize(CENT, ROUND_CEILING),
              ((rule * 100 * 11 // 10) + 1) / 100)
    return gross + fee


def tick_price(value):
    price = amount(value)
    snapped = price.quantize(ETF_PRICE_TICK)
    if price <= 0 or abs(price - snapped) > ETF_PRICE_TICK / 1000:
        raise Stop('Price not on the ETF tick grid')
    return snapped


def validated_mark(tick, reference, maximum_age, *, at=None):
    at = at or datetime.now(CN)
    if (tick.get('stockStatus') != 3 or tick.get('openInt') != 13
            or type(tick.get('time')) not in (int, float)):
        raise Wait('The ETF is not in continuous trading')
    source_at = datetime.fromtimestamp(tick['time'] / 1000, tz=CN)
    if source_at.date() != at.date() or not 0 <= (at - source_at).total_seconds() < maximum_age:
        raise Wait('The quote is missing or stale')
    if tick_price(tick.get('lastClose')) != reference:
        raise Stop('The broker previous close differs from the frozen plan')
    return tick_price(tick.get('lastPrice')), source_at + timedelta(seconds=maximum_age)


def validated_order_quote(detail, tick, reference, maximum_age, side, *, at=None):
    """Take the best ask for a buy or the best bid for a sell."""
    if side not in ('BUY', 'SELL'):
        raise Stop('Invalid order side')
    if type(detail.get('InstrumentStatus')) is not int or detail['InstrumentStatus'] > 0:
        raise Wait('The ETF is suspended or has an unknown instrument status')
    last, expires = validated_mark(tick, reference, maximum_age, at=at)
    try:
        if not (tick['askVol'][0] > 0 and tick['bidVol'][0] > 0):
            raise Wait('No executable top of book')
        bid = tick_price(tick['bidPrice'][0])
        ask = tick_price(tick['askPrice'][0])
    except (KeyError, IndexError, TypeError):
        raise Wait('No executable top of book') from None
    lower = tick_price(detail.get('DownStopPrice'))
    upper = tick_price(detail.get('UpStopPrice'))
    price = ask if side == 'BUY' else bid
    if not (lower <= bid <= ask <= upper and lower < price < upper and lower < last < upper):
        raise Wait('The ETF is at a price limit or its quote is outside the exchange band')
    return price, expires


def validated_buy_quote(detail, tick, reference, maximum_age, *, at=None):
    return validated_order_quote(detail, tick, reference, maximum_age, 'BUY', at=at)


class Execution:
    """One bounded step per call. The scheduler supplies the original plan and pricing rules."""

    def __init__(self, gate, journal, *, now=lambda: datetime.now(CN)):
        self.gate, self.path, self.now = gate, Path(journal), now

    def _in_window(self, leg):
        now = self.now()
        if now.utcoffset() is None:
            raise Stop('Execution requires a timezone-aware clock')
        local = now.astimezone(CN)
        return (leg.starts_at <= now < leg.deadline and local.weekday() < 5
                and (time(9, 30) <= local.time() < time(11, 30) or time(13) <= local.time() < time(14, 57)))

    def _load(self):
        scope = {'port': self.gate.port, 'strategy_id': self.gate.strategy_id}
        book = (json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists()
                else {'scope': scope, 'legs': {}})
        if book.get('scope') != scope:
            raise Stop('Journal belongs to a different port or strategy')
        return book

    def recover_pending(self):
        """Recover persisted requests across plan days without a pricing callback."""
        with locked(self.path.with_suffix('.lock')):
            pending = [row['binding'].copy() for row in self._load()['legs'].values()
                       if row['attempts'] and not row['attempts'][-1].get('outcome')]
        results = {}
        for row in pending:
            for field in ('starts_at', 'deadline'):
                row[field] = datetime.fromisoformat(row[field])
            leg = Leg(**row)
            results[f'{leg.plan_id}/{leg.leg_id}'] = self.advance(leg)
        return results

    def advance(self, leg, price_order=None, *, cancel=False):
        """Recover the recorded request first. Create a request only with a fresh pricing callback."""
        binding = leg.binding()
        with locked(self.path.with_suffix('.lock')):
            book = self._load()
            key = f'{leg.plan_id}/{leg.leg_id}'
            row = book['legs'].setdefault(key, {'binding': binding, 'attempts': []})
            if row['binding'] != binding:
                raise Stop('Persisted leg differs from the supplied plan')
            attempts = row['attempts']
            filled = sum(attempt.get('filled', 0) for attempt in attempts)
            if filled > leg.volume:
                raise Stop('Journal fill exceeds plan volume')
            if attempts:
                last = attempts[-1]
                if last.get('outcome') == 'STOP':
                    return self._progress('MANUAL', last['reason'], filled, last)
                if not last.get('outcome'):
                    return self._recover(book, leg, last, filled, cancel)
            if filled == leg.volume:
                return self._progress('DONE', 'TARGET_FILLED', filled, attempts[-1])
            if cancel or price_order is None:
                return Progress('IDLE', 'NO_NEW_REQUEST', filled)
            if self.now() >= leg.deadline or (leg.max_attempts is not None and len(attempts) >= leg.max_attempts):
                return Progress('INCOMPLETE', 'ORIGINAL_DEADLINE_OR_ATTEMPT_LIMIT', filled)
            if not self._in_window(leg):
                return Progress('WAIT', 'OUTSIDE_ORDER_WINDOW', filled)
            for other_key, other in book['legs'].items():
                if (other_key != key and other['binding']['security'] == leg.security and other['attempts']
                        and not other['attempts'][-1].get('outcome')):
                    return Progress('WAIT', 'RECOVER_OTHER_LEG_FOR_SECURITY', filled)

            try:
                request, expires_at = self._new_request(leg, filled, price_order)
                if not self._in_window(leg):
                    return Progress('WAIT', 'ORDER_WINDOW_CLOSED_DURING_PREFLIGHT', filled)
            except (Wait, Unknown) as exc:
                return Progress('WAIT', str(exc), filled)
            attempt = {'request': request, 'intent_id': None, 'filled': 0}
            attempts.append(attempt)
            save(self.path, book)  # The request must survive a crash during POST.
            if self.now() >= expires_at or not self._in_window(leg):
                # No POST has occurred in this process. Discard the unused local preparation.
                attempts.pop()
                save(self.path, book)
                return Progress('WAIT', 'PRICE_OR_WINDOW_EXPIRED_BEFORE_POST', filled)
            try:
                raw = self.gate.submit(request)
                if raw.get('status') == 'REJECTED' or raw.get('submission_outcome') == 'NOT_SUBMITTED':
                    return self._rejection(book, attempt, raw, filled)
                if not identity(raw.get('intent_id')):
                    raise Unknown('Submission did not return an intent identity')
                attempt['intent_id'] = raw['intent_id']
                save(self.path, book)
            except Unknown as exc:
                return self._progress('UNKNOWN', str(exc), filled, attempt)
            return self._progress('WORKING', 'AWAITING_CONFIRMED_EXECUTION', filled, attempt)

    @staticmethod
    def _progress(status, reason, filled, attempt):
        return Progress(status, reason, filled, attempt['request']['request_id'], attempt.get('intent_id'))

    def _new_request(self, leg, filled, price_order):
        account = self.gate.account()
        if not self.gate.write or not account.trading_enabled:
            raise Stop('Strategy order writes are disabled')
        position = account.positions.get(leg.security)
        owned = position.quantity if position else 0
        if position and position.blocked:
            raise Stop('Position is blocked for reconciliation')
        expected = leg.opening_quantity + (filled if leg.side == 'BUY' else -filled)
        if expected != owned:
            raise Stop('Own shares differ from plan and confirmed fills')
        if ((position and position.symbol_busy)
                or any(row.get('security') == leg.security for row in self.gate.intents())):
            raise Wait('SHARED_SYMBOL_BUSY_OR_OWN_INTENT_PENDING')
        cap = leg.volume - filled
        if leg.side == 'SELL':
            cap = min(cap, position.sellable if position else 0, owned)
        if leg.side == 'BUY' or cap != owned:
            cap = cap // 100 * 100
        if not cap:
            raise Wait('SHARED_SELLABLE_INSUFFICIENT' if leg.side == 'SELL' else 'REMAINING_BELOW_LOT')
        terms = price_order(account, cap)
        volume = quantity(terms.volume)
        if not volume:
            raise Wait('PLANNER_HAS_NO_EXECUTABLE_QUANTITY')
        if volume > cap or (volume % 100 and (leg.side == 'BUY' or volume != owned)):
            raise Stop('Priced quantity exceeds the available lot or strategy limit')
        price, fee = amount(terms.limit_price), amount(terms.fee_ceiling)
        if price <= 0:
            raise Stop('Invalid limit price')
        if terms.quote_expires_at.utcoffset() is None or self.now() >= terms.quote_expires_at:
            raise Wait('PRICE_EXPIRED')
        # The caller supplies the original plan's cash allowance (e.g. excluding same-day sale proceeds).
        if leg.side == 'BUY' and buy_reserve(volume, price, fee) > min(amount(terms.buy_budget), account.available_cash):
            raise Wait('BUY_EXCEEDS_OWN_AVAILABLE_BUDGET')
        rid = 'gs-' + uuid4().hex
        request = dict(request_id=rid, order_remark=rid, stock_code=leg.security, volume=volume,
                       direction=23 if leg.side == 'BUY' else 24, limit_price=format(price, 'f'),
                       send_price=format(price, 'f'), fee_ceiling=format(fee, 'f'))
        return request, terms.quote_expires_at

    def _rejection(self, book, attempt, raw, filled):
        rid = attempt['request']['request_id']
        iid = raw.get('intent_id')
        if (raw.get('submission_outcome') != 'NOT_SUBMITTED' or raw.get('request_id') != rid
                or not identity(iid) or attempt.get('intent_id') not in (None, iid)):
            raise Unknown('Missing NOT_SUBMITTED proof for the original request')
        reason = str(raw.get('reason', ''))
        code = next((value for value in RETRY_REASONS if reason.startswith(value)), None)
        reported = code or next((value for value in MANUAL_REASONS if reason.startswith(value)),
                                'CONFIRMED_REJECTION_REQUIRES_REVIEW')
        attempt.update(intent_id=iid, outcome='NOT_SUBMITTED' if code else 'STOP', reason=reported)
        save(self.path, book)
        return self._progress('WAIT' if code else 'MANUAL', reported, filled, attempt)

    def _find(self, leg, attempt):
        rid = attempt['request']['request_id']
        day = leg.starts_at.astimezone(CN).strftime('%Y%m%d')
        pending = self.gate.intents()
        events = self.gate.events(day)
        rejected = [row for row in events.get('local_rejections', []) if row.get('request_id') == rid]
        ids = {row.get('intent_id') for row in pending if row.get('request_id') == rid}
        ids.update(row.get('intent_id') for row in events['orders'] + events['trades']
                   if row.get('request_id') == rid or row.get('order_remark') == rid)
        if rejected:
            if len(rejected) != 1 or ids:
                raise Stop('Contradictory gateway facts for the recorded request')
            return rejected[0]
        if len(ids) != 1 or not identity(next(iter(ids))):
            raise Unknown('Original request has no unique gateway intent yet')
        return self.gate.order(next(iter(ids)))

    def _validate_order(self, leg, attempt, reply):
        req = attempt['request']
        if (reply.get('strategy_id') != self.gate.strategy_id or reply.get('request_id') != req['request_id']
                or reply.get('security') != leg.security or reply.get('side') != leg.side
                or reply.get('quantity') != req['volume'] or not identity(reply.get('intent_id'))
                or attempt.get('intent_id') not in (None, reply['intent_id'])
                or reply.get('trade_day') != leg.starts_at.astimezone(CN).strftime('%Y%m%d')):
            raise Stop('Gateway order identity differs from the saved request')
        filled = quantity(reply.get('filled'))
        if not attempt['filled'] <= filled <= req['volume']:
            raise Stop('Gateway cumulative fills decreased or exceed the original volume')
        return filled

    def _recover(self, book, leg, attempt, filled, cancel):
        try:
            if not attempt.get('intent_id'):
                observed = self._find(leg, attempt)
                if observed.get('submission_outcome') == 'NOT_SUBMITTED':
                    return self._rejection(book, attempt, observed, filled)
                self._validate_order(leg, attempt, observed)
                attempt['intent_id'] = observed['intent_id']
                save(self.path, book)
            iid = attempt['intent_id']
            # A read-only observer can recover identity/state, but cannot invoke broker reconciliation.
            reply = self.gate.refresh(iid) if self.gate.write else self.gate.order(iid)
            if reply.get('local_rejection'):
                self._validate_order(leg, attempt, reply)
                observed = self._find(leg, attempt)
                return self._rejection(book, attempt, observed, filled)
            current = self._validate_order(leg, attempt, reply)
            filled += current - attempt['filled']
            attempt['filled'] = current
            if (reply.get('state') in ('DONE', 'SELL_SETTLEMENT_PENDING')
                    and reply.get('execution_pending') is False and current
                    and reply.get('fees_confirmed') is not True):
                save(self.path, book)
                return self._progress('FEE_PENDING', 'EXECUTION_DONE_FEE_UNCONFIRMED', filled, attempt)
            if reply.get('state') == 'DONE' and reply.get('execution_pending') is False:
                attempt.update(outcome='SETTLED', final_fee_cents=reply.get('final_fee_cents'))
                save(self.path, book)
                status = 'DONE' if filled == leg.volume else 'READY'
                return self._progress(status, 'TERMINAL_QUANTITY_RECONCILED', filled, attempt)
            save(self.path, book)
            if reply.get('state') in ('UNKNOWN', 'DISPATCHING', 'PREPARED'):
                return self._progress('UNKNOWN', 'AWAITING_GATEWAY_RECONCILIATION', filled, attempt)
            if (cancel and self._in_window(leg) and self.gate.write and reply.get('state') == 'ACTIVE'
                    and not attempt.get('cancel_requested')):
                attempt['cancel_requested'] = True
                save(self.path, book)
                self.gate.cancel(iid)
            return self._progress('WORKING', 'CANCEL_OR_EXECUTION_NOT_CONFIRMED', filled, attempt)
        except Unknown as exc:
            return self._progress('UNKNOWN', str(exc), filled, attempt)
