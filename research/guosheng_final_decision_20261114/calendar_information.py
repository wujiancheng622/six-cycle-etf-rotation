"""Check how non-trading-day macro announcements interact with the signal cutoff."""

import numpy as np
import pandas as pd

import study as s


class CalendarPolicy(s.Policy):
    def __init__(self, returns, risk, state):
        super().__init__(returns, risk, state, s.Case("calendar_information"))
        self.scheduled = {day: (target, (pd.Timestamp(target)-pd.Timedelta(days=1)).date().isoformat())
                          for day, (target, _known) in self.scheduled.items()}

    def targets(self, day, known, i, nav):
        latest = (pd.Timestamp(day)-pd.Timedelta(days=1)).date().isoformat()
        result = super().targets(day, latest, i, nav)
        if result[0] is not None:
            self.receipts[-1]["price_known"] = known
        return result


def main():
    data = s.load()
    days = pd.date_range(data.state.index[0], s.END)
    state = data.state.reindex(days, method="ffill")
    credit = data.inputs.credit
    pmi = data.pmi
    state["credit"] = np.sign(s.last_asof(credit.credit_change.set_axis(pd.DatetimeIndex(credit.released)), days))
    state["credit_published"] = s.last_asof(pd.Series(credit.released.dt.strftime('%Y-%m-%d').to_numpy(),
                                                    index=pd.DatetimeIndex(credit.released)), days)
    state["growth"] = s.last_asof(pmi.direction.set_axis(pd.DatetimeIndex(pmi.available)), days)
    state["growth_available"] = s.last_asof(pd.Series(pmi.available.to_numpy(), index=pd.DatetimeIndex(pmi.available)), days)
    state = s.prior.mkstate(state, data.events)
    rows = []
    for delay in (0, 1, 3, 7):
        shifted = state.copy()
        shifted.index += pd.Timedelta(days=delay)
        delayed = shifted.reindex(days, method="ffill")
        delayed["stage"] = s.state_path(delayed.money, delayed.credit, delayed.growth)
        for mode, start in (("amount", s.START), ("actual", "2025-11-06")):
            factory = lambda n, r, p: CalendarPolicy(r, r, delayed)
            with s.capital(s.CAPITAL), s.prior.apply_policy(factory, s.KEYS):
                if mode == "actual":
                    q, adv, up, down = data.views.gs_quotes
                    account = s.ENGINE.actual("GS_monthly", q, adv, up, down, None, start, .0009)
                else:
                    account = s.ENGINE.economic("GS_monthly", data.views.hist, None, start, .0009)
            # Preserve the price cutoff and record the actual latest macro cutoff on each order.
            account.trades["price_known"] = account.trades.known
            lookup = account.signals.set_index("date").known
            account.trades["known"] = lookup.reindex(account.trades.date, method="ffill").to_numpy()
            s.record(account, f"calendar_{mode}_{delay}d", rows)
    s.write(pd.DataFrame(rows), "calendar_information_comparison.csv")
    print(pd.DataFrame(rows).query("period=='full'")[["case", "cagr", "vol", "mdd"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
