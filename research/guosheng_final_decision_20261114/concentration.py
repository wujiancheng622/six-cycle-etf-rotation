"""Symmetric tail-day checks and fixed-allocation cash controls."""

import numpy as np
import pandas as pd

import study as s
from postprocess import load_account


class CashControl(s.Policy):
    def targets(self, day, known, i, nav):
        pair = super().targets(day, known, i, nav)
        if pair[0] is not None:
            weights = pair[0].copy()
            weights[[self.keys.index(k) for k in ("growth", "gold")]] = 0.
            self.pending = weights
            self.receipts[-1].update(**dict(zip(self.keys, weights)), cash_target=1-weights.sum())
            return weights, weights
        return (None, self.pending) if i == self.last + 1 else (None, None)


def main():
    a = load_account("monthly_2014-01-02_9bp")
    b = load_account("V12_2014-01-02_9bp")
    g = np.log(a.ledger.nav / a.ledger.nav.shift(1, fill_value=s.CAPITAL))
    v = np.log(b.ledger.nav / b.ledger.nav.shift(1, fill_value=s.CAPITAL))
    assert g.index.equals(v.index)
    years = s.metrics(a)["years"]
    n = int(np.ceil(len(g) / 100))
    keep_g = ~g.index.isin(g.nlargest(n).index)
    keep_v = ~v.index.isin(v.nlargest(n).index)
    rows = []
    for case, x, y in (("base", g, v), ("same_GS_top_days", g[keep_g], v[keep_g]),
                       ("each_own_top_days", g[keep_g], v[keep_v])):
        rows.append(dict(case=case, gs_cagr=np.expm1(x.sum()/years), v12_cagr=np.expm1(y.sum()/years),
                         removed_days=0 if case=="base" else n))
    data = s.load()
    factory = lambda n, r, p: CashControl(r, r, data.state, s.Case("cash_top_two"))
    with s.capital(s.CAPITAL), s.prior.apply_policy(factory, s.KEYS):
        control = s.ENGINE.economic("cash_top_two", data.views.hist, None, s.START, .0009)
    assert control.pnl[["growth","gold"]].eq(0).all().all()
    extra = []
    s.record(control, "growth_gold_cash", extra)
    rows.append(dict(case="growth_gold_cash", gs_cagr=s.metrics(control)["cagr"],
                     v12_cagr=s.metrics(b)["cagr"], removed_days=0))
    s.write(pd.DataFrame(rows), "main_concentration.csv")
    s.write(pd.DataFrame(extra), "main_cash_control.csv")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
