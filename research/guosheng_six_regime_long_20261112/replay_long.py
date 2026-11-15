"""Replay the matched accounts and preregistered falsification controls."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from early_inputs import BASE, DATA, HERE, OLD, last_asof, prepare, provider, state_path
from run_study import ASSETS_BY_STAGE, CAPITAL, KEYS, Rotation, apply_policy, engine, measures

OUT = HERE / "accounts"
TABLES = HERE / "tables"
END = "2026-11-12"
MAIN = "GS_monthly"
SLIP = .0009
ACT_START = "2025-11-06"
RULES = HERE / "protocol.json"
AUX = HERE / "supplementary_protocol.json"


def save_csv(frame, name, **kwargs):
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / name, encoding="utf-8-sig", index=False, **kwargs)


def verify_frozen():
    for path in (RULES, AUX, BASE / "protocol.json"):
        assert provider.sha(path) == path.with_suffix(".sha256").read_text(encoding="ascii").strip(), path


@dataclass
class Inputs:
    hist: pd.DataFrame
    benchmark: pd.DataFrame
    macro: pd.DataFrame
    vintage: pd.DataFrame
    coverage: dict
    gs_quotes: tuple
    v12_quotes: tuple
    snapshots: object

    @classmethod
    def create(cls):
        return cls(*prepare())

    def earliest(self):
        excluded = {d for v in self.hist.attrs["excluded_risk_dates"].values() for d in v}
        for i, (known, day) in enumerate(zip(self.hist.index[:-1], self.hist.index[1:])):
            if i < 62 or pd.isna(self.macro.at[pd.Timestamp(known), "stage"]):
                continue
            if sum(d not in excluded for d in self.hist.index[:i+1]) >= 63:
                assert self.benchmark.index.get_loc(day) >= 63
                return day
        raise ValueError("No eligible date")

    def lag(self, start, n, actual=False):
        """Only the stage is lagged; risk weights use current trailing returns."""
        frame = self.macro.copy()
        idx = self.gs_quotes[0].returns.index if actual else self.hist.index
        if actual:
            idx = self.gs_quotes[0].panels["close"].index
        end = frame.groupby(frame.index.to_period("M"), sort=True).tail(1)
        end = end.set_axis(end.index.to_period("M"))
        columns = ["stage", "money_score", "credit", "growth", "credit_published", "growth_available"]
        receipts = []
        for i in range(idx.searchsorted(start), len(idx)):
            day, known = idx[i], idx[i-1]
            if day != start and day[:7] == known[:7]:
                continue
            lagged = pd.Period(day, freq="M") - 1 - n
            row = end.loc[lagged, columns]
            assert pd.notna(row.stage), (day, lagged)
            frame.loc[pd.Timestamp(known), columns] = row
            receipts.append(dict(trade_date=day, cutoff=known, lagged_month=str(lagged), stage=int(row.stage)))
        return frame, receipts

    def remap(self, n):
        frame = self.macro.copy()
        frame["stage"] = frame.stage.map(lambda x: (min(int(x), 5)-1+n)%5+1 if pd.notna(x) else pd.NA)
        frame["stage"] = frame["stage"].astype("Int64")
        return frame

    def publication_delay(self, n):
        """Delay credit/PMI independently without changing monthly schedule."""
        c = pd.read_csv(DATA / "credit_monthly.csv").set_index("released")
        c.index = pd.to_datetime(c.index) + pd.Timedelta(days=n)
        p = pd.read_csv(DATA / "pmi_as_reported.csv").set_index("available")
        p.index = pd.to_datetime(p.index) + pd.Timedelta(days=n)
        frame = self.macro.copy()
        frame["credit"] = np.sign(last_asof(c.credit_change, frame.index))
        frame["growth"] = last_asof(p.direction, frame.index)
        frame["credit_published"] = last_asof(pd.Series(c.index.strftime("%Y-%m-%d"), index=c.index), frame.index)
        frame["growth_available"] = last_asof(pd.Series(p.index, index=p.index), frame.index)
        frame["stage"] = state_path(frame.money, frame.credit, frame.growth)
        return frame


class CashAblation(Rotation):
    def __init__(self, name, returns, macro, assets):
        super().__init__(name, returns, macro)
        self.zero = [KEYS.index(a) for a in assets]

    def targets(self, day, known, i, nav):
        pair = super().targets(day, known, i, nav)
        out = []
        for a in pair:
            if a is not None:
                a = a.copy()
                a[self.zero] = 0.
            out.append(a)
        return tuple(out)


def run(inputs, mode, start, kind, slip=SLIP, *, state=None, cash_for=()):
    actual = mode == "actual"
    if kind == "V12":
        if actual:
            q, adv, up, down = inputs.v12_quotes
            return engine.actual("V12", q, adv, up, down, None, start, slip)
        return engine.economic("V12", inputs.benchmark, None, start, slip)
    state = inputs.macro if state is None else state
    policy_name = kind if kind in ("Six_asset_ERC", "GS_DR007_monthly", "GS_stage_switch") else MAIN
    factory = (lambda n, r, _: CashAblation(policy_name, r, state, cash_for)) if cash_for else (
        lambda n, r, _: Rotation(policy_name, r, state))
    with apply_policy(factory, KEYS):
        if actual:
            q, adv, up, down = inputs.gs_quotes
            return engine.actual(kind, q, adv, up, down, None, start, slip)
        return engine.economic(kind, inputs.hist, None, start, slip)


def stats(result):
    return measures(result.ledger, result.base_date, CAPITAL)


def verify_account(result, kind, mode):
    l, s, t = result.ledger, result.signals, result.trades
    assert l.index[-1] == END and l.index.is_unique and l.index.is_monotonic_increasing
    assert (s.known < s.date).all() and (t.known < t.date).all()
    assert s.date.is_unique
    if "first_return" in s:
        assert (s.first_return <= s.last_return).all() and (s.last_return <= s.known).all()
        assert (s.credit_date <= s.known).all() and (s.growth_date <= s.known).all()
        if kind != "Six_asset_ERC":
            assert all(r.selected == "/".join(ASSETS_BY_STAGE[r.phase]) for r in s.itertuples())
    if mode == "actual":
        assert (t.shares <= .01*t.adv+1e-8).all()
        assert (t.loc[t.side.eq("buy"), "shares"] % 100 == 0).all()
    prev = l.nav.shift(1, fill_value=CAPITAL)
    residual = (prev+result.pnl.sum(axis=1)-l.cost-l.nav).abs().max()
    assert residual <= 1e-7
    assert result.weights.min().min() >= -1e-9 and l.cash.min() >= -1e-7
    return dict(days=len(l), rebalances=len(s), trades=len(t), max_ledger_error=float(residual))


def save_account(value, mode, name, slip):
    r = verify_account(value, name, mode)
    folder = OUT / mode / value.ledger.index[0] / f"{name}_{slip*10000:g}bp"
    folder.mkdir(parents=True, exist_ok=True)
    for k in ("ledger", "weights", "pnl", "signals", "trades"):
        getattr(value, k).to_parquet(folder / f"{k}.parquet")
    provider.write(folder / "meta.json", dict(base_date=value.base_date, capital=CAPITAL,
                                               verification=r, ledger_sha256=provider.sha(folder/"ledger.parquet")))
    row = dict(mode=mode, strategy=name, slip_bps=round(slip*10000, 5), **stats(value))
    print(mode, row["start"], name, row["slip_bps"],
          *(round(row[k]*100, 3) for k in ("cagr", "vol", "mdd")), flush=True)
    return row


def load_account(mode, start, name, bps=9):
    path = OUT / mode / start / f"{name}_{bps:g}bp"
    return SimpleNamespace(base_date=json.loads((path/"meta.json").read_text(encoding="utf-8"))["base_date"],
                           **{k: pd.read_parquet(path / (k+".parquet"))
                              for k in ("ledger", "weights", "pnl", "signals", "trades")})


def log_attribution(value):
    l = value.ledger
    prev = l.nav.shift(1, fill_value=CAPITAL)
    r = l.nav / prev - 1.
    raw = value.pnl.div(prev, axis=0).assign(fee=-l.cost/prev)
    np.testing.assert_allclose(raw.sum(axis=1), r, rtol=0, atol=3e-15)
    scale = np.divide(np.log1p(r), r, out=np.ones(len(r)), where=r.ne(0))
    log = raw.mul(scale, axis=0)
    np.testing.assert_allclose(log.sum(axis=1), np.log1p(r), rtol=0, atol=3e-15)
    assert abs(log.to_numpy().sum()-np.log(l.nav.iloc[-1]/CAPITAL)) < 1e-11
    return r, raw, log


def break_even(inputs, mode, start, v12, target):
    """Stress candidate slippage only, keep V12 at the common 9 bp hurdle."""
    hurdle = stats(v12)["cagr"]
    values = {}
    def evaluate(bps):
        if bps not in values:
            x = target if bps == 9 else run(inputs, mode, start, MAIN, bps/1e4)
            values[bps] = stats(x)["cagr"]
        return values[bps] - hurdle
    lo = 0
    hi = None
    for n in (0, 9, 20, 40, 80, 160, 320, 640, 1280):
        if evaluate(n) <= 0:
            hi = n
            break
        lo = n
    if hi is not None:
        while hi-lo > 1:
            mid = (lo+hi)//2
            if evaluate(mid) > 0:
                lo = mid
            else:
                hi = mid
    print("break-even", mode, start, lo, hi, flush=True)
    return [dict(mode=mode, start=start, slip_bps=k, cagr=v, excess_vs_v12=v-hurdle, last_win_bp=lo,
                 first_lose_bp=hi) for k, v in sorted(values.items())]


def identical_to_prior(value, mode, start, name):
    path = BASE / "accounts" / mode / start / f"{name}_9bp"
    for k in ("ledger", "weights", "pnl", "trades", "signals"):
        pd.testing.assert_frame_equal(getattr(value, k), pd.read_parquet(path/f"{k}.parquet"),
                                      check_exact=False, rtol=0, atol=1e-12, check_index_type=False)


def main():
    verify_frozen()
    inputs = Inputs.create()
    early = inputs.earliest()
    check = dict(earliest_amount=early, actual_start=inputs.coverage["earliest_63_day_history"])
    rows, costs, lags = [], [], []
    amount_starts = [early, "2013-05-02", "2014-01-02", "2016-01-04", "2020-01-02"]
    actual_starts = [check["actual_start"], "2025-07-01", ACT_START, "2026-05-07"]
    for mode, starts in (("amount", amount_starts), ("actual", actual_starts)):
        for start in starts:
            gs = run(inputs, mode, start, MAIN)
            v = run(inputs, mode, start, "V12")
            assert gs.ledger.index.equals(v.ledger.index) and gs.base_date == v.base_date
            for a, n in ((gs, MAIN), (v, "V12")):
                rows.append(save_account(a, mode, n, SLIP))
                if start in ("2016-01-04", "2020-01-02", "2025-07-01", ACT_START):
                    identical_to_prior(a, mode, start, n)
            if start not in (early, ACT_START):
                continue
            for n in (MAIN, "V12"):
                rows.append(save_account(run(inputs, mode, start, n, .002), mode, n, .002))
            _, _, log = log_attribution(gs)
            top = tuple(log.loc[:, KEYS].sum().nlargest(2).index)
            for assets in [*( (a,) for a in KEYS), top]:
                label = "cash_" + "_".join(assets)
                rows.append(save_account(run(inputs, mode, start, label, cash_for=assets), mode, label, SLIP))
            for control in ("Six_asset_ERC", "GS_DR007_monthly"):
                rows.append(save_account(run(inputs, mode, start, control), mode, control, SLIP))
            for n in (1, 3):
                state, receipt = inputs.lag(start, n, actual=mode=="actual")
                lags.extend(dict(mode=mode, months=n, **a) for a in receipt)
                name = f"lag{n}m"
                rows.append(save_account(run(inputs, mode, start, name, state=state), mode, name, SLIP))
            for n in range(1, 5):
                name = f"rotation{n}"
                rows.append(save_account(run(inputs, mode, start, name, state=inputs.remap(n)), mode, name, SLIP))
            name = "PMI_credit_extra7days"
            rows.append(save_account(run(inputs, mode, start, name, state=inputs.publication_delay(7)), mode, name, SLIP))
            if mode == "amount":
                name = "PMI_current_vintage"
                rows.append(save_account(run(inputs, mode, start, name, state=inputs.vintage), mode, name, SLIP))
            costs.extend(break_even(inputs, mode, start, v, gs))
    save_csv(pd.DataFrame(rows), "replays.csv")
    save_csv(pd.DataFrame(costs), "cost_curve.csv")
    save_csv(pd.DataFrame(lags), "lag_decisions.csv")
    files = [*DATA.glob("tushare/*/*.parquet"), *DATA.glob("pbc/releases/*.html"),
             *DATA.glob("pbc/*.pdf"), DATA/"mfd_hsbc_final.html", DATA/"mfd_rows.json",
             DATA/"pbc_early_index.csv", OLD, BASE/"input_hashes.json", *inputs.snapshots.hashes,
             BASE/"market.py", BASE/"run_study.py", BASE/"regimes.py",
             Path(engine.__file__), Path(__file__), HERE/"early_inputs.py"]
    provider.write(HERE/"input_hashes.json", {
        p.relative_to(HERE.parent).as_posix():provider.sha(p) for p in files})
    provider.write(HERE/"replay_checks.json", check | {"saved_accounts":len(rows),
                                                     "prior_reproduced_accounts":8})


if __name__ == "__main__":
    main()
