"""Correctness checks and a bounded input-value uncertainty experiment."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

import study as s


def validate_saved():
    rows = []
    for folder in sorted((s.OUT / "accounts").iterdir()):
        if not folder.is_dir():
            continue
        meta = json.loads((folder / "meta.json").read_text())
        ledger = pd.read_parquet(folder / "ledger.parquet")
        weights = pd.read_parquet(folder / "weights.parquet")
        pnl = pd.read_parquet(folder / "pnl.parquet")
        trades = pd.read_parquet(folder / "trades.parquet")
        signals = pd.read_parquet(folder / "signals.parquet")
        old = ledger.nav.shift(1, fill_value=meta["capital"])
        assert old.notna().all() and ledger.index.is_unique and ledger.index.is_monotonic_increasing
        error = float((old + pnl.sum(axis=1) - ledger.cost - ledger.nav).abs().max())
        assert error < 1e-6 and ledger.cash.min() >= -1e-7 and weights.min().min() >= -1e-8
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-9)
        assert signals.known.lt(signals.date).all() and trades.known.lt(trades.date).all()
        if "last_return" in signals:
            assert signals.last_return.le(signals.known).all()
        if "shares" in trades:
            assert (trades.loc[trades.side.eq("buy"), "shares"] % 100 == 0).all()
        charges = trades.groupby("date").cost.sum().reindex(ledger.index, fill_value=0)
        np.testing.assert_allclose(charges, ledger.cost, atol=1e-8)
        rows.append(dict(case=folder.name, days=len(ledger), targets=len(signals), orders=len(trades),
                         max_nav_error=error, min_cash=float(ledger.cash.min()),
                         hindsight_diagnostic_only=meta.get("hindsight_diagnostic_only", False)))
    s.write(pd.DataFrame(rows), "account_validation.csv")
    print(f"Validated {len(rows)} saved accounts", flush=True)


def causality(data):
    rows = []
    for day in ("2014-01-02", "2015-03-02", "2020-03-02", "2023-05-04", "2025-11-06", "2026-11-02"):
        i = data.views.hist.index.get_loc(day)
        known = data.views.hist.index[i - 1]
        history = data.views.hist
        p = s.Policy(history, history, data.state, s.Case("base"))
        a = p.targets(day, known, 0, s.CAPITAL)[0]
        changed = history.copy()
        changed.loc[day:] = np.resize([-.4, .7], changed.loc[day:].shape)
        state = data.state.copy()
        state.loc[day:, "stage"] = 1
        q = s.Policy(changed, changed, state, s.Case("base"))
        b = q.targets(day, known, 0, s.CAPITAL)[0]
        np.testing.assert_array_equal(a, b)
        rows.append(dict(day=day, known=known, future_perturbation_max_difference=float(np.max(abs(a - b)))))
    np.testing.assert_allclose(s.risk_parity(np.diag([4., 9.])), [.6, .4], atol=1e-10)
    dates = pd.date_range("2020-01-01", periods=8)
    money = pd.Series([1, 1, -1, -1, -1, 1, 0, -1], index=dates)
    credit = pd.Series([1, 1, 1, -1, -1, -1, -1, 1], index=dates)
    growth = pd.Series([-1, 1, 1, 1, -1, -1, 1, -1], index=dates)
    assert s.state_path(money, credit, growth).tolist() == [1, 2, 3, 4, 5, 6, 6, 6]
    s.write(pd.DataFrame(rows), "causality_checks.csv")


def raw_uncertainty(data):
    conf = json.loads((s.HERE / "input_uncertainty.json").read_text())
    rng = np.random.default_rng(conf["seed"])
    rows = []
    for scenario in range(conf["scenarios"]):
        credit = data.inputs.credit
        changed = credit.middle_long_new_yi * (1 + rng.normal(0, conf["credit_multiplicative_noise_sd"], len(credit)))
        direction = np.sign(changed.rolling(12).sum().pct_change(12).diff(3))
        pmi = data.pmi[["manufacturing", "nonmanufacturing", "private"]]
        changed = pmi + rng.normal(0, conf["each_pmi_point_noise_sd"], pmi.shape)
        diffs = changed.rolling(12).mean().pct_change(12).diff(3)
        vote = pd.DataFrame(np.where(diffs > 0, 1, -1), index=diffs.index).sum(axis=1)
        vote = vote.where(diffs.notna().all(axis=1))
        state = data.state.copy()
        state["credit"] = s.last_asof(direction.set_axis(pd.DatetimeIndex(credit.released)), state.index)
        state["growth"] = np.sign(s.last_asof(vote.set_axis(pd.DatetimeIndex(data.pmi.available)), state.index))
        state["stage"] = s.state_path(state.money, state.credit, state.growth)
        name = f"raw_value_noise_{scenario:02}"
        a = s.run(data, s.Case(name), state=state)
        s.record(a, name, rows)
        real = s.run(data, s.Case(name), state=state, mode="actual", start="2025-11-06")
        s.record(real, "actual_" + name, rows)
    s.write(pd.DataFrame(rows), "input_value_uncertainty.csv")


def main():
    data = s.load()
    causality(data)
    raw_uncertainty(data)
    validate_saved()
    files = [s.HERE / "protocol.json", s.HERE / "input_uncertainty.json", *s.HERE.glob("*.py")]
    manifest = {str(p.relative_to(s.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (s.OUT / "code_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
