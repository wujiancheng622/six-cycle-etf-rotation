"""Replay the frozen high-risk rotation and its contemporaneous V12 comparator."""
import json

import numpy as np
import pandas as pd

from market import KEYS, PROXY_FILE, market_inputs
from provider_data import HERE, PRIOR, sha, write
from regimes import ASSETS_BY_STAGE, build

from allocation import risk_parity
from replay_local import Snapshots, apply_policy, engine, load_v12, measures, v12_amount

CONFIG = HERE / "protocol.json"
CFG = json.loads(CONFIG.read_text(encoding="utf-8"))
CAPITAL = CFG["capital"]
OUT = HERE / "accounts"
VARIANTS = ("GS_monthly", "GS_stage_switch", "GS_DR007_monthly", "Six_asset_ERC")


def freeze():
    frozen = HERE / "protocol.sha256"
    digest = sha(CONFIG)
    if frozen.exists():
        assert frozen.read_text(encoding="ascii").strip() == digest
    else:
        frozen.write_text(digest + "\n", encoding="ascii")


class Rotation:
    def __init__(self, name, returns, states):
        self.name = name
        self.returns = returns
        self.states = states
        self.receipts = []
        self.previous_assets = None
        self.pending = None
        self.last_rebalance = -2
        self.excluded_risk_dates = returns.attrs.get("excluded_risk_dates", {})

    def targets(self, day, known, i, nav):
        state_column = "stage_market" if self.name == "GS_DR007_monthly" else "stage"
        macro = self.states.loc[pd.Timestamp(known)]
        if pd.isna(macro[state_column]):
            return None, None
        phase = int(macro[state_column])
        assets = KEYS if self.name == "Six_asset_ERC" else ASSETS_BY_STAGE[phase]
        monthly = self.pending is None or day[:7] != known[:7]
        switched = self.name == "GS_stage_switch" and assets != self.previous_assets
        if monthly or switched:
            days = CFG["allocation"]["lookback_sessions"]
            exclusions = {date for asset in assets for date in self.excluded_risk_dates.get(asset, ())}
            train = self.returns.loc[:known, list(assets)].drop(index=sorted(exclusions), errors="ignore").tail(days)
            assert len(train) == days and np.isfinite(train.to_numpy()).all()
            cov = train.cov().to_numpy()
            assert np.linalg.eigvalsh(cov).min() > 0.
            fractions = risk_parity(cov)
            weights = pd.Series(fractions, index=assets).reindex(KEYS, fill_value=0.).to_numpy()
            self.previous_assets = assets
            self.last_rebalance = i
            self.pending = weights
            risk = fractions * (cov @ fractions)
            np.testing.assert_allclose(risk / risk.sum(), np.ones(len(risk)) / len(risk), atol=1e-8)
            self.receipts.append(dict(date=day, known=known, phase=phase, month_end=monthly,
                                      selected="/".join(assets), first_return=train.index[0],
                                      last_return=train.index[-1], money_score=float(macro.money_score),
                                      credit_direction=int(macro.credit), growth_direction=int(macro.growth),
                                      credit_date=macro.credit_published,
                                      growth_date=macro.growth_available.date().isoformat(),
                                      **dict(zip(KEYS, weights))))
            return weights, weights
        if i == self.last_rebalance + 1:
            return None, self.pending
        return None, None


def record(result, mode, start, name, slip):
    output = OUT / mode / start / f"{name}_{slip*10000:g}bp"
    output.mkdir(parents=True, exist_ok=True)
    for key in ("ledger", "weights", "pnl", "trades", "signals"):
        getattr(result, key).to_parquet(output / f"{key}.parquet")
    row = dict(mode=mode, strategy=name, slip_bps=slip*1e4,
               **measures(result.ledger, result.base_date, CAPITAL))
    nav = pd.concat([pd.Series([CAPITAL], index=[result.base_date]), result.ledger.nav])
    for days in (756, 1260):
        years = days//252
        row[f"worst_{years}y"] = (float(((nav/nav.shift(days))**(1/years)-1).min())
                                   if len(nav) > days else np.nan)
    row["rebalances"] = len(result.signals)
    row["end_nav"] = float(nav.iloc[-1])
    print(mode, start, name, f"slip={slip*10000:g}",
          *(round(row[k]*100, 3) for k in ("cagr", "vol", "mdd")), flush=True)
    return row


def main():
    freeze()
    market, adv, up, down, historical, sessions, coverage = market_inputs()
    phases, pmi = build(sessions)
    phases.to_csv(HERE / "daily_phases.csv", index_label="known_date", encoding="utf-8-sig")
    pmi.to_csv(HERE / "data/pmi_monthly.csv", index_label="month", encoding="utf-8-sig")
    write(HERE / "market_coverage.json", coverage)
    prior = Snapshots()
    baseline, ba, bu, bd = load_v12(prior)
    b_economic = v12_amount(baseline, prior)
    baseline_receipt = PRIOR / "accounts/actual_benchmark_2020-04-27/V12/ledger.parquet"
    if baseline_receipt.is_file():
        existing = pd.read_parquet(baseline_receipt)
        reproduced = engine.actual("V12", baseline, ba, bu, bd, None, "2020-04-27", .0009)
        pd.testing.assert_frame_equal(reproduced.ledger, existing)
    rows = []
    for mode, starts in (("amount", CFG["amount_starts"]), ("actual", CFG["actual_starts"])):
        for start in starts:
            for slip in (CFG["fees"]["slippage"], CFG["fees"]["stress_slippage"]):
                case_names = ("V12", *VARIANTS) if slip == CFG["fees"]["slippage"] else ("V12", VARIANTS[0])
                for name in case_names:
                    if name == "V12":
                        value = (engine.actual("V12", baseline, ba, bu, bd, None, start, slip)
                                 if mode == "actual" else engine.economic("V12", b_economic, None, start, slip))
                    else:
                        with apply_policy(lambda n, r, _: Rotation(n, r, phases), KEYS):
                            value = (engine.actual(name, market, adv, up, down, None, start, slip)
                                     if mode == "actual" else engine.economic(name, historical, None, start, slip))
                        assert (value.signals.last_return <= value.signals.known).all()
                        assert (value.signals.credit_date <= value.signals.known).all()
                        assert (value.signals.growth_date <= value.signals.known).all()
                    rows.append(record(value, mode, start, name, slip))
    frame = pd.DataFrame(rows)
    assert frame.end.eq(CFG["end"]).all()
    frame.to_csv(HERE / "comparison.csv", index=False, encoding="utf-8-sig")
    artifacts = [*HERE.glob("data/tushare/*/*.parquet"), *HERE.glob("source/*.json"),
                 HERE / "data/pbc/credit_monthly.csv", HERE / "data/pbc/raw_loans.csv", PROXY_FILE, *prior.hashes]
    write(HERE / "input_hashes.json", {p.relative_to(HERE.parent).as_posix(): sha(p) for p in artifacts})


if __name__ == "__main__":
    main()
