"""Reconstruct the published high-risk Guosheng rotation from archived local data."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path[:0] = [
    str(ROOT / "guosheng_parameter_robustness_20261113"),
    str(ROOT / "guosheng_fcf_cutoff_20261113"),
    str(ROOT / "guosheng_six_regime_long_20261112"),
]

from study_inputs import Inputs, check_frozen, DATA, PROVIDER
from replay_long import (ASSETS_BY_STAGE, BASE, CAPITAL, KEYS, MAIN, Rotation, SLIP,
                         apply_policy, engine, load_account, log_attribution, run,
                         state_path, verify_account)
from regimes import last_asof
from replay_fcf_cutoff import LOOKBACK, assert_clean, run_variant, trailing, window
from run_study import risk_parity


PROTOCOL = HERE / "protocol.json"
CFG = json.loads(PROTOCOL.read_text(encoding="utf-8"))
CATEGORIES = CFG["policy_categories"]
REPORT_END = CFG["end_paper"]
END = CFG["end_local"]
PRIOR_EVENTS = ROOT / "guosheng_source_alignment_20261113" / "public_events.csv"
EXTRA_EVENTS = HERE / "extra_policy_events.csv"
PMI_DATES = HERE / "pmi_release_dates.csv"
NBS_ARCHIVE = HERE / "nbs_pmi_archive.csv"
NBS_SUPPLEMENT = HERE / "nbs_pmi_supplement.csv"
METALS = HERE / "shfe_metals_index.csv"
WINDOW = pd.Timedelta(days=CFG["policy_window_calendar_days"])
ACTUAL_METALS = "2019-12-25"
OUT = HERE / "results"


def save_csv(frame, filename, index=False):
    frame.to_csv(OUT / filename, index=index, encoding="utf-8-sig")


def freeze():
    check_frozen()
    sha256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    receipt = PROTOCOL.with_suffix(".sha256")
    if receipt.exists():
        assert receipt.read_text(encoding="ascii").strip() == sha256
    else:
        receipt.write_text(sha256 + "\n", encoding="ascii")


def input_events(inputs):
    prior = pd.read_csv(PRIOR_EVENTS, parse_dates=["date"]).rename(columns={"date": "effective_date"})
    prior = prior.assign(known_date=prior.effective_date, evidence="前轮已核对的公开事件", dataset="prior")
    extra = pd.read_csv(EXTRA_EVENTS, parse_dates=["effective_date", "known_date"]).assign(dataset="added")
    snapshots = []
    for name, changes in inputs.policy.items():
        lpr = name.startswith("lpr_")
        changes = changes.dropna().loc[lambda x: x.ne(0)]
        snapshots.append(pd.DataFrame({
            "category": "loan" if lpr else "reserve",
            "effective_date": changes.index,
            "known_date": changes.index,
            "vote": -np.sign(changes.to_numpy()).astype(int),
            "evidence": name,
            "source": "Tushare shibor_lpr frozen snapshot" if lpr else
                "Eastmoney frozen reserve_ratio_changes.json; max(published, effective)",
            "dataset": "series",
        }))
    events = pd.concat([prior, extra, *snapshots], ignore_index=True)
    assert set(events.category) == set(CATEGORIES)
    assert events.vote.isin([-1, 1]).all() and events.known_date.notna().all()
    assert events.effective_date.notna().all() and events.source.notna().all()
    assert not events.duplicated(["category", "effective_date", "known_date", "evidence"]).any()
    events["usable_date"] = events[["effective_date", "known_date"]].max(axis=1)
    return events.sort_values(["usable_date", "category", "evidence"], ignore_index=True)


def votes_at(events, dates):
    """Use the most recent known category event; conflicting same-day votes are neutral."""
    votes = pd.DataFrame(0, index=dates, columns=CATEGORIES, dtype=np.int8)
    conflicts = []
    for category, category_events in events.groupby("category", sort=False):
        daily = category_events.groupby("usable_date").vote.agg(["min", "max"])
        conflicted = daily.index[daily["min"].ne(daily["max"])]
        conflicts.extend((category, date.date().isoformat()) for date in conflicted)
        direction = daily["min"].where(daily["min"].eq(daily["max"]), 0)
        observed = last_asof(pd.Series(daily.index, index=daily.index), dates)
        active = dates - observed < WINDOW
        votes[category] = last_asof(direction, dates).where(active, 0).astype(np.int8)
    return votes, conflicts


def growth_at(pmi, dates, replacements, official=None):
    pmi = pmi.copy()
    assert pmi.index.is_unique and replacements.index.is_unique
    assert replacements.index.difference(pmi.index).empty
    np.testing.assert_allclose(pmi.loc[replacements.index, "private"], replacements.private, rtol=0, atol=1e-12)
    assert replacements[["private_released", "official_released"]].notna().all().all()
    pmi["verified_official_date"] = pd.NaT
    pmi["private_released_used"] = pd.to_datetime(pmi.source_release_date)
    pmi["official_date_source"] = ""
    if official is not None:
        pmi.loc[official.index, "verified_official_date"] = official.official_released
        pmi.loc[official.index, "official_date_source"] = official.source
    pmi.loc[replacements.index, "verified_official_date"] = replacements.official_released
    pmi.loc[replacements.index, "private_released_used"] = replacements.private_released
    pmi.loc[replacements.index, "private_source"] = replacements.private_source
    pmi.loc[replacements.index, "official_date_source"] = replacements.official_source
    confirmed = pmi.verified_official_date.notna() & pmi.private_released_used.notna()
    pmi.loc[confirmed, "available"] = pmi.loc[confirmed, ["verified_official_date", "private_released_used"]].max(axis=1)
    release = pd.DatetimeIndex(pmi.available)
    assert release.is_unique and release.is_monotonic_increasing
    directions = pmi.direction.set_axis(release)
    return (last_asof(directions, dates), last_asof(pd.Series(release, index=release), dates), pmi)


def rebuild(inputs, events, pmi_releases, official):
    macro = inputs.macro
    dates = macro.index
    policy, conflicts = votes_at(events, dates)
    sparse_events = events.loc[events.dataset.ne("added")]
    sparse_policy, _ = votes_at(sparse_events, dates)
    cutoff = pd.Timestamp(END) - pd.Timedelta(days=120)
    past_policy, _ = votes_at(events.loc[events.usable_date.le(cutoff)], dates[dates <= cutoff])
    pd.testing.assert_frame_equal(past_policy, policy.loc[:cutoff])
    assert policy.abs().max().le(1).all()
    direction, available, pmi = growth_at(inputs.pmi, dates, pmi_releases, official)
    sparse_direction, sparse_available, _ = growth_at(inputs.pmi, dates, pmi_releases.loc[["2025-10"]])
    market_holes = dates.difference(inputs.rates.index)
    assert market_holes.empty, ("Need the one-month ChinaBond market fallback", market_holes)
    smooth = inputs.rates.rolling(30).mean()
    now = last_asof(smooth, dates)
    before = last_asof(smooth, dates - WINDOW)
    np.testing.assert_allclose(np.sign(before - now.set_axis(before.index)), macro.market,
                               rtol=0, atol=0, equal_nan=True)
    c = inputs.credit
    pulse = c.middle_long_new_yi.rolling(12).sum().pct_change(12, fill_method=None)
    np.testing.assert_allclose(pulse, c.pulse, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_allclose(pulse.diff(3), c.credit_change, atol=1e-12, rtol=0, equal_nan=True)
    pmi_delta = pmi[["manufacturing", "nonmanufacturing", "private"]].rolling(12).mean().pct_change(12).diff(3)
    pmi_votes = pd.DataFrame(np.where(pmi_delta.gt(0), 1, -1), index=pmi_delta.index).sum(axis=1)
    np.testing.assert_allclose(np.sign(pmi_votes.where(pmi_delta.notna().all(axis=1))), pmi.direction,
                               rtol=0, atol=1e-12, equal_nan=True)

    def panel(votes, growth, release):
        score = (votes.sum(axis=1) + macro.market) / 8
        state = votes.assign(market=macro.market, money_score=score, money=np.sign(score), credit=macro.credit,
                             credit_pulse=macro.credit_pulse, credit_published=macro.credit_published,
                             growth=growth, growth_available=release)
        state["stage"] = state_path(state.money, state.credit, state.growth)
        assert not state.loc["2014":, "stage"].isna().any()
        assert (state.growth_available.dropna() <= state.growth_available.dropna().index).all()
        return state

    sparse = panel(sparse_policy, sparse_direction, sparse_available)
    expanded = panel(policy, sparse_direction, sparse_available)
    corrected = panel(policy, direction, available)
    return sparse, expanded, corrected, pmi, conflicts, len(market_holes)


def exchange_metals(history):
    index = pd.read_csv(METALS, index_col="date")
    prelisting = history.index[history.index < ACTUAL_METALS]
    assert index.index.equals(prelisting), prelisting.difference(index.index)
    assert index.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert (index.close.gt(0) & index.previous_close.gt(0)).all()
    prior_close_error = (index.previous_close.iloc[1:] - index.close.shift().iloc[1:]).abs()
    assert prior_close_error.max() <= .011, prior_close_error.nlargest(8)
    updated = history.copy()
    updated.loc[prelisting, "metals"] = index.close / index.previous_close - 1
    assert updated.loc[ACTUAL_METALS:].equals(history.loc[ACTUAL_METALS:])
    assert updated.drop(columns="metals").equals(history.drop(columns="metals"))
    save_csv(pd.DataFrame({"old_proxy": history.loc[prelisting, "metals"],
                           "shfe_close_return": updated.loc[prelisting, "metals"]}),
             "metals_proxy_comparison.csv", index=True)
    return updated, float(prior_close_error.max()), len(index)


class SwitchOnlyRotation(Rotation):
    """Re-estimate the risk budget only when the tradable basket changes."""

    def targets(self, day, known, i, nav):
        phase = self.states.at[pd.Timestamp(known), "stage"]
        if pd.notna(phase) and ASSETS_BY_STAGE[int(phase)] != self.previous_assets:
            return super().targets(day, known, i, nav)
        if i == self.last_rebalance + 1:
            return None, self.pending
        return None, None


class PublishedIndexRotation(Rotation):
    """Require the covariance history to follow each relevant index publication."""

    def __init__(self, returns, states, first_returns, monthly, switch):
        super().__init__("GS_stage_switch", returns, states)
        self.monthly = monthly
        self.switch = switch
        self.first_returns = first_returns
        self.training = returns.copy()
        for asset, date in first_returns.items():
            self.training.loc[self.training.index < date, asset] = np.nan
        self.requested = None

    def targets(self, day, known, i, nav):
        macro = self.states.loc[pd.Timestamp(known)]
        if pd.isna(macro.stage):
            return None, None
        phase = int(macro.stage)
        requested = ASSETS_BY_STAGE[phase]
        first = self.pending is None
        month_open = day[:7] != known[:7]
        reweight = first or (self.monthly and month_open) or (self.switch and requested != self.requested)
        if not reweight:
            if i == self.last_rebalance + 1:
                return None, self.pending
            return None, None
        assets = tuple(asset for asset in requested if asset not in self.first_returns or
                       len(trailing(self.training, self.excluded_risk_dates, known, (asset,))) == LOOKBACK)
        train = trailing(self.training, self.excluded_risk_dates, known, assets)
        assert len(train) == LOOKBACK
        for asset in self.first_returns.keys() & set(assets):
            assert train.index[0] >= self.first_returns[asset]
        fractions = np.ones(len(assets))
        if len(assets) > 1:
            covariance = train.cov().to_numpy()
            assert np.linalg.eigvalsh(covariance).min() > 0.
            fractions = risk_parity(covariance)
            risk = fractions * (covariance @ fractions)
            np.testing.assert_allclose(risk / risk.sum(), 1 / len(assets), atol=1e-8)
        cash = (len(requested) - len(assets)) / len(requested)
        weights = (pd.Series(fractions * (1 - cash), index=assets)
                   .reindex(KEYS, fill_value=0.).to_numpy())
        self.requested = requested
        self.last_rebalance = i
        self.pending = weights
        self.receipts.append(dict(date=day, known=known, phase=phase, month_end=month_open or first,
                                  selected="/".join(assets), requested="/".join(requested),
                                  first_return=train.index[0], last_return=train.index[-1],
                                  credit_date=macro.credit_published, growth_date=macro.growth_available.date().isoformat(),
                                  fcf_eligible="fcf" in assets, cash_target=cash, **dict(zip(KEYS, weights))))
        return weights, weights


def switch_only(inputs, mode, start, state):
    kind = "GS_stage_switch"
    with apply_policy(lambda n, r, _: SwitchOnlyRotation(kind, r, state), KEYS):
        if mode == "actual":
            q, adv, up, down = inputs.gs_quotes
            account = engine.actual(kind, q, adv, up, down, None, start, SLIP)
        else:
            account = engine.economic(kind, inputs.hist, None, start, SLIP)
    assert not account.signals.selected.eq(account.signals.selected.shift()).any()
    return account


def period_rows(account, mode, start, case):
    spans = [("full", start, END), ("2026_H2", "2026-07-01", END)]
    if mode == "amount":
        spans.append(("paper", start, REPORT_END))
        spans.append(("2025_through_Oct", "2025-01-01", REPORT_END))
        spans.append(("2020_Covid", "2020-02-03", "2020-03-31"))
        spans.append(("2020-2022", "2020-01-01", "2022-12-31"))
        spans.append(("2023-2025", "2023-01-01", "2025-12-31"))
        spans.append(("2022_through_paper", "2022-01-01", REPORT_END))
        if start == "2014-01-02":
            spans.append(("2014-2019", start, "2019-12-31"))
            spans.append(("2018_through_paper", "2018-01-01", REPORT_END))
    spans.extend((f"{year}", f"{year}-01-01", f"{year}-12-31") for year in range(int(start[:4]), 2027))
    for label, lo, hi in spans:
        result = window(account, lo, hi)
        result["annual_one_way_turnover"] = result["annual_two_way_turnover"] / 2
        targets = int(account.signals.date.between(result["start"], result["end"]).sum())
        yield dict(case=case, mode=mode, account_start=start, period=label, rebalances=targets, **result)


def account_set(views, mode, start, sparse, policy, corrected, fcf_cutoff):
    original, improved = views
    old = run(original, mode, start, MAIN)
    prior = load_account(mode, start, MAIN)
    pd.testing.assert_frame_equal(old.ledger, prior.ledger)
    pd.testing.assert_frame_equal(old.signals, prior.signals)
    v12 = run(original, mode, start, "V12")
    pd.testing.assert_frame_equal(v12.ledger, load_account(mode, start, "V12").ledger)
    cases = dict(old_monthly=old, sparse_monthly=run(original, mode, start, MAIN, state=sparse),
                 expanded_policy=run(original, mode, start, MAIN, state=policy),
                 expanded_policy_pmi=run(original, mode, start, MAIN, state=corrected),
                 rebuilt_monthly=run(improved, mode, start, MAIN, state=corrected),
                 rebuilt_month_switch=run(improved, mode, start, "GS_stage_switch", state=corrected),
                 rebuilt_switch_only=switch_only(improved, mode, start, corrected), V12=v12)
    if mode == "amount":
        cases["rebuilt_monthly_fcf_pit"] = run_variant(improved.hist, corrected, start, fcf_cutoff, True)
        fcf_only = {"fcf": fcf_cutoff}
        dates = improved.hist.index
        published = {asset: dates[dates.searchsorted(date, side="right")]
                     for asset, date in CFG["index_publication_control"]["dates"].items()}
        assert published["fcf"] == fcf_cutoff
        for case, monthly, switch, allowed in (
            ("rebuilt_month_switch_fcf_pit", True, True, fcf_only),
            ("rebuilt_switch_only_fcf_pit", False, True, fcf_only),
            ("rebuilt_monthly_indices_pit", True, False, published),
            ("rebuilt_month_switch_indices_pit", True, True, published),
            ("rebuilt_switch_only_indices_pit", False, True, published),
        ):
            with apply_policy(lambda n, r, p: PublishedIndexRotation(r, corrected, allowed, monthly, switch), KEYS):
                account = engine.economic(case, improved.hist, None, start, SLIP)
            assert_clean(account, fcf_cutoff)
            for asset, date in allowed.items():
                signals = account.signals
                assert signals.loc[signals[asset].gt(0), "first_return"].ge(date).all()
                assert account.weights.loc[account.weights.index < date, asset].eq(0).all()
                assert account.pnl.loc[account.pnl.index < date, asset].eq(0).all()
            cases[case] = account
    else:
        pd.testing.assert_frame_equal(cases["expanded_policy_pmi"].ledger, cases["rebuilt_monthly"].ledger)
    return cases


def save_runs(views, sparse, policy, corrected, fcf_cutoff):
    verification, comparison, attribution, navs, targets, orders, positions = [], [], [], [], [], [], []
    cohorts = [("amount", start) for start in CFG["start_amount"]] + [("actual", CFG["start_actual"])]
    for mode, start in cohorts:
        cases = account_set(views, mode, start, sparse, policy, corrected, fcf_cutoff)
        dates = cases["V12"].ledger.index
        for case, account in cases.items():
            assert account.ledger.index.equals(dates), (mode, start, case)
            check_kind = "Six_asset_ERC" if case.endswith("_pit") else case
            check = verify_account(account, check_kind, mode)
            daily_trades = account.trades.groupby("date").agg(cost=("cost", "sum"), notional=("notional", "sum"),
                                                               trade_count=("side", "size"))
            actual = daily_trades.reindex(dates, fill_value=0)
            np.testing.assert_allclose(actual, account.ledger[actual.columns], rtol=0, atol=1e-8)
            assert (account.trades.notional.gt(0) & account.trades.cost.ge(5)).all()
            verification.append(dict(case=case, mode=mode, account_start=start, **check))
            comparison.extend(period_rows(account, mode, start, case))
            labels = dict(case=case, mode=mode, account_start=start)
            navs.append(account.ledger.rename_axis("date").reset_index().assign(**labels))
            targets.append(account.signals.assign(**labels))
            orders.append(account.trades.assign(**labels))
            positions.append(account.weights.rename_axis("date").reset_index().assign(**labels))
            if case != "V12" and start == "2014-01-02":
                daily, _, linked = log_attribution(account)
                for period, lo, hi in [("2015", "2015-01-01", "2015-12-31"),
                                       ("2020", "2020-01-01", "2020-12-31"),
                                       ("2020_Covid", "2020-02-03", "2020-03-31"),
                                       ("paper", start, REPORT_END)]:
                    asset_pnl = account.pnl.loc[lo:hi].sum()
                    log_contrib = linked.loc[lo:hi].sum()
                    assert np.isclose(log_contrib.sum(), np.log1p(daily.loc[lo:hi]).sum(), atol=1e-11)
                    for asset in (*KEYS, "fee"):
                        attribution.append(dict(**labels, period=period, asset=asset,
                                                pnl=asset_pnl.get(asset, -account.ledger.loc[lo:hi, "cost"].sum()),
                                                log_contribution=log_contrib[asset]))
        print(f"Checked {len(cases)} accounts for {mode} {start}", flush=True)

    comparison = pd.DataFrame(comparison)
    save_csv(comparison, "comparison.csv")
    save_csv(pd.DataFrame(verification), "verification.csv")
    save_csv(pd.DataFrame(attribution), "attribution.csv")
    save_csv(pd.concat(targets, ignore_index=True), "targets.csv")
    pd.concat(navs, ignore_index=True).to_parquet(OUT / "daily_ledger.parquet", index=False)
    pd.concat(orders, ignore_index=True).to_parquet(OUT / "orders.parquet", index=False)
    pd.concat(positions, ignore_index=True).to_parquet(OUT / "daily_weights.parquet", index=False)
    show = comparison.loc[comparison.period.isin(["paper", "full", "2026_H2"]),
                          ["mode", "account_start", "case", "period", "total", "cagr", "vol", "mdd"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def main():
    freeze()
    OUT.mkdir(exist_ok=True)
    inputs = Inputs.load()
    events = input_events(inputs)
    pmi_releases = pd.read_csv(PMI_DATES, index_col="month", parse_dates=["private_released", "official_released"])
    archive = pd.read_csv(NBS_ARCHIVE, parse_dates=["official_released"])
    supplement = pd.read_csv(NBS_SUPPLEMENT, parse_dates=["official_released"])
    official = pd.concat([archive, supplement], ignore_index=True)
    assert official.groupby("month").official_released.nunique().eq(1).all()
    official = official.drop_duplicates("month").set_index("month")
    sparse, policy, corrected, pmi, conflicts, market_holes = rebuild(inputs, events, pmi_releases, official)
    save_csv(events, "policy_events_used.csv")
    save_csv(pmi, "pmi_dates_used.csv", index=True)
    save_csv(corrected.rename_axis("known_date"), "daily_signal.csv", index=True)
    diff = pd.DataFrame({name: state.stage for name, state in (
        ("old", inputs.macro), ("sparse", sparse), ("policy", policy), ("corrected", corrected))})
    save_csv(diff.rename_axis("known_date"), "daily_phase_comparison.csv", index=True)
    hist, metal_error, n_metal = exchange_metals(inputs.hist)
    common = dict(benchmark=inputs.v12, macro=inputs.macro, gs_quotes=inputs.gs_quotes, v12_quotes=inputs.v12_quotes)
    views = (SimpleNamespace(hist=inputs.hist, **common), SimpleNamespace(hist=hist, **common))
    save_runs(views, sparse, policy, corrected, inputs.fcf_cutoff)
    inputs_hashed = (PROTOCOL, PRIOR_EVENTS, EXTRA_EVENTS, PMI_DATES, NBS_ARCHIVE, NBS_SUPPLEMENT, METALS,
                     BASE / "source" / "reserve_ratio_changes.json",
                     *(DATA / file for file in ("returns_gs.parquet", "returns_v12.parquet", "macro.parquet",
                                                "pmi_as_reported.csv", "credit_monthly.csv")))
    PROVIDER.write(OUT / "input_receipt.json", {
        "policy_sources": events.groupby(["category", "dataset"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "conflicts": conflicts, "market_double_missing_trading_sessions": market_holes,
        "new_private_pmi_date_sources": len(pmi_releases),
        "nbs_months_sourced": int(pmi.verified_official_date.notna().sum()),
        "pmi_availability_dates_changed_vs_old": int(pmi.available.ne(inputs.pmi.available).sum()),
        "shfe_replaced_return_sessions": n_metal,
        "max_shfe_previous_close_continuity_error_points": metal_error,
        "sha256": {str(path.relative_to(ROOT)): PROVIDER.sha(path) for path in inputs_hashed},
    })


if __name__ == "__main__":
    main()
