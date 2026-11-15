"""Compare the saved R6 audit with the newer local price and PMI reconstruction."""

import hashlib
import json
from pathlib import Path
from subprocess import run as render
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL = ROOT / "guosheng_original_reconstruction_20261114"
R6 = ROOT / "guosheng_policy_completion_20261114_r6"
RESULTS = HERE / "evidence"
sys.dont_write_bytecode = True
sys.path[:0] = [str(LOCAL), str(R6)]

from run_original import (ACTUAL_METALS, CFG, END, KEYS, MAIN, REPORT_END, SLIP, Inputs,
                          PublishedIndexRotation, apply_policy, assert_clean, check_frozen, engine,
                          log_attribution, run, switch_only, verify_account, votes_at, window)
from replay_policy import CATS, mkstate
from regimes import ASSETS_BY_STAGE


START = "2014-01-02"
REAL_START = "2025-11-06"
CAPITAL = 1_000_000
PAPER = (START, REPORT_END)
PERIODS = {
    "paper": PAPER, "full": (START, END), "2014-2019": (START, "2019-12-31"),
    "2015": ("2015-01-01", "2015-12-31"), "2020": ("2020-01-01", "2020-12-31"),
    "2020-2022": ("2020-01-01", "2022-12-31"), "2023-2025": ("2023-01-01", "2025-12-31"),
    "2025_through_Oct": ("2025-01-01", REPORT_END), "2026_H2": ("2026-07-01", END),
}
DOWNLOADS = Path("C:/Users/Administrator/Downloads")
BASKETS = {phase: "/".join(assets) for phase, assets in ASSETS_BY_STAGE.items()}
CHARTS = {
    "2024_fig24": {
        "pattern": "2024-12-28*国盛*中国经济六周期模型*.pdf", "page": 15,
        "sha256": "187312710f7684c0f2e3e0becc4e92512c2ec194727dde53e96a5924c0571185",
        "crop": (270, 960, 2740, 1040), "levels": (890, 742, 595, 445, 295, 149),
        "year_x": (51, 2080.5, 2565.5), "years": (2008, 2021, 2024),
        "xrange": (51, 2715), "end": "2024-11-30",
    },
    "2026_fig2": {
        "pattern": "2026-04-30*国盛*基于市场隐含状态与宏观偏离度*.pdf", "page": 3,
        "sha256": "68f09b47c85aa2d53f5d041f0dd1e199499809a36360f66e0571638a2ea79c8d",
        "crop": (960, 3060, 2195, 880), "levels": (710, 590, 470, 350, 230, 110),
        "year_x": (67, 176, 285, 394, 503, 612, 721, 831, 939, 1048, 1157, 1267,
                   1376, 1485, 1594, 1703, 1812, 1921, 2030), "years": tuple(range(2008, 2027)),
        "xrange": (67, 2061), "end": REPORT_END,
    },
}


def write(frame, name):
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def latest_local():
    check_frozen()
    source = LOCAL / "results" / "input_receipt.json"
    for name, digest in json.loads(source.read_text(encoding="utf-8"))["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    inputs = Inputs.load()
    state = pd.read_csv(LOCAL / "results" / "daily_signal.csv", index_col="known_date",
                        parse_dates=["known_date", "growth_available"], dtype={"stage": "Int64"})
    pd.testing.assert_index_equal(inputs.macro.index, state.index, check_names=False)
    metals = pd.read_csv(LOCAL / "shfe_metals_index.csv", index_col="date")
    history = inputs.hist.copy()
    assert history.index[history.index < ACTUAL_METALS].equals(metals.index)
    history.loc[metals.index, "metals"] = metals.close / metals.previous_close - 1
    coverage = json.loads((ROOT / "guosheng_six_regime_long_20261112" / "coverage.json").read_text(encoding="utf-8"))
    return state, SimpleNamespace(hist=history, benchmark=inputs.v12, macro=state,
                                 gs_quotes=inputs.gs_quotes, v12_quotes=inputs.v12_quotes,
                                 first_actual_return=coverage["actual_return_from"])


def r6_states(state):
    folder = R6 / "v2"
    for name, digest in json.loads((folder / "frozen_manifest.json").read_text(encoding="utf-8")).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    events = pd.read_csv(folder / "events_used.csv", parse_dates=["effective_date", "known_date", "usable_date"])
    pd.testing.assert_series_equal(events.usable_date, events[["effective_date", "known_date"]].max(axis=1),
                                   check_names=False)
    prior = pd.read_parquet(folder / "previous_documented_signals.parquet")
    expanded = pd.read_parquet(folder / "expanded_policy_only_signals.parquet")
    merged = mkstate(state, events)
    separate_votes, conflicts = votes_at(events, state.index)
    assert not conflicts
    pd.testing.assert_frame_equal(merged[CATS], separate_votes[CATS], check_dtype=False)
    pd.testing.assert_frame_equal(merged[["credit", "growth", "market"]], state[["credit", "growth", "market"]])
    pd.testing.assert_frame_equal(merged[CATS], expanded[CATS], check_index_type=False, check_names=False)
    print("R6 source hashes, dates, independent category scoring and unchanged latest PMI/credit verified", flush=True)
    return prior, expanded, merged, events


def check_web_ledger():
    table = pd.read_csv(R6 / "v2" / "comparison.csv")
    daily = pd.read_csv(R6 / "v2" / "daily_nav.csv.gz", index_col=0)
    checked = []
    for m in table.itertuples(index=False):
        series = daily.loc[m.start:m.end, m.runid]
        base = m.capital if m.start == m.account_start else daily.at[m.base_date, m.runid]
        nav = np.r_[base, series.to_numpy()]
        years = (pd.Timestamp(m.end) - pd.Timestamp(m.base_date)).days / 365.25
        r = nav[1:] / nav[:-1] - 1
        calculated = [nav[-1] / base - 1, (nav[-1] / base) ** (1 / years) - 1,
                      r.std(ddof=1) * np.sqrt(252), (1 - nav / np.maximum.accumulate(nav)).max()]
        assert len(series) == m.days and series.notna().all()
        np.testing.assert_allclose(calculated, [m.total, m.cagr, m.vol, m.mdd], atol=2e-12, rtol=0)
        checked.append(m.runid)
    print(f"Independently recalculated {4 * len(checked)} R6 metrics from archived NAV", flush=True)
    return len(checked)


def verify_policy_impact(prior, expanded, local, merged, events):
    spans = {"paper": PAPER, "full": (START, END), "recent": (REAL_START, END)}
    rows = []
    for label, old, new in (("R6_web_vs_sparse", prior, expanded), ("R6_on_latest_local", local, merged)):
        for period, (lo, hi) in spans.items():
            a, b = old.loc[lo:hi], new.loc[lo:hi]
            rows.append(dict(comparison=label, period=period, days=len(a),
                             different_money=int(a.money.ne(b.money).sum()),
                             different_phase=int(a.stage.ne(b.stage).sum()),
                             different_possible_basket=int(a.stage.map(BASKETS).ne(b.stage.map(BASKETS)).sum())))
    write(pd.DataFrame(rows), "policy_signal_impact.csv")
    last = merged.loc["2025-08-18":END, CATS]
    assert len(last) == 299 and last.eq(0).all().all()
    extra = events.loc[events.evidence_status.ne("inherited_pending_source_check")]
    write(extra, "r6_added_or_replaced_events.csv")
    for end in ("2015-03-31", "2020-04-30", "2025-08-29"):
        past = local.loc[:end]
        available = events.loc[events.usable_date.le(end)]
        pd.testing.assert_frame_equal(mkstate(past, available), merged.loc[:end])
    return mkstate(local, events.loc[events.tenor.ne("old LPR 1y")])


def read_ppm(file):
    with file.open("rb") as data:
        assert data.readline() == b"P6\n"
        width, height = map(int, data.readline().split())
        assert data.readline().strip() == b"255"
        return np.frombuffer(data.read(), dtype=np.uint8).reshape(height, width, 3)


def chart_pixels(spec, workdir):
    pdf, = DOWNLOADS.glob(spec["pattern"])
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == spec["sha256"]
    x, y, width, height = spec["crop"]
    page = str(spec["page"])
    file = workdir / "render"
    render(["pdftoppm", "-q", "-f", page, "-l", page, "-r", "400", "-x", str(x), "-y", str(y),
            "-W", str(width), "-H", str(height), str(pdf), str(file)], check=True, capture_output=True)
    output, = workdir.glob("render-*.ppm")
    pixels = read_ppm(output)[np.array(spec["levels"])]
    color = (pixels[:, :, 2].astype(int) > pixels[:, :, 0].astype(int) + 45)
    color &= (pixels[:, :, 0] < 120) & (pixels[:, :, 2] < 185)
    output.unlink()
    return np.where(color.sum(axis=0) == 1, color.argmax(axis=0) + 1, 0)


def year_position(dates):
    start = pd.to_datetime(dates.year.astype(str) + "-01-01")
    end = pd.to_datetime((dates.year + 1).astype(str) + "-01-01")
    return dates.year + (dates - start).days / (end - start).days


def phase_from_chart(spec, labels, dates):
    scale, zero = np.polyfit(spec["years"], spec["year_x"], 1)
    pixels = np.rint(scale * year_position(dates) + zero).astype(int)
    lo, hi = spec["xrange"]
    assert pixels.min() >= lo and pixels.max() < hi
    return labels[pixels]


def chart_comparison(old, expanded, local, merged, no_old_lpr):
    targets = pd.read_csv(LOCAL / "results" / "targets.csv", parse_dates=["date", "known"])
    targets = targets.loc[targets.case.eq("rebuilt_monthly") & targets.account_start.eq(START)]
    samples = []
    with TemporaryDirectory(prefix="guosheng_chart_", dir=HERE) as temp:
        for chart, spec in CHARTS.items():
            labels = chart_pixels(spec, Path(temp))
            for mode, dates in (("daily", local.loc[START:spec["end"]].index),
                                ("monthly_known", pd.DatetimeIndex(targets.loc[targets.date.le(spec["end"]), "known"]))):
                offsets = np.array([phase_from_chart(spec, labels, dates + pd.Timedelta(days=delta))
                                    for delta in range(-10, 11)])
                accepted = (offsets != 0).all(axis=0) & (offsets == offsets[10]).all(axis=0)
                # 0 means the date is too close to a visible boundary for a reliable reading.
                phase = np.where(accepted, offsets[10], 0)
                for d, p in zip(dates, phase):
                    samples.append(dict(chart=chart, mode=mode, known_date=d, author_chart_phase=int(p),
                                        sparse=int(old.at[d, "stage"]), r6=int(expanded.at[d, "stage"]),
                                        local=int(local.at[d, "stage"]), combined=int(merged.at[d, "stage"]),
                                        without_old_lpr=int(no_old_lpr.at[d, "stage"])))
    samples = pd.DataFrame(samples)
    write(samples, "chart_point_comparison.csv")
    rows = []
    for (chart, mode), points in samples.groupby(["chart", "mode"]):
        readable = points.loc[points.author_chart_phase.ne(0)]
        reference = readable.author_chart_phase
        for name in ("sparse", "r6", "local", "combined", "without_old_lpr"):
            phase = readable[name]
            rows.append(dict(chart=chart, mode=mode, version=name, dates=len(points), readable=len(readable),
                             exact_phase=int(reference.eq(phase).sum()),
                             same_basket=int(reference.map(BASKETS).eq(phase.map(BASKETS)).sum())))
    common = samples.pivot(index=["mode", "known_date"], columns="chart", values="author_chart_phase").dropna()
    common = common.loc[common.gt(0).all(axis=1)]
    for mode, points in common.groupby(level="mode"):
        a, b = points["2024_fig24"], points["2026_fig2"]
        rows.append(dict(chart="both_author_charts", mode=mode, version="author_vs_author", dates=len(points),
                         readable=len(points), exact_phase=int(a.eq(b).sum()),
                         same_basket=int(a.map(BASKETS).eq(b.map(BASKETS)).sum())))
    write(pd.DataFrame(rows), "chart_agreement.csv")


def run_published(views, state, allowed, monthly, switch, slip=SLIP):
    with apply_policy(lambda n, r, p: PublishedIndexRotation(r, state, allowed, monthly, switch), KEYS):
        result = engine.economic(MAIN, views.hist, None, START, slip)
    assert_clean(result, allowed["fcf"])
    for asset, date in allowed.items():
        present = result.signals[asset].gt(0)
        assert result.signals.loc[present, "first_return"].ge(date).all()
        assert result.weights.loc[:date, asset].eq(0).all()
        assert result.pnl.loc[:date, asset].eq(0).all()
    return result


def review_replays(views, local, merged):
    dates = views.hist.index
    allowed = {asset: dates[dates.searchsorted(day, side="right")]
               for asset, day in CFG["index_publication_control"]["dates"].items()}
    fcf_only = {"fcf": allowed["fcf"]}
    definitions = {
        "monthly": (True, False, None), "month_and_switch": (True, True, None),
        "switch_only": (False, True, None), "monthly_fcf_cut": (True, False, fcf_only),
        "monthly_three_cut": (True, False, allowed), "month_and_switch_three_cut": (True, True, allowed),
    }
    original_ledgers = pd.read_parquet(LOCAL / "results" / "daily_ledger.parquet")
    original_cases = dict(monthly="rebuilt_monthly", month_and_switch="rebuilt_month_switch",
                          switch_only="rebuilt_switch_only", monthly_fcf_cut="rebuilt_monthly_fcf_pit",
                          monthly_three_cut="rebuilt_monthly_indices_pit",
                          month_and_switch_three_cut="rebuilt_month_switch_indices_pit", V12="V12")
    accounts = {}

    def replay(mode, name, state, slip=SLIP):
        start = START if mode == "amount" else REAL_START
        if name == "V12":
            return run(views, mode, start, "V12", slip)
        monthly, switch, required = definitions[name]
        if required:
            return run_published(views, state, required, monthly, switch, slip)
        if not monthly:
            assert slip == SLIP
            return switch_only(views, mode, start, state)
        return run(views, mode, start, "GS_stage_switch" if switch else MAIN, slip, state=state)

    rows = []
    for mode in ("amount", "actual"):
        cases = list(definitions) if mode == "amount" else list(definitions)[:3]
        spans = PERIODS if mode == "amount" else {"recent": (REAL_START, END), "2026_H2": PERIODS["2026_H2"]}
        start = START if mode == "amount" else REAL_START
        for name in [*cases, "V12"]:
            for version, state in (("local", local), ("combined", merged)):
                a = replay(mode, name, state)
                accounts[(mode, name, version)] = a
                checked_name = "Six_asset_ERC" if name.endswith("_cut") else name
                verify_account(a, checked_name, mode)
                if version == "local":
                    original = original_ledgers.loc[original_ledgers.case.eq(original_cases[name])
                                                    & original_ledgers.account_start.eq(start)].set_index("date")
                    pd.testing.assert_frame_equal(a.ledger, original[a.ledger.columns], check_exact=True)
                elif mode == "actual":
                    for field in ("ledger", "weights", "pnl", "trades", "signals"):
                        pd.testing.assert_frame_equal(getattr(a, field), getattr(accounts[(mode, name, "local")], field),
                                                      check_exact=True)
                for period, (lo, hi) in spans.items():
                    m = window(a, lo, hi)
                    rows.append(dict(mode=mode, case=name, version=version, period=period, **m,
                                     target_count=int(a.signals.date.between(m["start"], m["end"]).sum())))
    write(pd.DataFrame(rows), "matched_replay.csv")
    targets = []
    for mode, name in (("amount", "monthly"), ("actual", "monthly")):
        a = accounts[(mode, name, "local")].signals.set_index("date")
        b = accounts[(mode, name, "combined")].signals.set_index("date")
        assert a.index.equals(b.index)
        m = pd.DataFrame(dict(date=a.index, known=a.known.to_numpy(), old_phase=a.phase.to_numpy(),
                              new_phase=b.phase.to_numpy(), old_basket=a.selected.to_numpy(),
                              new_basket=b.selected.to_numpy(), mode=mode))
        targets.append(m)
    write(pd.concat(targets), "monthly_target_comparison.csv")
    frozen = pd.read_csv(R6 / "v2" / "comparison.csv")
    prior_amount = frozen.loc[frozen.account_start.eq(START) & frozen.capital.eq(500_000) & frozen.bp.eq(9)
                              & frozen.window.eq("paper")].set_index("case")
    print(f"R6 archived monthly net annualized {prior_amount.loc['expanded_policy_only', 'cagr']:.5f}", flush=True)
    print("Matched replay: local ledgers reproduced exactly; R6 policy swapped in only", flush=True)
    return replay, accounts


def old_lpr_control(replay, accounts, local, merged, no_old_lpr):
    rows = []
    for case in ("monthly", "month_and_switch", "month_and_switch_three_cut"):
        a = accounts[("amount", case, "local")]
        b = accounts[("amount", case, "combined")]
        c = replay("amount", case, no_old_lpr)
        for name, state, account in (("local", local, a), ("combined", merged, b),
                                     ("without_old_lpr", no_old_lpr, c)):
            metrics = window(account, *PAPER)
            rows.append(dict(case=case, policy=name, cagr=metrics["cagr"], mdd=metrics["mdd"],
                             phase_days_changed_from_local=int(state.loc[START:REPORT_END, "stage"].ne(
                                 local.loc[START:REPORT_END, "stage"]).sum())))
    write(pd.DataFrame(rows), "old_lpr_control.csv")


def old_lpr_receipt():
    return {
        "source": "https://www.boc.cn/fimarkets/lilv/fd32/201310/t20131028_2578984.html",
        "instrument": "旧制1年期贷款基础利率",
        "date": "2018-04-08",
        "bank_archive": {"value_pct": 4.30, "change_bp": 0},
        "r6_vendor_snapshot": {"value_pct": 4.31, "change_bp": 1},
        "treatment": "原样复跑网页端事件表。另做排除全部13条旧LPR的敏感性，不自动以任何一方为标准。",
    }


def listing_control(views, state, accounts):
    """Keep unavailable ETF legs in cash and require actual post-listing risk inputs."""
    rows = []
    cases = {"monthly": run_published(views, state, views.first_actual_return, True, False),
             "month_and_switch": run_published(views, state, views.first_actual_return, True, True),
             "V12": accounts[("amount", "V12", "combined")]}
    for name, a in cases.items():
        for period, dates in PERIODS.items():
            m = window(a, *dates)
            ledger = a.ledger.loc[slice(*dates)]
            rows.append(dict(case=name, period=period, mean_cash=(ledger.cash / ledger.nav).mean(), **m))
    write(pd.DataFrame(rows), "listed_etfs_only_control.csv")


def risk_checks(replay, accounts, state, views):
    name = "month_and_switch_three_cut"
    cost_rows = []
    baseline = accounts[("amount", "V12", "combined")]
    hurdle = window(baseline, *PAPER)["cagr"]

    def margin(bp):
        a = replay("amount", name, state, bp / 10_000)
        m = window(a, *PAPER)
        excess = m["cagr"] - hurdle
        cost_rows.append(dict(slippage_bp=bp, v12_slippage_bp=9, cagr=m["cagr"], v12_cagr=hurdle,
                              excess=excess, mdd=m["mdd"]))
        return excess

    lower, upper = 9, 160
    assert margin(lower) > 0 and margin(upper) < 0
    margin(20)
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if margin(middle) > 0:
            lower = middle
        else:
            upper = middle
    write(pd.DataFrame(cost_rows).sort_values("slippage_bp"), "model_cost_threshold.csv")

    rows = []
    for mode, name, period, dates in (("actual", "month_and_switch", "recent", (REAL_START, END)),
                                      ("amount", "month_and_switch_three_cut", "paper", PAPER)):
        account = accounts[(mode, name, "combined")]
        v12 = accounts[(mode, "V12", "combined")]
        r, _, contributions = log_attribution(account)
        vr, _, _ = log_attribution(v12)
        a = np.log1p(r.loc[dates[0]:dates[1]])
        b = np.log1p(vr.loc[dates[0]:dates[1]])
        assert a.index.equals(b.index)
        years = (pd.Timestamp(a.index[-1]) - pd.Timestamp(account.base_date)).days / 365.25
        best = a.nlargest(int(np.ceil(len(a) * .01))).index
        months = pd.to_datetime(a.index).to_period("M")
        best_months = (a - b).groupby(months).sum().nlargest(3).index
        for test, retained in (("base", np.ones(len(a), dtype=bool)), ("omit_top_1pct_gs_days", ~a.index.isin(best)),
                               ("omit_top_3_relative_months", ~months.isin(best_months))):
            rows.append(dict(period=period, case=name, test=test, omitted_sessions=int((~retained).sum()),
                             cagr=np.expm1(a[retained].sum() / years), v12_cagr=np.expm1(b[retained].sum() / years)))
        if mode == "actual":
            top = contributions.loc[a.index, list(KEYS)].sum().nlargest(2)
            cash = run(views, mode, REAL_START, "GS_stage_switch", state=state, cash_for=tuple(top.index))
            rows.append(dict(period=period, case=name, test="two_largest_assets_in_cash", omitted_sessions=0,
                             cagr=window(cash, *dates)["cagr"], v12_cagr=np.expm1(b.sum() / years),
                             top_assets="/".join(top.index), share_of_gs_net_log=top.sum() / a.sum()))
    write(pd.DataFrame(rows), "concentration_controls.csv")
    print(f"Strict three-index model cost break-even in {lower}–{upper} bp with V12 fixed at 9 bp", flush=True)


def main():
    def guard(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("This replay uses frozen local data")

    sys.addaudithook(guard)
    RESULTS.mkdir(exist_ok=True)
    local, views = latest_local()
    prior, expanded, merged, events = r6_states(local)
    check_web_ledger()
    no_old_lpr = verify_policy_impact(prior, expanded, local, merged, events)
    r6_2018 = events.loc[events.tenor.eq("old LPR 1y") & events.effective_date.eq("2018-04-08")]
    assert len(r6_2018) == 1 and float(r6_2018.iloc[0].rate_after) == 4.31
    (RESULTS / "old_lpr_source_conflict.json").write_text(json.dumps(old_lpr_receipt(), ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    chart_comparison(prior, expanded, local, merged, no_old_lpr)
    replay, accounts = review_replays(views, local, merged)
    old_lpr_control(replay, accounts, local, merged, no_old_lpr)
    listing_control(views, merged, accounts)
    risk_checks(replay, accounts, merged, views)


if __name__ == "__main__":
    main()
