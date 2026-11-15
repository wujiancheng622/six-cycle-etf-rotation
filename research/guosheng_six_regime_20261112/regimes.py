"""Dated macro state reconstruction from public series and four policy tools."""
import json

import numpy as np
import pandas as pd

from provider_data import DATA, HERE, dated, get, write
from pbc_loans import ARCHIVE

STAGE_MAP = {
    (1, 1, -1): 1, (1, 1, 1): 2, (-1, 1, 1): 3,
    (-1, -1, 1): 4, (-1, -1, -1): 5, (1, -1, -1): 6,
}
ASSETS_BY_STAGE = {
    1: ("growth", "gold"),
    2: ("growth", "metals"),
    3: ("fcf", "metals"),
    4: ("fcf", "bond", "metals"),
    5: ("dividend", "bond", "gold"),
    6: ("dividend", "bond", "gold"),
}


def load_json(name):
    return json.loads((HERE / "source" / name).read_text(encoding="utf-8"))


def last_asof(series, dates):
    assert series.index.is_unique and series.index.is_monotonic_increasing
    return series.reindex(dates, method="ffill")


def pmi_data():
    official = get("cn_pmi", start_m="201301", end_m="202610").set_index("month").sort_index()
    official.index = pd.to_datetime(official.index, format="%Y%m").to_period("M")
    private = load_json("caixin_manufacturer.json")["data"]
    caixin = pd.Series(private["data"], index=pd.to_datetime(private["month"]).to_period("M"),
                       dtype=float, name="private")
    calendar = pd.DataFrame(load_json("caixin_release_history.json"),
                            columns=["released", "value", "forecast", "previous"])
    calendar["released"] = pd.to_datetime(calendar.released)
    calendar["value"] = pd.to_numeric(calendar.value, errors="coerce")
    # The early calendar combines preliminary and final releases.
    # A release in days 1–10 is for the prior month. Final releases on
    # the 28th or later belong to the current month.
    calendar = calendar.loc[calendar.released.dt.day.le(10) | calendar.released.dt.day.ge(28)]
    calendar["month"] = calendar.released.dt.to_period("M") - calendar.released.dt.day.le(10).astype(int)
    calendar = calendar.dropna(subset=["value"]).sort_values("released")
    early = calendar.loc[calendar.month.lt(caixin.index[0])].drop_duplicates("month", keep="last")
    series = pd.concat([early.set_index("month").value, caixin]).sort_index()
    primary = pd.DataFrame({"manufacturing": official.pmi010000.astype(float),
                            "nonmanufacturing": official.pmi020100.astype(float),
                            "private": series})
    primary = primary.loc["2013-01":"2026-10"]
    assert not primary.isna().any().any(), primary[primary.isna().any(axis=1)]
    assert primary.index.equals(pd.period_range("2013-01", "2026-10", freq="M"))
    # Use a calendar date only when its value agrees with the published series.
    aligned = calendar.loc[calendar.value.eq(calendar.month.map(series))].drop_duplicates("month", keep="first")
    release = aligned.set_index("month").released.reindex(primary.index)
    conservative = pd.Series((primary.index+1).to_timestamp()+pd.Timedelta(days=9), index=primary.index)
    availability = release.fillna(conservative)
    nbs_guard = (primary.index+1).to_timestamp()+pd.Timedelta(days=2)
    availability = availability.where(availability > nbs_guard, nbs_guard)
    assert availability.is_monotonic_increasing and availability.is_unique
    rolling = primary.rolling(12).mean()
    impulses = rolling.pct_change(12, fill_method=None)
    delta = impulses.diff(3)
    signed = pd.DataFrame(np.where(delta > 0, 1, -1), index=delta.index, columns=delta.columns)
    votes = signed.sum(axis=1).where(delta.notna().all(axis=1))
    return primary.join(delta.add_suffix("_change")).assign(
        votes=votes, direction=np.sign(votes), available=availability,
        source_release_date=release, private_calendar_match=release.notna(),
    )


def market_rate(dates):
    repo = dated("repo_daily", "DR007.IB").weight.astype(float)
    shibor = dated("shibor", None, date_column="date")["1w"].astype(float)
    dates_bank = repo.index.union(shibor.index)
    series = repo.reindex(dates_bank).combine_first(shibor.reindex(dates_bank)).dropna()
    series.index = pd.to_datetime(series.index, format="%Y%m%d")
    mean = series.rolling(30).mean()
    now = last_asof(mean, dates)
    before = last_asof(mean, dates - pd.Timedelta(days=90))
    return pd.Series(np.sign(before.to_numpy()-now.to_numpy()), index=dates, name="market")


def most_recent_event(change, dates):
    # Positive score is a cut or a reserve ratio reduction.
    events = -np.sign(change.dropna())
    events = events[events.ne(0)]
    direction = last_asof(events, dates)
    past = last_asof(pd.Series(events.index, index=events.index), dates)
    return direction.where((dates - past) < pd.Timedelta(days=90), 0.).fillna(0.)


def policy_events(dates):
    lpr = dated("shibor_lpr", None, date_column="date").astype(float)
    lpr.index = pd.to_datetime(lpr.index, format="%Y%m%d")
    # August 2019 starts the reformed quoting regime. Do not score the reset.
    lpr = lpr.loc["2019-08-20":]
    lpr = lpr.where(lpr.gt(0))
    output = {}
    for k in ("1y", "5y"):
        output[f"lpr_{k}"] = most_recent_event(lpr[k].dropna().round(3).diff(), dates)
    reserve = pd.DataFrame(load_json("reserve_ratio_changes.json")["result"]["data"])
    release = pd.to_datetime(reserve.PUBLISH_DATE.str.replace(r"[^\d]", "", regex=True),
                             format="%Y%m%d", errors="coerce")
    effective = pd.to_datetime(reserve.TRADE_DATE.str.replace(r"[^\d]", "", regex=True),
                               format="%Y%m%d", errors="coerce")
    reserve["usable"] = effective.where(effective >= release, release)
    for k in ("B", "S"):
        changes = reserve.assign(change=pd.to_numeric(reserve[f"CHANGE_RATE_{k}"], errors="coerce"))
        changes = changes.dropna(subset=["usable", "change"])
        changes = changes.groupby("usable").change.sum().sort_index()
        output[f"rrr_{k}"] = most_recent_event(changes, dates)
    return pd.DataFrame(output, index=dates)


def state_path(money, credit, growth):
    states = []
    stage = None
    for directions in zip(money, credit, growth):
        if np.isfinite(directions).all():
            candidate = STAGE_MAP.get(tuple(directions))
            if candidate is not None:
                stage = candidate
        states.append(stage)
    return pd.Series(states, index=money.index, dtype="Int64")


def build(dates):
    credit = pd.read_csv(ARCHIVE / "credit_monthly.csv", index_col="month", encoding="utf-8-sig")
    growth = pmi_data()
    mkt = market_rate(dates)
    policies = policy_events(dates)
    score = pd.concat([mkt, policies], axis=1).sum(axis=1, min_count=len(policies.columns)+1)
    c = credit.set_index(pd.to_datetime(credit.released))
    g = growth.set_index(growth.available)
    credit_direction = np.sign(last_asof(c.credit_change, dates))
    growth_direction = last_asof(g.direction, dates)
    panels = policies.assign(market=mkt, money_score=score, money=np.sign(score),
                             credit=credit_direction, growth=growth_direction,
                             credit_pulse=last_asof(c.pulse, dates),
                             credit_published=last_asof(c.released, dates),
                             growth_available=last_asof(g.available, dates),
                             growth_votes=last_asof(g.votes, dates))
    panels["stage"] = state_path(panels.money, panels.credit, panels.growth)
    panels["stage_market"] = state_path(mkt, panels.credit, panels.growth)
    assert (panels.credit_published.dropna() <=
            panels.index[panels.credit_published.notna()].strftime("%Y-%m-%d")).all()
    assert (panels.growth_available.dropna() <= panels.index[panels.growth_available.notna()]).all()
    return panels, growth
