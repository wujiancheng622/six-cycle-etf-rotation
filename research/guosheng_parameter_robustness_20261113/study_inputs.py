"""Read the parent snapshots and rebuild only the chosen macro parameter."""
from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "guosheng_six_regime_long_20261112"
BASE = ROOT / "guosheng_six_regime_20261112"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT))
import replay_long as parent
import early_inputs as early
from regimes import last_asof, most_recent_event, load_json, state_path
from run_study import CAPITAL, KEYS, Rotation, apply_policy, engine, measures, risk_parity
sys.path.remove(str(PARENT))

DATA = PARENT / "data"
PROVIDER = parent.provider
CFG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
END = CFG["end"]
POLICIES = ["lpr_1y", "lpr_5y", "rrr_B", "rrr_S"]
DEFAULT = dict(credit_change_months=3, pmi_change_months=3, pmi_smoothing_months=12,
               money_market_average_observations=30, money_market_comparison_calendar_days=90,
               policy_tool_expiry_calendar_days=90, macro_release_lag_calendar_days=0)


def check_frozen():
    assert PROVIDER.sha(HERE / "protocol.json") == (HERE / "protocol.sha256").read_text(
        encoding="ascii").strip()
    parent.verify_frozen()


def fund_returns(returns, first):
    """Recover only actual-fund histories for true-ETF risk estimation."""
    coverage = json.loads((PARENT / "coverage.json").read_text(encoding="utf-8"))
    masked = returns.copy()
    for k, date in coverage["actual_return_from"].items():
        masked.loc[masked.index < date,k] = np.nan
    pd.testing.assert_frame_equal(masked.loc[first:].astype(float),returns.loc[first:].astype(float))
    return masked


def market_series():
    legacy = early.ts("shibor",start_date="20100101",end_date="20131231").set_index("date")["1w"].astype(float)
    shibor = pd.concat([legacy, PROVIDER.dated("shibor",None,date_column="date")["1w"].astype(float)])
    repo = PROVIDER.dated("repo_daily","DR007.IB").weight.astype(float)
    x = repo.combine_first(shibor).dropna().sort_index()
    x.index = pd.to_datetime(x.index,format="%Y%m%d")
    assert x.index.is_unique and x.index.is_monotonic_increasing
    return x


def policy_change_events():
    lpr = PROVIDER.dated("shibor_lpr",None,date_column="date").astype(float)
    lpr.index = pd.to_datetime(lpr.index,format="%Y%m%d")
    lpr = lpr.loc["2019-08-20":].where(lpr.gt(0))
    out = {f"lpr_{k}":lpr[k].dropna().round(3).diff() for k in ("1y","5y")}
    reserve = pd.DataFrame(load_json("reserve_ratio_changes.json")["result"]["data"])
    dt = lambda s:pd.to_datetime(s.str.replace(r"[^\d]","",regex=True),format="%Y%m%d",errors="coerce")
    release,effective = dt(reserve.PUBLISH_DATE),dt(reserve.TRADE_DATE)
    reserve["usable"] = effective.where(effective>=release,release)
    for k in ("B","S"):
        t=reserve.assign(change=pd.to_numeric(reserve[f"CHANGE_RATE_{k}"],errors="coerce"))
        out[f"rrr_{k}"]=t.dropna(subset=["usable","change"]).groupby("usable").change.sum().sort_index()
    return out


@dataclass
class Inputs:
    hist: pd.DataFrame
    v12: pd.DataFrame
    macro: pd.DataFrame
    risk_actual: pd.DataFrame
    gs_quotes: tuple
    v12_quotes: tuple
    credit: pd.DataFrame
    pmi: pd.DataFrame
    rates: pd.Series
    policy: dict
    fcf_cutoff: str

    @classmethod
    def load(cls):
        hist = pd.read_parquet(DATA / "returns_gs.parquet")
        v12 = pd.read_parquet(DATA / "returns_v12.parquet")
        macro = pd.read_parquet(DATA / "macro.parquet")
        credit = pd.read_csv(DATA/"credit_monthly.csv",index_col="month",parse_dates=["released"])
        pmi = pd.read_csv(DATA/"pmi_as_reported.csv",index_col="month",parse_dates=["available"])
        for frame in (hist,v12,macro):
            assert frame.index.is_unique and frame.index.is_monotonic_increasing
        q,adv,hi,lo,*_=early.market_inputs()
        assert hist.index[-1]==v12.index[-1]==q.returns.index[-1]==END
        risk_actual=fund_returns(hist,q.returns.index[0])
        pd.testing.assert_frame_equal(risk_actual.loc[q.returns.index],q.returns,check_index_type=False)
        b,badv,bhi,blo=early.load_v12(early.Snapshots())
        return cls(hist,v12,macro,risk_actual,(q,adv,hi,lo),(b,badv,bhi,blo),
                   credit,pmi,market_series(),policy_change_events(),
                   hist.index[hist.index.searchsorted("2024-08-15",side="right")])

    def build_macro(self,overrides=None):
        params = DEFAULT | (overrides or {})
        dates = self.macro.index
        c = self.credit
        g = self.pmi
        credit_change = c.pulse.diff(params["credit_change_months"])
        gd = (g[["manufacturing","nonmanufacturing","private"]]
              .rolling(params["pmi_smoothing_months"]).mean().pct_change(12,fill_method=None)
              .diff(params["pmi_change_months"]))
        votes = pd.DataFrame(np.where(gd>0,1,-1),index=g.index).sum(axis=1).where(gd.notna().all(axis=1))
        delay = pd.Timedelta(days=params["macro_release_lag_calendar_days"])
        c_dates = pd.DatetimeIndex(c.released+delay)
        g_dates = pd.DatetimeIndex(g.available+delay)
        smooth = self.rates.rolling(params["money_market_average_observations"]).mean()
        prev = last_asof(smooth,dates-pd.Timedelta(days=params["money_market_comparison_calendar_days"]))
        now = last_asof(smooth,dates)
        market = pd.Series(np.sign(prev.to_numpy()-now.to_numpy()),index=dates)
        ttl = pd.Timedelta(days=params["policy_tool_expiry_calendar_days"])
        policy = {}
        for name,change in self.policy.items():
            if ttl.days==90:
                policy[name] = most_recent_event(change,dates)
                continue
            events = -np.sign(change.dropna())
            events = events[events.ne(0)]
            direction = last_asof(events,dates)
            event_dates = last_asof(pd.Series(events.index,index=events.index),dates)
            policy[name] = direction.where((dates-event_dates)<ttl,0.).fillna(0.)
        tools = pd.DataFrame(policy,index=dates)
        score = tools.sum(axis=1)+market
        credit = np.sign(last_asof(credit_change.set_axis(c_dates),dates))
        growth = np.sign(last_asof(votes.set_axis(g_dates),dates))
        panels = tools.assign(market=market,money_score=score,money=np.sign(score),credit=credit,growth=growth,
                              credit_published=last_asof(pd.Series(c_dates.strftime("%Y-%m-%d"),index=c_dates),dates),
                              growth_available=last_asof(pd.Series(g_dates,index=g_dates),dates))
        panels["stage"] = state_path(panels.money,panels.credit,panels.growth)
        return panels


def calendar_targets(days, schedule, delay=0):
    """Map an execution date to its signal target date and prior market date."""
    calendar = pd.Series(days,index=days)
    timestamp = pd.to_datetime(days)
    monthly = calendar.groupby(timestamp.to_period("M"),sort=True)
    if schedule.startswith("D"):
        ordinal = int(schedule[1:])
        selected = [g.iloc[ordinal-1] if len(g)>=ordinal else g.iloc[-1]
                    for _,g in monthly if len(g)>=ordinal or g.index[-1][:7]!=END[:7]]
    elif schedule=="ME":
        selected = [g.iloc[-1] for _,g in monthly if g.index[-1][:7]!=END[:7]]
    elif schedule=="W":
        selected = calendar.groupby(timestamp.to_period("W-SUN"),sort=True).first().tolist()
    elif schedule=="Q":
        selected = calendar.groupby(timestamp.to_period("Q"),sort=True).first().tolist()
    else:
        raise ValueError(schedule)
    out={}
    for d in selected:
        j=days.get_loc(d)
        if j>0 and j+delay<len(days):
            out[days[j+delay]]=(d,days[j-1])
    return out
