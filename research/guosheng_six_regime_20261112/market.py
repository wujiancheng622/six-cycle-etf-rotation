"""Build the six actual ETF carriers and the pre-listing amount-model inputs."""
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

from provider_data import CARRIERS, END, PRIOR, WORK, dated, get
from regimes import load_json

sys.path.insert(0, str(PRIOR))
from market_ht import Snapshots, calendar

KEYS = tuple(CARRIERS)
PROXY_FILE = WORK / "p75_risk15_codex_20261106/input/global_economic_returns.csv"


def market_inputs():
    session_digits = calendar(Snapshots())
    sessions = pd.to_datetime(session_digits, format="%Y%m%d")
    quote = {}
    total_returns = {}
    limits = {}
    dividends = []
    coverage = {}
    excluded_risk_dates = {}
    for key, code in CARRIERS.items():
        prices = dated("fund_daily", code).astype({c: float for c in ("open", "close", "pre_close", "vol")})
        listed = str(get("fund_basic", code, market="E").list_date.iloc[0])
        assert prices.index[0] >= listed and prices.index[-1] == END
        events = get("fund_div", code)
        events = events.loc[events.div_proc.eq("实施")].drop_duplicates(
            ["record_date", "ex_date", "pay_date", "div_cash"])
        cash = pd.Series(0., index=prices.index)
        for e in events.itertuples():
            assert e.imp_anndate <= e.record_date < e.ex_date <= e.pay_date, (code, e)
            if e.ex_date in cash:
                assert cash.loc[e.ex_date] == 0
                cash.loc[e.ex_date] = e.div_cash
                dividends.append(dict(key=key, event_id=f"{code}-{e.ex_date}",
                                      record_date=pd.Timestamp(e.record_date).date().isoformat(),
                                      ex_date=pd.Timestamp(e.ex_date).date().isoformat(),
                                      payment_date=pd.Timestamp(e.pay_date).date().isoformat(),
                                      cash_per_actual_share=e.div_cash))
        expected_ref = prices.close.shift(1)-cash
        np.testing.assert_allclose(prices.pre_close.iloc[1:], expected_ref.iloc[1:], atol=.0011, rtol=0,
                                   err_msg=code)
        days = session_digits[session_digits >= prices.index[0]]
        marks = prices.reindex(days)
        valid = marks.vol.gt(0)
        single_day = valid & valid.shift(1, fill_value=True)
        excluded_risk_dates[key] = tuple(days[~single_day])
        carried = marks.close.ffill()
        real_open = marks.open.where(valid, carried)
        vol = marks.vol.where(valid, 0.)
        quote[key] = pd.DataFrame({"close": carried, "open": real_open, "volume": vol})
        quote[key].index = pd.to_datetime(days, format="%Y%m%d")
        total_returns[key] = ((carried + cash.reindex(days, fill_value=0.)) /
                              carried.shift(1)-1).set_axis(quote[key].index)
        band = get("etf_limit", code, start_date="20240101", end_date=END)
        band.index = pd.to_datetime(band.pop("trade_date"), format="%Y%m%d")
        assert band.index.is_unique
        assert len(band), code
        limits[key] = band[["up_limit", "down_limit"]].astype(float)
        coverage[key] = dict(code=code, listing=listed, first_quote=prices.index[0],
                             cash_events=len(events), suspended_sessions=int((~valid).sum()),
                             omitted_covariance_dates=list(excluded_risk_dates[key]),
                             last=prices.index[-1])

    actual_returns = pd.DataFrame(total_returns).reindex(sessions)
    common = actual_returns.dropna().index
    assert common.equals(sessions[sessions >= common[0]])
    panels = {f: pd.DataFrame({k: q[f] for k, q in quote.items()}).reindex(common)
              for f in ("open", "close", "volume")}
    upper = pd.DataFrame({k: f.up_limit for k, f in limits.items()}).reindex(common)
    lower = pd.DataFrame({k: f.down_limit for k, f in limits.items()}).reindex(common)
    tradeable = panels["volume"].gt(0)
    assert not (upper.isna() & tradeable).any().any()
    assert not (lower.isna() & tradeable).any().any()
    assert ((panels["open"] <= upper+1e-7) | ~tradeable).all().all()
    assert ((panels["open"] >= lower-1e-7) | ~tradeable).all().all()

    bond = pd.Series(load_json("chinabond_30y_wealth.json")["CFZS_00"], dtype=float)
    bond.index = pd.to_datetime(pd.to_numeric(bond.index), unit="ms", utc=True).tz_convert(
        "Asia/Shanghai").tz_localize(None).normalize()
    bond = bond.sort_index()
    bond = bond.reindex(sessions, method="ffill").pct_change(fill_method=None)
    fcf, div = [dated("index_daily", code).close.astype(float) for code in ("980092.SZ", "H20955.CSI")]
    for v in (fcf, div):
        v.index = pd.to_datetime(v.index, format="%Y%m%d")
        assert v.index.equals(sessions)
    commodity = pd.read_csv(PROXY_FILE, index_col=0, parse_dates=True).metals
    proxies = pd.DataFrame({"metals": commodity.reindex(sessions),
                            "fcf": fcf.pct_change(fill_method=None), "bond": bond,
                            "dividend": div.pct_change(fill_method=None)}, index=sessions)
    economic = actual_returns.combine_first(proxies).loc[:, KEYS]
    first = economic.dropna().index[0]
    economic = economic.loc[first:]
    assert np.isfinite(economic.to_numpy()).all()
    assert economic.abs().max().max() <= .23
    for k in KEYS:
        coverage[k]["prelisting_proxy_sessions"] = int(actual_returns.loc[first:, k].isna().sum())
    coverage["earliest_economic_return"] = str(first.date())
    coverage["earliest_complete_etf_return"] = str(common[0].date())
    coverage["earliest_63_day_history"] = str(common[63].date())

    to_iso = lambda df: df.set_axis(df.index.strftime("%Y-%m-%d"))
    def risk_metadata(frame):
        out = to_iso(frame)
        out.attrs["excluded_risk_dates"] = {
            k: pd.to_datetime(v, format="%Y%m%d").strftime("%Y-%m-%d").tolist()
            for k, v in excluded_risk_dates.items()
        }
        return out
    return (
        SimpleNamespace(panels={name: to_iso(frame) for name, frame in panels.items()},
                        returns=risk_metadata(actual_returns.loc[common]), dividends=dividends, splits=[]),
        to_iso((panels["volume"]*100).rolling(20).mean().shift()),
        to_iso(upper), to_iso(lower), risk_metadata(economic), sessions, coverage,
    )
