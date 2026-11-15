"""Join auditable pre-2014 inputs to the frozen local reconstruction."""
from urllib.parse import quote, urljoin
import json
import re
import sys

import numpy as np
import pandas as pd

from early_probe import BASE, DATA, HERE, MFD, OLD, pbc, provider, snapshot, ts

sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE))
from run_study import KEYS, Snapshots, load_v12, v12_amount
from market import market_inputs
from regimes import last_asof, load_json, policy_events, state_path
sys.path.remove(str(BASE))

# Older releases predate the current wording. Read matching phrases in the
# stored originals; May 2011 is a PDF attachment, with figures on page 1.
MANUAL = {
    "2010-01": (3433., 7521., "monthly", "中长期贷款增加3433亿元", "中长期贷款增加7521亿元"),
    "2010-02": (1530., 5088., "monthly", "中长期贷款增加1530亿元", "中长期贷款增加5088亿元"),
    "2010-03": (6461., 17800., "year_to_date", "一季度本外币贷款增加", "中长期贷款增加1.78万亿元"),
    "2010-08": (1114., 2530., "monthly", "中长期贷款增加1114亿元", "中长期贷款增加2530亿元"),
    "2010-11": (1371., 1930., "monthly", "中长期贷款增加1371亿元", "中长期贷款增加1930亿元"),
    "2011-05": (1263., 1184., "monthly", "2011年5月金融统计数据报告(1).pdf", "附件"),
}
CFLP_TABLE = "https://finance.sina.cn/2014-08-07/detail-icesifvw9099684.d.html?from=wap"
# China Economic Net's dated compilation of then-published numbers, with CFLP
# spot checks (2010-05, 2011-01/02/10/11) and NBS overlap 2012-03 to 2013-12.
ORIGINAL_NONMAN = [
    55.1,46.4,58.4,63.0,62.7,57.4,60.1,60.1,61.7,60.5,53.2,56.5,
    56.4,44.1,60.2,62.5,61.9,57.0,59.6,57.6,59.3,57.7,49.7,56.0,
    52.9,48.4,58.0,56.1,55.2,56.7,55.6,56.3,53.7,55.5,55.6,56.1,
]
MFD_SPOT = {"2010-12":54.4, "2011-04":51.8, "2011-05":51.6, "2011-06":50.1,
            "2011-09":49.9, "2011-11":47.7, "2011-12":48.7, "2012-12":51.5}


def loan_data():
    early = pd.read_csv(DATA / "pbc/raw_loans.csv", dtype={"month":str}, encoding="utf-8-sig")
    early = early[early.month.ge("2010-01")].set_index("month")
    for m, (hh, f, basis, *needles) in MANUAL.items():
        file = DATA / "pbc/releases" / f"{m}.html"
        clean = re.sub(r"\s+", "", pbc.clean(file.read_text(encoding="utf-8")))
        assert all(n in clean for n in needles) if m != "2011-05" else needles[1] in clean
        if m == "2011-05":
            raw = file.read_text(encoding="utf-8")
            link = next(x for x in re.findall(r'href="([^"]+)"', raw) if needles[0] in x)
            url = urljoin(pbc.ROOT, quote(link, safe="/()"))
            raw = snapshot(url, DATA / "pbc" / "2011-05_attachment.pdf")
            assert raw.startswith(b"%PDF-")
            early.at[m, "supplementary_url"] = url
            early.at[m, "supplementary_sha256"] = provider.sha(DATA / "pbc" / "2011-05_attachment.pdf")
        early.loc[m, ["household_reported_yi", "firm_reported_yi", "basis", "hint", "extracted"]] = [
            hh, f, basis, "manual verified", " / ".join(needles)]
    assert early.basis.isin(["monthly", "year_to_date"]).all()
    raw = pd.read_csv(BASE / "data/pbc/raw_loans.csv", encoding="utf-8-sig")
    all_rows = pd.concat([early.reset_index(), raw], ignore_index=True)
    credit = pbc.credit_series(all_rows)
    preexisting = pd.read_csv(BASE / "data/pbc/credit_monthly.csv", index_col="month", encoding="utf-8-sig")
    columns = ["household_new_yi", "firm_new_yi", "middle_long_new_yi"]
    pd.testing.assert_frame_equal(credit.loc[preexisting.index, columns], preexisting[columns], check_exact=False)
    ix = preexisting.index[preexisting.credit_change.notna()]
    np.testing.assert_allclose(credit.loc[ix, "credit_change"], preexisting.loc[ix, "credit_change"],
                               atol=1e-13, rtol=0)
    all_rows.to_csv(DATA / "credit_release_sources.csv", index=False, encoding="utf-8-sig")
    credit.to_csv(DATA / "credit_monthly.csv", encoding="utf-8-sig")
    return credit


def pmi_data(original=True):
    official = ts("cn_pmi", start_m="200901", end_m="201212").set_index("month").sort_index()
    official.index = pd.to_datetime(official.index, format="%Y%m").to_period("M")
    ix = pd.period_range("2010-01", "2012-12", freq="M")
    early = official.loc[ix, ["pmi010000", "pmi020100"]].astype(float)
    early.columns = ["manufacturing", "nonmanufacturing"]
    early["nonmanufacturing_ts"] = early.nonmanufacturing
    if original:
        early["nonmanufacturing"] = ORIGINAL_NONMAN
    mfd = pd.DataFrame(json.loads((DATA / "mfd_rows.json").read_text(encoding="utf-8")),
                       columns=["date", "label", "private", "forecast", "previous", "revision"])
    mfd["released"] = pd.to_datetime(mfd.date.str[:10], format="%d.%m.%Y")
    mfd["month"] = mfd.released.dt.to_period("M")-mfd.released.dt.day.le(10).astype(int)
    mfd["private"] = pd.to_numeric(mfd.private, errors="coerce")
    mfd = mfd[mfd.month.isin(ix) & mfd.private.notna()]
    assert len(mfd) == len(ix) and mfd.month.is_unique, mfd
    mfd = mfd.set_index("month").reindex(ix)
    for m, n in MFD_SPOT.items():
        assert mfd.at[pd.Period(m), "private"] == n
    early["private"] = mfd.private
    early["source_release_date"] = mfd.released
    early["available"] = early.source_release_date.clip(lower=(ix+1).to_timestamp()+pd.Timedelta(days=2))
    early["nonmanufacturing_source"] = CFLP_TABLE if original else "Tushare current historical vintage"
    early["private_source"] = MFD
    base = pd.read_csv(BASE / "data/pmi_monthly.csv", index_col="month", encoding="utf-8-sig",
                       parse_dates=["available", "source_release_date"])
    base.index = pd.PeriodIndex(base.index, freq="M")
    columns = ["manufacturing", "nonmanufacturing", "private", "available", "source_release_date"]
    primary = pd.concat([early, base[columns]])
    assert primary.available.is_unique and primary.available.is_monotonic_increasing
    assert primary.index.equals(pd.period_range(ix[0], base.index[-1], freq="M"))
    raw = primary[columns[:3]]
    delta = raw.rolling(12).mean().pct_change(12, fill_method=None).diff(3)
    assert not raw.isna().any().any()
    votes = pd.DataFrame(np.where(delta>0, 1, -1), index=raw.index).sum(axis=1)
    votes = votes.where(delta.notna().all(axis=1))
    primary = primary.join(delta.add_suffix("_change")).assign(votes=votes, direction=np.sign(votes))
    allowed = base.index[base.direction.notna()]
    np.testing.assert_array_equal(primary.loc[allowed, "direction"], base.loc[allowed, "direction"])
    primary.to_csv(DATA / ("pmi_as_reported.csv" if original else "pmi_tushare_vintage.csv"),
                   index_label="month", encoding="utf-8-sig")
    return primary


def macro_inputs(sessions, credit, pmi):
    rates = ts("shibor", start_date="20100101", end_date="20131231").set_index("date")["1w"].astype(float)
    shibor = pd.concat([rates, provider.dated("shibor", None, date_column="date")["1w"].astype(float)])
    repo = provider.dated("repo_daily", "DR007.IB").weight.astype(float)
    assert shibor.index.is_unique and repo.index.is_unique
    rates = repo.combine_first(shibor).dropna().sort_index()
    rates.index = pd.to_datetime(rates.index, format="%Y%m%d")
    smooth = rates.rolling(30).mean()
    market = pd.Series(np.sign(last_asof(smooth, sessions-pd.Timedelta(days=90)).to_numpy()-
                               last_asof(smooth, sessions).to_numpy()), index=sessions, name="market")
    tools = policy_events(sessions)
    money_score = pd.concat([market, tools], axis=1).sum(axis=1, min_count=len(tools.columns)+1)
    c, g = credit.set_index(pd.to_datetime(credit.released)), pmi.set_index(pmi.available)
    macro = tools.assign(market=market, money_score=money_score, money=np.sign(money_score),
                         credit=np.sign(last_asof(c.credit_change, sessions)),
                         growth=last_asof(g.direction, sessions), credit_pulse=last_asof(c.pulse, sessions),
                         credit_published=last_asof(c.released, sessions),
                         growth_available=last_asof(g.available, sessions),
                         growth_votes=last_asof(g.votes, sessions))
    macro["stage"] = state_path(macro.money, macro.credit, macro.growth)
    macro["stage_market"] = state_path(market, macro.credit, macro.growth)
    assert (macro.credit_published.dropna() <= macro.credit_published.dropna().index.strftime("%Y-%m-%d")).all()
    assert (macro.growth_available.dropna() <= macro.growth_available.dropna().index).all()
    return macro


def s(endpoint, code, start, end, col="close"):
    f = ts(endpoint, code, start_date=start, end_date=end)
    assert f.trade_date.is_unique
    return f.set_index(pd.to_datetime(f.trade_date, format="%Y%m%d")).sort_index()[col].astype(float)


def load_inputs():
    q, adv, up, down, frozen, sessions, coverage = market_inputs()
    before = ts("trade_cal", exchange="SSE", start_date="20120101", end_date="20140115")
    before = pd.DatetimeIndex(pd.to_datetime(before.loc[before.is_open.eq(1), "cal_date"], format="%Y%m%d"))
    sessions = before.union(sessions).sort_values()
    old = pd.read_csv(OLD, index_col=0)
    past = old.set_axis(pd.to_datetime(old.index))
    early = sessions[sessions < pd.Timestamp(frozen.index[0])]
    bond = pd.Series(load_json("chinabond_30y_wealth.json")["CFZS_00"], dtype=float)
    bond.index = pd.to_datetime(pd.to_numeric(bond.index), unit="ms", utc=True).tz_convert(
        "Asia/Shanghai").tz_localize(None).normalize()
    bond = bond.sort_index().reindex(sessions, method="ffill").pct_change(fill_method=None)
    g = s("fund_daily", "159915.SZ", "20111201", "20140115").pct_change(fill_method=None)
    gold = s("fund_daily", "518880.SH", "20130101", "20140115").pct_change(fill_method=None)
    auc = s("sge_daily", "Au99.99", "20120101", "20140115").reindex(sessions)
    # Use the prior verified SGE return only if today's direct pair is absent.
    au = auc.pct_change(fill_method=None).combine_first(past.gold)
    no_au_pair = list(early[gold.reindex(early).isna() &
                            auc.reindex(early).pct_change(fill_method=None).isna()].strftime("%Y-%m-%d"))
    fcf = s("index_daily", "980092.SZ", "20120101", "20140116").pct_change(fill_method=None)
    div = s("index_daily", "H20955.CSI", "20120101", "20140116").pct_change(fill_method=None)
    early_ret = pd.DataFrame(dict(growth=g, gold=gold.combine_first(au), metals=past.metals,
                                  fcf=fcf, bond=bond, dividend=div), index=early).loc[:, KEYS]
    first = early_ret.dropna().index[0]
    early_ret = early_ret.loc[first:].set_axis(early_ret.loc[first:].index.strftime("%Y-%m-%d"))
    hist = pd.concat([early_ret, frozen]).set_axis([*early_ret.index, *frozen.index])
    hist.attrs.update(frozen.attrs)
    assert np.isfinite(hist.to_numpy()).all(), hist[hist.isna().any(axis=1)]
    assert hist.index.equals(sessions[sessions >= first].strftime("%Y-%m-%d"))
    pd.testing.assert_frame_equal(hist.loc[frozen.index], frozen, check_index_type=False)
    snapshots = Snapshots()
    bm, ba, bu, bd = load_v12(snapshots)
    tail = v12_amount(bm, snapshots)
    assert old.columns.equals(tail.columns)
    overlap = old.index.intersection(tail.index).intersection(tail.index[tail.index < bm.returns.index[0]])
    np.testing.assert_allclose(tail.loc[overlap], old.loc[overlap], atol=1e-14, rtol=0)
    v12 = pd.concat([old.loc[old.index < tail.index[0]], tail])
    assert v12.index.is_unique and v12.index.is_monotonic_increasing
    assert hist.index.difference(v12.index).empty
    coverage.update(early_first_complete_return=hist.index[0], prior_tail_start=frozen.index[0],
                    V12_input_splice=tail.index[0], V12_splice_overlap_verified=len(overlap),
                    early_gold_no_direct_pair=no_au_pair,
                    V12_earliest_source=v12.index[0],
                    V12_all_actual_returns_from=bm.returns.index[0])
    coverage["actual_return_from"] = {
        k: (str(g.first_valid_index().date()) if k == "growth" else
            str(gold.first_valid_index().date()) if k == "gold" else
            hist.index[hist.index.searchsorted(pd.Timestamp(coverage[k]["first_quote"]).date().isoformat())+1])
        for k in KEYS}
    coverage["extended_proxy_return_sessions"] = {
        k:int((hist.index < d).sum()) for k,d in coverage["actual_return_from"].items()}
    return hist, v12, sessions, coverage, (q, adv, up, down), (bm, ba, bu, bd), snapshots


def prepare():
    hist, v12, sessions, coverage, gs_q, v12_q, s = load_inputs()
    c = loan_data()
    primary = macro_inputs(sessions, c, pmi_data())
    vintage = macro_inputs(sessions, c, pmi_data(original=False))
    for frame, name in ((primary, "macro"), (vintage, "macro_vintage")):
        frame.to_parquet(DATA / f"{name}.parquet")
    old = pd.read_csv(BASE / "daily_phases.csv", index_col="known_date", parse_dates=True)
    ix = old.loc["2016-01-01":].index
    pd.testing.assert_series_equal(primary.loc[ix, "stage"], old.loc[ix, "stage"].astype("Int64"),
                                   check_names=False, check_index_type=False)
    hist.to_parquet(DATA / "returns_gs.parquet")
    v12.to_parquet(DATA / "returns_v12.parquet")
    provider.write(HERE / "coverage.json", coverage)
    return hist, v12, primary, vintage, coverage, gs_q, v12_q, s


if __name__ == "__main__":
    hist, v12, primary, vintage, coverage, *_ = prepare()
    print(hist.index[0], hist.index[63], primary.stage.first_valid_index(),
          int(primary.stage.loc["2013":"2015"].ne(vintage.stage.loc["2013":"2015"]).fillna(False).sum()), flush=True)
    print(json.dumps(coverage, ensure_ascii=True, indent=2))
