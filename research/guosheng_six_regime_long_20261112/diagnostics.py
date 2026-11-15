"""Matched calendar intervals and exact additive P&L attribution."""
import json

import numpy as np
import pandas as pd

from replay_long import (ACT_START, ASSETS_BY_STAGE, BASE, CAPITAL, DATA, END, HERE,
                         KEYS, MAIN, TABLES, load_account, log_attribution, measures,
                         provider, save_csv, verify_frozen)

LABELS = {"growth":"ChiNext", "gold":"Gold", "metals":"Metals", "fcf":"Free cash flow",
          "bond":"30Y CGB", "dividend":"Low vol dividend", "fee":"Expense"}


def window(a, start, end):
    l = a.ledger.loc[start:end]
    assert len(l), (start, end)
    i = a.ledger.index.get_loc(l.index[0])
    base = a.base_date if i == 0 else a.ledger.index[i-1]
    initial = CAPITAL if i == 0 else a.ledger.nav.iloc[i-1]
    return measures(l, base, initial)


def pair(gs, v12, start, end, label):
    a, b = window(gs, start, end), window(v12, start, end)
    out = dict(label=label, start=a["start"], end=a["end"], base_date=a["base_date"], days=a["sessions"])
    assert all(a[k] == b[k] for k in ("start", "end", "base_date", "sessions"))
    for k in ("total", "cagr", "vol", "mdd", "calmar", "sharpe", "peak", "trough", "recovery",
              "longest_underwater", "day_es_95", "annual_two_way_turnover", "fee_total"):
        out[f"gs_{k}"], out[f"v12_{k}"] = a[k], b[k]
    for k in ("cagr", "vol", "mdd"):
        out["delta_"+k] = a[k]-b[k]
    out["relative_total"] = (1+a["total"])/(1+b["total"])-1
    out["relative_log"] = float(np.log1p(out["relative_total"]))
    return out


def periods(g, v):
    start = g.ledger.index[0]
    blocks = [
        ("2013_partial", start, "2013-12-31"), ("2014-2016", "2014-01-01", "2016-12-31"),
        ("2017-2019", "2017-01-01", "2019-12-31"), ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-2025", "2023-01-01", "2025-12-31"), ("2026_YTD", "2026-01-01", END),
    ]
    stresses = [
        ("2015 equity crash", "2015-06-15", "2015-09-30"),
        ("2018 A-share bear", "2018-01-29", "2018-12-28"),
        ("2020 early Covid", "2020-02-03", "2020-03-31"),
        ("2022 early-year decline", "2022-01-04", "2022-10-31"),
        ("2024 September rebound", "2024-09-24", "2024-10-11"),
    ]
    years = [(str(y),f"{y}-01-01",f"{y}-12-31") for y in range(int(start[:4]),int(END[:4])+1)]
    return tuple(pd.DataFrame(pair(g,v,a,b,l) for l,a,b in group) for group in (blocks, years, stresses))


def rolling(gs, v):
    x = gs.ledger.index
    m = pd.Index(pd.to_datetime(x).to_period("M"))
    ends = pd.Series(x, index=m).groupby(level=0, sort=True).last()
    end_limit = pd.Timestamp(END).to_period("M")
    rows = []
    for n in (12,36,60):
        for mon, end in ends.items():
            if mon >= end_limit or mon-n not in ends:
                continue
            start = x[x.get_loc(ends.loc[mon-n])+1]
            a = pair(gs,v,start,end,str(mon))
            rows.append(dict(months=n, **a))
    return pd.DataFrame(rows)


def head_exclusions(gs, v, mode):
    """Zero out both marks on the same dates; no changed holdings or reoptimization."""
    gr = np.log1p(log_attribution(gs)[0])
    vr = np.log1p(log_attribution(v)[0])
    ex = gr - vr
    y = (pd.Timestamp(gr.index[-1])-pd.Timestamp(gs.base_date)).days/365.25
    selected = [
        ("original", [], np.zeros(len(gr), dtype=bool)),
        ("top_1pct_strategy_days", list(gr.nlargest(max(1,round(len(gr)*.01))).index), None),
        ("top_1pct_active_days", list(ex.nlargest(max(1,round(len(ex)*.01))).index), None),
    ]
    for role, r, gran, sizes in (("strategy",gr,7,(1,3)),("active",ex,7,(1,3,5)),("active",ex,4,(1,3))):
        grouping = gr.index.str[:gran]
        s = r.groupby(grouping).sum()
        if gran == 4:
            s = s.loc[(s.index > gr.index[0][:4]) & (s.index < gr.index[-1][:4])]
        if gran == 7:
            s = s.drop([gr.index[0][:7],gr.index[-1][:7]],errors="ignore")
        for n in sizes:
            if len(s) < n:
                continue
            labels = list(s.nlargest(n).index)
            selected.append((f"top{n}_{role}_{'years' if gran==4 else 'months'}", labels, grouping.isin(labels)))
    rows = []
    for label, removed, mask in selected:
        if mask is None:
            mask = gr.index.isin(removed)
        a, b = gr.iloc[~mask].sum(), vr.iloc[~mask].sum()
        rows.append(dict(mode=mode, test=label, removed="/".join(removed), removed_days=int(mask.sum()),
                         gs_cagr=float(np.expm1(a/y)), v12_cagr=float(np.expm1(b/y)),
                         delta_cagr=float(np.expm1(a/y)-np.expm1(b/y)),
                         relative_log_removed=float(ex.iloc[mask].sum()),
                         relative_log_share_removed=float(ex.iloc[mask].sum()/ex.sum()),
                         strategy_log_share_removed=float(gr.iloc[mask].sum()/gr.sum())))
    return pd.DataFrame(rows)


def mapped_stage(a):
    s = a.signals.set_index("date").phase
    assert a.signals.date.iloc[0] == a.ledger.index[0]
    phase = s.reindex(a.ledger.index).ffill()
    # The prior closing target describes the portfolio inherited into today.
    # The actual-open variant can change within the session; its transition
    # day is kept in the prior-target category, not asserted to be pure.
    phase = phase.shift(1, fill_value=0).astype(int)
    return phase


def asset_rows(a, strategy, mode, returns=None, real_from=None):
    r, raw, log = log_attribution(a)
    held = a.weights.drop(columns="cash").shift(1, fill_value=0.)
    phase = mapped_stage(a) if strategy==MAIN else None
    elapsed = (pd.Timestamp(r.index[-1])-pd.Timestamp(a.base_date)).days/365.25
    daily = log.rename(columns=lambda x:f"log_{x}")
    daily = daily.assign(ret=r, cost=a.ledger.cost, cash_at_close=a.weights.cash)
    daily.index.name = "date"
    if phase is not None:
        daily["phase_prior_target"] = phase
    rows = []
    groups = [("all", "all", np.ones(len(r), dtype=bool))]
    for year in r.index.str[:4].unique():
        groups.append(("year", year, r.index.str[:4] == year))
    for month in r.index.str[:7].unique():
        groups.append(("month", month, r.index.str[:7] == month))
    if phase is not None:
        groups += [("phase", str(p), phase.eq(p).to_numpy()) for p in range(1,7) if phase.eq(p).any()]
    for kind, label, mask in groups:
        total = log.loc[mask].to_numpy().sum()
        for key in log:
            proxy = np.zeros(len(r), dtype=bool)
            if key != "fee" and real_from is not None:
                proxy = r.index < real_from[key]
            row = dict(mode=mode, strategy=strategy, group=kind, label=label, asset=key,
                       days=int(mask.sum()), log_pnl=log.loc[mask,key].sum(),
                       proxy_log_pnl=log.loc[mask&proxy,key].sum(),
                       actual_log_pnl=log.loc[mask&~proxy,key].sum(),
                       share_net_log=log.loc[mask,key].sum()/total if abs(total)>1e-12 else None)
            if key != "fee":
                row |= dict(average_start_weight=held.loc[mask,key].mean(),
                            invested_days=int(held.loc[mask,key].gt(1e-9).sum()),
                            proxy_invested_days=int(held.loc[mask&proxy,key].gt(1e-9).sum()))
                if returns is not None:
                    x = returns.reindex(r.index).loc[mask,key]
                    if kind=="all":
                        row["underlying_cagr"] = np.expm1(np.log1p(x).sum()/elapsed)
                    row["underlying_total"] = np.expm1(np.log1p(x).sum())
                if kind=="all":
                    row["share_daily_variance"] = float(np.cov(raw[key],r,ddof=0)[0,1]/r.var(ddof=0))
            rows.append(row)
    return pd.DataFrame(rows), daily


def timing_rows(a, prices, name, dates):
    r, raw, log = log_attribution(a)
    x = prices.loc[a.ledger.index, a.pnl.columns]
    w = a.weights[x.columns].shift(1, fill_value=0.)
    np.testing.assert_allclose((w*x).to_numpy(), raw[x.columns].to_numpy(), atol=2e-15, rtol=0)
    phase = mapped_stage(a) if name==MAIN else None
    selected = None
    if phase is not None:
        selected = pd.DataFrame({k:phase.map(lambda p:k in ASSETS_BY_STAGE.get(p, ())) for k in KEYS})
    out = []
    for label, start, end in dates:
        frame = x.loc[start:end]
        wi = w.loc[frame.index]
        ri = raw.loc[frame.index]
        baseline = wi.mean()*frame.mean()
        covariance = ((wi-wi.mean())*(frame-frame.mean())).mean()
        np.testing.assert_allclose(baseline+covariance, ri[x.columns].mean(), atol=1e-15, rtol=0)
        subset = pd.Series(np.nan, index=x.columns)
        sizing = subset.copy()
        if selected is not None:
            active = selected.loc[frame.index]
            rates = active.mean()
            iso = active*(wi.mean()/rates.where(rates.gt(0))).fillna(0.)
            subset = (iso*(frame-frame.mean())).mean()
            sizing = ((wi-iso)*(frame-frame.mean())).mean()
            np.testing.assert_allclose(subset+sizing, covariance, atol=1e-15, rtol=0)
        for k in (*x.columns,"fee"):
            out.append(dict(strategy=name, period=label, asset=k, sample_days=len(frame),
                            exposure_annual_arithmetic=float(baseline.get(k,0.)*252),
                            weight_cov_annual_arithmetic=float(covariance.get(k,0.)*252),
                            selection_component=float(subset.get(k,0.)*252),
                            within_allocation_component=float(sizing.get(k,0.)*252),
                            fee_annual_arithmetic=float(ri[k].mean()*252 if k=="fee" else 0.),
                            net_annual_arithmetic=float(ri[k].mean()*252)))
    return pd.DataFrame(out)


def episodes(gs, v, mode):
    p = mapped_stage(gs)
    basket = p.replace({6:5})
    relative = np.log1p(log_attribution(gs)[0])-np.log1p(log_attribution(v)[0])
    g, b = (np.log1p(log_attribution(a)[0]) for a in (gs,v))
    ids = basket.ne(basket.shift()).cumsum()
    rows = []
    for _, periods in basket.groupby(ids):
        ix = periods.index
        stage = int(periods.iloc[0])
        if not stage:
            continue
        rows.append(dict(mode=mode,start=ix[0],end=ix[-1],canonical_basket=stage,
                         phases="/".join(map(str, sorted(p.loc[ix].unique()))),
                         members="/".join(ASSETS_BY_STAGE[stage]), days=len(ix),
                         gs_total=np.expm1(g.loc[ix].sum()),v12_total=np.expm1(b.loc[ix].sum()),
                         relative_log=relative.loc[ix].sum()))
    return pd.DataFrame(rows)


def market_bins(g, v, hist):
    idx = g.ledger.index
    group = idx.str[:7]
    gh, vh = (np.log1p(log_attribution(a)[0]).groupby(group).sum() for a in (g,v))
    x = np.log1p(hist.loc[idx, ["growth","metals"]]).groupby(group).sum()
    x = x.assign(gs=gh,v12=vh).iloc[1:-1]
    x["bucket"] = np.where(x.growth.ge(0),"growth_up","growth_down")+"/"+np.where(
        x.metals.ge(0),"metals_up","metals_down")
    x["active"] = x.gs-x.v12
    grp = x.groupby("bucket",sort=True)
    bins = grp[["gs","v12","active"]].sum().assign(
        months=grp.size(), mean_active=grp.active.mean(),
        win_rate=grp.active.apply(lambda z:z.gt(0).mean()),
    )
    x.index.name = "month"
    bins.index.name = "bucket"
    return x.reset_index(), bins.reset_index()


def main():
    verify_frozen()
    check = json.loads((HERE/"replay_checks.json").read_text(encoding="utf-8"))
    coverage = json.loads((HERE/"coverage.json").read_text(encoding="utf-8"))
    start = check["earliest_amount"]
    g,v = (load_account("amount",start,k) for k in (MAIN,"V12"))
    ag,av = (load_account("actual",ACT_START,k) for k in (MAIN,"V12"))
    hist = pd.read_parquet(DATA/"returns_gs.parquet")
    br = pd.read_parquet(DATA/"returns_v12.parquet")
    blocks, years, stress = periods(g,v)
    for frame, name in ((blocks,"blocks"), (years,"years"), (stress,"stress")):
        save_csv(frame,name+".csv")
    rolls = rolling(g,v)
    save_csv(rolls,"rolling_windows.csv")
    stats_rows = []
    for n, sub in rolls.groupby("months"):
        wi = sub.loc[sub.delta_cagr.idxmin()]
        best = sub.loc[sub.delta_cagr.idxmax()]
        stats_rows.append(dict(months=n, windows=len(sub), wins=int(sub.delta_cagr.gt(0).sum()),
                               gs_loss_windows=int(sub.gs_cagr.lt(0).sum()),
                               worst_delta=float(wi.delta_cagr), worst_start=wi.start, worst_end=wi.end,
                               gs_at_worst_delta=float(wi.gs_cagr),v12_at_worst_delta=float(wi.v12_cagr),
                               median_delta=float(sub.delta_cagr.median()), best_delta=float(best.delta_cagr),
                               worst_gs=float(sub.gs_cagr.min()), worst_v12=float(sub.v12_cagr.min()),
                               max_gs_extra_mdd=float(sub.delta_mdd.max()),
                               win_within_5pp_dd=int((sub.delta_cagr.gt(0)&sub.delta_mdd.le(.05)).sum())))
    save_csv(pd.DataFrame(stats_rows),"rolling_summary.csv")
    sub = [pair(ag,av,"2025-11-06","2025-12-31","2025_post_original"),
           pair(ag,av,"2026-01-01","2026-04-30","2026 Jan-Apr"),
           pair(ag,av,"2026-05-01",END,"2026 May-Nov carry-in")]
    save_csv(pd.DataFrame(sub),"actual_slices.csv")
    save_csv(pd.concat([head_exclusions(g,v,"amount"),head_exclusions(ag,av,"actual")]),"concentration.csv")
    asset_outputs = []
    for model, strategy, a, frame, listed in (
        ("amount",MAIN,g,hist,coverage["actual_return_from"]),
        ("amount","V12",v,br,{k:coverage["V12_all_actual_returns_from"] for k in br.columns}),
        ("actual",MAIN,ag,None,coverage["actual_return_from"]), ("actual","V12",av,None,None)
    ):
        at, daily = asset_rows(a,strategy,model,frame,listed)
        asset_outputs.append(at)
        daily.to_parquet(TABLES/f"daily_{model}_{strategy}.parquet")
    all_at = pd.concat(asset_outputs,ignore_index=True)
    save_csv(all_at,"asset_attribution.csv")
    dates = [("full",start,END),*((t.label,t.start,t.end) for t in blocks.itertuples())]
    t = pd.concat([timing_rows(g,hist,MAIN,dates),timing_rows(v,br,"V12",dates)],ignore_index=True)
    save_csv(t,"timing_attribution.csv")
    ep = pd.concat([episodes(g,v,"amount"),episodes(ag,av,"actual")],ignore_index=True)
    save_csv(ep,"phase_episodes.csv")
    b, bins = market_bins(g,v,hist)
    save_csv(b,"monthly_market_labels.csv")
    save_csv(bins,"market_bins.csv")
    period = years[years.label.between("2014","2025")].copy()
    period["years"] = (pd.to_datetime(period.end)-pd.to_datetime(period.base_date)).dt.days/365.25
    total = period.relative_log.sum()
    period["leave_one_out_annual_relative"] = np.expm1((total-period.relative_log)/(period.years.sum()-period.years))
    save_csv(period[["label","relative_log","years","leave_one_out_annual_relative"]],"year_jackknife.csv")
    print("diagnostics:",len(rolls),"rolling windows,",len(ep),"episodes,",len(all_at),"attribution rows",flush=True)


if __name__=="__main__":
    main()
