"""Cross-check the archived ledger, new contribution and timing tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import diagnose as d


def main():
    book=d.saved("monthly_2014-01-02_9bp")
    prior=pd.read_csv(d.SOURCE/"results/contribution_checks.csv").query(
        "case=='monthly_2014-01-02_9bp' and period=='full'").set_index("asset")
    new=pd.read_csv(d.OUT/"contribution.csv")
    base=new.query("run=='base' and period=='full'").set_index("asset").rename(index={"fees":"fee"})
    np.testing.assert_allclose(base.loc[prior.index,"log_contribution"],prior.log_contribution,atol=1e-12)
    detail=pd.read_csv(d.OUT/"asset_month_detail.csv")
    monthly=pd.read_csv(d.OUT/"month_detail.csv")
    for _, frame in monthly.groupby("run"):
        assert frame.date.iloc[1:].tolist()==frame.end.iloc[:-1].tolist()
        assert frame.known.lt(frame.date).all()
    reference=detail.loc[detail.run.eq("base")]
    for asset, frame in reference.groupby("asset"):
        source=book.signals.set_index("date").reindex(frame.date)
        np.testing.assert_array_equal(frame.selected.to_numpy(),source[asset].gt(0).to_numpy())
        assert len(frame)==154 and frame.date.nunique()==154
    switches=pd.read_csv(d.OUT/"basket_transitions.csv").query("run=='base'")
    growth_exits=switches.loc[switches.removed.str.contains("growth",na=False)]
    assert len(growth_exits)==13
    quotes=d.saved("actual_monthly_2025-06-04_9bp_cash0")
    exits=quotes.trades.query("date=='2026-08-03' and asset=='growth' and side=='sell'")
    assert len(exits)==1 and quotes.weights.loc["2026-08-03","growth"]==0
    assert quotes.weights.loc["2026-08-04","bond"]>.7
    nav=book.ledger.nav
    daily=np.log(nav/nav.shift(1,fill_value=book.capital))
    complete=daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).sum().iloc[:-1]
    m=np.expm1(complete)
    tests=dict(targets=len(book.signals),trade_days=book.trades.date.nunique(),
               model_fills=len(book.trades),calendar_months=len(m),positive_months=int(m.gt(0).sum()),
               mean_monthly_net=float(m.mean()),median_monthly_net=float(m.median()),
               monthly_compound_equivalent=float(np.expm1(np.log1p(m).mean())),
               minimum_monthly_net=float(m.min()),maximum_monthly_net=float(m.max()),
               closed_month_blocks=len(monthly.query("run=='base'")),closed_basket_transitions=len(switches),
               growth_exits=len(growth_exits),
               lower_predicted_risk_exits=int(growth_exits.new_weight_vol.lt(growth_exits.old_weight_vol).sum()),
               contribution_and_target_history_match=True,quote_august_sell_followed_by_bond_buy=True)
    (d.OUT/"verified_counts.json").write_text(json.dumps(tests,indent=2)+"\n")
    print(json.dumps(tests,indent=2))


if __name__=="__main__":
    main()
