"""Recalculate every reported scenario metric directly from its saved daily account."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
TABLES = ("main_comparison.csv", "neighbor_comparison.csv", "policy_uncertainty.csv",
          "symmetric_listing_comparison.csv", "premium_drift_comparison.csv", "updated_joint_stress.csv",
          "updated_proxy_overlap.csv", "input_value_uncertainty.csv", "all_stress_uncertainty.csv",
          "close_execution_comparison.csv", "main_cash_control.csv", "calendar_information_comparison.csv",
          "chart_sensitivity_comparison.csv")


def main():
    saved = {}
    checked, max_error = 0, 0.
    for name in TABLES:
        table = pd.read_csv(OUT / name)
        assert not table.duplicated(["case", "period"]).any(), name
        for row in table.itertuples():
            if row.case not in saved:
                path = OUT / "accounts" / row.case
                saved[row.case] = (pd.read_parquet(path / "ledger.parquet"),
                                   json.loads((path / "meta.json").read_text()))
            ledger, meta = saved[row.case]
            window = ledger.loc[row.start:row.end]
            where = ledger.index.get_loc(window.index[0])
            base = meta["capital"] if where == 0 else ledger.nav.iloc[where - 1]
            anchor = meta["base_date"] if where == 0 else ledger.index[where - 1]
            values = np.r_[base, window.nav.to_numpy()]
            returns = np.diff(values) / values[:-1]
            years = (pd.Timestamp(window.index[-1]) - pd.Timestamp(anchor)).days / 365.25
            metrics = [(values[-1]/base)**(1/years)-1, values[-1]/base-1,
                       np.std(returns, ddof=1)*np.sqrt(252),
                       np.max((np.maximum.accumulate(values)-values)/np.maximum.accumulate(values)),
                       np.mean(window.cash/window.nav), np.sum(window.cost)]
            expected = [row.cagr, row.total, row.vol, row.mdd, row.mean_cash, row.fees]
            np.testing.assert_allclose(metrics, expected, rtol=1e-12, atol=1e-8)
            max_error = max(max_error, float(np.max(np.abs(np.asarray(metrics)-expected))))
            checked += len(metrics)
    sources = [HERE / "protocol.json", HERE / "input_uncertainty.json", HERE / "final_joint_protocol.json",
               HERE / "chart_sensitivity_protocol.json",
               *HERE.glob("*.py"), HERE / "input/shfe_index.csv"]
    receipt = dict(accounts=len(saved), metric_values_checked=checked, max_metric_error=max_error,
                   source_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT / "final_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'}), flush=True)


if __name__ == "__main__":
    main()
