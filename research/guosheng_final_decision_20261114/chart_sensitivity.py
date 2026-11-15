"""Quantify readable source-chart disagreements as a hindsight-only diagnostic."""

import hashlib
import json

import pandas as pd

import study as s


def main():
    data = s.load()
    source = s.ROOT / "guosheng_cross_review_20261114/evidence/chart_point_comparison.csv"
    points = pd.read_csv(source, parse_dates=["known_date"])
    protocol = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    rule="For each of the two charts separately, replace only directly readable monthly signal-date phases (10-day boundary guard); ambiguous/absent dates retain the fixed local model.",
                    scope="Hindsight diagnostic only: source charts were not available at historical trade dates. This is not a tradable strategy or author-ledger reproduction.")
    (s.HERE / "chart_sensitivity_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, changes = [], []
    base = s.run(data, s.Case("monthly"))
    benchmark = s.benchmark(data)
    for chart, end in (("2024_fig24", "2024-11-30"), ("2026_fig2", "2025-10-31")):
        sample = points.loc[points.chart.eq(chart) & points["mode"].eq("monthly_known")]
        readable = sample.loc[sample.author_chart_phase.gt(0)].set_index("known_date")
        assert readable.index.is_unique
        state = data.state.copy()
        for day, p in readable.iterrows():
            before, after = int(state.at[day, "stage"]), int(p.author_chart_phase)
            state.at[day, "stage"] = after
            changes.append(dict(chart=chart, known=day.date().isoformat(), before=before, after=after,
                                different_basket=s.ASSETS[before] != s.ASSETS[after]))
        account = s.run(data, s.Case("chart_hindsight"), state=state)
        periods = {"chart_window": (s.START, end)}
        s.record(base, "chart_local_" + chart, rows, periods)
        s.record(account, "chart_override_" + chart, rows, periods)
        s.record(benchmark, "chart_V12_" + chart, rows, periods)
        meta_path = s.OUT / "accounts" / ("chart_override_" + chart) / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update(hindsight_diagnostic_only=True,
                    source_publication="2024-12-28" if chart == "2024_fig24" else "2026-04-30")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    s.write(pd.DataFrame(rows), "chart_sensitivity_comparison.csv")
    s.write(pd.DataFrame(changes), "chart_sensitivity_changes.csv")
    print(pd.DataFrame(rows)[["case", "start", "end", "cagr", "vol", "mdd"]].to_string(index=False))


if __name__ == "__main__":
    main()
