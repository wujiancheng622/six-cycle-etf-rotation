"""Combine the predeclared measurement perturbations with adverse execution assumptions."""

import json
import numpy as np
import pandas as pd

import study as s


def main():
    data = s.load()
    g, v = data.full.copy(), data.views.benchmark.copy()
    errors = pd.read_csv(s.OUT / "updated_proxy_error.csv")
    for row in errors.itertuples():
        if row.strategy == "GS":
            selected = g.index < data.views.first_actual_return[row.asset]
            g.loc[selected, row.asset] = np.expm1(np.log1p(g.loc[selected, row.asset]) - abs(row.annual_log_error) / 252)
        else:
            selected = v.index < "2020-01-20"
            v.loc[selected, row.asset] = np.expm1(np.log1p(v.loc[selected, row.asset]) + abs(row.annual_log_error) / 252)
    conf = json.loads((s.HERE / "input_uncertainty.json").read_text())
    registration = dict(fallback="old_value", capacity=.01, slip_bp=20, extra_macro_days=3,
                        credit_noise=conf["credit_multiplicative_noise_sd"], pmi_noise=conf["each_pmi_point_noise_sd"],
                        scenarios=conf["scenarios"], seed=conf["seed"], purpose="All specified adverse effects together")
    (s.HERE / "final_joint_protocol.json").write_text(json.dumps(registration, indent=2) + "\n")
    rng = np.random.default_rng(conf["seed"])
    rows = []
    c = data.inputs.credit
    p = data.pmi[["manufacturing", "nonmanufacturing", "private"]]
    for trial in range(conf["scenarios"]):
        change = c.middle_long_new_yi * (1 + rng.normal(0, conf["credit_multiplicative_noise_sd"], len(c)))
        credit = np.sign(change.rolling(12).sum().pct_change(12).diff(3))
        source = p + rng.normal(0, conf["each_pmi_point_noise_sd"], p.shape)
        diffs = source.rolling(12).mean().pct_change(12).diff(3)
        votes = pd.DataFrame(np.where(diffs > 0, 1, -1), index=p.index).sum(axis=1).where(diffs.notna().all(axis=1))
        state = data.state.copy()
        state["credit"] = s.last_asof(credit.set_axis(pd.DatetimeIndex(c.released)), state.index)
        state["growth"] = np.sign(s.last_asof(votes.set_axis(pd.DatetimeIndex(data.pmi.available)), state.index))
        shifted = state.copy()
        shifted.index += pd.Timedelta(days=3)
        state = shifted.reindex(state.index, method="ffill")
        state["stage"] = s.state_path(state.money, state.credit, state.growth)
        name = f"all_stress_value_{trial:02}"
        a = s.run(data, s.Case(name, fallback="old_value"), state=state, bp=20, returns=g, limit=True)
        s.record(a, name, rows)
    bench = s.benchmark(data, bp=20, returns=v)
    s.record(bench, "all_stress_V12", rows)
    s.write(pd.DataFrame(rows), "all_stress_uncertainty.csv")
    print(pd.DataFrame(rows).query("period=='full'")[["case", "cagr", "vol", "mdd"]].to_string(index=False), flush=True)
    hurdle = s.metrics(bench)["cagr"]
    state = s.macro(data, "all_macro_extra_calendar_lag", 3)
    cost = []
    def evaluate(bp):
        a = s.run(data, s.Case("cost", fallback="old_value"), state=state, bp=bp, returns=g, limit=True)
        row = s.metrics(a)
        cost.append(dict(bp=bp, cagr=row["cagr"], mdd=row["mdd"], v12_hurdle=hurdle))
        return row["cagr"] - hurdle
    lower, upper = 20, 200
    assert evaluate(lower) > 0 and evaluate(upper) < 0
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if evaluate(mid) > 0:
            lower = mid
        else:
            upper = mid
    s.write(pd.DataFrame(cost).sort_values("bp"), "joint_cost_break_even.csv")
    print(f"Adverse value alternative full-period model break-even {lower} to {upper} bp", flush=True)


if __name__ == "__main__":
    main()
