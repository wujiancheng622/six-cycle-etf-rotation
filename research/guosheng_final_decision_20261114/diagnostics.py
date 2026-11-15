"""Sensitivity and negative controls without selecting a replacement parameter."""

from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import study as s


def neighbors(data):
    rows = []
    cases = [s.Case("base")]
    cases += [s.Case(f"risk_{n}", risk_days=n) for n in (56, 70)]
    cases += [s.Case(f"read_D{n+1}", read_offset=n) for n in (1, 2, 5)]
    cases += [s.Case(f"execute_delay_{n}", delay=n) for n in (1, 2, 5)]
    for case in cases:
        for mode, start in (("amount", s.START), ("actual", s.CONFIG["dates"]["actual_start"])):
            account = s.run(data, case, start=start, mode=mode)
            s.record(account, f"{mode}_{case.name}", rows)
    for key, values in s.CONFIG["macro_neighbors"].items():
        for value in values:
            state = s.macro(data, key, value)
            for mode, start in (("amount", s.START), ("actual", s.CONFIG["dates"]["actual_start"])):
                name = f"{mode}_{key}_{value}"
                account = s.run(data, s.Case(name), state=state, mode=mode, start=start)
                s.record(account, name, rows)
        print(f"Completed macro neighbors {key}", flush=True)
    s.write(pd.DataFrame(rows), "neighbor_comparison.csv")


def policies(data):
    rows = []
    changes = [("market_only", "market_only", None)]
    changes += [(f"drop_{cat}", "drop", cat) for cat in s.prior.CATS]
    changes += [(f"recent_extra_vote_{v}", "extra_vote", v) for v in (-1, 1)]
    states = []
    for name, key, value in changes:
        state = s.macro(data, key, value)
        for mode, start in (("amount", s.START), ("actual", s.CONFIG["dates"]["actual_start"])):
            account = s.run(data, s.Case(name), state=state, mode=mode, start=start)
            s.record(account, name if mode == "amount" else "actual_" + name, rows)
        states.append(dict(case=name, changed_stages=int(state.loc[s.START:, "stage"].ne(
            data.state.loc[s.START:, "stage"]).sum())))
    s.write(pd.DataFrame(rows), "policy_uncertainty.csv")
    s.write(pd.DataFrame(states), "policy_stage_changes.csv")


def mapping_controls(data):
    rows = []
    baskets = [s.ASSETS[i] for i in range(1, 6)]
    intervals = {"full": s.PERIODS["full"], "paper": s.PERIODS["paper"], "2020-2022": s.PERIODS["2020-2022"],
                 "2023-2025": s.PERIODS["2023-2025"]}
    for number, order in enumerate(permutations(range(5))):
        mapping = {phase: baskets[order[min(phase, 5) - 1]] for phase in range(1, 7)}
        name = "mapping_" + "".join(str(i + 1) for i in order)
        account = s.run(data, s.Case(name), mapping=mapping)
        s.record(account, name, rows, intervals, save=False)
        if (number + 1) % 20 == 0:
            print(f"Mapping negative controls {number + 1}/120", flush=True)
    s.write(pd.DataFrame(rows), "mapping_negative_controls.csv")
    shifts = []
    for periods in range(0, 12):
        state = data.state.copy()
        # This intentionally destroys signal alignment for a descriptive negative control.
        phase = state.loc[s.START:, "stage"].to_numpy()
        state.loc[s.START:, "stage"] = np.roll(phase, periods * 126)
        name = f"phase_halfyear_shift_{periods}"
        account = s.run(data, s.Case(name), state=state)
        s.record(account, name, shifts, intervals, save=False)
    s.write(pd.DataFrame(shifts), "timing_negative_controls.csv")


def main():
    data = s.load()
    neighbors(data)
    policies(data)
    mapping_controls(data)


if __name__ == "__main__":
    main()
