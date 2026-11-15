"""Compare fractional close and actual-share opening execution on identical dates."""

import pandas as pd

import study as s


def main():
    data = s.load()
    rows = []
    for start in ("2025-06-04", "2025-11-06"):
        for bp in (9, 20):
            for cash in (False, True):
                for name, result in (("GS", s.run(data, s.Case("monthly"), start=start, bp=bp, same_cash=cash)),
                                     ("V12", s.benchmark(data, start=start, bp=bp, same_cash=cash))):
                    label = f"close_{name}_{start}_{bp}bp_cash{int(cash)}"
                    s.record(result, label, rows)
    s.write(pd.DataFrame(rows), "close_execution_comparison.csv")
    print("Matched close-account controls complete", flush=True)


if __name__ == "__main__":
    main()
