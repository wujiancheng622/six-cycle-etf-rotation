"""Recalculate net NAV metrics from the portable CSV using only Python stdlib."""
import csv
from collections import defaultdict
from datetime import date
from math import sqrt
from pathlib import Path
from statistics import stdev

path = Path(__file__).with_name("daily_nav.csv")
series = defaultdict(list)
with path.open(encoding="utf-8-sig",newline="") as file:
    for row in csv.DictReader(file):
        series[row["case"]].append((date.fromisoformat(row["date"]),float(row["nav"])))

def metrics(rows):
    days,nav=zip(*rows)
    assert all(b>a for a,b in zip(days,days[1:]))
    total=nav[-1]/nav[0]-1
    cagr=(1+total)**(365.25/(days[-1]-days[0]).days)-1
    vol=stdev(b/a-1 for a,b in zip(nav,nav[1:]))*sqrt(252)
    high=nav[0]
    mdd=0
    for value in nav:
        high=max(value,high)
        mdd=max(mdd,1-value/high)
    return cagr,vol,mdd,total

for label,rows in series.items():
    print(label, "CAGR/vol/MDD/total", *(f"{x:.4%}" for x in metrics(rows)))
    if label.startswith("2025_quote_"):
        sub=[row for row in rows if row[0]>=date(2026,6,30)]
        assert sub[0][0]==date(2026,6,30)
        print("  July onward total",f"{metrics(sub)[3]:.4%}")
