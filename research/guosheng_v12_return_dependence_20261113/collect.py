"""Collect bounded, dated fund and index snapshots via the established read-only client."""
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
PRIOR = WORK / "guosheng_six_regime_20261112"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR))
import provider_data as p
sys.path.remove(str(PRIOR))
p.DATA = HERE / "inputs" / "tushare"
END = "20261104"


def nav_inputs():
    inception = dict(growth="20111201", gold="20130701", metals="20191201",
                  fcf="20250201", bond="20230601", dividend="20200701")
    for key, code in p.CARRIERS.items():
        start = pd.Timestamp(inception[key])
        end = pd.Timestamp(END)
        starts = [start, *(pd.Timestamp(y,1,1) for y in range(start.year+3,end.year+1,3))]
        for a,b in zip(starts,[*(x-pd.Timedelta(days=1) for x in starts[1:]),end]):
            frame = p.get("fund_nav",code,start_date=a.strftime("%Y%m%d"),end_date=b.strftime("%Y%m%d"))
            assert len(frame)>0
            print("NAV",key,len(frame),frame.nav_date.min(),frame.nav_date.max(),flush=True)


def index_inputs():
    for code in ("399606.SZ","399606.CNI","399006.SZ","H20955.CSI","IMCI.SHFE"):
        frame = p.get("index_daily",code,start_date="20060101",end_date=END)
        if len(frame):
            print("INDEX",code,len(frame),frame.trade_date.min(),frame.trade_date.max(),flush=True)
        else:
            print("INDEX",code,0,flush=True)


if __name__ == "__main__":
    {"nav":nav_inputs,"index":index_inputs}[sys.argv[1]]()
