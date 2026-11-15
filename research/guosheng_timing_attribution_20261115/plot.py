"""Plot archived calculations for discussion and export."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read(name):
    with (RESULTS / (name + ".csv")).open(encoding="utf-8-sig", newline="") as stream:
        return [r for r in csv.DictReader(stream) if r["run"] == "base" and r["period"] == "full"]


def main():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "savefig.facecolor": "white"})
    items = ["growth","gold","bond","fcf","metals","dividend"]
    names = ["创业板","黄金","30年国债","自由现金流","有色期货","红利低波"]
    contribution = {r["asset"]:float(r["net_log_share"])*100 for r in read("contribution")}
    performance = {r["asset"]:r for r in read("asset_hold_vs_out")}
    green, blue, gray = "#006c62", "#335fa3", "#41566c"
    fig, (left,right) = plt.subplots(1,2,figsize=(12,5.6),layout="constrained")
    for axis in (left,right):
        axis.spines[["top","right","left"]].set_visible(False)
        axis.grid(axis="x",color="#dce5ec",alpha=.8)
        axis.set_axisbelow(True)
        axis.axvline(0,color=gray,linewidth=.8)
    y = list(range(len(items)))
    bars=left.barh(y,[contribution[x] for x in items],color=green,height=.58)
    left.bar_label(bars,fmt="%.1f%%",padding=5)
    left.set(xlim=(-8,47),yticks=y,yticklabels=names,xlabel="占净对数收益的比例",title="收益从哪些资产来")
    left.invert_yaxis()
    left.text(.02,-.19,"交易费用另计 −3.1%，资产贡献与费用合计 100%。",transform=left.transAxes,color=gray,fontsize=9)
    selected=[float(performance[x]["selected_next"])*100 for x in items]
    omitted=[float(performance[x]["unselected_next"])*100 for x in items]
    for values,delta,color,label in [(selected,-.17,green,"在篮子里"),(omitted,.17,blue,"不在篮子里")]:
        bars=right.barh([i+delta for i in y],values,color=color,height=.3,label=label)
        right.bar_label(bars,fmt="%+.2f%%",padding=4,fontsize=8)
    right.set(xlim=(-2.7,6),yticks=y,yticklabels=names,xlabel="下一调仓区间平均资产收益",title="同一资产进篮与不进篮的后续表现")
    right.invert_yaxis()
    right.legend(loc="lower right",frameon=False)
    right.text(.02,-.19,"154 个完整持有区间。按目标分组，早期包含代理。",transform=right.transAxes,color=gray,fontsize=9)
    fig.suptitle("国盛纯月频的收益与择时｜2014-01-02 至 2026-11-12",fontsize=15)
    fig.savefig(ROOT / "收益来源与择时.png",dpi=165)
    plt.close(fig)


if __name__=="__main__":
    main()
