"""Plot the parameter surface and the recent execution-date sensitivity."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
FONT=Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT.is_file():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.unicode_minus":False,"font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "text.color":"#203842","axes.labelcolor":"#203842",
                     "xtick.color":"#4b606a","ytick.color":"#4b606a"})
TABLES=HERE/"tables"
all_metrics=pd.read_csv(TABLES/"all_metrics.csv")
config=pd.read_csv(TABLES/"cases.csv")
grid=config[config.groups.str.split(",").apply(lambda x:"grid" in x)]
days=[1,2,3,4,5,8,10,15]
risk=[42,56,63,70,84,126]
v12="#aa5633"
gs="#116a78"
fig,axes=plt.subplots(2,2,figsize=(14,9.5))
fig.subplots_adjust(left=.082,right=.942,top=.853,bottom=.111,wspace=.31,hspace=.59)
fig.text(.081,.961,"国盛六周期｜参数小幅改变后，收益还稳吗？",fontsize=22,weight="bold",va="top")
fig.text(.082,.916,"固定六阶段规则  ·  单边滑点 9 bp  ·  截至 2026-11-12",
         fontsize=10.6,va="top",color="#4b606a")

def pick(start, window, clean=False):
    return all_metrics[all_metrics.start.eq(start)&all_metrics.fcf_clean.eq(clean)&
                       all_metrics.window.eq(window)&all_metrics.slip_bp.eq(9)]

def surface(ax,start,clean,title):
    z=pick(start,"full",clean)
    b=z.v12_cagr.iloc[0]*100
    q=z.merge(grid,on="case",how="inner").pivot(index="lookback",columns="schedule",values="cagr")
    q=q.reindex(index=risk,columns=[f"D{d}" for d in days])
    x=(q.to_numpy()-b/100)*100
    ax.imshow(x,aspect="auto",cmap="GnBu",norm=Normalize(vmin=0,vmax=21))
    for i,j in np.ndindex(x.shape):
        if np.isfinite(x[i,j]):
            ax.text(j,i,f"{x[i,j]:+.1f}",ha="center",va="center",
                    color="white" if x[i,j]>13 else "#203842",fontsize=9.5)
    ax.add_patch(plt.Rectangle((-.48,1.52),.96,.96,fill=False,lw=2.2,color="#ad512e"))
    ax.set_xticks(range(len(days)),days)
    ax.set_yticks(range(len(risk)),risk)
    ax.set_xlabel("每月第几个交易日")
    ax.set_ylabel("协方差窗口（交易日）")
    ax.set_title(title,loc="left",fontsize=13,weight="bold",pad=10)
    ax.tick_params(length=0)

surface(axes[0,0],"2014-01-02",True,"2014 年起｜剔除现金流回填")
surface(axes[0,1],"2025-11-06",False,"2025 年 11 月起｜真实 ETF 报价回测")

a,b=axes[1]
z=pick("2025-11-06","full")
sub=z[z.case.str.fullmatch(r"D[0-9]+_r63")].copy()
sub["day"]=sub.case.str.extract(r"D(\d+)").astype(int)
sub=sub.sort_values("day")
a.plot(sub.day,sub.cagr*100,color=gs,marker="o",markersize=4,lw=1.7)
a.axhline(z.v12_cagr.iloc[0]*100,color=v12,lw=1.4,linestyle="--",label="V12")
a.scatter(1,sub.cagr.iloc[0]*100,marker="D",color=v12,s=43,zorder=3)
a.set_title("真实报价全段｜只改每月调仓日",loc="left",fontsize=13,weight="bold",pad=10)
a.set_ylabel("净年化")
a.set_xlabel("每月第几个交易日")
a.set_xticks(range(1,16))
a.set_ylim(0,34)
a.legend(loc="upper right",frameon=False,fontsize=9)

z=pick("2025-11-06","2026-07 to end")
sub=z[z.case.str.fullmatch(r"D[0-9]+_r63")].copy()
sub["day"]=sub.case.str.extract(r"D(\d+)").astype(int)
sub=sub.sort_values("day")
b.bar(sub.day,sub.total*100,width=.61,color=gs)
b.bar(1,sub.total.iloc[0]*100,width=.61,color=v12)
b.axhline(z.v12_total.iloc[0]*100,color=v12,lw=1.4,linestyle="--",label="V12")
b.axhline(0,color="#81959d",lw=.7)
b.set_title("2026 年 7 月以来｜同一组调仓日",loc="left",fontsize=13,weight="bold",pad=10)
b.set_ylabel("区间累计收益")
b.set_xlabel("每月第几个交易日")
b.set_xticks(range(1,16))
b.set_ylim(-10,4)
b.legend(loc="upper right",frameon=False,fontsize=9)
for ax in (a,b):
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#dce5e8",lw=.7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100,decimals=0))
    ax.tick_params(length=0)
fig.text(.081,.039,"上方数值为相对同期 V12 的净年化差（百分点）。红框与橙色标记为原参数。",
         fontsize=10,color="#4b606a")
fig.savefig(HERE/"参数敏感性.png",dpi=149,facecolor="white")
plt.close(fig)
