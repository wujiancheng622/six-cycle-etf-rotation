"""Plot performance and trading burden for the same-frequency experiment."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT.is_file():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.unicode_minus":False,"font.size":10.5,"axes.spines.top":False,
                     "axes.spines.right":False,"text.color":"#19333d","axes.labelcolor":"#19333d",
                     "xtick.color":"#566b73","ytick.color":"#566b73","legend.frameon":False})

T = HERE/"tables"
raw = pd.read_csv(T/"frequency_and_cost.csv")
primary = raw.loc[raw.capital.eq(1e6)&raw.slip_bps.eq(9)&~raw.fcf_clean]
names = ["monthly","weekly","daily","stage_change","V12"]
labels = ["月频","周频","日频","月频＋\n换篮子加调","V12"]
colors = ["#285968","#3e8991","#81b7ba","#2b8272","#b36140"]
pos = np.arange(len(names))


def pick(start,window):
    return primary.loc[primary.start.eq(start)&primary.window.eq(window)].set_index("case").loc[names]


def style(ax,title,ylabel):
    ax.set_title(title,loc="left",pad=11,weight="bold",fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_xticks(pos,labels)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100,decimals=0))
    ax.grid(axis="y",color="#dce6e9",linewidth=.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both",length=0)
    ax.spines[["bottom","left"]].set_color("#c2d1d6")


def barlabels(ax,bars,values,fmt):
    ax.bar_label(bars,labels=[fmt.format(v) for v in values],padding=3,fontsize=9.7)


fig,axes = plt.subplots(2,2,figsize=(13,8.4))
fig.subplots_adjust(left=.085,right=.961,top=.80,bottom=.129,hspace=.60,wspace=.31)
fig.text(.084,.957,"国盛六周期｜提频后的收益、回撤与成本",fontsize=22,weight="bold",va="top")
fig.text(.085,.90,"各组采用相同标的池、63 日协方差和费用模型，统一截至 2026-11-12",
         fontsize=11,color="#536e79",va="top")

a,b,c,d = axes.flat
for ax,start,title in [(a,"2020-01-02","2020 年起｜长历史金额回测"),
                       (b,"2025-11-06","近一年｜真实 ETF 报价回测")]:
    x=pick(start,"full")
    y=x.cagr.to_numpy()*100
    h=x.mdd.to_numpy()*100
    bar=ax.bar(pos-.18,y,.33,color=colors,label="净年化")
    dd=ax.bar(pos+.18,h,.33,color="#d2dde1",edgecolor="#abbec5",linewidth=.45,label="最大回撤")
    barlabels(ax,bar,y,"{:.1f}%")
    barlabels(ax,dd,h,"{:.1f}%")
    ax.set_ylim(0,max(y.max(),h.max())+9)
    ax.legend(ncol=2,loc="upper right",fontsize=9.5)
    style(ax,title,"占比")

x = pick("2025-11-06","full")
fee1 = x.annual_commission.to_numpy()*100
fee2 = x.annual_slippage.to_numpy()*100
c.bar(pos,fee1,.56,color="#2a6972",label="佣金")
bars=c.bar(pos,fee2,.56,bottom=fee1,color="#a4cad0",label="滑点")
c.bar_label(bars,labels=[f"{r:.2f}%" for r in fee1+fee2],padding=3,fontsize=9.7)
c.set_ylim(0,1.90)
c.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.5))
c.legend(ncol=2,loc="upper right",fontsize=9.5)
style(c,"近一年｜年化模型交易成本","成本占净值")
c.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100,decimals=1))

x = pick("2025-11-06","2026-07 to end")
y=x.total.to_numpy()*100
bars=d.bar(pos,y,.55,color=colors)
barlabels(d,bars,y,"{:+.2f}%")
d.set_ylim(-8.25,3.9)
d.axhline(0,color="#8198a0",lw=.7)
style(d,"2026 年 7 月以来｜区间累计","累计净收益")

fig.text(.085,.04,"单边滑点 9 bp，佣金万 1.5、最低 5 元。长历史金额回测在部分 ETF 上市前使用代理。",
         fontsize=10,color="#536e79")
fig.savefig(HERE/"调仓频率与成本.png",dpi=153,facecolor="white")
plt.close(fig)
