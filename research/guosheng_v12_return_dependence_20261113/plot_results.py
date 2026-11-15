"""Render the same-period comparison from saved output."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
FONT=Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT.is_file():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.unicode_minus":False,"font.size":11,"axes.spines.top":False,
                     "axes.spines.right":False,"text.color":"#1d3038","axes.labelcolor":"#1d3038",
                     "xtick.color":"#556c76","ytick.color":"#556c76","legend.frameon":False})

COLOR={"GS":"#146b76","V12":"#ad512e"}
LABEL={"GS":"国盛六周期","V12":"V12"}
T=HERE/"tables"
sc=pd.read_csv(T/"scenarios.csv")
test=pd.read_csv(T/"timing_controls.csv")
fig,(a,b)=plt.subplots(2,1,figsize=(11.5,7.7))
fig.subplots_adjust(left=.105,right=.965,top=.81,bottom=.13,hspace=.63)
fig.text(.102,.951,"国盛六周期与 V12 的收益来源",fontsize=22,weight="bold",va="top")
fig.text(.104,.898,"人民币净年化收益，按同一金额模型复算",fontsize=11,color="#556c76",va="top")
for ax in (a,b):
    ax.set_axisbelow(True)
    ax.spines[["left","bottom"]].set_color("#c4d1d6")
    ax.yaxis.grid(color="#dce6ea",linewidth=.7)
    ax.tick_params(length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100,decimals=0))

a.set_title("2020.07.30–2026.11.04  |  同期收益对照",loc="left",weight="bold",pad=11)
x=np.arange(4)
cases=["market","premium_neutral","premium_neutral_whole","premium_neutral_pre2020"]
use=sc[sc.scope.eq("economic")&sc.start.eq("2020-07-30")].set_index(["strategy","scenario"])
for k,offset in (("GS",-.19),("V12",.19)):
    y=[float(use.loc[(k,c),"cagr"]*100) for c in cases]
    bars=a.bar(x+offset,y,.34,color=COLOR[k],label=LABEL[k])
    a.bar_label(bars,labels=[f"{z:.1f}%" for z in y],padding=3,color=COLOR[k],fontsize=10)
a.set_xticks(x,["原场内价","去折溢价变化","再用最长历史收益","再用 2020 年前收益"])
a.set_ylim(0,31)
a.legend(ncol=2,loc="upper right")

b.set_title("2014–2025  |  去溢价后，把同一批日收益错开宏观月份",loc="left",weight="bold",pad=11)
z=test[test.source.eq("premium_neutral")]
for k,center in (("GS",0),("V12",1)):
    column=k.lower()+"_cagr"
    val=z[z.shift_trading_days.ne(0)][column].sort_values().to_numpy()*100
    b.scatter(center+np.linspace(-.19,.19,len(val)),val,color=COLOR[k],alpha=.6,s=35,zorder=3)
    b.hlines(np.median(val),center-.22,center+.22,color=COLOR[k],lw=2,zorder=3)
    original=float(z.loc[z.shift_trading_days.eq(0),column].iloc[0]*100)
    b.scatter(center,original,s=74,marker="D",color=COLOR[k],zorder=4)
    b.annotate(f"原顺序 {original:.1f}%",(center,original),xytext=(15,3),textcoords="offset points",
               va="center",color=COLOR[k],fontsize=10)
    b.text(center-.24,1.8,f"错位中位数 {np.median(val):.1f}%",fontsize=10,color=COLOR[k])
b.set_xticks([0,1],[LABEL["GS"],LABEL["V12"]])
b.set_xlim(-.52,1.52)
b.set_ylim(0,27)
fig.text(.104,.04,"每个圆点对应一种固定错位。日收益和资产间同日涨跌保留，历史相位发生改变。",
         fontsize=10,color="#556c76")
fig.savefig(HERE/"return_dependence.png",dpi=152,facecolor="white")
plt.close(fig)
