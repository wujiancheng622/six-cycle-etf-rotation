"""Render the shareable figures from the verified account ledgers."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import dates as mdates, font_manager, pyplot as plt, ticker as mticker
import numpy as np
import pandas as pd

from replay_long import ACT_START, CAPITAL, HERE, MAIN, TABLES, load_account

GS = "#126578"
V12 = "#B85D31"
GRAY = "#62747B"
GRID = "#DCE3E6"
ACTUAL = "#317A77"
PROXY = "#B9884E"
FIG = HERE / "figures"


def configure():
    f = Path(r"C:\Windows\Fonts\msyh.ttc")
    if f.is_file():
        font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":11,"axes.titlesize":13,
                         "figure.dpi":140,"axes.edgecolor":GRID,"axes.labelcolor":"#162D34",
                         "text.color":"#162D34","xtick.color":GRAY,"ytick.color":GRAY,
                         "legend.frameon":False,"savefig.facecolor":"white"})
    FIG.mkdir(exist_ok=True)


def fig_title(fig, a, b, sub_y=.921):
    fig.text(.079,.964,a,fontsize=21,weight="bold",va="top")
    fig.text(.08,sub_y,b,fontsize=10.5,color=GRAY,va="top")


def format_ax(ax, *, percent=False):
    ax.grid(axis="y",color=GRID,linewidth=.7,alpha=.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.,decimals=0))


def panel(ax, title):
    ax.set_title(title,loc="left",pad=9,weight="bold")


def history():
    start=json.loads((HERE/"replay_checks.json").read_text(encoding="utf-8"))["earliest_amount"]
    a,v=(load_account("amount",start,k) for k in (MAIN,"V12"))
    roll=pd.read_csv(TABLES/"rolling_windows.csv",parse_dates=["end"])
    ys=pd.read_csv(TABLES/"years.csv")
    fig, (nav, rolling, annual)=plt.subplots(3,1,figsize=(13.4,9.8),
                                           gridspec_kw=dict(height_ratios=[1.3,1.03,1]))
    fig.subplots_adjust(left=.081,right=.972,top=.857,bottom=.103,hspace=.50)
    fig_title(fig,"国盛六周期 vs V12｜长历史的领先有多稳定？",
              "2013.04.12—2026.11.12  ·  同期扣费  ·  ETF 上市前使用指数或期货代理")
    d=pd.to_datetime([a.base_date,*a.ledger.index])
    for p,c,n in ((a,GS,"国盛六周期"),(v,V12,"V12")):
        value=np.r_[1,p.ledger.nav.to_numpy()/CAPITAL]
        nav.plot(d,value,color=c,lw=2,label=n)
        nav.annotate(f"{value[-1]:.2f}×",(d[-1],value[-1]),xytext=(-7,5),textcoords="offset points",
                     ha="right",color=c,weight="bold")
    nav.set_yscale("log")
    nav.set_yticks([1,2,4,8,12],["1×","2×","4×","8×","12×"])
    nav.set_ylim(.88,15.6)
    panel(nav,"累计净值｜对数刻度")
    nav.legend(loc="upper left",ncol=2)
    format_ax(nav)
    for n,c,w in ((12,"#A2B3B9",1.3),(36,GS,1.8),(60,"#9D583C",1.8)):
        t=roll[roll.months==n]
        rolling.plot(t.end,t.delta_cagr,color=c,lw=w,label=f"{n} 个月")
    rolling.axhline(0,color=GRAY,lw=.8)
    rolling.legend(loc="upper left",ncol=3)
    panel(rolling,"滚动年化差｜国盛减 V12")
    format_ax(rolling,percent=True)
    nav.sharex(rolling)
    nav.xaxis.set_major_locator(mdates.YearLocator(2))
    nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    nav.set_xlim(pd.Timestamp("2013-01-01"),pd.Timestamp("2027-02-01"))
    x=np.arange(len(ys))
    annual.bar(x-.19,ys.gs_total,.36,color=GS,label="国盛")
    annual.bar(x+.19,ys.v12_total,.36,color=V12,label="V12")
    annual.set_xticks(x,[str(k)+("*" if k in (2013,2026) else "") for k in ys.label])
    annual.axhline(0,color=GRAY,lw=.8)
    panel(annual,"逐年区间净收益｜完整年度 11 胜 1 负")
    format_ax(annual,percent=True)
    annual.set_ylim(-.12,.59)
    annual.annotate("2023 年落后",(10,.185),xytext=(9.5,.40),
                    color=GRAY,arrowprops=dict(arrowstyle="-",lw=.8,color=GRAY))
    fig.text(.081,.040,"* 2013 与 2026 为不完整年度。滚动窗口彼此重叠。长历史仅用于检验机制与风险。",
             fontsize=10,color=GRAY)
    fig.savefig(FIG/"long_vs_v12.png",dpi=152)
    plt.close(fig)


def exposures():
    t=pd.read_csv(TABLES/"asset_attribution.csv")
    t=t[t.strategy.eq(MAIN)&t.group.eq("all")]
    names=["创业板","黄金","有色","自由现金流","30年国债","红利低波","费用"]
    keys=["growth","gold","metals","fcf","bond","dividend","fee"]
    fig,(a,b)=plt.subplots(1,2,figsize=(12.9,6),sharey=True)
    fig.subplots_adjust(left=.128,right=.963,top=.792,bottom=.163,wspace=.27)
    fig_title(fig,"收益从哪里来？｜历史代理与真实 ETF 拆开看",
              "横轴为各资产对组合净对数收益的贡献（百分点），可逐项加总。",sub_y=.892)
    y=np.arange(len(keys))
    for ax,mode,title in ((a,"amount","2013.04—2026.11｜经济金额回测"),
                          (b,"actual","2025.11—2026.11｜真实 ETF 报价回测")):
        data=t[t["mode"].eq(mode)].set_index("asset").loc[keys]
        exact=data.actual_log_pnl*100
        est=data.proxy_log_pnl*100
        ax.barh(y,exact,color=ACTUAL,label="真实 ETF / 费用",height=.56)
        # A negative proxy is shown from zero instead of lying on top of a
        # positive actual bar; the signed portions still add to the net row.
        ax.barh(y,est,left=np.where(exact*est>0,exact,0),color=PROXY,label="上市前代理",height=.56)
        ax.axvline(0,color=GRAY,lw=.8)
        ax.set_yticks(y,names)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="x",color=GRID,lw=.65)
        ax.set_axisbelow(True)
        panel(ax,title)
        ax.set_xlim((-13,96) if mode=="amount" else (-2,13.5))
        ax.set_xlabel("对数收益贡献（百分点）",color=GRAY,fontsize=9.8,labelpad=7)
    a.invert_yaxis()
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="lower left",bbox_to_anchor=(.124,.024),fontsize=9.5)
    fig.text(.59,.041,"最近这一年，创业板与有色合计占净对数收益约 94%。",color=GRAY,fontsize=10)
    fig.savefig(FIG/"asset_concentration.png",dpi=152)
    plt.close(fig)


if __name__=="__main__":
    configure()
    history()
    exposures()
    print("Rendered two research figures.")
