"""Plot the asset and allocation attribution tables for the paper window."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FONT = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.titlepad": 17})
BLUE, TEAL, INK, GRID = "#344f87", "#147465", "#1e2b3f", "#dce4ea"


def format_axes(ax, title, xlabel):
    ax.set_title(title, loc="left", color=INK, fontproperties=FONT, fontsize=15)
    ax.set_xlabel(xlabel, loc="left", fontproperties=FONT, color=INK, labelpad=9)
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#a7b2bf", lw=.9)
    ax.tick_params(axis="y", length=0, labelcolor=INK, pad=6)
    ax.tick_params(axis="x", colors="#526078", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()


def bar_pairs(ax, y, left, right, width=.28):
    for values, offset, color in ((left, -width / 2, BLUE), (right, width / 2, TEAL)):
        bars = ax.barh(y + offset, values, height=width, color=color)
        for bar, value in zip(bars, values):
            is_negative = value < 0
            ax.annotate(f"{value:+.2f}", (value, bar.get_y() + bar.get_height() / 2),
                        xytext=(-5 if is_negative else 5, 0), textcoords="offset points", va="center",
                        ha="right" if is_negative else "left", fontsize=10, color=INK)


def main():
    assets = pd.read_csv(HERE / "asset_contributions.csv").query('mode == "amount" and period == "paper"')
    rules = pd.read_csv(HERE / "rule_contributions.csv").query('mode == "amount" and period == "paper"')
    axes_assets = {"growth": "创业板", "metals": "有色金属", "gold": "黄金", "bond": "30年国债",
                   "fcf": "自由现金流", "dividend": "红利低波", "fee": "交易成本"}
    axes_rules = {"base_annual_log": "全池等权底仓", "phase_basket": "经济阶段换篮",
                  "risk_weight": "等风险配权", "in_month_switch": "月内追加换仓"}
    a = assets.pivot(index="asset", columns="case", values="annual_contribution").loc[list(axes_assets)] * 100
    b = rules.set_index("history")[list(axes_rules)].T * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.1, 6.45), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.115, right=.965, top=.76, bottom=.16, wspace=.42)
    fig.text(.055, .945, "国盛六周期高波版收益归因", fontproperties=FONT, fontsize=23, color=INK)
    fig.text(.055, .895, "2014-01-02 至 2025-10-31 · 本地合并版 · 含交易成本", fontproperties=FONT,
             fontsize=11, color="#526078")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, TEAL)],
               labels=["使用原历史代理", "限制三指数发布前数据"], loc="upper right",
               bbox_to_anchor=(.969, .94), frameon=False, prop=FONT, ncols=1)
    y = np.arange(len(a))
    bar_pairs(ax1, y, a["switch"], a["switch_pit"])
    ax1.set_yticks(y, list(axes_assets.values()), fontproperties=FONT)
    ax1.set_xlim(-1.7, 7.8)
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    format_axes(ax1, "哪些资产贡献了收益", "分摊后的净年化贡献（百分点）")
    y2 = np.arange(len(b))
    bar_pairs(ax2, y2, b["as_archived"], b["publication_cutoffs"])
    ax2.set_yticks(y2, list(axes_rules.values()), fontproperties=FONT)
    ax2.set_xlim(-2.8, 13.25)
    ax2.xaxis.set_major_locator(MultipleLocator(3))
    format_axes(ax2, "哪些策略规则改变了收益", "年化对数收益贡献（百分点）")
    fig.text(.055, .056, "左图随净年化加总。右图用年化对数收益加总，阶段与配权的交互平均分摊。",
             fontproperties=FONT, fontsize=10, color="#526078")
    fig.savefig(HERE / "return_attribution.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
