"""Render the net asset value and drawdown comparison from saved ledgers."""
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

HERE = Path(__file__).resolve().parent
CASES = (
    ("amount", "2016-01-04", "2016-01-04 起 · 含上市前代理"),
    ("actual", "2025-11-06", "2025-11-06 起 · 真实 ETF"),
)


def main():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": "Microsoft YaHei" if "Microsoft YaHei" in fonts else "SimHei",
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), sharex="col", layout="constrained",
                             gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle("国盛六周期本地重建与 V12", fontsize=20, color="#182d40")
    for col, (mode, start, title) in enumerate(CASES):
        axes[0, col].set_title(title, loc="left", fontsize=12)
        for name, label, color in (
            ("V12", "V12", "#244c71"), ("GS_monthly", "六周期月频", "#af582e"),
            ("Six_asset_ERC", "同池静态风险平价", "#748985"),
        ):
            table = pd.read_parquet(HERE / "accounts" / mode / start / f"{name}_9bp" / "ledger.parquet")
            nav = table.nav/1e6
            t = pd.to_datetime(nav.index)
            axes[0, col].plot(t, nav, label=label, color=color, lw=1.45)
            dd = nav / nav.cummax().clip(lower=1.) - 1
            axes[1, col].plot(t, dd, color=color, lw=1.0)
        axes[0, col].set_ylabel("扣费净值")
        axes[1, col].set_ylabel("历史回撤")
        axes[0, col].legend(loc="upper left", frameon=False)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[1, col].set_xlim(pd.Timestamp(start), pd.Timestamp("2026-11-12"))
        axes[1, col].xaxis.set_major_locator(mdates.YearLocator(2) if mode == "amount"
                                            else mdates.MonthLocator(interval=3))
        axes[1, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y" if mode == "amount" else "%Y-%m"))
        for ax in axes[:, col]:
            ax.grid(axis="y", linewidth=.7, color="#e0e7ea")
            ax.spines[["right", "top"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#cad4dc")
            ax.tick_params(axis="both", length=0, colors="#455868")
    fig.supxlabel("每期从 100 万元开始 · 单边滑点 9bp，另计佣金 · 截至 2026-11-12", fontsize=10)
    fig.savefig(HERE / "净值回撤对照.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
