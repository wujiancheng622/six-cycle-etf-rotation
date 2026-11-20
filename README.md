# 六周期多资产 ETF 轮动

国盛证券研究所六周期框架的本地实现。8001 模拟盘选用固定周频，每个自然周的首个交易日读取上一交易日已知阶段，在相应篮子内用过去63个有效交易日的协方差做等风险贡献配权。周内维持本周目标，成交模型允许下一交易日补买。宏观三方向未匹配六阶段时沿用上一有效阶段。周频入口为 [`weekly.py`](weekly.py)，冻结月频对照通过 [`monthly.py`](monthly.py) 复算。

| 资产 | ETF |
|---|---|
| 创业板 | 159915.SZ |
| 黄金 | 518880.SH |
| 有色 | 159980.SZ |
| 自由现金流 | 159201.SZ |
| 30年国债 | 511090.SH |
| 红利低波 | 515100.SH |

本地回放从2014-01-02至2026-11-12，初始50万元，单边9bp滑点，佣金1.5bp、最低5元。周频净年化22.75%、年化波动11.98%、最大回撤14.57%；冻结月频对照分别为22.01%、12.22%、18.61%。早期使用指数代理，属于金额回放；真实ETF历史、交易执行、代理偏差及样本外边界见研究报告。国盛研报与本地重建差异见[原始资料索引](docs/sources.md)。

## 复算

Python 3.11。准备输入后运行：

```powershell
python -m pip install -r requirements.txt
python weekly.py --inputs inputs --output runs/weekly
python monthly.py --inputs inputs --output runs/monthly-shadow
```

输入目录接收 `returns.parquet`、`state.parquet` 和 `risk_exclusions.json`。收益表以递增 `YYYY-MM-DD` 字符串为索引，六列名与 `monthly.py` 的 `KEYS` 一致，数值是含资产适用复权的日总收益小数，含暖机期。阶段表以交易日时间戳为索引，至少含 `stage`、`credit_published` 和 `growth_available`。风险剔除表按资产键保存日期列表，无剔除时提供 `{}`。该入口离线消费已重建的宏观状态；状态重建及数据来源见研究归档。具有原研究工作区和授权数据缓存的使用者可以导出同版输入：

```powershell
python -m pip install pyyaml==6.0.3
python tools/export_local_inputs.py --workspace PATH_TO_ORIGINAL_WORKSPACE --output inputs
python monthly.py --inputs inputs --output runs/baseline
python tools/verify_reference.py runs/baseline PATH_TO_ORIGINAL_FROZEN_ACCOUNT
```

参考账户位于原工作区 `guosheng_final_decision_20261114/results/accounts/monthly_2014-01-02_9bp`。逐日账本、权重、损益、交易和目标输出在指定输出目录。授权行情缓存保留在各自研究环境。研究归档中的旧脚本保留历史目录关系，部分需要原工作区和缓存。

策略账户、共享持仓和订单恢复接口见 [qmt_gate 接入说明](docs/qmt-shared.md)。

## 研究记录

| 主题 | 文档 |
|---|---|
| 原文核对与实现 | [原文重建与回测](research/guosheng_original_reconstruction_20261114/国盛六周期原文重建与回测.md) · [研报出处](docs/sources.md) |
| 主结论和回测 | [最终结论](research/guosheng_final_decision_20261114/最终结论.md) · [运行与修正记录](research/guosheng_final_decision_20261114/运行与修正记录.md) · [主对照数据](research/guosheng_final_decision_20261114/results/main_comparison.csv) |
| 过拟合与数据审计 | [过拟合速检](research/guosheng_cross_review_20261114/过拟合速检复核.md) · [70项覆盖](research/guosheng_final_decision_20261114/70项审计覆盖.md) · [可信度交叉复核](research/guosheng_cross_review_20261114/国盛六周期可信度交叉复核.md) |
| 参数、调仓与择时 | [参数稳健性](research/guosheng_parameter_robustness_20261113/参数稳健性复核.md) · [调仓机制](research/guosheng_rebalance_design_20261114/调仓机制分析与结论.md) · [收益与择时](research/guosheng_timing_attribution_20261115/收益来源与择时审查.md) |
| 收益归因与代理压力 | [综合评审](research/guosheng_return_sources_20261114/国盛策略综合评审文档.md) · [其他环节审查](research/guosheng_remaining_review_20261114/其他环节最终审查与调整建议.md) |
| 后续改进实验 | [新增数据冻结方案](research/guosheng_data_expansion_review_20261115/国盛新增分析数据源整改方案_R2交叉评审修订稿.md) · [网页端逐项回应](research/guosheng_data_expansion_review_20261115/网页端第二轮评审逐项回应.md) · [PPI与订单R13验证](research/guosheng_h1_h2_validation_20261115_r13/研究结论.md) |

早期研究版本、对照表和能公开的研究脚本按原目录归档在 [`research/`](research/)。频率备选和新增宏观变量结果用于研究追溯。
