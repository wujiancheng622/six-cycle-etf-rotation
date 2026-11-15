# 国盛原文口径增量复算

截至 2026-11-12。这里的“修正版”先对现有两类政策工具去重，再修正 2025 年 10 月私人 PMI 的可用日，并补入 public_events.csv 中 15 条有来源的旧贷款基准利率、7 天逆回购和 MLF 利率事件。补入的是确认事件子集，未覆盖其他期限、类别或完整事件史，未取得国盛原始阶段表、上市前收益拼接、协方差窗口或成交规则。缺失事件没有作为完整覆盖的证据。

运行 uv run --offline --quiet --no-project --python 3.11 --with numpy==2.4.6 --with pandas==3.0.5 --with pyarrow==25.0.1 --with pyyaml==6.0.3 python replay.py。沿用上轮本地快照、63 日等风险、100 万、万 1.5 最低 5 元及单边 9bp。逐日核对原本地月频账本。PMI 日期修正与仅去重的整个账户账本相等。

- comparison.csv 提供全段、原研报截止日、关键年份以及 2026 年 7 月起的区间指标。
- targets.csv 提供实际使用的信号日期及篮子。信号事件版在原月初更新之外，发现篮子变更时额外更新。其换仓约定属于本地解释。
- signal_comparison.csv 提供旧版、去重、再修 PMI、再补公告后的阶段对照。
- 历史严格现金流情景在风险窗口与收益两处排除 2024-08-15 以前的回填，未满足窗口时留一个现金位。其他早期代理仍保留。

PMI 来源为[标普全球 2025 年 10 月 PMI 发布稿](https://www.pmi.spglobal.com/Public/Home/PressRelease/e56564f4cd484bc2bae7c245658d751d)和[国家统计局公告](https://app.www.gov.cn/govdata/gov/202510/31/539359/article.html)。[2025 国盛原文条目](https://www.fxbaogao.com/detail/5137956)的高波版披露 23.0% 年化、11.3% 波动、12.0% 最大回撤。
