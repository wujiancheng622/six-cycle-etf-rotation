# 国盛接入 qmt_gate 共享持仓

接入代码由 `qmt_adapter.py` 和 `qmt_execution.py` 组成。前者按策略读取资金、份额、成本、委托及成交，后者负责委托前落盘、按原身份恢复、续单和撤单确认。读取及执行设计以网关 `5463202` 的《共享 ETF 详细设计》9.2 节为准。

8001 模拟盘按固定周频接入，预算50万元。每个自然周的首个交易日读取上一交易日已知的阶段，重算篮内63日风险平价。`weekly.py` 是对应回放入口，`monthly.py` 用于影子对照。`weekly_execute.py` 接收当日国盛周频计划，恢复时使用原委托身份。

V12 与国盛在 8001 共用的证券是黄金 ETF `518880.SH` 和有色期货 ETF `159980.SZ`。两者使用独立的策略身份、资金预算和委托记录，按各自的策略视图计算差额。同一证券有未结执行时以网关返回的 `symbol_busy` 为准，等原窗口内重查再计算剩余目标。周初生成新计划时，先核对上一期尚未完成的卖出及补买，再按自己的实际持仓和可用资金算差额。

国盛买入以实时卖一、卖出以实时买一作为当笔委托价格。`validated_order_quote` 检查当日新鲜行情、连续竞价、盘口和涨跌停。周频执行覆盖当天连续竞价时段，午休暂停提交，下午继续。已提交的委托保留原身份等待成交；卖出全部核对且卖款与费用已结算后，按实时卖一和当时可用资金核算买入手数。14:57后只复查原单，未完成的腿保留在执行记录中。

## 周频计划准备

每天按北京时间先运行 `weekly_prepare.py check`。`DUE` 表示当天是该自然周的第一个沪深交易日，`known` 为上一交易日。`SKIP` 表示当天无需准备，`BLOCKED` 表示日历需要补齐或修正。日历文件位于 `runtime/weekly/calendar.json`，保留交易所公告的来源及有效期限。

调仓日核对截至 `known` 已公开的政策、信用与增长数据。将核对结果保存为 `runtime/weekly/macro_<known>.json`，顶层含 `known` 与 `days`。`days` 按日期保存，从已对账首仓计划的信息截止日的下一交易日开始连续记录到 `known`。每日记录下列字段。

- `policy_votes` 含 `reverse_repo`、`slf`、`mlf`、`reserve`、`reserve_interest`、`loan`、`deposit`。原口径可核实的方向填 `1`、`0` 或 `-1`，缺证据填 `null`。
- `credit` 和 `growth` 按原公式取方向，`credit_published` 和 `growth_available` 记录最新数据实际可用日，`sources` 写入用于核对的公开链接。DR007 票由脚本读取并按原30日均值和90日比较计算。

随后按仓库既有研究环境运行 `weekly_prepare.py prepare`。脚本核查不同政策未知值下阶段是否一致，验证前一交易日收盘行情和复权信息，沿用原63日风险平价，结合8001国盛自己的持仓与现金输出 `runtime/weekly/plan_<今日>.json`。其中的整手差额按上一交易日收盘价估算，输出仅供审阅。宏观证据不完整、复权变更待核或持仓有未决委托时，输出具体阻断原因。

调仓日可用 `python weekly_execute.py runtime/weekly/plan_<今日>.json` 预览当日份额。模拟执行入口使用同一命令加 `--execute`。重新启动使用同一日计划和本地订单记录，先核对原请求。计划准备的定时任务只生成供审阅的计划。

## 调用边界

- `LocalGateway(port, strategy_id, token)` 默认仅能读国盛自己的策略视图。8001 是模拟交易，8003 是真实交易。行情使用 8002 的独立只读客户端。运行令牌由私有配置或环境变量提供。
- 上游以同一个策略账户视图生成 `Leg`，在其中固定计划、交易腿、起始自身份额、目标数量、时间窗口和尝试上限。价格回调按原规则读取行情并提供限价、有效期、费用上限和该腿可用买入预算。预算已经反映组合现金目标和当日是否允许使用卖款。适配器会再检查自身可用资金，不向其他策略借用余额。
- 一个端口、一个策略使用同一个私有 journal。重启时用 `recover_pending()` 找回最近未确认的请求，原请求查不到时仍保留记录，等网关证据或人工核对。恢复过程不需要重新生成价格或调用新下单。
- `advance(leg, price_order)` 每次只推进当前状态。`WAIT` 可依原窗口和轮询频率重查。`READY` 表示上一请求已终结，下一次只按剩余目标定价。窗口或尝试耗尽保留实际仓位。`UNKNOWN`、`FEE_PENDING` 和 `MANUAL` 分别对应回执未知、数量已核对而费用待结、需要人工处理。`cancel=True` 对原窗口内已知活动委托请求撤单。收到撤单回执仍需刷新同一意图确认，不释放本地目标量。
- 买卖委托要求独立策略、已核对的 shared 模式、已登记的共享 ETF。卖出按 `min(策略自身份额, 策略视图 sellable, 剩余目标)` 限量。`account_quantity` 和 `broker_quantity` 仅供对账，`account_sellable` 是公共额度；`symbol_busy` 为真时等待。网关仍负责最终准入及资金、数量与归属复验。
- 共享忙或可卖不足必须同时有同一个 `request_id`、有效 `intent_id` 和 `NOT_SUBMITTED` 证明，才可在原窗口和尝试限额内更新订单。网关已确认终态并核对数量时，可重新计算未成部分。已成交费用待结时，国盛暂缓该腿续单，费用与预留继续由原策略账本处理。

## 只读核对

开设专用身份并确认目标柜台后，在私有环境中注入 `GUOSHENG_QMT_TOKEN`，再用对应端口和策略 ID 检查。正常输出只有模式和记录数量。

```powershell
python qmt_adapter.py --port 8001 --strategy-id YOUR_GUOSHENG_STRATEGY_ID
```

实盘迁移完成后，用同一入口和实际的实盘身份核对 8003。`/health` 的 `write_ready` 只表明写通道状态，仍需确认柜台身份、结构 1、策略 `position_mode=shared`、目标 ETF 的共享登记与 Policy 的 exclusive 归属。

## 离线复验

在本仓库根目录执行。网关 Python、源码和测试夹具均来自同一个 qmt_gate 工作区。

```powershell
$env:PYTHONPATH = 'D:/src/qmt_gate/src;D:/src/qmt_gate/tests'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:/src/qmt_gate/.venv/Scripts/python.exe' -m pytest tests/test_qmt_shared.py -o addopts= -q
```

测试使用临时 SQLite、真实 FastAPI 策略路由和模拟柜台。覆盖两策略持仓与成本隔离、同证券并发、部分成交后撤单、卖出上限恢复、费用待结、丢回执复原、拒单证明校验、过期和账实差异。真实模拟柜台验证需使用国盛已授权的模拟身份及窗口。
