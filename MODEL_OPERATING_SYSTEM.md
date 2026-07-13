# 模型操作系统

本文件定义 `high_risk_quant_model3` 的日常运行边界、检查点和质量要求。目标不是把模型变成自动交易系统，而是把研究、筛选、复盘、审计和风险提示做成可重复、可追踪的本地流程。

## 当前定位

- 项目根目录：`D:\codex\智能化\high_risk_quant_model3`
- 数据源优先级：先读 `D:\codex\daily-market-data\ths_exports\normalized`，缺少最新数据时再参考 `D:\codex\2026-06-15-exchange-data-ingest`
- 研究方向：A 股主板 + 创业板，高波动/高弹性标的，重点观察涨幅大于 5%、强趋势、趋势二波、强势平台蓄势和模型排序靠前的股票
- 输出用途：研究、观察、复盘和人工决策辅助
- 明确边界：不连接券商、不自动下单、不把模型结果视为买卖建议

## 日更入口

每天数据更新后，从项目根目录运行：

```bat
cd /d D:\codex\智能化\high_risk_quant_model3
python run_daily_model_pipeline.py --asof-date YYYY-MM-DD
```

如果当天训练权重已经存在，可走快速刷新：

```bat
python run_daily_model_pipeline.py --asof-date YYYY-MM-DD --skip-train
```

## 月度自进化研究

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_YYYYMMDD.csv `
  --benchmark D:\codex\daily-market-data\benchmarks\510300.csv `
  --asof-date YYYY-MM-DD `
  --dry-run
```

这是月度复核认可的验证入口。默认参数选择期为 2025-01-01 至 2025-06-30，只允许该区间决定组赢家和最终候选路径。该区间按实际 A 股日历共有 117 个交易日；`min_validation_days: 100` 为少量停牌或数据缺口保留余量，`rolling_window_days: 63` 在完整样本上形成 54 个滚动窗口。此次可行性修正没有降低选择、核心测试或最终保留集的回撤、Sharpe、换手率、负窗口率及 PnL 集中度门槛。每个组候选必须同时通过整段旧版选择门槛，以及至少 `min_folds` 个选择期本地、按时间排序且互不重叠分段上的通用硬门槛，才能成为下一组父参数。路径锁定后，系统再把 2025-07-01 至 2025-12-31 划为独立的非重叠核心测试段；每段都把全部价格矩阵和市场暴露物理截断到段末，并只比较锁定候选与起始 champion。锁定候选通过核心门槛后，才允许打开 2026-01-01 起的最终保留集。时间边界完全由 `periods` 定义；旧 `evolution_core.train_days`、`validation_days`、`test_days`、`step_days` 已从严格 schema 删除，不能再作为兼容输入。

CLI 为遗留或一般纸面研究保留可选 `--benchmark`；省略时市场暴露回退为全程 `1.0`，这种结果不能作为基准化验证证据。任何全局 shadow 状态变更都要求同时使用 `--no-dry-run --promote-shadow`、`--asof-date` 等于面板原始最大日期，并要求清洗后的全部价格矩阵和基准有效序列以有限数值实际到达该日期。可写状态只能位于专用 `evolution_state` 目录；正式 YAML、券商和订单路径始终不在该流程内。

全局状态转换使用同目录的 promotion journal 记录 `pending`、`committed` 或 `rejected`。已有 shadow 在选择期复核、锁定核心测试或最终保留集被判定失败时，回滚会先作为独立持久化事件提交并结束本次运行；dry-run 只记录回滚建议，不改变状态，也不会在同一运行晋级替代者。下一次启动会核对 journal 与实际状态，并幂等完成已 committed 但清单或决定仍为 pending/running 的旧运行；单次运行清单同时记录耗时、峰值内存和覆盖重放/指标直接依赖的策略代码指纹。

运行前必须确认：

- 当天文件存在，例如 `D:\codex\daily-market-data\ths_exports\normalized\ths_hs_a_share_YYYY-MM-DD.xls`
- `D:\codex\2026-06-15-exchange-data-ingest\scripts\market_data_utils.py` 中的 `get_latest_fetch_status()` 或 `ensure_latest_fetch_ok()` 可作为备用状态检查
- 如果状态函数落后，但主目录已有当天标准化文件，以主目录文件时间为更强执行依据

## 固定输出

每日核心产物在 `outputs\high_return_v2` 下：

- `daily_personal_overlay_report_YYYYMMDD.md`：中文日报
- `daily_personal_overlay_selected_YYYYMMDD.csv`：个人行为叠加后的入选表
- `daily_personal_overlay_changes_YYYYMMDD.csv`：新增、移除、降权变化
- `early_pattern_watchlist_YYYYMMDD.csv` / `_cn.csv`：早期形态观察池
- `merged_model_decision_table_YYYYMMDD.csv` / `_cn.csv`：模型决策明细
- `merged_priority_watchlist_YYYYMMDD.csv` / `_cn.csv`：优先观察表
- `core_risk_filter_finalist_stability_YYYYMMDD.md`：核心风控稳定性报告

快速查看优先使用：

```text
outputs\high_return_v2\merged_priority_watchlist_YYYYMMDD_cn.csv
```

## 分层规则

优先观察表按以下层级理解：

- `action_focus`：人工明确决策为买入或强关注时的最高层
- `model_focus`：模型排序 + 个人行为过滤后的主观察层
- `risk_watch`：风险名称或风险状态，仅保留可见，不作为执行优先层
- `pattern_watch`：形态观察层，主要用于发现早期结构
- `review_later`：延后复查层

ST / *ST 名称必须进入 `risk_watch`，不能进入执行优先层。默认可见 `risk_watch` 数量应保持受控，避免挤占正常观察名单。

## 每日质量检查

每次日更完成后至少确认：

- 优先观察表行数正常，通常为 50 行
- `stock_name` 缺失数为 0
- `model_focus`、`pattern_watch`、`risk_watch` 分层合理
- 早期形态数量与类别正常，重点看 `趋势二波启动` 和 `强势平台蓄势`
- ST / *ST 已被降到 `risk_watch`
- 自动测试通过：`python -m pytest -q`
- 将本次证据追加到 `daily_run_state.jsonl`

## 风险边界

允许为了收益率适度提高风险，但以下边界不变：

- 当前回撤容忍上限按约 `-40%` 管理，超过后必须优先做归因和风控审计
- 不因为单日高收益放松 ST / *ST、异常数据、未来函数和路径漂移检查
- 不用未确认来源的数据覆盖主日更数据
- 不把成交额缺失或估算成交额作为硬性失败，除非策略重新依赖成交额过滤
- 不做 ETF 和基金

## 需要人工介入的情况

出现以下情况时，应暂停自动推进并让用户确认：

- 当天主数据目录缺少最新文件
- 数据文件行数明显异常
- 优先表缺失股票名称
- 测试失败
- 输出写入 `_pending` 文件且原文件可能被表格软件占用
- 风控稳定性报告显示最大回撤、收益、胜率同时恶化
- 新策略会改变交易边界、股票池范围或风险容忍上限

## 复盘和改进节奏

- 每日：跑日更、看优先观察表、记录状态账本
- 每周：汇总入选后表现，检查成功/失败原因
- 每月：复查参数稳定性，避免只追逐短期最优
- 策略改动前：先写清楚改变了什么、为什么、如何验证
- 策略改动后：必须和上一版输出做对比，而不是只看单次收益
