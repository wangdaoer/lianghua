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

