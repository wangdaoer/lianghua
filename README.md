# 高风险量化交易模型3 · 回测骨架

该目录用于承接 `019ecbc7-7b5b-7c13-a80a-3f328010bab8` 的策略建模工作，先给出“可落地的第一版”：

- 可切换信号类型（可替换）：
  - `trend_momentum_ratio`：短线与中线动量比值分数（默认保底）
  - `trend_breakout_accel`：突破+加速动量（高风险更激进）
  - `mean_reversion_mr`：均值回归反转（防御性）
- 多标的多空头寸配资与仓位归一
- 风控钩子（最大回撤、仓位上限、杠杆/净中性、目标波动率）
- 成本模型（滑点+交易成本）
- 回测结果与指标输出（权益曲线、交易日志、绩效摘要）

> 说明：本框架为研究用途示例，不构成投资建议；请先在纸面环境复核。

## 月度自进化研究

本验证流程是只读研究：必须显式提供沪深 300 基准并使用 `--dry-run`，不接受省略 `--benchmark` 的验证结果。CLI 本身仍为遗留/一般纸面研究保留可选的 `--benchmark`；省略时会回退为全程满仓市场暴露，但不能据此更新 shadow。默认时间契约把参数选择限制在 2025-01-01 至 2025-06-30；候选路径锁定后，才在 2025-07-01 至 2025-12-31 的非重叠核心测试段上直接比较锁定候选与起始 champion；只有核心门槛通过后才打开 2026-01-01 起的最终保留集。每个核心测试段都会把价格和市场暴露物理截断到该段 `test_end` 后独立重放，指标只来自该段日期。运行专属产物写入 `<output-root>/<run_id>`（默认 `outputs/evolution_runs/<run_id>`）；成功完成的运行还会在跨进程锁下原子更新 output root 下的 `latest.json` 和去重后的 `evolution_registry.jsonl`。即使是 `--dry-run`，它也绝不修改全局影子状态、正式配置或券商/订单。

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_20260710.csv `
  --benchmark D:\codex\daily-market-data\benchmarks\510300.csv `
  --asof-date 2026-07-10 `
  --dry-run
```

该命令仅供人工或每月研究使用，不接入每日默认流水线，也不会自动修改任何生产配置。

若人工复核后需要更新影子注册表，必须单独、显式执行以下研究命令：

```powershell
python run_strong_pullback_evolution.py `
  --config configs/evolution_strong_pullback.yaml `
  --data data_panel_history_main_chinext_20220101_20260710.csv `
  --benchmark D:\codex\daily-market-data\benchmarks\510300.csv `
  --asof-date 2026-07-10 `
  --no-dry-run `
  --promote-shadow
```

即使带有 `--no-dry-run --promote-shadow`，它也只会在资格门槛通过、`--asof-date` 等于面板原始最大日期、且清洗后的全部价格矩阵和基准有效序列都实际到达该日期时更新影子注册表 `outputs/evolution_state/strong_pullback.json`；状态保存的是这个有效加载日期。自定义状态路径也必须是专用 `evolution_state` 目录下的 JSON。已有 shadow 会先重新评估；失败时先以独立 CAS/journal 事件持久化回滚并立即结束本次运行，替代候选只能在下一次运行搜索。状态旁的 `strong_pullback.promotion_journal.json` 记录 `pending`、`committed` 或 `rejected`；下次启动会恢复 pending 事实，也会把已 committed 但尚未完成清单/决定发布的运行幂等收尾。状态 schema v2 强制保存语义数据日期；缺少该字段的旧 v1 状态会被明确拒绝，必须先人工复核并重建。该流程不会修改正式 YAML，也不会产生任何券商订单。

## 使用说明

1. 准备日线数据文件 `csv`，至少包含：
   - `date`（交易日）
   - `symbol`（标的）
   - `close`（收盘价）
   - 可选：`open`、`high`、`low`、`volume`

2. 安装依赖：
   - `pandas`
   - `numpy`
   - `pyyaml`

3. 运行：

```bash
python .\run_backtest.py --data "你的数据.csv" --config .\configs\high_risk_strategy.yaml
```

4. 输出文件写入 `outputs` 目录：
   - `equity_curve.csv`
   - `metrics.json`
   - `trade_log.csv`
   - `backtest_report.md`

## 目标与风控默认参数（第一阶段）

- `signal_type`：可在上述三种信号间切换
- `long_exposure`: `1.0`，`short_exposure`: `1.0`（默认净中性）
- `max_position_weight`: 每只标的最大仓位 `0.08`
- `max_drawdown`: 单位时间最大回撤 `18%`，触发后硬平仓 `2` 天
- `target_annualized_vol`: `0.55`，按名义波动自动缩放
- `commission_bps`: `1.2`，`impact_bps`: `0.8`

如果你确认策略方向（比如“情绪反转/突破/事件驱动”），我会在下一步把信号生成替换成对应规则并增加更完整的风险审计项。

### 参数扫描（下一步）

当你准备比较参数时，可直接运行：

```bash
python .\sweep_backtests.py --data "你的数据.csv" --config .\configs\high_risk_strategy.yaml
```

默认会自动扫描一组保守网格；如需自定义网格，请提供 YAML 文件，例如：

```yaml
signal.short_window: [3, 5, 8]
signal.long_window: [15, 20, 25]
risk.max_drawdown: [0.12, 0.16, 0.2]
portfolio.leverage: [1.1, 1.3, 1.5]
```

并执行：

```bash
python .\sweep_backtests.py --data "你的数据.csv" --config .\configs\high_risk_strategy.yaml --sweep .\configs\sweep_grid.yaml
```

结果文件：

- `outputs/sweep_YYYYMMDD_HHMMSS/sweep_results.csv`
- `outputs/sweep_YYYYMMDD_HHMMSS/top_k.md`
- `outputs/sweep_YYYYMMDD_HHMMSS/sweep_summary.md`

## 真实数据接入前的标准化步骤（阶段性）

当前仓库中的回测脚本仍要求输入标准面板：`date,symbol,open,high,low,close,volume`。  
若你先只想基于现有产物（如 `paper_account_*\\stock_targets.csv`）做一次参数探索，可先执行标准化再回测：

1. 先生成映射后的标准面板

```bash
python .\normalize_data.py `
  --input .\paper_account_today\stock_targets.csv `
  --output .\high_risk_quant_model3\data_panel.csv `
  --mapping .\configs\field_map_stock_targets.yaml
```

2. 如果不想单独维护 mapping 文件，可直接把映射放到命令里（示例）

```bash
python .\normalize_data.py `
  --input .\paper_account_today\stock_targets.csv `
  --output .\high_risk_quant_model3\data_panel.csv `
  --mapping "{ \"date\": \"price_date\", \"symbol\": \"code\", \"close\": \"close_price\" }"
```

3. 然后回测/扫参

```bash
python .\run_backtest.py --data .\high_risk_quant_model3\data_panel.csv --config .\configs\high_risk_strategy.yaml
```

```bash
python .\sweep_backtests.py --data .\high_risk_quant_model3\data_panel.csv --config .\configs\high_risk_strategy.yaml
```

说明：
- `--mapping` 支持 YAML 文件或 JSON 字符串，字段建议至少提供：`date`、`symbol`、`close`。
- 标准化脚本会自动用 `close` 填补缺失的 `open/high/low`，`volume` 缺省时填为空值。
- 注意：`stock_targets.csv` 这类文件通常是每日截面快照，不一定包含足够历史日线用于严格回测，结果仅可用于流程连通验证。

可选 mapping 文件示例（保存到 `high_risk_quant_model3/configs/field_map_stock_targets.yaml`）：

```yaml
date: price_date
symbol: code
close: close_price
open: close_price
high: close_price
low: close_price
volume: ""   # 留空则自动置空
```

### 宽表格式的补充用法

如果你的历史文件是“宽表”（例如 `date,000001,000002,...`，或 `close_000001,close_000002,...`），可加 `--wide` 运行：

```bash
python .\normalize_data.py `
  --input .\some_wide_prices.csv `
  --output .\high_risk_quant_model3\data_panel.csv `
  --wide
```

若列名为 `code_open`、`code_close` 这种 `symbol_field` 形式，脚本会自动识别；  
若只有 `000001,000002...` 这类纯 symbol 列，默认会把它们当 `close` 处理（可用 `--wide-value-field` 修改）。
