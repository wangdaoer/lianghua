# A-share ETF Quant Backtest Lab

This project is a research-only backtesting workspace for A-share ETFs.
It does not connect to brokers, place orders, or provide investment advice.

## Quick Start

```powershell
cd /d D:\codex\量化
python -m pip install -r requirements.txt
python -m quant_etf_lab data update --start 20180101
python -m quant_etf_lab backtest --config configs/etf_trend.yaml
python -m quant_etf_lab report --run-id latest
```

The backtest command writes each run under `outputs/backtests/<run-id>/`.

## Commands

```powershell
python -m quant_etf_lab data update --start 20180101
python -m quant_etf_lab backtest --config configs/etf_trend.yaml
python -m quant_etf_lab report --run-id latest
```

## CSI 300 Constituents

Run the current CSI 300 constituent stock test:

```powershell
python -m quant_etf_lab data update --config configs/csi300_trend.yaml --start 20180101
python -m quant_etf_lab backtest --config configs/csi300_trend.yaml
python -m quant_etf_lab report --run-id latest
```

The constituent list is fetched from AKShare's CSIndex interface and cached under `data/universe/`.
This is a current-constituent backtest, so it has survivorship bias and is best used as a strategy smoke test before implementing historical constituent membership.

## A-Share Main Board Universe

Run a small 80-stock main-board smoke test first:

```powershell
python -m quant_etf_lab data update --config configs/ashare_main_board_thsdk_robust_sample80.yaml --start 20220101 --skip-existing --pause-seconds 0.1 --continue-on-error
python -m quant_etf_lab backtest --config configs/ashare_main_board_thsdk_robust_sample80.yaml --skip-missing
python -m quant_etf_lab report --run-id latest
```

Run the full current A-share main-board universe after the smoke test:

```powershell
python -m quant_etf_lab data update --config configs/ashare_main_board_thsdk_robust.yaml --start 20180101 --skip-existing --pause-seconds 0.2 --continue-on-error
python -m quant_etf_lab backtest --config configs/ashare_main_board_thsdk_robust.yaml --skip-missing
python -m quant_etf_lab report --run-id latest
```

The main-board universe is fetched from AKShare Shanghai and Shenzhen stock-name lists, then filtered to current listed A-share main-board codes: `000/001/002/003/600/601/603/605`. This excludes ChiNext, STAR Market, and Beijing Stock Exchange names. It is a current-universe test and has survivorship bias.

Run the expanded current main-board plus ChiNext universe:

```powershell
python -m quant_etf_lab data update --config configs/ashare_main_chinext_thsdk_robust.yaml --start 20180101 --skip-existing --pause-seconds 0.2 --continue-on-error
python -m quant_etf_lab backtest --config configs/ashare_main_chinext_thsdk_robust.yaml --skip-missing
python -m quant_etf_lab report --run-id latest
```

This expanded universe keeps the main-board codes and adds ChiNext `300/301`, while still excluding STAR Market and Beijing Stock Exchange names.

## A-Share Main Board + ChiNext Multi-Factor Stable

Run a 120-stock smoke test before the full expanded universe:

```powershell
python -m quant_etf_lab data update --config configs/ashare_main_chinext_multifactor_stable_sample120.yaml --start 20180101 --skip-existing --pause-seconds 0.1 --continue-on-error
python -m quant_etf_lab walk-forward --config configs/ashare_main_chinext_multifactor_stable_sample120.yaml --preset main-chinext-stable --skip-missing
```

Run the full current main-board plus ChiNext multi-factor rolling training after the smoke test:

```powershell
python -m quant_etf_lab data update --config configs/ashare_main_chinext_multifactor_stable.yaml --start 20180101 --skip-existing --pause-seconds 0.2 --continue-on-error
python -m quant_etf_lab walk-forward --config configs/ashare_main_chinext_multifactor_stable.yaml --preset main-chinext-stable --skip-missing
```

The `main-chinext-stable` preset uses a 4-year training window, 6-month out-of-sample windows, the `stable` candidate grid, the `stable` objective, and a 25% training drawdown gate. It writes `overfit_audit.csv` beside the normal walk-forward summary so train/OOS performance gaps can be reviewed before further strategy changes.

If early OOS windows are still weak, run the stricter v2 preset:

```powershell
python -m quant_etf_lab walk-forward --config configs/ashare_main_chinext_multifactor_stable.yaml --preset main-chinext-stable-v2 --skip-missing
```

The v2 preset only searches stricter portfolio-level risk profiles, lowers the training drawdown gate to 20%, and penalizes high average risk exposure in the stable objective.

## Custom Watchlist

Run the local custom stock pool test:

```powershell
python -m quant_etf_lab data update --config configs/custom_watchlist_trend.yaml --start 20180101 --skip-existing
python -m quant_etf_lab backtest --config configs/custom_watchlist_trend.yaml
python -m quant_etf_lab report --run-id latest
```

Edit `configs/custom_stocks.csv` to replace the stock pool. Required columns are `code` and `name`; `asset_type` defaults to `stock`.

Retest the same pool with the THSDK monitor-style daily conditions:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk.yaml
python -m quant_etf_lab report --run-id latest
```

Recommended lower-turnover variant from the holding-period test:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20.yaml
python -m quant_etf_lab report --run-id latest
```

Risk-controlled variant:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20_risk.yaml
python -m quant_etf_lab report --run-id latest
```

Balanced market-filter variant from the risk-control sweep:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20_market40.yaml
python -m quant_etf_lab report --run-id latest
```

Exit-rule variants:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20_market40_tp30.yaml
python -m quant_etf_lab report --run-id latest

python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20_market40_sl8_tp30.yaml
python -m quant_etf_lab report --run-id latest
```

Score-threshold variants after the exit-rule sweep:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20_market40_tp30_score65.yaml
python -m quant_etf_lab report --run-id latest

python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20_market40_tp30_score70_lowdd.yaml
python -m quant_etf_lab report --run-id latest
```

Robust market-filter variant from split-sample validation:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_thsdk_hold20_market10_tp30_score65_robust.yaml
python -m quant_etf_lab report --run-id latest
python -m quant_etf_lab validate --config configs/custom_watchlist_thsdk_hold20_market10_tp30_score65_robust.yaml
```

Optional loss-cooldown parameters are available as `loss_cooldown_days` and `loss_cooldown_min_losses`.

Run the custom stock pool with multi-factor ranking:

```powershell
python -m quant_etf_lab data update --config configs/custom_watchlist_multifactor.yaml --start 20180101 --skip-existing
python -m quant_etf_lab backtest --config configs/custom_watchlist_multifactor.yaml
python -m quant_etf_lab report --run-id latest
```

The first multi-factor model uses daily OHLCV-derived factors: 60-day momentum, 120-day trend strength, 5-day reversal, 20-day low volatility, and 20-day liquidity. It ranks the stock pool cross-sectionally and rebalances every 5 trading days.

Run the custom stock pool with portfolio-level risk controls:

```powershell
python -m quant_etf_lab data update --config configs/custom_watchlist_multifactor_risk.yaml --start 20180101 --skip-existing
python -m quant_etf_lab backtest --config configs/custom_watchlist_multifactor_risk.yaml
python -m quant_etf_lab report --run-id latest
```

Risk-enabled runs write `risk_curve.csv` and `risk_events.csv` beside the normal backtest outputs.

Config inheritance can reduce YAML sprawl. A variant file may use `extends` to inherit a base YAML and only override changed fields:

```yaml
extends: base/custom_watchlist_multifactor_risk_base.yaml

project:
  name: custom_watchlist_multifactor_risk_extends
strategy:
  max_positions: 3
```

The loader deep-merges nested dictionaries, so overriding one `factor_weights` key preserves the other inherited factor weights. Lists such as `risk.drawdown_levels` are replaced as a whole. Example:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_multifactor_risk_extends.yaml
```

Run the tuned multi-factor risk configuration:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_multifactor_tuned.yaml
python -m quant_etf_lab report --run-id latest
```

Run train/validation/out-of-sample checks for the tuned configuration:

```powershell
python -m quant_etf_lab validate --config configs/custom_watchlist_multifactor_tuned.yaml
```

The validation command writes a summary under `outputs/validation/` and creates one full backtest report per split under `outputs/backtests/`.

Run rolling training with walk-forward out-of-sample tests:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml
```

The default walk-forward run uses a 3-year training window, a 12-month test window, a 12-month step, and the `robust` training objective. It selects the best candidate on each training window, then tests only the selected candidate on the next out-of-sample window. Outputs are written under `outputs/walk_forward/`.

To reduce training-window overfit, hold out the latest part of each training window for internal candidate validation:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --selection-validation-months 6
```

With `--selection-validation-months N`, each rolling window uses the earlier training segment for `train_*` diagnostics, ranks candidates by the held-out `validation_*` segment, then evaluates the selected candidate on the normal next out-of-sample window. Use a separate `--run-id-prefix` for validation experiments so old checkpoints are not mixed with the new selection rule.

Compare the older balanced objective with the robust objective:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --objective balanced
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --objective robust
```

The `robust` objective penalizes losing years, worst annual return, annual drawdown, weak bear-year behavior, and very high turnover.

Run a bear-market risk-control grid:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --preset bear-v2
```

The `bear` grid keeps the factor search small, but lets rolling training choose stricter benchmark-off exposure, crash exposure, drawdown de-risking, and protection/recovery thresholds. The 4-year training window keeps an older weak-market sample in scope for the 2022 out-of-sample decision.

Run the static v2 configuration selected by the latest 4-year bear-market walk-forward window:

```powershell
python -m quant_etf_lab backtest --config configs/custom_watchlist_multifactor_bear_v2.yaml
python -m quant_etf_lab report --run-id latest
```

The v2 static config uses the defensive low-volatility factor mix, 20-day rebalance, and strict bear-market risk controls. It is a research baseline for the current stock pool, not an auto-trading instruction.

Run the moderate-risk opportunity variant:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --preset opportunity-v2
python -m quant_etf_lab backtest --config configs/custom_watchlist_multifactor_opportunity_v2.yaml
python -m quant_etf_lab report --run-id latest
```

The opportunity variant loosens the strict v2 risk controls: benchmark-off exposure rises from 35% to 50%, portfolio drawdown protection waits until 30%, and the drawdown de-risking ladder is less abrupt. Use it only as a higher-risk research comparison against `bear-v2`.
The `opportunity-v2` walk-forward preset still uses a 25% training drawdown gate to prevent the search from falling back to the old high-risk profile.

For a higher-return-oriented sweep of the same candidate set, use the growth preset:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --preset opportunity-v2-growthfull
```

This preset uses the `stable_v2` factor/risk search surface with the new `opportunity_growth` score, which increases the weight on CAGR/absolute return while retaining drawdown and execution penalties.

For the upgraded high-return `opportunity_q_full` research flow, use the dedicated objective:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --preset opportunity-v2-q-full
```

This variant keeps the same 4-year/12-month/6-step structure as the existing full-quality workflow, but uses a stronger return-weighted `opportunity_q_full` objective with tightened tail-risk penalties and 6-month internal validation selection.

If you want a more return-seeking variant first, try:

```powershell
python -m quant_etf_lab walk-forward --config configs/custom_watchlist_multifactor_tuned.yaml --preset opportunity-v2-q-full-aggressive
```

The aggressive variant keeps the same candidate surface and objective, but raises the training drawdown gate to 28% and runs on the full training sample (no internal validation split).

Run the full main-board plus ChiNext THS-style research model:

```powershell
python scripts\cache_market_data.py --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light.yaml --audit-only --output-dir outputs\cache_runs\main_chinext_full
python -m quant_etf_lab backtest --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light.yaml --run-id ashare_main_chinext_hold_prefer_warm120_dd_light
python -m quant_etf_lab report --run-id ashare_main_chinext_hold_prefer_warm120_dd_light
```

This variant keeps existing positions when the signal remains valid, requires 120 local trading bars before a stock can enter, and uses a light portfolio drawdown guard. It is a research baseline only.

Run the concentrated THS-style candidate:

```powershell
python -m quant_etf_lab backtest --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3.yaml --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3
python -m quant_etf_lab report --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3
python -m quant_etf_lab diagnostics --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3
python -m quant_etf_lab attribution --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3 --min-depth 0.05 --top-n 5
python -m quant_etf_lab validate --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3.yaml --run-id-prefix ashare_main_chinext_warm120_dd_light_pos3_split
```

This variant lowers `max_positions` from 5 to 3 while keeping the same 120-bar warmup and light drawdown guard. In the latest local full-sample run it improved return and profit factor, but still needs weak-market treatment for 2022-2023.

Run the targeted rebalance-loss guard experiment:

```powershell
python -m quant_etf_lab backtest --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d20.yaml --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d20 --skip-missing
python -m quant_etf_lab report --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d20
python -m quant_etf_lab diagnostics --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d20
python -m quant_etf_lab attribution --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d20 --min-depth 0.05 --top-n 5
python -m quant_etf_lab validate --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d20.yaml --run-id-prefix ashare_main_chinext_warm120_dd_light_pos3_lossguard5d20_split
```

This variant only delays ordinary rebalance sells when a position has a small unrealized loss within the configured guard window; stop-loss, take-profit, trailing-stop, and explicit risk exits are not blocked. In the latest full-sample run it reduced max drawdown from -37.92% to -36.05% while total return moved from 221.18% to 211.47%. Split validation improved 2018-2021 and 2024-latest, but 2022-2023 remained negative and was slightly worse than the pos3 baseline, so it is a candidate enhancement rather than the final baseline.

Run the selected rebalance-loss guard sweep winner:

```powershell
python scripts\lossguard_sweep.py --config configs\ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3.yaml --output-dir outputs\sensitivity\lossguard_pos3_quick_20260613 --run-prefix lossguard_pos3_quick_20260613 --mode quick
python -m quant_etf_lab backtest --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40.yaml --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40 --skip-missing
python -m quant_etf_lab report --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40
python -m quant_etf_lab diagnostics --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40
python -m quant_etf_lab attribution --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40 --min-depth 0.05 --top-n 5
python -m quant_etf_lab validate --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40.yaml --run-id-prefix ashare_main_chinext_warm120_dd_light_pos3_lossguard5d40_split
```

The quick guard sweep selected `5%/40d` by the current risk-adjusted score. Its full-sample result was 240.11% total return, -36.05% max drawdown, 0.954 Sharpe, and 1.631 profit factor, with rebalance-loss protection active on 19.05% of days. Split validation was 20.76% / -36.05% / 0.376 Sharpe for 2018-2021, -4.43% / -13.50% / -0.120 Sharpe for 2022-2023, and 84.37% / -19.87% / 1.290 Sharpe for 2024-latest. This is the preferred general guard candidate for the next round, while weak-market specialization remains paused.

Run the QuantsPlaybook-inspired moving-average convergence overlay check:

```powershell
python -m quant_etf_lab backtest --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40_convergence.yaml --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_convergence --skip-missing
python -m quant_etf_lab report --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_convergence
python -m quant_etf_lab diagnostics --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_convergence
python -m quant_etf_lab attribution --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_convergence --min-depth 0.05 --top-n 5
```

This experiment borrows QuantsPlaybook's moving-average convergence idea from `B-因子构建类/开源证券-开源量化评论（91）：形态识别，均线的收敛与发散`, adapted to a relative price-normalized factor so cross-stock ranks are not dominated by absolute price level. The direct overlay result was poor: -13.27% total return, -40.07% max drawdown, -0.110 Sharpe, and 0.911 profit factor. Treat `convergence` as an available research factor only; do not promote this overlay into the baseline.

Run the second QuantsPlaybook-inspired quick factor screen:

```powershell
python scripts\external_factor_sweep.py --config configs\ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40.yaml --output-dir outputs\sensitivity\external_factor_quick_20260613 --run-prefix external_factor_quick_20260613 --max-symbols 800
```

This screen tested volume-price correlation, volume-price divergence, lower-shadow support, and a volume-shadow combination on an 800-stock cached A-share sample. The baseline remained best by the current quick research score: 75.29% return, -38.06% max drawdown, 0.499 Sharpe, and 1.270 profit factor. The best external-factor overlay was `volume_shadow_combo`, but it only reached 27.64% return, -40.01% max drawdown, 0.332 Sharpe, and 1.225 profit factor. Keep these factors available for research, but do not promote them into the baseline without a stronger parameter study or a different usage mode.

Run the local factor lab before promoting new signals into strategy configs:

```powershell
python -m quant_etf_lab factor-lab --data-dir data\processed\stocks --output-dir outputs\research\factor_lab_sample_20260614 --max-symbols 300 --start-date 2023-01-01 --horizons 1,5,20 --quantiles 5 --min-obs 100
```

For wider runs, use `--no-save-panel` to avoid writing a very large `factor_panel.csv` while still producing IC, quantile, correlation, snapshot, and Markdown outputs:

```powershell
python -m quant_etf_lab factor-lab --data-dir data\processed\stocks --output-dir outputs\research\factor_lab_full_20260614 --start-date 2023-01-01 --horizons 1,5,20 --quantiles 5 --min-obs 1000 --no-save-panel
```

The factor lab reads cached OHLCV CSV files only and builds same-close factors with forward-return labels. It writes `ic_by_date.csv`, `ic_summary.csv`, `quantile_returns.csv`, `quantile_summary.csv`, `factor_correlation.csv`, `factor_lab_snapshot.json`, and `factor_lab.md`; it also writes `factor_panel.csv` unless `--no-save-panel` is used. The full cached-stock run from 2023-01-03 to 2026-06-12 loaded 4,617 symbols and 3,733,613 factor rows with no file failures. The strongest absolute IC came from `liquidity_log_amount` at the 20-day horizon, with negative mean IC. The best positive long-short spread came from `reversal_5d` at the 20-day horizon, with about 0.72% average top-minus-bottom forward return and a 57.27% win ratio. Treat these as research evidence, not a trading signal, until the factor is wired into walk-forward strategy comparisons.

Run the reversal-factor follow-up experiment:

```powershell
python -m quant_etf_lab backtest --config configs\ashare_main_chinext_multifactor_reversal_sample120.yaml --run-id ashare_main_chinext_reversal_sample120_smoke --skip-missing
python -m quant_etf_lab walk-forward --config configs\ashare_main_chinext_multifactor_reversal_sample120.yaml --preset main-chinext-reversal-v1 --run-id-prefix smoke_main_chinext_reversal_v1 --skip-missing
python -m quant_etf_lab walk-forward --config configs\ashare_main_chinext_multifactor_reversal.yaml --preset main-chinext-reversal-v1 --run-id-prefix main_chinext_reversal_v1_full --skip-missing --resume
```

The `main-chinext-reversal-v1` preset turns the full-universe factor-lab evidence into a controlled walk-forward grid. It compares `reversal_focus`, `reversal_defensive`, and the existing `reversal_lowvol` factor mixes over 20-day and 40-day rebalance intervals while keeping defensive benchmark and drawdown gates. Liquidity remains a small tradability weight rather than a direct low-liquidity alpha target.

The 2026-06-14 sample120 smoke run is mixed. The static reversal-heavy backtest underperformed the stable sample baseline: `ashare_main_chinext_reversal_sample120_smoke` returned -12.05% with -30.09% max drawdown, while `ashare_main_chinext_stable_sample120_compare` returned -2.02% with -26.42% max drawdown on the same sample pool. The reversal walk-forward grid was more useful than the static config: `smoke_main_chinext_reversal_v1` stitched OOS return was 6.83%, max drawdown was -26.77%, and Sharpe was 0.177. Treat reversal as a candidate selected by rolling training, not as a fixed replacement for the defensive core.

The full main-board plus ChiNext run `main_chinext_reversal_v1_full` completed on 2026-06-14 with 72 candidate-window rows. Stitched OOS return was 25.95%, max drawdown was -22.95%, Sharpe was 0.420, positive OOS window ratio was 66.67%, and the worst OOS window return was -11.12%. This is useful evidence that reversal can be a research sleeve, but it is weaker than the current stable-v2 defensive core at 28.64% return, -14.45% max drawdown, and 0.521 Sharpe, and much weaker than the activation allocator at 43.28% return, -11.51% max drawdown, and 1.047 Sharpe. Do not promote reversal-v1 as the core; keep it as a monitored candidate for future allocator or ensemble tests.

The 2026-06-15 allocation and exit smoke tests added `score_weighted_allocation` modes but did not promote them. On `configs/etf_trend.yaml`, equal allocation returned 1.43%, raw score allocation returned -4.11%, and bounded score tilt returned -3.76%. On `ashare_main_chinext_multifactor_stable_sample120`, equal allocation returned -2.02%, while bounded score tilt returned -5.95%. Current `trade_score` is useful for ranking entries but not yet reliable for position sizing. The better sample result came from adding an 8% hard stop to the stable sample config: `smoke_stable_sample120_sl8_v2` returned 14.26%, max drawdown improved slightly to -25.65%, Sharpe rose to 0.197, win rate rose to 66.99%, and average exposure rose to 40.07%. This was captured as `configs/ashare_main_chinext_multifactor_stable_sl8_sample120.yaml` for validation.

The follow-up same-sample walk-forward validation is recorded at `outputs\research\stable_sl8_validation_20260615\stable_sl8_validation.md`. Both variants used the refreshed `sample120` universe, the `main-chinext-stable-v2` preset, and data end date `20260612`. The original stable baseline still won: OOS return 27.69%, max drawdown -13.39%, Sharpe 0.516, positive window ratio 77.78%, and worst window -5.66%. The `stable_sl8` candidate returned 25.07%, max drawdown -14.67%, Sharpe 0.461, positive window ratio 77.78%, and worst window -6.69%. Do not promote the 8% stop-loss variant into the defensive core or allocator input; keep it as research-only evidence that static backtest improvements must pass rolling validation.

The next adaptive stop experiment is recorded at `outputs\research\stable_volstop_validation_20260615\stable_volstop_validation.md`. It adds a 20-day realized-volatility stop to the stable sample config, using multiplier 2.5 and clipping the stop range to 4%-14%. The candidate is much stronger than the fixed 8% stop-loss variant but still does not beat the current stable baseline in rolling validation. Same-sample OOS metrics: stable baseline 27.69% return, -13.39% max drawdown, 0.515 Sharpe, and 77.78% positive windows; volatility-stop candidate 26.46% return, -13.29% max drawdown, 0.509 Sharpe, and 66.67% positive windows; fixed 8% stop reference 14.45% return, -16.07% max drawdown, and 0.314 Sharpe. Keep `configs/ashare_main_chinext_multifactor_stable_volstop_w20_m25_sample120.yaml` as a watched research candidate only. The evidence suggests volatility stops can help drawdown control, but this version should not replace the defensive core.

The 2026-06-15 time-stop follow-up is recorded at `outputs\research\stable_timestop_validation_20260615\stable_timestop_validation.md`. It tested the volatility-stop candidate plus a stagnant-position exit after 40 holding days when open return stayed below 2%. The combination underperformed both the stable baseline and standalone volatility stop: OOS return 15.59%, max drawdown -16.68%, Sharpe 0.333, and 66.67% positive windows. Reject the time-stop combination for the stable core. The result suggests unconditional holding-age exits cut recovery windows more than they improve capital efficiency; future capital-efficiency work should happen at allocator or regime-exposure level rather than with a blanket time stop.

Run the RSRS benchmark risk-filter experiment:

```powershell
python scripts\rsrs_risk_sweep.py --config configs\ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40.yaml --output-dir outputs\sensitivity\rsrs_risk_quick_20260613 --run-prefix rsrs_risk_quick_20260613 --mode quick --max-symbols 800
python -m quant_etf_lab backtest --config configs\ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40_rsrs_z240.yaml --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_rsrs_z240 --skip-missing
python -m quant_etf_lab report --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_rsrs_z240
python -m quant_etf_lab diagnostics --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_rsrs_z240
python -m quant_etf_lab attribution --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40_rsrs_z240 --min-depth 0.05 --top-n 5
python -m quant_etf_lab validate --config configs\ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40_rsrs_z240.yaml --run-id-prefix ashare_main_chinext_warm120_dd_light_pos3_lossguard5d40_rsrs_z240_split
```

The quick RSRS sweep selected `RSRS 18 / z-score 240 / threshold -0.5 / off exposure 35%`: on the 800-stock sample it improved return from 75.29% to 128.25%, max drawdown from -38.06% to -24.67%, and Sharpe from 0.499 to 0.630. Full-universe validation was more mixed. Full-sample return fell from 240.11% to 122.85%, max drawdown improved from -36.05% to -32.87%, and Sharpe fell from 0.954 to 0.654. Split validation improved 2018-2021 and 2022-2023, but hurt the 2024-latest strong market window: 30.06% / -32.87% / 0.472 Sharpe for 2018-2021, -0.50% / -10.45% / 0.038 Sharpe for 2022-2023, and 23.62% / -24.27% / 0.561 Sharpe for 2024-latest. Treat RSRS as a defensive research overlay, not the promoted baseline.

Run the market sentiment state reference report:

```powershell
python -m quant_etf_lab sentiment --config configs\ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_lossguard5d40.yaml --output-dir outputs\research\market_sentiment_state_20260613 --window 120
```

The sentiment command reads the local cached A-share histories only and writes `market_sentiment_timeseries.csv`, `latest_market_sentiment.json`, and `latest_market_sentiment.md`. It adapts the sentiment-timing idea into a market-state reference using same-day breadth, limit-up/limit-down balance, market average return, and the current-day return of stocks that hit limit-up on the previous trading day. It is not connected to broker actions and does not change any backtest strategy. In the 2026-06-13 run, the latest local market date was 2026-06-12, with `warm` sentiment, score `0.440`, 4,586-stock coverage, 72.92% advancing ratio, 2.33% limit-up ratio, 0.61% limit-down ratio, and 1.08% prior limit-up premium.

The `diagnostics` command adds a research layer for linear algebra, statistics, and econometrics. It reads `equity_curve.csv` plus `benchmark.csv`, then writes `diagnostics/return_diagnostics.csv`, covariance and correlation matrices, PCA eigen-components, distribution statistics, and an OLS market model with Newey-West/HAC alpha and beta t-statistics.

The `attribution` command decomposes drawdown episodes using `equity_curve.csv`, `trades.csv`, `benchmark.csv`, and `risk_curve.csv`. It writes `attribution/drawdown_periods.csv`, `drawdown_attribution.csv`, `top_losing_symbols.csv`, `exit_reason_attribution.csv`, and a Markdown report. Use this before changing strategy rules so changes target the actual loss source instead of another broad parameter sweep.

Run batch diagnostics across the current baseline, rejected variants, and selected walk-forward windows:

```powershell
python -m quant_etf_lab diagnostics-batch ashare_main_chinext_hold_prefer_warm120_dd_light_pos3 ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_breadth_soft30 --run-glob "walk_forward_20260613_074336_wf_*_test" --batch-output-dir outputs\diagnostics_batch\main_chinext_review_20260613
```

The batch command writes `diagnostic_compare.csv`, `diagnostic_failures.csv`, and one diagnostics folder per run under the batch output directory.

Market-breadth entry filters are available through `market_breadth_enabled`, `market_breadth_window`, `market_breadth_min_ratio`, and `market_breadth_min_count`. A softer exposure reducer is also available through `market_breadth_exposure_enabled`, `market_breadth_weak_ratio`, and `market_breadth_weak_exposure`. The initial 30% and 40% hard breadth-threshold experiments were too restrictive for this THS-style strategy, and the 30% soft exposure experiment also failed to beat the concentrated pos3 baseline, so these are kept as research comparisons rather than promoted as the current baseline.

Run the cross-sectional score overlay experiment:

```powershell
python -m quant_etf_lab backtest --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light_pos3_xscore.yaml --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_xscore
python -m quant_etf_lab report --run-id ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_xscore
```

This keeps the original THS-style entry conditions but changes the ranking among same-day candidates using configurable cross-sectional signal, momentum, trend, liquidity, and low-volatility ranks. It is an experimental selection-quality layer, not a promoted baseline until full-sample and split-sample validation beat `ashare_main_chinext_hold_prefer_warm120_dd_light_pos3`.

Run split-sample validation for the current THS-style baseline:

```powershell
python -m quant_etf_lab validate --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light.yaml --run-id-prefix ashare_main_chinext_warm120_dd_light_split
python -m quant_etf_lab validate --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_guard.yaml --run-id-prefix ashare_main_chinext_warm120_dd_guard_split
```

Run a small parameter sensitivity sweep around the THS-style baseline:

```powershell
python scripts\sensitivity_sweep.py --config configs/ashare_main_chinext_thsdk_hold_prefer_warm120_dd_light.yaml --output-dir outputs\sensitivity\warm120_dd_light_quick --run-prefix sens_warm120_dd_light --mode quick
```

The sensitivity sweep writes `summary.csv` and `summary.md`; it is intended to catch parameter overfitting before treating any full-sample run as stable.

Run the main-board plus ChiNext stable walk-forward baseline:

```powershell
python -m quant_etf_lab walk-forward --config configs/ashare_main_chinext_multifactor_stable.yaml --preset main-chinext-stable-v2 --run-id-prefix main_chinext_stable_v2_full --skip-missing
```

The `main-chinext-stable-v2` preset uses 4-year training windows, 6-month out-of-sample windows, the `stable_v2` grid, and the `stable` objective. It is the current defensive core baseline for the broader A-share main-board plus ChiNext universe.

Run the true rolling-trained satellite strategy:

```powershell
python -m quant_etf_lab walk-forward --config configs/ashare_main_chinext_multifactor_satellite.yaml --preset main-chinext-satellite-v1 --run-id-prefix main_chinext_satellite_v1_full --skip-missing
```

The satellite preset uses the same 4-year train / 6-month out-of-sample structure as the defensive core, but searches growth-momentum, trend-quality, and pullback-growth factor mixes with looser satellite risk profiles. The `satellite` objective rewards higher CAGR and market participation while still penalizing excessive drawdown, losing years, concentrated return sources, and unstable training-only results.

Run the v2 satellite strategy with individual-stock entry quality and market-breadth controls:

```powershell
python -m quant_etf_lab walk-forward --config configs/ashare_main_chinext_multifactor_satellite_quality_sample120.yaml --preset main-chinext-satellite-v2 --run-id-prefix smoke_main_chinext_satellite_quality_v2 --skip-missing
python -m quant_etf_lab walk-forward --config configs/ashare_main_chinext_multifactor_satellite_quality.yaml --preset main-chinext-satellite-v2 --run-id-prefix main_chinext_satellite_quality_v2_full --skip-missing
```

The v2 satellite keeps the same factor-ranking family as v1, but only allows selected stocks through when their 120-day trend and 20-day momentum are not badly damaged. It also applies market-breadth checks to multi-factor signals, and the `satellite_v2` grid lets rolling training choose between the original satellite guard and a stricter risk-on-only profile that goes flat when the CSI 300 ETF benchmark is risk-off. All of these checks use only information known after the prior close for next-open execution.

Run the first core-satellite allocation layer:

```powershell
python -m quant_etf_lab portfolio combine --config configs/portfolio_core_satellite_v1.yaml
```

This combines the stable-v2 stitched out-of-sample equity curve as the core with the existing `warm120_dd_light` curve as a risk-on satellite proxy. The satellite allocation is only enabled from the next trading day after the CSI 300 ETF is above its 120-day moving average and the 20-day drop is not worse than the configured threshold. Outputs are written under `outputs/portfolios/`.

After the true satellite walk-forward run is complete, use the stricter rolling-trained satellite portfolio configs:

```powershell
python -m quant_etf_lab portfolio combine --config configs/portfolio_core_satellite_rolling_v1.yaml
python -m quant_etf_lab portfolio combine --config configs/portfolio_core_satellite_rolling_sat30.yaml
python -m quant_etf_lab portfolio combine --config configs/portfolio_core_satellite_rolling_guarded.yaml
```

`portfolio_core_satellite_rolling_v1.yaml` uses a 25% risk-on satellite weight. `portfolio_core_satellite_rolling_sat30.yaml` raises the risk-on satellite weight to 30%, which was the highest-Sharpe point in the quick 0%-40% weight sweep on the true rolling satellite curve.
`portfolio_core_satellite_rolling_guarded.yaml` keeps the 30% nominal risk-on satellite weight, but adds a satellite health filter using prior satellite equity momentum and drawdown. In the current run it retained most of the return lift while bringing maximum drawdown back near the defensive core.

Run portfolio-level walk-forward validation for allocation parameters:

```powershell
python -m quant_etf_lab portfolio walk-forward --config configs/portfolio_core_satellite_rolling_guarded.yaml --train-months 24 --test-months 6 --step-months 6 --grid guarded --max-train-drawdown 0.16 --run-id-prefix main_chinext_portfolio_alloc_wf_guarded
```

This step treats the core and true rolling satellite equity curves as inputs, then trains only the allocation layer. It tests whether the satellite weight and health-filter parameters can be selected from prior data without using future returns. If the walk-forward allocator selects `core_only` for many windows, treat that as evidence that the satellite overlay is not yet robust enough to force on every risk-on regime.

Run activation-condition training for the quality-gated v2 satellite:

```powershell
python -m quant_etf_lab portfolio walk-forward --config configs/portfolio_core_satellite_quality_v2_guarded.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --run-id-prefix main_chinext_portfolio_activation_wf_quality_v2
```

The `activation` grid trains the satellite risk-on rule itself: it compares 60/120/200-day benchmark MA gates, stricter or looser 20-day benchmark-drop thresholds, satellite weights from 10% to 25%, and selected satellite-health filters. This is the current acceptance gate before promoting the v2 satellite into the live research portfolio.

In the 2026-06-13 activation run, the stitched out-of-sample allocation result was 43.28% total return, -11.51% max drawdown, and 1.047 Sharpe. The allocator selected `core_only` in two middle windows and selected `sat20_ma60_drop03_unfiltered` for the latest 2026-01-04 to 2026-06-12 window, so the v2 satellite should remain allocator-gated rather than fixed-weight.

Run the reversal-v1 allocator reconnect test:

```powershell
python -m quant_etf_lab portfolio combine --config configs\portfolio_core_reversal_v1_guarded.yaml --run-id main_chinext_core_reversal_v1_guarded
python -m quant_etf_lab portfolio walk-forward --config configs\portfolio_core_reversal_v1_guarded.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --run-id-prefix main_chinext_portfolio_reversal_activation_v1
```

The 2026-06-14 reconnect test uses the stable-v2 defensive core plus the completed reversal-v1 walk-forward curve as a gated satellite. The fixed guarded portfolio returned 31.32%, with -15.10% max drawdown and 0.558 Sharpe. The reversal activation allocator improved to 39.18%, with -11.51% max drawdown and 0.966 Sharpe, but it still trailed the existing quality-v2 activation allocator at 43.28%, -11.51%, and 1.047 Sharpe. Keep reversal-v1 as a monitored research sleeve or future ensemble input; do not replace the current phase-2 allocator.

Run the quality-v2 plus reversal-v1 ensemble allocator comparison:

```powershell
python -m quant_etf_lab portfolio walk-forward --config configs\portfolio_core_ensemble_quality70_reversal30_guarded.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --run-id-prefix main_chinext_portfolio_ensemble_q70_r30_activation_v1
python -m quant_etf_lab portfolio walk-forward --config configs\portfolio_core_ensemble_quality50_reversal50_guarded.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --run-id-prefix main_chinext_portfolio_ensemble_q50_r50_activation_v1
python -m quant_etf_lab portfolio walk-forward --config configs\portfolio_core_ensemble_quality30_reversal70_guarded.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --run-id-prefix main_chinext_portfolio_ensemble_q30_r70_activation_v1
```

The 2026-06-14 ensemble comparison writes composite satellite curves under `outputs\research\ensemble_curves_20260614` and records the conclusion in `outputs\research\ensemble_allocator_review_20260614.md`. The best blend was 70% quality-v2 and 30% reversal-v1 at 42.90% stitched OOS return, -11.82% max drawdown, and 1.043 Sharpe, which is close to but still below the current quality-v2 activation allocator. Do not raise reversal-v1 to a larger fixed weight; the next allocator step should test source selection between quality, reversal, blended satellite, and core-only candidates.

Run true satellite source-selection across quality, reversal, and blended sources:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --run-id-prefix main_chinext_portfolio_source_selection_quality_reversal_v1
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --default-source quality --source-switch-margin 0.12 --run-id-prefix main_chinext_portfolio_source_selection_quality_reversal_guarded_v1
```

The 2026-06-14 source-selection review is recorded at `outputs\research\source_selection_review_20260614\source_selection_review.md`. Free source selection underperformed at 38.57% stitched OOS return, -12.06% max drawdown, and 0.959 Sharpe because it chased reversal-heavy sources in training. The guarded run matched the existing quality-v2 activation allocator at 43.28%, -11.51%, and 1.047 Sharpe by requiring a 0.12 training-score edge before switching away from quality. Keep source selection as a diagnostic tool; do not promote it unless a future source-level validation layer improves out-of-sample results beyond quality-v2.

Run source-selection with an internal source-validation period:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --source-validation-months 3 --run-id-prefix main_chinext_portfolio_source_selection_validation3_v1
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --source-validation-months 6 --run-id-prefix main_chinext_portfolio_source_selection_validation6_v1
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --source-validation-months 9 --run-id-prefix main_chinext_portfolio_source_selection_validation9_v1
```

The source-validation review is recorded at `outputs\research\source_validation_review_20260614\source_validation_review.md`. The 6-month validation run was the strongest result so far at 49.47% stitched OOS return, -9.58% max drawdown, 1.166 Sharpe, 100.00% positive OOS windows, and 0.08% worst OOS window. It first entered the grouped promotion gate as a candidate; the later execution-cost stress review supplied the independent evidence group that allowed the evidence-aware promotion gate below to promote it for research governance.

The follow-up 36-month training-window promotion gate completed as `main_chinext_portfolio_source_selection_validation6_train36_v1`: 38.39% stitched OOS return, -9.58% max drawdown, 1.637 Sharpe, 100.00% positive OOS windows, and 1.13% worst OOS window across 3 OOS windows. Treat this as supportive risk evidence for `source_validation_6m`, not enough to replace the quality-v2 default yet because the longer training window shortens the comparable OOS sample.

The source-stability penalty check completed as `main_chinext_portfolio_source_selection_validation6_stability02_v1`:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --source-validation-months 6 --source-stability-penalty 0.02 --run-id-prefix main_chinext_portfolio_source_selection_validation6_stability02_v1
```

This run reached 48.17% stitched OOS return, -9.91% max drawdown, 1.144 Sharpe, 100.00% positive OOS windows, and 0.08% worst OOS window. It supports the 6-month validation candidate under the full-sample promotion gate, but it is one sensitivity check rather than enough evidence to promote the default allocator by itself.

The lighter quality-default guarded source-switch check completed as `main_chinext_portfolio_source_selection_validation6_margin03_v1`:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --default-source quality --source-switch-margin 0.03 --source-validation-months 6 --run-id-prefix main_chinext_portfolio_source_selection_validation6_margin03_v1
```

This run reached 48.34% stitched OOS return, -9.91% max drawdown, 1.147 Sharpe, 100.00% positive OOS windows, and 0.08% worst OOS window. It supports the candidate at the run level, but it is correlated with `stability02` and `margin06`, so these conservative guard variants are counted as one evidence group in the promotion gate.

The quality-default guarded source-switch check completed as `main_chinext_portfolio_source_selection_validation6_margin06_v1`:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --default-source quality --source-switch-margin 0.06 --source-validation-months 6 --run-id-prefix main_chinext_portfolio_source_selection_validation6_margin06_v1
```

This run reached 43.61% stitched OOS return, -12.49% max drawdown, 1.051 Sharpe, 80.00% positive OOS windows, and -3.11% worst OOS window. It is a useful guardrail diagnostic, but it does not support promotion because the return edge over quality-v2 is only about 0.33 percentage points and max drawdown worsens versus the baseline.

The source-switch margin diagnostic is recorded at `outputs\research\source_switch_margin_diagnostic_20260614\report.html`. It showed that `margin06` failed because PF02's raw-best reversal source beat the default quality/core source by 0.0339 validation-score points, but the 0.06 guard blocked that switch and selected a weaker quality candidate. A narrower 0.03 guard admits PF02's stronger reversal source while still blocking PF04's fragile 0.0083 source edge:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --default-source quality --source-switch-margin 0.03 --source-validation-months 6 --run-id-prefix main_chinext_portfolio_source_selection_validation6_margin03_v1
```

The `margin03` run reached 48.34% stitched OOS return, -9.91% max drawdown, 1.147 Sharpe, 100.00% positive OOS windows, and 0.08% worst OOS window. Treat `margin03` as the current source-switch guardrail evidence; do not adopt `margin06`.

Run the sensitivity-only allocator promotion gate before changing the phase-2 default:

```powershell
python -m quant_etf_lab allocator-promotion-review --baseline-dir outputs\portfolio_walk_forward\main_chinext_portfolio_activation_wf_quality_v2 --candidate-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation3_v1 --sensitivity-group validation_3m --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation9_v1 --sensitivity-group validation_9m --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_train36_v1 --sensitivity-group train36 --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_stability02_v1 --sensitivity-group conservative_guard --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_margin03_v1 --sensitivity-group conservative_guard --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_margin06_v1 --sensitivity-group conservative_guard --output-dir outputs\research\allocator_promotion_grouped_20260614 --min-return-edge 0.03 --min-sharpe-edge 0.05 --max-drawdown-worsening 0 --min-positive-window-ratio 0.80 --min-sensitivity-support 2
```

The grouped 2026-06-14 promotion review is recorded at `outputs\research\allocator_promotion_grouped_20260614\allocator_promotion_review.md`. Decision: `watch_candidate`, not `promote_candidate`. The 6-month source-validation candidate passes the headline gate versus quality-v2 with +6.20 percentage points total return, +0.119 Sharpe, lower drawdown, 100.00% positive OOS windows, and 0.08% worst OOS window. Run-level sensitivity support is 2, but independent group support is only 1 / 2 because `stability02` and `margin03` both belong to the correlated `conservative_guard` family; the 3-month, 9-month, 36-month training, and `margin06` diagnostics do not beat the baseline under the same full-sample thresholds. Keep it as the leading watch candidate instead of switching the operational allocator.

The staged source-selection candidate probe is recorded at `outputs\research\daily_pipeline_promoted_candidate_probe_fixed_20260614\daily_pipeline.md`. It runs the candidate allocator through the same daily-pipeline, paper-account, dashboard, market-cap, model-audit, promotion-review, and after-close gates. `phase2_review` now reads both legacy top-level selected-params files and source-selection nested `allocation` selected-params files. `paper-account` also applies source-selection nested allocation parameters and `source_path` satellite overrides, so the probe now matches the source-selection stitched OOS result: dashboard posture `core_base_watch_allocator_gate`, paper latest regime `risk_off`, paper-account total return 49.47%, max drawdown -9.58%, and Sharpe 1.165. This was the compatibility probe before the evidence-aware promotion gate added execution-cost stress evidence.

The execution-cost stress review is recorded at `outputs\research\execution_cost_stress_20260614\execution_cost_stress_review.md`. It compares the default quality-v2 allocator and the source-selection candidate under allocation-level rebalance cost rates of 0.00%, 0.03%, and 0.10%. The candidate supports across all three costs: at the strictest 0.10% cost, return edge is +5.41 percentage points, Sharpe edge is +0.106, and max drawdown is 1.79 percentage points lower than quality-v2. This is now passed into the formal promotion gate as the independent `execution_cost_stress` evidence group:

```powershell
python -m quant_etf_lab allocator-promotion-review --baseline-dir outputs\portfolio_walk_forward\main_chinext_portfolio_activation_wf_quality_v2 --candidate-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation3_v1 --sensitivity-group validation_3m --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation9_v1 --sensitivity-group validation_9m --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_train36_v1 --sensitivity-group train36 --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_stability02_v1 --sensitivity-group conservative_guard --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_margin03_v1 --sensitivity-group conservative_guard --sensitivity-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_margin06_v1 --sensitivity-group conservative_guard --evidence-snapshot outputs\research\execution_cost_stress_20260614\execution_cost_stress_snapshot.json --evidence-group execution_cost_stress --output-dir outputs\research\allocator_promotion_with_execution_cost_20260614 --min-return-edge 0.03 --min-sharpe-edge 0.05 --max-drawdown-worsening 0 --min-positive-window-ratio 0.80 --min-sensitivity-support 2
```

The evidence-aware 2026-06-14 promotion review is recorded at `outputs\research\allocator_promotion_with_execution_cost_20260614\allocator_promotion_review.md`. Decision: `promote_candidate` for research governance, with headline gate passed, sensitivity run support 2, pure sensitivity-group support 1, evidence-group support 1, and total independent support groups 2 / 2 from `conservative_guard` plus `execution_cost_stress`. After the switch-readiness probe passed, the research CLI defaults and refresh script now use the source-selection 6-month allocator as the default research allocator; this still does not connect to brokers, place orders, or change live positions.

The 2026-06-15 capital-efficiency allocator follow-up is recorded at `outputs\research\capital_efficiency_allocator_validation_20260615\capital_efficiency_allocator_validation.md`. It added an explicit `--score-mode capital_efficiency` option for portfolio and source-selection walk-forward training, rewarding excess return per average satellite allocation while keeping the drawdown and overuse penalties. The candidate run `main_chinext_portfolio_source_selection_validation6_capital_efficiency_v1` reached 48.40% stitched OOS return, -9.91% max drawdown, and 1.148 Sharpe versus the current default `validation6_v1` at 49.54%, -9.58%, and 1.166. Decision: reject for default allocator; keep the score mode as a diagnostic candidate only. It correctly avoided the fragile PF04 blend by selecting `core_only`, but the lower total return and worse drawdown do not justify replacing the default allocator.

The score-mode edge gate follow-up is recorded at `outputs\research\capital_efficiency_edge_allocator_validation_20260615\capital_efficiency_edge_allocator_validation.md`. It added `--score-mode-min-edge`, a default-off guard that requires non-balanced scoring to beat the balanced-score selection by a minimum active-score edge before switching. With no source-switch default/margin guard, `capital_efficiency --score-mode-min-edge 0.05` reproduced the current default allocator exactly: 49.54% stitched OOS return, -9.58% max drawdown, 1.166 Sharpe, and the same PF04 blend selection. With the existing quality-default/source-switch margin guard, PF04 still fell back to `core_only`, leaving the weaker 48.40%, -9.91%, 1.148 result. Decision: keep `--score-mode-min-edge` as a diagnostic/safety switch only; it does not create a new promotion candidate on this sample.

The source-guard decomposition follow-up is recorded at `outputs\research\source_guard_decomposition_20260615\source_guard_decomposition.md`, with a score-edge-only control at `outputs\research\source_guard_decomposition_20260615_score_edge_only\source_guard_decomposition.md`. It adds a lightweight comparison command for two source-selection runs:

```powershell
python -m quant_etf_lab source-guard-decomposition --default-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --candidate-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_capital_efficiency_edge05_v1 --output-dir outputs\research\source_guard_decomposition_20260615
```

The PF04 change is attributed to `source_switch_guard`, not `score_mode_gate`: the 0.03 margin forced `blend_q30_r70__sat25_ma60_drop03_health_ma60_mom20_dd15` back to `core_only` even though the raw edge was only about 0.010. That worsened PF04 OOS return by 0.77 percentage points, max drawdown by 0.33 percentage points, and Sharpe by 0.103 versus the current default. The score-edge-only control changed 0 / 5 windows. Decision: keep the current default allocator unchanged; if PF04 fragility needs additional control, review a separate risk-budget or market-state exposure rule instead of promoting the source-switch margin guard as the default.

The source risk-budget review is recorded at `outputs\research\source_risk_budget_review_20260615\source_risk_budget_review.md`. It adds a research-only `source-risk-budget-review` command that combines source-score edge, average satellite weight, satellite filter-off ratio, core-relative excess return, and optional source-guard decomposition deltas:

```powershell
python -m quant_etf_lab source-risk-budget-review --source-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --guard-decomposition-dir outputs\research\source_guard_decomposition_20260615 --output-dir outputs\research\source_risk_budget_review_20260615 --fragile-source-edge 0.03 --max-low-risk-satellite-weight 0.05 --min-filter-off-ratio 0.50
```

The review flags 2 fragile non-core source windows. PF04 is classified as `observe_without_source_switch`: its source edge is 0.008, average satellite weight is only 3.25%, the satellite filter is off 86.18% of days, and the source-switch guard reduced OOS return by 0.77 percentage points. PF02 is classified as `review_satellite_budget_cap`: its source edge is 0.029 with 11.29% average satellite exposure, positive core-relative return, and better drawdown, so the next experiment should test whether a smaller reversal sleeve cap preserves the edge with less allocator fragility. This remains diagnostic only and does not change defaults or live positions.

The source budget cap20 validation is recorded at `outputs\research\source_budget_cap20_validation_20260615\source_budget_cap20_validation.md`. It added a default-off `portfolio source-selection --max-satellite-weight` option and ran the 6-month source-validation allocator with a 20% risk-on satellite cap:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --source-validation-months 6 --max-satellite-weight 0.20 --run-id-prefix main_chinext_portfolio_source_selection_validation6_cap20_v1
```

The cap20 run reached 47.20% stitched OOS return, -9.64% max drawdown, and 1.124 Sharpe, versus the active default at 49.54%, -9.58%, and 1.166. PF02 moved from `reversal__sat25_ma60_drop03_unfiltered` to `reversal__sat20_ma60_drop03_unfiltered`, but its OOS return fell from 0.08% to -0.49%; PF04 and PF05 also gave up return. Decision: reject the global 20% cap for the default allocator. Keep `--max-satellite-weight` as a diagnostic experiment knob only.

The source-specific reversal cap20 validation is recorded at `outputs\research\source_reversal_cap20_validation_20260615\source_reversal_cap20_validation.md`. It added a default-off `--source-max-satellite-weight SOURCE=WEIGHT` option for source-selection runs and tested only `reversal=0.20` while leaving quality/blend candidates uncapped:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --source-validation-months 6 --source-max-satellite-weight reversal=0.20 --run-id-prefix main_chinext_portfolio_source_selection_validation6_reversal_cap20_v1
```

The source-specific cap is cleaner than global cap20 because PF04 and PF05 stay on the default sat25 blend/quality choices, while only PF02/PF03 reversal sleeves move to sat20. It still underperforms: 48.69% stitched OOS return, -9.58% max drawdown, and 1.149 Sharpe, down 0.85 percentage points of return and 0.017 Sharpe versus the active default. PF02 again turns from a tiny gain to a small loss. Decision: reject reversal-only cap20 for the default allocator and keep `--source-max-satellite-weight` as a diagnostic knob only.

The aggressive-source guard follow-up is recorded at `outputs\research\strategy_iteration_review_20260616\strategy_iteration_review.md`. It added a default-off `portfolio source-selection --source-switch-margin-by-source SOURCE=MARGIN` option so fragile research sources can require a larger validation-score edge without forcing a global incumbent-source margin. Both `highgain_pos8` and `pos10_risk_lean` failed default promotion at 10% caps because they were selected in PF04 on small validation-score edges and reduced stitched OOS results. The guarded run restored the active default allocator result:

```powershell
python -m quant_etf_lab portfolio source-selection --config configs\portfolio_core_source_selection_quality_reversal_highgain_pos8_pos10.yaml --train-months 24 --test-months 6 --step-months 6 --grid activation --max-train-drawdown 0.16 --source-validation-months 6 --source-max-satellite-weight highgain_pos8=0.05 --source-max-satellite-weight pos10_risk_lean=0.10 --source-switch-margin-by-source pos10_risk_lean=0.02 --source-switch-margin-by-source highgain_pos8=0.02 --run-id-prefix main_chinext_source_selection_pos10_highgain_source_margin02_validation6_v1
```

Result: 49.54% stitched OOS return, -9.58% max drawdown, and 1.166 Sharpe, with PF04 staying on `blend_q30_r70` instead of switching to `pos10_risk_lean`. Decision: keep the active default allocator unchanged and use source-specific margins as a guardrail when evaluating aggressive research sleeves.

The chip-reversal daily proxy is wired as a default-off `multi_factor` factor named `chip_reversal`. The lab edge is research-only because true chip distribution and 9:20-9:25 auction confirmation are not in the daily cache. The grid builder now preserves configured extra factors, so named walk-forward candidates keep `chip_reversal` instead of silently dropping it.

```powershell
python -m quant_etf_lab walk-forward --config configs\custom_watchlist_multifactor_momentum_iter_h_breadth_soft_focus_boost_highgain_pos8_chip_reversal05.yaml --train-years 4 --test-months 6 --step-months 6 --grid risk --objective robust --skip-missing --window-limit 1 --run-id-prefix smoke_pos8_chip_reversal05_wf1_gridfix
```

Decision: do not promote fixed positive chip-reversal factor sleeves. The valid 5% smoke at `outputs\walk_forward\smoke_pos8_chip_reversal05_wf1_gridfix` selected a candidate with `chip_reversal: 0.05`, but its first OOS window underperformed the `focus_highgain_pos8 robust/risk` baseline: 1.67% return, -6.40% drawdown, and 0.379 Sharpe versus the baseline 5.56%, -6.39%, and 1.031. The 2% diagnostic at `outputs\walk_forward\smoke_pos8_chip_reversal02_wf1_gridfix` also underperformed at 1.81%, -7.28%, and 0.324. Keep chip-reversal as a watch/screen signal; next experiments should be conditional flags, exclusion/penalty logic, or observation-queue validation. The PIT audit at `outputs\research\chip_reversal_pit_audit_sample500_20260616` passed on 2,000 events with 0 mismatches, confirming forward labels are not used as selector columns.

For chip-reversal observation candidates, run outcome backfill only as a research gate. It overlays the unified daily market snapshot by default, using `D:\codex\daily-market-data` before the exchange-ingest fallback, and it does not generate orders:

```powershell
python -m quant_etf_lab chip-reversal-candidate-outcomes --candidates-path outputs\research\chip_reversal_daily_candidates_20260616\chip_reversal_daily_candidates.csv --output-dir outputs\research\chip_reversal_candidate_outcomes_20260616_overlay_gate --horizons 1,2 --success-threshold 0.03 --min-ready-per-horizon 30 --min-success-rate 0.55
```

The 2026-06-16 outcome gate is currently blocked: 54 candidates produced 108 pending 1D/2D outcomes because the latest overlay snapshot is still 2026-06-16. Do not test chip-reversal as a conditional filter or penalty until the gate has at least 30 ready outcomes per horizon and at least 55% success rate.

Historical calibration for the latest chip-reversal gate is recorded at `outputs\research\chip_reversal_historical_gate_20260609_20260612`. Candidate generation for 2026-06-09 through 2026-06-12 disabled market-snapshot overlay, while outcome evaluation used the unified daily market snapshot only to mature forward labels. Only 1 of 4 daily gates passed; aggregate 1D success was 24.50% and aggregate 2D success was 43.00% across 400 ready candidates per horizon. Keep chip-reversal watch-only and do not promote it into factor weights, conditional filters, penalties, or allocator inputs until a broader rolling calibration passes.

The PF02 day-level attribution is recorded at `outputs\research\pf02_daily_attribution_20260615\portfolio_window_attribution.md`. It added a research-only `portfolio-window-attribution` command that replays selected source-allocation params for one window, compares variants against an optional core-only baseline, and writes per-variant metrics plus daily indexed-equity/daily-return deltas:

```powershell
python -m quant_etf_lab portfolio-window-attribution --config configs\portfolio_core_source_selection_quality_reversal_v1.yaml --start 20240704 --end 20250103 --variant default=outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1\pf_02_20240704_20250103_selected_params.json --variant reversal_cap20=outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_reversal_cap20_v1\pf_02_20240704_20250103_selected_params.json --include-core-only --output-dir outputs\research\pf02_daily_attribution_20260615
```

PF02 attribution confirms that the cap20 loss is mostly under-sizing a reversal sleeve that was net helpful in that window, not removing a damaging source. Default sat25 finished at +0.08% with -8.71% max drawdown and 0.109 Sharpe; reversal cap20 finished at -0.49%, -9.24%, and 0.052; core-only finished at -2.76%, -11.35%, and -0.172. The default-minus-cap20 indexed-equity edge ended at +0.57 percentage points and peaked at +0.67 percentage points on 2024-12-24. This supports keeping the active default unchanged and treating satellite caps as diagnostics rather than promotion candidates.

The allocator switch-readiness probe is recorded at `outputs\research\allocator_switch_readiness_20260614\allocator_switch_readiness.md`, with machine-readable deltas in `allocator_switch_readiness_comparison.csv` and `allocator_switch_readiness_snapshot.json`. It reran the default quality-v2 allocator and the source-selection 6-month candidate through the same daily-pipeline gates on 2026-06-12, using the evidence-aware promotion review. The candidate passed the same freshness/dashboard/after-close gates and improved paper-account total return from 43.28% to 49.47%, max drawdown from -11.51% to -9.58%, and Sharpe from 1.046 to 1.165. Both runs were still `risk_off` with 100% core / 0% satellite at the latest date. A model-audit timestamp generated after the as-of date is now accepted as a valid research-audit timestamp when its action list is clean, so the fixed probe has `model_audit_status=ok`, alert level `info`, action stage `monitor`, and no warning-level blocker. The report now separates default-switch readiness from satellite risk-budget readiness: `decision=ready_for_controlled_default_switch`, but `risk_budget_decision=wait_for_outcome_samples` until stock-target outcome samples pass readiness gates; the next review checkpoint is 2026-06-15 / 1d. The optional `--outcome-analysis-path` input loads `stock_target_review_outcome_analysis.json` into the same report, adding horizon gate status plus best/worst outcome groups for risk-budget review. Remaining items are monitored drawdown-review targets, suppressed-layer visibility, and outcome sample maturity. Rebuild the comparison after new default/candidate pipeline probes with:

```powershell
python -m quant_etf_lab allocator-switch-readiness --default-snapshot outputs\research\daily_pipeline_default_switch_probe_modelaudit_fixed_20260614\daily_pipeline_snapshot.json --candidate-snapshot outputs\research\daily_pipeline_candidate_switch_probe_modelaudit_fixed_20260614\daily_pipeline_snapshot.json --output-dir outputs\research\allocator_switch_readiness_20260614 --default-label "Default quality-v2" --candidate-label "Candidate source-selection 6m" --outcome-analysis-path outputs\research\paper_account_candidate_switch_probe_modelaudit_fixed_20260614\stock_target_review_outcome_analysis.json
```

The controlled default switch was then applied to the research CLI defaults, module defaults, and `scripts\run_daily_research_refresh.ps1`. The active research allocator default is now `outputs\portfolio_source_selection\main_chinext_source_selection_highgain_pos8_dd50_cap30_activation_dd50_20260624`, and the active phase-2 component snapshot is `outputs\research\phase2_model_status_source_selection_default_20260614\phase2_components.csv`. The verification run `outputs\research\daily_pipeline_default_candidate_switched_20260614\daily_pipeline.md` completed with dashboard posture `core_base_watch_allocator_gate`, model audit `ok`, promotion decision `promote_candidate`, alert level `info`, and paper-account metrics matching the candidate: 49.47% total return, -9.58% max drawdown, and 1.165 Sharpe. The old quality-v2 allocator remains the rollback/baseline comparison path, not the active research default.

Run the post-switch observation report after each default-candidate daily pipeline run:

```powershell
python -m quant_etf_lab allocator-observation --pipeline-snapshot outputs\research\daily_pipeline_default_candidate_switched_20260614\daily_pipeline_snapshot.json --output-dir outputs\research\allocator_observation_default_candidate_20260614 --baseline-label "Rollback quality-v2"
```

The 2026-06-14 observation report is recorded at `outputs\research\allocator_observation_default_candidate_20260614\allocator_observation.md`. It writes `allocator_observation_snapshot.json` and `allocator_observation_checklist.csv`, with status `waiting_for_outcome_samples`, next action stage `accumulate_outcome_samples`, no blocking items, monitor-only items for reviewed drawdown targets, suppressed-layer visibility, manual-pending rows, and outcome maturity. The next 1D outcome review date is 2026-06-15; 5D/10D/20D dates are 2026-06-19, 2026-06-26, and 2026-07-10. Keep the risk-budget decision at `hold_default_core_base` until outcome sample gates are ready.

Run the standalone satellite risk-budget review when stock-target outcomes start to mature:

```powershell
python -m quant_etf_lab satellite-risk-budget-review --pipeline-snapshot outputs\research\daily_pipeline_candidate_switch_probe_modelaudit_fixed_20260614\daily_pipeline_snapshot.json --outcome-analysis-path outputs\research\paper_account_candidate_switch_probe_modelaudit_fixed_20260614\stock_target_review_outcome_analysis.json --output-dir outputs\research\satellite_risk_budget_review_20260614 --trial-satellite-budget 0.05 --max-satellite-budget 0.20 --min-overall-win-rate 0.55 --min-overall-avg-return 0.0 --max-worst-group-loss -0.05
```

The review writes `satellite_risk_budget_review.md`, `satellite_risk_budget_snapshot.json`, and `satellite_risk_budget_checklist.csv`. It is research-only and does not change model parameters, target weights, or broker positions. The default gate allows at most a 5% satellite trial only when at least one outcome horizon is ready, the overall win rate is at least 55%, the overall average return is non-negative, and the worst ready group does not breach the -5% average-return guard. Otherwise it keeps the budget at `hold_default_core_base` or `wait_for_outcome_samples`. The daily pipeline now runs this review automatically after paper-account/outcome analysis and before history/alerts, then records `satellite_risk_budget_decision`, `satellite_risk_budget_recommended_satellite_weight`, and the report/checklist/snapshot paths in `daily_pipeline_snapshot.json`, `daily_pipeline_history.csv`, `daily_pipeline.md`, and `alerts.md`. The standalone command remains useful for isolated threshold checks.

The equal-window robustness review is recorded at `outputs\research\allocator_equal_window_review_20260614\equal_window_review.md`. Clipping quality-v2, `source_validation_6m`, and train36 to the common OOS range `2025-01-06` to `2026-06-12` shows that `source_validation_6m` and train36 are identical on the overlapping windows: 38.39% return, -9.58% max drawdown, and 1.637 Sharpe, versus quality-v2 at 36.54%, -9.91%, and 1.582. This adds supportive equal-window context for the 6-month source-validation candidate; the evidence-aware promotion review with execution-cost stress is the controlling promotion gate.

The source-stability penalty review is recorded at `outputs\research\source_stability_review_20260614\source_stability_review.md`. A `source_stability_penalty` of `0.02` reduced one tiny source switch in the 6-month validation run, but OOS return fell from 49.47% to 48.17%, max drawdown worsened from -9.58% to -9.91%, and Sharpe fell from 1.166 to 1.144. Keep the feature available for diagnostics, but do not adopt `source_stability_penalty=0.02` as the default.

The holdout time-slice review is recorded at `outputs\research\allocator_holdout_review_20260614\holdout_review.md`. Across five independent half-year slices, `source_validation_6m` wins return in 3 / 5 and wins return, drawdown, and Sharpe together in 3 / 5. `margin06` wins return in only 1 / 5 and is not adopted. The evidence-aware promotion gate is the current controlling promotion review; it promotes the 6-month source-validation candidate at the research-governance level after adding independent execution-cost evidence, and the research default allocator now uses that candidate while outcome samples continue to mature.

Run the source-decision explainability report used by the promotion review:

```powershell
python -m quant_etf_lab source-decision-review --source-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --baseline-dir outputs\portfolio_walk_forward\main_chinext_portfolio_activation_wf_quality_v2 --output-dir outputs\research\source_decision_review_20260614
```

The source-decision review is recorded at `outputs\research\source_decision_review_20260614\source_decision_review.md`. It shows all 5 6-month validation selections were raw-best source selections: `core_only` once, `reversal` twice, `blend_q30_r70` once, and `quality` once. The weakest source edge is `pf_04_20250704_20260103`, where `blend_q30_r70` beat the next source by only 0.008 score points, so promotion review should treat that window as fragile.

Run the phase-2 construction status report:

```powershell
python -m quant_etf_lab phase2 --output-dir outputs\research\phase2_model_status_source_selection_default_20260614 --portfolio-wf-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1
```

The phase-2 report reads the current full-sample THS-style baseline, the defensive core walk-forward, the v2 satellite walk-forward, the guarded v2 core-satellite portfolio, and the active source-selection allocation walk-forward. It writes `phase2_components.csv` and `phase2_status.md`, marking each layer as complete, running, or incomplete. In the 2026-06-14 source-selection default status run, all 5 components were complete. Treat the defensive core as the base and use allocator-gated satellite exposure only.

Run the phase-2 monitoring review after the source-selection default allocator is complete:

```powershell
python -m quant_etf_lab phase2-review --phase2-status-dir outputs\research\phase2_model_status_source_selection_default_20260614 --allocator-wf-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --output-dir outputs\research\phase2_monitoring_review_source_selection_default_20260614
```

The monitoring review reads `outputs\research\phase2_model_status_source_selection_default_20260614\phase2_components.csv` and `outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1` by default, then writes `phase2_review.md` and `phase2_review_snapshot.json`. The historical activation-v2 quality allocator can still be reviewed by explicitly passing `outputs\research\phase2_model_status_activation_v2_20260613\phase2_components.csv` and `outputs\portfolio_walk_forward\main_chinext_portfolio_activation_wf_quality_v2`.

To surface the allocator promotion watchlist in the same review, add the promotion review directory:

```powershell
python -m quant_etf_lab phase2-review --phase2-status-dir outputs\research\phase2_model_status_source_selection_default_20260614 --allocator-wf-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --promotion-review-dir outputs\research\allocator_promotion_with_execution_cost_20260614 --output-dir outputs\research\phase2_monitoring_review_with_execution_cost_promotion_20260614
```

Run the after-close daily model check:

```powershell
python -m quant_etf_lab daily-check --date-stamp
```

The daily check wraps the phase-2 review, reads the active source-selection 6-month research allocator by default, writes `daily_model_check.md` and `daily_model_check_snapshot.json`, and labels local data freshness before interpreting the model posture. Add `--promotion-review-dir outputs\research\allocator_promotion_with_execution_cost_20260614` when the daily report should carry the current allocator promotion decision, return edge, Sharpe edge, drawdown change, total independent support groups, pure sensitivity-group support, evidence-group support, and raw sensitivity-run support. With `--date-stamp`, output is archived under `outputs\research\daily_model_check_YYYYMMDD`. It is still a research/status artifact only; stock-level alerts remain in the separate trigger-monitor workflow under `D:\codex\outputs`.

Build the paper-account ledger and rebalance audit from the active source-selection allocator:

```powershell
python -m quant_etf_lab paper-account --config configs\portfolio_core_satellite_quality_v2_guarded.yaml --allocator-wf-dir outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1 --output-dir outputs\research\paper_account_source_selection_default_20260614
```

The paper account writes `ledger.csv`, `rebalance_audit.csv`, `monthly_returns.csv`, `target_holdings.csv`, `target_holdings.json`, `target_holdings.md`, `stock_targets.csv`, `stock_targets.json`, `stock_targets.md`, `stock_target_review.csv`, `stock_target_review.json`, `stock_target_review.md`, `stock_target_review_actions.csv`, `stock_target_review_actions.json`, `stock_target_review_actions.md`, `stock_target_review_outcomes.csv`, `stock_target_review_outcomes.json`, `stock_target_review_outcomes.md`, `metrics.json`, and `paper_account.md`. It also maintains a persistent manual review note sheet at `outputs\research\stock_target_review_notes.csv` by default and writes a run-local `stock_target_review_notes.csv` snapshot beside the paper-account outputs. It is a research-only simulation account: the core and satellite curves are treated as already cost-adjusted by their own backtests, while `--rebalance-cost-rate` can optionally estimate additional portfolio allocation-level target-change costs. `target_holdings.*` is a layer-level target sheet for core, satellite, and cash. `stock_targets.*` reconstructs stock-level target decomposition from local walk-forward backtest `trades.csv` files and scales positions by the current core/satellite target weights; suppressed satellite rows remain visible with zero portfolio target weight. The stock-level output also records each source layer's strategy profile, factor-weight explanation, source risk/filter status, unrealized return diagnostic, and latest trigger-monitor match from `D:\codex\outputs\signal_history\signals_latest.csv` when that file is available; override this with `--trigger-signal-path` for another signal CSV. `stock_target_review.*` ranks the current target stocks for human review by trigger-monitor matches, unrealized loss thresholds, layer suppression, target weight, and recent source trades; it does not alter target weights or generate broker orders. `stock_target_review_actions.*` is the smaller daily checklist for rows that still need manual action: pending `review_required` rows, `watch` follow-ups, `exclude_candidate` rows, unrecognized manual statuses, and due `next_review_date` items. `stock_target_review_outcomes.*` tracks later 1/5/10/20 trading-day returns from local OHLCV when those future bars exist; unavailable horizons stay `pending` rather than being guessed. The persistent note sheet preserves `manual_status`, `manual_note`, `reviewed_at`, `reviewed_by`, and `next_review_date` across reruns, while model fields such as `last_review_stage` and `last_model_reason` are refreshed. Supported `manual_status` values are `unreviewed`, `reviewed`, `watch`, `resolved`, and `exclude_candidate`; Chinese aliases such as `待复核`, `已复核`, `观察`, `已解决`, and `排除候选` are also classified. `exclude_candidate` is only an audit label in v1 and does not remove the row from model targets. Override the note-sheet location with `--stock-review-notes-path`. The review thresholds are configurable with `--stock-review-drawdown-threshold`, `--stock-review-watch-drawdown-threshold`, `--stock-review-loss-attention-threshold`, `--stock-review-gain-attention-threshold`, and `--stock-review-watch-score-threshold`; for example, use `-0.08` for stricter drawdown review or `-0.12` for a wider review band. These files are not broker order lists. In the 2026-06-13 activation-v2 run, the default no-extra-cost ledger ended at 1,432,764.52, with 43.28% total return, -11.51% max drawdown, 1.046 Sharpe, and 29 audit events.

The 2026-06-13 activation-v2 metrics above are retained as historical baseline context. In the 2026-06-14 controlled default-switch verification, the active source-selection default ledger ended at 1,494,744.88, with 49.47% total return, -9.58% max drawdown, 1.165 Sharpe, and 42 audit events.

The paper account also writes `stock_target_review_outcome_calendar.csv`, `stock_target_review_outcome_calendar.json`, `stock_target_review_outcome_calendar.md`, `stock_target_review_outcome_due.csv`, `stock_target_review_outcome_due.json`, and `stock_target_review_outcome_due.md`. The standalone calendar converts pending/evaluable 1/5/10/20 trading-day outcomes into the next estimated review date, sample-readiness state, source history rows, and research-only/no-broker-action markers. The due queue filters that calendar to horizons whose estimated maturity date is on or before the latest local paper-account date but whose outcomes are still pending.

Recommended after-close one-command workflow:

```powershell
scripts\run_daily_research_refresh.bat
```

`scripts\run_daily_research_refresh.bat` first refreshes `data\processed\stock_market_cap_yi.csv`, then runs `daily-pipeline` with the active core-satellite config, the active source-selection 6-month allocator directory, the evidence-aware promotion review directory, `--stock-tracking-max-market-cap-yi 1000000`, the persistent stock-review outcome history path, and `--stock-review-warning-only-after-close`. If the live market-cap refresh fails but a non-empty local cache already exists, the script records the market-cap stderr under `outputs\logs\market_cap_update_*.err.log`, warns, and continues with the existing cache; if no cache exists, it still fails. The daily pipeline also writes an automatic satellite risk-budget review under `outputs\research\satellite_risk_budget_review_YYYYMMDD` unless `--satellite-risk-budget-output-dir` is supplied. For a quick verification run that reuses the existing market-cap cache, use `powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_refresh.ps1 --skip-market-cap-update`.

The pipeline runs `daily-check`, `paper-account`, `dashboard`, an A-share trading-day/after-close data gate, the automatic satellite risk-budget review, an automatic history review, and daily alert rules in sequence, using dated output directories by default. It writes `daily_pipeline.md`, `daily_pipeline_snapshot.json`, `alerts.md`, and `alerts.json` under `outputs\research\daily_pipeline_YYYYMMDD`, updates `outputs\research\latest_alerts.md`, appends one run row to `outputs\research\daily_pipeline_history.csv` unless `--history-file` points elsewhere, and writes matching pipeline-history and satellite risk-budget reviews under `outputs\research\pipeline_history_review_YYYYMMDD` and `outputs\research\satellite_risk_budget_review_YYYYMMDD`. The daily report links the current paper-account layer target sheet, stock-level target decomposition, stock-target review panel, stock-target action queue, stock-target outcome tracker, satellite risk-budget review, model-build hygiene audit, allocator promotion watch report, and persistent review notes file so core/satellite/cash weights, reconstructed stock targets, human-review status, open manual actions, later outcome availability, risk-budget readiness, walk-forward cleanup items, and allocator promotion evidence can be checked from the same entry point. The history row records freshness gates, trading-day gate state, posture, paper-account weights, stock-target review counts, manual-note counts, manual-status counts, stock-target action counts, stock-target outcome counts, satellite risk-budget decision/recommended weight, dashboard same-run history-refresh status, model-audit status/action counts, allocator promotion status/decision, promotion total independent support, evidence support, sensitivity group/run support, final equity, drawdown, Sharpe, and key report paths so changes can be reviewed across runs. The history review now also displays latest promotion decision/support counts and latest satellite risk-budget status, decision, selected horizon, report path, recommended budget, decision change, and recommended-weight change. It marks health `watch` when run-level promotion support is sufficient but independent group support is still below the promotion gate, when satellite risk-budget review fails, when a small satellite trial becomes eligible, or when the risk-budget decision/recommended weight changes across runs. The recommended refresh script includes `--promotion-review-dir outputs\research\allocator_promotion_review_dd50_vs_validation6_v1_relaxed2` so the current `promote_candidate` research-governance evidence appears in `daily-check`, dashboard, daily-pipeline snapshot, and daily-pipeline history; the active research allocator default is `outputs\portfolio_source_selection\main_chinext_source_selection_highgain_pos8_dd50_cap30_activation_dd50_20260624`. The alert rules flag stale data gates, dashboard wait/defensive states, dashboard same-run history-refresh failures, paper-account drawdown breaches, day-to-day return drops, posture changes, satellite target-weight changes, satellite risk-budget wait/hold/trial/failed states, weekend/non-trading runs, after-close data-not-ready blocks, market-cap cache health, model-build audit action items, correlated allocator-promotion evidence, stock-target review queues, review-required rows that still have no manual note, manual watch rows, manual exclusion candidates, unrecognized manual statuses, and stock-target outcome maturity/readiness. Model-audit status with open action/resume/archive candidates raises a `warning` with a `review_model_build_audit` action item and links both the audit report and `walk_forward_run_actions.csv`; a generated-after-as-of audit with zero open action rows is retained as `ok` provenance. `drawdown_review` or trigger-monitor matches in `stock_target_review.*` promote the alert posture to `warning` / `review_required` only when there are unreviewed rows or open stock-target actions; reviewed/resolved drawdown rows are retained as `info` / `monitor`. Satellite risk-budget `review_failed` raises `warning`, while wait/hold/trial-eligible states stay `info` because they are research-review prompts, not broker actions. Unavailable, empty, or stale market-cap cache also raises `warning` because the 1500 Yi Yuan tracking exclusion may be incomplete. Correlated promotion evidence, suppressed-layer rows, watch rows, missing manual notes, partial market-cap coverage, manual-status follow-ups, and outcome-maturity waits are kept as `info` / `monitor`. On a trading-day-ready run with no higher-priority review item, correlated promotion evidence becomes the pipeline next-step stage `monitor_allocator_promotion_evidence`; non-trading-day runs still prioritize `wait_for_next_trading_close`. Add `--model-audit-dir outputs\research\model_build_audit_YYYYMMDD` to pin the dashboard and pipeline to a specific audit run; otherwise they look for the latest model-build audit directory. Add `--stock-review-warning-only-after-close` when scheduled weekend/non-trading runs should record target-stock review items as `info` unless the A-share after-close gate is ready. The alert report also writes an `Action Playbook`, mapping each alert to a `blocked`, `review_required`, `monitor`, or `routine` action stage. On 2026-06-14, the controlled default-switch verification wrote `outputs\research\daily_pipeline_default_candidate_switched_20260614\daily_pipeline.md` with promotion decision `promote_candidate`, dashboard posture `core_base_watch_allocator_gate`, trading-day gate `trading_day_data_ready`, after-close data `ready`, history health `ok`, model audit `ok`, alert level `info`, and next step `accumulate_outcome_samples`.

By default, alert and history-health findings are written to reports without failing the command. For scheduled monitoring, add `--fail-on-alert-level warning` or `--fail-on-history-health watch`; the command then exits with code `4` for alert-level breaches and `5` for history-health breaches, while still writing the report artifacts first.

The daily pipeline snapshot, history row, and alert report now carry the outcome-calendar and due-queue report paths plus next action date/horizon, so the next outcome review can be found from `daily_pipeline.md`, `alerts.md`, or `daily_pipeline_history.csv` without opening the full analysis JSON.

Chip-reversal candidate outcomes stay disabled in `daily-pipeline` unless explicitly requested. To refresh the watch-only maturity/gate snapshot inside the daily run, add:

```powershell
python -m quant_etf_lab daily-pipeline --chip-reversal-candidate-outcomes --chip-reversal-candidates-path outputs\research\chip_reversal_daily_candidates_20260616\chip_reversal_daily_candidates.csv --chip-reversal-outcome-horizons 1,2 --chip-reversal-outcome-success-threshold 0.03 --chip-reversal-outcome-min-ready-per-horizon 30 --chip-reversal-outcome-min-success-rate 0.55
```

This only updates research reports, pipeline snapshots, and history fields; it keeps `broker_action=none`.

Review the accumulated pipeline history separately when you only need to re-check prior daily runs:

```powershell
python -m quant_etf_lab pipeline-history --as-of-date 2026-06-13
```

The history review reads `outputs\research\daily_pipeline_history.csv`, writes `pipeline_history_review.md` and `pipeline_history_review_snapshot.json`, and labels the research workflow as `ok`, `watch`, or `blocked`. It checks whether the latest pipeline row is fresh, whether hard data gates failed, whether reference gates need review, whether the paper account breached configurable current-drawdown or Sharpe watch thresholds, and whether the dashboard same-run history refresh failed. The report also displays the dashboard-refresh status, dashboard pipeline-history health, and the risk-budget decision seen by that refreshed dashboard.

The dashboard command can also surface the latest pipeline-history review and post-switch allocator observation. Pass `--pipeline-history-dir outputs\research\pipeline_history_review_YYYYMMDD` to pin a specific history review; otherwise it looks for the latest `pipeline_history_review_*` directory. Pass `--daily-run-status-dir outputs\research\daily_run_status_latest` or `--allocator-observation-dir outputs\research\allocator_observation_YYYYMMDD` to pin the scheduled-refresh status and observation snapshot. This adds the history health state, alert count, latest satellite risk-budget decision, selected outcome horizon, recommended satellite budget, decision/recommended-weight changes, daily run state, observation status, outcome-ready horizon count, next review date/horizon, and allocator-observation risk-budget decision to `latest_dashboard.md` without changing dashboard posture gates or any model/account targets. The daily-pipeline command now refreshes the same dashboard output once more after a completed history review, so the one-command run's dashboard includes the same-run pipeline-history context.

Windows helper scripts are available for manual runs or Task Scheduler:

```powershell
scripts\run_daily_research_refresh.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_refresh_logged.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_refresh.ps1 --skip-market-cap-update
scripts\run_daily_pipeline.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_pipeline.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_daily_pipeline_logged.ps1
scripts\run_daily_pipeline_preflight_smoke.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_pipeline_preflight_smoke.ps1
scripts\run_live_shadow_preflight_smoke.bat
powershell -ExecutionPolicy Bypass -File scripts\run_live_shadow_preflight_smoke.ps1
scripts\run_daily_model_check.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_model_check.ps1
scripts\run_daily_research_refresh_with_observation.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_refresh_with_observation.ps1
scripts\run_daily_research_refresh_with_observation_preflight_smoke.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_refresh_with_observation.ps1 --run-daily-pipeline-preflight-smoke
scripts\install_daily_research_preflight_task.bat
powershell -ExecutionPolicy Bypass -File scripts\install_daily_research_preflight_task.ps1
scripts\check_daily_research_status.bat
powershell -ExecutionPolicy Bypass -File scripts\check_daily_research_status.ps1
scripts\check_daily_research_preflight_task_status.bat
powershell -ExecutionPolicy Bypass -File scripts\check_daily_research_preflight_task_status.ps1
scripts\run_daily_research_preflight_smoke_and_status.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_preflight_smoke_and_status.ps1
scripts\run_daily_research_preflight_ops.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_preflight_ops.ps1
``` 

For a one-command smoke run + immediate status snapshot, run:

```powershell
scripts\run_daily_research_preflight_smoke_and_status.bat
powershell -ExecutionPolicy Bypass -File scripts\run_daily_research_preflight_smoke_and_status.ps1
```

For an end-to-end manual ops flow (preflight status check + smoke + status summary + dashboard), run:

```powershell
scripts\run_daily_research_preflight_ops.bat
```

To skip dashboard generation in this flow (for automation only), run:

```powershell
scripts\run_daily_research_preflight_ops.bat -NoDashboard
```

To make the preflight flow fail-fast when live preflight is blocked (exit code 7), run:

```powershell
scripts\run_daily_research_preflight_ops.bat -FailOnBlocked
```

To create a scheduled task that runs this one-command smoke+status flow, include `-WithStatusCheck` when running installer.
To create a scheduled task that runs the full ops flow (preflight status + smoke + status summary + dashboard), include `-WithOps` when running installer.

Exit-code notes:
- `75`: another daily preflight run is already executing (scheduler lock is held).
- non-zero otherwise: underlying preflight/observation pipeline error; review the printed wrapper status file under `outputs\logs`.

## 实盘前置 SOP（手工执行）

建议按以下顺序核对（从前到后）：

1. 先看任务状态（确认是否有锁/是否有可用结果）
```powershell
scripts\check_daily_research_preflight_task_status.bat
```

2. 运行一键预检 + 当前状态快照
```powershell
scripts\run_daily_research_preflight_smoke_and_status.bat
```

3. 按返回码处理
- `75`：有同类运行在进行中，通常 5-10 分钟后再试（排他锁未释放）。
- `1`：失败，先看 `outputs\logs\daily_research_refresh_with_observation_YYYYMMDD_HHMMSS.status.json` 中的 `final_stage` / `refresh_exit_code`。
- 其他非零：继续核对 `pipeline_preflight_smoke_exit_code` / `observation_exit_code` / `daily_run_status_exit_code`。

4. 成功后，刷 status 并更新看板
```powershell
scripts\check_daily_research_status.bat
python -m quant_etf_lab dashboard
```

建议最小执行链：`run_daily_research_preflight_ops.bat`

A Windows Task Scheduler task named `QuantETF Daily Pipeline` runs the research refresh every Monday-Friday at 16:10 Asia/Shanghai. For the full research refresh plus post-switch allocator observation path, point the scheduler to `scripts\run_daily_research_refresh_with_observation.bat`; for stricter daily quality checks, use `scripts\run_daily_research_refresh_with_observation_preflight_smoke.bat` (or `scripts\run_daily_research_refresh_with_observation.ps1 --run-daily-pipeline-preflight-smoke`). It uses an exclusive scheduler-level lock at `outputs\locks\daily_research_refresh_with_observation.lock` so overlapping manual/scheduled starts fail fast instead of running duplicate refreshes. The wrapper writes `daily_research_refresh_with_observation_YYYYMMDD_HHMMSS.status.json` with the final stage plus refresh, observation, and status-summary exit codes. Refresh logs are written to `outputs\logs\daily_research_refresh_YYYYMMDD_HHMMSS.out.log`, `outputs\logs\daily_research_refresh_YYYYMMDD_HHMMSS.err.log`, and `outputs\logs\daily_research_refresh_YYYYMMDD_HHMMSS.status.json`; the follow-up observation writes `allocator_observation_YYYYMMDD_HHMMSS.*` logs and `outputs\research\allocator_observation_YYYYMMDD`; the wrapper then automatically runs `daily-run-status`, writing `daily_run_status_YYYYMMDD_HHMMSS.*` logs and refreshing `outputs\research\daily_run_status_latest\daily_run_status.md`. You can add `--run-daily-pipeline-preflight-smoke` to run the pre-flight smoke check right after refresh and before allocator observation; it writes `pipeline_preflight_smoke_YYYYMMDD_HHMMSS.{out,err,status}.json` under `outputs\logs`. The older `run_daily_research_refresh_logged.ps1` remains available for refresh-only diagnostics, and `run_daily_pipeline_logged.ps1` remains available for low-level daily-pipeline-only diagnostics.

For automation or pre-trade gateways, you can also request machine-readable status:

```powershell
python -m quant_etf_lab daily-run-status --print-json --json-path outputs\research\daily_run_status_payload.json
```

The command returns exit code `6` when `--fail-on-problem-state` is enabled and `problem_state` is true.

To install a scheduled preflight-quality task once, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_daily_research_preflight_task.ps1
```

The installer creates a weekly Monday-Friday task named `QuantETF Daily Pipeline (Preflight Smoke)` (or `QuantETF Daily Pipeline (Preflight Smoke + Status)` when `-WithStatusCheck` is used with default name) with `/SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:10 /RU $env:USERDOMAIN\$env:USERNAME /RL HIGHEST` and runs `scripts\run_daily_research_refresh_with_observation_preflight_smoke.bat` (or `scripts\run_daily_research_preflight_smoke_and_status.bat` with `-WithStatusCheck`). You can override `-TaskName`, `-StartTime`, `-RunAsUser`, and `-RunLevel` (`HIGHEST` default, `LIMITED` fallback when `HIGHEST` is blocked by user privileges). Pass `-DryRun` for a dry-run preview before committing.
When `-WithOps` is used (with or without `-WithStatusCheck`), installer writes the task to run `scripts\run_daily_research_preflight_ops.bat` and defaults the name to `QuantETF Daily Pipeline (Preflight Smoke + Ops)`.

For example, to install the status-enriched scheduled task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_daily_research_preflight_task.ps1 -WithStatusCheck -TaskName "QuantETF Daily Pipeline (Preflight Smoke + Status)"
```

For example, to install the full manual-ops flow in schedule:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_daily_research_preflight_task.ps1 -WithOps -TaskName "QuantETF Daily Pipeline (Preflight Ops)"
```

For a fail-fast scheduled preflight (exit early when live preflight is blocked), add `-FailOnBlocked`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_daily_research_preflight_task.ps1 -WithOps -FailOnBlocked -TaskName "QuantETF Daily Pipeline (Preflight Ops, FailFast)"
```

For a fail-fast status-check-only task, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_daily_research_preflight_task.ps1 -WithStatusCheck -FailOnBlocked -TaskName "QuantETF Daily Pipeline (Preflight Smoke + Status, FailFast)"
```

After the scheduled task runs, open `outputs\research\daily_run_status_latest\daily_run_status.md` first. Use `python -m quant_etf_lab daily-run-status` or `scripts\check_daily_research_status.bat` only when you want to refresh that read-only status page manually. Add `--fail-on-problem-state` for automation gates; the command still writes the status artifacts, returns exit code `6` only for problem states, and treats `waiting_for_scheduled_run` plus `waiting_for_outcome_samples` as non-failing waiting states. The status check summarizes the latest wrapper final stage, refresh exit code, allocator-observation exit code, pipeline preflight-smoke exit code/status path, stock-cache latest date, ready outcome horizons, next review date/horizon, and satellite risk-budget decision. If a wrapper-level failure such as `lock_conflict`, `daily_pipeline_preflight_smoke_failed`, `wrapper_failed`, `refresh_failed`, or `observation_failed` is present, it takes priority over older successful refresh logs when computing `run_state`. Before the expected schedule time, currently `16:10`, a same-day outcome review with no wrapper/observation log is labeled `waiting_for_scheduled_run` rather than a missing-observation failure.

Build the one-page local dashboard after daily-check and sentiment reports are ready:

```powershell
python -m quant_etf_lab dashboard
```

The dashboard writes `latest_dashboard.md` and `latest_dashboard_snapshot.json` under `outputs\research\latest_dashboard`. It reads the latest daily model check, paper-account ledger, market sentiment report, cached OHLCV files, activation allocator inputs, model-build hygiene audit, post-switch allocator observation, daily-run status, and `D:\codex\outputs\trigger_reports\latest_trigger.md`, then labels the overall posture without placing orders or changing any strategy state. Its freshness gates check the daily model date, market cache coverage, allocator curve and summary dates, paper-account date, sentiment date, and trigger-monitor date before accepting a review posture; the model-audit and allocator-observation sections are reported as build/monitoring context and do not by themselves change the trading posture. Helper scripts:

```powershell
scripts\run_latest_dashboard.bat
powershell -ExecutionPolicy Bypass -File scripts\run_latest_dashboard.ps1
```

Long-running commands use process locks under `outputs/locks/` to avoid duplicate runs. If an equivalent `data update`, `backtest`, `validate`, `sentiment`, `factor-lab`, `daily-check`, `paper-account`, `dashboard`, `daily-pipeline`, `pipeline-history`, `walk-forward`, `portfolio combine`, or `portfolio walk-forward` command is already running, the CLI prints `Duplicate process blocked` and exits with code `3`. A stale lock is automatically replaced when the recorded PID is no longer running. For long walk-forward runs, add `--resume` with the same `--run-id-prefix` to skip completed windows and continue writing `walk_forward_summary.csv`, `candidate_results.csv`, `overfit_audit.csv`, and `oos_equity_stitched.csv`.

Walk-forward locks are keyed by config, output directory, run-id, train/test/step settings, grid, objective, drawdown limit, and selection-validation months. `--window-limit` is intentionally excluded so a smoke run and a later `--resume` run with the same `--run-id-prefix` cannot write the same output directory concurrently.

Project-level Codex subagents are defined under `.codex\agents` for internal delegation during long research work. Use `quant-research-reviewer` for strategy and leakage review, `walk-forward-operator` for resumable validation runs, `data-cache-auditor` for local cache freshness and coverage checks, and `model-governance-reviewer` for promote/watch/reject decisions. These agents are project-specific guardrails for research velocity and evidence quality; they do not replace `pytest`, `compileall`, `model-audit`, or the daily pipeline gates.

For quick walk-forward smoke tests, add `--window-limit N` to run only the first `N` rolling windows before launching the full grid:

```powershell
python -m quant_etf_lab walk-forward --config configs\ashare_main_chinext_multifactor_reversal_sample120.yaml --preset main-chinext-reversal-v1 --run-id-prefix smoke_reversal_one_window --skip-missing --window-limit 1
```

Run a model-build hygiene audit after adding a new strategy family or completing a long walk-forward run:

```powershell
python -m quant_etf_lab model-audit --output-dir outputs\research\model_build_audit_YYYYMMDD
```

The audit scans config-section duplication, config inheritance coverage, incomplete walk-forward output directories, process-lock state, and speed-control reminders. It also writes `config_inheritance_map.csv`, a root/base config map with direct `extends` links, plus `walk_forward_run_actions.csv`, a non-destructive action list for interrupted or suspicious walk-forward runs. The 2026-06-14 post-migration audit scanned 52 root config files and 27 base/fragment configs, found 0 duplicate config-section groups, 0 root configs without `extends`, 4 incomplete or suspicious walk-forward runs, 3 resume/finalize candidates, 1 archive-review candidate requiring manual confirmation, and 0 active external lock files.

Resolved or superseded walk-forward directories can be recorded without deleting historical outputs:

```csv
run_name,resolution_status,replacement_run,resolved_at,resolved_by,resolution_note
old_partial_run,superseded,later_complete_run,2026-06-14,codex,Replaced by later complete walk-forward evidence.
```

By default `model-audit` reads `outputs\research\walk_forward_run_resolutions.csv`. Resolved rows stay visible in `walk_forward_runs.csv` with their original partial status, but they are no longer included in `walk_forward_run_actions.csv`. The current registry resolves 3 old interrupted walk-forward directories as superseded by later complete runs, leaving 1 empty `missing_outputs` directory as an archive-review candidate that still requires manual confirmation before moving or deleting.

## Notes

- Signals are generated after the close and executed at the next available open.
- Default ETF costs: 0.03% commission, 0.05% slippage, and no stamp tax.
- The CSI 300 stock config adds 0.05% sell-side stamp tax and 100-share lot sizing.
- Historical OHLCV backtest data is cached under `data/raw/`, `data/processed/`, and `data/meta/`.
- Daily exchange snapshot data is read first from `D:\codex\daily-market-data`; if the latest snapshot is missing there, the reader falls back to `D:\codex\2026-06-15-exchange-data-ingest` after checking `scripts\market_data_utils.py` via `ensure_latest_fetch_ok()` / `get_latest_fetch_status()`. Dashboard and daily-pipeline market-cache freshness now default to this unified daily hub.
- AKShare data is used for research only and external data interfaces may change.
