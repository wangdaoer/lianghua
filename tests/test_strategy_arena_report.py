import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_strategy_arena_report import (
    ArenaEntrant,
    build_pairwise_comparison,
    build_portfolio_division,
    build_signal_observation_division,
    load_equity_curve,
    run_arena,
    update_arena_history,
)


def write_curve(path: Path, dates: pd.DatetimeIndex, returns: list[float], exposure: float) -> None:
    nav = 1_000_000.0 * pd.Series(1.0 + np.asarray(returns), index=dates).cumprod()
    pd.DataFrame(
        {
            "date": dates,
            "equity": nav.to_numpy(),
            "turnover": 0.10,
            "cost": 0.0001,
            "gross_exposure": exposure,
        }
    ).to_csv(path, index=False)


def write_metrics(path: Path, total_return: float = 0.0) -> None:
    path.write_text(
        json.dumps(
            {
                "total_return": total_return,
                "max_drawdown": -0.10,
                "sharpe_like": 0.5,
            }
        ),
        encoding="utf-8",
    )


class StrategyArenaReportTest(unittest.TestCase):
    def test_equity_loader_rejects_future_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "equity.csv"
            write_curve(
                path,
                pd.to_datetime(["2026-07-14", "2026-07-16"]),
                [0.01, 0.01],
                0.5,
            )

            with self.assertRaisesRegex(ValueError, "after 2026-07-15"):
                load_equity_curve(path, asof_date="2026-07-15")

    def test_common_window_recomputation_finds_pareto_challenger_without_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.bdate_range("2025-01-02", periods=30)
            champion_curve = root / "champion.csv"
            challenger_curve = root / "challenger.csv"
            champion_metrics = root / "champion.json"
            challenger_metrics = root / "challenger.json"
            champion_returns = [0.0] + [0.001, -0.002] * 14 + [0.001]
            challenger_returns = [0.0] + [0.0015, -0.0005] * 14 + [0.0015]
            write_curve(champion_curve, dates, champion_returns, 0.8)
            write_curve(challenger_curve, dates, challenger_returns, 0.5)
            write_metrics(champion_metrics)
            write_metrics(challenger_metrics)
            entrants = (
                ArenaEntrant(
                    "champion",
                    "正式",
                    "core",
                    "核心联赛",
                    "next_open_v1",
                    "production_champion",
                    champion_curve,
                    champion_metrics,
                ),
                ArenaEntrant(
                    "challenger",
                    "挑战",
                    "core",
                    "核心联赛",
                    "next_open_v1",
                    "shadow_challenger",
                    challenger_curve,
                    challenger_metrics,
                ),
            )

            leaderboard, curves, common = build_portfolio_division(
                entrants,
                asof_date=dates[-1],
                min_common_days=20,
                tracking_summary={"status": "collecting", "valid_observation_count": 2, "target_days": 20},
            )
            challenger = leaderboard.set_index("entrant_id").loc["challenger"]

            self.assertEqual(len(common["core"]), 30)
            self.assertGreater(challenger["total_return"], leaderboard.iloc[0]["total_return"])
            self.assertTrue(
                bool(challenger["historical_pareto_dominates_league_reference"])
            )
            self.assertFalse(bool(challenger["promotion_eligible"]))
            pairwise = build_pairwise_comparison(leaderboard, curves, common)
            self.assertEqual(len(pairwise), 1)
            self.assertEqual(pairwise.loc[0, "common_return_days"], 29)

    def test_signal_observers_never_enter_portfolio_division(self):
        health = pd.DataFrame(
            [
                {
                    "strategy_family": "trend_momentum",
                    "strategy_family_cn": "趋势动量",
                    "signal_count": 100,
                    "completed_count": 40,
                    "selected_horizon_days": 1,
                    "avg_return": 0.02,
                    "win_rate": 0.7,
                    "worst_adverse_return": -0.1,
                    "family_health_status": "normal",
                    "family_health_reason": "正常观察",
                }
            ]
        )

        result = build_signal_observation_division(
            health,
            czsc_metadata={"status": "partial", "candidate_count": 200, "analyzed_count": 194},
            flow_metadata={"status": "warmup", "latest_source_rows": 5000},
        )

        self.assertEqual(len(result), 3)
        self.assertFalse(result["portfolio_comparable"].any())
        self.assertFalse(result["promotion_eligible"].any())
        self.assertEqual(
            result.set_index("observer_id").loc["family:trend_momentum", "completed_outcome_count"],
            40,
        )

    def test_history_update_is_idempotent_for_same_date_and_entrant(self):
        leaderboard = pd.DataFrame(
            [
                {
                    "entrant_id": "champion",
                    "entrant_name": "正式",
                    "league_id": "core",
                    "league_name": "核心联赛",
                    "contract_id": "next_open_v1",
                    "role": "production_champion",
                    "source_latest_date": "2026-07-15",
                    "total_return": 0.1,
                    "max_drawdown": -0.1,
                    "sharpe_like": 0.5,
                    "latest_daily_return": 0.01,
                    "average_turnover": 0.2,
                    "average_gross_exposure": 0.8,
                    "historical_pareto_dominates_league_reference": False,
                    "same_contract_comparable": True,
                    "arena_observation_days": 0,
                    "independent_observation_status": "production_reference",
                    "independent_observation_count": 0,
                    "independent_observation_target": 0,
                    "promotion_eligible": False,
                    "promotion_reason": "保持",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.csv"
            first, history1 = update_arena_history(
                history_path, leaderboard, asof_date="2026-07-15"
            )
            second, history2 = update_arena_history(
                history_path, leaderboard, asof_date="2026-07-15"
            )

            self.assertEqual(len(history1), 1)
            self.assertEqual(len(history2), 1)
            self.assertEqual(first.loc[0, "arena_observation_days"], 1)
            self.assertEqual(second.loc[0, "arena_observation_days"], 1)

    def test_end_to_end_writes_all_audit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.bdate_range("2025-01-02", periods=30)
            entrants = []
            for entrant_id, league_id, role, gain in (
                ("core_rank", "core", "production_champion", 0.0010),
                ("core_breadth_guard", "core", "shadow_challenger", 0.0011),
                ("pullback_baseline", "satellite", "league_reference", 0.0012),
                ("pullback_dynamic", "satellite", "shadow_challenger", 0.0014),
            ):
                equity = root / f"{entrant_id}.csv"
                metrics = root / f"{entrant_id}.json"
                write_curve(equity, dates, [0.0] + [gain] * 29, 0.6)
                write_metrics(metrics)
                entrants.append(
                    ArenaEntrant(
                        entrant_id,
                        entrant_id,
                        league_id,
                        "核心联赛" if league_id == "core" else "卫星联赛",
                        "next_open_v1" if league_id == "core" else "satellite_v1",
                        role,
                        equity,
                        metrics,
                    )
                )
            tracking = root / "tracking.json"
            tracking.write_text(
                json.dumps(
                    {
                        "status": "collecting",
                        "valid_observation_count": 2,
                        "target_days": 20,
                        "cumulative_return_delta": -0.005,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "outputs"

            paths, metadata = run_arena(
                entrants,
                asof_date=dates[-1].strftime("%Y-%m-%d"),
                output_dir=output,
                history_path=output / "strategy_arena_history.csv",
                tracking_summary_path=tracking,
                min_common_days=20,
                generated_at="2026-07-15T16:00:00",
            )

            for path in paths.values():
                self.assertTrue(path.exists(), path)
            self.assertEqual(metadata["production_champion"], "core_rank")
            self.assertFalse(metadata["automatic_promotion"])
            self.assertFalse(metadata["llm_trading_enabled"])
            chinese = pd.read_csv(paths["portfolio_cn"])
            self.assertIn("决策器名称", chinese.columns)
            report = paths["report"].read_text(encoding="utf-8")
            self.assertIn("| 29", report)
            self.assertNotIn("2900.00%", report)


if __name__ == "__main__":
    unittest.main()
