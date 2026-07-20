from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.broker_statement import (
    GF_COLUMNS,
    build_fifo_analysis,
    normalize_broker_statement,
    run_broker_statement_review,
)
from quant_etf_lab.cli import build_parser


def _sample_statement() -> pd.DataFrame:
    def row(
        *,
        trade_date: str,
        trade_time: str,
        sequence: str,
        symbol: str,
        name: str,
        business: str,
        quantity: str,
        price: str,
        commission: str,
        stamp_tax: str,
        transfer_fee: str,
        regulatory_fee: str,
        handling_fee: str,
        other_fee: str,
        settlement_amount: str,
        order_id: str,
    ) -> dict[str, str]:
        values = {
            "业务日期": trade_date,
            "发生时间": trade_time,
            "流水序号": sequence,
            "资金账号": "26848426",
            "证券代码": symbol,
            "证券名称": name,
            "业务标志名称": business,
            "成交数量": quantity,
            "成交价格": price,
            "净佣金": commission,
            "印花税": stamp_tax,
            "过户费": transfer_fee,
            "证管费": regulatory_fee,
            "经手费": handling_fee,
            "其他费": other_fee,
            "清算金额": settlement_amount,
            "货币名称": "人民币",
            "委托编号": order_id,
            "应计利息": "0.0000",
        }
        return {column: f"\t{values[column]}\t" for column in GF_COLUMNS}

    return pd.DataFrame(
        [
            row(
                trade_date="2026-04-15",
                trade_time="09:25:00",
                sequence="805951149",
                symbol="2645",
                name="华宏科技",
                business="证券买入",
                quantity="200.0000",
                price="21.3400",
                commission="4.7200",
                stamp_tax="0.0000",
                transfer_fee="0.0400",
                regulatory_fee="0.0900",
                handling_fee="0.1500",
                other_fee="0.0000",
                settlement_amount="-4273.0000",
                order_id="408",
            ),
            row(
                trade_date="2026-04-16",
                trade_time="09:31:00",
                sequence="805951150",
                symbol="002645",
                name="华宏科技",
                business="证券卖出",
                quantity="-100.0000",
                price="22.5500",
                commission="4.8500",
                stamp_tax="1.1300",
                transfer_fee="0.0200",
                regulatory_fee="0.0500",
                handling_fee="0.0800",
                other_fee="0.0000",
                settlement_amount="2248.8700",
                order_id="1737",
            ),
        ]
    )


class BrokerStatementTests(unittest.TestCase):
    def test_normalize_statement_strips_sensitive_account_fields(self) -> None:
        normalized = normalize_broker_statement(_sample_statement())

        self.assertEqual(list(normalized["symbol"]), ["002645", "002645"])
        self.assertEqual(list(normalized["side"]), ["buy", "sell"])
        self.assertIn("statement_row_id", normalized.columns)
        self.assertIn("total_fee", normalized.columns)
        self.assertNotIn("璧勯噾璐﹀彿", normalized.columns)
        self.assertNotIn("娴佹按搴忓彿", normalized.columns)
        self.assertNotIn("濮旀墭缂栧彿", normalized.columns)
        self.assertAlmostEqual(float(normalized.iloc[0]["total_fee"]), 5.0)

    def test_fifo_analysis_calculates_realized_pnl_and_open_cost(self) -> None:
        normalized = normalize_broker_statement(_sample_statement())
        realized, open_positions = build_fifo_analysis(normalized)

        self.assertEqual(len(realized), 1)
        self.assertAlmostEqual(float(realized.iloc[0]["matched_cost"]), 2136.5)
        self.assertAlmostEqual(float(realized.iloc[0]["net_proceeds"]), 2248.87)
        self.assertAlmostEqual(float(realized.iloc[0]["realized_pnl"]), 112.37)
        self.assertEqual(int(realized.iloc[0]["hold_days"]), 1)
        self.assertEqual(len(open_positions), 1)
        self.assertAlmostEqual(float(open_positions.iloc[0]["quantity"]), 100.0)
        self.assertAlmostEqual(float(open_positions.iloc[0]["cost_total"]), 2136.5)

    def test_run_broker_statement_review_writes_research_only_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "statement.csv"
            _sample_statement().to_csv(source, index=False, encoding="utf-8-sig")

            result = run_broker_statement_review(source, output_dir=root / "broker_review")

            self.assertTrue(result.normalized_path.exists())
            self.assertTrue(result.normalized_cn_path.exists())
            self.assertTrue(result.realized_path.exists())
            self.assertTrue(result.report_path.exists())
            snapshot = json.loads(result.snapshot_path.read_text(encoding="utf-8-sig"))
            self.assertEqual(snapshot["broker_action"], "none")
            self.assertEqual(snapshot["row_count"], 2)
            self.assertEqual(snapshot["symbol_count"], 1)
            self.assertAlmostEqual(float(snapshot["realized_pnl"]), 112.37)
            self.assertNotIn("璧勯噾璐﹀彿", result.normalized_path.read_text(encoding="utf-8-sig"))
            report = result.report_path.read_text(encoding="utf-8-sig")
            self.assertIn(result.normalized_path.name, report)
            self.assertIn(result.snapshot_path.name, report)
            self.assertEqual(snapshot["broker_action"], "none")

    def test_broker_statement_cli_parser(self) -> None:
        args = build_parser().parse_args(["broker-statement", "--input", "statement.csv"])

        self.assertEqual(args.command, "broker-statement")
        self.assertEqual(args.input, "statement.csv")
        self.assertEqual(args.output_dir, "outputs/research/broker_statement_latest")
