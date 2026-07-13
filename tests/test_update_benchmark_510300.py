import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from update_benchmark_510300 import (
    BenchmarkRow,
    parse_sina_quote,
    parse_sohu_history,
    parse_yahoo_chart,
    refresh_benchmark,
)


SINA_QUOTE = (
    'var hq_str_sh510300="沪深300ETF华泰柏瑞,4.802,4.829,4.744,4.821,4.719,'
    '4.744,4.745,1619052968,7705111563.000,578400,4.744,4848400,4.743,'
    '1680300,4.742,216300,4.741,1973400,4.740,208500,4.745,79800,4.746,'
    '17500,4.747,48800,4.748,57100,4.749,2026-07-13,15:34:59,00,'
    'D|304600|1445022.40";'
)

SOHU_HISTORY = (
    'historySearchHandler([{"status":0,"hq":['
    '["2026-07-13","4.802","4.744","-0.085","-1.76%","4.719",'
    '"4.821","16190529","770511.125","3.61%","3046.000"],'
    '["2026-07-10","4.918","4.829","-0.087","-1.77%","4.827",'
    '"4.949","8242800","403950.719","1.84%","4021.000"]],'
    '"code":"cn_510300"}])'
)

YAHOO_CHART = {
    "chart": {
        "result": [
            {
                "timestamp": [1783647000, 1783906200],
                "indicators": {
                    "quote": [
                        {
                            "open": [4.918000221252441, 4.802000045776367],
                            "high": [4.948999881744385, 4.821000099182129],
                            "low": [4.827000141143799, 4.718999862670898],
                            "close": [4.828999996185303, 4.74399995803833],
                            "volume": [824280021, 1619052968],
                        }
                    ]
                },
            }
        ],
        "error": None,
    }
}


class UpdateBenchmark510300Test(unittest.TestCase):
    def test_parses_three_source_formats(self):
        sina = parse_sina_quote(SINA_QUOTE, "510300")
        sohu = parse_sohu_history(SOHU_HISTORY)
        yahoo = parse_yahoo_chart(YAHOO_CHART)

        self.assertEqual(sina.date, date(2026, 7, 13))
        self.assertEqual(sina.close, 4.744)
        self.assertEqual(sina.amount, 7_705_111_563)
        self.assertEqual(sohu[date(2026, 7, 13)].volume, 1_619_052_900)
        self.assertEqual(sohu[date(2026, 7, 13)].amount, 7_705_111_250)
        self.assertEqual(yahoo[date(2026, 7, 13)].volume, 1_619_052_968)

    def test_rejects_intraday_sina_snapshot_for_current_asof(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = Path(tmp) / "510300.csv"
            self._write_base(benchmark)
            intraday = parse_sina_quote(
                SINA_QUOTE.replace("15:34:59", "14:59:59"),
                "510300",
            )
            _, sohu, yahoo = self._fetchers()

            with self.assertRaisesRegex(ValueError, "before market close"):
                refresh_benchmark(
                    benchmark,
                    date(2026, 7, 13),
                    fetch_sina=lambda symbol: intraday,
                    fetch_sohu=sohu,
                    fetch_yahoo=yahoo,
                )

    def test_prior_day_catchup_ignores_next_day_intraday_sina_quote(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = Path(tmp) / "510300.csv"
            self._write_base(benchmark)
            next_day_intraday = parse_sina_quote(
                SINA_QUOTE.replace("2026-07-13,15:34:59", "2026-07-14,14:59:59"),
                "510300",
            )
            _, sohu, yahoo = self._fetchers()

            result = refresh_benchmark(
                benchmark,
                date(2026, 7, 13),
                fetch_sina=lambda symbol: next_day_intraday,
                fetch_sohu=sohu,
                fetch_yahoo=yahoo,
            )

            self.assertEqual(result["status"], "updated")
            self.assertEqual(result["sources"], ["Sohu", "Yahoo"])
            self.assertEqual(pd.read_csv(benchmark).iloc[-1]["date"], "2026-07-13")

    def test_refresh_appends_verified_row_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "510300.csv"
            status_path = root / "status.json"
            self._write_base(benchmark)
            sina, sohu, yahoo = self._fetchers()

            result = refresh_benchmark(
                benchmark,
                date(2026, 7, 13),
                status_path=status_path,
                fetch_sina=sina,
                fetch_sohu=sohu,
                fetch_yahoo=yahoo,
            )

            frame = pd.read_csv(benchmark)
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "updated")
            self.assertEqual(result["rows_added"], 1)
            self.assertEqual(frame["date"].tolist(), ["2026-07-10", "2026-07-13"])
            self.assertEqual(int(frame.iloc[-1]["volume"]), 1_619_052_968)
            self.assertEqual(int(frame.iloc[-1]["amount"]), 7_705_111_563)
            self.assertTrue(status["source_agreement"])

    def test_source_disagreement_leaves_original_file_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = Path(tmp) / "510300.csv"
            self._write_base(benchmark)
            original = benchmark.read_bytes()
            sina, sohu, yahoo = self._fetchers()

            def bad_yahoo(symbol: str, start: date, end: date):
                rows = yahoo(symbol, start, end)
                row = rows[date(2026, 7, 13)]
                rows[date(2026, 7, 13)] = BenchmarkRow(
                    row.date, row.open, row.high, row.low, 4.900, row.volume, row.amount
                )
                return rows

            with self.assertRaisesRegex(ValueError, "source disagreement"):
                refresh_benchmark(
                    benchmark,
                    date(2026, 7, 13),
                    fetch_sina=sina,
                    fetch_sohu=sohu,
                    fetch_yahoo=bad_yahoo,
                )

            self.assertEqual(benchmark.read_bytes(), original)

    def test_status_commit_failure_rolls_back_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "510300.csv"
            invalid_status_path = root / "status-directory"
            invalid_status_path.mkdir()
            self._write_base(benchmark)
            original = benchmark.read_bytes()
            sina, sohu, yahoo = self._fetchers()

            with self.assertRaises(OSError):
                refresh_benchmark(
                    benchmark,
                    date(2026, 7, 13),
                    status_path=invalid_status_path,
                    fetch_sina=sina,
                    fetch_sohu=sohu,
                    fetch_yahoo=yahoo,
                )

            self.assertEqual(benchmark.read_bytes(), original)

    def test_already_fresh_file_is_verified_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = Path(tmp) / "510300.csv"
            self._write_base(benchmark)
            with benchmark.open("a", encoding="utf-8", newline="") as handle:
                handle.write("2026-07-13,4.802,4.821,4.719,4.744,1619052968,7705111563\n")
            sina, sohu, yahoo = self._fetchers()

            result = refresh_benchmark(
                benchmark,
                date(2026, 7, 13),
                fetch_sina=sina,
                fetch_sohu=sohu,
                fetch_yahoo=yahoo,
            )

            frame = pd.read_csv(benchmark)
            self.assertEqual(result["status"], "already_fresh")
            self.assertEqual(result["rows_added"], 0)
            self.assertEqual(len(frame), 2)

    @staticmethod
    def _write_base(path: Path) -> None:
        path.write_text(
            "date,open,high,low,close,volume,amount\n"
            "2026-07-10,4.918,4.949,4.827,4.829,824280021,4039507185\n",
            encoding="utf-8",
        )

    @staticmethod
    def _fetchers():
        sina_row = parse_sina_quote(SINA_QUOTE, "510300")
        sohu_rows = parse_sohu_history(SOHU_HISTORY)
        yahoo_rows = parse_yahoo_chart(YAHOO_CHART)

        def sina(symbol: str):
            return sina_row

        def sohu(symbol: str, start: date, end: date):
            return dict(sohu_rows)

        def yahoo(symbol: str, start: date, end: date):
            return dict(yahoo_rows)

        return sina, sohu, yahoo


if __name__ == "__main__":
    unittest.main()
