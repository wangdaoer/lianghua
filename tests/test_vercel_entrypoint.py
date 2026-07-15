import importlib.util
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import HTTPServer
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vercel_entrypoint", ROOT / "api" / "index.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class VercelEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), MODULE.handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_root_returns_research_workbench(self) -> None:
        with urlopen(f"{self.base_url}/", timeout=2) as response:
            html = response.read().decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get_content_type(), "text/html")
        self.assertIn("<title>lianghua · 量化研究工作台</title>", html)
        self.assertIn('id="overview"', html)

    def test_index_path_returns_research_workbench(self) -> None:
        with urlopen(f"{self.base_url}/index.html", timeout=2) as response:
            html = response.read().decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get_content_type(), "text/html")
        self.assertIn("<main", html)

    def test_api_returns_health_payload(self) -> None:
        with urlopen(f"{self.base_url}/api", timeout=2) as response:
            payload = json.load(response)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get_content_type(), "application/json")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["project"], "lianghua")
        self.assertTrue(payload["research_only"])
        self.assertFalse(payload["trade_instruction"])

    def test_health_returns_health_payload(self) -> None:
        with urlopen(f"{self.base_url}/health", timeout=2) as response:
            payload = json.load(response)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["status"], "ok")

    def _research_snapshot(self, asof_date: str | None = None) -> dict:
        asof_date = (
            asof_date
            or MODULE.datetime.now(MODULE.timezone.utc).date().isoformat()
        )
        return {
            "schema_version": 1,
            "project": "lianghua",
            "research_only": True,
            "trade_instruction": False,
            "asof_date": asof_date,
            "generated_at": "2026-07-14T22:00:13",
            "published_at": "2026-07-15T05:00:00Z",
            "run_status": "success",
            "freshness": {
                "status": "fresh",
                "age_days": 1,
                "stale_after_days": 3,
            },
            "summary": {"priority_rows": 50},
            "coverage": {"database_latest_date": "2026-07-14"},
            "quality": {"warnings": []},
            "watchlist": {"top10": []},
            "source_integrity": {
                "run_card_sha256": "a" * 64,
                "watchlist_sha256": "b" * 64,
            },
        }

    def test_research_api_returns_verified_snapshot(self) -> None:
        snapshot = self._research_snapshot()
        with patch.object(MODULE, "_fetch_research_snapshot", return_value=snapshot):
            with urlopen(f"{self.base_url}/api/research/latest", timeout=2) as response:
                payload = json.load(response)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["data"]["summary"]["priority_rows"], 50)
        self.assertEqual(response.headers["Cache-Control"], "private, no-store")

    def test_research_api_reports_missing_snapshot(self) -> None:
        with patch.object(
            MODULE,
            "_fetch_research_snapshot",
            side_effect=MODULE.ResearchSnapshotMissing,
        ):
            with self.assertRaises(HTTPError) as context:
                urlopen(f"{self.base_url}/api/research/latest", timeout=2)

        self.assertEqual(context.exception.code, 404)
        payload = json.load(context.exception)
        self.assertEqual(payload["status"], "missing")

    def test_research_api_recomputes_stale_state(self) -> None:
        snapshot = self._research_snapshot(asof_date="2020-01-01")
        with patch.object(MODULE, "_fetch_research_snapshot", return_value=snapshot):
            with urlopen(f"{self.base_url}/api/research/latest", timeout=2) as response:
                payload = json.load(response)

        self.assertEqual(payload["status"], "stale")
        self.assertEqual(payload["data"]["freshness"]["status"], "stale")

    def test_research_api_hides_invalid_snapshot_error(self) -> None:
        snapshot = self._research_snapshot()
        snapshot["personal_target_weight"] = 0.75
        with patch.object(MODULE, "_fetch_research_snapshot", return_value=snapshot):
            with self.assertRaises(HTTPError) as context:
                urlopen(f"{self.base_url}/api/research/latest", timeout=2)

        self.assertEqual(context.exception.code, 502)
        payload = json.load(context.exception)
        self.assertEqual(payload["status"], "error")
        self.assertNotIn("personal_target_weight", json.dumps(payload))

    def test_private_blob_read_bypasses_cached_overwrite(self) -> None:
        url = MODULE._consistent_blob_url(
            "https://store.private.blob.vercel-storage.com/research/latest.json?x=1"
        )

        self.assertIn("x=1", url)
        self.assertIn("cache=0", url)

    def test_unknown_path_returns_not_found(self) -> None:
        with self.assertRaises(HTTPError) as context:
            urlopen(f"{self.base_url}/missing", timeout=2)

        self.assertEqual(context.exception.code, 404)

    def test_head_has_no_response_body(self) -> None:
        request = Request(f"{self.base_url}/api", method="HEAD")
        with urlopen(request, timeout=2) as response:
            body = response.read()

        self.assertEqual(response.status, 200)
        self.assertEqual(body, b"")
        self.assertGreater(int(response.headers["Content-Length"]), 0)


if __name__ == "__main__":
    unittest.main()
