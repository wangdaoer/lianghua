import importlib.util
import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import HTTPServer


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
