import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VercelFrontendTests(unittest.TestCase):
    def test_homepage_contains_required_dashboard_sections(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn("<main", html)
        self.assertIn('id="overview"', html)
        self.assertIn('id="capabilities"', html)
        self.assertIn('id="boundaries"', html)
        self.assertIn('fetch("/api"', html)
        self.assertIn('fetch("/api/research/latest"', html)
        self.assertIn('id="research-panel"', html)
        self.assertIn('id="research-badge"', html)
        self.assertIn('id="database-badge"', html)
        self.assertIn('id="market-badge"', html)
        self.assertIn('id="connection-badge"', html)
        self.assertNotIn("Lorem Ipsum", html)

    def test_research_ui_preserves_read_only_boundary_and_all_api_states(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn('snapshot.research_only !== true', html)
        self.assertIn('snapshot.trade_instruction !== false', html)
        self.assertIn('["ok", "stale", "failed"]', html)
        for state in ("missing", "unavailable", "error"):
            self.assertIn(f'{state}: [', html)
        self.assertIn("页面不产生券商订单", html)
        self.assertNotIn("personal_target_weight", html)
        self.assertNotIn("personal_action", html)

    def test_root_is_static_and_health_rewrites_to_api(self) -> None:
        config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

        rewrites = config["rewrites"]
        self.assertIn({"source": "/", "destination": "/index.html"}, rewrites)
        self.assertNotIn({"source": "/", "destination": "/api"}, rewrites)
        self.assertIn({"source": "/health", "destination": "/api"}, rewrites)

    def test_vercel_bundle_includes_homepage(self) -> None:
        ignore_rules = (ROOT / ".vercelignore").read_text(encoding="utf-8")
        config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

        self.assertIn("!api", ignore_rules.splitlines())
        self.assertIn("!index.html", ignore_rules.splitlines())
        self.assertIn("!vercel.json", ignore_rules.splitlines())
        self.assertEqual(
            config["functions"]["api/index.py"]["includeFiles"], "index.html"
        )


if __name__ == "__main__":
    unittest.main()
