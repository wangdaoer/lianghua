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
        self.assertNotIn("Lorem Ipsum", html)

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
