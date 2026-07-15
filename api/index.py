"""Vercel entry point for the research workbench and its health API."""

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import urlsplit


RESPONSE = {
    "status": "ok",
    "project": "lianghua",
    "service": "deployment-health",
    "message": "The quantitative research repository is available.",
    "research_only": True,
    "trade_instruction": False,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_ROOT / "index.html"
HTML_PATHS = {"/", "/index.html"}
API_PATHS = {"/api", "/api/", "/health", "/health/"}


class handler(BaseHTTPRequestHandler):
    """Serve the homepage and dependency-free health responses on Vercel."""

    def _response_for_path(self) -> tuple[int, str, bytes]:
        path = urlsplit(self.path).path

        if path in HTML_PATHS:
            try:
                return 200, "text/html; charset=utf-8", INDEX_FILE.read_bytes()
            except OSError:
                body = json.dumps(
                    {"status": "error", "message": "Homepage is unavailable."}
                ).encode("utf-8")
                return 500, "application/json; charset=utf-8", body

        if path in API_PATHS:
            body = json.dumps(RESPONSE, ensure_ascii=False).encode("utf-8")
            return 200, "application/json; charset=utf-8", body

        body = json.dumps(
            {"status": "not_found", "message": "Route not found."}
        ).encode("utf-8")
        return 404, "application/json; charset=utf-8", body

    def _send_response(self, *, include_body: bool) -> None:
        status, content_type, body = self._response_for_path()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_response(include_body=True)

    def do_HEAD(self) -> None:
        self._send_response(include_body=False)

    def log_message(self, format: str, *args: object) -> None:
        return
