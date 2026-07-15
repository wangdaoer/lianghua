"""Minimal Vercel entry point for the research repository."""

from http.server import BaseHTTPRequestHandler
import json


RESPONSE = {
    "status": "ok",
    "project": "lianghua",
    "service": "deployment-health",
    "message": "The quantitative research repository is available.",
    "research_only": True,
    "trade_instruction": False,
}


class handler(BaseHTTPRequestHandler):
    """Serve a dependency-free health response on Vercel."""

    def _send_headers(self, body_length: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(body_length))
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.end_headers()

    def do_GET(self) -> None:
        body = json.dumps(RESPONSE, ensure_ascii=False).encode("utf-8")
        self._send_headers(len(body))
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        body = json.dumps(RESPONSE, ensure_ascii=False).encode("utf-8")
        self._send_headers(len(body))

    def log_message(self, format: str, *args: object) -> None:
        return
