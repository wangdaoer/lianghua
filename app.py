"""Minimal read-only HTTP entrypoint for Vercel deployments."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any


def build_response(path: str) -> tuple[int, dict[str, Any]]:
    """Build a small deployment status response without touching local data."""
    if path not in {"/", "/health"}:
        return 404, {
            "status": "not_found",
            "service": "quant-etf-lab",
        }

    return 200, {
        "status": "ok",
        "service": "quant-etf-lab",
        "mode": "research_only",
        "broker_action": "none",
        "message": "The full research pipeline runs in the local data workspace.",
    }


class handler(BaseHTTPRequestHandler):
    """Vercel Python serverless handler."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        status, payload = build_response(self.path.split("?", 1)[0])
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
