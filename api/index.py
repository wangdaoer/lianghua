"""Vercel entry point for the research workbench and its health API."""

from copy import deepcopy
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


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
RESEARCH_PATHS = {"/api/research/latest", "/api/research/latest/"}
MAX_RESEARCH_SNAPSHOT_BYTES = 1024 * 1024
FORBIDDEN_RESEARCH_KEYS = {
    "project_root",
    "commands",
    "argv",
    "artifacts",
    "personal_target_weight",
    "personal_action",
    "shadow_account_notes",
    "target_leverage",
}


class ResearchSnapshotMissing(Exception):
    """The configured private snapshot does not exist."""


class ResearchSnapshotUnavailable(Exception):
    """Private research storage has not been configured."""


def _fetch_research_snapshot() -> dict[str, Any]:
    url = os.environ.get("RESEARCH_BLOB_URL", "").strip()
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()
    if not url or not token:
        raise ResearchSnapshotUnavailable
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".private.blob.vercel-storage.com")
    ):
        raise ValueError("RESEARCH_BLOB_URL must reference a private Vercel Blob")

    request = Request(
        _consistent_blob_url(url),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = response.read(MAX_RESEARCH_SNAPSHOT_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 404:
            raise ResearchSnapshotMissing from exc
        raise RuntimeError("Private research storage returned an HTTP error") from exc
    except URLError as exc:
        raise RuntimeError("Private research storage could not be reached") from exc
    if len(payload) > MAX_RESEARCH_SNAPSHOT_BYTES:
        raise ValueError("Research snapshot exceeds the maximum allowed size")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Research snapshot must be a JSON object")
    return value


def _consistent_blob_url(url: str) -> str:
    """Bypass private Blob cache so an overwritten daily snapshot is current."""
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["cache"] = "0"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def _validate_research_snapshot(
    snapshot: dict[str, Any], *, today: date | None = None
) -> dict[str, Any]:
    value = deepcopy(snapshot)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported research snapshot schema")
    if value.get("project") != "lianghua":
        raise ValueError("Research snapshot project does not match")
    if value.get("research_only") is not True:
        raise ValueError("Research snapshot must remain research-only")
    if value.get("trade_instruction") is not False:
        raise ValueError("Research snapshot must not contain trade instructions")
    if _contains_forbidden_research_key(value):
        raise ValueError("Research snapshot contains a forbidden field")

    asof_value = value.get("asof_date")
    try:
        asof_date = date.fromisoformat(str(asof_value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Research snapshot has an invalid asof_date") from exc
    current_date = today or datetime.now(timezone.utc).date()
    age_days = (current_date - asof_date).days
    if age_days < 0:
        raise ValueError("Research snapshot asof_date is in the future")

    freshness = value.get("freshness")
    if not isinstance(freshness, dict):
        raise ValueError("Research snapshot is missing freshness metadata")
    stale_after_days = freshness.get("stale_after_days")
    if not isinstance(stale_after_days, int) or stale_after_days < 0:
        raise ValueError("Research snapshot has invalid freshness metadata")
    freshness["age_days"] = age_days
    freshness["status"] = "fresh" if age_days <= stale_after_days else "stale"
    if value.get("run_status") != "success":
        freshness["status"] = "failed"
    return value


def _contains_forbidden_research_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RESEARCH_KEYS:
                return True
            if _contains_forbidden_research_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_research_key(item) for item in value)
    return False


def _research_response(
    fetcher: Callable[[], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    fetcher = fetcher or _fetch_research_snapshot
    base = {
        "project": "lianghua",
        "research_only": True,
        "trade_instruction": False,
    }
    try:
        snapshot = _validate_research_snapshot(fetcher())
    except ResearchSnapshotMissing:
        return 404, {
            **base,
            "status": "missing",
            "message": "Research data is not published.",
        }
    except ResearchSnapshotUnavailable:
        return 503, {
            **base,
            "status": "unavailable",
            "message": "Research storage is not configured.",
        }
    except Exception:
        return 502, {
            **base,
            "status": "error",
            "message": "Research data could not be verified.",
        }

    freshness_status = snapshot["freshness"]["status"]
    if freshness_status == "stale":
        status = "stale"
    elif freshness_status == "failed":
        status = "failed"
    else:
        status = "ok"
    return 200, {**base, "status": status, "data": snapshot}


class handler(BaseHTTPRequestHandler):
    """Serve the homepage and dependency-free health responses on Vercel."""

    def _response_for_path(self) -> tuple[int, str, str, bytes]:
        path = urlsplit(self.path).path

        if path in HTML_PATHS:
            try:
                return (
                    200,
                    "text/html; charset=utf-8",
                    "public, max-age=0, must-revalidate",
                    INDEX_FILE.read_bytes(),
                )
            except OSError:
                body = json.dumps(
                    {"status": "error", "message": "Homepage is unavailable."}
                ).encode("utf-8")
                return 500, "application/json; charset=utf-8", "no-store", body

        if path in API_PATHS:
            body = json.dumps(RESPONSE, ensure_ascii=False).encode("utf-8")
            return (
                200,
                "application/json; charset=utf-8",
                "public, max-age=0, must-revalidate",
                body,
            )

        if path in RESEARCH_PATHS:
            status, payload = _research_response()
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return status, "application/json; charset=utf-8", "private, no-store", body

        body = json.dumps(
            {"status": "not_found", "message": "Route not found."}
        ).encode("utf-8")
        return 404, "application/json; charset=utf-8", "no-store", body

    def _send_response(self, *, include_body: bool) -> None:
        status, content_type, cache_control, body = self._response_for_path()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_response(include_body=True)

    def do_HEAD(self) -> None:
        self._send_response(include_body=False)

    def log_message(self, format: str, *args: object) -> None:
        return
