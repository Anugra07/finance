from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ai_analyst.api.service import (
    analyst_brief_payload,
    analyst_context_payload,
    analyst_decision_payload,
    analyst_health_payload,
    analyst_research_payload,
)
from ai_analyst.config import Settings
from ai_analyst.utils.dates import parse_iso_datetime


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(rendered)))
    handler.end_headers()
    handler.wfile.write(rendered)


def build_handler(settings: Settings) -> type[BaseHTTPRequestHandler]:
    class AnalystHandler(BaseHTTPRequestHandler):
        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/analyst/v1/health":
                _json_response(self, HTTPStatus.OK, analyst_health_payload(settings))
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            body = self._body()
            as_of = parse_iso_datetime(str(body.get("as_of") or "2024-01-31T20:00:00Z"))
            ticker = str(body.get("ticker") or "SPY").upper()
            if self.path == "/api/analyst/v1/context-pack":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    analyst_context_payload(
                        settings,
                        ticker=ticker,
                        as_of=as_of,
                        mode=str(body.get("mode") or "research"),
                    ),
                )
                return
            if self.path == "/api/analyst/v1/research":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    analyst_research_payload(settings, ticker=ticker, as_of=as_of),
                )
                return
            if self.path == "/api/analyst/v1/forecast":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    analyst_decision_payload(settings, ticker=ticker, as_of=as_of),
                )
                return
            if self.path == "/api/analyst/v1/brief":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    analyst_brief_payload(settings, as_of=as_of),
                )
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return AnalystHandler


def serve(settings: Settings, *, host: str = "127.0.0.1", port: int = 8181) -> None:
    handler = build_handler(settings)
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
