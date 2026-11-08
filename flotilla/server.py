"""HTTP API (stdlib only).

Run sandboxes on one machine (a CPU box, an HPC node) and drive them from
another (the GPU trainer). Routes::

    GET    /health
    POST   /api/swarms                    start (Idempotency-Key header supported)
    GET    /api/swarms?limit=50           list, newest first
    GET    /api/swarms/{id}               run + every sandbox
    POST   /api/swarms/{id}/cancel
    GET    /api/swarms/{id}/results       JSON Lines, input order
    POST   /api/sessions                  open a long-lived sandbox
    GET    /api/sessions
    POST   /api/sessions/{id}/exec        {"command", "timeoutSeconds", "env"}
    PUT    /api/sessions/{id}/files       {"path", "content", "encoding": "utf-8"|"base64"}
    GET    /api/sessions/{id}/files?path=/work/x
    DELETE /api/sessions/{id}

Auth: when a token is configured every /api route needs
``Authorization: Bearer <token>``. Binds to 127.0.0.1 by default.
"""

from __future__ import annotations

import base64
import hmac
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__
from .session import SessionManager
from .spec import SpecError
from .swarm import IdempotencyConflict, NotFound, SwarmRunner

MAX_BODY = 64 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str = ""):
        super().__init__(message or code)
        self.status, self.code, self.message = status, code, message or code


def make_handler(runner: SwarmRunner, sessions: SessionManager, token: str | None):
    routes = []

    def route(method: str, pattern: str):
        rx = re.compile("^" + pattern + "$")

        def deco(fn):
            routes.append((method, rx, fn))
            return fn

        return deco

    @route("GET", r"/health")
    def health(h, q, body):
        return 200, {"ok": True, "version": __version__, "backend": runner.backend.name,
                     "isolated": runner.backend.isolated, "maxConcurrency": runner.max_concurrency}

    @route("POST", r"/api/swarms")
    def start(h, q, body):
        key = h.headers.get("Idempotency-Key") or (body or {}).pop("idempotencyKey", None)
        run, created = runner.start(body or {}, idempotency_key=key)
        return (201 if created else 200), run

    @route("GET", r"/api/swarms")
    def list_runs(h, q, body):
        return 200, {"swarms": runner.list(int(q.get("limit", ["50"])[0]))}

    @route("GET", r"/api/swarms/(?P<id>[\w-]+)")
    def get_run(h, q, body, id):
        return 200, runner.get(id)

    @route("POST", r"/api/swarms/(?P<id>[\w-]+)/cancel")
    def cancel(h, q, body, id):
        return 200, runner.cancel(id)

    @route("GET", r"/api/swarms/(?P<id>[\w-]+)/results")
    def results(h, q, body, id):
        lines = "".join(json.dumps(r) + "\n" for r in runner.results(id))
        return 200, ("application/x-ndjson", lines.encode())

    @route("POST", r"/api/sessions")
    def open_session(h, q, body):
        b = body or {}
        s = sessions.open(
            image=b.get("image") or b.get("template"),
            size=b.get("size"),
            network=bool(b.get("network", False)),
            env=b.get("env"),
            files=b.get("files"),
            timeout_seconds=float(b.get("timeoutSeconds", 120)),
        )
        return 201, s.info()

    @route("GET", r"/api/sessions")
    def list_sessions(h, q, body):
        return 200, {"sessions": sessions.list()}

    @route("POST", r"/api/sessions/(?P<id>[\w-]+)/exec")
    def exec_(h, q, body, id):
        b = body or {}
        if not isinstance(b.get("command"), str):
            raise SpecError("command is required")
        res = sessions.get(id).exec(
            b["command"],
            timeout=b.get("timeoutSeconds"),
            env=b.get("env"),
            output_limit=int(b.get("outputLimit", 64 * 1024)),
        )
        return 200, res.to_dict()

    @route("PUT", r"/api/sessions/(?P<id>[\w-]+)/files")
    def put_file(h, q, body, id):
        b = body or {}
        content = b.get("content", "")
        if b.get("encoding") == "base64":
            content = base64.b64decode(content)
        sessions.get(id).write_file(b.get("path", ""), content)
        return 200, {"ok": True}

    @route("GET", r"/api/sessions/(?P<id>[\w-]+)/files")
    def get_file(h, q, body, id):
        path = q.get("path", [""])[0]
        data = sessions.get(id).read_file(path)
        if data is None:
            raise ApiError(404, "file_not_found", path)
        return 200, {"path": path, "encoding": "base64", "content": base64.b64encode(data).decode()}

    @route("DELETE", r"/api/sessions/(?P<id>[\w-]+)")
    def close_session(h, q, body, id):
        sessions.close(id)
        return 200, {"ok": True}

    class Handler(BaseHTTPRequestHandler):
        server_version = f"flotilla/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter default logging
            if getattr(self.server, "verbose", False):
                super().log_message(fmt, *args)

        def _send(self, status: int, payload):
            if isinstance(payload, tuple):
                ctype, data = payload
            else:
                ctype, data = "application/json", json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _dispatch(self, method: str):
            url = urlparse(self.path)
            try:
                if url.path.startswith("/api/") and token:
                    got = self.headers.get("Authorization", "")
                    if not hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
                        raise ApiError(401, "unauthorized", "missing or bad bearer token")
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise ApiError(413, "body_too_large")
                body = None
                if length:
                    try:
                        body = json.loads(self.rfile.read(length))
                    except ValueError as e:
                        raise ApiError(422, "invalid_json", str(e)) from e
                q = parse_qs(url.query)
                for m, rx, fn in routes:
                    if m != method:
                        continue
                    match = rx.match(url.path)
                    if match:
                        status, payload = fn(self, q, body, **match.groupdict())
                        return self._send(status, payload)
                raise ApiError(404, "not_found", url.path)
            except ApiError as e:
                self._send(e.status, {"error": e.code, "message": e.message})
            except IdempotencyConflict as e:
                self._send(409, {"error": "idempotency_conflict", "message": str(e)})
            except NotFound as e:
                self._send(404, {"error": "not_found", "message": f"unknown id {e}"})
            except (SpecError, ValueError, TypeError) as e:
                self._send(422, {"error": "invalid_request", "message": str(e)})
            except Exception as e:  # pragma: no cover - defensive
                self._send(500, {"error": "internal", "message": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_PUT(self):
            self._dispatch("PUT")

        def do_DELETE(self):
            self._dispatch("DELETE")

    return Handler


def make_server(runner: SwarmRunner, sessions: SessionManager, host: str = "127.0.0.1",
                port: int = 8642, token: str | None = None, verbose: bool = False) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(runner, sessions, token))
    srv.daemon_threads = True
    srv.verbose = verbose
    return srv
