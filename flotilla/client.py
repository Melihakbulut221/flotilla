"""One client API for both modes.

``Flotilla()`` runs sandboxes in this process. ``Flotilla(base_url=...)``
talks to a ``flotilla serve`` instance over HTTP. Both expose::

    fl.swarms.start / run / get / list / cancel / wait / results
    fl.sessions.open(...) -> session with exec / write_file / read_file / close
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

from .backends import ExecResult
from .spec import SandboxInput, Size, SpecError, SwarmSpec


def _spec(command, count=None, inputs=None, name="", image=None, size=None,
          timeout_seconds=600, concurrency=None, result_path=None, network=False, env=None) -> SwarmSpec:
    return SwarmSpec(
        command=command, count=count,
        inputs=[SandboxInput.from_any(i) for i in inputs] if inputs is not None else None,
        name=name, image=image, size=Size.from_any(size), timeout_seconds=timeout_seconds,
        concurrency=concurrency, result_path=result_path, network=network, env=dict(env or {}),
    ).validate()


class ApiError(RuntimeError):
    def __init__(self, status: int, payload: Any):
        msg = payload.get("message") if isinstance(payload, dict) else payload
        super().__init__(f"HTTP {status}: {msg}")
        self.status, self.payload = status, payload


class _Http:
    def __init__(self, base_url: str, token: str | None, timeout: float, max_retries: int):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries

    def request(self, method: str, path: str, body: Any = None, headers: dict | None = None,
                raw: bool = False, timeout: float | None = None):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Content-Type": "application/json", **(headers or {})}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    payload = r.read()
                    return payload if raw else json.loads(payload or b"null")
            except urllib.error.HTTPError as e:
                text = e.read()
                try:
                    payload = json.loads(text)
                except ValueError:
                    payload = text.decode(errors="replace")
                if e.code >= 500 and attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 8)
                    continue
                raise ApiError(e.code, payload) from None
            except urllib.error.URLError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 8)


class RemoteSwarms:
    def __init__(self, http: _Http):
        self.http = http

    def start(self, command, idempotency_key: str | None = None, **kw) -> dict:
        spec = _spec(command, **kw)
        key = idempotency_key or uuid.uuid4().hex
        return self.http.request("POST", "/api/swarms", spec.to_dict(), {"Idempotency-Key": key})

    def get(self, run_id: str) -> dict:
        return self.http.request("GET", f"/api/swarms/{run_id}")

    def list(self, limit: int = 50) -> list[dict]:
        return self.http.request("GET", f"/api/swarms?limit={limit}")["swarms"]

    def cancel(self, run_id: str) -> dict:
        return self.http.request("POST", f"/api/swarms/{run_id}/cancel")

    def results(self, run_id: str) -> list[dict]:
        raw = self.http.request("GET", f"/api/swarms/{run_id}/results", raw=True)
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]

    def wait(self, run_id: str, interval: float = 1.0, timeout: float | None = None) -> dict:
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            run = self.get(run_id)
            if run["status"] != "running":
                return run
            if end is not None and time.monotonic() > end:
                raise TimeoutError(f"run {run_id} still running after {timeout}s")
            time.sleep(interval)

    def run(self, command, timeout: float | None = None, **kw) -> dict:
        return self.wait(self.start(command, **kw)["id"], timeout=timeout)


class LocalSwarms:
    def __init__(self, runner):
        self.runner = runner

    def start(self, command, idempotency_key: str | None = None, **kw) -> dict:
        return self.runner.start(_spec(command, **kw), idempotency_key=idempotency_key)[0]

    def get(self, run_id):
        return self.runner.get(run_id)

    def list(self, limit=50):
        return self.runner.list(limit)

    def cancel(self, run_id):
        return self.runner.cancel(run_id)

    def results(self, run_id):
        return self.runner.results(run_id)

    def wait(self, run_id, interval: float = 0, timeout: float | None = None):
        return self.runner.wait(run_id, timeout=timeout)

    def run(self, command, timeout: float | None = None, **kw) -> dict:
        return self.wait(self.start(command, **kw)["id"], timeout=timeout)


class RemoteSession:
    def __init__(self, http: _Http, info: dict, default_timeout: float):
        self.http = http
        self.id = info["id"]
        self.default_timeout = default_timeout
        self.closed = False

    def exec(self, command: str, timeout: float | None = None, env: dict | None = None,
             output_limit: int = 64 * 1024) -> ExecResult:
        t = timeout or self.default_timeout
        d = self.http.request("POST", f"/api/sessions/{self.id}/exec",
                              {"command": command, "timeoutSeconds": t, "env": env,
                               "outputLimit": output_limit}, timeout=t + 30)
        return ExecResult(
            exit_code=d["exitCode"], stdout=d["stdout"], stderr=d["stderr"],
            timed_out=d["timedOut"], cancelled=d["cancelled"],
            stdout_truncated=d["stdoutTruncated"], stderr_truncated=d["stderrTruncated"],
            duration=d["durationSeconds"],
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        body = {"path": path}
        if isinstance(content, bytes):
            body.update(content=base64.b64encode(content).decode(), encoding="base64")
        else:
            body.update(content=content)
        self.http.request("PUT", f"/api/sessions/{self.id}/files", body)

    def read_file(self, path: str, max_bytes: int | None = None) -> bytes | None:
        try:
            d = self.http.request("GET", f"/api/sessions/{self.id}/files?path={urllib.parse.quote(path)}")
        except ApiError as e:
            if e.status == 404:
                return None
            raise
        data = base64.b64decode(d["content"])
        return data if max_bytes is None else data[:max_bytes]

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.http.request("DELETE", f"/api/sessions/{self.id}")
            except ApiError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RemoteSessions:
    def __init__(self, http: _Http):
        self.http = http

    def open(self, *, image=None, size=None, network=False, env=None, files=None,
             timeout_seconds: float = 120) -> RemoteSession:
        body = {"image": image, "size": Size.from_any(size).to_dict(), "network": network,
                "env": env or {}, "files": files or {}, "timeoutSeconds": timeout_seconds}
        return RemoteSession(self.http, self.http.request("POST", "/api/sessions", body), timeout_seconds)

    def list(self) -> list[dict]:
        return self.http.request("GET", "/api/sessions")["sessions"]


class Flotilla:
    """Entry point.

    >>> fl = Flotilla()                                   # in-process, auto backend
    >>> fl = Flotilla(backend="local")                    # no isolation
    >>> fl = Flotilla(base_url="http://cpu-node:8642")    # remote server
    """

    def __init__(self, base_url: str | None = None, *, token: str | None = None,
                 backend: str = "auto", state_dir: str | None = None,
                 max_concurrency: int | None = None, timeout: float = 60, max_retries: int = 2,
                 **backend_kwargs):
        base_url = base_url or os.environ.get("FLOTILLA_URL")
        token = token or os.environ.get("FLOTILLA_TOKEN")
        self.remote = bool(base_url)
        if self.remote:
            http = _Http(base_url, token, timeout, max_retries)
            self.swarms = RemoteSwarms(http)
            self.sessions = RemoteSessions(http)
        else:
            from .session import SessionManager
            from .swarm import SwarmRunner

            runner = SwarmRunner(backend=backend, state_dir=state_dir,
                                 max_concurrency=max_concurrency, **backend_kwargs)
            self.runner = runner
            self.swarms = LocalSwarms(runner)
            self.sessions = SessionManager(backend=runner.backend, state_dir=runner.home)

    @property
    def isolated(self) -> bool:
        if self.remote:
            return True
        return self.runner.backend.isolated


__all__ = ["Flotilla", "ApiError", "SpecError"]
