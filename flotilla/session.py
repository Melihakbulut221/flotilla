"""Long-lived sandboxes for multi-turn environments.

A swarm sandbox runs one command and dies. Agentic RL needs the opposite: a
sandbox that survives across turns so the policy can run a command, read the
output, edit a file and run again. A ``Session`` is that: open it, ``exec``
and move files as often as needed, then ``close`` it (sessions idle for
longer than ``idle_seconds`` are reaped).
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

from .backends import Backend, ExecResult, Sandbox, SandboxConfig, get_backend
from .spec import Size, SpecError, check_abs_path
from .swarm import NotFound, default_home

DEFAULT_OUTPUT_LIMIT = 64 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024


class Session:
    def __init__(self, session_id: str, sandbox: Sandbox, host_dir: Path, default_timeout: float):
        self.id = session_id
        self.sandbox = sandbox
        self.host_dir = host_dir
        self.default_timeout = default_timeout
        self.created_at = time.time()
        self.last_used = time.monotonic()
        self.lock = threading.Lock()
        self.closed = False

    def exec(self, command: str, timeout: float | None = None, env: dict[str, str] | None = None,
             output_limit: int = DEFAULT_OUTPUT_LIMIT) -> ExecResult:
        with self.lock:
            self._check()
            try:
                return self.sandbox.exec(command, timeout=timeout or self.default_timeout,
                                         env=env, output_limit=output_limit)
            finally:
                self.last_used = time.monotonic()

    def write_file(self, path: str, content: str | bytes) -> None:
        with self.lock:
            self._check()
            self.sandbox.write_file(check_abs_path(path, "path"), content)
            self.last_used = time.monotonic()

    def read_file(self, path: str, max_bytes: int = MAX_FILE_BYTES) -> bytes | None:
        with self.lock:
            self._check()
            got = self.sandbox.read_file(check_abs_path(path, "path"), max_bytes)
            self.last_used = time.monotonic()
            return None if got is None else got[0]

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                self.sandbox.close()
            finally:
                shutil.rmtree(self.host_dir, ignore_errors=True)

    def info(self) -> dict:
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "idleSeconds": round(time.monotonic() - self.last_used, 1),
            "closed": self.closed,
        }

    def _check(self):
        if self.closed:
            raise SpecError(f"session {self.id} is closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SessionManager:
    def __init__(self, backend: str | Backend = "auto", state_dir: str | Path | None = None,
                 idle_seconds: float = 1800, max_sessions: int = 256, **backend_kwargs):
        self.backend = get_backend(backend, **backend_kwargs)
        self.dir = (Path(state_dir) if state_dir else default_home()) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.idle_seconds = idle_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True, name="flotilla-reaper").start()

    def open(self, *, image: str | None = None, size: Size | dict | str | None = None,
             network: bool = False, env: dict[str, str] | None = None,
             files: dict[str, str] | None = None, timeout_seconds: float = 120) -> Session:
        with self._lock:
            live = sum(not s.closed for s in self._sessions.values())
            if live >= self.max_sessions:
                raise SpecError(f"session limit reached ({self.max_sessions})")
        sid = f"ses-{uuid.uuid4().hex[:16]}"
        host_dir = self.dir / sid
        cfg = SandboxConfig(size=Size.from_any(size), network=network, image=image,
                            env=dict(env or {}), label=sid)
        sandbox = self.backend.open(host_dir, cfg)
        sess = Session(sid, sandbox, host_dir, timeout_seconds)
        try:
            for p, c in (files or {}).items():
                sess.write_file(p, c)
        except Exception:
            sess.close()
            raise
        with self._lock:
            self._sessions[sid] = sess
        return sess

    def get(self, session_id: str) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None or s.closed:
            raise NotFound(session_id)
        return s

    def close(self, session_id: str) -> None:
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s is None:
            raise NotFound(session_id)
        s.close()

    def list(self) -> list[dict]:
        with self._lock:
            return [s.info() for s in self._sessions.values() if not s.closed]

    def close_all(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for s in sessions:
            s.close()

    def _reaper(self) -> None:
        while True:
            time.sleep(min(30.0, max(1.0, self.idle_seconds / 4)))
            now = time.monotonic()
            with self._lock:
                stale = [k for k, s in self._sessions.items()
                         if s.closed or now - s.last_used > self.idle_seconds]
                victims = [self._sessions.pop(k) for k in stale]
            for s in victims:
                if not s.lock.locked():
                    s.close()
                else:  # mid-exec; put it back for the next sweep
                    with self._lock:
                        self._sessions[s.id] = s
