"""Swarm runner: one command, many fresh sandboxes, collected results.

Lifecycle of each sandbox::

    queued -> booting -> running -> succeeded | failed | cancelled

A run is ``running`` until every sandbox has ended, then ``completed`` (or
``cancelled`` if it was cancelled). Every sandbox's exit code, the last 16 KB
of stdout/stderr and the optional result file (up to 64 KB) are kept; the
sandbox itself, disk included, is destroyed as soon as its command ends.

Runs are persisted under ``$FLOTILLA_HOME/runs/<id>/`` (``run.json`` while
running, plus ``results.jsonl`` in input order when finished).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends import Backend, SandboxConfig, get_backend
from .spec import RESULT_MAX_BYTES, TAIL_BYTES, SpecError, SwarmSpec

TERMINAL = ("succeeded", "failed", "cancelled")
STATES = ("queued", "booting", "running") + TERMINAL


class IdempotencyConflict(SpecError):
    pass


class NotFound(KeyError):
    pass


def default_home() -> Path:
    return Path(os.environ.get("FLOTILLA_HOME", Path.home() / ".flotilla"))


def _now() -> float:
    return time.time()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class _CpuPool:
    """Hands out disjoint host CPU sets so sandboxes do not fight over cores."""

    def __init__(self):
        try:
            cpus = sorted(os.sched_getaffinity(0))
        except AttributeError:  # non-Linux
            cpus = []
        self.free = list(cpus)
        self.lock = threading.Lock()

    def take(self, n: int) -> list[int] | None:
        with self.lock:
            if len(self.free) < n:
                return None
            got, self.free = self.free[:n], self.free[n:]
            return got

    def give(self, cpus: list[int] | None) -> None:
        if cpus:
            with self.lock:
                self.free.extend(cpus)


class _Run:
    def __init__(self, run_id: str, spec: SwarmSpec, concurrency: int, backend: str):
        self.id = run_id
        self.spec = spec
        self.concurrency = concurrency
        self.backend = backend
        self.created_at = _now()
        self.finished_at: float | None = None
        self.status = "running"
        self.error: str | None = None
        self.waiting_reason: str | None = None
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.persist_lock = threading.Lock()
        self.last_persist = 0.0
        short = run_id.split("-", 1)[-1][:8]
        self.sandboxes = [
            {
                "index": i,
                "sandboxId": f"sbx-{short}-{i}",
                "status": "queued",
                "exitCode": None,
                "stdoutTail": "",
                "stderrTail": "",
                "result": None,
                "resultTruncated": False,
                "error": None,
                "startedAt": None,
                "finishedAt": None,
                "durationSeconds": None,
            }
            for i in range(spec.total)
        ]

    def counts(self) -> dict:
        c = {s: 0 for s in STATES}
        for sb in self.sandboxes:
            c[sb["status"]] += 1
        return c

    def summary(self) -> dict:
        counts = self.counts()
        end = self.finished_at or _now()
        busy = sum(sb["durationSeconds"] or 0 for sb in self.sandboxes)
        return {
            "id": self.id,
            "name": self.spec.name,
            "status": self.status,
            "backend": self.backend,
            "count": self.spec.total,
            "size": self.spec.size.to_dict(),
            "concurrency": self.concurrency,
            "timeoutSeconds": self.spec.timeout_seconds,
            "counts": counts,
            "done": sum(counts[s] for s in TERMINAL),
            "waitingReason": self.waiting_reason,
            "error": self.error,
            "createdAt": self.created_at,
            "finishedAt": self.finished_at,
            "durationSeconds": round(end - self.created_at, 3),
            "sandboxSeconds": round(busy, 3),
        }

    def full(self) -> dict:
        d = self.summary()
        d["sandboxes"] = [dict(sb) for sb in self.sandboxes]
        return d


class SwarmRunner:
    """In-process swarm engine. Thread-safe; one instance per host is enough.

    ``max_concurrency`` is the global cap on sandboxes alive at once across
    all runs (the "awake limit"); a run's own ``concurrency`` is clamped to it.
    """

    def __init__(
        self,
        backend: str | Backend = "auto",
        state_dir: str | Path | None = None,
        max_concurrency: int | None = None,
        keep_workdirs: bool = False,
        pin_cpus: bool = True,
        **backend_kwargs,
    ):
        self.backend = get_backend(backend, **backend_kwargs)
        self.home = Path(state_dir) if state_dir else default_home()
        self.runs_dir = self.home / "runs"
        self.work_dir = self.home / "work"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        cpu = os.cpu_count() or 2
        self.max_concurrency = max_concurrency or max(1, cpu // 2)
        self.keep_workdirs = keep_workdirs
        self.pin_cpus = pin_cpus and self.backend.name in ("local", "bwrap", "apptainer")
        self._slots = threading.BoundedSemaphore(self.max_concurrency)
        self._cpus = _CpuPool()
        self._runs: dict[str, _Run] = {}
        self._lock = threading.RLock()
        self._idem_path = self.home / "idempotency.json"
        self._idem: dict[str, list[str]] = {}
        self._load()

    # ---------------------------------------------------------------- public

    def start(self, spec: SwarmSpec | dict, idempotency_key: str | None = None) -> tuple[dict, bool]:
        """Start a run in the background. Returns (run, created)."""
        if isinstance(spec, dict):
            spec = SwarmSpec.from_dict(spec)
        else:
            spec.validate()
        digest = hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest()
        with self._lock:
            if idempotency_key:
                prev = self._idem.get(idempotency_key)
                if prev:
                    if prev[0] != digest:
                        raise IdempotencyConflict("idempotency key reused with a different body")
                    if prev[1] in self._runs:
                        return self._runs[prev[1]].summary(), False
            run_id = f"swarm-{uuid.uuid4().hex[:16]}"
            conc = min(spec.concurrency or self.max_concurrency, self.max_concurrency, spec.total)
            run = _Run(run_id, spec, conc, self.backend.name)
            self._runs[run_id] = run
            if idempotency_key:
                self._idem[idempotency_key] = [digest, run_id]
                _atomic_write(self._idem_path, json.dumps(self._idem))
        (self.runs_dir / run_id).mkdir(parents=True, exist_ok=True)
        (self.runs_dir / run_id / "spec.json").write_text(json.dumps(spec.to_dict()))
        self._persist(run)
        threading.Thread(target=self._execute, args=(run,), name=run_id, daemon=True).start()
        return run.summary(), True

    def run(self, spec: SwarmSpec | dict, timeout: float | None = None) -> dict:
        """Start a run and block until it finishes."""
        summary, _ = self.start(spec)
        return self.wait(summary["id"], timeout=timeout)

    def get(self, run_id: str) -> dict:
        return self._run(run_id).full()

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return [r.summary() for r in runs[:limit]]

    def cancel(self, run_id: str) -> dict:
        run = self._run(run_id)
        run.cancel.set()
        return run.summary()

    def wait(self, run_id: str, timeout: float | None = None) -> dict:
        run = self._run(run_id)
        if not run.done.wait(timeout):
            raise TimeoutError(f"run {run_id} still running after {timeout}s")
        return run.full()

    def results(self, run_id: str) -> list[dict]:
        run = self._run(run_id)
        return [dict(sb) for sb in run.sandboxes]

    # -------------------------------------------------------------- internals

    def _run(self, run_id: str) -> _Run:
        with self._lock:
            if run_id not in self._runs:
                raise NotFound(run_id)
            return self._runs[run_id]

    def _execute(self, run: _Run) -> None:
        try:
            with ThreadPoolExecutor(max_workers=run.concurrency, thread_name_prefix=run.id) as pool:
                list(pool.map(lambda i: self._run_one(run, i), range(run.spec.total)))
        except Exception as e:  # pragma: no cover - defensive
            run.error = f"runner error: {e}"
        run.waiting_reason = None
        run.finished_at = _now()
        run.status = "cancelled" if run.cancel.is_set() else "completed"
        self._persist(run, final=True)
        if not self.keep_workdirs:
            shutil.rmtree(self.work_dir / run.id, ignore_errors=True)
        run.done.set()

    def _acquire_slot(self, run: _Run) -> bool:
        waited = False
        while not run.cancel.is_set():
            if self._slots.acquire(timeout=0.25):
                return True
            if not waited:
                waited = True
                run.waiting_reason = "awake_limit"
        return False

    def _set(self, run: _Run, i: int, **fields) -> None:
        with run.lock:
            run.sandboxes[i].update(fields)

    def _run_one(self, run: _Run, i: int) -> None:
        spec = run.spec
        if run.cancel.is_set() or not self._acquire_slot(run):
            self._set(run, i, status="cancelled", finishedAt=_now())
            return
        run.waiting_reason = None
        cpus = self._cpus.take(spec.size.vcpus) if self.pin_cpus else None
        host_dir = self.work_dir / run.id / f"{i:06d}"
        sandbox = None
        t0 = _now()
        self._set(run, i, status="booting", startedAt=t0)
        try:
            inp = spec.input_for(i)
            env = {**spec.env, **inp.env}
            for prefix in ("FLOTILLA", "MARITIME"):
                env[f"{prefix}_SWARM_INDEX"] = str(i)
                env[f"{prefix}_SWARM_COUNT"] = str(spec.total)
                env[f"{prefix}_SWARM_ID"] = run.id
            cfg = SandboxConfig(
                size=spec.size, network=spec.network, image=spec.image,
                env=env, cpus=cpus, label=f"{run.id}/{i}",
            )
            sandbox = self.backend.open(host_dir, cfg)
            for path, content in inp.files.items():
                sandbox.write_file(path, content)
            if spec.result_path:
                sandbox.ensure_parent(spec.result_path)
            if run.cancel.is_set():
                self._set(run, i, status="cancelled", finishedAt=_now())
                return
            self._set(run, i, status="running")
            res = sandbox.exec(
                spec.command_for(i), timeout=spec.timeout_seconds,
                cancel=run.cancel, output_limit=TAIL_BYTES,
            )
            fields = {"stdoutTail": res.stdout, "stderrTail": res.stderr, "exitCode": res.exit_code}
            if spec.result_path and not res.cancelled:
                got = sandbox.read_file(spec.result_path, RESULT_MAX_BYTES)
                if got is not None:
                    fields["result"] = got[0].decode("utf-8", errors="replace")
                    fields["resultTruncated"] = got[1]
            if res.cancelled:
                fields.update(status="cancelled", error="cancelled")
            elif res.timed_out:
                fields.update(status="failed", error=f"timeout after {spec.timeout_seconds:g}s")
            elif res.exit_code == 0:
                fields.update(status="succeeded")
            else:
                fields.update(status="failed", error=f"exit code {res.exit_code}")
        except Exception as e:
            fields = {"status": "failed", "error": f"boot failed: {type(e).__name__}: {e}"}
        finally:
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception:
                    pass
            if not self.keep_workdirs:
                shutil.rmtree(host_dir, ignore_errors=True)
            self._cpus.give(cpus)
            self._slots.release()
        t1 = _now()
        self._set(run, i, finishedAt=t1, durationSeconds=round(t1 - t0, 3), **fields)
        self._persist(run)

    def _persist(self, run: _Run, final: bool = False) -> None:
        with run.persist_lock:
            now = time.monotonic()
            if not final and now - run.last_persist < 1.0:
                return
            run.last_persist = now
            d = self.runs_dir / run.id
            d.mkdir(parents=True, exist_ok=True)
            with run.lock:
                snap = run.full()
            _atomic_write(d / "run.json", json.dumps(snap))
            if final:
                lines = "".join(json.dumps(sb) + "\n" for sb in snap["sandboxes"])
                _atomic_write(d / "results.jsonl", lines)

    def _load(self) -> None:
        if self._idem_path.exists():
            try:
                self._idem = json.loads(self._idem_path.read_text())
            except ValueError:
                self._idem = {}
        for d in sorted(self.runs_dir.iterdir()) if self.runs_dir.exists() else []:
            try:
                snap = json.loads((d / "run.json").read_text())
                spec = SwarmSpec.from_dict(json.loads((d / "spec.json").read_text()))
            except (OSError, ValueError, SpecError):
                continue
            run = _Run(snap["id"], spec, snap.get("concurrency", 1), snap.get("backend", "?"))
            run.created_at = snap.get("createdAt", run.created_at)
            run.finished_at = snap.get("finishedAt")
            run.status = snap.get("status", "completed")
            run.error = snap.get("error")
            run.sandboxes = snap.get("sandboxes", run.sandboxes)
            if run.status == "running":
                run.status = "completed"
                run.error = "runner stopped before the run finished"
                run.finished_at = run.finished_at or _now()
                for sb in run.sandboxes:
                    if sb["status"] not in TERMINAL:
                        sb["status"], sb["error"] = "failed", "runner stopped"
                self._persist(run, final=True)
            run.done.set()
            self._runs[run.id] = run
