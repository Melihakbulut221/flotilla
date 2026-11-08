"""Backend interface.

A backend turns a host scratch directory into a ``Sandbox``: something that
can receive files, run shell commands with a timeout, and hand files back.
Swarm jobs use a sandbox for exactly one command; RL sessions keep one open
for many commands. Every backend must make ``close()`` destroy all state.
"""

from __future__ import annotations

import os
import posixpath
import shlex
import signal
import stat
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..spec import Size, SpecError, check_abs_path

DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
WORKDIR = "/work"


@dataclass
class SandboxConfig:
    size: Size = field(default_factory=Size)
    network: bool = False
    image: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cpus: list[int] | None = None  # host CPU ids to pin to, if the backend can
    label: str = "sandbox"


@dataclass
class ExecResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "exitCode": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timedOut": self.timed_out,
            "cancelled": self.cancelled,
            "stdoutTruncated": self.stdout_truncated,
            "stderrTruncated": self.stderr_truncated,
            "durationSeconds": round(self.duration, 3),
        }


class BackendUnavailable(RuntimeError):
    pass


def read_tail(path: Path, limit: int) -> tuple[str, bool]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return "", False
    with open(path, "rb") as f:
        if size > limit:
            f.seek(size - limit)
        data = f.read(limit)
    return data.decode("utf-8", errors="replace"), size > limit


def spawn(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: str | Path | None,
    timeout: float,
    log_dir: Path,
    output_limit: int,
    cancel: threading.Event | None = None,
    preexec=None,
    on_kill=None,
    stdin_data: bytes | None = None,
) -> ExecResult:
    """Run ``argv`` in its own process group, enforcing timeout and cancel.

    stdout/stderr go to files (never pipes) so a chatty process cannot
    deadlock us; only the last ``output_limit`` bytes are returned.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.monotonic_ns()}"
    out_path = log_dir / f"stdout.{stamp}"
    err_path = log_dir / f"stderr.{stamp}"
    start = time.monotonic()
    timed_out = cancelled = False
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen(
            argv,
            env=env,
            cwd=cwd,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
            preexec_fn=preexec,
        )
        if stdin_data is not None:
            try:
                proc.stdin.write(stdin_data)
                proc.stdin.close()
            except BrokenPipeError:
                pass
        deadline = start + timeout
        while True:
            try:
                proc.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                cancelled = True
            elif time.monotonic() >= deadline:
                timed_out = True
            if timed_out or cancelled:
                if on_kill is not None:
                    try:
                        on_kill()
                    except Exception:
                        pass
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
                break
    stdout, so_trunc = read_tail(out_path, output_limit)
    stderr, se_trunc = read_tail(err_path, output_limit)
    out_path.unlink(missing_ok=True)
    err_path.unlink(missing_ok=True)
    code = None if (timed_out or cancelled) else proc.returncode
    return ExecResult(
        exit_code=code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        cancelled=cancelled,
        stdout_truncated=so_trunc,
        stderr_truncated=se_trunc,
        duration=time.monotonic() - start,
    )


def make_preexec(memory_gb: float | None, cpus: list[int] | None, limit_memory: bool):
    """rlimit + CPU affinity applied in the child before exec."""

    def _pre():
        import resource

        if limit_memory and memory_gb:
            b = int(memory_gb * (1 << 30))
            try:
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
            except (ValueError, OSError):
                pass
        if cpus:
            try:
                os.sched_setaffinity(0, cpus)
            except (AttributeError, OSError):
                pass

    return _pre


class Sandbox(ABC):
    """A booted environment. Paths given to it are sandbox-absolute."""

    def __init__(self, host_dir: Path, config: SandboxConfig):
        self.host_dir = host_dir
        self.config = config
        self.closed = False

    @abstractmethod
    def write_file(self, path: str, content: str | bytes) -> None: ...

    @abstractmethod
    def read_file(self, path: str, max_bytes: int) -> tuple[bytes, bool] | None:
        """Return (data, truncated) or None if the file does not exist."""

    @abstractmethod
    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: dict[str, str] | None = None,
        cancel: threading.Event | None = None,
        output_limit: int = 16 * 1024,
    ) -> ExecResult: ...

    @abstractmethod
    def close(self) -> None: ...

    def ensure_parent(self, path: str) -> None:
        """Make sure the parent dir of a sandbox path exists."""
        parent = posixpath.dirname(check_abs_path(path, "sandbox path"))
        self.exec(f"mkdir -p {shlex.quote(parent)}", timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Backend(ABC):
    name: str = "base"
    #: whether model-written code is isolated from the host user account
    isolated: bool = True

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return True, ""

    @abstractmethod
    def open(self, host_dir: Path, config: SandboxConfig) -> Sandbox: ...


class StagedFsSandbox(Sandbox):
    """Shared logic for backends that keep the sandbox's writable state in a
    host directory tree (``host_dir/fs`` mirrors sandbox-absolute paths)."""

    #: top-level sandbox dirs that are read-only system dirs and cannot hold files
    reserved_tops: frozenset[str] = frozenset()

    def __init__(self, host_dir: Path, config: SandboxConfig):
        super().__init__(host_dir, config)
        self.fs = host_dir / "fs"
        self.logs = host_dir / "logs"
        for d in (self.fs / "work", self.fs / "tmp", self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def host_path(self, path: str) -> Path:
        norm = check_abs_path(path, "sandbox path")
        top = norm.split("/")[1]
        if top in self.reserved_tops:
            raise SpecError(
                f"{path!r} is inside read-only system dir /{top}; use /work, /tmp or a new top-level dir"
            )
        return self.fs / norm.lstrip("/")

    def tops(self) -> list[str]:
        return sorted(p.name for p in self.fs.iterdir() if p.is_dir() and not p.is_symlink())

    def confined(self, hp: Path) -> Path:
        """Refuse host paths that escape ``fs`` through symlinks the sandbox planted.

        Without this, a command could point /work/result.json at a host file
        (an SSH key, a token) and the host would read or overwrite it.
        """
        root = os.path.realpath(self.fs)
        real = os.path.realpath(hp)
        if real != root and not real.startswith(root + os.sep):
            raise SpecError(f"refusing to follow a symlink out of the sandbox: {hp}")
        return hp

    def write_file(self, path: str, content: str | bytes) -> None:
        hp = self.confined(self.host_path(path))
        hp.parent.mkdir(parents=True, exist_ok=True)
        self.confined(hp)
        data = content.encode() if isinstance(content, str) else content
        fd = os.open(hp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(data)

    def read_file(self, path: str, max_bytes: int) -> tuple[bytes, bool] | None:
        hp = self.host_path(path)
        try:
            self.confined(hp)
            fd = os.open(hp, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except (OSError, SpecError):
            return None
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return None
            data = f.read(max_bytes + 1)
        return data[:max_bytes], len(data) > max_bytes

    def ensure_parent(self, path: str) -> None:
        hp = self.confined(self.host_path(path))
        hp.parent.mkdir(parents=True, exist_ok=True)
        self.confined(hp)

    def base_env(self, extra: dict[str, str] | None, home: str = WORKDIR) -> dict[str, str]:
        env = {
            "PATH": DEFAULT_PATH,
            "HOME": home,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
        }
        env.update(self.config.env)
        if extra:
            env.update(extra)
        return env
