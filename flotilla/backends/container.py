"""OCI container backends: Docker/Podman and Apptainer (Singularity).

Docker/Podman: one long-lived container per sandbox (``sleep infinity``),
commands via ``exec``, destroyed with ``rm -f``. Gives real CPU/memory/pid
limits through cgroups and any image you like (``python:3.12-slim``,
an image with your task's toolchain, ...).

Apptainer: the usual choice on HPC clusters without root (e.g. TRUBA). It
has no long-lived daemon, so state is kept in staged host dirs bound into
the container, the same way the bwrap backend does it.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from .base import (
    WORKDIR,
    Backend,
    ExecResult,
    Sandbox,
    SandboxConfig,
    StagedFsSandbox,
    make_preexec,
    spawn,
)
from .bwrap import RESERVED
from ..spec import check_abs_path

DEFAULT_IMAGE = "python:3.12-slim"

# Run the command under coreutils `timeout` when the image has it, so a timed
# out command does not keep running inside a session container.
_TIMEOUT_WRAPPER = (
    'if command -v timeout >/dev/null 2>&1; then exec timeout -s KILL "$1" sh -c "$2"; fi; '
    'exec sh -c "$2"'
)


class DockerSandbox(Sandbox):
    def __init__(self, host_dir: Path, config: SandboxConfig, backend: "DockerBackend"):
        super().__init__(host_dir, config)
        self.backend = backend
        self.logs = host_dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.name = f"flotilla-{uuid.uuid4().hex[:12]}"
        size = config.size
        argv = [
            backend.exe, "run", "-d", "--init", "--name", self.name,
            "--network", "bridge" if config.network else "none",
            "--cpus", str(size.vcpus),
            "--memory", f"{int(size.memory_gb * 1024)}m",
            "--pids-limit", str(backend.pids_limit),
            "-w", WORKDIR,
            "--entrypoint", "sleep",
            config.image or backend.default_image, "infinity",
        ]
        r = subprocess.run(argv, capture_output=True, timeout=backend.start_timeout)
        if r.returncode != 0:
            raise RuntimeError(f"{backend.name} run failed: {r.stderr.decode(errors='replace').strip()}")

    def _env_args(self, extra: dict[str, str] | None) -> list[str]:
        env = {"PYTHONUNBUFFERED": "1", "HOME": WORKDIR, **self.config.env, **(extra or {})}
        out = []
        for k, v in env.items():
            out += ["-e", f"{k}={v}"]
        return out

    def write_file(self, path: str, content: str | bytes) -> None:
        path = check_abs_path(path, "sandbox path")
        data = content.encode() if isinstance(content, str) else content
        r = subprocess.run(
            [self.backend.exe, "exec", "-i", self.name, "sh", "-c",
             'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", path],
            input=data, capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            raise RuntimeError(f"write {path} failed: {r.stderr.decode(errors='replace').strip()}")

    def read_file(self, path: str, max_bytes: int) -> tuple[bytes, bool] | None:
        path = check_abs_path(path, "sandbox path")
        r = subprocess.run(
            [self.backend.exe, "exec", self.name, "sh", "-c",
             f'test -f "$1" && head -c {max_bytes + 1} "$1"', "sh", path],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            return None
        return r.stdout[:max_bytes], len(r.stdout) > max_bytes

    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: dict[str, str] | None = None,
        cancel: threading.Event | None = None,
        output_limit: int = 16 * 1024,
    ) -> ExecResult:
        argv = [self.backend.exe, "exec", "-w", WORKDIR, *self._env_args(env), self.name,
                "sh", "-c", _TIMEOUT_WRAPPER, "sh", str(int(timeout) + 1), command]
        res = spawn(
            argv, env=None, cwd=None, timeout=timeout + 5, log_dir=self.logs,
            output_limit=output_limit, cancel=cancel,
        )
        if res.exit_code in (124, 137) and res.duration >= timeout:
            res.timed_out, res.exit_code = True, None
        return res

    def close(self) -> None:
        if not self.closed:
            subprocess.run([self.backend.exe, "rm", "-f", self.name], capture_output=True, timeout=60)
            self.closed = True


class DockerBackend(Backend):
    name = "docker"
    isolated = True

    def __init__(self, exe: str = "docker", default_image: str = DEFAULT_IMAGE,
                 pids_limit: int = 4096, start_timeout: float = 600):
        found = shutil.which(exe)
        if not found:
            raise RuntimeError(f"{exe} not found")
        self.exe = found
        self.name = Path(exe).name
        self.default_image = default_image
        self.pids_limit = pids_limit
        self.start_timeout = start_timeout

    @classmethod
    def available(cls, exe: str = "docker") -> tuple[bool, str]:
        found = shutil.which(exe)
        if not found:
            return False, f"{exe} not installed"
        r = subprocess.run([found, "info"], capture_output=True, timeout=20)
        if r.returncode != 0:
            return False, f"{exe} daemon not reachable"
        return True, found

    def open(self, host_dir: Path, config: SandboxConfig) -> DockerSandbox:
        return DockerSandbox(host_dir, config, self)


class PodmanBackend(DockerBackend):
    name = "podman"

    def __init__(self, **kw):
        super().__init__(exe="podman", **kw)

    @classmethod
    def available(cls, exe: str = "podman") -> tuple[bool, str]:
        return DockerBackend.available.__func__(cls, exe)


class ApptainerSandbox(StagedFsSandbox):
    reserved_tops = RESERVED

    def __init__(self, host_dir: Path, config: SandboxConfig, backend: "ApptainerBackend"):
        super().__init__(host_dir, config)
        self.backend = backend

    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: dict[str, str] | None = None,
        cancel: threading.Event | None = None,
        output_limit: int = 16 * 1024,
    ) -> ExecResult:
        b = self.backend
        a = [b.exe, "exec", "--contain", "--cleanenv", "--no-home", "--pwd", WORKDIR]
        if not self.config.network and b.isolate_network:
            a += ["--net", "--network", "none"]
        for top in self.tops():
            a += ["-B", f"{self.fs / top}:/{top}"]
        envd = {"PYTHONUNBUFFERED": "1", "HOME": WORKDIR, "TMPDIR": "/tmp", **self.config.env, **(env or {})}
        for k, v in envd.items():
            a += ["--env", f"{k}={v}"]
        a += [self.config.image or b.default_image, "sh", "-c", command]
        return spawn(
            a, env=None, cwd=str(self.host_dir), timeout=timeout, log_dir=self.logs,
            output_limit=output_limit, cancel=cancel,
            preexec=make_preexec(self.config.size.memory_gb, self.config.cpus, b.limit_memory),
        )

    def close(self) -> None:
        self.closed = True


class ApptainerBackend(Backend):
    name = "apptainer"
    isolated = True

    def __init__(self, default_image: str = f"docker://{DEFAULT_IMAGE}",
                 isolate_network: bool = True, limit_memory: bool = False):
        exe = shutil.which("apptainer") or shutil.which("singularity")
        if not exe:
            raise RuntimeError("apptainer/singularity not found")
        self.exe = exe
        self.default_image = default_image
        self.isolate_network = isolate_network
        self.limit_memory = limit_memory

    @classmethod
    def available(cls) -> tuple[bool, str]:
        exe = shutil.which("apptainer") or shutil.which("singularity")
        return (True, exe) if exe else (False, "apptainer/singularity not installed")

    def open(self, host_dir: Path, config: SandboxConfig) -> ApptainerSandbox:
        return ApptainerSandbox(host_dir, config, self)
