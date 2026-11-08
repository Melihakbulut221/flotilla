"""Plain-process backend: no isolation, works everywhere.

Sandbox-absolute paths are rooted under ``$FLOTILLA_ROOT`` on the host and
the command runs with ``$FLOTILLA_ROOT/work`` as its working directory, so
commands that use relative paths (``python solve.py``) behave the same here
as in the isolated backends. Do not run untrusted model output with it.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from .base import (
    DEFAULT_PATH,
    Backend,
    ExecResult,
    SandboxConfig,
    StagedFsSandbox,
    make_preexec,
    spawn,
)


class LocalSandbox(StagedFsSandbox):
    def __init__(self, host_dir: Path, config: SandboxConfig, limit_memory: bool):
        super().__init__(host_dir, config)
        self.limit_memory = limit_memory

    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: dict[str, str] | None = None,
        cancel: threading.Event | None = None,
        output_limit: int = 16 * 1024,
    ) -> ExecResult:
        root = str(self.fs.resolve())
        e = self.base_env(env, home=f"{root}/work")
        e["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', DEFAULT_PATH)}"
        e["TMPDIR"] = f"{root}/tmp"
        e["FLOTILLA_ROOT"] = root
        return spawn(
            ["/bin/sh", "-c", command],
            env=e,
            cwd=f"{root}/work",
            timeout=timeout,
            log_dir=self.logs,
            output_limit=output_limit,
            cancel=cancel,
            preexec=make_preexec(self.config.size.memory_gb, self.config.cpus, self.limit_memory),
        )

    def close(self) -> None:
        self.closed = True


class LocalBackend(Backend):
    name = "local"
    isolated = False

    def __init__(self, limit_memory: bool = True):
        self.limit_memory = limit_memory

    def open(self, host_dir: Path, config: SandboxConfig) -> LocalSandbox:
        return LocalSandbox(host_dir, config, self.limit_memory)
