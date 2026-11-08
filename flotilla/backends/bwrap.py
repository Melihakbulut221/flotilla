"""Bubblewrap backend: rootless Linux namespaces, the default isolated backend.

Each command runs in fresh user/pid/ipc/uts/cgroup namespaces (and a fresh,
empty network namespace unless ``network=True``). The root filesystem is an
empty tmpfs with only system dirs bound read-only; the host home directory,
SSH keys and tokens are not visible. Writable state lives in staged host dirs
(``/work``, ``/tmp`` and any new top-level dir a file is written to), which
is what makes multi-command sessions keep their files between commands.

Not a VM: the kernel is shared with the host. That is the right trade-off for
reward computation on model-written code on a workstation or HPC node; for
hostile multi-tenant workloads use a VM-based backend.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from .base import (
    DEFAULT_PATH,
    WORKDIR,
    Backend,
    ExecResult,
    SandboxConfig,
    StagedFsSandbox,
    make_preexec,
    spawn,
)

SYSTEM_DIRS = ("usr", "bin", "sbin", "lib", "lib32", "lib64", "libx32", "etc", "opt")
RESERVED = frozenset(SYSTEM_DIRS + ("proc", "dev", "sys", "run"))


def _python_prefixes() -> list[str]:
    out = []
    for p in {sys.prefix, sys.base_prefix}:
        rp = os.path.realpath(p)
        top = rp.split("/")[1] if rp != "/" else ""
        if top and top not in SYSTEM_DIRS and rp not in out:
            out.append(rp)
    return out


class BwrapSandbox(StagedFsSandbox):
    reserved_tops = RESERVED

    def __init__(self, host_dir: Path, config: SandboxConfig, backend: "BwrapBackend"):
        super().__init__(host_dir, config)
        self.backend = backend

    def argv(self, env: dict[str, str]) -> list[str]:
        b = self.backend
        a = [b.bwrap, "--unshare-all", "--die-with-parent", "--new-session"]
        if self.config.network:
            a += ["--share-net"]
        a += ["--hostname", "flotilla", "--tmpfs", "/"]
        for name in SYSTEM_DIRS:
            hp = f"/{name}"
            if os.path.islink(hp):
                a += ["--symlink", os.readlink(hp), hp]
            elif os.path.isdir(hp):
                a += ["--ro-bind", hp, hp]
        if self.config.network:
            a += ["--ro-bind-try", "/run/systemd/resolve", "/run/systemd/resolve"]
        a += ["--proc", "/proc", "--dev", "/dev"]
        for top in self.tops():
            a += ["--bind", str(self.fs / top), f"/{top}"]
        for p in b.expose:
            a += ["--ro-bind-try", p, p]
        a += ["--chdir", WORKDIR, "--clearenv"]
        for k, v in env.items():
            a += ["--setenv", k, v]
        return a

    def exec(
        self,
        command: str,
        *,
        timeout: float,
        env: dict[str, str] | None = None,
        cancel: threading.Event | None = None,
        output_limit: int = 16 * 1024,
    ) -> ExecResult:
        e = self.base_env(env)
        e["TMPDIR"] = "/tmp"
        if self.backend.expose_python:
            e["PATH"] = ":".join(f"{p}/bin" for p in _python_prefixes()) + ":" + DEFAULT_PATH
        argv = self.argv(e) + ["/bin/sh", "-c", command]
        return spawn(
            argv,
            env={"PATH": DEFAULT_PATH},
            cwd=str(self.host_dir),
            timeout=timeout,
            log_dir=self.logs,
            output_limit=output_limit,
            cancel=cancel,
            preexec=make_preexec(self.config.size.memory_gb, self.config.cpus, self.backend.limit_memory),
        )

    def close(self) -> None:
        self.closed = True


class BwrapBackend(Backend):
    name = "bwrap"
    isolated = True

    def __init__(
        self,
        expose: list[str] | None = None,
        expose_python: bool = True,
        limit_memory: bool = True,
        bwrap: str | None = None,
    ):
        found = bwrap or shutil.which("bwrap")
        if not found:
            raise RuntimeError("bwrap not found; install bubblewrap or use another backend")
        self.bwrap = found
        self.expose_python = expose_python
        self.limit_memory = limit_memory
        ex = [os.path.realpath(p) for p in (expose or [])]
        if expose_python:
            ex = _python_prefixes() + ex
        self.expose = list(dict.fromkeys(ex))

    @classmethod
    def available(cls) -> tuple[bool, str]:
        exe = shutil.which("bwrap")
        if not exe:
            return False, "bwrap not installed (apt install bubblewrap)"
        try:
            r = subprocess.run(
                [exe, "--unshare-all", "--die-with-parent", "--ro-bind", "/", "/", "true"],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"bwrap failed to start: {e}"
        if r.returncode != 0:
            return False, "bwrap cannot create namespaces: " + r.stderr.decode(errors="replace").strip()
        return True, exe

    def open(self, host_dir: Path, config: SandboxConfig) -> BwrapSandbox:
        return BwrapSandbox(host_dir, config, self)
