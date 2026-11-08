from __future__ import annotations

from .base import Backend, BackendUnavailable, ExecResult, Sandbox, SandboxConfig
from .bwrap import BwrapBackend
from .container import ApptainerBackend, DockerBackend, PodmanBackend
from .local import LocalBackend

BACKENDS = {
    "bwrap": BwrapBackend,
    "docker": DockerBackend,
    "podman": PodmanBackend,
    "apptainer": ApptainerBackend,
    "local": LocalBackend,
}

# Order tried by "auto": isolated backends first, plain processes last.
AUTO_ORDER = ("bwrap", "podman", "docker", "apptainer")


def doctor() -> dict[str, tuple[bool, str]]:
    report = {}
    for name, cls in BACKENDS.items():
        try:
            report[name] = cls.available()
        except Exception as e:  # pragma: no cover - defensive
            report[name] = (False, str(e))
    return report


def get_backend(name: str = "auto", **kwargs) -> Backend:
    if isinstance(name, Backend):
        return name
    if name == "auto":
        for cand in AUTO_ORDER:
            ok, _ = BACKENDS[cand].available()
            if ok:
                return BACKENDS[cand](**kwargs)
        raise BackendUnavailable(
            "no isolated backend found (bwrap, podman, docker, apptainer); "
            "install bubblewrap, or pass backend='local' to run without isolation"
        )
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    ok, why = BACKENDS[name].available()
    if not ok:
        raise BackendUnavailable(f"backend {name!r} unavailable: {why}")
    return BACKENDS[name](**kwargs)


__all__ = [
    "Backend",
    "BackendUnavailable",
    "ExecResult",
    "Sandbox",
    "SandboxConfig",
    "BACKENDS",
    "doctor",
    "get_backend",
]
