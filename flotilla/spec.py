"""Swarm request model and validation.

A swarm runs one shell command on many fresh sandboxes. Each sandbox gets
either nothing but its index (``count``) or its own entry from ``inputs``
(extra args, env vars, files). The field names accept both snake_case and
camelCase so JSON bodies written for hosted swarm services port unchanged.
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, field
from typing import Any

TAIL_BYTES = 16 * 1024
RESULT_MAX_BYTES = 64 * 1024
MIN_TIMEOUT = 1
MAX_TIMEOUT = 6 * 3600
MAX_COUNT = 100_000


class SpecError(ValueError):
    """Raised when a swarm or session request is invalid."""


def _get(d: dict, *names: str, default: Any = None) -> Any:
    for n in names:
        if n in d:
            return d[n]
    return default


def check_abs_path(path: str, what: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise SpecError(f"{what} must be an absolute path, got {path!r}")
    norm = posixpath.normpath(path)
    if norm == "/" or ".." in norm.split("/"):
        raise SpecError(f"{what} is not a usable path: {path!r}")
    return norm


@dataclass
class Size:
    vcpus: int = 2
    memory_gb: float = 4.0

    @classmethod
    def from_any(cls, value: Any) -> "Size":
        if value is None:
            return cls()
        if isinstance(value, Size):
            return value
        if isinstance(value, str):
            # "2x4" -> 2 vCPU, 4 GB
            try:
                c, m = value.lower().split("x", 1)
                return cls(int(c), float(m)).validate()
            except ValueError as e:
                raise SpecError(f"size must look like '2x4', got {value!r}") from e
        if isinstance(value, dict):
            return cls(
                int(_get(value, "vcpus", default=2)),
                float(_get(value, "memoryGb", "memory_gb", default=4)),
            ).validate()
        raise SpecError(f"bad size: {value!r}")

    def validate(self) -> "Size":
        if self.vcpus < 1 or self.vcpus > 256:
            raise SpecError("size.vcpus must be in 1..256")
        if self.memory_gb <= 0 or self.memory_gb > 1024:
            raise SpecError("size.memoryGb must be in (0, 1024]")
        return self

    def to_dict(self) -> dict:
        return {"vcpus": self.vcpus, "memoryGb": self.memory_gb}


@dataclass
class SandboxInput:
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "SandboxInput":
        if isinstance(value, SandboxInput):
            return value
        if not isinstance(value, dict):
            raise SpecError(f"each input must be an object, got {type(value).__name__}")
        args = [str(a) for a in value.get("args", []) or []]
        env = {str(k): str(v) for k, v in (value.get("env", {}) or {}).items()}
        files = {}
        for p, c in (value.get("files", {}) or {}).items():
            if not isinstance(c, str):
                raise SpecError(f"file contents for {p!r} must be a string")
            files[check_abs_path(p, "input file path")] = c
        return cls(args, env, files)

    def to_dict(self) -> dict:
        return {"args": self.args, "env": self.env, "files": self.files}


@dataclass
class SwarmSpec:
    command: str | list[str]
    count: int | None = None
    inputs: list[SandboxInput] | None = None
    name: str = ""
    image: str | None = None
    size: Size = field(default_factory=Size)
    timeout_seconds: float = 600
    concurrency: int | None = None
    result_path: str | None = None
    network: bool = False
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SwarmSpec":
        if not isinstance(d, dict):
            raise SpecError("request body must be a JSON object")
        inputs = _get(d, "inputs")
        spec = cls(
            command=_get(d, "command"),
            count=_get(d, "count"),
            inputs=[SandboxInput.from_any(i) for i in inputs] if inputs is not None else None,
            name=str(_get(d, "name", default="") or ""),
            image=_get(d, "image", "template"),
            size=Size.from_any(_get(d, "size")),
            timeout_seconds=_get(d, "timeoutSeconds", "timeout_seconds", default=600),
            concurrency=_get(d, "concurrency"),
            result_path=_get(d, "resultPath", "result_path"),
            network=bool(_get(d, "network", default=False)),
            env={str(k): str(v) for k, v in (_get(d, "env", default={}) or {}).items()},
        )
        return spec.validate()

    def validate(self) -> "SwarmSpec":
        if isinstance(self.command, list):
            if not self.command or not all(isinstance(a, str) for a in self.command):
                raise SpecError("command list must be non-empty strings")
        elif not isinstance(self.command, str) or not self.command.strip():
            raise SpecError("command is required")
        if (self.count is None) == (self.inputs is None):
            raise SpecError("give exactly one of count or inputs")
        if self.count is not None:
            if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
                raise SpecError("count must be a positive integer")
            if self.count > MAX_COUNT:
                raise SpecError(f"count exceeds {MAX_COUNT}")
        if self.inputs is not None:
            if not self.inputs:
                raise SpecError("inputs must not be empty")
            if len(self.inputs) > MAX_COUNT:
                raise SpecError(f"inputs exceed {MAX_COUNT}")
        try:
            self.timeout_seconds = float(self.timeout_seconds)
        except (TypeError, ValueError) as e:
            raise SpecError("timeoutSeconds must be a number") from e
        if not MIN_TIMEOUT <= self.timeout_seconds <= MAX_TIMEOUT:
            raise SpecError(f"timeoutSeconds must be in {MIN_TIMEOUT}..{MAX_TIMEOUT}")
        if self.concurrency is not None:
            if not isinstance(self.concurrency, int) or self.concurrency < 1:
                raise SpecError("concurrency must be a positive integer")
        if self.result_path is not None:
            self.result_path = check_abs_path(self.result_path, "resultPath")
        self.size.validate()
        return self

    @property
    def total(self) -> int:
        return self.count if self.count is not None else len(self.inputs or [])

    def input_for(self, index: int) -> SandboxInput:
        if self.inputs is None:
            return SandboxInput()
        return self.inputs[index]

    def command_for(self, index: int) -> str:
        base = self.command if isinstance(self.command, str) else shlex.join(self.command)
        args = self.input_for(index).args
        return f"{base} {shlex.join(args)}" if args else base

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "count": self.count,
            "inputs": [i.to_dict() for i in self.inputs] if self.inputs is not None else None,
            "name": self.name,
            "image": self.image,
            "size": self.size.to_dict(),
            "timeoutSeconds": self.timeout_seconds,
            "concurrency": self.concurrency,
            "resultPath": self.result_path,
            "network": self.network,
            "env": self.env,
        }


def parse_duration(text: str | int | float) -> float:
    """'90' -> 90, '10m' -> 600, '1h' -> 3600, '1500ms' -> 1.5."""
    if isinstance(text, (int, float)):
        return float(text)
    t = text.strip().lower()
    for suffix, mult in (("ms", 0.001), ("s", 1), ("m", 60), ("h", 3600)):
        if t.endswith(suffix) and t[: -len(suffix)].replace(".", "", 1).isdigit():
            return float(t[: -len(suffix)]) * mult
    try:
        return float(t)
    except ValueError as e:
        raise SpecError(f"bad duration: {text!r}") from e
