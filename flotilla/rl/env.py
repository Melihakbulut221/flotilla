"""Multi-turn tool-use environment on a persistent sandbox.

The policy sees a task, then on each turn replies with either a shell block::

    ```bash
    python app.py
    ```

which is executed in the sandbox (the observation is its output), or a line
``SUBMIT`` which ends the episode and runs the task's verifier. The episode
also ends (and is verified) after ``max_turns`` actions.

Task dict::

    {
      "id": "fix-off-by-one",
      "prompt": "...what the agent must achieve...",
      "files": {"/work/app.py": "..."},            # visible starting state
      "setup": "pip install ...",                   # optional, run once at reset
      "verify_files": {"/verify/check.py": "..."},  # hidden, written only at verify
      "verify": "python3 /verify/check.py",         # exit 0 -> reward 1.0
      "max_turns": 8
    }

A verifier may print ``FLOTILLA_REWARD=<float>`` as its last stdout line to
give partial credit.

Isolated verification (the default): the verifier never runs in the sandbox
the agent controlled. At the end of the episode the agent's ``/work`` is
packed, sanitised on the host (regular files and directories only: no
symlinks, devices, absolute or ``..`` paths) and unpacked into a fresh
sandbox. There the task ``files`` outside ``/work`` are written again
untouched, ``setup`` runs again, the hidden ``verify_files`` are written and
``verify`` runs. Anything the agent changed outside ``/work`` (a reference
file it overwrote, something planted in ``/tmp``, a shadowing module next to
the verifier) is gone. Deliverables must therefore live under ``/work``. Pass
``isolated_verify=False`` for tasks whose result is state outside ``/work``.

Verifiers should still judge artifacts (files, program output run as a
subprocess) rather than import agent code into their own process.
"""

from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass, field
from typing import Any

_BASH = re.compile(r"```(?:bash|sh|shell|console)?[ \t]*\n(.*?)```", re.S)
_SUBMIT = re.compile(r"^\s*SUBMIT\s*$", re.M)
_REWARD = re.compile(r"^FLOTILLA_REWARD=([-+0-9.eE]+)\s*$", re.M)

SYSTEM_PROMPT = (
    "You are working in a Linux sandbox with a POSIX shell; your working directory is /work. "
    "Each reply must contain exactly one action:\n"
    "- a shell command block:\n```bash\n<commands>\n```\n"
    "- or the single line SUBMIT when the task is done.\n"
    "You will see the command output after each command. There is no network access."
)


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


def parse_action(text: str) -> tuple[str, str | None]:
    """-> ("bash", command) | ("submit", None) | ("invalid", None)."""
    m = _BASH.findall(text or "")
    if m:
        return "bash", m[-1].strip()
    if _SUBMIT.search(text or ""):
        return "submit", None
    return "invalid", None


def _fmt(res) -> str:
    out = res.stdout
    if res.stderr:
        out += ("\n" if out and not out.endswith("\n") else "") + res.stderr
    head = "[output truncated, showing the end]\n" if (res.stdout_truncated or res.stderr_truncated) else ""
    if res.timed_out:
        tail = "\n[command timed out]"
    else:
        tail = f"\n[exit code {res.exit_code}]"
    return head + out.rstrip("\n") + tail


class ShellTaskEnv:
    def __init__(
        self,
        client,
        task: dict,
        *,
        image: str | None = None,
        size: str | dict = "1x2",
        command_timeout: float = 30,
        verify_timeout: float = 120,
        output_limit: int = 4000,
        invalid_penalty: float = 0.0,
        isolated_verify: bool = True,
        max_workspace_bytes: int = 16 * 1024 * 1024,
    ):
        self.client = client
        self.task = task
        self.image = image
        self.size = size
        self.command_timeout = command_timeout
        self.verify_timeout = verify_timeout
        self.output_limit = output_limit
        self.invalid_penalty = invalid_penalty
        self.isolated_verify = isolated_verify
        self.max_workspace_bytes = max_workspace_bytes
        self.max_turns = int(task.get("max_turns", 8))
        self.session = None
        self.turn = 0
        self.done = False

    def _open(self, files: dict | None = None):
        sess = self.client.sessions.open(
            image=self.image, size=self.size, network=False,
            files=files or {}, timeout_seconds=self.command_timeout,
        )
        if self.task.get("setup"):
            r = sess.exec(self.task["setup"], timeout=max(self.command_timeout, 300))
            if r.exit_code != 0:
                sess.close()
                raise RuntimeError(f"task setup failed: {r.stderr[-500:]}")
        return sess

    def reset(self) -> str:
        self.close()
        self.turn, self.done = 0, False
        self.session = self._open(self.task.get("files"))
        return self.task["prompt"]

    def step(self, action_text: str) -> StepResult:
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if self.session is None:
            raise RuntimeError("call reset() first")
        self.turn += 1
        kind, cmd = parse_action(action_text)
        if kind == "submit":
            return self._finish("submitted")
        if kind == "invalid":
            obs = "No action found. Reply with a ```bash``` block or the line SUBMIT."
            if self.turn >= self.max_turns:
                return self._finish("max_turns", extra_obs=obs)
            return StepResult(obs, -self.invalid_penalty, False, {"action": "invalid", "turn": self.turn})
        res = self.session.exec(cmd, timeout=self.command_timeout, output_limit=self.output_limit)
        obs = _fmt(res)
        if self.turn >= self.max_turns:
            return self._finish("max_turns", extra_obs=obs)
        return StepResult(obs, 0.0, False, {"action": "bash", "turn": self.turn, "exit_code": res.exit_code})

    def verify(self) -> tuple[float, dict]:
        """Score the episode. Sandbox failures raise (they are infra errors);
        anything the agent caused yields a reward of 0 with a reason."""
        if not self.isolated_verify:
            return self._run_verifier(self.session)
        tar_bytes, problem = self._snapshot_workspace()
        if problem:
            return 0.0, {"verify_error": problem}
        # task inputs outside /work are restored pristine; /work comes from the agent
        pristine = {p: c for p, c in (self.task.get("files") or {}).items()
                    if not (p == "/work" or p.startswith("/work/"))}
        grader = self._open(pristine)
        try:
            grader.write_file("/tmp/.flotilla_ws.tar", tar_bytes)
            r = grader.exec("tar -xf /tmp/.flotilla_ws.tar -C /work && rm -f /tmp/.flotilla_ws.tar",
                            timeout=300)
            if r.exit_code != 0:
                raise RuntimeError(f"workspace restore failed: {r.stderr[-300:]}")
            reward, info = self._run_verifier(grader)
            info["isolated_verify"] = True
            return reward, info
        finally:
            grader.close()

    def _run_verifier(self, sess) -> tuple[float, dict]:
        for p, c in (self.task.get("verify_files") or {}).items():
            sess.write_file(p, c)
        r = sess.exec(self.task["verify"], timeout=self.verify_timeout, output_limit=8000)
        m = _REWARD.findall(r.stdout)
        if r.timed_out:
            reward = 0.0
        elif m:
            reward = float(m[-1])
        else:
            reward = 1.0 if r.exit_code == 0 else 0.0
        return reward, {"verify_exit": r.exit_code, "verify_timed_out": r.timed_out,
                        "verify_output": (r.stdout + r.stderr)[-1000:]}

    def _snapshot_workspace(self) -> tuple[bytes | None, str | None]:
        """Pack the agent's /work and rebuild it on the host as a clean tar."""
        r = self.session.exec("tar -cf /tmp/.flotilla_snapshot.tar -C /work . 2>&1", timeout=300,
                              output_limit=2000)
        if r.timed_out or r.exit_code not in (0, 1):  # GNU tar: 1 = some files changed while reading
            return None, f"could not pack /work (exit {r.exit_code}): {r.stdout[-300:]}"
        raw = self.session.read_file("/tmp/.flotilla_snapshot.tar", self.max_workspace_bytes + 1)
        if raw is None:
            return None, "workspace archive missing"
        if len(raw) > self.max_workspace_bytes:
            return None, f"workspace larger than {self.max_workspace_bytes} bytes"
        try:
            return sanitize_tar(raw, self.max_workspace_bytes), None
        except (tarfile.TarError, ValueError) as e:
            return None, f"bad workspace archive: {e}"

    def _finish(self, why: str, extra_obs: str = "") -> StepResult:
        self.done = True
        try:
            reward, info = self.verify()
        finally:
            self.close()
        info.update(end=why, turn=self.turn)
        return StepResult(extra_obs or "Episode finished.", reward, True, info)

    def close(self) -> None:
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        self.close()


def sanitize_tar(raw: bytes, max_bytes: int) -> bytes:
    """Rebuild a tar keeping only regular files and directories with safe,
    relative names. Symlinks, hard links, devices and FIFOs are dropped."""
    src = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    out = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=out, mode="w:") as dst:
        for m in src.getmembers():
            name = m.name
            while name.startswith("./"):
                name = name[2:]
            if not name or name == ".":
                continue
            parts = name.split("/")
            if name.startswith("/") or ".." in parts:
                continue
            if m.isdir():
                info = tarfile.TarInfo(name)
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                dst.addfile(info)
            elif m.isreg():
                total += m.size
                if total > max_bytes:
                    raise ValueError("workspace too large")
                info = tarfile.TarInfo(name)
                info.size, info.mode = m.size, (0o755 if m.mode & 0o111 else 0o644)
                dst.addfile(info, src.extractfile(m))
    return out.getvalue()
