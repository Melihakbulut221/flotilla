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
give partial credit. Verifiers should judge artifacts (files, program output
run as a subprocess) rather than import agent code into their own process,
which an agent could patch.
"""

from __future__ import annotations

import re
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
    ):
        self.client = client
        self.task = task
        self.image = image
        self.size = size
        self.command_timeout = command_timeout
        self.verify_timeout = verify_timeout
        self.output_limit = output_limit
        self.invalid_penalty = invalid_penalty
        self.max_turns = int(task.get("max_turns", 8))
        self.session = None
        self.turn = 0
        self.done = False

    def reset(self) -> str:
        self.close()
        self.turn, self.done = 0, False
        self.session = self.client.sessions.open(
            image=self.image, size=self.size, network=False,
            files=self.task.get("files") or {}, timeout_seconds=self.command_timeout,
        )
        if self.task.get("setup"):
            r = self.session.exec(self.task["setup"], timeout=max(self.command_timeout, 300))
            if r.exit_code != 0:
                raise RuntimeError(f"task setup failed: {r.stderr[-500:]}")
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
        for p, c in (self.task.get("verify_files") or {}).items():
            self.session.write_file(p, c)
        r = self.session.exec(self.task["verify"], timeout=self.verify_timeout, output_limit=8000)
        m = _REWARD.findall(r.stdout)
        if r.timed_out:
            reward = 0.0
        elif m:
            reward = float(m[-1])
        else:
            reward = 1.0 if r.exit_code == 0 else 0.0
        return reward, {"verify_exit": r.exit_code, "verify_timed_out": r.timed_out,
                        "verify_output": (r.stdout + r.stderr)[-1000:]}

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
