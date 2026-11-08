"""Verifiable code reward: run model-written code against tests in a swarm.

One completion -> one fresh sandbox. Typical use inside a GRPO step::

    verifier = CodeVerifier(Flotilla())
    rewards = verifier.rewards(completions, tasks)   # list[float] in [0, 1]

A task is a dict:

    {"entry_point": "add", "tests": [{"args": [1, 2], "expected": 3}, ...]}   # io mode
    {"tests": "def test_a():\\n    assert add(1, 2) == 3\\n"}                    # assert mode

io mode is the hack-resistant one (see harness.py); prefer it for training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .shaping import is_degenerate

HARNESS_SRC = (Path(__file__).parent / "harness.py").read_text()
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def extract_code(text: str, language: str = "python", require_fence: bool = True) -> str | None:
    """Return the last fenced code block (preferring ``language`` blocks)."""
    if not isinstance(text, str):
        return None
    blocks = _FENCE.findall(text)
    tagged = [b for lang, b in blocks if lang.lower() in (language, language[:2], "")]
    if tagged:
        return tagged[-1]
    if blocks:
        return blocks[-1][1]
    return None if require_fence else text


def completion_text(c: Any) -> str:
    """TRL passes either strings or chat message lists."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return str(c[-1].get("content", ""))
    if isinstance(c, dict):
        return str(c.get("content", ""))
    return str(c)


@dataclass
class Score:
    reward: float
    passed: int
    total: int
    error: str | None = None
    status: str = "scored"
    infra: bool = False  # sandbox failure, not the policy's fault

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class CodeVerifier:
    def __init__(
        self,
        client,
        *,
        timeout: float = 60,
        per_test_timeout: float = 5,
        size: str | dict = "1x2",
        image: str | None = None,
        concurrency: int | None = None,
        partial_credit: bool = True,
        require_fence: bool = True,
        language: str = "python",
        python: str = "python3",
        repetition_zero: bool = True,
        infra_reward: float = 0.0,
    ):
        self.client = client
        self.timeout = timeout
        self.per_test_timeout = per_test_timeout
        self.size = size
        self.image = image
        self.concurrency = concurrency
        self.partial_credit = partial_credit
        self.require_fence = require_fence
        self.language = language
        self.python = python
        self.repetition_zero = repetition_zero
        self.infra_reward = infra_reward

    def _task_json(self, task: dict) -> str:
        tests = task["tests"]
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except ValueError:
                pass  # assert-mode source code
        mode = "assert" if isinstance(tests, str) else "io"
        body = {
            "mode": mode,
            "tests": tests,
            "per_test_timeout": self.per_test_timeout,
            "total_timeout": max(1.0, self.timeout - 5),
        }
        if mode == "io":
            if not task.get("entry_point"):
                raise ValueError("io-mode task needs an entry_point")
            body["entry_point"] = task["entry_point"]
        return json.dumps(body)

    def score(self, completions: list[Any], tasks: list[dict], name: str = "code-reward") -> list[Score]:
        if len(completions) != len(tasks):
            raise ValueError("completions and tasks must have the same length")
        scores: list[Score | None] = [None] * len(completions)
        inputs, slots = [], []
        for i, (c, t) in enumerate(zip(completions, tasks)):
            text = completion_text(c)
            if self.repetition_zero and is_degenerate(text):
                scores[i] = Score(0.0, 0, 0, "degenerate repetition", status="repetition")
                continue
            code = extract_code(text, self.language, self.require_fence)
            if not code or not code.strip():
                scores[i] = Score(0.0, 0, 0, "no code block", status="format")
                continue
            inputs.append({"files": {
                "/work/solution.py": code,
                "/work/.task.json": self._task_json(t),
                "/work/.harness.py": HARNESS_SRC,
            }})
            slots.append(i)
        if inputs:
            run = self.client.swarms.run(
                f"{self.python} -I .harness.py",
                inputs=inputs, name=name, image=self.image, size=self.size,
                timeout_seconds=self.timeout, concurrency=self.concurrency,
                result_path="/work/.flotilla_result.json",
            )
            for sb in run["sandboxes"]:
                i = slots[sb["index"]]
                scores[i] = self._parse(sb)
        return scores  # type: ignore[return-value]

    def _parse(self, sb: dict) -> Score:
        err = sb.get("error") or ""
        if sb["status"] == "cancelled" or err.startswith("boot failed"):
            return Score(0.0, 0, 0, err or "cancelled", status=sb["status"], infra=True)
        if not sb.get("result"):
            # the harness always writes a result unless the sandbox itself was
            # killed (timeout); a clean exit without one means a broken harness
            infra = sb["status"] == "succeeded"
            return Score(0.0, 0, 0, err or "no result", status=sb["status"], infra=infra)
        try:
            r = json.loads(sb["result"])
            passed, total = int(r["passed"]), max(1, int(r["total"]))
        except (ValueError, KeyError, TypeError):
            return Score(0.0, 0, 0, "unreadable result", status="error")
        frac = passed / total
        reward = frac if self.partial_credit else float(passed == total)
        return Score(reward, passed, total, r.get("error"))

    def rewards(self, completions: list[Any], tasks: list[dict]) -> list[float]:
        """Plain floats for trainers that cannot mask samples. Infra failures
        get ``infra_reward``; use ``score()`` to see and drop them instead."""
        return [self.infra_reward if s.infra else s.reward for s in self.score(completions, tasks)]


def trl_reward_func(verifier: CodeVerifier, tests_key: str = "tests", entry_point_key: str = "entry_point"):
    """Adapter for TRL's GRPOTrainer(reward_funcs=[...]).

    TRL calls ``f(prompts=..., completions=..., **dataset_columns)``; the
    dataset must carry ``tests`` (and ``entry_point`` for io mode) columns.
    """

    def code_reward(prompts=None, completions=None, **kwargs) -> list[float]:
        tests = kwargs[tests_key]
        eps = kwargs.get(entry_point_key) or [None] * len(tests)
        tasks = [{"tests": t, "entry_point": e} for t, e in zip(tests, eps)]
        return verifier.rewards(completions, tasks)

    code_reward.__name__ = "code_reward"
    return code_reward
