"""Rollout collection: drive a policy through environments in parallel.

A policy is any callable ``policy(messages) -> str`` taking OpenAI-style chat
messages. ``OpenAIChatPolicy`` points at any OpenAI-compatible server, e.g.
your own model behind ``vllm serve`` or ``llama.cpp --server``.

Trajectories are plain dicts (messages + reward + info) written as JSON
Lines, the input format for rejection-sampling fine-tuning, offline GRPO or
any trainer that accepts scored conversations.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

from .env import SYSTEM_PROMPT, ShellTaskEnv
from .shaping import assistant_text, group_advantages, is_degenerate  # noqa: F401  (re-export)

Policy = Callable[[list[dict]], str]


class OpenAIChatPolicy:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 1024, timeout: float = 300, **extra):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.params = {"temperature": temperature, "max_tokens": max_tokens, **extra}
        self.timeout = timeout

    def __call__(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, **self.params}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"] or ""


def run_episode(env: ShellTaskEnv, policy: Policy, system_prompt: str = SYSTEM_PROMPT,
                repetition_zero: bool = True) -> dict:
    """Play one episode. Any exception (sandbox, network, model server) is an
    infra error: the trajectory is kept for debugging but flagged
    ``infra_error`` so it does not count as a policy failure."""
    t0 = time.time()
    messages = [{"role": "system", "content": system_prompt}]

    def out(reward: float, info: dict, infra: bool = False) -> dict:
        return {"task_id": env.task.get("id"), "messages": messages, "reward": reward,
                "turns": env.turn, "info": info, "infra_error": infra,
                "seconds": round(time.time() - t0, 2)}

    try:
        messages.append({"role": "user", "content": env.reset()})
        while True:
            action = policy(messages)
            messages.append({"role": "assistant", "content": action})
            step = env.step(action)
            if step.done:
                result = out(step.reward, step.info)
                if repetition_zero and is_degenerate(assistant_text(result)):
                    result["info"]["repetition_zeroed"] = result["reward"]
                    result["reward"] = 0.0
                return result
            messages.append({"role": "user", "content": step.observation})
    except Exception as e:
        return out(0.0, {"error": f"{type(e).__name__}: {e}"}, infra=True)
    finally:
        env.close()


def run_episodes(client, policy: Policy, tasks: Iterable[dict], *, samples_per_task: int = 1,
                 parallel: int = 8, env_kwargs: dict | None = None) -> list[dict]:
    """Run ``samples_per_task`` episodes per task (GRPO groups) in parallel."""
    jobs = [(t, k) for t in tasks for k in range(samples_per_task)]

    def one(job):
        task, k = job
        out = run_episode(ShellTaskEnv(client, task, **(env_kwargs or {})), policy)
        out["sample"] = k
        return out

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(one, jobs))


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
