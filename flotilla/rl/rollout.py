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


def run_episode(env: ShellTaskEnv, policy: Policy, system_prompt: str = SYSTEM_PROMPT) -> dict:
    t0 = time.time()
    messages = [{"role": "system", "content": system_prompt}]
    try:
        messages.append({"role": "user", "content": env.reset()})
        while True:
            action = policy(messages)
            messages.append({"role": "assistant", "content": action})
            step = env.step(action)
            if step.done:
                return {"task_id": env.task.get("id"), "messages": messages, "reward": step.reward,
                        "turns": env.turn, "info": step.info, "seconds": round(time.time() - t0, 2)}
            messages.append({"role": "user", "content": step.observation})
    except Exception as e:
        return {"task_id": env.task.get("id"), "messages": messages, "reward": 0.0, "turns": env.turn,
                "info": {"error": f"{type(e).__name__}: {e}"}, "seconds": round(time.time() - t0, 2)}
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


def group_advantages(trajectories: list[dict], key: str = "task_id") -> list[dict]:
    """Add GRPO-style group-normalised advantages: (r - mean_group) / (std_group + eps)."""
    groups: dict = {}
    for t in trajectories:
        groups.setdefault(t[key], []).append(t["reward"])
    for t in trajectories:
        rs = groups[t[key]]
        mean = sum(rs) / len(rs)
        std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5
        t["advantage"] = (t["reward"] - mean) / (std + 1e-6)
    return trajectories


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
