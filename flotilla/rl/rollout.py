"""Rollout collection: drive a policy through environments in parallel.

A policy is any callable taking OpenAI-style chat messages:

- for ``ShellTaskEnv`` it returns the reply text;
- for ``ToolTaskEnv`` it returns either an assistant message dict (which may
  carry ``tool_calls``) or plain text containing ``<tool_call>`` blocks.

``OpenAIChatPolicy`` (text) and ``OpenAIToolPolicy`` (native function
calling) point at any OpenAI-compatible server, e.g. your own model behind
``vllm serve`` or ``llama.cpp --server``.

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

from .env import ShellTaskEnv
from .tools import TOOLS, ToolTaskEnv, render_tools_prompt
from .shaping import assistant_text, group_advantages, is_degenerate  # noqa: F401  (re-export)

Policy = Callable[[list[dict]], "str | dict"]


class OpenAIChatPolicy:
    """Text completions from an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 1024, timeout: float = 300, **extra):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.params = {"temperature": temperature, "max_tokens": max_tokens, **extra}
        self.timeout = timeout

    def _complete(self, messages: list[dict], **extra) -> dict:
        body = json.dumps({"model": self.model, "messages": messages, **self.params, **extra}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]

    def __call__(self, messages: list[dict]) -> str:
        return self._complete(messages).get("content") or ""


class OpenAIToolPolicy(OpenAIChatPolicy):
    """Native function calling. With vLLM, serve the model with
    ``--enable-auto-tool-choice --tool-call-parser <parser>`` (e.g. hermes)."""

    def __init__(self, base_url: str, model: str, tools: list[dict] = TOOLS, **kw):
        super().__init__(base_url, model, **kw)
        self.tools = tools

    def __call__(self, messages: list[dict]) -> dict:
        msg = self._complete(messages, tools=self.tools, tool_choice="auto")
        return _assistant_message(msg)


def _assistant_message(action: "str | dict") -> dict:
    """Normalise a policy reply into a clean assistant message."""
    if isinstance(action, str):
        return {"role": "assistant", "content": action}
    msg = {"role": "assistant", "content": action.get("content") or ""}
    if action.get("tool_calls"):
        calls = []
        for i, c in enumerate(action["tool_calls"]):
            fn = c.get("function") or {}
            args = fn.get("arguments", {})
            calls.append({"id": c.get("id") or f"call_{i}", "type": "function",
                          "function": {"name": fn.get("name"),
                                       "arguments": args if isinstance(args, str) else json.dumps(args)}})
        msg["tool_calls"] = calls
    return msg


def run_episode(env: ShellTaskEnv, policy: Policy, system_prompt: str | None = None,
                repetition_zero: bool = True, native_tools: bool = True) -> dict:
    """Play one episode. Any exception (sandbox, network, model server) is an
    infra error: the trajectory is kept for debugging but flagged
    ``infra_error`` so it does not count as a policy failure.

    For a ``ToolTaskEnv``, ``native_tools=False`` appends the tool schemas and
    the ``<tool_call>`` format to the system prompt, for servers without a
    tool-call parser.
    """
    tool_env = isinstance(env, ToolTaskEnv)
    if system_prompt is None:
        system_prompt = env.system_prompt
        if tool_env and not native_tools:
            system_prompt += "\n\n" + render_tools_prompt(env.tools)
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
            if tool_env:
                action = _assistant_message(action)
                messages.append(action)
                step = env.step(action)
                messages.extend(step.observation)
            else:
                messages.append({"role": "assistant", "content": action})
                step = env.step(action)
            if step.done:
                result = out(step.reward, step.info)
                if repetition_zero and is_degenerate(assistant_text(result)):
                    result["info"]["repetition_zeroed"] = result["reward"]
                    result["reward"] = 0.0
                return result
            if not tool_env:
                messages.append({"role": "user", "content": step.observation})
    except Exception as e:
        return out(0.0, {"error": f"{type(e).__name__}: {e}"}, infra=True)
    finally:
        env.close()


def run_episodes(client, policy: Policy, tasks: Iterable[dict], *, samples_per_task: int = 1,
                 parallel: int = 8, env_cls: type = ShellTaskEnv, env_kwargs: dict | None = None,
                 episode_kwargs: dict | None = None) -> list[dict]:
    """Run ``samples_per_task`` episodes per task (GRPO groups) in parallel.

    ``env_cls`` is ``ShellTaskEnv`` (text actions) or ``ToolTaskEnv``
    (function calling); ``episode_kwargs`` go to ``run_episode``."""
    jobs = [(t, k) for t in tasks for k in range(samples_per_task)]

    def one(job):
        task, k = job
        out = run_episode(env_cls(client, task, **(env_kwargs or {})), policy, **(episode_kwargs or {}))
        out["sample"] = k
        return out

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(one, jobs))


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
