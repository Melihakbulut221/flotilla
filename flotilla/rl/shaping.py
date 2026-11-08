"""Reward shaping and GRPO group post-processing.

Order matters. Apply these after every per-rollout reward is final, so a
later step cannot silently overwrite an earlier zero::

    trajs = run_episodes(...)                    # rewards final, infra errors flagged
    trajs = length_penalty(trajs)                # optional: shorter passing rollouts win ties
    trajs = group_advantages(trajs)              # infra rollouts excluded from group stats
    kept, stats = filter_groups(trajs)           # drop groups with no learning signal

Infra errors (a sandbox that failed to boot, an unreachable model server)
are not the policy's fault. Scoring them 0 would push the advantages of the
whole group around for no reason, so they are flagged ``infra_error`` and
left out of their group: advantage 0, excluded from mean and std.
"""

from __future__ import annotations

import math
import re
from typing import Callable

_WORD = re.compile(r"\S+")


def ngram_uniqueness(text: str, n: int = 4) -> float:
    """Distinct word n-grams / total word n-grams (1.0 = no repetition)."""
    words = _WORD.findall(text or "")
    if len(words) < n:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def is_degenerate(text: str, *, n: int = 4, threshold: float = 0.15, min_words: int = 500) -> bool:
    """True for long, highly repetitive text (a model stuck in a loop)."""
    if len(_WORD.findall(text or "")) < min_words:
        return False
    return ngram_uniqueness(text, n) < threshold


def assistant_text(traj: dict) -> str:
    return "\n".join(m.get("content") or "" for m in traj.get("messages", []) if m.get("role") == "assistant")


def _groups(trajs: list[dict], key: str) -> dict:
    groups: dict = {}
    for t in trajs:
        groups.setdefault(t.get(key), []).append(t)
    return groups


def _percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def length_penalty(
    trajs: list[dict],
    *,
    key: str = "task_id",
    length: Callable[[dict], float] = lambda t: len(assistant_text(t)),
    pass_threshold: float = 0.5,
    min_pass_fraction: float = 0.5,
    percentile: float = 30,
    max_penalty: float = 0.2,
) -> list[dict]:
    """Penalise long rollouts among the passing ones in a group.

    Only applies when more than ``min_pass_fraction`` of the group passed, so
    hard tasks never discourage the long attempts that eventually succeed.
    A passing rollout longer than the group's ``percentile`` length loses up
    to ``max_penalty``, scaled linearly up to the longest passing rollout.
    """
    for group in _groups(trajs, key).values():
        valid = [t for t in group if not t.get("infra_error")]
        passing = [t for t in valid if t["reward"] >= pass_threshold]
        if not valid or len(passing) / len(valid) <= min_pass_fraction:
            continue
        lens = {id(t): float(length(t)) for t in passing}
        base = _percentile(list(lens.values()), percentile)
        top = max(lens.values())
        for t in passing:
            over = lens[id(t)] - base
            if over > 0 and top > base:
                pen = max_penalty * min(1.0, over / (top - base))
                t["reward"] = t["reward"] - pen
                t.setdefault("info", {})["length_penalty"] = round(pen, 4)
    return trajs


def group_advantages(trajs: list[dict], key: str = "task_id", normalize_std: bool = True,
                     eps: float = 1e-6) -> list[dict]:
    """GRPO advantages: (r - mean) / (std + eps) within each group.

    Rollouts flagged ``infra_error`` get advantage 0 and do not count towards
    their group's mean or std. ``normalize_std=False`` gives (r - mean).
    """
    for group in _groups(trajs, key).values():
        valid = [t for t in group if not t.get("infra_error")]
        rs = [t["reward"] for t in valid]
        mean = sum(rs) / len(rs) if rs else 0.0
        std = (sum((r - mean) ** 2 for r in rs) / len(rs)) ** 0.5 if rs else 0.0
        for t in group:
            if t.get("infra_error"):
                t["advantage"] = 0.0
            else:
                d = t["reward"] - mean
                t["advantage"] = d / (std + eps) if normalize_std else d
    return trajs


def filter_groups(trajs: list[dict], key: str = "task_id", min_valid: int = 2) -> tuple[list[dict], dict]:
    """Drop groups that carry no gradient: every valid reward identical, or
    fewer than ``min_valid`` valid rollouts. Returns (kept, stats)."""
    kept: list[dict] = []
    stats = {"groups": 0, "kept_groups": 0, "uniform_groups": 0, "too_few_valid": 0,
             "infra_rollouts": sum(1 for t in trajs if t.get("infra_error"))}
    for group in _groups(trajs, key).values():
        stats["groups"] += 1
        valid = [t for t in group if not t.get("infra_error")]
        if len(valid) < min_valid:
            stats["too_few_valid"] += 1
            continue
        rs = [t["reward"] for t in valid]
        if max(rs) - min(rs) < 1e-9:
            stats["uniform_groups"] += 1
            continue
        stats["kept_groups"] += 1
        kept.extend(valid)
    return kept, stats
