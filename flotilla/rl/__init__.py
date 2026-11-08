"""RL building blocks on top of swarms and sessions."""

from .code import CodeVerifier, Score, extract_code, trl_reward_func
from .env import SYSTEM_PROMPT, ShellTaskEnv, StepResult, parse_action
from .rollout import OpenAIChatPolicy, run_episode, run_episodes, write_jsonl
from .shaping import (
    filter_groups,
    group_advantages,
    is_degenerate,
    length_penalty,
    ngram_uniqueness,
)

__all__ = [
    "CodeVerifier",
    "Score",
    "extract_code",
    "trl_reward_func",
    "ShellTaskEnv",
    "StepResult",
    "SYSTEM_PROMPT",
    "parse_action",
    "OpenAIChatPolicy",
    "run_episode",
    "run_episodes",
    "group_advantages",
    "length_penalty",
    "filter_groups",
    "is_degenerate",
    "ngram_uniqueness",
    "write_jsonl",
]
