"""RL building blocks on top of swarms and sessions."""

from .code import CodeVerifier, Score, extract_code, trl_reward_func
from .env import SYSTEM_PROMPT, ShellTaskEnv, StepResult, parse_action
from .tools import TOOL_SYSTEM_PROMPT, TOOLS, ToolTaskEnv, parse_tool_calls, render_tools_prompt
from .rollout import OpenAIChatPolicy, OpenAIToolPolicy, run_episode, run_episodes, write_jsonl
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
    "ToolTaskEnv",
    "TOOLS",
    "TOOL_SYSTEM_PROMPT",
    "parse_tool_calls",
    "render_tools_prompt",
    "OpenAIChatPolicy",
    "OpenAIToolPolicy",
    "run_episode",
    "run_episodes",
    "group_advantages",
    "length_penalty",
    "filter_groups",
    "is_degenerate",
    "ngram_uniqueness",
    "write_jsonl",
]
