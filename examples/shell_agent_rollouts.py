"""Collect multi-turn tool-use rollouts in persistent sandboxes.

    # dry run with a scripted policy that solves fix-off-by-one (no model needed)
    python examples/shell_agent_rollouts.py --scripted

    # your model behind an OpenAI-compatible server, 8 samples per task (a GRPO group)
    python examples/shell_agent_rollouts.py --base-url http://localhost:8000/v1 --model my-model \\
        --samples 8 --out rollouts.jsonl

Each output line is one episode: messages, reward, turns, verifier info and
a group-normalised advantage, ready for rejection sampling or GRPO.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flotilla import Flotilla  # noqa: E402
from flotilla.rl import OpenAIChatPolicy, group_advantages, run_episodes, write_jsonl  # noqa: E402


def scripted_policy(messages):
    """Solves fix-off-by-one; does nothing useful on other tasks."""
    n_actions = sum(m["role"] == "assistant" for m in messages)
    task = messages[1]["content"]
    if "stats.py" not in task:
        return "SUBMIT"
    return ["```bash\ncat stats.py; python3 stats.py\n```",
            "```bash\nsed -i 's/range(1, 100)/range(1, 101)/' stats.py && python3 stats.py\n```",
            "SUBMIT"][min(n_actions, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks" / "shell_tasks.jsonl"))
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--out", default="rollouts.jsonl")
    ap.add_argument("--url", help="flotilla server; default runs sandboxes in-process")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
    if args.scripted:
        policy = scripted_policy
    elif args.base_url and args.model:
        policy = OpenAIChatPolicy(args.base_url, args.model, api_key=args.api_key)
    else:
        ap.error("use --scripted or give --base-url and --model")

    fl = Flotilla(base_url=args.url)
    trajs = group_advantages(run_episodes(fl, policy, tasks, samples_per_task=args.samples,
                                          parallel=args.parallel))
    write_jsonl(args.out, trajs)
    for t in trajs:
        err = t["info"].get("error")
        print(f"{t['task_id']:16s} sample {t['sample']}  reward {t['reward']:.2f}  turns {t['turns']}"
              + (f"  error: {err}" if err else ""))
    print(f"wrote {len(trajs)} episodes to {args.out}")


if __name__ == "__main__":
    main()
