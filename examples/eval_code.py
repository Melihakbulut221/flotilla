"""Score a code model on examples/tasks/code_tasks.jsonl.

    # sanity gate: every reference solution must score 1.0
    python examples/eval_code.py --references

    # your model behind any OpenAI-compatible server (vllm serve, llama.cpp, ...)
    python examples/eval_code.py --base-url http://localhost:8000/v1 --model my-model --samples 4
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flotilla import Flotilla  # noqa: E402
from flotilla.rl import CodeVerifier, OpenAIChatPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks" / "code_tasks.jsonl"))
    ap.add_argument("--references", action="store_true", help="score the reference solutions")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--url", help="flotilla server; default runs sandboxes in-process")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
    fl = Flotilla(base_url=args.url, backend=args.backend)
    verifier = CodeVerifier(fl, timeout=60, per_test_timeout=5)

    if args.references:
        completions = [f"```python\n{t['reference']}```" for t in tasks]
        flat_tasks = tasks
    else:
        if not (args.base_url and args.model):
            ap.error("--base-url and --model are required unless --references")
        policy = OpenAIChatPolicy(args.base_url, args.model, api_key=args.api_key,
                                  temperature=args.temperature)
        flat_tasks = [t for t in tasks for _ in range(args.samples)]
        with ThreadPoolExecutor(16) as pool:
            completions = list(pool.map(lambda t: policy([{"role": "user", "content": t["prompt"]}]),
                                        flat_tasks))

    scores = verifier.score(completions, flat_tasks)
    per_task: dict = {}
    for t, s in zip(flat_tasks, scores):
        per_task.setdefault(t["id"], []).append(s)
    for tid, ss in per_task.items():
        solved = sum(s.reward == 1.0 for s in ss)
        errs = {s.error for s in ss if s.error}
        print(f"{tid:20s} solved {solved}/{len(ss)}  mean reward {sum(s.reward for s in ss) / len(ss):.2f}"
              + (f"  errors: {sorted(errs)[:2]}" if errs else ""))
    pass1 = sum(s.reward == 1.0 for s in scores) / len(scores)
    print(f"\npass@1 = {pass1:.3f} over {len(scores)} samples")
    if args.references and pass1 < 1.0:
        sys.exit("reference solutions must all pass; the task set is broken")


if __name__ == "__main__":
    main()
