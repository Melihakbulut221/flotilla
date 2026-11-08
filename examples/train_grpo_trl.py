"""GRPO on verifiable code rewards with TRL, rewards computed in sandboxes.

Needs a GPU and ``pip install trl datasets transformers accelerate``.
Every generated completion is executed against the task's tests in its own
sandbox; the pass fraction is the reward. Nothing here is specific to the
model; pass any causal LM checkpoint with --model.

    python examples/train_grpo_trl.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200

Run the sandboxes on another machine with ``flotilla serve`` there and
``--flotilla-url http://host:8642`` (plus FLOTILLA_TOKEN) here.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flotilla import Flotilla  # noqa: E402
from flotilla.rl import CodeVerifier, trl_reward_func  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks" / "code_tasks.jsonl"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--output-dir", default="runs/grpo-code")
    ap.add_argument("--flotilla-url")
    ap.add_argument("--use-vllm", action="store_true")
    args = ap.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(line) for line in open(args.tasks) if line.strip()]
    ds = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            # TRL forwards extra columns to reward functions as kwargs
            "tests": json.dumps(r["tests"]),
            "entry_point": r["entry_point"],
        }
        for r in rows
    ])

    fl = Flotilla(base_url=args.flotilla_url)
    if not fl.isolated:
        print("warning: sandboxes are not isolated (local backend)", file=sys.stderr)
    verifier = CodeVerifier(fl, timeout=60, per_test_timeout=5, partial_credit=True)

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        use_vllm=args.use_vllm,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[trl_reward_func(verifier)],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()
