"""Score ABC music against prompt constraints (examples/tasks/music_tasks.jsonl).

    # sanity gate: a plain compliant piece for every task must get full adherence
    python examples/eval_music.py --references

    # your model behind any OpenAI-compatible server; also write MIDI files to listen to
    python examples/eval_music.py --base-url http://localhost:8000/v1 --model my-model \\
        --samples 4 --midi-dir midi/

Prints, per task, the reward and its parts: adherence per constraint
(key, meter, tempo, bars, voices) and musicality.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flotilla.rl import OpenAIChatPolicy, reference_piece  # noqa: E402
from flotilla.rl.music import MusicReward, extract_abc, parse_abc, to_midi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks" / "music_tasks.jsonl"))
    ap.add_argument("--references", action="store_true", help="score generated compliant pieces")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, help="only the first N tasks")
    ap.add_argument("--midi-dir", help="write a .mid file per scored completion")
    ap.add_argument("--out", help="write completions and score breakdowns as JSON Lines")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open(args.tasks) if line.strip()][: args.limit]
    if args.references:
        flat = tasks
        completions = [f"```abc\n{reference_piece(t['constraints'], i)}\n```" for i, t in enumerate(tasks)]
    else:
        if not (args.base_url and args.model):
            ap.error("--base-url and --model are required unless --references")
        policy = OpenAIChatPolicy(args.base_url, args.model, api_key=args.api_key,
                                  temperature=args.temperature, max_tokens=args.max_tokens)
        flat = [t for t in tasks for _ in range(args.samples)]
        with ThreadPoolExecutor(16) as pool:
            completions = list(pool.map(lambda t: policy([{"role": "user", "content": t["prompt"]}]), flat))

    scores = MusicReward().score(completions, flat)
    parts = ("key", "meter", "tempo", "bars", "voices")
    print(f"{'task':12s} {'reward':>6s}  " + " ".join(f"{p:>6s}" for p in parts) + "  music  status")
    for t, s in zip(flat, scores):
        cols = " ".join(f"{s.adherence.get(p, float('nan')):6.2f}" for p in parts)
        print(f"{t['id']:12s} {s.reward:6.3f}  {cols}  {s.musicality.get('score', 0):5.2f}  {s.status}")
    n = len(scores)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    print(f"\nmean reward {mean([s.reward for s in scores]):.3f} over {n} samples")
    for p in parts:
        vals = [s.adherence[p] for s in scores if p in s.adherence]
        print(f"  {p:7s} adherence {mean(vals):.3f}")
    print(f"  format failures {sum(s.status in ('format', 'invalid') for s in scores)}/{n}")

    if args.midi_dir:
        d = Path(args.midi_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, (t, c) in enumerate(zip(flat, completions)):
            abc = extract_abc(c)
            if abc:
                (d / f"{t['id']}-{i}.mid").write_bytes(to_midi(parse_abc(abc)))
        print(f"MIDI written to {d}/")
    if args.out:
        with open(args.out, "w") as f:
            for t, c, s in zip(flat, completions, scores):
                f.write(json.dumps({"task_id": t["id"], "completion": c, **s.to_dict()}) + "\n")
    if args.references and min(s.adherence.get("score", 0) for s in scores) < 1.0:
        sys.exit("a reference piece missed full adherence; the scorer or task set is broken")


if __name__ == "__main__":
    main()
