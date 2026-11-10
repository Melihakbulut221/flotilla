"""Generate completions for a task file with transformers (no server needed).

    python examples/generate_hf.py --model runs/sft-music/merged --tasks tasks.jsonl --out completions.jsonl

``--subtract-adapter DIR`` rebuilds the base model from a merged one by
subtracting the LoRA update (B @ A x alpha / r) from every adapted weight.
That allows a same-stack base-vs-fine-tuned comparison when only the merged
weights and the adapter are on disk. The only difference from the original
base is bf16 rounding.

Writes one JSON line per task: {"task_id", "completion"}. Score the file with
``eval_music.py --completions``.
"""

import argparse
import json
from pathlib import Path


def subtract_adapter(model, adapter_dir: str) -> int:
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    params = dict(model.named_parameters())
    done = 0
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        b = tensors[key.replace(".lora_A.", ".lora_B.")]
        name = key.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
        w = params[name]
        with torch.no_grad():
            delta = (b.float().to(w.device) @ a.float().to(w.device)) * scale
            w.copy_((w.float() - delta).to(w.dtype))
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subtract-adapter", help="rebuild the base model from merged weights")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", help="system prompt to prepend")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if args.subtract_adapter:
        print(f"subtracted the LoRA update from {subtract_adapter(model, args.subtract_adapter)} weights")

    tasks = [json.loads(line) for line in open(args.tasks) if line.strip()][: args.limit]
    sys_msg = [{"role": "system", "content": args.system}] if args.system else []
    prompts = [tok.apply_chat_template(sys_msg + [{"role": "user", "content": t["prompt"]}], tokenize=False,
                                       add_generation_prompt=True) for t in tasks]
    with open(args.out, "w") as f:
        for i in range(0, len(tasks), args.batch_size):
            batch = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**batch, do_sample=True, temperature=args.temperature,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            for t, text in zip(tasks[i:i + args.batch_size], texts):
                f.write(json.dumps({"task_id": t["id"], "completion": text}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(i + args.batch_size, len(tasks))}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
