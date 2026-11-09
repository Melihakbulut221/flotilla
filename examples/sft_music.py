"""LoRA SFT warm-up on the music set, before GRPO with the music reward.

Needs ``pip install torch transformers "trl>=0.20" peft datasets``. Sized
for one 8 GB GPU: a 1.5B model in bf16 with LoRA, gradient checkpointing
and sequences up to 4096 tokens. The data is fed as conversational
prompt/completion pairs, so the loss covers the assistant's ABC only.
Written against the TRL 0.20 API; not run in this repository's CI.

    python examples/build_music_sft.py --download
    python examples/sft_music.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 2

Then measure it the same way as the baseline. Serve the merged model with
vLLM and run::

    python examples/eval_music.py --tasks runs/mimo-music-baseline/tasks.jsonl \\
        --base-url http://localhost:8000/v1 --model runs/sft-music/merged
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/music_sft")
    ap.add_argument("--output-dir", default="runs/sft-music")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--no-synthetic", action="store_true", help="train on the real tunes only")
    ap.add_argument("--no-merge", action="store_true", help="keep only the LoRA adapter")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    files = {"train": str(Path(args.data) / "train.jsonl"), "validation": str(Path(args.data) / "val.jsonl")}
    ds = load_dataset("json", data_files=files)
    if args.no_synthetic:
        ds = ds.filter(lambda r: r["source"] != "synthetic")
    # conversational prompt/completion: TRL computes the loss on the completion only,
    # which works with any chat template (assistant_only_loss needs template markers)
    ds = ds.map(lambda r: {"prompt": r["messages"][:1], "completion": r["messages"][1:]},
                remove_columns=ds["train"].column_names)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        max_length=args.max_length,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    )
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["validation"],
                         peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.output_dir)
    if not args.no_merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(args.output_dir) / "merged")
        trainer.processing_class.save_pretrained(Path(args.output_dir) / "merged")


if __name__ == "__main__":
    main()
