# Music SFT set

A supervised warm-up set for the music environment (`flotilla/rl/music.py`).
Every example pairs a prompt that states the RL constraints (key, meter, tempo,
bars, voices) with an ABC answer that the RL reward scores at full adherence.

## Why it exists

Measured with `examples/eval_music.py` on 200 randomly sampled rows of the MiMo
music split. The model is Qwen2.5-1.5B-Instruct served by Ollama, sampled at
temperature 0.8 with a 4096-token limit.

| Setup | Mean reward | Non-zero | Valid ABC header |
|---|:-:|:-:|:-:|
| MiMo prompt as is (200 rows) | 0.000 | 0 / 200 | 0 |
| same, plus an ABC format system prompt (40 rows) | 0.016 | 7 / 40 | 19 / 40 |

Even with the format prompt, meter and bar-count adherence stayed at 0. The
model writes a header and then junk. GRPO from this start gets almost no
signal, so the model first has to learn to write ABC at all.

## Build

```bash
python examples/build_music_sft.py --download    # Nottingham ABC, pinned commit
python examples/build_music_sft.py               # -> data/music_sft/{train,val}.jsonl, stats.json
```

The build is deterministic for a given `--seed`: a fresh download reproduces
`train.jsonl` byte for byte.

## Contents (seed 0)

| | |
|---|---|
| source tunes | 1034 (Nottingham Music Database, cleaned ABC) |
| tunes used | 871 |
| rejected | 97 key or meter change mid-tune; 61 irregular bars; 54 variants whose notes do not fit the labelled key well enough for full key credit |
| examples | 6060 (train 5747, val 313; split by tune) |
| by source | Nottingham 5151, synthetic 909 |
| by language | English 3044, Chinese 3016 |
| by voices | 1: 2707, 2: 2674, 3: 222, 4: 243, 5: 97, 6: 117 |
| by meter | 4/4 2853, 6/8 1952, 3/4 450, 2/4 364, 9/8 231, 7/8 159, 2/2 27, 3/2 18, 6/4 6 |
| keys | 30 distinct |
| mean reward | 0.980 (adherence 1.0 for every example) |
| longest example | about 3200 characters, well under a 4096-token limit |

Each real tune yields up to six examples:

- **Keys:** the original key plus two random transpositions within the same
  mode, kept in a playable register.
- **Voices:** each key as melody only, and as melody plus a bass voice taken
  from the chord symbols (a slash chord's bass note wins).

Repeats are written out once. The two partial bars at a repeat boundary are
joined into one bar. Tempo is drawn from a range typical of the tune type,
with dotted-quarter beats for compound meters.

Synthetic examples come from `reference_piece()`. They cover what the
collection lacks: three to six voices, 7/8 and 9/8, up to 96 bars, and
modal keys. They are filtered by the same musicality floor as the real
tunes. Drop them with `--no-synthetic` in `sft_music.py`, or set
`--synthetic-fraction 0`.

Prompts mirror the MiMo music split:

- English or Chinese, three phrasings each;
- half of them add an explicit ABC format hint;
- the key and bar count can be parsed back from every prompt (a test checks
  this).

## Format

```json
{"id": "nottingham:reelsd-g:X35:C-bass", "group": "nottingham:reelsd-g:X35",
 "source": "nottingham", "license": "GPL-3.0", "lang": "en",
 "constraints": {"key": "C", "meter": "4/4", "tempo": 124, "bars": 32, "voices": 2},
 "reward": 0.9728,
 "messages": [{"role": "user", "content": "I need a reel: key of C major, meter 4/4, ..."},
              {"role": "assistant", "content": "```abc\nX:1\nT:Eyes of Blue\nM:4/4\n..."}]}
```

`examples/sft_music.py` trains a LoRA on it with TRL. It uses the
prompt/completion format, so the loss covers the ABC only.

## Results

### Training

- **Hardware:** RTX 4060 Laptop, 8 GB.
- **Software:** transformers 5.22, TRL 1.17, PEFT 0.21.
- **Setup:** LoRA r=32 on all attention and MLP projections; 2 epochs, 720
  steps at an effective batch of 16; learning rate 2e-4 with cosine decay.
  About 3.6 hours; peak GPU memory 4.6 GB.

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 720 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| eval loss | 0.877 | 0.826 | **0.808** | **0.808** | 0.820 | 0.825 | 0.829 |

Train loss went from 1.46 to 0.52, and eval token accuracy settled at 0.80.
Eval loss was lowest at steps 300-400 and rose slightly after, so one epoch
is probably enough.

Three things the first run on this GPU surfaced, all fixed in the script:

- TRL loads a model given by name in **float32** unless `model_init_kwargs`
  sets a dtype. For 1.5B parameters that is 6 GB, and the run ran out of
  memory on the first steps.
- The 152k-token vocabulary makes the logits the largest tensor, so the
  script uses micro-batches of one, with 16 steps accumulated.
- transformers 5 removed `warmup_ratio`; `warmup_steps` now takes the ratio.

### Evaluation

Measured on the same 200 MiMo music rows as the baseline, at temperature 0.8
with a 4096-token limit, and scored with `MusicReward`:

- **Same stack for both models.** Both are generated with
  `examples/generate_hf.py` in bf16.
- **How the base model was rebuilt.** The base model's weights were not kept
  on disk. They were rebuilt from the merged model by subtracting the LoRA
  update (`--subtract-adapter`). The re-merge error is below bf16 rounding.
- **Quantization does not matter here.** The bf16 base matches the Ollama Q4
  baseline above.

| Model | Prompt | n | Mean reward | Valid ABC | key | meter | tempo | bars | voices | musicality | zh | en |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| base | MiMo prompt | 40 | 0.000 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.36 | 0.00 | 0.00 |
| base | + format system prompt | 40 | 0.021 | 10 | 0.60 | 0.00 | 0.40 | 0.00 | 0.56 | 0.49 | 0.01 | 0.03 |
| **SFT** | MiMo prompt | **200** | **0.708** | **197** | 0.99 | 0.87 | 1.00 | 0.50 | 0.70 | 0.79 | 0.89 | 0.53 |
| SFT | MiMo prompt, same 40 | 40 | 0.665 | 40 | 0.95 | 0.85 | 1.00 | 0.53 | 0.61 | 0.71 | 0.84 | 0.54 |
| SFT | + format system prompt | 40 | 0.601 | 40 | 0.97 | 0.25 | 1.00 | 0.15 | 0.92 | 0.86 | 0.67 | 0.55 |

The adherence and musicality columns average over valid completions only.

### What it shows

- **The format problem is solved.** Valid ABC went from 0 to 197 of 200, key
  and tempo adherence are nearly perfect, and meter fit is 0.87. GRPO now has
  a dense signal to work with.
- **English prompts are out of distribution.** They score 0.53, against 0.89
  for Chinese. MiMo's English prompts are long (median 736 characters) and
  describe orchestration, texture, sections and dynamics. The SFT prompts are
  short, like the Chinese ones (median 155). On the long prompts the model
  often writes a single voice (voice adherence 0.42) and misses the bar count
  (0.21).
- **The model copies the synthetic examples.** 145 of the 200 completions are
  titled `T:Reference`, the title the synthetic pieces carry, and reuse their
  mechanical random-walk style, mostly on many-voice requests. They score on
  adherence but are weak music.
- **The format system prompt hurts after SFT.** It drops the score from 0.708
  to 0.601. Meter fit falls to 0.25, apparently because its header template
  steers bar lengths. It does raise voice adherence to 0.92.

### Next steps

1. Fix the data, then retrain for one epoch:
   - give synthetic pieces realistic titles and lower their share;
   - add long, MiMo-style English prompts that describe texture and
     sections;
   - add more real multi-voice material if a usable licensed source turns
     up.
2. Then run GRPO with `MusicReward`, from the SFT model.

## Licenses

- **Nottingham examples** derive from the Nottingham Music Database as cleaned
  in [jukedeck/nottingham-dataset](https://github.com/jukedeck/nottingham-dataset)
  (commit `0992bb6`), which is GPL-3.0. They are GPL-3.0.
- **Synthetic examples** are generated by this repository's code and are
  Apache-2.0.
- **Where the data lives:** the generated files are written to `data/`,
  which is git-ignored, so the Apache-2.0 repository holds only the code
  that builds them. Publish the data separately, for example as a Hugging
  Face dataset, with the GPL-3.0 license for the Nottingham part.
- **Excluded source:** The Session's data dump was considered and excluded.
  Its license forbids processing the material with, or training, large
  language models.
