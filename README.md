# flotilla

Self-hosted sandbox swarms and RL environments for LLM training.

flotilla runs one command on many fresh, isolated sandboxes at once and
collects every result: exit code, output tails and a result file. On top of
that it provides what an RL loop for a language model needs:

- **Verifiable code rewards**: each completion runs against tests in its
  own sandbox, and the pass fraction is the reward. A hack-resistant
  input/output mode is included.
- **Multi-turn tool-use environments**: persistent sandboxes where the
  policy runs shell commands over several turns, then a hidden verifier
  scores the final state.
- **Rollout collection** from any OpenAI-compatible server (`vllm serve`,
  llama.cpp, ...), with GRPO group advantages.
- A **TRL `GRPOTrainer` adapter**.

It runs on your own hardware: a workstation, a CPU server next to the GPU
box, or an HPC node. There are no runtime dependencies beyond the Python
standard library and a sandbox tool already on most Linux machines.

```
trainer (GPU)                         sandbox host (CPU, can be the same machine)
  GRPO step                             flotilla serve
    completions ──── HTTP / in-process ──▶ swarm: N fresh sandboxes, concurrency-capped
    rewards     ◀──────────────────────── exit code, stdout/stderr tails, result file
```

## Install

```bash
git clone https://github.com/Melihakbulut221/flotilla && cd flotilla
pip install -e .            # no dependencies
sudo apt install bubblewrap # isolated backend; usually already present
flotilla doctor             # shows which backends work here
```

## Swarms

A swarm is one shell command, run either `count` times or once per entry in
`inputs`. Each sandbox is created fresh. Its input files are written, then the
command runs with these variables set:

- `FLOTILLA_SWARM_INDEX`
- `FLOTILLA_SWARM_COUNT`
- `FLOTILLA_SWARM_ID`

The same variables are also set with a `MARITIME_` prefix, so scripts written
for hosted swarm services run unchanged.

When the command exits, flotilla keeps:

- the exit code
- the last 16 KB of stdout and of stderr
- the file at `resultPath`, up to 64 KB

The sandbox, disk included, is then destroyed.

```python
from flotilla import Flotilla

fl = Flotilla()  # in-process, best available isolated backend
run = fl.swarms.run("python3 eval.py --case $FLOTILLA_SWARM_INDEX", count=50, timeout_seconds=600)
print(run["counts"])  # {'succeeded': 48, 'failed': 2, ...}

run = fl.swarms.run(
    "python3 rollout.py",
    inputs=[
        {"args": ["--seed", "1"]},
        {"args": ["--seed", "2"], "env": {"TEMPERATURE": "0.7"}},
        {"files": {"/work/task.json": '{"id": 3}'}},
    ],
    result_path="/work/result.json",
    size="2x4",          # vCPUs x GB
    concurrency=16,
)
for sb in run["sandboxes"]:
    print(sb["index"], sb["status"], sb["exitCode"], sb["result"])
```

Sandbox states are `queued`, `booting`, `running`, then one of `succeeded`,
`failed` or `cancelled`:

- **succeeded**: exit 0.
- **failed**: a non-zero exit, a timeout or a boot error. The `error` field
  says which.
- **cancelled**: `fl.swarms.cancel(id)` stops running sandboxes within a
  fraction of a second, and queued ones never start.

A run is `completed` (or `cancelled`) once every sandbox has ended.

Concurrency has two limits:

- **Global cap**: `max_concurrency`, the number of sandboxes alive at once
  across all runs. It defaults to half the CPU count.
- **Per-run cap**: `concurrency`, clamped to the global cap.

Each sandbox is pinned to its own `vcpus` host cores when enough are free.
Runs are persisted to `~/.flotilla/runs/<id>/`: `run.json` and, at the end,
`results.jsonl` in input order.

### CLI

```bash
flotilla run --count 50 --command 'python3 eval.py --case $FLOTILLA_SWARM_INDEX' --size 2x4 --timeout 10m -o results.jsonl
flotilla run --inputs inputs.jsonl --command 'python3 rollout.py' --result-path /work/result.json
flotilla list | flotilla get ID | flotilla results ID -o r.jsonl | flotilla cancel ID
```

`flotilla run` exits 0 only if every sandbox succeeded.

### Server mode

Run sandboxes on one machine and drive them from another:

```bash
FLOTILLA_TOKEN=change-me flotilla serve --host 0.0.0.0 --port 8642
```

```python
fl = Flotilla(base_url="http://sandbox-host:8642", token="change-me")
```

| Route | Purpose |
|---|---|
| `POST /api/swarms` | start a run (`Idempotency-Key` header supported) |
| `GET /api/swarms?limit=50` | list runs, newest first |
| `GET /api/swarms/{id}` | run plus every sandbox |
| `POST /api/swarms/{id}/cancel` | cancel |
| `GET /api/swarms/{id}/results` | JSON Lines, input order |
| `POST /api/sessions` | open a persistent sandbox |
| `POST /api/sessions/{id}/exec` | run a command in it |
| `PUT` / `GET /api/sessions/{id}/files` | write or read a file |
| `DELETE /api/sessions/{id}` | destroy it |

Request bodies accept the same fields as the Python API, in either camelCase
or snake_case:

- `command`
- `count` or `inputs[]`, each input with optional `args`, `env` and `files`
- `size {vcpus, memoryGb}`
- `timeoutSeconds`
- `concurrency`
- `resultPath`
- `image`
- `network`
- `env`
- `name`

The server refuses to bind a non-loopback address without a token.

## RL

### Verifiable code reward (single turn, GRPO)

```python
from flotilla import Flotilla
from flotilla.rl import CodeVerifier

verifier = CodeVerifier(Flotilla(), timeout=60, per_test_timeout=5)
task = {"entry_point": "add", "tests": [{"args": [1, 2], "expected": 3}, {"args": [-1, 1], "expected": 0}]}
rewards = verifier.rewards(completions, [task] * len(completions))   # floats in [0, 1]
```

Each completion's last fenced code block runs in its own sandbox. A
completion without a code block gets 0 and costs no sandbox.

**`io` mode** (tests are `args`/`expected` pairs) is built to be hard to game:

- The model's code runs in a child process that only receives the inputs.
- The parent compares the outputs with expected values that the child never
  sees. The task file is deleted before the child starts, and the parent is
  marked non-dumpable so the child cannot read its memory.
- The child's stdout is redirected, so printing a fake result does nothing.
- Exiting early scores 0.
- Objects that compare equal to everything do not survive JSON
  serialization.
- Every process the solution spawned is killed before the result is written.
  The parent is a child subreaper, so a daemonised writer cannot overwrite
  the result afterwards.

`tests/test_rl.py` contains each of these attacks as a test.

**`assert` mode** (tests given as Python source) is supported for
convenience. It runs tests in the same process as the solution, so a policy
determined to cheat can cheat. Prefer `io` mode for training.

With TRL:

```python
from trl import GRPOTrainer
from flotilla.rl import trl_reward_func
trainer = GRPOTrainer(model=..., reward_funcs=[trl_reward_func(verifier)], train_dataset=ds, args=cfg)
```

The dataset needs `tests` and `entry_point` columns.
`examples/train_grpo_trl.py` is a complete script.

### Multi-turn tool use

```python
from flotilla.rl import ShellTaskEnv, OpenAIChatPolicy, run_episodes, group_advantages

policy = OpenAIChatPolicy("http://localhost:8000/v1", "my-model")  # e.g. `vllm serve my-model`
trajs = group_advantages(run_episodes(fl, policy, tasks, samples_per_task=8, parallel=16))
```

Each turn, the policy replies with one of:

- a ` ```bash ` block, which is executed in a persistent sandbox (files survive
  between turns);
- `SUBMIT`, which ends the episode.

At the end, hidden `verify_files` are written into the sandbox and `verify`
runs:

- exit 0 gives reward 1;
- otherwise the reward is 0, unless `verify` prints a
  `FLOTILLA_REWARD=<float>` line, which sets the reward directly.

See `examples/tasks/shell_tasks.jsonl` for the task format and
`examples/shell_agent_rollouts.py --scripted` for a model-free dry run.

## Backends and threat model

| Backend | Isolation | Needs | Notes |
|---|---|---|---|
| `bwrap` (default) | user/pid/net/ipc/uts namespaces, empty tmpfs root, system dirs read-only, host home hidden, no network | `bubblewrap`, unprivileged user namespaces | ~120 ms per sandbox; ~80 sandboxes/s on a 20-core workstation |
| `podman` / `docker` | containers with cgroup CPU/memory/pid limits, any image | a container engine | `image=` selects the toolchain |
| `apptainer` | containers on HPC without root | `apptainer` / `singularity` | state kept in staged host dirs |
| `local` | none | nothing | for development only; never for untrusted model output |

What the `bwrap` backend guarantees is covered by `tests/test_isolation.py`:

- the host home directory, dotfiles and keys are invisible;
- host environment variables do not leak in;
- there is no network unless `network=True`;
- system directories are read-only;
- a symlink planted in the sandbox cannot make the host read or write a host
  file;
- background processes die with the sandbox;
- each sandbox has its own pid namespace.

Its limits:

- **Shared kernel.** It is a namespace sandbox, not a VM, so it shares the
  kernel with the host. That is a sound trade-off for scoring your own
  model's code on your own machine. For hostile multi-tenant code, use a
  VM-based sandbox.
- **Memory.** It is limited with `RLIMIT_AS`; CPU is limited by core
  pinning.
- **Process count.** It is not limited (no cgroups without root). Use the
  container backends if you need hard pid and memory limits.

The Docker, Podman and Apptainer backends are implemented against the
engines' standard CLIs. The test suite here exercises `bwrap` and `local`.

## Layout

```
flotilla/spec.py        request model and validation
flotilla/swarm.py       swarm runner: queue, concurrency, cancel, persistence, idempotency
flotilla/session.py     persistent sandboxes for multi-turn environments
flotilla/backends/      bwrap, docker/podman, apptainer, local
flotilla/server.py      HTTP API (stdlib)
flotilla/client.py      one client for in-process and remote use
flotilla/rl/            code verifier + harness, shell task env, rollouts, TRL adapter
examples/               task sets, eval, GRPO training, rollout collection
tests/                  swarm, isolation, RL reward-hacking and server tests
```

## Tests

```bash
pip install pytest && python3 -m pytest -q
```

## License

Apache-2.0
