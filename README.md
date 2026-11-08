<p align="center">
  <img src="docs/assets/banner.svg" alt="flotilla: self-hosted sandbox swarms and RL environments for LLM training" width="100%">
</p>

<p align="center">
  <a href="https://github.com/Melihakbulut221/flotilla/actions/workflows/test.yml"><img src="https://github.com/Melihakbulut221/flotilla/actions/workflows/test.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/dependencies-none-2dd4bf" alt="no dependencies">
  <img src="https://img.shields.io/badge/isolation-bubblewrap%20%7C%20podman%20%7C%20docker%20%7C%20apptainer-0d2c45" alt="backends">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#swarms">Swarms</a> &middot;
  <a href="#rl-environments">RL environments</a> &middot;
  <a href="#reward-hacking-resistance">Reward hacking</a> &middot;
  <a href="#backends-and-threat-model">Backends</a> &middot;
  <a href="#how-it-compares">Compare</a>
</p>

---

flotilla runs one command on many fresh, isolated sandboxes at once and
collects every result: exit code, output tails and a result file. On top of
that it provides what an RL loop for a language model needs, on your own
hardware, with nothing to install beyond Python and a sandbox tool most Linux
machines already have.

<table>
<tr>
<td width="33%" valign="top">

**Swarms**<br>
One command, `count` copies or one per input. Concurrency caps, timeouts,
cancel, idempotency keys, JSON Lines results in input order.

</td>
<td width="33%" valign="top">

**Verifiable code rewards**<br>
Each completion runs against tests in its own sandbox; the pass fraction
is the reward. Drops into TRL's `GRPOTrainer`.

</td>
<td width="33%" valign="top">

**Multi-turn tool use**<br>
Persistent sandboxes where the policy runs shell commands across turns,
then a hidden verifier scores the final state.

</td>
</tr>
<tr>
<td valign="top">

**Rootless isolation**<br>
Namespaces via bubblewrap: no network, host home and env hidden, read-only
system dirs. Containers and Apptainer for cgroups or HPC.

</td>
<td valign="top">

**Hack-resistant judging**<br>
The judge never shares a process with model code. Five classic reward
hacks are part of the test suite and all score 0.

</td>
<td valign="top">

**Local or remote**<br>
In-process for one box, or `flotilla serve` on a CPU host and drive it over
HTTP from the GPU trainer.

</td>
</tr>
</table>

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture: trainer sends completions to flotilla, which runs them in fresh sandboxes and returns rewards" width="100%">
</p>

## Quickstart

```bash
git clone https://github.com/Melihakbulut221/flotilla && cd flotilla
pip install -e .            # no dependencies
sudo apt install bubblewrap # isolated backend; usually already present
flotilla doctor             # shows which backends work here
```

<p align="center">
  <img src="docs/assets/terminal.svg" alt="Terminal session: flotilla doctor, a 64-sandbox swarm and the example task check" width="100%">
</p>

<sub>Captured output, condensed where marked with <code>...</code>. The swarm ran
64 sandboxes on the bwrap backend at concurrency 10; sandbox 17 fails on purpose.</sub>

## Swarms

A swarm is one shell command, run either `count` times or once per entry in
`inputs`. Each sandbox is created fresh, its input files are written, and the
command runs with `FLOTILLA_SWARM_INDEX`, `FLOTILLA_SWARM_COUNT` and
`FLOTILLA_SWARM_ID` set. The same variables are also exported with a
`MARITIME_` prefix, so scripts written for hosted swarm services run
unchanged. When the command exits, flotilla keeps the exit code, the last
16 KB of stdout and stderr, and the file at `resultPath` (up to 64 KB). The
sandbox, disk included, is then destroyed.

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

```
queued ──▶ booting ──▶ running ──┬──▶ succeeded   exit 0
                                 ├──▶ failed      non-zero exit, timeout or boot error (see "error")
                                 └──▶ cancelled   fl.swarms.cancel(id)
```

A run is `completed` (or `cancelled`) once every sandbox has ended.

- **Global cap:** `max_concurrency`, the number of sandboxes alive at once
  across all runs. It defaults to half the CPU count.
- **Per-run cap:** `concurrency`, clamped to the global cap.
- **CPU pinning:** each sandbox is pinned to its own `vcpus` cores while
  enough are free.
- **Persistence:** runs are saved under `~/.flotilla/runs/<id>/`.

<details>
<summary><b>CLI</b></summary>

```bash
flotilla run --count 50 --command 'python3 eval.py --case $FLOTILLA_SWARM_INDEX' --size 2x4 --timeout 10m -o results.jsonl
flotilla run --inputs inputs.jsonl --command 'python3 rollout.py' --result-path /work/result.json
flotilla list | flotilla get ID | flotilla results ID -o r.jsonl | flotilla cancel ID
```

`flotilla run` exits 0 only if every sandbox succeeded. See
[`examples/swarm_basics.sh`](examples/swarm_basics.sh).

</details>

<details>
<summary><b>Server mode and HTTP API</b></summary>

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

Request bodies accept these fields, in camelCase or snake_case:

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

</details>

## RL environments

### Verifiable code reward (single turn, GRPO)

```python
from flotilla import Flotilla
from flotilla.rl import CodeVerifier

verifier = CodeVerifier(Flotilla(), timeout=60, per_test_timeout=5)
task = {"entry_point": "add", "tests": [{"args": [1, 2], "expected": 3}, {"args": [-1, 1], "expected": 0}]}
rewards = verifier.rewards(completions, [task] * len(completions))   # floats in [0, 1]
```

The last fenced code block of each completion runs in its own sandbox. A
completion without a code block gets 0 and costs no sandbox. With TRL:

```python
from trl import GRPOTrainer
from flotilla.rl import trl_reward_func

trainer = GRPOTrainer(model=..., reward_funcs=[trl_reward_func(verifier)], train_dataset=ds, args=cfg)
```

The dataset needs `tests` and `entry_point` columns.
[`examples/train_grpo_trl.py`](examples/train_grpo_trl.py) is a complete
script, and [`examples/eval_code.py`](examples/eval_code.py) scores any
OpenAI-compatible endpoint on the example task set.

### Multi-turn tool use

```python
from flotilla.rl import OpenAIChatPolicy, group_advantages, run_episodes

policy = OpenAIChatPolicy("http://localhost:8000/v1", "my-model")  # e.g. `vllm serve my-model`
trajs = group_advantages(run_episodes(fl, policy, tasks, samples_per_task=8, parallel=16))
```

```
system   You are working in a Linux sandbox ... reply with a ```bash``` block or SUBMIT.
user     /work/stats.py should print the sum of 1..100 but prints the wrong value. Fix it.
assistant  ```bash
           cat stats.py; python3 stats.py
           ```
user     total = 0 / for i in range(1, 100): ... / 4950  [exit code 0]
assistant  ```bash
           sed -i 's/range(1, 100)/range(1, 101)/' stats.py && python3 stats.py
           ```
user     5050  [exit code 0]
assistant  SUBMIT                                   -> hidden verifier runs -> reward 1.0
```

Files persist between turns. At the end, hidden `verify_files` are written
and `verify` runs:

- exit 0 gives reward 1, anything else gives 0;
- a `FLOTILLA_REWARD=<float>` line in its output gives partial credit instead.

See [`examples/tasks/shell_tasks.jsonl`](examples/tasks/shell_tasks.jsonl) for
the task format, and run
[`examples/shell_agent_rollouts.py`](examples/shell_agent_rollouts.py)
`--scripted` for a dry run that needs no model.

## Reward hacking resistance

A policy under RL pressure will find any shortcut to a high reward. In `io`
mode (tests given as `args`/`expected` pairs), the model's code runs in a
child process that only receives the inputs. The judge is a separate
process that holds the expected values.

| Attack in the submitted code | What stops it | Reward |
|---|---|:-:|
| `os._exit(0)` before any test runs | pass/fail comes from compared outputs, not exit codes | 0 |
| print a fake "all passed" result | the solution's stdout is redirected away from the protocol channel | 0 |
| read the expected answers from disk | the task file is deleted before the child starts; the judge is non-dumpable | 0 |
| return an object whose `__eq__` is always true | outputs are JSON-serialised before comparison | 0 |
| fork a daemon that overwrites the result file later | the judge is a child subreaper and kills every descendant before writing | 0 |

Each row is a test in [`tests/test_rl.py`](tests/test_rl.py). The last one
was confirmed to score 1.0 with the protection switched off, so the test
proves the protection works. `assert` mode (tests as Python source) is also
supported for convenience, but it shares a process with the solution. Prefer
`io` mode for training.

## Backends and threat model

| Backend | Isolation | Needs | Notes |
|---|---|---|---|
| `bwrap` (default) | user/pid/net/ipc/uts namespaces, empty tmpfs root, read-only system dirs, host home hidden, no network | `bubblewrap`, unprivileged user namespaces | ~120 ms per sandbox, ~80 sandboxes/s on a 20-core workstation |
| `podman` / `docker` | containers with cgroup CPU, memory and pid limits; any image | a container engine | `image=` picks the toolchain |
| `apptainer` | containers on HPC clusters without root | `apptainer` / `singularity` | state kept in staged host dirs |
| `local` | none | nothing | development only, never for untrusted model output |

[`tests/test_isolation.py`](tests/test_isolation.py) checks what the `bwrap`
backend guarantees:

- the host home directory, dotfiles and keys are invisible;
- host environment variables do not leak in;
- there is no network unless you ask for it;
- system dirs are read-only;
- a symlink planted in the sandbox cannot make the host read or write host
  files;
- background processes die with the sandbox;
- each sandbox has its own pid namespace.

The `bwrap` backend has three limits:

- It shares the kernel with the host, because it is a namespace sandbox and
  not a VM. That is a sound trade-off for scoring your own model's code on
  your own machine. For hostile multi-tenant code, use a VM-based sandbox.
- Memory is capped with `RLIMIT_AS` and CPU by core pinning.
- The process count is not capped, since cgroups need root.

The Docker, Podman and Apptainer backends use the engines' standard CLIs.
CI covers `bwrap` and `local`.

## How it compares

|  | Hosted micro-VM swarm services | flotilla |
|---|---|---|
| Isolation | a VM per sandbox (stronger) | namespaces or containers |
| Cost | billed per vCPU-second and GB-second | your own hardware |
| Parallelism | set by the plan | set by your cores |
| Per-sandbox lifecycle | one command, then destroyed | one command (swarms) or many turns (sessions) |
| Rewards and verifiers | left to your command | code verifier, shell task env, TRL adapter |
| Setup | an API key | `pip install -e .` |

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
tests/                  swarm, isolation, reward-hacking and server tests
```

## Tests

```bash
pip install pytest && python3 -m pytest -q
```

## License

[Apache-2.0](LICENSE)
