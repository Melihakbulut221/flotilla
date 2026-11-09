"""verl data format: export flotilla tasks as verl RL rows, import verl and
MiMo rows as flotilla tasks, and score verl rollouts with flotilla rewards.

A verl RL row (one per prompt, usually stored as Parquet)::

    {"data_source": "flotilla/code",
     "prompt": [{"role": "user", "content": "..."}],
     "ability": "code",
     "reward_model": {"style": "rule", "ground_truth": "<JSON string>"},
     "extra_info": {"index": 0, "split": "train", "task_id": "...", "kind": "code",
                    "task_json": "<the whole flotilla task as JSON>"}}

``ground_truth`` is always a JSON string, so every row of a file has the same
Parquet schema. Its content depends on the kind:

- ``code``: ``{"entry_point", "tests"}``;
- ``music``: the constraints;
- ``shell`` and ``tool``: empty, because those tasks are graded by their
  environment.

Scoring inside verl. Point the trainer at this file::

    custom_reward_function.path=$(python -c "import flotilla.verl as v; print(v.__file__)")
    custom_reward_function.name=compute_score            # per-sample (naive manager)
    # or, to run every code completion of a batch in one swarm:
    reward_model.reward_manager=batch custom_reward_function.name=compute_score_batch

Rows whose ``data_source`` is not ``flotilla/*`` fall back to verl's
``default_compute_score`` when verl is installed.

Importing MiMo-V2.6-RL-oss rows (``import_rows``):

- ``music``: becomes a flotilla music task. Tempo, meter and voice count come
  from ``extra_info``; key and bar count are parsed from the prompt (Chinese
  or English).
- ``mimoagent/terminal_bench`` (the general split): becomes a tool task on
  the row's image. Its ``tests_files`` are written at verification, and the
  reward is read from the ``reward.txt`` that its ``test.sh`` writes.
- ``opensource-code``: becomes a tool task on the row's image. The files
  the hidden test patch touches are restored from the image, then the patch
  is applied and ``test_command`` runs.
- ``arvo`` (cyber) and ``blackbox/webdev``: skipped, with a reason. They
  need an in-image root grading server, or a browser and a vision judge.

Image tasks need a container backend (docker, podman or apptainer). The
mapping from the rows' bare image names to tags in Docker Hub's
``xiaomimimo/mimo-v2.6-rl-oss`` repository is inferred from that
repository's tag list, and is not documented by the dataset.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import threading
from collections import Counter
from pathlib import Path
from typing import Any

KINDS = ("code", "music", "shell", "tool")
MIMO_IMAGE_REPO = "docker.io/xiaomimimo/mimo-v2.6-rl-oss"


class Unsupported(ValueError):
    """A row this module cannot turn into a flotilla task."""


# ------------------------------------------------------------------ export


def to_verl_row(task: dict, kind: str, index: int = 0, split: str = "train") -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if kind == "code":
        tests = task["tests"]
        gt = json.dumps({"entry_point": task.get("entry_point"),
                         "tests": json.loads(tests) if isinstance(tests, str) and tests.lstrip()[:1] == "[" else tests})
    elif kind == "music":
        gt = json.dumps(task["constraints"])
    else:
        gt = ""
    return {
        "data_source": f"flotilla/{kind}",
        "prompt": [{"role": "user", "content": task["prompt"]}],
        "ability": kind,
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {"index": index, "split": split, "task_id": str(task.get("id", index)),
                       "kind": kind, "task_json": json.dumps(task)},
    }


def to_verl_rows(tasks: list[dict], kind: str, split: str = "train") -> list[dict]:
    return [to_verl_row(t, kind, i, split) for i, t in enumerate(tasks)]


# ------------------------------------------------------------------ import


def mimo_image(name: str, repo: str = MIMO_IMAGE_REPO, apptainer: bool = False) -> str:
    """Map a MiMo row's bare image name to a Docker Hub reference (inferred)."""
    base, _, tag = (name or "").partition(":")
    if base.startswith("format-code-task-"):
        t = base
    elif base == "arvo-rl" and tag.startswith("v1-arvo-"):
        t = "arvo-v1-" + tag[len("v1-arvo-"):]
    elif base.startswith("general-agent-env-"):
        t = base
    elif base.startswith("webdev-rl-opensource"):
        t = "webdev-rl-opensource"
    else:
        return name
    ref = f"{repo}:{t}"
    return f"docker://{ref}" if apptainer else ref


_ZH_KEY = re.compile(r"(升|降)?\s*([A-G])\s*([#b♯♭])?\s*(大调|小调)")
_EN_KEY = re.compile(r"\b([A-G])(?:\s*(#|b|♯|♭|-flat|-sharp| flat| sharp))?\s+"
                     r"(major|minor|dorian|mixolydian|lydian|phrygian|locrian|aeolian|ionian)\b", re.I)
_BARS = re.compile(r"(\d+)\s*(?:小节|bars?\b|measures?\b)", re.I)


def _acc(sym: str | None) -> str:
    s = (sym or "").strip().lower()
    if s in ("#", "♯", "-sharp", "sharp", "升"):
        return "#"
    if s in ("b", "♭", "-flat", "flat", "降"):
        return "b"
    return ""


def music_constraints_from_prompt(text: str) -> dict:
    out: dict = {}
    m = _ZH_KEY.search(text)
    if m:
        out["key"] = f"{m.group(2)}{_acc(m.group(1) or m.group(3))}" + ("m" if m.group(4) == "小调" else "")
    else:
        m = _EN_KEY.search(text)
        if m:
            mode = m.group(3).lower()
            suffix = {"major": "", "minor": "m", "ionian": "", "aeolian": "m"}.get(mode, f" {mode}")
            out["key"] = f"{m.group(1).upper()}{_acc(m.group(2))}{suffix}"
    m = _BARS.search(text)
    if m:
        out["bars"] = int(m.group(1))
    return out


def _prompt_text(row: dict) -> str:
    p = row.get("prompt")
    if isinstance(p, str):
        return p
    users = [m.get("content", "") for m in (p or []) if m.get("role", "user") == "user"]
    return users[-1] if users else ""


def _instance(row: dict) -> dict:
    ij = (row.get("extra_info") or {}).get("instance_json")
    if not ij:
        raise Unsupported("row has no extra_info.instance_json")
    return json.loads(ij) if isinstance(ij, str) else ij


def _size(ij: dict) -> dict | None:
    if ij.get("cpus") or ij.get("memory_mb"):
        return {"vcpus": max(1, int(ij.get("cpus") or 1)),
                "memoryGb": max(1.0, float(ij.get("memory_mb") or 2048) / 1024)}
    return None


def _from_mimo_music(row: dict) -> tuple[str, dict]:
    prompt = _prompt_text(row)
    ex = row.get("extra_info") or {}
    c = music_constraints_from_prompt(prompt)
    if ex.get("meter"):
        c["meter"] = str(ex["meter"])
    if ex.get("nvoice_want"):
        c["voices"] = int(ex["nvoice_want"])
    notes = []
    if ex.get("bpm"):
        num, _, den = c.get("meter", "4/4").partition("/")
        if den == "8" and num.isdigit() and int(num) % 3 == 0 and int(num) > 3:
            notes.append("tempo dropped: BPM beat unit is ambiguous in compound meter")
        else:
            c["tempo"] = float(ex["bpm"])
    if not c:
        raise Unsupported("no constraints found in the music row")
    task = {"id": f"mimo-music-{ex.get('src_id', ex.get('index', ''))}", "prompt": prompt, "constraints": c}
    if notes:
        task["import_notes"] = notes
    return "music", task


def _from_mimo_terminal_bench(row: dict, repo: str, apptainer: bool) -> tuple[str, dict]:
    ij = _instance(row)
    files = ij.get("tests_files") or {}
    files = json.loads(files) if isinstance(files, str) else files
    verify_files: dict = {}
    for name, b64 in files.items():
        try:
            raw = base64.b64decode(b64)
        except ValueError as e:
            raise Unsupported(f"tests file {name} is not valid base64") from e
        try:
            verify_files[f"/tests/{name}"] = raw.decode("utf-8")
        except UnicodeDecodeError:
            verify_files[f"/tests/{name}"] = {"base64": b64}      # binary or non-UTF-8 fixture
    if "/tests/test.sh" not in verify_files:
        raise Unsupported("terminal_bench row without tests/test.sh")
    verify = ("mkdir -p /logs/verifier && sh /tests/test.sh > /tmp/flotilla_verify.log 2>&1; "
              "tail -c 1500 /tmp/flotilla_verify.log; "
              "r=$(head -c 32 /logs/verifier/reward.txt 2>/dev/null | tr -dc '0-9.'); "
              'echo; echo "FLOTILLA_REWARD=${r:-0}"')
    task = {
        "id": f"mimo-general-{ij.get('instance_id', '')}",
        "prompt": ij.get("problem_statement") or _prompt_text(row),
        "image": mimo_image(ij.get("docker_image", ""), repo, apptainer),
        "workdir": ij.get("cwd") or "/app",
        "files": {},
        "verify_files": verify_files,
        "verify": verify,
        "verify_timeout": float(ij.get("verifier_timeout_sec") or 120),
    }
    size = _size(ij)
    if size:
        task["size"] = size
    return "tool", task


_PATCH_PATHS = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)


def _from_mimo_code(row: dict, repo: str, apptainer: bool) -> tuple[str, dict]:
    ij = _instance(row)
    patch, test_cmd = ij.get("test_patch") or "", ij.get("test_command") or ""
    if not patch or not test_cmd:
        raise Unsupported("opensource-code row without test_patch or test_command")
    paths = sorted({p for pair in _PATCH_PATHS.findall(patch) for p in pair if p != "/dev/null"})
    quoted = " ".join(shlex.quote(p) for p in paths)
    wd = ij.get("cwd") or "/testbed"
    # in the grader, before the agent's tree replaces the image's: keep pristine copies
    prepare = (f"mkdir -p /tmp/flotilla/pristine; cd {shlex.quote(wd)} 2>/dev/null || exit 0; for f in {quoted}; do "
               'if [ -e "$f" ]; then mkdir -p "/tmp/flotilla/pristine/$(dirname "$f")" && '
               'cp -p "$f" "/tmp/flotilla/pristine/$f"; fi; done; true')
    # restore those files, apply the hidden tests, run them; the last line is ours, not the tests'
    verify = (f"for f in {quoted}; do rm -rf \"$f\"; if [ -e \"/tmp/flotilla/pristine/$f\" ]; then "
              'mkdir -p "$(dirname "$f")" && cp -p "/tmp/flotilla/pristine/$f" "$f"; fi; done && '
              "git apply --whitespace=nowarn /tmp/flotilla/test.patch && "
              f"( {test_cmd} ) > /tmp/flotilla/test.log 2>&1; s=$?; tail -c 1500 /tmp/flotilla/test.log; "
              'echo; echo "flotilla: tests exited $s"; exit $s')
    task = {
        "id": f"mimo-code-{ij.get('instance_id', '')}",
        "prompt": ij.get("problem_statement") or _prompt_text(row),
        "image": mimo_image(ij.get("docker_image", ""), repo, apptainer),
        "workdir": wd,
        "files": {},
        "verify_prepare": prepare,
        "verify_files": {"/tmp/flotilla/test.patch": patch},
        "verify": verify,
        "verify_timeout": float(ij.get("verifier_timeout_sec") or 1800),
    }
    return "tool", task


def from_verl_row(row: dict, *, image_repo: str = MIMO_IMAGE_REPO, apptainer: bool = False) -> tuple[str, dict]:
    """-> (kind, flotilla task). Raises ``Unsupported`` with a reason."""
    src = str(row.get("data_source") or "")
    ex = row.get("extra_info") or {}
    if src.startswith("flotilla/"):
        return ex.get("kind") or src.split("/", 1)[1], json.loads(ex["task_json"])
    if src == "music":
        return _from_mimo_music(row)
    if src == "mimoagent/terminal_bench" or ex.get("dataset_type") == "terminal_bench":
        return _from_mimo_terminal_bench(row, image_repo, apptainer)
    if src == "opensource-code" or ex.get("dataset_type") == "opensource-code":
        return _from_mimo_code(row, image_repo, apptainer)
    if src == "arvo":
        raise Unsupported("arvo (cyber) needs an in-image root grading server")
    if "webdev" in src:
        raise Unsupported("webdev needs a browser renderer and a vision judge")
    raise Unsupported(f"unknown data_source {src!r}")


def import_rows(rows: list[dict], **kw) -> tuple[dict[str, list[dict]], Counter]:
    """Convert many rows. Returns ({kind: [task, ...]}, Counter of skip reasons)."""
    out: dict[str, list[dict]] = {}
    skipped: Counter = Counter()
    for row in rows:
        try:
            kind, task = from_verl_row(row, **kw)
        except Unsupported as e:
            skipped[str(e)] += 1
            continue
        except (KeyError, ValueError, TypeError) as e:
            skipped[f"malformed row ({type(e).__name__}: {e})"[:120]] += 1
            continue
        out.setdefault(kind, []).append(task)
    return out, skipped


# ------------------------------------------------------------------ files


def read_rows(path: str | Path) -> list[dict]:
    """.parquet (needs pyarrow), .jsonl, or .json (a list, or a Hugging Face
    datasets-server dump with {"rows": [{"row": ...}]})."""
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError("reading Parquet needs pyarrow: pip install pyarrow") from e
        return pq.read_table(path).to_pylist()
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "rows" in data:
        return [r.get("row", r) for r in data["rows"]]
    return data if isinstance(data, list) else [data]


def write_rows(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError("writing Parquet needs pyarrow (pip install pyarrow); or use .jsonl") from e
        pq.write_table(pa.Table.from_pylist(rows), path)
        return
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------ verl reward functions

_lock = threading.Lock()
_state: dict = {}


def _code_verifier():
    with _lock:
        if "code" not in _state:
            from .client import Flotilla
            from .rl.code import CodeVerifier

            _state["code"] = CodeVerifier(Flotilla(), timeout=60, per_test_timeout=5)
        return _state["code"]


def _music_reward():
    with _lock:
        if "music" not in _state:
            from .rl.music import MusicReward

            _state["music"] = MusicReward()
        return _state["music"]


def _fallback(data_source, solution_str, ground_truth, extra_info):
    try:
        from verl.utils.reward_score import default_compute_score
    except ImportError as e:
        raise ValueError(f"no flotilla scorer for data_source {data_source!r} and verl is not importable") from e
    return default_compute_score(data_source, solution_str, ground_truth, extra_info)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None, **_) -> list:
    """verl batch reward manager entry point. Code completions of the whole
    batch run in one swarm; every flotilla result is ``{"score", "infra"}``."""
    n = len(solution_strs)
    extra_infos = list(extra_infos) if extra_infos is not None else [None] * n
    results: list[Any] = [None] * n
    code_idx = [i for i in range(n) if data_sources[i] == "flotilla/code"]
    music_idx = [i for i in range(n) if data_sources[i] == "flotilla/music"]
    if code_idx:
        tasks = []
        for i in code_idx:
            gt = json.loads(ground_truths[i])
            tasks.append({"entry_point": gt.get("entry_point"), "tests": gt["tests"]})
        scores = _code_verifier().score([solution_strs[i] for i in code_idx], tasks)
        for i, s in zip(code_idx, scores):
            results[i] = {"score": 0.0 if s.infra else s.reward, "infra": float(s.infra)}
    if music_idx:
        scores = _music_reward().score([solution_strs[i] for i in music_idx],
                                       [{"constraints": json.loads(ground_truths[i])} for i in music_idx])
        for i, s in zip(music_idx, scores):
            results[i] = {"score": s.reward, "infra": 0.0}
    for i in range(n):
        if results[i] is None:
            results[i] = _fallback(data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i])
    return results


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_):
    """verl per-sample entry point (naive / prime reward managers)."""
    return compute_score_batch([data_source], [solution_str], [ground_truth], [extra_info])[0]
