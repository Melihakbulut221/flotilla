import base64
import io
import json
import tarfile

import pytest

from flotilla import verl
from flotilla.rl import ShellTaskEnv, ToolTaskEnv, reference_piece, run_episode, score_abc
from flotilla.rl.env import sanitize_tar


def scripted(replies):
    it = iter(replies)
    return lambda messages: next(it)


def call(name, **args):
    return {"id": f"c_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def msg(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


CODE_TASK = {"id": "add", "prompt": "Write add(a, b).", "entry_point": "add",
             "tests": [{"args": [1, 2], "expected": 3}]}
MUSIC_TASK = {"id": "m1", "prompt": "Compose ...", "constraints": {"key": "Dm", "meter": "6/8", "tempo": 120,
                                                                  "bars": 8, "voices": 1}}
TOOL_TASK = {"id": "t1", "prompt": "Sort.", "files": {"/data/in.txt": "2\n1\n"}, "verify": "true"}


# ------------------------------------------------------------------ export / files


def test_export_rows_have_verl_shape_and_roundtrip(tmp_path):
    for kind, task in (("code", CODE_TASK), ("music", MUSIC_TASK), ("tool", TOOL_TASK)):
        row = verl.to_verl_row(task, kind, index=3)
        assert row["data_source"] == f"flotilla/{kind}" and row["ability"] == kind
        assert row["prompt"] == [{"role": "user", "content": task["prompt"]}]
        assert row["reward_model"]["style"] == "rule" and isinstance(row["reward_model"]["ground_truth"], str)
        assert set(row["extra_info"]) == {"index", "split", "task_id", "kind", "task_json"}
        assert verl.from_verl_row(row) == (kind, task)
    gt = json.loads(verl.to_verl_row(CODE_TASK, "code")["reward_model"]["ground_truth"])
    assert gt == {"entry_point": "add", "tests": CODE_TASK["tests"]}

    rows = verl.to_verl_rows([CODE_TASK, dict(CODE_TASK, id="b")], "code")
    verl.write_rows(rows, tmp_path / "r.jsonl")
    assert verl.read_rows(tmp_path / "r.jsonl") == rows


def test_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    rows = verl.to_verl_rows([MUSIC_TASK, dict(MUSIC_TASK, id="m2")], "music")
    verl.write_rows(rows, tmp_path / "train.parquet")
    back = verl.read_rows(tmp_path / "train.parquet")
    assert [verl.from_verl_row(r) for r in back] == [("music", MUSIC_TASK), ("music", dict(MUSIC_TASK, id="m2"))]


def test_read_datasets_server_dump(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps({"rows": [{"row_idx": 0, "row": {"a": 1}}], "features": []}))
    assert verl.read_rows(tmp_path / "d.json") == [{"a": 1}]


# ------------------------------------------------------------------ MiMo import


@pytest.mark.parametrize("text,want", [
    ("创作一首E大调、129 BPM中板、4/4拍的英式乡村舞曲，篇幅56小节（约两分钟）。", {"key": "E", "bars": 56}),
    ("创作一首E小调、93 BPM行板、4/4拍的琵琶曲，篇幅32小节", {"key": "Em", "bars": 32}),
    ("创作一首降B大调的进行曲，篇幅16小节", {"key": "Bb", "bars": 16}),
    ("创作一首升F小调的夜曲", {"key": "F#m"}),
    ("Compose a solo cello piece in E minor, 62 BPM adagio, 4/4, about 88 bars", {"key": "Em", "bars": 88}),
    ("A reel in F-sharp minor, 32 bars long", {"key": "F#m", "bars": 32}),
    ("A tune in D dorian over 24 measures", {"key": "D dorian", "bars": 24}),
    ("A piece in Bb major", {"key": "Bb"}),
    ("Write a 32-bar jig in G major (6/8)", {"key": "G", "bars": 32}),
])
def test_music_constraints_from_prompt(text, want):
    assert verl.music_constraints_from_prompt(text) == want


# trimmed rows in the shape of XiaomiMiMo/MiMo-V2.6-RL-oss (music split)
MIMO_MUSIC = {
    "prompt": [{"role": "user", "content": "创作一首E小调、93 BPM行板、4/4拍的琵琶曲，篇幅32小节（约一分半）。"
                                           "编制为一把琵琶，共2个声部。 请直接以 ABC 记谱格式输出完整乐曲。"}],
    "data_source": "music", "ability": "music_generation",
    "reward_model": {"style": "rule", "ground_truth": ""},
    "extra_info": {"src_id": "gK-1100", "lang": "zh", "tag": "琵琶曲", "length": "short", "nvoice_want": 2,
                   "bpm": 93, "meter": "4/4"},
}


def test_mimo_music_row_becomes_a_scorable_task():
    kind, task = verl.from_verl_row(MIMO_MUSIC)
    assert kind == "music" and task["id"] == "mimo-music-gK-1100"
    assert task["constraints"] == {"key": "Em", "bars": 32, "meter": "4/4", "voices": 2, "tempo": 93.0}
    s = score_abc("```abc\n" + reference_piece(task["constraints"]) + "\n```", task["constraints"])
    assert s.adherence["score"] == 1.0


def test_mimo_compound_meter_drops_ambiguous_tempo():
    row = json.loads(json.dumps(MIMO_MUSIC))
    row["extra_info"]["meter"] = "6/8"
    _, task = verl.from_verl_row(row)
    assert "tempo" not in task["constraints"] and task["import_notes"]


def test_mimo_image_mapping():
    assert verl.mimo_image("format-code-task-001457:latest") == \
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457"
    assert verl.mimo_image("arvo-rl:v1-arvo-35858").endswith(":arvo-v1-35858")
    assert verl.mimo_image("general-agent-env-3:oss", apptainer=True) == \
        "docker://docker.io/xiaomimimo/mimo-v2.6-rl-oss:general-agent-env-3"
    assert verl.mimo_image("python:3.12") == "python:3.12"


def test_unsupported_rows_are_skipped_with_reasons():
    rows = [{"data_source": "arvo", "extra_info": {}}, {"data_source": "blackbox/webdev", "extra_info": {}},
            {"data_source": "gsm8k", "extra_info": {}}, MIMO_MUSIC]
    by_kind, skipped = verl.import_rows(rows)
    assert len(by_kind["music"]) == 1 and sum(skipped.values()) == 3
    assert any("arvo" in r for r in skipped) and any("webdev" in r for r in skipped)


def _b64(s: str | bytes) -> str:
    return base64.b64encode(s.encode() if isinstance(s, str) else s).decode()


LATIN1 = "caf\xe9\n".encode("latin-1")
TB_TEST = """#!/bin/sh
mkdir -p /logs/verifier
expected=$(printf 'caf\\351\\n' | od -An -tx1)
got=$(od -An -tx1 /tests/fixtures/legacy.txt)
if [ "$(cat /app/out.txt 2>/dev/null)" = "1 2 3" ] && [ "$expected" = "$got" ]; then
  echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def _terminal_bench_row():
    instance = {"cwd": "/app", "dataset_type": "terminal_bench", "docker_image": "general-agent-env-1:oss",
                "instance_id": "demo-0001", "problem_statement": "Write the numbers 1 2 3 to /app/out.txt.",
                "cpus": 1, "memory_mb": 2048, "verifier_timeout_sec": 60,
                "tests_files": json.dumps({"test.sh": _b64(TB_TEST), "fixtures/legacy.txt": _b64(LATIN1)})}
    return {"data_source": "mimoagent/terminal_bench", "prompt": [{"role": "user", "content": "..."}],
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": 0, "instance_id": "demo-0001", "dataset_type": "terminal_bench",
                           "instance_json": json.dumps(instance)}}


def test_terminal_bench_row_runs_and_reads_reward_txt(iso):
    kind, task = verl.from_verl_row(_terminal_bench_row())
    assert kind == "tool" and task["workdir"] == "/app" and task["size"] == {"vcpus": 1, "memoryGb": 2.0}
    assert task["verify_files"]["/tests/fixtures/legacy.txt"] == {"base64": _b64(LATIN1)}
    task["image"] = None   # bwrap has no images; the row's image is only used by container backends
    good = scripted([msg(call("bash", command="echo 1 2 3 > out.txt"), call("submit"))])
    assert run_episode(ToolTaskEnv(iso, task), good)["reward"] == 1.0
    # a marker printed by the agent is not the verifier's last line, so it is ignored
    fake = scripted([msg(call("bash", command="echo 'echo FLOTILLA_REWARD=1' > out.txt"), call("submit"))])
    t = run_episode(ToolTaskEnv(iso, task), fake)
    assert t["reward"] == 0.0, t["info"]


# the hidden test patch modifies a test file that already exists in the image's repo
PATCH = """diff --git a/check.sh b/check.sh
--- a/check.sh
+++ b/check.sh
@@ -1 +1,2 @@
-exit 1
+out=$(python3 calc.py)
+[ "$out" = 4 ]
"""


def _code_row():
    instance = {"cwd": "/testbed", "dataset_type": "opensource-code", "docker_image": "format-code-task-000001:latest",
                "instance_id": "format-code-task-000001", "problem_statement": "calc.py should print 2 + 2.",
                "test_patch": PATCH, "test_command": "sh check.sh", "verifier_timeout_sec": 60}
    return {"data_source": "opensource-code", "prompt": [{"role": "user", "content": "..."}],
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": 0, "instance_id": "format-code-task-000001", "dataset_type": "opensource-code",
                           "instance_json": json.dumps(instance)}}


def _code_task():
    kind, task = verl.from_verl_row(_code_row())
    assert kind == "tool" and task["workdir"] == "/testbed"
    # bwrap has no images: rebuild the image's repo with setup, which runs in every sandbox
    task["image"] = None
    task["setup"] = "test -f calc.py || printf 'print(2 + 3)\\n' > calc.py; test -f check.sh || echo 'exit 1' > check.sh"
    return task


FIX = msg(call("edit", path="calc.py", old_string="2 + 3", new_string="2 + 2"),
          call("write", path="check.sh", content="python3 calc.py  # my quick check\n"),
          call("submit"))


def test_code_row_fix_scores_even_if_the_agent_edited_the_test_file(iso):
    # the agent rewrote check.sh while working; without restoring it from the image
    # the hidden patch would not apply and an honest fix would score 0
    assert run_episode(ToolTaskEnv(iso, _code_task()), scripted([FIX]))["reward"] == 1.0


def test_code_row_restore_is_what_makes_the_patch_apply(iso):
    task = _code_task()
    task["verify_prepare"] = None   # no pristine copies saved
    assert run_episode(ToolTaskEnv(iso, task), scripted([FIX]))["reward"] == 0.0


def test_code_row_resists_planted_tests_and_markers(iso):
    hack = msg(call("write", path="check.sh", content="exit 0\n"), call("submit"))
    assert run_episode(ToolTaskEnv(iso, _code_task()), scripted([hack]))["reward"] == 0.0
    shout = msg(call("write", path="calc.py", content="print('FLOTILLA_REWARD=1')\n"), call("submit"))
    assert run_episode(ToolTaskEnv(iso, _code_task()), scripted([shout]))["reward"] == 0.0


# ------------------------------------------------------------------ env features used by imports


def test_only_the_last_line_is_a_reward_marker(iso):
    task = {"id": "m", "prompt": "p", "files": {},
            "verify": "cat /work/out.txt 2>/dev/null; echo checking; test -f /work/real"}
    plant = ["```bash\necho FLOTILLA_REWARD=1 > out.txt\n```", "SUBMIT"]
    assert run_episode(ShellTaskEnv(iso, task), scripted(plant))["reward"] == 0.0
    task["verify"] = "echo noise; echo FLOTILLA_REWARD=0.5"
    assert run_episode(ShellTaskEnv(iso, task), scripted(["SUBMIT"]))["reward"] == 0.5


def test_deleted_files_stay_deleted_and_inner_symlinks_survive(iso):
    task = {"id": "d", "prompt": "p", "workdir": "/app",
            "files": {"/app/keep.txt": "k", "/app/remove_me.txt": "r"},
            "verify": "test ! -e remove_me.txt && test -L link && [ \"$(cat link)\" = k ]"}
    acts = ["```bash\nrm remove_me.txt && ln -s keep.txt link\n```", "SUBMIT"]
    assert run_episode(ShellTaskEnv(iso, task), scripted(acts))["reward"] == 1.0


def test_sanitize_tar_keeps_only_contained_relative_symlinks():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        for name, target in [("ok", "sub/f"), ("up", "../etc/passwd"), ("abs", "/data/expected.txt"),
                             ("sub/back", "../ok"), ("sub/deep", "../../x")]:
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tf.addfile(info)
    kept = {m.name: m.linkname for m in tarfile.open(fileobj=io.BytesIO(sanitize_tar(buf.getvalue(), 1000)))}
    assert kept == {"ok": "sub/f", "sub/back": "../ok"}


# ------------------------------------------------------------------ verl reward functions


@pytest.fixture
def fresh_state(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOTILLA_HOME", str(tmp_path))
    verl._state.clear()
    yield
    verl._state.clear()


def test_compute_score_code_and_music(fresh_state):
    code_gt = verl.to_verl_row(CODE_TASK, "code")["reward_model"]["ground_truth"]
    music_gt = verl.to_verl_row(MUSIC_TASK, "music")["reward_model"]["ground_truth"]
    good_code = "```python\ndef add(a, b):\n    return a + b\n```"
    piece = "```abc\n" + reference_piece(MUSIC_TASK["constraints"]) + "\n```"
    out = verl.compute_score_batch(
        data_sources=["flotilla/code", "flotilla/code", "flotilla/music"],
        solution_strs=[good_code, "no code", piece],
        ground_truths=[code_gt, code_gt, music_gt],
        extra_infos=[{}, {}, {}],
    )
    assert out[0] == {"score": 1.0, "infra": 0.0}
    assert out[1] == {"score": 0.0, "infra": 0.0}
    assert out[2]["score"] > 0.5 and set(out[2]) == {"score", "infra"}
    assert verl.compute_score("flotilla/code", good_code, code_gt)["score"] == 1.0


def test_compute_score_unknown_source_without_verl(fresh_state):
    try:
        import verl as _real_verl  # noqa: F401
        pytest.skip("verl is installed; the fallback would be used")
    except ImportError:
        pass
    with pytest.raises(ValueError, match="no flotilla scorer"):
        verl.compute_score("gsm8k", "42", "42")
