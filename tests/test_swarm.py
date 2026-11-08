import json
import threading
import time

import pytest

from flotilla import Flotilla, SpecError, SwarmSpec
from flotilla.spec import parse_duration
from flotilla.swarm import IdempotencyConflict, SwarmRunner


def test_count_run_sets_index_vars(fl):
    run = fl.swarms.run("echo $FLOTILLA_SWARM_INDEX/$FLOTILLA_SWARM_COUNT/$MARITIME_SWARM_INDEX", count=5,
                        timeout_seconds=30)
    assert run["status"] == "completed"
    assert run["counts"]["succeeded"] == 5
    outs = [sb["stdoutTail"].strip() for sb in run["sandboxes"]]
    assert outs == [f"{i}/5/{i}" for i in range(5)]


def test_inputs_args_env_files_and_result(fl):
    inputs = [
        {"args": ["--seed", "1"]},
        {"args": ["--seed", "2"], "env": {"TEMPERATURE": "0.7"}},
        {"files": {"/work/task.json": '{"id": 3}'}},
    ]
    cmd = ("echo args=\"$*\" t=$TEMPERATURE; "
           "if [ -f task.json ]; then cat task.json > result.json; else echo '{}' > result.json; fi; true")
    run = fl.swarms.run(["sh", "-c", cmd, "sh"], inputs=inputs, result_path="/work/result.json",
                        timeout_seconds=30)
    sbs = run["sandboxes"]
    assert [sb["status"] for sb in sbs] == ["succeeded"] * 3
    assert "args=--seed 1" in sbs[0]["stdoutTail"]
    assert "t=0.7" in sbs[1]["stdoutTail"]
    assert json.loads(sbs[2]["result"]) == {"id": 3}
    assert json.loads(sbs[0]["result"]) == {}


def test_failure_and_timeout(fl):
    run = fl.swarms.run("if [ $FLOTILLA_SWARM_INDEX = 0 ]; then exit 3; else sleep 30; fi",
                        count=2, timeout_seconds=1)
    a, b = run["sandboxes"]
    assert a["status"] == "failed" and a["exitCode"] == 3 and a["error"] == "exit code 3"
    assert b["status"] == "failed" and b["exitCode"] is None and "timeout" in b["error"]
    assert b["durationSeconds"] < 5


def test_output_tail_is_truncated(fl):
    run = fl.swarms.run("python3 -c \"print('x' * 100000); print('END')\"", count=1, timeout_seconds=30)
    tail = run["sandboxes"][0]["stdoutTail"]
    assert len(tail) == 16 * 1024 and tail.rstrip().endswith("END")


def test_result_is_capped_at_64k(fl):
    run = fl.swarms.run("python3 -c \"open('r', 'w').write('y' * 70000)\"", count=1,
                        result_path="/work/r", timeout_seconds=30)
    sb = run["sandboxes"][0]
    assert len(sb["result"]) == 64 * 1024 and sb["resultTruncated"] is True


def test_cancel_stops_running_and_queued(tmp_path):
    fl = Flotilla(backend="local", state_dir=str(tmp_path), max_concurrency=2)
    run = fl.swarms.start("sleep 30", count=6, timeout_seconds=60)
    time.sleep(0.5)
    fl.swarms.cancel(run["id"])
    done = fl.swarms.wait(run["id"], timeout=10)
    assert done["status"] == "cancelled"
    assert done["counts"]["cancelled"] == 6


def test_global_concurrency_cap(tmp_path):
    fl = Flotilla(backend="local", state_dir=str(tmp_path), max_concurrency=2)
    run = fl.swarms.run("sleep 0.5", count=4, concurrency=10, timeout_seconds=30)
    assert run["concurrency"] == 2
    assert run["durationSeconds"] >= 0.9  # two waves of two


def test_idempotency(tmp_path):
    r = SwarmRunner(backend="local", state_dir=tmp_path)
    a, created_a = r.start({"command": "true", "count": 1}, idempotency_key="k1")
    b, created_b = r.start({"command": "true", "count": 1}, idempotency_key="k1")
    assert created_a and not created_b and a["id"] == b["id"]
    with pytest.raises(IdempotencyConflict):
        r.start({"command": "false", "count": 1}, idempotency_key="k1")


def test_runs_persist_and_reload(tmp_path):
    r = SwarmRunner(backend="local", state_dir=tmp_path)
    run = r.run({"command": "echo hi", "count": 2})
    lines = (tmp_path / "runs" / run["id"] / "results.jsonl").read_text().splitlines()
    assert [json.loads(x)["index"] for x in lines] == [0, 1]
    r2 = SwarmRunner(backend="local", state_dir=tmp_path)
    assert r2.get(run["id"])["counts"]["succeeded"] == 2


def test_spec_validation():
    with pytest.raises(SpecError):
        SwarmSpec.from_dict({"command": "x"})
    with pytest.raises(SpecError):
        SwarmSpec.from_dict({"command": "x", "count": 1, "inputs": [{}]})
    with pytest.raises(SpecError):
        SwarmSpec.from_dict({"command": "x", "count": 1, "timeoutSeconds": 0})
    with pytest.raises(SpecError):
        SwarmSpec.from_dict({"command": "x", "inputs": [{"files": {"rel/path": "a"}}]})
    with pytest.raises(SpecError):
        SwarmSpec.from_dict({"command": "x", "count": 1, "resultPath": "work/result.json"})
    s = SwarmSpec.from_dict({"command": ["python", "a b.py"], "inputs": [{"args": ["--x", "y z"]}],
                             "size": {"vcpus": 1, "memoryGb": 2}, "timeoutSeconds": 10})
    assert s.command_for(0) == "python 'a b.py' --x 'y z'"
    assert parse_duration("10m") == 600 and parse_duration("1500ms") == 1.5 and parse_duration("90") == 90


def test_many_sandboxes_threads_clean_up(tmp_path):
    fl = Flotilla(backend="local", state_dir=str(tmp_path), max_concurrency=8)
    before = threading.active_count()
    run = fl.swarms.run("true", count=40, timeout_seconds=30)
    assert run["counts"]["succeeded"] == 40
    time.sleep(0.2)
    assert threading.active_count() <= before + 2
    assert not any((tmp_path / "work").iterdir())
