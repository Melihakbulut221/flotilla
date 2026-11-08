import io
import tarfile

import pytest

from flotilla.rl import (
    CodeVerifier,
    ShellTaskEnv,
    filter_groups,
    group_advantages,
    is_degenerate,
    length_penalty,
    ngram_uniqueness,
    run_episode,
)
from flotilla.rl.env import sanitize_tar


def scripted(actions):
    it = iter(actions)
    return lambda messages: next(it)


# ------------------------------------------------------------ repetition


def test_ngram_uniqueness_and_degenerate():
    loop = "the cat sat down " * 200
    varied = " ".join(f"w{i}" for i in range(800))
    assert ngram_uniqueness(loop) < 0.05
    assert ngram_uniqueness(varied) == 1.0
    assert is_degenerate(loop)
    assert not is_degenerate(varied)
    assert not is_degenerate("the cat " * 50)  # short text is never flagged


def test_code_verifier_zeroes_repetition_without_a_sandbox(fl):
    v = CodeVerifier(fl, timeout=30)
    text = "```python\ndef add(a, b):\n    return a + b\n```\n" + "I am sure this is right. " * 200
    s = v.score([text], [{"entry_point": "add", "tests": [{"args": [1, 2], "expected": 3}]}])[0]
    assert s.status == "repetition" and s.reward == 0.0


# ------------------------------------------------------------ infra errors


def test_infra_errors_are_flagged_not_scored(iso):
    def broken_policy(messages):
        raise ConnectionError("model server down")

    t = run_episode(ShellTaskEnv(iso, FIX_TASK), broken_policy)
    assert t["infra_error"] is True and "ConnectionError" in t["info"]["error"]


def test_model_failure_is_not_infra(iso):
    t = run_episode(ShellTaskEnv(iso, FIX_TASK), scripted(["SUBMIT"]))
    assert t["infra_error"] is False and t["reward"] == 0.0


def test_code_verifier_infra_classification(fl):
    v = CodeVerifier(fl, infra_reward=0.0)
    boot = v._parse({"status": "failed", "error": "boot failed: RuntimeError: x", "result": None})
    timeout = v._parse({"status": "failed", "error": "timeout after 60s", "result": None})
    cancelled = v._parse({"status": "cancelled", "error": "cancelled", "result": None})
    assert boot.infra and cancelled.infra and not timeout.infra


def test_group_advantages_exclude_infra():
    trajs = [
        {"task_id": "a", "reward": 1.0},
        {"task_id": "a", "reward": 0.0},
        {"task_id": "a", "reward": 0.0, "infra_error": True},
        {"task_id": "a", "reward": 0.0, "infra_error": True},
    ]
    group_advantages(trajs)
    assert trajs[2]["advantage"] == 0.0 and trajs[3]["advantage"] == 0.0
    # mean over the two valid rollouts is 0.5, so they get +1 / -1
    assert trajs[0]["advantage"] == pytest.approx(1.0, abs=1e-4)
    assert trajs[1]["advantage"] == pytest.approx(-1.0, abs=1e-4)


# ------------------------------------------------------------ group filtering


def test_filter_groups_drops_uniform_and_infra_heavy_groups():
    trajs = (
        [{"task_id": "mixed", "reward": r} for r in (1.0, 0.0, 1.0)]
        + [{"task_id": "all-pass", "reward": 1.0} for _ in range(3)]
        + [{"task_id": "all-fail", "reward": 0.0} for _ in range(3)]
        + [{"task_id": "broken", "reward": 1.0}, {"task_id": "broken", "reward": 0.0, "infra_error": True}]
    )
    kept, stats = filter_groups(trajs)
    assert {t["task_id"] for t in kept} == {"mixed"} and len(kept) == 3
    assert stats == {"groups": 4, "kept_groups": 1, "uniform_groups": 2, "too_few_valid": 1,
                     "infra_rollouts": 1}


def test_length_penalty_only_when_most_of_the_group_passes():
    def traj(task, reward, n_chars):
        return {"task_id": task, "reward": reward,
                "messages": [{"role": "assistant", "content": "x" * n_chars}]}

    easy = [traj("easy", 1.0, n) for n in (100, 200, 300, 1000)]
    hard = [traj("hard", 1.0, 1000)] + [traj("hard", 0.0, 100) for _ in range(3)]
    length_penalty(easy + hard)
    assert easy[0]["reward"] == 1.0                   # shortest passing: no penalty
    assert easy[3]["reward"] == pytest.approx(0.8)    # longest passing: full 0.2
    assert 0.8 < easy[2]["reward"] < 1.0
    assert hard[0]["reward"] == 1.0                   # only 1/4 passed: untouched


# ------------------------------------------------------------ isolated verification

FIX_TASK = {
    "id": "fix-sum",
    "prompt": "Make `python3 app.py` print 15.",
    "files": {"/work/app.py": "print(sum(range(5)))\n"},
    "verify_files": {"/verify/check.sh": "out=$(python3 /work/app.py); [ \"$out\" = 15 ]\n"},
    "verify": "sh /verify/check.sh",
    "max_turns": 4,
}

# the reference answer is a task input outside /work; the agent can overwrite it
REF_TASK = {
    "id": "ref",
    "prompt": "Write the sorted numbers from /data/in.txt to /work/out.txt, one per line.",
    "files": {"/data/in.txt": "3\n1\n2\n", "/data/expected.txt": "1\n2\n3\n"},
    "verify": "cmp -s /work/out.txt /data/expected.txt",
    "max_turns": 3,
}
HACK = ["```bash\necho wrong > out.txt; echo wrong > /data/expected.txt\n```", "SUBMIT"]


def test_reference_overwrite_hack_works_without_isolation(iso):
    """Proves the attack is real: judged in the agent's own sandbox it scores 1."""
    t = run_episode(ShellTaskEnv(iso, REF_TASK, isolated_verify=False), scripted(HACK))
    assert t["reward"] == 1.0


def test_reference_overwrite_hack_fails_with_isolated_verify(iso):
    t = run_episode(ShellTaskEnv(iso, REF_TASK), scripted(HACK))
    assert t["reward"] == 0.0 and t["info"]["isolated_verify"] is True


def test_isolated_verify_keeps_honest_work(iso):
    honest = ["```bash\nsort -n /data/in.txt > out.txt && mkdir -p sub && echo x > sub/y\n```", "SUBMIT"]
    t = run_episode(ShellTaskEnv(iso, REF_TASK), scripted(honest))
    assert t["reward"] == 1.0


def test_tmp_is_not_carried_to_the_grader(iso):
    task = dict(REF_TASK, verify="test ! -e /tmp/planted && cmp -s /work/out.txt /data/expected.txt")
    acts = ["```bash\nsort -n /data/in.txt > out.txt; touch /tmp/planted\n```", "SUBMIT"]
    assert run_episode(ShellTaskEnv(iso, task), scripted(acts))["reward"] == 1.0
    assert run_episode(ShellTaskEnv(iso, task, isolated_verify=False), scripted(acts))["reward"] == 0.0


def test_workspace_symlinks_do_not_reach_the_grader(iso):
    task = dict(REF_TASK, verify="test ! -e /work/link && test ! -L /work/link && test -f /work/out.txt")
    acts = ["```bash\necho ok > out.txt; ln -s /etc/passwd link\n```", "SUBMIT"]
    assert run_episode(ShellTaskEnv(iso, task), scripted(acts))["reward"] == 1.0


def test_oversized_workspace_scores_zero_not_infra(iso):
    acts = ["```bash\nhead -c 3000000 /dev/zero > big.bin; echo ok > out.txt\n```", "SUBMIT"]
    t = run_episode(ShellTaskEnv(iso, REF_TASK, max_workspace_bytes=1_000_000), scripted(acts))
    assert t["reward"] == 0.0 and t["infra_error"] is False and "larger" in t["info"]["verify_error"]


def test_sanitize_tar_drops_dangerous_members():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        def add(name, data=b"x", **kw):
            info = tarfile.TarInfo(name)
            for k, v in kw.items():
                setattr(info, k, v)
            if info.isreg():
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                tf.addfile(info)
        add("./good.txt", b"hello")
        add("./dir", type=tarfile.DIRTYPE)
        add("./dir/inner.py", b"print(1)")
        add("/etc/abs", b"bad")
        add("../escape", b"bad")
        add("./a/../../escape2", b"bad")
        add("./link", type=tarfile.SYMTYPE, linkname="/etc/passwd")
        add("./hard", type=tarfile.LNKTYPE, linkname="good.txt")
        add("./dev", type=tarfile.CHRTYPE)
    clean = tarfile.open(fileobj=io.BytesIO(sanitize_tar(buf.getvalue(), 10_000)))
    assert sorted(m.name for m in clean.getmembers()) == ["dir", "dir/inner.py", "good.txt"]
    assert clean.extractfile("good.txt").read() == b"hello"
