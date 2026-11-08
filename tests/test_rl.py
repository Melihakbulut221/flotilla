import json

import pytest

from flotilla.rl import CodeVerifier, ShellTaskEnv, extract_code, group_advantages, parse_action, run_episode

ADD_TASK = {"entry_point": "add", "tests": [{"args": [1, 2], "expected": 3},
                                            {"args": [-1, 1], "expected": 0},
                                            {"args": [0.1, 0.2], "expected": 0.3}]}


def wrap(code):
    return f"Here you go:\n```python\n{code}\n```\n"


def test_extract_code():
    assert extract_code("```python\na=1\n```\ntext\n```py\nb=2\n```") == "b=2\n"
    assert extract_code("no code") is None
    assert extract_code("no code", require_fence=False) == "no code"


def test_io_mode_scores(fl):
    v = CodeVerifier(fl, timeout=30, per_test_timeout=2)
    comps = [
        wrap("def add(a, b):\n    return a + b"),
        wrap("def add(a, b):\n    return a - b"),
        wrap("def add(a, b):\n    while True: pass"),
        "I refuse to write code.",
        wrap("def add(a, b):\n    raise ValueError('x')"),
        wrap("syntax error here ("),
    ]
    scores = v.score(comps, [ADD_TASK] * len(comps))
    assert scores[0].reward == 1.0
    assert scores[1].passed == 0
    assert scores[2].reward == 0.0
    assert scores[3].status == "format" and scores[3].reward == 0.0
    assert scores[4].reward == 0.0
    assert scores[5].reward == 0.0 and "SyntaxError" in (scores[5].error or "")


HACKS = {
    # exit 0 before any test runs
    "early_exit": "import os\nos._exit(0)",
    # print a fake perfect result on the protocol channel and to stdout
    "fake_result": ("import os, json\nos.write(1, b'{\"passed\": 3, \"total\": 3}\\n')\n"
                    "print(json.dumps({'load_error': None, 'results': [{'ok': True, 'value': 0}]*3}))\n"
                    "def add(a, b):\n    return None"),
    # daemon that overwrites the result file after the harness writes it
    "late_writer": ("import os, time, json\n"
                    "if os.fork() == 0:\n"
                    "    os.setsid()\n"
                    "    os.closerange(0, 1024)\n"
                    "    t = time.time()\n"
                    "    while time.time() - t < 20:\n"
                    "        if os.path.exists('.flotilla_result.json'):\n"
                    "            for _ in range(50):\n"
                    "                try:\n"
                    "                    os.unlink('.flotilla_result.json')\n"
                    "                except OSError: pass\n"
                    "                open('.flotilla_result.json', 'w').write(json.dumps({'passed': 3, 'total': 3}))\n"
                    "            os._exit(0)\n"
                    "        time.sleep(0.0005)\n"
                    "    os._exit(0)\n"
                    "def add(a, b):\n    return None"),
    # try to read the expected answers
    "peek_task": ("import os\nX = os.path.exists('/work/.task.json')\n"
                  "def add(a, b):\n    if X:\n        raise SystemExit\n    return None"),
    # a value that compares equal to anything
    "eq_object": ("class A:\n    def __eq__(self, o): return True\n"
                  "def add(a, b):\n    return A()"),
}


@pytest.mark.parametrize("name", sorted(HACKS))
def test_io_mode_resists_reward_hacks(iso, name):
    v = CodeVerifier(iso, timeout=30, per_test_timeout=2)
    s = v.score([wrap(HACKS[name])], [ADD_TASK])[0]
    assert s.reward == 0.0, (name, s)


def test_assert_mode(fl):
    v = CodeVerifier(fl, timeout=30)
    task = {"tests": "def test_a():\n    assert sq(3) == 9\n\ndef test_b():\n    assert sq(-2) == 4\n"}
    good = v.score([wrap("def sq(x):\n    return x*x")], [task])[0]
    half = v.score([wrap("def sq(x):\n    return 9")], [task])[0]
    assert (good.passed, good.total, good.reward) == (2, 2, 1.0)
    assert (half.passed, half.total) == (1, 2)


def test_tests_given_as_json_string(fl):
    v = CodeVerifier(fl, timeout=30)
    task = {"entry_point": "add", "tests": json.dumps(ADD_TASK["tests"])}
    assert v.rewards([wrap("def add(a,b): return a+b")], [task]) == [1.0]


def test_parse_action():
    assert parse_action("ok\n```bash\nls -la\n```") == ("bash", "ls -la")
    assert parse_action("done\nSUBMIT\n") == ("submit", None)
    assert parse_action("```python\nprint(1)\n```") == ("invalid", None)


FIX_TASK = {
    "id": "fix-sum",
    "prompt": "Make `python3 app.py` print 15.",
    "files": {"/work/app.py": "print(sum(range(5)))\n"},
    "verify_files": {"/verify/check.sh": "out=$(python3 /work/app.py); [ \"$out\" = 15 ]\n"},
    "verify": "sh /verify/check.sh",
    "max_turns": 4,
}


def scripted(actions):
    it = iter(actions)
    return lambda messages: next(it)


def test_shell_env_solves_and_keeps_state(iso):
    env = ShellTaskEnv(iso, FIX_TASK)
    traj = run_episode(env, scripted([
        "```bash\npython3 app.py\n```",
        "```bash\necho 'print(sum(range(6)))' > app.py\n```",
        "```bash\npython3 app.py\n```",
        "SUBMIT",
    ]))
    assert traj["reward"] == 1.0, traj
    obs = [m["content"] for m in traj["messages"] if m["role"] == "user"]
    assert "10" in obs[1] and "15" in obs[3]


def test_shell_env_failure_and_max_turns(iso):
    env = ShellTaskEnv(iso, FIX_TASK)
    traj = run_episode(env, scripted(["thinking...", "```bash\ntrue\n```", "hmm", "```bash\ntrue\n```"]))
    assert traj["reward"] == 0.0 and traj["info"]["end"] == "max_turns"


def test_partial_reward_line(iso):
    task = dict(FIX_TASK, verify="echo FLOTILLA_REWARD=0.25", verify_files={})
    env = ShellTaskEnv(iso, task)
    assert run_episode(env, scripted(["SUBMIT"]))["reward"] == 0.25


def test_group_advantages():
    t = group_advantages([{"task_id": "a", "reward": 1.0}, {"task_id": "a", "reward": 0.0},
                          {"task_id": "b", "reward": 0.5}])
    assert t[0]["advantage"] > 0 > t[1]["advantage"] and abs(t[2]["advantage"]) < 1e-6
