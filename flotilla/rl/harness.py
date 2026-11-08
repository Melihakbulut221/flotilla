"""Test harness shipped into each sandbox by CodeVerifier. Standalone: stdlib only.

Two modes:

io      Tests are (args, expected) pairs for one entry-point function. A child
        process imports the solution and computes outputs for the args only;
        this parent compares them with the expected values, which the child
        never sees (the task file is deleted before the child starts, and this
        process marks itself non-dumpable so the child cannot read its memory).
        Model code cannot fake a pass by exiting early, patching assert, or
        printing a fake result.

assert  Tests are Python source with test_* functions (or bare asserts) run
        in the same process as the solution, HumanEval style. Convenient but
        hackable by a determined policy (it shares a process with its judge).

Before writing the result every leftover process the solution spawned is
killed (this process is a child subreaper), so nothing can overwrite the
result after it is written.
"""

import ctypes
import json
import math
import os
import subprocess
import sys
import time

# Relative to the working directory (/work in isolated backends), so the
# harness also runs under the plain-process backend.
TASK = ".task.json"
RESULT = ".flotilla_result.json"
SOLUTION = "solution.py"

CHILD = r'''
import copy, json, os, signal, sys, traceback
out = os.dup(1)
os.dup2(2, 1)                      # solution prints go to stderr, not the protocol channel
spec = json.loads(sys.stdin.read())
def _alarm(*a):
    raise TimeoutError("timed out")
signal.signal(signal.SIGALRM, _alarm)
res = {"load_error": None, "results": []}
ns = {"__name__": "solution"}
try:
    signal.setitimer(signal.ITIMER_REAL, spec["per_test_timeout"])
    with open(spec["solution"]) as f:
        exec(compile(f.read(), "solution.py", "exec"), ns)
    signal.setitimer(signal.ITIMER_REAL, 0)
except BaseException as e:
    signal.setitimer(signal.ITIMER_REAL, 0)
    res["load_error"] = f"{type(e).__name__}: {e}"[:500]
if res["load_error"] is None:
    if spec["mode"] == "io":
        fn = ns.get(spec["entry_point"])
        if not callable(fn):
            res["load_error"] = f"entry point {spec['entry_point']!r} not defined"
        else:
            for args in spec["calls"]:
                try:
                    signal.setitimer(signal.ITIMER_REAL, spec["per_test_timeout"])
                    v = fn(*copy.deepcopy(args))
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    try:
                        v = json.loads(json.dumps(v))
                    except (TypeError, ValueError):
                        v = {"__repr__": repr(v)[:200]}
                    res["results"].append({"ok": True, "value": v})
                except BaseException as e:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    res["results"].append({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
    else:
        src = spec["tests"]
        try:
            exec(compile(src, "tests.py", "exec"), ns)
            names = [k for k, v in list(ns.items()) if k.startswith("test") and callable(v)]
        except BaseException as e:
            names = None
            res["results"].append({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        if names == []:
            res["results"].append({"ok": True})
        for name in names or []:
            try:
                signal.setitimer(signal.ITIMER_REAL, spec["per_test_timeout"])
                ns[name]()
                signal.setitimer(signal.ITIMER_REAL, 0)
                res["results"].append({"ok": True, "name": name})
            except BaseException as e:
                signal.setitimer(signal.ITIMER_REAL, 0)
                res["results"].append({"ok": False, "name": name, "error": f"{type(e).__name__}: {e}"[:300]})
os.write(out, ("\n" + json.dumps(res) + "\n").encode())
'''


def _prctl(option, arg):
    try:
        ctypes.CDLL(None, use_errno=True).prctl(option, arg, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def _children(me):
    kids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me:
            kids.append(int(d))
    return kids


def reap_descendants():
    me = os.getpid()
    for _ in range(100):
        kids = _children(me)
        if not kids:
            return
        for k in kids:
            try:
                os.kill(k, 9)
            except OSError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        time.sleep(0.01)


def same(a, b, rel=1e-6, abs_=1e-9):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, rel, abs_) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rel, abs_) for k in a)
    return a == b


def write_result(res):
    try:
        if os.path.isdir(RESULT) and not os.path.islink(RESULT):
            os.rmdir(RESULT)
        else:
            os.unlink(RESULT)
    except OSError:
        pass
    fd = os.open(RESULT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(res, f)


def main():
    _prctl(36, 1)  # PR_SET_CHILD_SUBREAPER: orphans of the solution come back to us
    _prctl(4, 0)   # PR_SET_DUMPABLE=0: the solution cannot read this process's memory
    with open(TASK) as f:
        task = json.load(f)
    os.unlink(TASK)
    mode = task.get("mode", "io")
    per = float(task.get("per_test_timeout", 5))
    total_timeout = float(task.get("total_timeout", 60))
    if mode == "io":
        tests = task["tests"]
        spec = {"mode": "io", "entry_point": task["entry_point"],
                "calls": [t["args"] for t in tests], "per_test_timeout": per, "solution": SOLUTION}
        n_total = len(tests)
    else:
        spec = {"mode": "assert", "tests": task["tests"], "per_test_timeout": per, "solution": SOLUTION}
        n_total = None
    res = {"passed": 0, "total": n_total, "error": None, "failures": []}
    try:
        p = subprocess.run([sys.executable, "-I", "-c", CHILD], input=json.dumps(spec).encode(),
                           capture_output=True, timeout=total_timeout, start_new_session=True)
        lines = [ln for ln in p.stdout.decode(errors="replace").splitlines() if ln.strip()]
        child = json.loads(lines[-1]) if lines else None
        if child is None:
            res["error"] = f"no result from child (exit {p.returncode}): " + p.stderr.decode(errors="replace")[-300:]
        elif child["load_error"]:
            res["error"] = child["load_error"]
        elif mode == "io":
            got = child["results"]
            for i, t in enumerate(tests):
                r = got[i] if i < len(got) else {"ok": False, "error": "missing"}
                if r["ok"] and same(r["value"], t["expected"]):
                    res["passed"] += 1
                elif len(res["failures"]) < 5:
                    res["failures"].append({"index": i, "error": r.get("error"),
                                            "got": r.get("value") if r["ok"] else None})
        else:
            got = child["results"]
            res["total"] = len(got)
            res["passed"] = sum(1 for r in got if r["ok"])
            res["failures"] = [r for r in got if not r["ok"]][:5]
    except subprocess.TimeoutExpired:
        res["error"] = f"total timeout {total_timeout:g}s"
    except (ValueError, KeyError) as e:
        res["error"] = f"bad child output: {e}"
    reap_descendants()
    if res["total"] is None:
        res["total"] = 1
    write_result(res)
    print(json.dumps({k: res[k] for k in ("passed", "total", "error")}))


if __name__ == "__main__":
    main()
