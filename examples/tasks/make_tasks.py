"""Generate the example task files (code_tasks.jsonl, shell_tasks.jsonl).

Each code task has a prompt, an entry point, io tests and a reference
solution; ``python examples/eval_code.py --references`` checks that every
reference scores 1.0, which is the sanity gate for a task set.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

CODE = [
    ("run_length_encode",
     "Write `run_length_encode(s: str) -> list` that returns [[char, count], ...] for consecutive runs.",
     "def run_length_encode(s):\n    out = []\n    for ch in s:\n        if out and out[-1][0] == ch:\n            out[-1][1] += 1\n        else:\n            out.append([ch, 1])\n    return out\n",
     [[["aaabcc"]], [[""]], [["x"]], [["abba"]]]),
    ("is_balanced",
     "Write `is_balanced(s: str) -> bool`: True if every (, [ and { in s is closed in the right order. Ignore other characters.",
     "def is_balanced(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    st = []\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif c in pairs:\n            if not st or st.pop() != pairs[c]:\n                return False\n    return not st\n",
     [[["([]{})"]], [["(]"]], [["((a)"]], [[""]], [["a{b[c]d}e"]], [["}{"]]]),
    ("kth_largest",
     "Write `kth_largest(nums: list, k: int) -> int` returning the k-th largest element (k=1 is the maximum).",
     "def kth_largest(nums, k):\n    return sorted(nums, reverse=True)[k - 1]\n",
     [[[[3, 1, 4, 1, 5], 1]], [[[3, 1, 4, 1, 5], 3]], [[[7], 1]], [[[2, 2, 2], 2]]]),
    ("merge_intervals",
     "Write `merge_intervals(iv: list) -> list` merging overlapping [start, end] intervals (touching ones merge too). Return them sorted by start.",
     "def merge_intervals(iv):\n    out = []\n    for s, e in sorted(iv):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n",
     [[[[[1, 3], [2, 6], [8, 10]]]], [[[[1, 4], [4, 5]]]], [[[]]], [[[[5, 7], [1, 2]]]]]),
    ("count_islands",
     "Write `count_islands(grid: list) -> int` counting 4-connected groups of 1s in a 0/1 grid.",
     "def count_islands(grid):\n    seen = set()\n    n = 0\n    for r in range(len(grid)):\n        for c in range(len(grid[0]) if grid else 0):\n            if grid[r][c] == 1 and (r, c) not in seen:\n                n += 1\n                st = [(r, c)]\n                while st:\n                    y, x = st.pop()\n                    if (y, x) in seen or not (0 <= y < len(grid) and 0 <= x < len(grid[0])) or grid[y][x] != 1:\n                        continue\n                    seen.add((y, x))\n                    st += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]\n    return n\n",
     [[[[[1, 1, 0], [0, 1, 0], [1, 0, 1]]]], [[[[0]]]], [[[]]], [[[[1, 0, 1, 0, 1]]]]]),
    ("roman_to_int",
     "Write `roman_to_int(s: str) -> int` converting a Roman numeral (I..MMMCMXCIX) to an integer.",
     "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    t = 0\n    for i, ch in enumerate(s):\n        if i + 1 < len(s) and v[ch] < v[s[i + 1]]:\n            t -= v[ch]\n        else:\n            t += v[ch]\n    return t\n",
     [[["III"]], [["IX"]], [["LVIII"]], [["MCMXCIV"]], [["MMMCMXCIX"]]]),
    ("moving_average",
     "Write `moving_average(xs: list, w: int) -> list` returning the mean of each full window of size w, as floats.",
     "def moving_average(xs, w):\n    return [sum(xs[i:i + w]) / w for i in range(len(xs) - w + 1)]\n",
     [[[[1, 2, 3, 4], 2]], [[[5], 1]], [[[1, 2], 3]], [[[0.1, 0.2, 0.3], 3]]]),
    ("word_frequencies",
     "Write `word_frequencies(text: str) -> dict` mapping each lowercase word (split on whitespace, strip .,!? from both ends) to its count. Skip empty words.",
     "def word_frequencies(text):\n    d = {}\n    for w in text.split():\n        w = w.strip('.,!?').lower()\n        if w:\n            d[w] = d.get(w, 0) + 1\n    return d\n",
     [[["The cat. the CAT!"]], [[""]], [["a, b ... a"]]]),
    ("binary_search",
     "Write `binary_search(xs: list, target) -> int` returning the index of target in sorted xs, or -1.",
     "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        if xs[m] == target:\n            return m\n        if xs[m] < target:\n            lo = m + 1\n        else:\n            hi = m - 1\n    return -1\n",
     [[[[1, 3, 5, 7], 5]], [[[1, 3, 5, 7], 4]], [[[], 1]], [[[2], 2]]]),
    ("primes_upto",
     "Write `primes_upto(n: int) -> list` returning all primes <= n in increasing order.",
     "def primes_upto(n):\n    if n < 2:\n        return []\n    s = [True] * (n + 1)\n    s[0] = s[1] = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if s[i]:\n            s[i * i::i] = [False] * len(s[i * i::i])\n    return [i for i, p in enumerate(s) if p]\n",
     [[[1]], [[2]], [[30]], [[97]]]),
]


def code_tasks():
    rows = []
    for ep, prompt, ref, cases in CODE:
        ns = {}
        exec(ref, ns)
        tests = [{"args": c[0], "expected": json.loads(json.dumps(ns[ep](*c[0])))} for c in cases]
        rows.append({
            "id": ep,
            "prompt": prompt + " Answer with a single ```python``` code block.",
            "entry_point": ep,
            "tests": tests,
            "reference": ref,
        })
    return rows


SHELL = [
    {
        "id": "fix-off-by-one",
        "prompt": ("/work/stats.py should print the sum of the numbers 1 to 100 inclusive, "
                   "but it prints the wrong value. Fix the file, then reply SUBMIT."),
        "files": {"/work/stats.py": "total = 0\nfor i in range(1, 100):\n    total += i\nprint(total)\n"},
        "verify_files": {"/verify/check.sh": "out=$(python3 /work/stats.py)\n[ \"$out\" = 5050 ]\n"},
        "verify": "sh /verify/check.sh",
        "max_turns": 6,
    },
    {
        "id": "csv-filter",
        "prompt": ("/work/data.csv has columns name,score. Write the names whose score is at least 80, "
                   "one per line in file order, to /work/answer.txt. Then reply SUBMIT."),
        "files": {"/work/data.csv": "name,score\nada,91\nbob,79\ncem,80\ndeniz,12\nece,100\n"},
        "verify_files": {"/verify/check.py": (
            "import sys\n"
            "try:\n    got = open('/work/answer.txt').read().split()\n"
            "except OSError:\n    sys.exit(1)\n"
            "want = ['ada', 'cem', 'ece']\n"
            "hit = sum(1 for w in want if w in got)\n"
            "extra = len([g for g in got if g not in want])\n"
            "print(f'FLOTILLA_REWARD={max(0.0, (hit - extra) / len(want)):.3f}')\n")},
        "verify": "python3 /verify/check.py",
        "max_turns": 6,
    },
    {
        "id": "make-cli",
        "prompt": ("Create /work/rev.py: a script that reads lines from stdin and prints each line reversed "
                   "(characters in reverse order), keeping line order. Then reply SUBMIT."),
        "files": {},
        "verify_files": {"/verify/check.sh": (
            "out=$(printf 'abc\\nhello world\\n' | python3 /work/rev.py)\n"
            "[ \"$out\" = \"$(printf 'cba\\ndlrow olleh')\" ]\n")},
        "verify": "sh /verify/check.sh",
        "max_turns": 6,
    },
]


if __name__ == "__main__":
    with open(HERE / "code_tasks.jsonl", "w") as f:
        for r in code_tasks():
            f.write(json.dumps(r) + "\n")
    with open(HERE / "shell_tasks.jsonl", "w") as f:
        for r in SHELL:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(CODE), "code tasks and", len(SHELL), "shell tasks")
