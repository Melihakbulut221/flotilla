"""Build a supervised fine-tuning set for the music environment.

The RL reward in ``music.py`` only gives a signal once a model writes valid
ABC. A small model starts at zero (see the baseline in the README), so it
first needs a warm-up on real tunes paired with prompts in the exact form
the RL tasks use.

Every example is a chat pair::

    user:      "Compose a jig in Bb major, 6/8 time, at dotted quarter = 116, 32 bars long, ..."
    assistant: "```abc\\nX:1\\nT:...\\nM:6/8\\nL:1/8\\nQ:3/8=116\\nK:Bb\\n...```"

Each example is built from a real tune in four steps:

1. Parse the tune. Tunes whose key or meter changes mid-tune are skipped,
   because their constraints are ambiguous.
2. Optionally transpose it to a randomly chosen key in the same mode, so
   all twelve tonics appear and not only the D, G and A majors folk tunes
   favour.
3. Optionally add a bass voice from the tune's chord symbols. A slash
   chord's bass note wins over its root.
4. Rewrite the tune as clean ABC with ``write_abc``: one bar per bar line,
   ties across bar lines, explicit accidentals spelled for the key, and a
   ``Q:`` tempo field.

Synthetic pieces from ``reference_piece`` cover the requests the tune
collection lacks: three to six voices, 7/8, long pieces and modal keys.
They are tagged ``source="synthetic"`` so they can be down-weighted or
dropped.

Every example is checked with the RL scorer against its own prompt's
constraints. Only examples with full adherence are kept, so SFT never
teaches anything the reward would penalise. Prompts come in English and
Chinese (mirroring the MiMo music split). Half of them carry an explicit
format hint and half do not.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as Fr
from pathlib import Path

from .music import (
    _LETTER_PC,
    _LETTERS,
    MODES,
    Key,
    Note,
    Piece,
    features,
    musicality,
    parse_abc,
    parse_key,
    reference_piece,
    score_abc,
)

MAJOR_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
MINOR_KEYS = ["Am", "Em", "Bm", "F#m", "C#m", "G#m", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"]

STYLES = {  # Nottingham file -> (English, Chinese)
    "ashover": ("English country dance tune", "英式乡村舞曲"),
    "hpps": ("hornpipe", "角笛舞曲"),
    "jigs": ("jig", "吉格舞曲"),
    "morris": ("morris dance tune", "莫里斯舞曲"),
    "playford": ("Playford country dance", "普莱福德乡村舞曲"),
    "reels": ("reel", "里尔舞曲"),
    "slip": ("slip jig", "滑步吉格舞曲"),
    "waltzes": ("waltz", "圆舞曲"),
    "xmas": ("Christmas carol", "圣诞颂歌"),
}
TEMPO = {  # style -> (beat field, low, high); compound meters count dotted quarters
    "jigs": ("3/8", 96, 126), "slip": ("3/8", 96, 120), "reels": ("1/4", 96, 124), "hpps": ("1/4", 76, 100),
    "waltzes": ("1/4", 120, 168), "morris": ("1/4", 80, 112), "playford": ("1/4", 84, 120),
    "ashover": ("1/4", 84, 120), "xmas": ("1/4", 72, 108),
}
_ZH_MODE = {"dorian": "多利亚调式", "mixolydian": "混合利底亚调式", "lydian": "利底亚调式", "phrygian": "弗里吉亚调式",
            "locrian": "洛克利亚调式"}


# ------------------------------------------------------------------ writing ABC


def _dur_token(d: Fr, unit: Fr) -> str:
    m = d / unit
    if m == 1:
        return ""
    if m.denominator == 1:
        return str(m.numerator)
    if m.numerator == 1:
        return "/" if m.denominator == 2 else f"/{m.denominator}"
    return f"{m.numerator}/{m.denominator}"


def _spell(pitch: int, key: Key, state: dict) -> str:
    """Spell a MIDI pitch in ``key``. ``state`` holds the accidentals in force
    in the current bar, (letter, octave) -> semitones, and is updated."""
    pc = pitch % 12
    sig = dict(zip(_LETTERS, key.signature))
    letter = next((L for L in _LETTERS if (_LETTER_PC[L] + sig[L]) % 12 == pc), None)
    if letter is not None:
        semis = sig[letter]
    else:
        # not in the key: prefer a natural sign, then the key's own direction
        # (sharps in sharp keys, flats in flat keys), then the other, then doubles
        flats = any(v < 0 for v in key.signature)
        order = [0, -1, 1, -2, 2] if flats else [0, 1, -1, 2, -2]
        for want in order:
            hit = next((L for L in _LETTERS if (pc - _LETTER_PC[L]) % 12 == want % 12), None)
            if hit is not None:
                letter, semis = hit, want
                break
    octave, rem = divmod(pitch - 60 - _LETTER_PC[letter] - semis, 12)
    assert rem == 0
    in_force = state.get((letter, octave), sig[letter])
    acc = ""
    if semis != in_force:
        acc = {2: "^^", 1: "^", 0: "=", -1: "_", -2: "__"}[semis]
        state[(letter, octave)] = semis
    name = letter if octave <= 0 else letter.lower()
    return acc + name + ("," * -octave if octave < 0 else "'" * (octave - 1))


def _events(notes: list[Note]) -> list[tuple[Fr, Fr, list[int]]]:
    """(start, duration, pitches) with gaps filled by rests and chords grouped."""
    by_start: dict = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)
    out, t = [], Fr(0)
    for s in sorted(by_start):
        if s < t:          # overlapping material is dropped
            continue
        if s > t:
            out.append((t, s - t, []))
        group = by_start[s]
        out.append((s, group[0].dur, [n.pitch for n in group if n.pitch is not None]))
        t = s + group[0].dur
    return out


def _voice_body(notes: list[Note], bars: list[Fr], meter: Fr, key: Key, unit: Fr, per_line: int = 4) -> str:
    bounds, acc = [], Fr(0)
    for b in bars:
        acc += b
        bounds.append(acc)
    out, line, bar_tokens, state, nbar = [], [], [], {}, 0

    def next_bound(t: Fr) -> Fr:
        for b in bounds:
            if b > t:
                return b
        k = 1
        while bounds and bounds[-1] + k * meter <= t:
            k += 1
        return (bounds[-1] if bounds else Fr(0)) + k * meter

    def close_bar():
        nonlocal bar_tokens, state, nbar, line
        line.append("".join(bar_tokens) + "|")
        bar_tokens, state = [], {}
        nbar += 1
        if nbar % per_line == 0:
            out.append("".join(line))
            line = []

    for start, dur, pitches in _events(notes):
        t, end = start, start + dur
        while t < end:
            nb = next_bound(t)
            seg_end = min(end, nb)
            d = seg_end - t
            tok = _dur_token(d, unit)
            if not pitches:
                bar_tokens.append("z" + tok)
            else:
                names = [_spell(pch, key, state) for pch in sorted(pitches)]
                body = names[0] if len(names) == 1 else "[" + "".join(names) + "]"
                tie = "-" if seg_end < end else ""
                bar_tokens.append(body + tok + tie)
            t = seg_end
            if t == nb:
                close_bar()
    if bar_tokens:
        line.append("".join(bar_tokens) + "|")
    if line:
        out.append("".join(line))
    text = "\n".join(out)
    return text[:-1] + "|]" if text.endswith("|") else text


def merge_partial_bars(bars: list[Fr], meter: Fr) -> list[Fr]:
    """Join neighbouring partial bars that add up to one bar (the bar a repeat
    sign cut short plus the next section's pickup), as ``bar_stats`` counts them."""
    out, i = [], 0
    while i < len(bars):
        if i + 1 < len(bars) and bars[i] != meter and bars[i + 1] != meter and bars[i] + bars[i + 1] == meter:
            out.append(meter)
            i += 2
        else:
            out.append(bars[i])
            i += 1
    return out


def write_abc(voices: list[list[Note]], bars: list[Fr], key: Key, *, meter_text: str, title: str,
              tempo_field: str, key_field: str, unit: Fr = Fr(1, 8)) -> str:
    meter = Fr(*map(int, meter_text.split("/")))
    head = ["X:1", f"T:{title}", f"M:{meter_text}", "L:1/8" if unit == Fr(1, 8) else f"L:{unit}",
            f"Q:{tempo_field}", f"K:{key_field}"]
    if len(voices) == 1:
        return "\n".join(head + [_voice_body(voices[0], bars, meter, key, unit)])
    body = []
    for i, notes in enumerate(voices, 1):
        body += [f"V:{i}", _voice_body(notes, bars, meter, key, unit)]
    return "\n".join(head + body)


# ------------------------------------------------------------------ transposition and bass


def key_field(k: Key) -> str:
    if k.mode == "ionian":
        return k.tonic
    if k.mode == "aeolian":
        return k.tonic + "m"
    return f"{k.tonic} {k.mode}"


def transpose_interval(src: Key, dst: Key, pitches: list[int], lo: int = 55, hi: int = 93) -> int:
    t = (dst.tonic_pc - src.tonic_pc) % 12
    if t > 6:
        t -= 12
    if pitches:
        while min(pitches) + t < lo and max(pitches) + t + 12 <= hi:
            t += 12
        while max(pitches) + t > hi and min(pitches) + t - 12 >= lo - 12:
            t -= 12
    return t


_ROOT = re.compile(r"^\s*\(?\s*([A-G])([#b]?)")
_SLASH = re.compile(r"/\s*([A-Ga-g])([#b+-]?)")


def _chord_bass_pc(symbol: str) -> int | None:
    m = _SLASH.search(symbol) or _ROOT.match(symbol)
    if not m:
        return None
    acc = {"#": 1, "+": 1, "b": -1, "-": -1}.get(m.group(2), 0)
    return (_LETTER_PC[m.group(1).upper()] + acc) % 12


def bass_from_chords(chords: list, total: Fr, transpose: int = 0, lo: int = 40) -> list[Note]:
    """One held bass note per chord symbol, in the octave starting at ``lo``."""
    pts = [(t, _chord_bass_pc(sym)) for _, t, sym in chords]
    pts = [(t, pc) for t, pc in pts if pc is not None and t < total]
    out = []
    for i, (t, pc) in enumerate(pts):
        end = pts[i + 1][0] if i + 1 < len(pts) else total
        if end <= t:
            continue
        pitch = lo + ((pc + transpose - lo) % 12)
        out.append(Note(pitch, t, end - t, "2"))
    return out


# ------------------------------------------------------------------ prompts


def key_name_zh(k: Key) -> str:
    acc = {"#": "升", "b": "降"}.get(k.tonic[1:], "")
    if k.mode == "ionian":
        return f"{acc}{k.tonic[0]}大调"
    if k.mode == "aeolian":
        return f"{acc}{k.tonic[0]}小调"
    return f"{k.tonic}{_ZH_MODE[k.mode]}"


def make_prompt(c: dict, style: tuple[str, str], lang: str, rng: random.Random, format_hint: bool) -> str:
    k = parse_key(c["key"])
    meter, bars, voices = c["meter"], c["bars"], c["voices"]
    compound = meter in ("6/8", "9/8", "12/8")
    beat = round(c["tempo"] / 1.5) if compound else round(c["tempo"])
    if lang == "en":
        tempo = (f"dotted quarter = {beat}" if compound else
                 rng.choice([f"quarter note = {beat}", f"{beat} BPM (quarter note)"]))
        v = ("a single melody line" if voices == 1 else
             "two voices, melody and bass (use V: lines)" if voices == 2 else f"{voices} voices (use V: lines)")
        base = rng.choice([
            f"Compose a {style[0]} in {k.name}, {meter} time, at {tempo}, {bars} bars long, for {v}.",
            f"Write a {bars}-bar {style[0]} in {k.name} ({meter}, {tempo}) for {v}.",
            f"I need a {style[0]}: key of {k.name}, meter {meter}, tempo {tempo}, exactly {bars} bars, {v}.",
        ])
        hint = (" Write it in ABC notation inside a ```abc code block, with X:, T:, M:, L:, Q: and K: header "
                "fields." if format_hint else " Answer in ABC notation.")
        return base + hint
    tempo = f"附点四分音符 = {beat}" if compound else rng.choice([f"{beat} BPM", f"四分音符 = {beat}"])
    v = "单声部旋律" if voices == 1 else ("旋律与低音两个声部" if voices == 2 else f"共{voices}个声部")
    base = rng.choice([
        f"创作一首{key_name_zh(k)}、{tempo}、{meter}拍的{style[1]}，篇幅{bars}小节，{v}。",
        f"请写一首{style[1]}：{key_name_zh(k)}，{meter}拍，速度{tempo}，共{bars}小节，{v}。",
    ])
    hint = ("请直接以 ABC 记谱格式输出完整乐曲，放在 ```abc 代码块中，包含 X:、T:、M:、L:、Q:、K: 等头部字段。"
            if format_hint else "请直接以 ABC 记谱格式输出完整乐曲。")
    return base + hint


# ------------------------------------------------------------------ building


@dataclass
class Example:
    id: str
    group: str            # examples from the same tune share a group (used for the split)
    source: str
    license: str
    lang: str
    constraints: dict
    prompt: str
    completion: str
    reward: float

    def to_dict(self) -> dict:
        return {"id": self.id, "group": self.group, "source": self.source, "license": self.license,
                "lang": self.lang, "constraints": self.constraints, "reward": self.reward,
                "messages": [{"role": "user", "content": self.prompt},
                             {"role": "assistant", "content": self.completion}]}


def split_tunes(text: str) -> list[str]:
    return [t for t in re.split(r"(?m)^(?=X:)", text) if t.startswith("X:")]


def _style_of(filename: str) -> str:
    stem = Path(filename).stem
    return "reels" if stem.startswith("reels") else stem


def tunes_from_nottingham(directory: str | Path) -> list[tuple[str, str, str]]:
    """[(style, tune id, abc)] from the Nottingham ABC_cleaned files."""
    out = []
    for f in sorted(Path(directory).glob("*.abc")):
        for tune in split_tunes(f.read_text(encoding="latin-1")):
            x = re.match(r"X:\s*(\d+)", tune)
            out.append((_style_of(f.name), f"{f.stem}:X{x.group(1) if x else '?'}", tune))
    return out


def examples_from_tune(style: str, tune_id: str, abc: str, rng: random.Random, *, transpositions: int = 2,
                       bass: bool = True, min_musicality: float = 0.5) -> tuple[list[Example], Counter]:
    rejected: Counter = Counter()
    if len(re.findall(r"(?m)^K:", abc)) > 1 or len(re.findall(r"(?m)^M:", abc)) > 1 or re.search(r"\[[KM]:", abc):
        rejected["key or meter changes mid-tune"] += 1
        return [], rejected
    p: Piece = parse_abc(abc)
    f = features(p)
    if p.key is None or p.meter_text is None or "1" not in p.voices:
        rejected["unparsed"] += 1
        return [], rejected
    if f["bar_fit"] < 1.0 or f["unknown_ratio"] > 0:
        rejected["irregular bars or unreadable text"] += 1
        return [], rejected
    if musicality(f)["score"] < min_musicality:
        rejected["low musicality"] += 1
        return [], rejected
    melody = p.voices["1"]
    bars = merge_partial_bars(p.bars["1"], p.meter)   # repeats are written out once, so join their halves
    total = sum(bars, Fr(0))
    title = re.search(r"(?m)^T:\s*(.+)$", abc)
    title = title.group(1).strip() if title else "Untitled"
    pool = MAJOR_KEYS if p.key.mode == "ionian" else MINOR_KEYS if p.key.mode == "aeolian" else None
    targets = [p.key] + ([parse_key(k) for k in rng.sample([k for k in pool if not parse_key(k).same_as(p.key)],
                                                          min(transpositions, len(pool) - 1))] if pool else [])
    has_chords = bass and len(p.chords) >= max(2, len(bars) // 2)
    beat, lo, hi = TEMPO.get(style, ("1/4", 84, 120))
    if p.meter_text in ("6/8", "9/8", "12/8"):
        beat = "3/8"
    elif beat == "3/8":
        beat = "1/4"
    out = []
    for target in targets:
        t = transpose_interval(p.key, target, [n.pitch for n in melody if n.pitch is not None])
        mel = [Note(None if n.pitch is None else n.pitch + t, n.start, n.dur, "1") for n in melody]
        for with_bass in ([False, True] if has_chords else [False]):
            voices = [mel]
            if with_bass:
                b = bass_from_chords([c for c in p.chords if c[0] == "1"], total, transpose=t)
                if not b:
                    continue
                voices.append(b)
            bpm = rng.randrange(lo, hi + 1, 2)
            qpm = bpm * 1.5 if beat == "3/8" else bpm
            text = write_abc(voices, bars, target, meter_text=p.meter_text, title=title,
                             tempo_field=f"{beat}={bpm}", key_field=key_field(target))
            full_bars = features(parse_abc(text))["full_bars"]
            c = {"key": key_field(target), "meter": p.meter_text, "tempo": qpm, "bars": full_bars,
                 "voices": len(voices)}
            lang = rng.choice(["en", "zh"])
            prompt = make_prompt(c, STYLES.get(style, (style, style)), lang, rng, format_hint=rng.random() < 0.5)
            completion = f"```abc\n{text}\n```"
            s = score_abc(completion, c)
            if s.adherence.get("score") != 1.0:
                rejected["rewritten tune misses full adherence"] += 1
                continue
            suffix = f"{key_field(target).replace(' ', '')}{'-bass' if with_bass else ''}"
            out.append(Example(f"nottingham:{tune_id}:{suffix}", f"nottingham:{tune_id}", "nottingham",
                               "GPL-3.0", lang, c, prompt, completion, s.reward))
    return out, rejected


_SYN_KEYS = MAJOR_KEYS + MINOR_KEYS + ["D dorian", "E dorian", "G mixolydian", "A mixolydian", "E phrygian",
                                       "F lydian"]
_SYN_STYLES = [("chorale", "众赞歌"), ("folk song", "民歌"), ("march", "进行曲"), ("dance", "舞曲"),
               ("string quartet movement", "弦乐四重奏乐章"), ("brass piece", "铜管乐曲")]


def synthetic_examples(n: int, rng: random.Random, min_musicality: float = 0.5) -> list[Example]:
    out = []
    i = 0
    while len(out) < n:
        i += 1
        voices = rng.choice([3, 3, 4, 4, 5, 6, 1, 2])
        meter = rng.choice(["4/4", "3/4", "2/4", "6/8", "7/8", "9/8"])
        c = {"key": rng.choice(_SYN_KEYS), "meter": meter, "bars": rng.choice([12, 16, 24, 32, 48, 64, 96]),
             "voices": voices}
        compound = meter in ("6/8", "9/8")
        c["tempo"] = (rng.randrange(40, 120, 2) * 1.5) if compound else rng.randrange(60, 168, 2)
        abc = re.sub(r"(?m)^X:\d+$", "X:1", reference_piece(c, seed=rng.randrange(10**9)), count=1)
        if compound and meter == "9/8":
            abc = abc.replace(f"Q:1/4={round(c['tempo'])}", f"Q:3/8={round(c['tempo'] / 1.5)}")
        lang = rng.choice(["en", "zh"])
        prompt = make_prompt(c, rng.choice(_SYN_STYLES), lang, rng, format_hint=rng.random() < 0.5)
        completion = f"```abc\n{abc}\n```"
        s = score_abc(completion, c)
        if s.adherence.get("score") != 1.0 or s.musicality.get("score", 0) < min_musicality:
            continue
        out.append(Example(f"synthetic:{i}", f"synthetic:{i}", "synthetic", "Apache-2.0", lang, c, prompt,
                           completion, s.reward))
    return out


def build_dataset(nottingham_dir: str | Path, *, seed: int = 0, transpositions: int = 2,
                  synthetic_fraction: float = 0.15, val_fraction: float = 0.05) -> tuple[list[dict], list[dict], dict]:
    """-> (train rows, val rows, stats). The split is by tune, so the
    transpositions and bass variants of one tune never straddle it."""
    rng = random.Random(seed)
    examples: list[Example] = []
    rejected: Counter = Counter()
    tunes = tunes_from_nottingham(nottingham_dir)
    for style, tid, abc in tunes:
        ex, rej = examples_from_tune(style, tid, abc, rng, transpositions=transpositions)
        examples += ex
        rejected += rej
    n_syn = int(len(examples) * synthetic_fraction / max(1e-9, 1 - synthetic_fraction))
    examples += synthetic_examples(n_syn, rng)
    groups = sorted({e.group for e in examples})
    rng.shuffle(groups)
    val_groups = set(groups[: max(1, int(len(groups) * val_fraction))])
    train = [e.to_dict() for e in examples if e.group not in val_groups]
    val = [e.to_dict() for e in examples if e.group in val_groups]
    rng.shuffle(train)
    stats = {
        "tunes_in": len(tunes),
        "tunes_used": len({e.group for e in examples if e.source == "nottingham"}),
        "rejected": dict(rejected),
        "examples": len(examples), "train": len(train), "val": len(val),
        "by_source": dict(Counter(e.source for e in examples)),
        "by_lang": dict(Counter(e.lang for e in examples)),
        "by_voices": dict(sorted(Counter(e.constraints["voices"] for e in examples).items())),
        "by_meter": dict(Counter(e.constraints["meter"] for e in examples).most_common()),
        "distinct_keys": len({e.constraints["key"] for e in examples}),
        "mean_reward": round(sum(e.reward for e in examples) / max(1, len(examples)), 4),
    }
    return train, val, stats


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


__all__ = ["build_dataset", "write_abc", "merge_partial_bars", "examples_from_tune", "synthetic_examples", "make_prompt",
           "bass_from_chords", "transpose_interval", "split_tunes", "key_field", "key_name_zh", "write_jsonl",
           "MODES"]
