"""Music environment: ABC notation with a prompt-adherence reward.

Single turn: the prompt asks for a piece with concrete constraints (key,
meter, tempo, number of bars, number of voices); the completion is ABC
notation; the reward checks that the piece does what was asked.

Reward = validity x adherence x (0.4 + 0.6 x musicality), in [0, 1].

The product means neither part can be ignored:

- a fluent piece in the wrong key, meter and tempo has low adherence, so it
  scores low;
- a drone that satisfies every header field has low musicality, so it
  scores about half.

* Adherence. Each requested constraint is scored, and a header claim only
  earns credit if the music backs it:

  - **key:** the ``K:`` field must name the key, and the notes must fit its
    scale;
  - **meter:** ``M:`` must match, and the bars must actually have that
    length;
  - **tempo:** taken from ``Q:``;
  - **voices:** the number of voices that contain notes;
  - **bars:** the number of full bars.

  Writing ``K:Dm`` over notes in another key earns nothing for the key.
* Musicality. Heuristics that need no reference corpus: scale fit, a
  balance of steps and leaps, few huge leaps, not too many repeated notes,
  a sensible range, more than one note length, motifs that repeat without
  looping, and an ending on the tonic. They catch degenerate output (one
  note held forever, random pitches, an endless loop). They are not a model
  of taste.

  The mean of these parts is scaled by a degeneracy gate: the worst of the
  repeats, range, motifs and scale-fit scores. A drone, a loop or random
  chromatic notes cannot make up for that with free points elsewhere (no
  big leaps, a tonic ending).
* Validity. The parse must succeed with enough notes, and the share of
  characters the parser could not read must stay low. Otherwise the reward
  is 0.

This is the piece missing from scorers that only measure similarity to
human music statistics: there, a fluent piece in the wrong key, meter and
tempo can score as well as one that follows the prompt.

The parser covers the ABC subset LLMs actually write:

- the header fields ``X T M L Q K V``, plus inline ``[K:]``, ``[M:]``,
  ``[L:]``, ``[V:]`` and ``[Q:]``;
- notes with accidentals, octave marks and lengths, plus rests ``z x Z``;
- chords ``[CEG]``;
- ties, broken rhythm (``>`` ``<``) and tuplets ``(3``;
- bar lines and repeats.

It skips decorations, chord symbols, grace notes and lyrics. It never
executes anything, and its input is capped in size.
"""

from __future__ import annotations

import json
import random
import re
import struct
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import Any

MAX_CHARS = 30_000
MAX_NOTES = 8_000

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_LETTERS = "CDEFGAB"
MODES = {
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}
_MODE_ALIASES = [
    ("mixolydian", "mixolydian"), ("phrygian", "phrygian"), ("aeolian", "aeolian"), ("locrian", "locrian"),
    ("dorian", "dorian"), ("lydian", "lydian"), ("ionian", "ionian"), ("major", "ionian"), ("minor", "aeolian"),
    ("maj", "ionian"), ("min", "aeolian"), ("mix", "mixolydian"), ("dor", "dorian"), ("phr", "phrygian"),
    ("lyd", "lydian"), ("loc", "locrian"), ("aeo", "aeolian"), ("ion", "ionian"), ("m", "aeolian"),
]
_KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)\s*([A-Za-z]*)")


# ------------------------------------------------------------------ keys and meters


@dataclass(frozen=True)
class Key:
    tonic: str          # e.g. "F#"
    mode: str           # one of MODES
    tonic_pc: int
    signature: tuple    # accidental (semitones) per letter C..B

    @property
    def scale(self) -> frozenset:
        return frozenset((self.tonic_pc + i) % 12 for i in MODES[self.mode])

    @property
    def name(self) -> str:
        if self.mode == "ionian":
            return f"{self.tonic} major"
        if self.mode == "aeolian":
            return f"{self.tonic} minor"
        return f"{self.tonic} {self.mode}"

    def same_as(self, other: "Key") -> bool:
        return self.tonic_pc == other.tonic_pc and MODES[self.mode] == MODES[other.mode]


def parse_key(text: str) -> Key | None:
    """'D', 'Dm', 'F# minor', 'Bb mix', 'E dorian' -> Key. 'none'/'HP' -> C major."""
    t = (text or "").strip()
    low = t.lower()
    if low.startswith("none") or low.startswith("hp") or not t:
        return make_key("C", "ionian")
    m = _KEY_RE.match(t)
    if not m:
        return None
    letter, acc, mode_word = m.group(1).upper(), m.group(2), m.group(3).lower()
    mode = "ionian"
    if mode_word:
        for alias, full in _MODE_ALIASES:
            if mode_word.startswith(alias) and (alias != "m" or mode_word == "m" or mode_word.startswith("min")):
                mode = full
                break
        else:
            if not mode_word.startswith(("clef", "treble", "bass", "alto", "tenor", "transpose")):
                return None
    return make_key(letter + acc, mode)


def make_key(tonic: str, mode: str) -> Key:
    letter, acc = tonic[0].upper(), tonic[1:]
    tonic_pc = (_LETTER_PC[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12
    start = _LETTERS.index(letter)
    sig = [0] * 7
    for degree, interval in enumerate(MODES[mode]):
        li = (start + degree) % 7
        target = (tonic_pc + interval) % 12
        d = (target - _LETTER_PC[_LETTERS[li]]) % 12
        sig[li] = d - 12 if d > 6 else d
    return Key(letter + acc, mode, tonic_pc, tuple(sig))


def parse_meter(text: str) -> Fr | None:
    """'6/8' -> 3/4 (whole notes per bar), 'C' -> 1, 'C|' -> 1, '2+3/8' -> 5/8."""
    t = (text or "").strip()
    if t.startswith("C|"):
        return Fr(1)
    if t.startswith("C"):
        return Fr(1)
    m = re.match(r"^\(?([\d+]+)\)?\s*/\s*(\d+)", t)
    if not m:
        return None
    try:
        num = sum(int(x) for x in m.group(1).split("+") if x)
        den = int(m.group(2))
    except ValueError:
        return None
    return Fr(num, den) if num > 0 and den > 0 else None


def meter_text(text: str) -> str | None:
    t = (text or "").strip()
    if t.startswith("C|"):
        return "2/2"
    if t.startswith("C"):
        return "4/4"
    m = re.match(r"^\(?([\d+]+)\)?\s*/\s*(\d+)", t)
    if not m:
        return None
    return f"{sum(int(x) for x in m.group(1).split('+') if x)}/{m.group(2)}"


def parse_tempo(text: str, unit: Fr) -> float | None:
    """Q: field -> quarter notes per minute. '1/4=120', '3/8=60', '"Allegro" 1/4=120', '120'."""
    t = re.sub(r'"[^"]*"', " ", text or "")
    m = re.search(r"((?:\d+/\d+\s*)+)=\s*(\d+(?:\.\d+)?)", t)
    if m:
        beat = sum((Fr(f) for f in m.group(1).split()), Fr(0))
        return float(m.group(2)) * float(beat / Fr(1, 4))
    m = re.search(r"C\s*=\s*(\d+)", t)
    if m:
        return float(m.group(1)) * float(unit / Fr(1, 4))
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", t)
    if m:
        return float(m.group(1)) * float(unit / Fr(1, 4))
    return None


# ------------------------------------------------------------------ parsing


@dataclass
class Note:
    pitch: int | None     # MIDI number; None for a rest
    start: Fr             # in whole notes
    dur: Fr
    voice: str


@dataclass
class _Voice:
    time: Fr = Fr(0)
    bar_dur: Fr = Fr(0)
    bars: list = field(default_factory=list)
    accidentals: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    tie: bool = False
    tuplet_left: int = 0
    tuplet_factor: Fr = Fr(1)
    broken_next: Fr = Fr(1)


@dataclass
class Piece:
    title: str = ""
    key: Key | None = None
    key_text: str = ""
    meter: Fr | None = None
    meter_text: str | None = None
    unit: Fr | None = None
    tempo_qpm: float | None = None
    voices: dict = field(default_factory=dict)      # id -> list[Note]
    bars: dict = field(default_factory=dict)        # id -> list[Fr] (bar durations)
    unknown_chars: int = 0
    body_chars: int = 0
    errors: list = field(default_factory=list)

    @property
    def notes(self) -> list:
        return [n for ns in self.voices.values() for n in ns]


_NOTE = re.compile(r"(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)(\d*)(/*)(\d*)")
_REST = re.compile(r"([zx])(\d*)(/*)(\d*)")
_MREST = re.compile(r"Z(\d*)")
_FIELD_LINE = re.compile(r"^([A-Za-z]):(.*)$")
_DECORATION_LETTERS = set("HIJKLMNOPQRSTUVWYhijklmnopqrstuvwy~.")


def _length(num: str, slashes: str, den: str) -> Fr:
    n = Fr(int(num)) if num else Fr(1)
    if slashes:
        d = Fr(int(den)) if den else Fr(2 ** len(slashes))
        n = n / d
    return n


def extract_abc(text: str) -> str | None:
    """The ABC tune in a completion: a ```abc block, else text from 'X:' on."""
    if not isinstance(text, str):
        return None
    m = re.findall(r"```[ \t]*(?:abc)?[ \t]*\n(.*?)```", text, re.S | re.I)
    for block in reversed(m):
        if re.search(r"^K:", block, re.M):
            return block
    m = re.search(r"^X:.*", text, re.M | re.S)
    if m and re.search(r"^K:", m.group(0), re.M):
        return m.group(0)
    return None


def parse_abc(src: str) -> Piece:
    p = Piece()
    if len(src) > MAX_CHARS:
        p.errors.append(f"tune longer than {MAX_CHARS} characters")
        src = src[:MAX_CHARS]
    voices: dict[str, _Voice] = {}
    cur = ["1"]
    in_body = [False]
    key_sig = [(0,) * 7]

    def voice() -> _Voice:
        return voices.setdefault(cur[0], _Voice())

    def set_field(f: str, val: str) -> None:
        val = val.split("%", 1)[0].strip()
        if f == "T" and not p.title:
            p.title = val
        elif f == "M":
            p.meter, p.meter_text = parse_meter(val), meter_text(val)
            if p.unit is None and not in_body[0]:
                p.unit = Fr(1, 16) if p.meter is not None and p.meter < Fr(3, 4) else Fr(1, 8)
        elif f == "L":
            m = re.match(r"\s*(\d+)\s*/\s*(\d+)", val)
            if m and int(m.group(1)) > 0 and int(m.group(2)) > 0:
                p.unit = Fr(int(m.group(1)), int(m.group(2)))
        elif f == "Q":
            q = parse_tempo(val, p.unit or Fr(1, 8))
            if q is not None and p.tempo_qpm is None:
                p.tempo_qpm = q
        elif f == "K":
            k = parse_key(val)
            if k is None:
                p.errors.append(f"unreadable key {val!r}")
            else:
                if p.key is None:
                    p.key, p.key_text = k, val
                key_sig[0] = k.signature
            in_body[0] = True
        elif f == "V":
            vid = val.split()[0] if val.split() else "1"
            cur[0] = vid
            voice()

    def close_bar(v: _Voice) -> None:
        if v.bar_dur > 0:
            v.bars.append(v.bar_dur)
        v.bar_dur = Fr(0)
        v.accidentals = {}

    def add(v: _Voice, pitch: int | None, dur: Fr) -> None:
        if len(v.notes) >= MAX_NOTES:
            return
        dur = dur * v.broken_next
        v.broken_next = Fr(1)
        if v.tuplet_left > 0:
            dur *= v.tuplet_factor
            v.tuplet_left -= 1
        if v.tie and pitch is not None and v.notes and v.notes[-1].pitch == pitch:
            v.notes[-1].dur += dur          # tied: one longer note
        else:
            v.notes.append(Note(pitch, v.time, dur, cur[0]))
        v.tie = False
        v.time += dur
        v.bar_dur += dur

    def pitch_of(v: _Voice, acc: str | None, letter: str, octs: str) -> int:
        up = letter.upper()
        octave = 0 if letter.isupper() else 1
        octave += octs.count("'") - octs.count(",")
        key = (up, octave)
        if acc:
            semis = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}[acc]
            v.accidentals[key] = semis
        elif key in v.accidentals:
            semis = v.accidentals[key]
        else:
            semis = key_sig[0][_LETTERS.index(up)]
        return 60 + 12 * octave + _LETTER_PC[up] + semis

    for raw in src.splitlines():
        line = raw.rstrip()
        if line.startswith("%"):
            continue
        fm = _FIELD_LINE.match(line)
        if fm and fm.group(1) in "XTMLQKVCOARSNZBDGHIFPW" + "w":
            if fm.group(1) not in "wW":
                set_field(fm.group(1), fm.group(2))
            continue
        if not in_body[0]:
            continue
        line = line.split("%", 1)[0]
        i, n = 0, len(line)
        while i < n:
            c = line[i]
            v = voice()
            p.body_chars += 1
            if c in " \t\\`":
                i += 1
            elif c == '"':
                j = line.find('"', i + 1)
                i = n if j == -1 else j + 1
            elif c in "!+" and line.find(c, i + 1) != -1 and line.find(c, i + 1) - i < 40:
                i = line.find(c, i + 1) + 1
            elif c == "{":
                j = line.find("}", i + 1)
                i = n if j == -1 else j + 1
            elif c == "[" and i + 2 < n and line[i + 1].isalpha() and line[i + 2] == ":":
                j = line.find("]", i)
                j = n if j == -1 else j
                set_field(line[i + 1], line[i + 3:j])
                i = j + 1
            elif c == "[" and i + 1 < n and line[i + 1].isdigit():
                i += 2
            elif c == "[" and i + 1 < n and line[i + 1] == "|":
                close_bar(v)
                i += 2
            elif c == "[":
                j = line.find("]", i)
                if j == -1:
                    p.unknown_chars += 1
                    i += 1
                    continue
                inner = line[i + 1:j]
                chord = [m for m in _NOTE.finditer(inner)]
                k = j + 1
                m = re.match(r"(\d*)(/*)(\d*)", line[k:])
                outer = _length(*m.groups()) if m and m.group(0) else Fr(1)
                k += len(m.group(0)) if m else 0
                if chord:
                    unit = p.unit or Fr(1, 8)
                    first_len = _length(chord[0].group(4), chord[0].group(5), chord[0].group(6))
                    dur = unit * first_len * outer
                    start = v.time
                    pitches = [pitch_of(v, ch.group(1), ch.group(2), ch.group(3)) for ch in chord]
                    add(v, pitches[0], dur)
                    for extra in pitches[1:]:
                        if len(v.notes) < MAX_NOTES:
                            v.notes.append(Note(extra, start, v.notes[-1].dur, cur[0]))
                else:
                    p.unknown_chars += 1
                i = k
            elif c in "|:":
                j = i
                while j < n and line[j] in "|:]":
                    j += 1
                close_bar(v)
                while j < n and (line[j].isdigit() or line[j] in ",-"):
                    j += 1
                i = j
            elif c == "]":
                i += 1
            elif c == "(":
                m = re.match(r"\((\d)(?::(\d)?)?(?::(\d))?", line[i:])
                if m:
                    pn = int(m.group(1))
                    q = int(m.group(2)) if m.group(2) else (3 if pn in (2, 4, 8) else 2)
                    r = int(m.group(3)) if m.group(3) else pn
                    if pn > 0:
                        v.tuplet_factor, v.tuplet_left = Fr(q, pn), r
                    i += len(m.group(0))
                else:
                    i += 1
            elif c == ")":
                i += 1
            elif c == "-":
                v.tie = True
                i += 1
            elif c in "<>":
                j = i
                while j < n and line[j] == c:
                    j += 1
                depth = min(j - i, 3)
                short = Fr(1, 2 ** depth)
                long_ = 2 - short
                if v.notes:
                    prev = v.notes[-1]
                    old = prev.dur
                    factor = long_ if c == ">" else short
                    for x in v.notes[-8:]:          # the previous note and its chord tones
                        if x.start == prev.start:
                            x.dur = old * factor
                    delta = old * (factor - 1)
                    v.time += delta
                    v.bar_dur += delta
                v.broken_next = short if c == ">" else long_
                i = j
            else:
                m = _NOTE.match(line, i)
                if m:
                    dur = (p.unit or Fr(1, 8)) * _length(m.group(4), m.group(5), m.group(6))
                    add(v, pitch_of(v, m.group(1), m.group(2), m.group(3)), dur)
                    i = m.end()
                    continue
                m = _REST.match(line, i)
                if m:
                    add(v, None, (p.unit or Fr(1, 8)) * _length(m.group(2), m.group(3), m.group(4)))
                    i = m.end()
                    continue
                m = _MREST.match(line, i)
                if m:
                    count = int(m.group(1) or 1)
                    bar = p.meter or Fr(1)
                    close_bar(v)
                    for _ in range(min(count, 512)):
                        v.notes.append(Note(None, v.time, bar, cur[0]))
                        v.time += bar
                        v.bars.append(bar)
                    i = m.end()
                    continue
                if c in _DECORATION_LETTERS:
                    i += 1
                    continue
                p.unknown_chars += 1
                i += 1
    for vid, v in voices.items():
        close_bar(v)
        if any(nt.pitch is not None for nt in v.notes):
            p.voices[vid] = v.notes
            p.bars[vid] = v.bars
    if p.key is None:
        p.errors.append("missing K: field")
    return p


# ------------------------------------------------------------------ scoring


def _band(x: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """Trapezoid: 0 at or below lo0, rising to 1 on [lo1, hi1], 0 at or above hi0."""
    if x < lo1:
        return 0.0 if x <= lo0 else (x - lo0) / (lo1 - lo0)
    if x > hi1:
        return 0.0 if x >= hi0 else (hi0 - x) / (hi0 - hi1)
    return 1.0


def features(p: Piece, key: Key | None = None) -> dict:
    """Numbers the reward is built from. ``key`` defaults to the declared key."""
    key = key or p.key
    pitched = [n for n in p.notes if n.pitch is not None]
    top_id = next(iter(p.voices), None)
    top = [n.pitch for n in p.voices.get(top_id, []) if n.pitch is not None]
    iv = [abs(b - a) for a, b in zip(top, top[1:])]
    grams = [tuple(top[i:i + 4]) for i in range(len(top) - 3)]
    durs = {n.dur for n in p.voices.get(top_id, []) if n.pitch is not None}
    full_bars, interior_ok, interior = 0, 0, 0
    for bars in p.bars.values():
        if p.meter:
            full_bars = max(full_bars, sum(1 for b in bars if b == p.meter))
            inner = bars[1:-1] if len(bars) > 2 else bars
            interior += len(inner)
            interior_ok += sum(1 for b in inner if b == p.meter)
    return {
        "notes": len(pitched),
        "voices": len(p.voices),
        "full_bars": full_bars,
        "bar_fit": interior_ok / interior if interior else 0.0,
        "scale_fit": (sum(1 for n in pitched if n.pitch % 12 in key.scale) / len(pitched)) if key and pitched else 0.0,
        "step_ratio": sum(1 for d in iv if 1 <= d <= 2) / len(iv) if iv else 0.0,
        "repeat_ratio": sum(1 for d in iv if d == 0) / len(iv) if iv else 1.0,
        "big_leap_ratio": sum(1 for d in iv if d > 12) / len(iv) if iv else 0.0,
        "range": (max(top) - min(top)) if top else 0,
        "distinct_durations": len(durs),
        "uniqueness": len(set(grams)) / len(grams) if grams else 0.0,
        "ends_on_tonic": bool(top and key and top[-1] % 12 == key.tonic_pc),
        "unknown_ratio": p.unknown_chars / p.body_chars if p.body_chars else 1.0,
    }


def musicality(f: dict) -> dict:
    parts = {
        "scale_fit": _band(f["scale_fit"], 0.6, 0.9, 1.0, 1.01),
        "steps": _band(f["step_ratio"], 0.1, 0.35, 0.85, 1.0),
        "repeats": _band(f["repeat_ratio"], -0.01, 0.0, 0.3, 0.7),
        "big_leaps": _band(f["big_leap_ratio"], -0.01, 0.0, 0.02, 0.15),
        "range": _band(f["range"], 4, 9, 24, 36),
        "rhythm": min(1.0, {0: 0.0, 1: 0.3, 2: 0.8}.get(f["distinct_durations"], 1.0)),
        "motifs": _band(f["uniqueness"], 0.15, 0.35, 0.95, 1.2),
        "cadence": 1.0 if f["ends_on_tonic"] else 0.0,
    }
    mean = sum(parts.values()) / len(parts)
    parts["gate"] = min(parts["repeats"], parts["range"], parts["motifs"], parts["scale_fit"])
    parts["score"] = mean * (0.5 + 0.5 * parts["gate"])
    return parts


def adherence(p: Piece, c: dict, f_target: dict | None) -> dict:
    """Score each requested constraint; header claims need musical support."""
    out = {}
    if "key" in c:
        want = parse_key(c["key"])
        header_ok = bool(p.key and want and p.key.same_as(want))
        fit = _band(f_target["scale_fit"], 0.6, 0.85, 1.0, 1.01) if f_target else 0.0
        out["key"] = fit if header_ok else 0.0
    if "meter" in c:
        header_ok = p.meter_text == meter_text(c["meter"])
        f = features(p)
        out["meter"] = _band(f["bar_fit"], 0.5, 0.95, 1.0, 1.01) if header_ok else 0.0
    if "tempo" in c:
        if p.tempo_qpm is None:
            out["tempo"] = 0.0
        else:
            err = abs(p.tempo_qpm / float(c["tempo"]) - 1)
            out["tempo"] = 1.0 if err <= 0.05 else max(0.0, 1 - (err - 0.05) / 0.20)
    if "voices" in c:
        d = abs(len(p.voices) - int(c["voices"]))
        out["voices"] = 1.0 if d == 0 else 0.3 if d == 1 else 0.0
    if "bars" in c:
        want = int(c["bars"])
        got = features(p)["full_bars"]
        miss = max(0, abs(got - want) - 1)        # one bar of slack for pickups and endings
        out["bars"] = max(0.0, 1 - miss / (0.5 * want))
    out["score"] = sum(out.values()) / len(out) if out else 1.0
    return out


@dataclass
class MusicScore:
    reward: float
    status: str                         # scored | format | invalid
    adherence: dict = field(default_factory=dict)
    musicality: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def score_abc(text: str, constraints: dict, *, min_notes: int = 8, w_musicality: float = 0.6) -> MusicScore:
    abc = extract_abc(text)
    if abc is None:
        return MusicScore(0.0, "format", error="no ABC tune with a K: field found")
    p = parse_abc(abc)
    f = features(p)
    if p.key is None or f["notes"] < min_notes:
        return MusicScore(0.0, "invalid", features=f,
                          error="; ".join(p.errors) or f"fewer than {min_notes} notes")
    validity = max(0.0, 1 - 5 * f["unknown_ratio"])
    want = parse_key(constraints["key"]) if "key" in constraints else None
    f_target = features(p, want) if want else None
    adh = adherence(p, constraints, f_target)
    mus = musicality(f)
    reward = validity * adh["score"] * (1 - w_musicality + w_musicality * mus["score"])
    return MusicScore(round(reward, 4), "scored", adh, mus, f,
                      error="; ".join(p.errors) or None)


class MusicReward:
    """Batch scorer with the same shape as CodeVerifier (no sandbox needed:
    nothing is executed, the ABC text is only parsed)."""

    def __init__(self, repetition_zero: bool = True, **score_kwargs):
        self.repetition_zero = repetition_zero
        self.score_kwargs = score_kwargs

    def score(self, completions: list[Any], tasks: list[dict]) -> list[MusicScore]:
        from .code import completion_text
        from .shaping import is_degenerate

        out = []
        for c, t in zip(completions, tasks):
            text = completion_text(c)
            cons = t.get("constraints", {})
            if isinstance(cons, str):
                cons = json.loads(cons)
            if self.repetition_zero and is_degenerate(text):
                out.append(MusicScore(0.0, "repetition", error="degenerate repetition"))
            else:
                out.append(score_abc(text, cons, **self.score_kwargs))
        return out

    def rewards(self, completions: list[Any], tasks: list[dict]) -> list[float]:
        return [s.reward for s in self.score(completions, tasks)]


def trl_music_reward(scorer: MusicReward, constraints_key: str = "constraints"):
    """Adapter for TRL's GRPOTrainer; the dataset needs a ``constraints`` column."""

    def music_reward(prompts=None, completions=None, **kwargs) -> list[float]:
        return scorer.rewards(completions, [{"constraints": c} for c in kwargs[constraints_key]])

    music_reward.__name__ = "music_reward"
    return music_reward


# ------------------------------------------------------------------ tasks


_KEYS = ["C", "G", "D", "A", "F", "Bb", "Eb", "Am", "Em", "Bm", "Dm", "Gm", "Cm", "D dorian", "G mixolydian",
         "E phrygian"]
_STYLES = {"4/4": ["march", "folk song", "hymn", "reel", "chorale"], "3/4": ["waltz", "minuet", "lullaby"],
           "6/8": ["jig", "barcarolle", "lullaby"], "2/4": ["polka", "march", "dance"]}


def make_music_task(rng: random.Random, idx: int = 0) -> dict:
    key = rng.choice(_KEYS)
    meter = rng.choice(list(_STYLES))
    style = rng.choice(_STYLES[meter])
    bars = rng.choice([8, 12, 16, 16, 24, 32])
    voices = rng.choice([1, 1, 2])
    k = parse_key(key)
    if meter == "6/8":
        beat_bpm = rng.randrange(40, 120, 4)
        qpm, tempo_words, q_field = beat_bpm * 1.5, f"dotted quarter = {beat_bpm}", f"Q:3/8={beat_bpm}"
    else:
        qpm = rng.randrange(60, 168, 4)
        tempo_words, q_field = f"quarter note = {qpm}", f"Q:1/4={qpm}"
    voice_words = "a single melody line" if voices == 1 else f"{voices} voices (use V: lines)"
    prompt = (f"Compose a {style} in {k.name}, {meter} time, at {tempo_words}, {bars} bars long, for "
              f"{voice_words}. Write it in ABC notation inside a ```abc code block, with X:, T:, M:, L:, "
              f"Q: (e.g. {q_field}) and K: header fields.")
    return {"id": f"music-{idx:04d}", "prompt": prompt,
            "constraints": {"key": key, "meter": meter, "tempo": qpm, "bars": bars, "voices": voices}}


def make_music_tasks(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [make_music_task(rng, i) for i in range(n)]


_RHYTHMS = {8: [[2, 1, 1, 2, 2], [1, 1, 1, 1, 2, 2], [2, 2, 1, 1, 2]], 6: [[2, 1, 2, 1], [3, 1, 1, 1], [1, 1, 1, 3]],
            4: [[1, 1, 2], [2, 1, 1], [2, 2]]}


def reference_piece(c: dict, seed: int = 0) -> str:
    """A plain piece that satisfies ``c`` exactly: a diatonic random walk in
    the requested key and meter, ending on the tonic, with a bass voice when
    two voices are asked for. Used to check that every task is solvable and
    that the scorer gives full adherence to a compliant piece."""
    rng = random.Random(seed)
    key = parse_key(c["key"])
    num, den = (int(x) for x in c["meter"].split("/"))
    eighths = int(Fr(num, den) / Fr(1, 8))
    rhythms = _RHYTHMS.get(eighths, [[1] * eighths])
    start = _LETTERS.index(key.tonic[0])

    def token(a: int, n: int) -> str:
        letter, octave = _LETTERS[a % 7], a // 7
        t = letter if octave <= 0 else letter.lower()
        t += "," * max(0, -octave) + "'" * max(0, octave - 1)
        return t + ("" if n == 1 else str(n))

    a, lines = start, []
    melody = []
    for bar in range(int(c["bars"])):
        if bar == int(c["bars"]) - 1:
            melody.append(token(start + 7 if a > start + 3 else start, eighths))
            break
        notes = []
        for n in rng.choice(rhythms):
            a = max(start - 2, min(start + 9, a + rng.choice([-2, -1, -1, 1, 1, 2, 0, 3])))
            notes.append(token(a, n))
        melody.append("".join(notes))
    q = (f"Q:3/8={round(float(c['tempo']) / 1.5)}" if c["meter"] == "6/8"
         else f"Q:1/4={round(float(c['tempo']))}")
    head = [f"X:{seed + 1}", "T:Reference", f"M:{c['meter']}", "L:1/8", q, f"K:{c['key']}"]
    if int(c.get("voices", 1)) == 1:
        return "\n".join(head + ["|".join(melody) + "|]"])
    bass = []
    for bar in range(int(c["bars"])):
        deg = 0 if bar % 4 in (0, 3) or bar == int(c["bars"]) - 1 else 4
        bass.append(token(start - 7 + deg, eighths))
    body = ["V:1", "|".join(melody) + "|]", "V:2", "|".join(bass) + "|]"]
    return "\n".join(head + body)


# ------------------------------------------------------------------ MIDI


def _varlen(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def to_midi(p: Piece, ticks_per_quarter: int = 480, program: int = 0) -> bytes:
    """Type-1 standard MIDI file, one track per voice."""
    tempo = int(60_000_000 / (p.tempo_qpm or 120))
    tracks = [b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big") + b"\x00\xff\x2f\x00"]
    for ch, notes in enumerate(p.voices.values()):
        ch %= 16
        events = []
        for nt in notes:
            if nt.pitch is None or not 0 <= nt.pitch <= 127:
                continue
            on = int(nt.start * 4 * ticks_per_quarter)
            off = int((nt.start + nt.dur) * 4 * ticks_per_quarter)
            events.append((on, 1, bytes([0x90 | ch, nt.pitch, 80])))
            events.append((off, 0, bytes([0x80 | ch, nt.pitch, 0])))
        events.sort(key=lambda e: (e[0], e[1]))
        body, last = bytearray(b"\x00" + bytes([0xC0 | ch, program])), 0
        for t, _, data in events:
            body += _varlen(t - last) + data
            last = t
        body += b"\x00\xff\x2f\x00"
        tracks.append(bytes(body))
    out = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ticks_per_quarter)
    for t in tracks:
        out += b"MTrk" + struct.pack(">I", len(t)) + t
    return out
