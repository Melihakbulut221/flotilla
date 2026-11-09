"""Music SFT builder. The tunes below were written for these tests (no
third-party material), in the style of the folk-tune collections the builder
targets: chord symbols, pickups, repeats, ties, accidentals."""

import random
from fractions import Fraction as Fr

import pytest

from flotilla.rl.music import bar_stats, features, parse_abc, parse_key, score_abc
from flotilla.rl.music_sft import (
    MAJOR_KEYS,
    MINOR_KEYS,
    bass_from_chords,
    build_dataset,
    examples_from_tune,
    key_name_zh,
    merge_partial_bars,
    transpose_interval,
    write_abc,
)
from flotilla.verl import music_constraints_from_prompt

REEL = """X:1
T:Test Reel
M:4/4
L:1/8
K:G
|:"G"GABc dBGB|"C"c2e2 "G"d2B2|"Am"ABcA "D7"FGAF|"G"G2B2 G4:|
|:"Em"e2ge "Bm"dBGB|"C"cBcd "G"B2G2|"Am"A2c2 "D7"F2A2|"G"G6 z2:|
"""

JIG = """X:2
T:Test Jig
M:6/8
L:1/8
K:D
A|:"D"dfa afd|"G/b"gbg "A7"e2^c|"D"d2f "Bm"a2-a|"A"e3 e2A|
"D"dfa afd|"G"gbg "A7"e^dc|"D"d3 d2:|
"""


def pitches_and_lengths(abc, voice="1"):
    p = parse_abc(abc)
    return [(n.pitch, n.dur) for n in p.voices[voice]]


def rewrite(abc, key_text=None, t=0, bass=False):
    p = parse_abc(abc)
    key = parse_key(key_text) if key_text else p.key
    from flotilla.rl.music import Note

    mel = [Note(None if n.pitch is None else n.pitch + t, n.start, n.dur, "1") for n in p.voices["1"]]
    voices = [mel]
    bars = merge_partial_bars(p.bars["1"], p.meter)
    if bass:
        voices.append(bass_from_chords(p.chords, sum(bars, Fr(0)), transpose=t))
    return write_abc(voices, bars, key, meter_text=p.meter_text, title="t", tempo_field="1/4=100",
                     key_field=key_text or p.key_text)


def test_rewrite_preserves_the_music():
    for tune in (REEL, JIG):
        assert pitches_and_lengths(rewrite(tune)) == pitches_and_lengths(tune)


@pytest.mark.parametrize("target", ["Eb", "B", "F#", "Ab"])
def test_transposition_shifts_every_pitch_and_respells(target):
    src = parse_abc(JIG)
    t = transpose_interval(src.key, parse_key(target), [n.pitch for n in src.voices["1"] if n.pitch])
    out = rewrite(JIG, target, t)
    assert [p for p, _ in pitches_and_lengths(out)] == [p + t for p, _ in pitches_and_lengths(JIG)]
    assert f"K:{target}" in out
    # the chromatic D# of the original stays one chromatic note, spelled with an explicit accidental
    f = features(parse_abc(out))
    assert f["scale_fit"] == features(parse_abc(JIG))["scale_fit"]


@pytest.mark.parametrize("key", MAJOR_KEYS + MINOR_KEYS + ["D dorian", "G mixolydian"])
def test_every_pitch_class_spells_back_in_every_key(key):
    from flotilla.rl.music import Note

    notes = [Note(60 + i, Fr(i, 8), Fr(1, 8), "1") for i in range(24)]          # two chromatic octaves
    out = write_abc([notes], [Fr(1)] * 3, parse_key(key), meter_text="4/4", title="c", tempo_field="1/4=90",
                    key_field=key)
    assert [p for p, _ in pitches_and_lengths(out)] == list(range(60, 84))


def test_bars_pairing_and_merging():
    bars = [Fr(1, 8), Fr(3, 4), Fr(5, 8), Fr(1, 8), Fr(3, 4), Fr(5, 8)]
    assert bar_stats(bars, Fr(3, 4)) == (3, 4, 4)
    assert merge_partial_bars(bars, Fr(3, 4)) == [Fr(1, 8), Fr(3, 4), Fr(3, 4), Fr(3, 4), Fr(5, 8)]


def test_bass_from_chords_follows_slash_chords():
    p = parse_abc(JIG)
    bass = bass_from_chords(p.chords, sum(p.bars["1"], Fr(0)))
    assert all(40 <= n.pitch <= 51 for n in bass)
    assert sum((n.dur for n in bass), Fr(0)) == sum(p.bars["1"], Fr(0)) - p.chords[0][1]
    # "G/b" puts B in the bass, not G
    g_over_b = next(n for n in bass if n.start == p.chords[1][1])
    assert g_over_b.pitch % 12 == 11


def test_tie_across_a_bar_line_is_written_with_a_tie():
    out = rewrite("X:1\nM:4/4\nL:1/8\nK:C\nc8-|c2 d2 e4|\n")
    assert "c8-|" in out or "c8-" in out
    assert pitches_and_lengths(out) == [(72, Fr(5, 4)), (74, Fr(1, 4)), (76, Fr(1, 2))]


def test_examples_have_full_adherence_and_both_languages():
    rng = random.Random(1)
    ex, rejected = examples_from_tune("jigs", "test:X2", JIG, rng, transpositions=3)
    ex2, _ = examples_from_tune("reels", "test:X1", REEL, rng, transpositions=3)
    ex += ex2
    assert not rejected and len(ex) == 2 * 4 * 2      # 2 tunes x 4 keys x (melody, melody+bass)
    for e in ex:
        assert score_abc(e.completion, e.constraints).adherence["score"] == 1.0
        assert e.license == "GPL-3.0" and e.source == "nottingham"
    assert {e.lang for e in ex} == {"en", "zh"}
    assert {e.constraints["voices"] for e in ex} == {1, 2}
    for e in ex:                                       # what the prompt says is what is graded
        back = music_constraints_from_prompt(e.prompt)
        assert parse_key(back["key"]).same_as(parse_key(e.constraints["key"]))
        assert back["bars"] == e.constraints["bars"]


def test_mid_tune_key_changes_are_rejected():
    tune = REEL.replace("|:\"Em\"", "K:D\n|:\"Em\"")
    ex, rejected = examples_from_tune("reels", "x", tune, random.Random(0))
    assert not ex and rejected["key or meter changes mid-tune"] == 1


def test_key_names_in_chinese():
    assert key_name_zh(parse_key("F#m")) == "升F小调"
    assert key_name_zh(parse_key("Bb")) == "降B大调"
    assert key_name_zh(parse_key("D dorian")) == "D多利亚调式"


def test_build_dataset_splits_by_tune(tmp_path):
    (tmp_path / "jigs.abc").write_text(JIG)
    (tmp_path / "reelsa-c.abc").write_text(REEL)
    train, val, stats = build_dataset(tmp_path, seed=3, transpositions=2, val_fraction=0.3)
    groups_train = {r["group"] for r in train}
    groups_val = {r["group"] for r in val}
    assert groups_val and not groups_train & groups_val
    assert stats["tunes_in"] == 2 and stats["tunes_used"] == 2
    assert stats["by_source"]["synthetic"] >= 1
    for r in train + val:
        assert r["messages"][0]["role"] == "user" and r["messages"][1]["content"].startswith("```abc\nX:1")
