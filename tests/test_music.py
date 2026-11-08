from fractions import Fraction as Fr

import pytest

from flotilla.rl.music import (
    MusicReward,
    extract_abc,
    make_key,
    make_music_tasks,
    parse_abc,
    parse_key,
    parse_meter,
    parse_tempo,
    reference_piece,
    score_abc,
    to_midi,
    trl_music_reward,
)

HEAD = "X:1\nM:4/4\nL:1/8\nK:{key}\n"


def notes(body, key="C", head=None):
    p = parse_abc((head or HEAD.format(key=key)) + body)
    return [(n.pitch, n.dur) for n in p.voices["1"]], p


# ------------------------------------------------------------------ parser


@pytest.mark.parametrize("text,tonic,mode", [
    ("C", "C", "ionian"), ("Am", "A", "aeolian"), ("A minor", "A", "aeolian"), ("Amin", "A", "aeolian"),
    ("F#m", "F#", "aeolian"), ("Bb", "Bb", "ionian"), ("D dor", "D", "dorian"), ("G Mix", "G", "mixolydian"),
    ("E phrygian", "E", "phrygian"), ("C clef=bass", "C", "ionian"), ("Gmaj", "G", "ionian"),
])
def test_parse_key(text, tonic, mode):
    k = parse_key(text)
    assert (k.tonic, k.mode) == (tonic, mode)


def test_key_signatures():
    assert make_key("G", "ionian").signature == (0, 0, 0, 1, 0, 0, 0)          # F#
    assert make_key("F#", "aeolian").signature == (1, 0, 0, 1, 1, 0, 0)      # F# C# G#
    assert make_key("Eb", "ionian").signature == (0, 0, -1, 0, 0, -1, -1)    # Bb Eb Ab
    assert make_key("D", "dorian").signature == (0,) * 7
    assert parse_key("Dm").same_as(parse_key("D aeolian")) and not parse_key("Dm").same_as(parse_key("D"))


def test_meter_and_tempo():
    assert parse_meter("6/8") == Fr(3, 4) and parse_meter("C") == 1 and parse_meter("C|") == 1
    assert parse_meter("2+3/8") == Fr(5, 8) and parse_meter("none") is None
    assert parse_tempo("1/4=120", Fr(1, 8)) == 120
    assert parse_tempo("3/8=60", Fr(1, 8)) == 90
    assert parse_tempo('"Allegro" 1/4=132', Fr(1, 8)) == 132
    assert parse_tempo("120", Fr(1, 8)) == 60       # bare number counts L units


def test_pitches_octaves_and_accidentals():
    ns, _ = notes("C, C c c' | ^F F =F | F", key="C")
    assert [p for p, _ in ns] == [48, 60, 72, 84, 66, 66, 65, 65]
    ns, _ = notes("F =F F | F _B", key="G")
    assert [p for p, _ in ns] == [66, 65, 65, 66, 70]


def test_lengths_tuplets_broken_ties_chords():
    ns, _ = notes("A2 A/2 A/ A// A3/2")
    assert [d for _, d in ns] == [Fr(1, 4), Fr(1, 16), Fr(1, 16), Fr(1, 32), Fr(3, 16)]
    ns, _ = notes("(3abc d")
    assert [d for _, d in ns] == [Fr(1, 12)] * 3 + [Fr(1, 8)]
    ns, _ = notes("a>b c<d")
    assert [d for _, d in ns] == [Fr(3, 16), Fr(1, 16), Fr(1, 16), Fr(3, 16)]
    ns, _ = notes("A2-A2 B")
    assert ns == [(69, Fr(1, 2)), (71, Fr(1, 8))]
    ns, p = notes("[CEG]2 c")
    assert [x for x, _ in ns] == [60, 64, 67, 72] and p.voices["1"][1].start == 0
    assert p.voices["1"][3].start == Fr(1, 4)


def test_bars_voices_rests_and_ignored_markup():
    tune = ('X:1\nM:3/4\nL:1/4\nV:1\nV:2\nK:D\n[V:1] "D"!trill!A B c | d3 |]\n'
            "[V:2] D,3 | Z2 |]\nw: la la la\n")
    p = parse_abc(tune)
    assert set(p.voices) == {"1", "2"}
    assert p.bars["1"] == [Fr(3, 4), Fr(3, 4)]
    assert p.bars["2"] == [Fr(3, 4), Fr(3, 4), Fr(3, 4)]
    assert p.unknown_chars == 0


def test_repeats_endings_and_pickup():
    p = parse_abc(HEAD.format(key="G") + "D|:G2B2 d4|[1 c2A2 F4:|[2 c2A2 G4|]")
    assert p.bars["1"] == [Fr(1, 8), 1, 1, 1]


def test_extract_abc():
    assert extract_abc("Here:\n```abc\nX:1\nK:C\nCDE\n```") == "X:1\nK:C\nCDE\n"
    assert extract_abc("Sure!\nX:1\nT:t\nK:C\nCDEF").startswith("X:1")
    assert extract_abc("no tune here") is None
    assert extract_abc("```abc\nCDEF\n```") is None      # no K: field


def test_parser_is_bounded():
    p = parse_abc("X:1\nK:C\n" + "C" * 100_000)
    assert len(p.voices["1"]) <= 8000 and p.errors


# ------------------------------------------------------------------ reward

C = {"key": "Dm", "meter": "6/8", "tempo": 120, "bars": 8, "voices": 1}
GOOD = """```abc
X:1
T:Good jig
M:6/8
L:1/8
Q:3/8=80
K:Dm
|:D2E FED|A2d cAF|G2E F2D|E3 A,3|
D2E FED|A2d cAF|GFE ^CDE|D3 D3:|
```"""
WRONG = """```abc
X:1
M:4/4
L:1/8
Q:1/4=90
K:G
|:G2B d2B c2|A F2A G2B d|f2a g4 G2|B d2B c2A F|A2B2 G2A2|F2G6:|
```"""
HEADER_HACK = ("```abc\nX:1\nM:6/8\nL:1/8\nQ:3/8=80\nK:Dm\n"
               + "^C^D^F ^G^A^c|^d^f^g ^a^c'^d'|" * 4 + "\n```")
DRONE = "```abc\nX:1\nM:6/8\nL:1/8\nQ:3/8=80\nK:Dm\n" + "D6|" * 8 + "\n```"


def test_following_the_prompt_beats_everything_else():
    good, wrong, hack, drone = (score_abc(t, C) for t in (GOOD, WRONG, HEADER_HACK, DRONE))
    assert good.reward > 0.95
    assert good.adherence["score"] == 1.0
    assert wrong.reward < 0.3 and wrong.musicality["score"] > 0.7      # fluent, ignores the prompt
    assert hack.adherence["key"] == 0.0                                 # K:Dm over notes not in Dm
    assert drone.musicality["gate"] == 0.0
    assert good.reward > max(hack.reward, drone.reward) + 0.4


def test_header_claims_need_musical_support():
    # right header, an interior bar of the wrong length (first and last bars may be partial)
    bad_bars = GOOD.replace("|A2d cAF|G2E", "|A2d cAFG|G2E")
    assert score_abc(bad_bars, C).adherence["meter"] < 1.0
    # right notes, wrong key named in the header
    assert score_abc(GOOD.replace("K:Dm", "K:F"), C).adherence["key"] == 0.0


def test_tempo_and_bar_tolerance():
    assert score_abc(GOOD.replace("Q:3/8=80", "Q:3/8=82"), C).adherence["tempo"] == 1.0
    assert score_abc(GOOD.replace("Q:3/8=80", "Q:3/8=100"), C).adherence["tempo"] == 0.0
    assert score_abc(GOOD.replace("Q:3/8=80\n", ""), C).adherence["tempo"] == 0.0
    assert score_abc(GOOD, dict(C, bars=9)).adherence["bars"] == 1.0      # one bar of slack
    assert score_abc(GOOD, dict(C, bars=16)).adherence["bars"] < 0.2


def test_format_and_invalid():
    assert score_abc("I cannot write music.", C).status == "format"
    assert score_abc("```abc\nX:1\nK:Dm\nDE\n```", C).status == "invalid"


def test_every_generated_task_is_solvable():
    tasks = make_music_tasks(120, seed=7)
    assert len({t["constraints"]["key"] for t in tasks}) > 8
    for i, t in enumerate(tasks):
        s = score_abc(f"```abc\n{reference_piece(t['constraints'], i)}\n```", t["constraints"])
        assert s.adherence["score"] == 1.0, (t["constraints"], s.adherence)
        assert s.reward > 0.5


def test_task_prompts_state_the_constraints():
    t = make_music_tasks(1, seed=3)[0]
    c = t["constraints"]
    assert parse_key(c["key"]).name in t["prompt"] and c["meter"] in t["prompt"]
    assert f"{c['bars']} bars" in t["prompt"]
    assert make_music_tasks(5, seed=3) == make_music_tasks(5, seed=3)


def test_batch_scorer_and_trl_adapter():
    scorer = MusicReward()
    rewards = scorer.rewards([GOOD, "nope", "la " * 600 + GOOD], [{"constraints": C}] * 3)
    assert rewards[0] > 0.95 and rewards[1] == 0.0 and rewards[2] == 0.0   # last: degenerate repetition
    import json

    fn = trl_music_reward(scorer)
    out = fn(prompts=["p"], completions=[[{"role": "assistant", "content": GOOD}]], constraints=[json.dumps(C)])
    assert out[0] > 0.95


def test_midi_output():
    p = parse_abc(extract_abc(reference_piece({"key": "G", "meter": "3/4", "tempo": 90, "bars": 4,
                                               "voices": 2})))
    data = to_midi(p)
    assert data[:4] == b"MThd" and data.count(b"MTrk") == 3        # tempo track + 2 voices
    assert int.from_bytes(data[10:12], "big") == 3
