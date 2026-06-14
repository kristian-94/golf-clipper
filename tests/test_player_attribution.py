"""Tests for phone-of-origin player attribution and per-player shot tallies.

Run:
    /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/

In Kristian's playgroup each golfer films the *other*, so a single hole holds
clips of more than one player. The shot badge (SHOT n/total) must therefore be
numbered per subject player — counting a player's own clips against that
player's own stroke count — not by assuming every clip on a hole is the
owner's. `device_subjects` (model -> the player that phone films) is what
supplies each clip's subject, derived from the recording device.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the project root importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from correlate import correlate, tag_players_from_devices  # noqa: E402

UTC = timezone.utc


def _iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def _mixed_scores() -> dict:
    """Two players. On hole 1 Kristian took 2 strokes, Simon took 3. Both
    holes' scores go in early enough that every test clip lands on hole 1."""
    base = datetime(2026, 6, 13, 3, 40, tzinfo=UTC)

    def sc(hole: int, i: int, strokes: int) -> dict:
        return {"hole": hole, "strokes": strokes, "putts": 2,
                "created_at": _iso(base + timedelta(minutes=14 * i))}

    return {
        "round_id": "test",
        "course": "Bexley GC",
        "holes": [{"number": 1, "par": 4}, {"number": 2, "par": 4}],
        "players": [
            {"name": "Kristian", "is_owner": True,
             "scores": [sc(1, 0, 2), sc(2, 1, 4)]},
            {"name": "Simon", "is_owner": False,
             "scores": [sc(1, 0, 3), sc(2, 1, 5)]},
        ],
    }


def _clip(minute: int, player: str | None = None) -> dict:
    """A sidecar recorded before hole 1's boundary (03:40), so it maps to hole 1."""
    rec = datetime(2026, 6, 13, 3, 30, tzinfo=UTC) + timedelta(minutes=minute)
    data = {"raw": f"c{minute}.MOV", "recorded_at": _iso(rec)}
    if player is not None:
        data["player"] = player
    return data


def test_mixed_camera_hole_numbers_each_player_separately():
    """The core regression. Five clips on hole 1: two of Kristian (2 strokes),
    three of Simon (3 strokes). Each player's clips must be numbered 1..n
    against *their own* stroke count. The old owner-only tally compared the
    owner's 2 strokes to all 5 clips, matched nothing, and emitted no shot
    badges at all — so this would have failed before per-player grouping."""
    scores = _mixed_scores()
    sidecars = {
        "k1": _clip(1, "Kristian"),
        "k2": _clip(2, "Kristian"),
        "s1": _clip(3, "Simon"),
        "s2": _clip(4, "Simon"),
        "s3": _clip(5, "Simon"),
    }
    out = correlate(scores, sidecars)

    assert all(out[s]["hole"] == 1 for s in sidecars), "all five clips are hole 1"

    # Kristian: 2 strokes -> his two clips numbered 1/2, 2/2 by recording time.
    assert (out["k1"]["shot_index"], out["k1"]["shot_total"]) == (1, 2)
    assert (out["k2"]["shot_index"], out["k2"]["shot_total"]) == (2, 2)

    # Simon: 3 strokes -> his three clips numbered 1/3, 2/3, 3/3.
    assert (out["s1"]["shot_index"], out["s1"]["shot_total"]) == (1, 3)
    assert (out["s2"]["shot_index"], out["s2"]["shot_total"]) == (2, 3)
    assert (out["s3"]["shot_index"], out["s3"]["shot_total"]) == (3, 3)


def test_untagged_clips_default_to_owner():
    """Clips with no `player` are the owner's. Two untagged clips on hole 1
    match Kristian's 2 strokes and get numbered against him."""
    scores = _mixed_scores()
    sidecars = {"a": _clip(1), "b": _clip(2)}  # no player field
    out = correlate(scores, sidecars)
    assert (out["a"]["shot_index"], out["a"]["shot_total"]) == (1, 2)
    assert (out["b"]["shot_index"], out["b"]["shot_total"]) == (2, 2)


def test_count_mismatch_leaves_shots_unnumbered():
    """The guard still holds per player: if a player's clip count doesn't equal
    their stroke count (e.g. a missed or extra clip), don't guess shot numbers
    for that player — a wrong badge is worse than none."""
    scores = _mixed_scores()  # Simon has 3 strokes on hole 1
    sidecars = {"s1": _clip(3, "Simon"), "s2": _clip(4, "Simon")}  # only 2 clips
    out = correlate(scores, sidecars)
    assert "shot_index" not in out["s1"]
    assert "shot_total" not in out["s1"]


def _roster() -> dict:
    return {"players": [{"name": "Kristian", "is_owner": True},
                        {"name": "Simon", "is_owner": False}],
            "device_subjects": {"iPhone 17 Pro Max": "Kristian",
                                "iPhone 14 Pro Max": "Simon"}}


def test_tag_from_devices_sets_subject_by_model():
    """Each phone tags its clips with the player it films, stamping the source
    as 'device' so a later manual edit can be told apart."""
    scores = _roster()
    sidecars = {
        "a": {"device_model": "iPhone 17 Pro Max"},  # dad's phone -> Kristian
        "b": {"device_model": "iPhone 14 Pro Max"},  # Kristian's phone -> Simon
    }
    n = tag_players_from_devices(scores, sidecars)
    assert n == 2
    assert sidecars["a"]["player"] == "Kristian"
    assert sidecars["a"]["player_source"] == "device"
    assert sidecars["b"]["player"] == "Simon"


def test_tag_from_devices_does_not_clobber_existing_player():
    """A clip already carrying a `player` (a manual /assign override, or a
    prior tag) is left untouched, so re-correlating never undoes the user."""
    scores = _roster()
    sidecars = {"c": {"device_model": "iPhone 14 Pro Max", "player": "Kristian"}}
    n = tag_players_from_devices(scores, sidecars)
    assert n == 0
    assert sidecars["c"]["player"] == "Kristian"


def test_tag_from_devices_ignores_unknown_model_and_player():
    """Unmapped cameras and mappings to a non-roster name are no-ops — never
    invent a `player` we can't back with a real scorecard entry."""
    scores = {"players": [{"name": "Kristian", "is_owner": True}],
              "device_subjects": {"iPhone 14 Pro Max": "Ghost"}}  # Ghost not on roster
    sidecars = {
        "x": {"device_model": "iPhone 14 Pro Max"},  # maps to invalid player
        "y": {"device_model": "Pixel 8"},            # model not in map
        "z": {"device_model": None},                 # no model at all
    }
    n = tag_players_from_devices(scores, sidecars)
    assert n == 0
    assert all("player" not in sidecars[s] for s in sidecars)


def test_tag_from_devices_noop_without_map():
    """No device_subjects configured -> nothing to do (single-phone rounds)."""
    sidecars = {"a": {"device_model": "iPhone 14 Pro Max"}}
    assert tag_players_from_devices({"players": []}, sidecars) == 0
    assert "player" not in sidecars["a"]
