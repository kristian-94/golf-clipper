"""Tests for the timestamp-based hole correlator (live-logging path).

Run:
    /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/

When the scorecard is logged live, each hole's `created_at` marks when that
hole was completed, so we map a clip to the hole that was being played when it
was filmed. The tricky case is a shotgun / back-nine-first round: holes are NOT
played in numeric order, so the correlator must walk holes in *play order*
(score-entry time), not by hole number. These tests pin that behaviour — a
regression here silently collapses every early clip onto hole 1.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the project root importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from correlate import (  # noqa: E402
    assign_hole,
    correlate,
    hole_boundaries,
    leaderboard_order,
    players_through,
    _parse,
)


def _iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def _scores(play_order: list[int]) -> dict:
    """Build a single-player scorecard where holes are completed in `play_order`,
    each 14 minutes apart. `created_at` therefore encodes play order, which may
    differ from hole number (back-nine-first, shotgun, etc.)."""
    base = datetime(2026, 5, 24, 3, 51, tzinfo=timezone.utc)
    scores = [
        {
            "hole": h,
            "strokes": 4,
            "putts": 2,
            "created_at": _iso(base + timedelta(minutes=14 * i)),
        }
        for i, h in enumerate(play_order)
    ]
    return {
        "round_id": "test",
        "course": "Test GC",
        "holes": [{"number": n, "par": 4} for n in range(1, 19)],
        "players": [
            {"session_id": "o", "name": "Kristian", "is_owner": True, "scores": scores}
        ],
    }


# Played holes 10..18 then 1..7 — the real Georges River round shape.
BACK_NINE_FIRST = [10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7]


def test_back_nine_first_does_not_collapse_early_clips_onto_hole_1():
    """The regression: a clip filmed at the very start of play (during hole 10)
    must map to hole 10. Hole 1 was played 10th, so its score-entry time is
    *late*; walking holes by number would grab hole 1 for every early clip."""
    boundaries = hole_boundaries(_scores(BACK_NINE_FIRST))
    early = _parse("2026-05-24T03:45:00Z")  # before any score was entered
    assert assign_hole(early, boundaries) == 10


def test_back_nine_first_front_nine_clip_maps_to_front_nine():
    """A clip filmed once they reached the front nine (played 10th) maps to
    hole 1, not back to a numerically-earlier-but-already-finished hole."""
    boundaries = hole_boundaries(_scores(BACK_NINE_FIRST))
    # Hole 1 was played 10th: its score went in 9*14 min after the 03:51 start
    # = 05:57. A clip in (hole 18's 05:43, hole 1's 05:57] is during hole 1.
    during_hole_1 = _parse("2026-05-24T05:50:00Z")
    assert assign_hole(during_hole_1, boundaries) == 1


def test_clip_after_last_score_falls_back_to_last_hole_played():
    """Trailing clips (after the final score entry) fall back to the last hole
    *played* (7), not the highest hole *number* (18)."""
    boundaries = hole_boundaries(_scores(BACK_NINE_FIRST))
    after = _parse("2026-05-24T23:00:00Z")
    assert assign_hole(after, boundaries) == 7


def test_numeric_order_round_is_unaffected():
    """Sanity: a normal front-to-back round still assigns by the same time
    windows — the fix is a no-op when play order matches hole number."""
    boundaries = hole_boundaries(_scores(list(range(1, 19))))
    # Hole 5 went in at 03:51 + 4*14 = 04:47; a clip just before is hole 5.
    during_hole_5 = _parse("2026-05-24T04:45:00Z")
    assert assign_hole(during_hole_5, boundaries) == 5


def test_correlate_end_to_end_back_nine_first():
    """Through the full correlator: early clip -> hole 10, later clip -> hole 1."""
    scores = _scores(BACK_NINE_FIRST)
    sidecars = {
        "clip_a": {"raw": "a.MOV", "recorded_at": "2026-05-24T03:45:00Z"},
        "clip_b": {"raw": "b.MOV", "recorded_at": "2026-05-24T05:50:00Z"},
    }
    out = correlate(scores, sidecars)
    assert out["clip_a"]["hole"] == 10
    assert out["clip_b"]["hole"] == 1
    assert out["clip_a"]["par"] == 4
    # The hole-10 clip is the first hole played: leaderboard shows 0 strokes.
    assert out["clip_a"]["players"][0]["total"] == 0
    # The hole-1 clip is the 10th hole played: 9 back-nine holes already done.
    assert out["clip_b"]["players"][0]["total"] == 9 * 4


def test_leaderboard_totals_follow_play_order_not_hole_number():
    """In time order the running total going into a hole counts the holes the
    player actually finished first — NOT every numerically-lower hole. This is
    the leaderboard half of the back-nine-first fix."""
    scores = _scores(BACK_NINE_FIRST)
    # Hole 10 was teed off first → nothing played yet.
    into_10 = players_through(scores, 10, order="time")[0]
    assert into_10["total"] == 0
    assert into_10["par_through"] == 0
    # Hole 1 was the 10th hole → the back nine (10..18, nine holes) is done, and
    # crucially hole 1 itself is NOT counted.
    into_1 = players_through(scores, 1, order="time")[0]
    assert into_1["total"] == 9 * 4
    assert into_1["par_through"] == 9 * 4


def test_leaderboard_time_order_handles_partner_who_left_early():
    """A player with no score for the target hole falls back to the field's
    score-entry time for it, so they still show their played holes (rather than
    vanishing or double-counting)."""
    base = datetime(2026, 5, 24, 3, 51, tzinfo=timezone.utc)

    def sc(h, i):
        return {"hole": h, "strokes": 4, "putts": 2,
                "created_at": _iso(base + timedelta(minutes=14 * i))}

    scores = {
        "holes": [{"number": n, "par": 4} for n in range(1, 19)],
        "players": [
            {"name": "Kristian", "is_owner": True, "scores": [sc(10, 0), sc(1, 2)]},
            {"name": "Guest", "is_owner": False, "scores": [sc(10, 1)]},  # left after 10
        ],
    }
    rows = {r["name"]: r for r in players_through(scores, 1, order="time")}
    assert rows["Kristian"]["total"] == 4   # played hole 10 before hole 1
    assert rows["Guest"]["total"] == 4      # hole 10 counted via field-time fallback


def test_leaderboard_order_selected_by_assignment_mode():
    """Gaps mode (clustered timestamps) keeps number ordering; everything else
    uses play-order time."""
    assert leaderboard_order({"assignment_mode": "gaps"}) == "number"
    assert leaderboard_order({"assignment_mode": "timestamps"}) == "time"
    assert leaderboard_order({}) == "time"
