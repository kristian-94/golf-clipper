"""Tests for the gap-based hole/shot correlator (manual-scorecard path).

Run:
    /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/

The gap correlator is used when the user enters their scorecard after the
round, so `score.created_at` clusters and can't be trusted. We assign holes
greedily by walking the scorecard's owner stroke counts and eating clips in
chronological order.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

# Make the project root importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from correlate import (  # noqa: E402
    apply_to_sidecars,
    correlate_by_gaps,
    players_through,
)


def _scores(strokes_per_hole: list[int], pars: list[int] | None = None) -> dict:
    """Build a minimal SmartCaddy-shaped scorecard for testing.

    `strokes_per_hole[i]` is what the (only) player scored on hole i+1.
    `pars` defaults to par 4 for every hole.
    """
    pars = pars or [4] * len(strokes_per_hole)
    return {
        "round_id": "test",
        "course": "Test GC",
        "fetched_at": "2026-05-03T00:00:00+00:00",
        "holes": [{"number": i + 1, "par": pars[i]} for i in range(len(strokes_per_hole))],
        "players": [
            {
                "session_id": "owner",
                "name": "Kristian",
                "is_owner": True,
                "scores": [
                    {
                        "hole": i + 1,
                        "strokes": s,
                        "putts": 2,
                        # Deliberately bogus timestamps — gap-based ignores them.
                        "created_at": "2026-05-03T20:00:00+00:00",
                    }
                    for i, s in enumerate(strokes_per_hole)
                ],
            },
        ],
    }


def _sidecars(timestamps: list[str]) -> dict[str, dict]:
    """Build sidecars with monotonic stems and the given recorded_at timestamps."""
    return {
        f"clip_{i:02d}": {"raw": f"clip_{i:02d}.MOV", "recorded_at": ts}
        for i, ts in enumerate(timestamps)
    }


def test_exact_match_one_clip_per_stroke():
    """When clip count equals total strokes, every shot gets the right shot_index."""
    scores = _scores([4, 3])  # 4 clips on hole 1, 3 clips on hole 2
    sidecars = _sidecars([
        "2026-05-03T10:00:00+00:00",
        "2026-05-03T10:01:00+00:00",
        "2026-05-03T10:02:00+00:00",
        "2026-05-03T10:03:00+00:00",
        "2026-05-03T10:10:00+00:00",
        "2026-05-03T10:11:00+00:00",
        "2026-05-03T10:12:00+00:00",
    ])
    result = correlate_by_gaps(scores, sidecars)

    # First 4 clips → hole 1, shots 1..4 of 4
    for i in range(4):
        r = result[f"clip_{i:02d}"]
        assert r["hole"] == 1, f"clip_{i:02d} should be hole 1"
        assert r["shot_index"] == i + 1
        assert r["shot_total"] == 4
        assert r["par"] == 4
        assert "correlate_warning" in r

    # Next 3 clips → hole 2, shots 1..3 of 3
    for i, j in enumerate(range(4, 7)):
        r = result[f"clip_{j:02d}"]
        assert r["hole"] == 2
        assert r["shot_index"] == i + 1
        assert r["shot_total"] == 3


def test_chronological_not_filename():
    """Clips are walked in recorded_at order, not stem order."""
    scores = _scores([2])
    # clip_00 has a LATER timestamp than clip_01 — the chronological ordering
    # should put clip_01 as shot 1 and clip_00 as shot 2.
    sidecars = {
        "clip_00": {"raw": "a.MOV", "recorded_at": "2026-05-03T10:05:00+00:00"},
        "clip_01": {"raw": "b.MOV", "recorded_at": "2026-05-03T10:00:00+00:00"},
    }
    result = correlate_by_gaps(scores, sidecars)
    assert result["clip_01"]["shot_index"] == 1
    assert result["clip_00"]["shot_index"] == 2


def test_fewer_clips_than_strokes():
    """If we only recorded 2 of 4 shots on hole 1, those still go to hole 1
    with shot_total reflecting the scorecard truth, and we move on to hole 2."""
    scores = _scores([4, 3])
    sidecars = _sidecars([
        "2026-05-03T10:00:00+00:00",
        "2026-05-03T10:01:00+00:00",
        # Missed shots 3+4 on hole 1; first hole-2 clip is next.
        "2026-05-03T10:10:00+00:00",
        "2026-05-03T10:11:00+00:00",
    ])
    result = correlate_by_gaps(scores, sidecars)
    # All 4 clips end up on hole 1 (greedy walk). This is the expected
    # behaviour — the gap correlator has no way to know shots were skipped,
    # which is exactly why the review UI exists. shot_total stays at 4
    # (the scorecard truth) so the user sees "shot 1/4" etc.
    for stem in ("clip_00", "clip_01", "clip_02", "clip_03"):
        assert result[stem]["hole"] == 1
        assert result[stem]["shot_total"] == 4
    assert result["clip_00"]["shot_index"] == 1
    assert result["clip_03"]["shot_index"] == 4
    # Every assignment must carry the warning so the UI flags it.
    for stem in ("clip_00", "clip_01", "clip_02", "clip_03"):
        assert result[stem]["correlate_warning"]


def test_more_clips_than_strokes_overflow():
    """Excess clips past the last hole spill into that hole with overflow shot indices."""
    scores = _scores([2, 2])  # only 4 strokes total
    sidecars = _sidecars([
        "2026-05-03T10:00:00+00:00",
        "2026-05-03T10:01:00+00:00",
        "2026-05-03T10:10:00+00:00",
        "2026-05-03T10:11:00+00:00",
        "2026-05-03T10:20:00+00:00",  # extra
        "2026-05-03T10:21:00+00:00",  # extra
    ])
    result = correlate_by_gaps(scores, sidecars)
    # First 4 placed normally
    assert result["clip_00"]["hole"] == 1 and result["clip_00"]["shot_index"] == 1
    assert result["clip_03"]["hole"] == 2 and result["clip_03"]["shot_index"] == 2
    # Overflow lands on the last hole (2) with shot_index past shot_total.
    assert result["clip_04"]["hole"] == 2
    assert result["clip_04"]["shot_index"] == 3
    assert "overflow" in result["clip_04"]["correlate_warning"]


def test_hole_with_zero_strokes_skipped():
    """A hole the owner didn't play (strokes None) is skipped — the next hole
    gets the next clips. This matters for a 9-hole subset of an 18-hole card."""
    scores = _scores([4, 0, 3])
    # Manually clobber hole 2's strokes to None to mimic an unplayed hole.
    scores["players"][0]["scores"][1]["strokes"] = None
    sidecars = _sidecars([
        "2026-05-03T10:00:00+00:00",
        "2026-05-03T10:01:00+00:00",
        "2026-05-03T10:02:00+00:00",
        "2026-05-03T10:03:00+00:00",
        "2026-05-03T11:00:00+00:00",
        "2026-05-03T11:01:00+00:00",
        "2026-05-03T11:02:00+00:00",
    ])
    result = correlate_by_gaps(scores, sidecars)
    # Hole 2 entirely skipped; hole 3 picks up after hole 1's 4.
    assert result["clip_03"]["hole"] == 1
    assert result["clip_04"]["hole"] == 3
    assert result["clip_06"]["hole"] == 3


def test_clip_without_recorded_at_flagged():
    """A sidecar with no recorded_at can't be ordered — flag it as an error."""
    scores = _scores([2])
    sidecars = {
        "clip_00": {"raw": "a.MOV", "recorded_at": "2026-05-03T10:00:00+00:00"},
        "clip_01": {"raw": "b.MOV"},  # no recorded_at
    }
    result = correlate_by_gaps(scores, sidecars)
    # The dated clip gets shot 1.
    assert result["clip_00"]["hole"] == 1
    assert result["clip_00"]["shot_index"] == 1
    # The undated clip is flagged. It would have been assigned to shot 2 by
    # position, but the function refuses it and surfaces the error so the
    # user can fix the sidecar (extract date from EXIF/mtime).
    assert result["clip_01"]["hole"] is None
    assert result["clip_01"]["correlate_error"]


def test_players_through_running_totals_unchanged():
    """`players_through` is shared between both correlators and must report
    cumulative strokes/par BEFORE the assigned hole (so the overlay shows
    the leaderboard as it was when the player teed off)."""
    scores = _scores([4, 5, 3], pars=[4, 4, 3])
    rows = players_through(scores, hole=3)
    assert len(rows) == 1
    assert rows[0]["total"] == 9       # holes 1+2 = 4+5
    assert rows[0]["par_through"] == 8  # par 4+4


def test_apply_to_sidecars_persists_mode_and_preserves_manual_edits(tmp_path):
    """End-to-end: gaps mode writes assignment_mode to scores.json and a
    second pass leaves user-modified sidecars alone."""
    round_dir = tmp_path / "test-round"
    (round_dir / "meta").mkdir(parents=True)
    scores = _scores([2, 2])
    (round_dir / "scores.json").write_text(json.dumps(scores))
    timestamps = [
        "2026-05-03T10:00:00+00:00",
        "2026-05-03T10:01:00+00:00",
        "2026-05-03T10:10:00+00:00",
        "2026-05-03T10:11:00+00:00",
    ]
    sidecars = _sidecars(timestamps)
    for stem, data in sidecars.items():
        (round_dir / "meta" / f"{stem}.json").write_text(json.dumps(data))

    summary = apply_to_sidecars(round_dir, fetch_map=False, mode="gaps")
    assert summary["correlated"] == 4

    # Mode persisted into scores.json
    persisted = json.loads((round_dir / "scores.json").read_text())
    assert persisted["assignment_mode"] == "gaps"

    # Sidecars now have hole/shot assignments
    sc = json.loads((round_dir / "meta" / "clip_00.json").read_text())
    assert sc["hole"] == 1 and sc["shot_index"] == 1

    # User manually moves clip_03 from hole 2 → hole 1, shot 3 of 3
    sc3 = json.loads((round_dir / "meta" / "clip_03.json").read_text())
    sc3["hole"] = 1
    sc3["shot_index"] = 3
    sc3["shot_total"] = 3
    sc3.pop("correlate_warning", None)
    (round_dir / "meta" / "clip_03.json").write_text(json.dumps(sc3))

    # Re-run apply_to_sidecars (e.g. server restart). The manual edit must survive.
    apply_to_sidecars(round_dir, fetch_map=False)  # mode auto-reads from scores.json
    sc3_after = json.loads((round_dir / "meta" / "clip_03.json").read_text())
    assert sc3_after["hole"] == 1, "manual hole edit was clobbered"
    assert sc3_after["shot_index"] == 3, "manual shot_index was clobbered"
    # `players` was refreshed from the current scorecard — confirm it's consistent
    # with hole 1 (i.e. running totals BEFORE hole 1 = 0).
    assert sc3_after["players"][0]["total"] == 0


def test_played_through_drops_player_from_leaderboard():
    """A player with `played_through=9` is excluded from any hole > 9."""
    scores = _scores([4, 3, 5])  # 3 holes, par 4 each
    scores["players"].append({
        "session_id": "guest",
        "name": "Kabir",
        "is_owner": False,
        "played_through": 1,  # left after hole 1
        "scores": [
            {"hole": 1, "strokes": 5, "putts": 2, "created_at": "x"},
            {"hole": 2, "strokes": 4, "putts": 2, "created_at": "x"},
            {"hole": 3, "strokes": 6, "putts": 2, "created_at": "x"},
        ],
    })
    # Hole 1 leaderboard (the "before hole 1" snapshot): both players, both at 0.
    rows = players_through(scores, hole=1)
    names = [r["name"] for r in rows]
    assert "Kabir" in names

    # Hole 2 leaderboard: Kabir's `played_through` (1) is not < 2 — wait, 1 < 2,
    # so he IS dropped. The convention is "played_through < hole means gone."
    rows = players_through(scores, hole=2)
    names = [r["name"] for r in rows]
    assert "Kabir" not in names, "Kabir should be dropped for hole 2 onwards"
    assert "Kristian" in names

    # Hole 3 leaderboard: still no Kabir.
    rows = players_through(scores, hole=3)
    assert "Kabir" not in [r["name"] for r in rows]


def test_played_through_none_means_all_holes():
    """No `played_through` (or null) keeps a player on every leaderboard."""
    scores = _scores([4, 3])
    scores["players"].append({
        "session_id": "guest",
        "name": "Kabir",
        "is_owner": False,
        # No played_through set
        "scores": [
            {"hole": 1, "strokes": 5, "putts": 2, "created_at": "x"},
            {"hole": 2, "strokes": 4, "putts": 2, "created_at": "x"},
        ],
    })
    for h in (1, 2):
        rows = players_through(scores, hole=h)
        assert "Kabir" in [r["name"] for r in rows]


def test_played_through_at_boundary_keeps_player():
    """`played_through=9` keeps the player on hole 9's overlay (which shows
    leaderboard *before* hole 9), drops them on hole 10."""
    scores = _scores([4] * 10)  # 10 holes
    scores["players"].append({
        "session_id": "guest",
        "name": "Kabir",
        "is_owner": False,
        "played_through": 9,
        "scores": [
            {"hole": h, "strokes": 4, "putts": 2, "created_at": "x"}
            for h in range(1, 11)
        ],
    })
    # Hole 9 — the leaderboard shows scores going INTO hole 9, when Kabir was
    # still around and about to play. Keep him.
    rows = players_through(scores, hole=9)
    assert "Kabir" in [r["name"] for r in rows]
    # Hole 10 — he's gone.
    rows = players_through(scores, hole=10)
    assert "Kabir" not in [r["name"] for r in rows]


def test_apply_to_sidecars_default_mode_is_timestamps(tmp_path):
    """When no mode is set anywhere, fall back to the timestamp-based correlator."""
    round_dir = tmp_path / "test-round"
    (round_dir / "meta").mkdir(parents=True)
    # SmartCaddy-style: per-shot timestamps that actually mean something.
    scores = deepcopy(_scores([2, 2]))
    scores["players"][0]["scores"][0]["created_at"] = "2026-05-03T10:05:00+00:00"
    scores["players"][0]["scores"][1]["created_at"] = "2026-05-03T10:15:00+00:00"
    (round_dir / "scores.json").write_text(json.dumps(scores))
    sidecars = _sidecars([
        "2026-05-03T10:01:00+00:00",  # before hole 1 boundary → hole 1
        "2026-05-03T10:04:00+00:00",  # before hole 1 boundary → hole 1
        "2026-05-03T10:10:00+00:00",  # before hole 2 boundary → hole 2
        "2026-05-03T10:14:00+00:00",  # before hole 2 boundary → hole 2
    ])
    for stem, data in sidecars.items():
        (round_dir / "meta" / f"{stem}.json").write_text(json.dumps(data))

    apply_to_sidecars(round_dir, fetch_map=False)
    persisted = json.loads((round_dir / "scores.json").read_text())
    assert persisted.get("assignment_mode", "timestamps") == "timestamps"
    sc0 = json.loads((round_dir / "meta" / "clip_00.json").read_text())
    sc2 = json.loads((round_dir / "meta" / "clip_02.json").read_text())
    assert sc0["hole"] == 1
    assert sc2["hole"] == 2
