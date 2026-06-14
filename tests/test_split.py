"""Tests for the split-clip flow (multiple shots from one raw video).

Run:
    /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/

The split feature lets one raw video produce multiple trims — handy when
a single clip captures several putts on the same green for both players.
Sidecar stems become independent of raw filenames: the original keeps the
bare stem (`IMG_9641`), sub-clips suffix with letters (`_b`, `_c`, ...).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import _next_split_stem  # noqa: E402


def test_next_split_stem_first_split_is_b():
    """The original clip keeps the raw stem, so the first sub-clip is `_b`."""
    assert _next_split_stem("IMG_9641", {"IMG_9641"}) == "IMG_9641_b"


def test_next_split_stem_skips_used_letters():
    """Subsequent splits walk the alphabet, jumping over taken letters."""
    used = {"IMG_9641", "IMG_9641_b", "IMG_9641_c"}
    assert _next_split_stem("IMG_9641", used) == "IMG_9641_d"


def test_next_split_stem_returns_none_when_exhausted():
    """All 25 letters taken (b–z) → caller should surface a 409."""
    used = {"IMG_9641"} | {f"IMG_9641_{ch}" for ch in "bcdefghijklmnopqrstuvwxyz"}
    assert _next_split_stem("IMG_9641", used) is None


def test_next_split_stem_only_collides_on_exact_match():
    """A different raw's sub-clips don't reserve letters from this raw's pool."""
    used = {"IMG_9641", "IMG_9999_b", "OTHER_x"}
    assert _next_split_stem("IMG_9641", used) == "IMG_9641_b"


def _setup_state(tmp_path):
    """Wire server.STATE to a sandboxed round folder. Returns (round_dir, paths)."""
    import server
    round_dir = tmp_path / "round"
    raw_dir = round_dir / "raw"
    trims_dir = round_dir / "trims"
    meta_dir = round_dir / "meta"
    for d in (raw_dir, trims_dir, meta_dir):
        d.mkdir(parents=True)
    (round_dir / "scores.json").write_text(json.dumps({
        "round_id": "x", "course": "Test", "fetched_at": "x",
        "holes": [{"number": 1, "par": 4}],
        "players": [{"session_id": "p1", "name": "K", "is_owner": True, "scores": []}],
    }))
    server.STATE.clear()
    server.STATE.update({
        "round": round_dir,
        "raw": raw_dir,
        "trims": trims_dir,
        "meta": meta_dir,
        "scores": round_dir / "scores.json",
    })
    server.RENDER_STATUS.clear()
    return round_dir


def _stub_render_executor():
    """Replace asyncio.get_event_loop().run_in_executor with a no-op that
    just records which stems would have been rendered."""
    import asyncio as _asyncio
    queued: list[str] = []

    class _Loop:
        def run_in_executor(self, executor, fn, *args):
            queued.append(args[0])

    orig = _asyncio.get_event_loop
    _asyncio.get_event_loop = lambda: _Loop()  # type: ignore
    return queued, lambda: setattr(_asyncio, "get_event_loop", orig)


def test_split_endpoint_clones_and_resets(tmp_path):
    """End-to-end: /api/clips/{stem}/split clones the sidecar, sets the new
    impact_s, and resets per-trim state (review/has_overlay/edited)."""
    import asyncio
    import server
    from server import api_split_clip, SplitRequest

    round_dir = _setup_state(tmp_path)
    queued, restore = _stub_render_executor()
    try:
        parent = {
            "raw": "IMG_9641.MOV",
            "recorded_at": "2026-05-10T08:00:00Z",
            "impact_s": 8.5,
            "pre": 1.5, "post": 4.0,
            "confidence": "strong", "flagged": False, "edited": False,
            "review": "approved", "has_overlay": True,
            "trimmed_at": "2026-05-10T22:00:00",
            "hole": 7, "par": 4,
            "players": [{"name": "K", "is_owner": True, "total": 0, "par_through": 0}],
            "shot_index": 1, "shot_total": 4,
        }
        (round_dir / "meta" / "IMG_9641.json").write_text(json.dumps(parent))

        new = asyncio.run(api_split_clip("IMG_9641", SplitRequest(impact_s=25.3)))
        assert new["stem"] == "IMG_9641_b"
        assert new["impact_s"] == 25.3
        assert new["raw"] == "IMG_9641.MOV"
        # Per-trim state reset for the new sub-clip
        assert new["review"] == "unreviewed"
        assert new["has_overlay"] is False
        assert new["confidence"] == "manual"
        assert new["edited"] is True
        # Hole/shot/players inherited — same hole is the common case
        assert new["hole"] == 7
        assert new["shot_index"] == 1
        assert (round_dir / "meta" / "IMG_9641_b.json").exists()
        assert "IMG_9641_b" in queued

        # Original sidecar untouched
        orig = json.loads((round_dir / "meta" / "IMG_9641.json").read_text())
        assert orig["impact_s"] == 8.5
        assert orig["review"] == "approved"

        # Second split picks `_c`.
        new2 = asyncio.run(api_split_clip("IMG_9641", SplitRequest(impact_s=41.0)))
        assert new2["stem"] == "IMG_9641_c"
    finally:
        restore()


def test_discard_keeps_raw_when_siblings_exist(tmp_path):
    """Discarding one of several sub-clips moves the sidecar+trim to trash
    but LEAVES the raw in place. Discarding the LAST sidecar sweeps the raw."""
    import asyncio
    from server import api_discard_clip

    round_dir = _setup_state(tmp_path)
    raw_file = round_dir / "raw" / "IMG_9641.MOV"
    raw_file.write_bytes(b"fake raw bytes")

    parent = {"raw": "IMG_9641.MOV", "impact_s": 8.5, "pre": 1.5, "post": 4.0}
    sub = {"raw": "IMG_9641.MOV", "impact_s": 25.3, "pre": 1.5, "post": 4.0}
    (round_dir / "meta" / "IMG_9641.json").write_text(json.dumps(parent))
    (round_dir / "meta" / "IMG_9641_b.json").write_text(json.dumps(sub))
    (round_dir / "trims" / "IMG_9641.mp4").write_bytes(b"trim bytes")
    (round_dir / "trims" / "IMG_9641_b.mp4").write_bytes(b"sub trim bytes")

    # Discard the sub-clip — raw must stay because parent still references it.
    res = asyncio.run(api_discard_clip("IMG_9641_b"))
    assert res["raw_moved"] is False
    assert raw_file.exists()
    assert not (round_dir / "meta" / "IMG_9641_b.json").exists()
    assert not (round_dir / "trims" / "IMG_9641_b.mp4").exists()
    assert (round_dir / "trash" / "meta" / "IMG_9641_b.json").exists()

    # Now discard the parent — last reference → raw goes to trash.
    res2 = asyncio.run(api_discard_clip("IMG_9641"))
    assert res2["raw_moved"] is True
    assert not raw_file.exists()
    assert (round_dir / "trash" / "raw" / "IMG_9641.MOV").exists()
