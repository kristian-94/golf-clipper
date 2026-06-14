"""Tests for the per-clip rotation feature.

We don't shell out to ffmpeg — too slow and flaky in CI. Instead we patch
`subprocess.Popen` in overlay.render_video to capture the constructed
command and assert the right `transpose` filter is in the chain.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import overlay  # noqa: E402


def _canvas() -> dict:
    """Minimal canvas dict with the keys render_video reads."""
    return {
        "width": 1920, "height": 1080,
        "fps": 30, "fps_str": "30/1",
        "pix_fmt": "yuv420p", "profile": "high",
        "primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
    }


def _capture_filter_chain(rotate: int) -> str:
    """Run render_video with stubs and return the -filter_complex string."""
    captured: dict = {}

    class FakeProc:
        pid = 99999
        def wait(self): return 0
        def communicate(self): return (b"", b"")
        @property
        def returncode(self): return 0

    def fake_popen(cmd, *a, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    with patch.object(overlay.subprocess, "Popen", side_effect=fake_popen), \
         patch.object(overlay, "_probe_src_fps", return_value=30.0):
        try:
            overlay.render_video(
                Path("/tmp/fake.mov"),  # raw_path — never opened thanks to the Popen patch
                start=0.0, duration=1.0,
                dst=Path("/tmp/out.mp4"),
                canvas=_canvas(),
                rotate=rotate,
            )
        except Exception:
            # render_video may try to wait/probe the fake output — that's fine,
            # we only care about the captured command.
            pass

    cmd = captured["cmd"]
    idx = cmd.index("-filter_complex")
    return cmd[idx + 1]


def test_no_rotation_chain_omitted():
    """rotate=0 produces a filter chain with no `transpose=` filter."""
    chain = _capture_filter_chain(0)
    assert "transpose" not in chain


def test_rotate_90_clockwise():
    """90° clockwise → single transpose=1, applied BEFORE scale."""
    chain = _capture_filter_chain(90)
    assert "transpose=1" in chain
    # The rotation must come before the scale, so the rotated dimensions
    # are what get fitted into the canvas.
    assert chain.index("transpose=1") < chain.index("scale=")


def test_rotate_180_double_transpose():
    """180° → two transpose=2 in sequence (cheap, avoids hflip+vflip pair)."""
    chain = _capture_filter_chain(180)
    assert "transpose=2,transpose=2" in chain


def test_rotate_270_counter_clockwise():
    """270° (i.e. 90° CCW) → transpose=2."""
    chain = _capture_filter_chain(270)
    assert "transpose=2" in chain
    # Must NOT be the 180° pattern — exactly one transpose=2 token, plus the
    # rest of the chain after it.
    assert "transpose=2,transpose=2" not in chain


def test_rotate_invalid_angle_raises():
    """Anything other than 0/90/180/270 is rejected at the render layer too,
    so a bug in the API can't silently produce an invalid filter chain."""
    import pytest
    with pytest.raises(ValueError, match="rotate must be"):
        # We don't even need the patch — render_video raises before Popen.
        overlay.render_video(
            Path("/tmp/fake.mov"), 0.0, 1.0, Path("/tmp/out.mp4"),
            canvas=_canvas(), rotate=45,
        )


def test_rotate_field_round_trips_through_sidecar(tmp_path):
    """The rotate API endpoint persists `rotate` to the sidecar and removes
    it (rather than storing 0) when the user cycles back to upright."""
    import asyncio
    import json
    import server
    from server import api_rotate_clip, RotateRequest

    # Minimal STATE wiring (same shape as test_split.py)
    round_dir = tmp_path / "round"
    for d in ("raw", "trims", "meta"):
        (round_dir / d).mkdir(parents=True)
    server.STATE.clear()
    server.STATE.update({
        "round": round_dir,
        "raw": round_dir / "raw",
        "trims": round_dir / "trims",
        "meta": round_dir / "meta",
        "scores": round_dir / "scores.json",
    })
    server.RENDER_STATUS.clear()

    sidecar = round_dir / "meta" / "IMG_9645.json"
    sidecar.write_text(json.dumps({
        "raw": "IMG_9645.MOV", "impact_s": 5.0, "pre": 1.5, "post": 4.0,
        "has_overlay": True, "review": "approved",
    }))

    # Stub the executor so we don't actually run ffmpeg.
    import asyncio as _asyncio

    class _Loop:
        def run_in_executor(self, executor, fn, *args): pass

    orig = _asyncio.get_event_loop
    _asyncio.get_event_loop = lambda: _Loop()  # type: ignore
    try:
        # Set rotation to 90
        result = asyncio.run(api_rotate_clip("IMG_9645", RotateRequest(rotate=90)))
        assert result["rotate"] == 90
        # has_overlay must flip false because the on-disk trim is now stale.
        assert result["has_overlay"] is False
        on_disk = json.loads(sidecar.read_text())
        assert on_disk["rotate"] == 90

        # Cycle to 180
        result = asyncio.run(api_rotate_clip("IMG_9645", RotateRequest(rotate=180)))
        assert result["rotate"] == 180

        # Back to 0 → field is removed (we don't keep `rotate: 0` noise).
        result = asyncio.run(api_rotate_clip("IMG_9645", RotateRequest(rotate=0)))
        assert "rotate" not in result or result.get("rotate") in (None, 0)
        on_disk = json.loads(sidecar.read_text())
        assert "rotate" not in on_disk
    finally:
        _asyncio.get_event_loop = orig
