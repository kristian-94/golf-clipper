# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "soundfile", "pillow"]
# ///
"""Batch-trim every video in a round folder.

Layout:
    clips/<round>/
        raw-<round>/   ← input videos (untouched)
        trims/         ← .mp4 outputs with overlays baked in (created)
        meta/          ← .json sidecars per clip (created)
        scores.json    ← optional, from smartcaddy.py — enables overlays

Detection (audio impact + ffprobe creation_time) and rendering (single
ffmpeg pass that trims + composites the scorecard if available) are kept
separate so we can:
  • run detection on a fresh batch without needing the scorecard yet
  • re-render any clip from its sidecar (Save in compare view, bulk re-render)
  • avoid paying the encode cost twice

Sidecar shape:
    {
      "raw": "003_IMG_9356.MOV",
      "recorded_at": "2026-04-18T02:51:46.000000Z",
      "impact_s": 12.345,
      "pre": 1.5,
      "post": 4.0,
      "confidence": "strong" | "ok" | "weak",
      "flagged": false,
      "reasons": [...],
      "edited": false,
      "trimmed_at": "2026-04-19T08:50:00",
      "review": "approved",            # added later by the web UI
      "hole": 7, "par": 4, "players": [...],   # added by correlate.py
      "shot_index": 2, "shot_total": 4         # optional
    }
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from impact_trim import SR, assess, extract_audio, find_impacts
from overlay import render_card, render_video

VIDEO_EXTS = {".mov", ".mp4", ".m4v"}
DEFAULT_PRE = 1.5
DEFAULT_POST = 4.0


def newest_round(clips_root: Path) -> Path:
    rounds = [d for d in clips_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not rounds:
        sys.exit(f"no round folders under {clips_root}")
    return max(rounds, key=lambda d: d.stat().st_mtime)


def round_paths(round_dir: Path) -> tuple[Path, Path, Path]:
    """Return (raw_dir, trims_dir, meta_dir). Creates trims/meta if missing."""
    raw_candidates = [
        d for d in round_dir.iterdir()
        if d.is_dir() and d.name.startswith("raw-")
    ]
    raw = raw_candidates[0] if raw_candidates else round_dir
    trims = round_dir / "trims"
    meta = round_dir / "meta"
    trims.mkdir(exist_ok=True)
    meta.mkdir(exist_ok=True)
    return raw, trims, meta


def list_videos(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")
    )


def probe_recorded_at(src: Path) -> str | None:
    """Return ISO-8601 creation_time from container metadata, or None.

    Used downstream to correlate the clip with a SmartCaddy hole boundary.
    Missing on screen-recorded or metadata-stripped files — caller should flag.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format_tags=creation_time",
         "-of", "default=nw=1:nk=1", str(src)],
        capture_output=True, text=True,
    )
    val = out.stdout.strip()
    return val or None


def ensure_recorded_at(meta_dir: Path, raw_dir: Path) -> int:
    """Backfill `recorded_at` on existing sidecars that don't have it.

    Returns the count of sidecars updated. Safe to call repeatedly.
    """
    n = 0
    for meta_path in sorted(meta_dir.glob("*.json")):
        data = json.loads(meta_path.read_text())
        if data.get("recorded_at"):
            continue
        src = raw_dir / data["raw"]
        if not src.exists():
            continue
        ts = probe_recorded_at(src)
        data["recorded_at"] = ts
        if ts is None:
            data["flagged"] = True
            data.setdefault("reasons", []).append("no creation_time in container metadata")
        meta_path.write_text(json.dumps(data, indent=2))
        n += 1
    return n


def detect_clip(
    src: Path,
    meta_dir: Path,
    pre: float = DEFAULT_PRE,
    post: float = DEFAULT_POST,
    force: bool = False,
) -> dict:
    """Detect impact + record_at, write sidecar. **Does not render.**

    Render is deferred to render_from_sidecar so a single ffmpeg pass can
    bake the scorecard overlay in once correlation has run.
    """
    meta_path = meta_dir / f"{src.stem}.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    audio = extract_audio(src)
    cands = find_impacts(audio, SR)
    recorded_at = probe_recorded_at(src)

    if not cands:
        data = {
            "raw": src.name,
            "recorded_at": recorded_at,
            "impact_s": None,
            "pre": pre,
            "post": post,
            "confidence": "weak",
            "flagged": True,
            "reasons": ["no impact-like transient detected"],
            "edited": False,
            "trimmed_at": None,
        }
        if recorded_at is None:
            data["reasons"].append("no creation_time in container metadata")
        meta_path.write_text(json.dumps(data, indent=2))
        return data

    best = cands[0]
    verdict = assess(cands)
    reasons = list(verdict.reasons)
    flagged = verdict.flagged
    if recorded_at is None:
        reasons.append("no creation_time in container metadata")
        flagged = True

    data = {
        "raw": src.name,
        "recorded_at": recorded_at,
        "impact_s": best.time_s,
        "pre": pre,
        "post": post,
        "confidence": verdict.confidence,
        "flagged": flagged,
        "reasons": reasons,
        "edited": False,
        "trimmed_at": None,
    }
    meta_path.write_text(json.dumps(data, indent=2))
    return data


def _card_for(data: dict):
    """Build the overlay card from sidecar fields, or None if not enough info."""
    if data.get("hole") is None or not data.get("players"):
        return None
    shot = None
    if data.get("shot_index") and data.get("shot_total"):
        shot = (data["shot_index"], data["shot_total"])
    return render_card(data["hole"], data["par"], data["players"], shot=shot)


def render_from_sidecar(meta_path: Path, raw_dir: Path, trims_dir: Path) -> str:
    """Render the trim from sidecar data, baking in the scorecard if present.

    Returns "ok" / "no-impact" / "no-raw".
    """
    data = json.loads(meta_path.read_text())
    if data.get("impact_s") is None:
        return "no-impact"
    src = raw_dir / data["raw"]
    if not src.exists():
        return "no-raw"
    dst = trims_dir / f"{meta_path.stem}.mp4"
    start = max(0.0, data["impact_s"] - data["pre"])
    duration = data["pre"] + data["post"]
    render_video(src, start, duration, dst, card=_card_for(data))
    data["trimmed_at"] = datetime.now().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(data, indent=2))
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=Path, help="round folder (default: newest under --clips-root)")
    ap.add_argument("--clips-root", type=Path, default=Path("clips"))
    ap.add_argument("--force", action="store_true", help="re-detect clips that already have sidecars")
    ap.add_argument("--re-render", action="store_true",
                    help="skip detection; re-render every clip from its sidecar "
                         "(uses current scorecard correlation if scores.json exists)")
    args = ap.parse_args()

    round_dir = args.round or newest_round(args.clips_root)
    raw_dir, trims_dir, meta_dir = round_paths(round_dir)

    if args.re_render:
        metas = sorted(meta_dir.glob("*.json"))
        print(f"round: {round_dir}  ({len(metas)} sidecars)")
        for i, meta_path in enumerate(metas, 1):
            print(f"[{i:>3}/{len(metas)}] {meta_path.stem}", end=" ", flush=True)
            try:
                status = render_from_sidecar(meta_path, raw_dir, trims_dir)
                print(f"-> {status}")
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
        return 0

    videos = list_videos(raw_dir)
    print(f"round: {round_dir}  ({len(videos)} clips)")
    for i, src in enumerate(videos, 1):
        print(f"[{i:>3}/{len(videos)}] {src.name}", end=" ", flush=True)
        try:
            data = detect_clip(src, meta_dir, force=args.force)
            flag = " FLAG" if data["flagged"] else ""
            print(f"-> {data['confidence']}{flag}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()

    # Render after detection so we only encode each clip once. If a scorecard
    # was correlated, the overlay is included.
    metas = sorted(meta_dir.glob("*.json"))
    print(f"\nrendering {len(metas)} trims")
    for i, meta_path in enumerate(metas, 1):
        print(f"[{i:>3}/{len(metas)}] {meta_path.stem}", end=" ", flush=True)
        try:
            print(f"-> {render_from_sidecar(meta_path, raw_dir, trims_dir)}")
        except Exception as e:
            print(f"ERROR: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
