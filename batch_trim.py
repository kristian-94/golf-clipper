# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "soundfile", "pillow", "httpx"]
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
from datetime import datetime, timezone
from pathlib import Path

from course_map import extract_gps, load_course_geom, render_hole_map
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


def _to_utc_z(raw: str) -> str:
    """Normalize an ISO-8601 timestamp (possibly with a tz offset) to UTC with
    a trailing Z and microseconds — the format every other `recorded_at` uses,
    so the string-sorted chronology stays correct. Returns the input unchanged
    if it can't be parsed."""
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def probe_recorded_at(src: Path) -> str | None:
    """Return the clip's true capture time as ISO-8601 UTC (…Z), or None.

    Prefers Apple's `com.apple.quicktime.creationdate` — it carries the
    original capture instant *and* its timezone. iPhones reset the track-level
    `creation_time` to the transfer time when a clip is AirDropped or
    downloaded from another device, so for those clips `creation_time` is when
    it landed on this phone, not when it was filmed. We fall back to
    `creation_time` when the QuickTime tag is absent (e.g. Android clips).

    Used downstream to correlate the clip with a SmartCaddy hole boundary.
    Missing on screen-recorded or metadata-stripped files — caller should flag.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries",
         "format_tags=com.apple.quicktime.creationdate,creation_time",
         "-of", "default=noprint_wrappers=1", str(src)],
        capture_output=True, text=True,
    )
    tags: dict[str, str] = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("TAG:") and "=" in line:
            k, _, v = line[4:].partition("=")
            if v.strip():
                tags[k.strip()] = v.strip()
    val = tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
    return _to_utc_z(val) if val else None


def ensure_recorded_at(meta_dir: Path, raw_dir: Path) -> int:
    """Backfill `recorded_at` and `gps`/`gps_accuracy` on older sidecars.

    Returns the count of sidecars updated. Safe to call repeatedly.
    """
    n = 0
    for meta_path in sorted(meta_dir.glob("*.json")):
        data = json.loads(meta_path.read_text())
        changed = False

        if not data.get("recorded_at"):
            src = raw_dir / data["raw"]
            if src.exists():
                ts = probe_recorded_at(src)
                data["recorded_at"] = ts
                if ts is None:
                    data["flagged"] = True
                    data.setdefault("reasons", []).append(
                        "no creation_time in container metadata")
                changed = True

        # GPS field absent (sidecar predates the map feature).
        if "gps" not in data:
            src = raw_dir / data["raw"]
            if src.exists():
                gps, acc = extract_gps(src)
                data["gps"] = list(gps) if gps else None
                data["gps_accuracy"] = acc
                changed = True

        if changed:
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
    gps, gps_acc = extract_gps(src)

    if not cands:
        data = {
            "raw": src.name,
            "recorded_at": recorded_at,
            "gps": list(gps) if gps else None,
            "gps_accuracy": gps_acc,
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
        "gps": list(gps) if gps else None,
        "gps_accuracy": gps_acc,
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


def _card_for(data: dict, course_geom: dict | None = None, scale: float = 1.0):
    """Build the overlay card from sidecar fields, or None if not enough info.

    `course_geom` is optional — when present and the hole has OSM geometry,
    a small map is composited on the left of the scorecard.

    `scale` is passed through so the card matches the canvas resolution
    (1.0 at 1080p, ~2.0 at 4K).
    """
    if data.get("hole") is None or not data.get("players"):
        return None
    shot = None
    if data.get("shot_index") and data.get("shot_total"):
        shot = (data["shot_index"], data["shot_total"])

    hole_map = None
    if course_geom is not None:
        gps = data.get("gps")
        gps_tuple = (gps[0], gps[1]) if gps else None
        hole_map = render_hole_map(
            course_geom, data["hole"], gps_tuple, data.get("gps_accuracy"),
        )

    return render_card(
        data["hole"], data["par"], data["players"],
        shot=shot, hole_map=hole_map, scale=scale,
        active_player=data.get("player"),
    )


def render_from_sidecar(
    meta_path: Path,
    raw_dir: Path,
    trims_dir: Path,
    canvas: dict | None = None,
    pid_set: set | None = None,
    with_overlay: bool = True,
) -> str:
    """Render the trim from sidecar data, baking in the scorecard if present.

    Each trim is normalised to `canvas` so all of the round's trims share
    one codec/resolution/fps/colorspace and can be byte-copied together by
    `finalise.py`. If `canvas` is None we compute it from `raw_dir` — but
    callers in a loop should compute it once and pass it in (otherwise we
    re-probe every clip on every render).

    Pass `with_overlay=False` to skip the scorecard composite — useful for
    the initial preview pass (faster encode, just confirms the trim window).
    The sidecar's `has_overlay` flag records what the on-disk trim contains
    so an approve can decide whether a re-render is needed.

    Returns "ok" / "no-impact" / "no-raw".
    """
    data = json.loads(meta_path.read_text())
    if data.get("impact_s") is None:
        return "no-impact"
    src = raw_dir / data["raw"]
    if not src.exists():
        return "no-raw"
    if canvas is None:
        # Lazy single-clip path; expensive in a loop, fine for one-off CLI use.
        from finalise import pick_canvas_for_round
        canvas = pick_canvas_for_round(raw_dir)
    dst = trims_dir / f"{meta_path.stem}.mp4"
    start = max(0.0, data["impact_s"] - data["pre"])
    duration = data["pre"] + data["post"]
    card = None
    if with_overlay:
        # Round dir is the meta dir's parent; course geom is optional.
        course_geom = load_course_geom(meta_path.parent.parent)
        # Card layout was tuned at 1080p. Scale every pixel-space dimension
        # by canvas_h/1080 so the card stays the same fraction at 4K.
        card_scale = max(1.0, canvas["height"] / 1080.0)
        card = _card_for(data, course_geom=course_geom, scale=card_scale)
    render_video(src, start, duration, dst,
                 canvas=canvas,
                 card=card,
                 pid_set=pid_set,
                 rotate=int(data.get("rotate", 0) or 0))
    # Race-safe write-back: re-read the sidecar in case it was modified
    # while ffmpeg ran (cascade renumber, manual edit, player change…).
    # We only own `trimmed_at` and `has_overlay` — everything else stays
    # at whatever's currently on disk so we don't clobber user edits.
    # `has_overlay` is true only when the on-disk hole/shot/player still
    # match what we rendered against — otherwise leave it False so the
    # render queue will pick this clip up again.
    on_disk = json.loads(meta_path.read_text())
    rendered_against = (
        data.get("hole"), data.get("shot_index"),
        data.get("shot_total"), data.get("player"),
        data.get("impact_s"), data.get("rotate"),
    )
    current = (
        on_disk.get("hole"), on_disk.get("shot_index"),
        on_disk.get("shot_total"), on_disk.get("player"),
        on_disk.get("impact_s"), on_disk.get("rotate"),
    )
    on_disk["trimmed_at"] = datetime.now().isoformat(timespec="seconds")
    on_disk["has_overlay"] = (card is not None) and rendered_against == current
    meta_path.write_text(json.dumps(on_disk, indent=2))
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

    # Compute the canvas spec once for the round so every per-clip render
    # uses the same target (lets finalise.py concat-copy with no re-encode).
    from finalise import pick_canvas_for_round
    canvas = pick_canvas_for_round(raw_dir)
    print(f"canvas: {canvas['width']}x{canvas['height']} @ {canvas['fps_str']} "
          f"({canvas['pix_fmt']}, {canvas['primaries']}/{canvas['transfer']}/"
          f"{canvas['matrix']})")

    if args.re_render:
        metas = sorted(meta_dir.glob("*.json"))
        print(f"round: {round_dir}  ({len(metas)} sidecars)")
        for i, meta_path in enumerate(metas, 1):
            print(f"[{i:>3}/{len(metas)}] {meta_path.stem}", end=" ", flush=True)
            try:
                status = render_from_sidecar(meta_path, raw_dir, trims_dir, canvas)
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
            print(f"-> {render_from_sidecar(meta_path, raw_dir, trims_dir, canvas)}")
        except Exception as e:
            print(f"ERROR: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
