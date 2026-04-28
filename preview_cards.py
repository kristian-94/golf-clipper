# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "numpy", "scipy", "soundfile"]
# ///
"""Render every card design to PNG so we can iterate on look & feel fast.

Outputs go to `vids/`:
    preview-overlay.png        — in-game corner card alone
    preview-overlay-on-frame.png — same card composited onto a real 4K frame
    preview-title.png          — title card
    preview-scorecard.png      — full 18-hole scorecard card
    preview-summary.png        — round summary card

Usage:
    uv run preview_cards.py                          # newest round
    uv run preview_cards.py --round clips/27-april-2026
    uv run preview_cards.py --hole 7                 # which hole the overlay shows
    uv run preview_cards.py --frame-from IMG_9482.mp4

Re-run after editing overlay.py / finalise.py — the PNGs refresh in place.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from batch_trim import newest_round, round_paths
from finalise import (
    pick_canvas_for_round,
    render_scorecard_card,
    render_summary_card,
    render_title_card,
)
from overlay import MARGIN, render_card

OUT = Path(__file__).parent / "vids"


def _player_breakdown_through(player: dict, holes: list[dict], up_to: int) -> dict:
    """Mock the same scoreboard format `render_card` consumes — totals through hole N."""
    par_by_hole = {h["number"]: h["par"] for h in holes}
    total = 0
    par_through = 0
    for s in player["scores"]:
        if s["hole"] <= up_to and s.get("strokes"):
            total += s["strokes"]
            par_through += par_by_hole.get(s["hole"], 0)
    return {
        "name": player["name"],
        "is_owner": player.get("is_owner", False),
        "total": total,
        "par_through": par_through,
    }


def grab_frame(video: Path, dst: Path, t: float = 1.0) -> None:
    """Pull a still frame so the overlay preview shows real-world contrast."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", f"{t:.2f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(dst)],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=Path, help="round folder (default: newest)")
    ap.add_argument("--clips-root", type=Path, default=Path("clips"))
    ap.add_argument("--hole", type=int, default=7,
                    help="hole number for the in-game overlay preview")
    ap.add_argument("--frame-from", default=None,
                    help="trim filename to grab a still frame from for the composite preview")
    args = ap.parse_args()

    round_dir = args.round or newest_round(args.clips_root)
    print(f"using round: {round_dir}")
    OUT.mkdir(parents=True, exist_ok=True)

    raw_dir, trims_dir, _ = round_paths(round_dir)
    canvas = pick_canvas_for_round(raw_dir)
    print(f"canvas: {canvas['width']}x{canvas['height']} @ {canvas['fps_str']}")

    scores_path = round_dir / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else None

    # --- in-game overlay ------------------------------------------------
    if scores:
        hole_obj = next((h for h in scores["holes"] if h["number"] == args.hole), scores["holes"][0])
        players = [_player_breakdown_through(p, scores["holes"], hole_obj["number"])
                   for p in scores["players"]]
        par = hole_obj["par"]
        hole_n = hole_obj["number"]
    else:
        # Fallback so the script works on rounds without a scorecard.
        players = [
            {"name": "Kristian", "is_owner": True,  "total": 32},
            {"name": "Raden",    "is_owner": False, "total": 38},
        ]
        par, hole_n = 4, args.hole

    card_scale = max(1.0, canvas["height"] / 1080.0)
    card = render_card(hole_n, par, players, shot=(2, 4), scale=card_scale)
    overlay_path = OUT / "preview-overlay.png"
    card.save(overlay_path)
    print(f"  → {overlay_path}  ({card.width}x{card.height})")

    # --- overlay composited onto a real frame ---------------------------
    candidates = sorted(p for p in trims_dir.glob("*.mp4") if p.stem != "final")
    if args.frame_from:
        src = trims_dir / args.frame_from
    elif candidates:
        src = candidates[0]
    else:
        src = None

    if src and src.exists():
        frame_tmp = OUT / "preview-frame.jpg"
        grab_frame(src, frame_tmp)
        bg = Image.open(frame_tmp).convert("RGBA")
        # Match the canvas spec, just like the live render does.
        if bg.size != (canvas["width"], canvas["height"]):
            bg = bg.resize((canvas["width"], canvas["height"]), Image.LANCZOS)
        margin = max(MARGIN, round(MARGIN * canvas["height"] / 1080))
        bg.alpha_composite(card, (canvas["width"] - card.width - margin, margin))
        composite_path = OUT / "preview-overlay-on-frame.png"
        bg.convert("RGB").save(composite_path, "PNG")
        print(f"  → {composite_path}  (composited on {src.name})")

    # --- full-screen cards ----------------------------------------------
    title = render_title_card(
        canvas,
        course=(scores or {}).get("course") or "Sample Course",
        round_name=round_dir.name,
        players=[p["name"] for p in (scores or {}).get("players", [])] or ["Kristian", "Raden"],
    )
    title_path = OUT / "preview-title.png"
    title.save(title_path)
    print(f"  → {title_path}")

    sc = render_scorecard_card(canvas, scores)
    sc_path = OUT / "preview-scorecard.png"
    sc.save(sc_path)
    print(f"  → {sc_path}")

    summary = render_summary_card(canvas, scores)
    sum_path = OUT / "preview-summary.png"
    summary.save(sum_path)
    print(f"  → {sum_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
