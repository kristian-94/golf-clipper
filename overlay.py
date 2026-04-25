# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Scorecard card rendering + a single-pass trim+overlay+normalize encoder.

The card image is built with Pillow; trim, overlay composite, canvas
normalisation (scale/pad/fps/pix_fmt) and final encode all happen in one
ffmpeg call so each clip is encoded exactly once. The output is then
byte-copied straight into the round's `final.mp4` by `finalise.py`'s
concat demuxer — no second encode, no quality drop.

Every per-clip output matches the round's canvas spec (dimensions, fps,
pix_fmt, profile, color tags, audio params) so the concat demuxer can
stream-copy them. The canvas is computed once for the round by
`finalise.pick_canvas_for_round`.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Card style ---------------------------------------------------------

# Position relative to the source video's pixel space.
MARGIN = 32

PAD_X = 24
PAD_Y = 18
ROW_GAP = 8
HEADER_GAP = 14
WIDTH = 260  # scoreboard column; total card width grows when a map is added

# Hole-map layout (only used when render_card receives `hole_map`)
MAP_PAD = 8          # gutter between map and scoreboard

# Colors (RGBA)
BG       = (14, 17, 22, 215)        # near-black, ~85% opaque
BORDER   = (255, 255, 255, 30)      # very faint hairline
ACCENT   = (88, 166, 255, 255)      # blue — matches the web UI
TEXT     = (230, 237, 243, 255)
MUTED    = (139, 148, 158, 255)
OWNER    = (255, 255, 255, 255)
RADIUS   = 14

FONT_PATHS = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            try:
                # HelveticaNeue.ttc index 1 ~ bold; 0 ~ regular. Best-effort.
                idx = 1 if bold and path.endswith(".ttc") else 0
                return ImageFont.truetype(path, size, index=idx)
            except OSError:
                continue
    return ImageFont.load_default()


def render_card(
    hole: int,
    par: int,
    players: list[dict],
    shot: tuple[int, int] | None = None,
    hole_map: Image.Image | None = None,
) -> Image.Image:
    """Build a transparent RGBA card. `players` is leaderboard-ordered.

    `shot` is an optional (index, total) tuple — only set when clip count
    for the hole matched the owner's strokes for that hole.

    `hole_map` is an optional pre-rendered RGBA image of the hole geometry
    (from `course_map.render_hole_map`). When supplied, it's composited as a
    column on the left of the scoreboard and the card grows accordingly.
    """
    f_header = _font(22, bold=True)
    f_meta   = _font(14, bold=False)
    f_name   = _font(18, bold=True)
    f_score  = _font(20, bold=True)

    tmp = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(tmp)

    def text_h(font: ImageFont.FreeTypeFont) -> int:
        bbox = d.textbbox((0, 0), "Hg", font=font)
        return bbox[3] - bbox[1]

    header_h = text_h(f_header)
    row_h    = max(text_h(f_name), text_h(f_score))
    sb_height = (
        PAD_Y
        + header_h
        + HEADER_GAP
        + row_h * len(players)
        + ROW_GAP * max(0, len(players) - 1)
        + PAD_Y
    )

    # Map column width is just the map's width + a small gap to the scoreboard.
    # No box around the map — it floats freely on the left.
    map_col_w = (hole_map.width + MAP_PAD) if hole_map else 0
    total_w = map_col_w + WIDTH
    # Total card height fits whichever is taller — the scoreboard box or the map.
    total_h = max(sb_height, hole_map.height) if hole_map else sb_height

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    sb_left = map_col_w
    sb_right = total_w
    sb_top = 0  # scoreboard anchored to the top — the card itself is top-right pinned

    # Box wraps the scoreboard column only — the map is unboxed.
    draw.rounded_rectangle(
        (sb_left, sb_top, sb_right - 1, sb_top + sb_height - 1),
        radius=RADIUS, fill=BG, outline=BORDER, width=1,
    )

    if hole_map:
        # Top-align the map with the scoreboard so the visual top edge is shared.
        img.paste(hole_map, (0, 0), hole_map)

    bar_top = sb_top + PAD_Y - 2
    bar_bot = sb_top + PAD_Y + header_h + 2
    draw.rounded_rectangle(
        (sb_left + PAD_X - 10, bar_top, sb_left + PAD_X - 6, bar_bot),
        radius=2, fill=ACCENT,
    )

    y = sb_top + PAD_Y
    head_left = sb_left + PAD_X
    draw.text((head_left, y), f"HOLE {hole}", font=f_header, fill=TEXT)
    hole_w = draw.textlength(f"HOLE {hole}", font=f_header)
    sep = "   ·   "
    draw.text((head_left + hole_w, y + 2), sep, font=f_meta, fill=MUTED)
    sep_w = draw.textlength(sep, font=f_meta)
    draw.text((head_left + hole_w + sep_w, y), f"PAR {par}", font=f_header, fill=ACCENT)
    if shot:
        idx, total = shot
        tag = f"SHOT {idx}/{total}"
        tag_w = draw.textlength(tag, font=f_meta)
        draw.text((sb_right - PAD_X - tag_w, y + 4), tag, font=f_meta, fill=MUTED)

    y += header_h + HEADER_GAP // 2
    draw.line((sb_left + PAD_X, y, sb_right - PAD_X, y), fill=BORDER, width=1)
    y += HEADER_GAP // 2

    for i, p in enumerate(players):
        if i:
            y += ROW_GAP
        name_color = OWNER if p.get("is_owner") else TEXT
        draw.text((sb_left + PAD_X, y), p["name"], font=f_name, fill=name_color)
        score = str(p["total"])
        score_w = draw.textlength(score, font=f_score)
        draw.text((sb_right - PAD_X - score_w, y - 2), score, font=f_score, fill=name_color)
        y += row_h

    return img


# --- Encoder ------------------------------------------------------------

# Encode quality for the per-clip render. This is now the ONLY encode in
# the whole pipeline (concat is byte-copy), so we spend bits here.
CRF = "12"
PRESET = "slow"


def render_video(
    raw_path: Path,
    start: float,
    duration: float,
    dst: Path,
    canvas: dict,
    card: Image.Image | None = None,
) -> None:
    """Trim raw_path[start..start+duration] → dst, normalised to canvas.

    Single ffmpeg pass: trim → scale+pad to canvas dims → fps adjust →
    pix_fmt convert → optional card overlay → libx264 encode with canvas
    color tags. Audio is normalised to 48 kHz stereo AAC 192 k. The output
    matches every other clip in the round so `finalise.py` can stream-copy
    them all together with no second encode.

    `canvas` is the dict returned by `finalise.pick_canvas_for_round`:
    keys `width`, `height`, `fps_str`, `pix_fmt`, `profile`, `primaries`,
    `transfer`, `matrix`.
    """
    start = max(0.0, start)
    dst.parent.mkdir(parents=True, exist_ok=True)

    W = canvas["width"]
    H = canvas["height"]
    fps_str = canvas["fps_str"]
    pix_fmt = canvas["pix_fmt"]
    profile = canvas["profile"]

    # Build the video filter chain. Lower-res clips are upscaled with Lanczos
    # and letterboxed; lower-fps clips get frame-doubled by the fps filter
    # (no interpolation — visually the clip plays at its source rate inside
    # a higher-fps container, which is what we want).
    base_chain = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={fps_str},setsar=1,format={pix_fmt}"
    )

    # `-ss` and `-t` are both input options for the source video here. Putting
    # `-t` after the first `-i` would make it bind to the *next* input (the
    # PNG), not cap the output — bug we hit earlier where trims ran the full
    # length from impact to end.
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(raw_path),
    ]

    png_path: Path | None = None
    try:
        if card is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                png_path = Path(tmp.name)
            card.save(png_path)
            # The overlay filter promotes the canvas to 4:4:4 to do alpha
            # blending with the RGBA card; libx264's `high10` profile only
            # supports 4:2:0 / 4:2:2, so we explicitly format back to the
            # canvas pix_fmt after the overlay to keep the encoder happy.
            cmd += [
                "-i", str(png_path),
                "-filter_complex",
                f"[0:v]{base_chain}[base];"
                f"[base][1:v]overlay=x=W-w-{MARGIN}:y={MARGIN}:format=auto,"
                f"format={pix_fmt}[v];"
                f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a]",
                "-map", "[v]", "-map", "[a]",
            ]
        else:
            cmd += [
                "-filter_complex",
                f"[0:v]{base_chain}[v];"
                f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a]",
                "-map", "[v]", "-map", "[a]",
            ]
        cmd += [
            "-c:v", "libx264", "-crf", CRF, "-preset", PRESET,
            "-profile:v", profile,
            # Tag the output with the canvas's color characteristics — both
            # the libavformat container flags AND the x264 VUI need this, or
            # the file ends up "color_*=unknown" and players fall back to
            # generic BT.709 decoding (visible washed-out look).
            "-color_primaries", canvas["primaries"],
            "-color_trc", canvas["transfer"],
            "-colorspace", canvas["matrix"],
            "-color_range", "tv",
            "-x264-params",
            f"colorprim={canvas['primaries']}:transfer={canvas['transfer']}"
            f":colormatrix={canvas['matrix']}",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
    finally:
        if png_path:
            png_path.unlink(missing_ok=True)
