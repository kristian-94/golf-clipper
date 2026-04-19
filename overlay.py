# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Scorecard card rendering + a single-pass trim+overlay encoder.

The card image is built with Pillow; trim+overlay is one ffmpeg call so we
only re-encode each clip once. When `card` is None the function just trims.

Source resolution, framerate, pixel format, and bit depth pass through
unchanged so 1080p60 / 4K60 / 10-bit footage is preserved.
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
WIDTH = 260  # fixed; height grows with player count

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
) -> Image.Image:
    """Build a transparent RGBA card. `players` is leaderboard-ordered.

    `shot` is an optional (index, total) tuple — only set when clip count
    for the hole matched the owner's strokes for that hole.
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
    height = (
        PAD_Y
        + header_h
        + HEADER_GAP
        + row_h * len(players)
        + ROW_GAP * max(0, len(players) - 1)
        + PAD_Y
    )

    img = Image.new("RGBA", (WIDTH, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (0, 0, WIDTH - 1, height - 1),
        radius=RADIUS, fill=BG, outline=BORDER, width=1,
    )

    bar_top = PAD_Y - 2
    bar_bot = PAD_Y + header_h + 2
    draw.rounded_rectangle(
        (PAD_X - 10, bar_top, PAD_X - 6, bar_bot),
        radius=2, fill=ACCENT,
    )

    y = PAD_Y
    head_left = PAD_X
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
        draw.text((WIDTH - PAD_X - tag_w, y + 4), tag, font=f_meta, fill=MUTED)

    y += header_h + HEADER_GAP // 2
    draw.line((PAD_X, y, WIDTH - PAD_X, y), fill=BORDER, width=1)
    y += HEADER_GAP // 2

    for i, p in enumerate(players):
        if i:
            y += ROW_GAP
        name_color = OWNER if p.get("is_owner") else TEXT
        draw.text((PAD_X, y), p["name"], font=f_name, fill=name_color)
        score = str(p["total"])
        score_w = draw.textlength(score, font=f_score)
        draw.text((WIDTH - PAD_X - score_w, y - 2), score, font=f_score, fill=name_color)
        y += row_h

    return img


# --- Encoder ------------------------------------------------------------

def _probe_pix_fmt(src: Path) -> str | None:
    """Return the source video stream's pixel format, e.g. 'yuv420p10le'."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt",
         "-of", "default=nw=1:nk=1", str(src)],
        capture_output=True, text=True,
    )
    val = out.stdout.strip()
    return val or None


def render_video(
    raw_path: Path,
    start: float,
    duration: float,
    dst: Path,
    card: Image.Image | None = None,
) -> None:
    """Trim raw_path[start..start+duration] → dst as H.264.

    If `card` is provided, composites it top-right in the same pass — only
    one re-encode per clip. Pixel format / bit depth are matched to the
    source so 10-bit / 4K / 60fps survive untouched.
    """
    start = max(0.0, start)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_pix_fmt = _probe_pix_fmt(raw_path) or "yuv420p"
    profile = "high10" if "10" in src_pix_fmt else "high"

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
            cmd += [
                "-i", str(png_path),
                "-filter_complex",
                f"[0:v][1:v]overlay=x=W-w-{MARGIN}:y={MARGIN}:format=auto,format={src_pix_fmt}",
            ]
        cmd += [
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-profile:v", profile,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
    finally:
        if png_path:
            png_path.unlink(missing_ok=True)
