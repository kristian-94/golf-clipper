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

PAD_X = 28
PAD_Y = 22
ROW_GAP = 10
HEADER_GAP = 18
WIDTH = 310          # scoreboard column; total card width grows when a map is added

# Hole-map layout (only used when render_card receives `hole_map`)
MAP_PAD = 10         # gutter between map and scoreboard

# Colors (RGBA). The in-game card uses BG (translucent) so the frame shows
# through. The full-screen cards in finalise.py use BG_TOP/BG_BOT (opaque
# gradient endpoints).
BG        = (8, 12, 18, 170)        # translucent cool-blue-tinted dark glass
BG_TOP    = (10, 14, 22, 255)
BG_BOT    = (4, 6, 12, 255)
BORDER    = (255, 255, 255, 28)
ACCENT    = (96, 165, 250, 255)     # broadcast blue (sky-400-ish, not bootstrap)
ACCENT2   = (74, 175, 110, 255)     # course green — secondary accent
TEXT      = (240, 245, 250, 255)
MUTED     = (148, 156, 165, 255)
OWNER     = (255, 255, 255, 255)
OWNER_BG  = (96, 165, 250, 38)      # soft blue wash behind owner row
UNDER_PAR = (96, 165, 250, 255)     # birdie/par → blue
OVER_PAR  = (232, 120, 110, 255)    # triple+ → soft red
RADIUS    = 16

# Pillow's `truetype` accepts variable fonts; we apply a weight via
# set_variation_by_name when the font supports it (SF Pro / SF Compact do).
# Order: SF Compact (display, scoreboard-y) → SF Pro → Helvetica fallback.
FONT_CANDIDATES = [
    ("/System/Library/Fonts/SFCompact.ttf", "variable"),
    ("/System/Library/Fonts/SFNS.ttf", "variable"),
    ("/System/Library/Fonts/HelveticaNeue.ttc", "ttc"),
    ("/System/Library/Fonts/Helvetica.ttc", "ttc"),
    ("/Library/Fonts/Arial.ttf", "plain"),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path, kind in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            if kind == "ttc":
                idx = 1 if bold else 0
                return ImageFont.truetype(path, size, index=idx)
            font = ImageFont.truetype(path, size)
            if kind == "variable":
                # SF variable fonts expose named instances like "Bold",
                # "Semibold", "Regular". Pillow raises if the name isn't
                # found, so guard each call.
                try:
                    font.set_variation_by_name("Bold" if bold else "Regular")
                except (OSError, ValueError):
                    pass
            return font
        except OSError:
            continue
    return ImageFont.load_default()


def _vgradient(w: int, h: int, top: tuple, bot: tuple) -> Image.Image:
    """Vertical RGBA gradient — used as the card body so it doesn't read flat."""
    grad = Image.new("RGBA", (w, h), top)
    px = grad.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = round(top[0] + (bot[0] - top[0]) * t)
        g = round(top[1] + (bot[1] - top[1]) * t)
        b = round(top[2] + (bot[2] - top[2]) * t)
        a = round(top[3] + (bot[3] - top[3]) * t)
        for x in range(w):
            px[x, y] = (r, g, b, a)
    return grad


def _rounded_mask(w: int, h: int, radius: int) -> Image.Image:
    """L-mode mask of a rounded rectangle — used to clip the gradient body."""
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    return m


def render_card(
    hole: int,
    par: int,
    players: list[dict],
    shot: tuple[int, int] | None = None,
    hole_map: Image.Image | None = None,
    scale: float = 1.0,
    active_player: str | None = None,
) -> Image.Image:
    """Build a translucent RGBA card. `players` is leaderboard-ordered.

    Inline header ("HOLE 7 · PAR 4 ... SHOT 2/4"), divider, then one row per
    player: name | ±par | total. Background is a single semi-transparent
    fill — the frame shows through, which reads better in motion than an
    opaque gradient.
    """
    s = scale
    px = lambda v: max(1, round(v * s))
    pad_x = px(PAD_X); pad_y = px(PAD_Y)
    row_gap = px(ROW_GAP); header_gap = px(HEADER_GAP)
    width = px(WIDTH); map_pad = px(MAP_PAD); radius = px(RADIUS)

    f_header = _font(px(22), bold=True)
    f_meta   = _font(px(14), bold=False)
    f_name   = _font(px(19), bold=True)
    f_score  = _font(px(22), bold=True)
    f_diff   = _font(px(15), bold=True)

    if hole_map is not None and s != 1.0:
        hole_map = hole_map.resize(
            (px(hole_map.width), px(hole_map.height)),
            Image.LANCZOS,
        )

    tmp = Image.new("RGBA", (1, 1))
    d = ImageDraw.Draw(tmp)

    def text_h(font: ImageFont.FreeTypeFont) -> int:
        bbox = d.textbbox((0, 0), "Hg", font=font)
        return bbox[3] - bbox[1]

    header_h = text_h(f_header)
    row_h    = max(text_h(f_name), text_h(f_score))
    sb_height = (
        pad_y
        + header_h
        + header_gap
        + row_h * len(players)
        + row_gap * max(0, len(players) - 1)
        + pad_y
    )

    map_col_w = (hole_map.width + map_pad) if hole_map else 0
    total_w = map_col_w + width
    total_h = max(sb_height, hole_map.height) if hole_map else sb_height

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    sb_left = map_col_w
    sb_right = total_w
    sb_top = 0

    # Single semi-transparent fill — translucent enough to see the frame
    # through it. BG carries an alpha of ~150 (see constants).
    draw.rounded_rectangle(
        (sb_left, sb_top, sb_right - 1, sb_top + sb_height - 1),
        radius=radius, fill=BG, outline=BORDER, width=1,
    )

    if hole_map:
        img.paste(hole_map, (0, 0), hole_map)

    # Small gold accent bar to the left of the header.
    bar_top = sb_top + pad_y - px(2)
    bar_bot = sb_top + pad_y + header_h + px(2)
    draw.rounded_rectangle(
        (sb_left + pad_x - px(10), bar_top, sb_left + pad_x - px(6), bar_bot),
        radius=px(2), fill=ACCENT,
    )

    # --- Inline header: HOLE 7 · PAR 4   …   SHOT 2/4
    y = sb_top + pad_y
    head_left = sb_left + pad_x
    draw.text((head_left, y), f"HOLE {hole}", font=f_header, fill=TEXT)
    hole_w = draw.textlength(f"HOLE {hole}", font=f_header)
    sep = "   ·   "
    draw.text((head_left + hole_w, y + px(2)), sep, font=f_meta, fill=MUTED)
    sep_w = draw.textlength(sep, font=f_meta)
    draw.text((head_left + hole_w + sep_w, y), f"PAR {par}",
              font=f_header, fill=ACCENT)
    if shot:
        idx, total = shot
        tag = f"SHOT {idx}/{total}"
        tag_w = draw.textlength(tag, font=f_meta)
        draw.text((sb_right - pad_x - tag_w, y + px(4)),
                  tag, font=f_meta, fill=MUTED)

    y += header_h + header_gap // 2
    draw.line((sb_left + pad_x, y, sb_right - pad_x, y),
              fill=BORDER, width=1)
    y += header_gap // 2

    # --- Player rows: name | ±par | total
    score_col_w = px(46)
    for i, p in enumerate(players):
        if i:
            y += row_gap
        name_color = OWNER if p.get("is_owner") else TEXT
        is_active = active_player is not None and p.get("name") == active_player

        # Highlight the active player — just a thin accent bar to the left
        # of the row, no background fill. Card stays uniform & opaque.
        if is_active:
            row_top = y - px(3)
            row_bot = y + row_h + px(2)
            draw.rounded_rectangle(
                (sb_left + pad_x - px(10), row_top + px(1),
                 sb_left + pad_x - px(6), row_bot - px(1)),
                radius=px(2), fill=ACCENT,
            )

        # Name
        draw.text((sb_left + pad_x, y), p["name"],
                  font=f_name, fill=name_color)

        # Total (far right)
        score = str(p["total"])
        score_w = draw.textlength(score, font=f_score)
        draw.text((sb_right - pad_x - score_w, y - px(2)),
                  score, font=f_score, fill=name_color)

        # ±par sits between the name and the total. Sidecars without
        # par_through (older clips) just don't render this column.
        diff = p.get("diff")
        if diff is None and "par_through" in p:
            diff = p["total"] - p["par_through"]
        if diff is not None:
            if diff < 0:
                diff_str, diff_color = str(diff), UNDER_PAR
            elif diff > 0:
                diff_str, diff_color = f"+{diff}", OVER_PAR
            else:
                diff_str, diff_color = "E", MUTED
            diff_w = draw.textlength(diff_str, font=f_diff)
            diff_x = sb_right - pad_x - score_col_w - diff_w
            draw.text((diff_x, y + px(4)),
                      diff_str, font=f_diff, fill=diff_color)

        y += row_h

    return img


# --- Encoder ------------------------------------------------------------

# Encode quality for the per-clip render. This is now the ONLY encode in
# the whole pipeline (concat is byte-copy), so we spend bits here.
CRF = "16"
PRESET = "medium"


def _probe_src_fps(path: Path) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    s = out.stdout.strip()
    if not s:
        return None
    if "/" in s:
        n, _, d = s.partition("/")
        try:
            return float(n) / float(d) if float(d) else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def render_video(
    raw_path: Path,
    start: float,
    duration: float,
    dst: Path,
    canvas: dict,
    card: Image.Image | None = None,
    pid_set: set | None = None,
    rotate: int = 0,
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
    canvas_fps = float(canvas["fps"])
    pix_fmt = canvas["pix_fmt"]
    profile = canvas["profile"]
    # Margin scales with canvas height so the card-to-edge gap stays
    # visually constant at 1080p / 4K.
    margin = max(MARGIN, round(MARGIN * H / 1080))

    # Detect source fps. If the source is faster than the canvas (e.g. iPhone
    # 240fps slo-mo on a 60fps canvas), we want iMovie-style behaviour: keep
    # every frame, stretch duration to the canvas fps so it plays as actual
    # slow motion. Without retiming, the fps filter would drop 3 of every 4
    # frames and play at "real" speed — losing the slo-mo intent entirely.
    src_fps = _probe_src_fps(raw_path) or canvas_fps
    slowmo = src_fps > canvas_fps * 1.05  # small margin for 60 vs 60000/1001
    if slowmo:
        retime = src_fps / canvas_fps
        retime_chain = f"setpts=PTS*{retime:.6f},"
        # Match audio duration to the stretched video. Slo-mo audio at the
        # source pitch sounds wrong (low-pitched), so we mute it — same as
        # iMovie's default slo-mo behaviour. atempo can't go below 0.5 in
        # one step, so chain factors of 0.5 until we reach the target.
        speed = 1.0 / retime
        factors: list[float] = []
        while speed < 0.5:
            factors.append(0.5); speed *= 2
        factors.append(speed)
        audio_extra = "," + ",".join(f"atempo={f:.6f}" for f in factors) + ",volume=0"
    else:
        retime_chain = ""
        audio_extra = ""

    # Optional rotation. Applied BEFORE scale so a 90°-rotated portrait
    # source is letterboxed correctly into a landscape canvas.
    #   90  → transpose=1 (90° clockwise)
    #   180 → transpose=2,transpose=2 (180°)
    #   270 → transpose=2 (90° counter-clockwise)
    if rotate == 0:
        rotate_chain = ""
    elif rotate == 90:
        rotate_chain = "transpose=1,"
    elif rotate == 180:
        rotate_chain = "transpose=2,transpose=2,"
    elif rotate == 270:
        rotate_chain = "transpose=2,"
    else:
        raise ValueError(f"rotate must be 0/90/180/270, got {rotate}")

    # Build the video filter chain. Lower-res clips are upscaled with Lanczos
    # and letterboxed; lower-fps clips get frame-doubled by the fps filter
    # (no interpolation — visually the clip plays at its source rate inside
    # a higher-fps container, which is what we want).
    base_chain = (
        f"{retime_chain}"
        f"{rotate_chain}"
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
                f"[base][1:v]overlay=x=W-w-{margin}:y={margin}:format=auto,"
                f"format={pix_fmt}[v];"
                f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo{audio_extra}[a]",
                "-map", "[v]", "-map", "[a]",
            ]
        else:
            cmd += [
                "-filter_complex",
                f"[0:v]{base_chain}[v];"
                f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo{audio_extra}[a]",
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
        # Popen + register PID so the server can SIGTERM only OUR ffmpegs
        # on cancel (instead of `pgrep -x ffmpeg` which also catches an
        # unrelated finalise running in parallel).
        proc = subprocess.Popen(cmd)
        if pid_set is not None:
            pid_set.add(proc.pid)
        try:
            rc = proc.wait()
        finally:
            if pid_set is not None:
                pid_set.discard(proc.pid)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    finally:
        if png_path:
            png_path.unlink(missing_ok=True)
