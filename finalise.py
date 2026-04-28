# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Concatenate approved trims into a single export-ready video.

Output: clips/<round>/final.mp4 — a title card, every approved trim in
chronological order, then a scorecard card.

**No re-encode.** Each per-clip trim is already encoded to the round's
canvas spec by `overlay.render_video` (one encode per clip, ever). This
module just renders the title/scorecard cards as short mp4 segments
matching the same canvas spec, then runs the ffmpeg concat demuxer with
`-c copy` to byte-append everything into final.mp4. Total runtime for
the concat step is seconds; the per-card encodes dominate.

This is the architecture that fixes the colour/grain regression we saw
when finalise re-encoded the (already-encoded) trims a second time.

Canvas selection (`pick_canvas_for_round`):
  • width  = max longer-side across all raw clips
  • height = max shorter-side
  • fps    = max framerate
  • HDR    = HLG (bt2020/arib-std-b67/bt2020nc, 10-bit, profile high10)
             if any clip is HDR; otherwise BT.709 SDR. We don't tonemap.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from overlay import (
    ACCENT, ACCENT2, BG_BOT, BG_TOP, BORDER, MUTED, OVER_PAR, OWNER, OWNER_BG,
    TEXT, UNDER_PAR, _font, _rounded_mask, _vgradient,
)

# Default per-card durations. The HTTP API can override per-render.
DEFAULT_DURATIONS = {
    "title": 3.0,
    "scorecard": 7.0,
    "summary": 6.0,
}
DEFAULT_START_CARDS = [{"kind": "title", "seconds": DEFAULT_DURATIONS["title"]}]
DEFAULT_END_CARDS = [{"kind": "scorecard", "seconds": DEFAULT_DURATIONS["scorecard"]}]
# Encoder settings for card mp4s — the ONLY encode in this module. Trims
# arrive already encoded (by overlay.render_video) at canvas spec; concat
# is byte-copy. Match overlay's settings so cards stream-copy with trims.
CRF = "12"
PRESET = "slow"


# --- Probe / canvas selection -----------------------------------------

@dataclass
class StreamInfo:
    path: Path
    width: int
    height: int
    fps_str: str
    fps: float
    pix_fmt: str
    color_space: str       # e.g. "bt709", "bt2020nc", or "" if untagged
    color_primaries: str   # e.g. "bt709", "bt2020"
    color_transfer: str    # e.g. "bt709", "arib-std-b67" (HLG), "smpte2084" (PQ)
    recorded_at: str
    seconds: float

    def is_hdr(self) -> bool:
        """True if the clip is HDR or wide-gamut (HLG / PQ / BT.2020 primaries / 10-bit).

        Phones (iPhones in particular) record HLG by default. We preserve HLG
        through to the output rather than tonemapping to SDR — Apple devices
        and modern displays render HLG correctly with full HDR brightness, and
        any tonemap-to-SDR step (libx264 has none built in; the FOSS options
        are all approximations) has produced visibly worse results than just
        passing the metadata through. See `pick_canvas` for the output choice.
        """
        ct = (self.color_transfer or "").lower()
        cp = (self.color_primaries or "").lower()
        cs = (self.color_space or "").lower()
        if ct in ("arib-std-b67", "smpte2084"):  # HLG, PQ
            return True
        if "2020" in cp or "2020" in cs:
            return True
        if "10" in self.pix_fmt:  # 10-bit usually pairs with HDR on phones
            return True
        return False


def _ffprobe_video_stream(path: Path) -> dict:
    """Return {width, height, r_frame_rate, pix_fmt, duration, color_*} for the video."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=width,height,r_frame_rate,pix_fmt,"
         "color_space,color_primaries,color_transfer:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    j = json.loads(out.stdout)
    s = j["streams"][0]
    s["duration"] = float(j["format"]["duration"])
    return s


def _fps_to_float(fps_str: str) -> float:
    if "/" in fps_str:
        n, d = fps_str.split("/")
        return float(n) / float(d) if float(d) else 0.0
    return float(fps_str)


def gather_approved(meta_dir: Path, trims_dir: Path) -> list[StreamInfo]:
    """Return approved clips with rendered trims, sorted by recorded_at."""
    rows: list[tuple[str, Path]] = []
    for sc in sorted(meta_dir.glob("*.json")):
        data = json.loads(sc.read_text())
        if data.get("review") != "approved":
            continue
        trim = trims_dir / f"{sc.stem}.mp4"
        if not trim.exists():
            continue
        rows.append((data.get("recorded_at") or "", trim))
    # Sort chronologically; clips missing recorded_at sink to the start
    # but at least keep stable filename order amongst themselves.
    rows.sort(key=lambda x: (x[0] == "", x[0]))
    out: list[StreamInfo] = []
    for rec, trim in rows:
        s = _ffprobe_video_stream(trim)
        out.append(StreamInfo(
            path=trim,
            width=int(s["width"]),
            height=int(s["height"]),
            fps_str=s["r_frame_rate"],
            fps=_fps_to_float(s["r_frame_rate"]),
            pix_fmt=s.get("pix_fmt", "yuv420p"),
            color_space=s.get("color_space", "") or "",
            color_primaries=s.get("color_primaries", "") or "",
            color_transfer=s.get("color_transfer", "") or "",
            recorded_at=rec,
            seconds=float(s["duration"]),
        ))
    return out


def pick_canvas(infos: list[StreamInfo]) -> dict:
    """Choose canvas dims/fps + color tags for the final timeline.

    Always landscape-oriented (YouTube-first): width = max longer side,
    height = max shorter side. Framerate is the highest seen.

    Color handling is pass-through: if any clip is HDR (HLG/BT.2020/10-bit)
    we output 10-bit `yuv420p10le` with `high10` profile and stamp the
    output as HLG (`bt2020 / arib-std-b67 / bt2020nc`). Otherwise BT.709
    SDR. We don't tonemap — the HLG metadata travels through to the file
    so HDR-aware players (QuickTime, Photos, iMovie, modern browsers,
    YouTube) display the round at full brightness, and naive players
    still decode the luma roughly correctly.
    """
    width = max(max(i.width, i.height) for i in infos)
    height = max(min(i.width, i.height) for i in infos)
    # Cap canvas fps at 60. Slo-mo (240fps) clips would otherwise drag the
    # whole round to 240fps and force a 4× upsample on every other clip.
    # `overlay.render_video` retimes higher-fps sources with setpts so they
    # land on the canvas as actual slow motion (à la iMovie).
    fps_pick = max(infos, key=lambda i: i.fps)
    if fps_pick.fps > 60:
        fps_str, fps_val = "60/1", 60.0
    else:
        fps_str, fps_val = fps_pick.fps_str, fps_pick.fps
    any_hdr = any(i.is_hdr() for i in infos)
    if any_hdr:
        color = {
            "pix_fmt": "yuv420p10le",
            "profile": "high10",
            "primaries": "bt2020",
            "transfer": "arib-std-b67",
            "matrix": "bt2020nc",
        }
    else:
        color = {
            "pix_fmt": "yuv420p",
            "profile": "high",
            "primaries": "bt709",
            "transfer": "bt709",
            "matrix": "bt709",
        }
    return {
        "width": width,
        "height": height,
        "fps_str": fps_str,
        "fps": fps_val,
        **color,
    }


def pick_canvas_for_round(raw_dir: Path) -> dict:
    """Compute the canvas spec from every video in `raw_dir`.

    This is the single source of truth for the round's output spec —
    `overlay.render_video` uses it to normalise every per-clip trim, and
    `render_final` uses the same spec for the title/scorecard cards so the
    concat demuxer can byte-copy everything together.
    """
    # Late import: batch_trim imports finalise, finalise imports batch_trim
    # — both via late imports so neither blocks at module-load time.
    from batch_trim import VIDEO_EXTS

    videos = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith(".")
    )
    if not videos:
        raise RuntimeError(f"no videos under {raw_dir}")
    infos: list[StreamInfo] = []
    for v in videos:
        s = _ffprobe_video_stream(v)
        infos.append(StreamInfo(
            path=v,
            width=int(s["width"]),
            height=int(s["height"]),
            fps_str=s["r_frame_rate"],
            fps=_fps_to_float(s["r_frame_rate"]),
            pix_fmt=s.get("pix_fmt", "yuv420p"),
            color_space=s.get("color_space", "") or "",
            color_primaries=s.get("color_primaries", "") or "",
            color_transfer=s.get("color_transfer", "") or "",
            recorded_at="",
            seconds=float(s["duration"]),
        ))
    return pick_canvas(infos)


# --- Card rendering ---------------------------------------------------

def _round_label(round_name: str) -> str:
    """'18-april-2026' -> '18 April 2026'. Falls through to the raw name."""
    m = re.match(r"^(\d{1,2})-([a-z]+)-(\d{4})$", round_name.lower())
    if not m:
        return round_name
    day, month, year = m.groups()
    return f"{int(day)} {month.capitalize()} {year}"


def _centered(draw: ImageDraw.ImageDraw, text: str, font, cx: int, cy: int,
              color) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((cx - w // 2, cy - h // 2 - bbox[1]), text, font=font, fill=color)


def _card_background(W: int, H: int) -> Image.Image:
    """Full-frame card background: vertical gradient + faint vignette + gold corner brackets.

    Matches the in-game overlay's tonal palette so the title/scorecard/summary
    cards read as the same broadcast template, just at full-screen scale.
    """
    # Reuse overlay's gradient endpoints but force full opacity (this is RGB).
    top = (BG_TOP[0], BG_TOP[1], BG_TOP[2], 255)
    bot = (BG_BOT[0], BG_BOT[1], BG_BOT[2], 255)
    grad = _vgradient(W, H, top, bot).convert("RGB")

    draw = ImageDraw.Draw(grad)
    s = H / 1080.0
    # Gold corner brackets — broadcast scoreboard cliché, but it works.
    bw = int(2 * s)
    arm = int(64 * s)
    inset = int(48 * s)
    pts = [
        # top-left
        ((inset, inset), (inset + arm, inset), (inset, inset), (inset, inset + arm)),
        # top-right
        ((W - inset - arm, inset), (W - inset, inset),
         (W - inset, inset), (W - inset, inset + arm)),
        # bottom-left
        ((inset, H - inset), (inset + arm, H - inset),
         (inset, H - inset - arm), (inset, H - inset)),
        # bottom-right
        ((W - inset - arm, H - inset), (W - inset, H - inset),
         (W - inset, H - inset - arm), (W - inset, H - inset)),
    ]
    for p1, p2, p3, p4 in pts:
        draw.line([p1, p2], fill=ACCENT[:3], width=bw)
        draw.line([p3, p4], fill=ACCENT[:3], width=bw)
    return grad


def render_title_card(
    canvas: dict,
    course: str | None,
    round_name: str,
    players: list[str],
) -> Image.Image:
    W, H = canvas["width"], canvas["height"]
    img = _card_background(W, H)
    draw = ImageDraw.Draw(img)

    s = H / 1080.0
    f_eyebrow = _font(int(28 * s), bold=True)
    f_course = _font(int(140 * s), bold=True)
    f_date = _font(int(42 * s), bold=False)
    f_players_label = _font(int(22 * s), bold=True)
    f_players = _font(int(48 * s), bold=True)

    course_text = (course or "Round").upper()
    date_text = _round_label(round_name).upper()
    players_text = "   ·   ".join(p.upper() for p in players) if players else ""

    # Eyebrow above the course name — gives the title some breathing room.
    _centered(draw, "ROUND HIGHLIGHTS", f_eyebrow, W // 2, int(H * 0.22),
              ACCENT[:3])

    # Course name — the hero. Slight letter-spacing emulated by uppercasing
    # and using a heavy weight; PIL doesn't expose tracking directly.
    _centered(draw, course_text, f_course, W // 2, int(H * 0.40), TEXT[:3])

    # Gold rule + date stamp.
    rule_w = int(W * 0.12)
    rule_h = max(2, int(3 * s))
    rule_y = int(H * 0.54)
    draw.rectangle(
        ((W - rule_w) // 2, rule_y, (W + rule_w) // 2, rule_y + rule_h),
        fill=ACCENT[:3],
    )
    _centered(draw, date_text, f_date, W // 2, int(H * 0.60), MUTED[:3])

    if players_text:
        _centered(draw, "PLAYERS", f_players_label, W // 2, int(H * 0.74),
                  ACCENT2[:3])
        _centered(draw, players_text, f_players, W // 2, int(H * 0.80),
                  TEXT[:3])
    return img


def render_scorecard_card(canvas: dict, scores: dict | None) -> Image.Image:
    """Full 18-hole scorecard. Falls back to a placeholder if no scores."""
    W, H = canvas["width"], canvas["height"]
    img = _card_background(W, H)
    draw = ImageDraw.Draw(img)
    s = H / 1080.0

    if not scores:
        f = _font(int(56 * s), bold=True)
        _centered(draw, "Scorecard unavailable", f, W // 2, H // 2, MUTED[:3])
        return img

    holes = scores["holes"]
    players = scores["players"]

    # Empty rounds (no holes logged yet, or no players) fall back to the
    # same placeholder rather than dividing by zero in the cell-width math.
    if not holes or not players:
        f = _font(int(56 * s), bold=True)
        _centered(draw, "Scorecard unavailable", f, W // 2, H // 2, MUTED[:3])
        return img

    # Header — eyebrow + title + accent rule, mirrors the title card.
    f_eyebrow = _font(int(22 * s), bold=True)
    f_title = _font(int(72 * s), bold=True)
    _centered(draw, "SCORECARD", f_eyebrow, W // 2, int(H * 0.07), ACCENT[:3])
    title = (scores.get("course") or "Scorecard").upper()
    _centered(draw, title, f_title, W // 2, int(H * 0.13), TEXT[:3])
    rule_w = int(W * 0.08); rule_h = max(2, int(3 * s))
    rule_y = int(H * 0.19)
    draw.rectangle(
        ((W - rule_w) // 2, rule_y, (W + rule_w) // 2, rule_y + rule_h),
        fill=ACCENT[:3],
    )

    # Table layout
    side_margin = int(W * 0.05)
    grid_w = W - 2 * side_margin
    label_col_w = int(grid_w * 0.17)
    total_col_w = int(grid_w * 0.08)
    hole_cols_w = grid_w - label_col_w - total_col_w
    cell_w = hole_cols_w // len(holes)
    table_left = side_margin + (hole_cols_w - cell_w * len(holes)) // 2

    table_top = int(H * 0.26)
    table_bot = int(H * 0.93)
    rows = 2 + len(players)
    row_h = (table_bot - table_top) // rows

    f_label = _font(int(30 * s), bold=True)
    f_head  = _font(int(28 * s), bold=True)
    f_par   = _font(int(32 * s), bold=True)
    f_val   = _font(int(34 * s), bold=True)
    f_total = _font(int(38 * s), bold=True)

    def cell(text: str, font, x: int, y: int, w: int, h: int, color) -> None:
        b = draw.textbbox((0, 0), text, font=font)
        tw = b[2] - b[0]; th = b[3] - b[1]
        draw.text((x + (w - tw) // 2, y + (h - th) // 2 - b[1]),
                  text, font=font, fill=color)

    total_x = table_left + label_col_w + cell_w * len(holes)
    table_right = total_x + total_col_w

    # Tints: card body is RGB, so we pre-blend against the average BG colour
    # rather than passing RGBA fills (which PIL would silently treat as opaque).
    bg_mid = tuple((BG_TOP[i] + BG_BOT[i]) // 2 for i in range(3))

    def tint(rgb: tuple, a: int) -> tuple:
        # Alpha 0..255, blended over bg_mid.
        return tuple(round(bg_mid[i] + (rgb[i] - bg_mid[i]) * a / 255)
                     for i in range(3))

    owner_band = tint(ACCENT[:3], 56)     # subtle gold wash
    par_band   = tint(ACCENT2[:3], 36)    # whisper of course-green
    stripe     = tint((255, 255, 255), 8)  # near-invisible zebra

    # Striped row backgrounds for player rows so the eye can track across 18 cols.
    for i in range(len(players)):
        y = table_top + (2 + i) * row_h
        is_owner = players[i].get("is_owner")
        if is_owner:
            draw.rectangle((table_left, y, table_right, y + row_h), fill=owner_band)
            # gold left edge bar
            draw.rectangle((table_left - int(8 * s), y,
                            table_left - int(2 * s), y + row_h),
                           fill=ACCENT[:3])
        elif i % 2:
            draw.rectangle((table_left, y, table_right, y + row_h), fill=stripe)

    # HOLE row
    y = table_top
    cell("HOLE", f_label, table_left, y, label_col_w, row_h, MUTED[:3])
    for i, hole in enumerate(holes):
        cell(str(hole["number"]), f_head,
             table_left + label_col_w + i * cell_w, y, cell_w, row_h,
             MUTED[:3])
    cell("TOT", f_label, total_x, y, total_col_w, row_h, MUTED[:3])

    # PAR row — gets a dim band so it visually separates from the player rows.
    y += row_h
    draw.rectangle((table_left, y, table_right, y + row_h), fill=par_band)
    cell("PAR", f_label, table_left, y, label_col_w, row_h, ACCENT2[:3])
    par_total = 0
    for i, hole in enumerate(holes):
        cell(str(hole["par"]), f_par,
             table_left + label_col_w + i * cell_w, y, cell_w, row_h,
             ACCENT2[:3])
        par_total += hole["par"]
    cell(str(par_total), f_par, total_x, y, total_col_w, row_h, ACCENT2[:3])

    # Divider under PAR
    line_y = y + row_h
    draw.line((table_left, line_y, table_right, line_y),
              fill=ACCENT[:3], width=max(1, int(2 * s)))

    # Score-type palette tuned for high-handicap golfers: par is celebrated
    # like a birdie, bogey is treated as a "good" hole, doubles are the
    # baseline, triples warn, quad+ are flagged. Five distinct treatments
    # so the eye can pick a hole's quality from across the room.
    BIRDIE_FG  = (10, 22, 38)            # near-black text on filled blue pill
    BIRDIE_BG  = ACCENT[:3]              # filled blue (only score type with a chip)
    PAR_FG     = ACCENT[:3]              # blue text, no chip
    BOGEY_FG   = ACCENT2[:3]             # course-green
    DOUBLE_FG  = (240, 180, 96)          # amber
    TRIPLE_FG  = OVER_PAR[:3]            # soft red — covers triple AND worse

    # Player rows
    for p in players:
        y += row_h
        name_color = OWNER[:3] if p.get("is_owner") else TEXT[:3]
        cell(p["name"], f_label, table_left, y, label_col_w, row_h, name_color)
        strokes_by_hole = {sc["hole"]: sc.get("strokes") for sc in p["scores"]}
        total = 0
        for i, hole in enumerate(holes):
            v = strokes_by_hole.get(hole["number"])
            txt = "—" if v is None else str(v)
            color = name_color
            if v is not None:
                diff = v - hole["par"]
                cx = table_left + label_col_w + i * cell_w
                # Pill geometry — used for birdie (filled) and par (outline).
                # Centred chip ~80% of cell width, square so it reads as a coin
                # instead of a vertical ellipse.
                chip_d = min(cell_w, row_h) - int(20 * s)
                cx_mid = cx + cell_w // 2
                cy_mid = y + row_h // 2
                pill_box = (cx_mid - chip_d // 2, cy_mid - chip_d // 2,
                            cx_mid + chip_d // 2, cy_mid + chip_d // 2)
                if diff < 0:
                    draw.ellipse(pill_box, fill=BIRDIE_BG)
                    color = BIRDIE_FG
                elif diff == 0:
                    color = PAR_FG
                elif diff == 1:
                    color = BOGEY_FG
                elif diff == 2:
                    color = DOUBLE_FG
                else:
                    color = TRIPLE_FG
                total += v
            cell(txt, f_val,
                 table_left + label_col_w + i * cell_w, y, cell_w, row_h, color)
        cell(str(total), f_total, total_x, y, total_col_w, row_h, name_color)

    # Legend across the bottom — explains the colour bucketing without
    # cluttering the table itself.
    legend_y = table_bot + int(20 * s)
    if legend_y + int(50 * s) < H:
        f_legend = _font(int(20 * s), bold=True)
        items = [
            ("BIRDIE",  BIRDIE_FG, BIRDIE_BG),    # filled chip — only one
            ("PAR",     PAR_FG, None),
            ("BOGEY",   BOGEY_FG, None),
            ("DOUBLE",  DOUBLE_FG, None),
            ("TRIPLE+", TRIPLE_FG, None),
        ]
        gap = int(36 * s)
        chip_d = int(28 * s)
        # Pre-measure widths to centre the legend strip.
        widths = []
        for label, _, _ in items:
            tw = draw.textlength(label, font=f_legend)
            widths.append(chip_d + int(8 * s) + int(tw))
        total_w = sum(widths) + gap * (len(items) - 1)
        x = (W - total_w) // 2
        for (label, fg, bg), w in zip(items, widths):
            chip_y = legend_y + int(8 * s)
            chip_box = (x, chip_y, x + chip_d, chip_y + chip_d)
            if bg is not None:
                draw.ellipse(chip_box, fill=bg)
            else:
                draw.ellipse(chip_box, outline=fg, width=max(2, int(2 * s)))
            draw.text((x + chip_d + int(8 * s), legend_y + int(10 * s)),
                      label, font=f_legend, fill=MUTED[:3])
            x += w + gap

    return img


def _player_breakdown(player: dict, holes: list[dict]) -> dict:
    """Per-player aggregates over an 18-hole round.

    Returns: strokes, par, diff, putts, birdies, pars, bogeys, doubles_plus.
    Holes the player didn't score are skipped (no penalty).
    """
    pars = {h["number"]: h["par"] for h in holes}
    strokes = putts = par_total = 0
    birdies = par_count = bogeys = doubles_plus = 0
    for sc in player["scores"]:
        s = sc.get("strokes")
        if s is None:
            continue
        p = pars.get(sc["hole"])
        if p is None:
            continue
        strokes += s
        putts += sc.get("putts") or 0
        par_total += p
        diff = s - p
        if diff < 0:
            birdies += 1
        elif diff == 0:
            par_count += 1
        elif diff == 1:
            bogeys += 1
        else:
            doubles_plus += 1
    return {
        "strokes": strokes,
        "par": par_total,
        "diff": strokes - par_total,
        "putts": putts,
        "birdies": birdies,
        "pars": par_count,
        "bogeys": bogeys,
        "doubles_plus": doubles_plus,
    }


def render_summary_card(canvas: dict, scores: dict | None) -> Image.Image:
    """Per-player round summary: total / vs par / putts / score-type counts."""
    W, H = canvas["width"], canvas["height"]
    img = _card_background(W, H)
    draw = ImageDraw.Draw(img)
    s = H / 1080.0

    if not scores:
        f = _font(int(56 * s), bold=True)
        _centered(draw, "Summary unavailable", f, W // 2, H // 2, MUTED[:3])
        return img

    holes = scores["holes"]
    players = scores["players"]
    if not players:
        f = _font(int(56 * s), bold=True)
        _centered(draw, "Summary unavailable", f, W // 2, H // 2, MUTED[:3])
        return img
    breakdowns = [(_player_breakdown(p, holes), p) for p in players]

    # Header
    f_eyebrow = _font(int(22 * s), bold=True)
    f_title = _font(int(76 * s), bold=True)
    _centered(draw, "FINAL", f_eyebrow, W // 2, int(H * 0.07), ACCENT[:3])
    _centered(draw, "ROUND SUMMARY", f_title, W // 2, int(H * 0.13), TEXT[:3])
    rule_w = int(W * 0.10); rule_h = max(2, int(3 * s))
    rule_y = int(H * 0.19)
    draw.rectangle(
        ((W - rule_w) // 2, rule_y, (W + rule_w) // 2, rule_y + rule_h),
        fill=ACCENT[:3],
    )

    # Card-per-player layout — each player gets a panel with the same gradient
    # body as the in-game card, gold edge bar on owner, big score, stats grid.
    n = len(players)
    side_margin = int(W * 0.06)
    gap = int(W * 0.02)
    col_w = (W - 2 * side_margin - gap * (n - 1)) // n
    col_top = int(H * 0.27)
    col_bot = int(H * 0.93)
    col_h = col_bot - col_top
    radius = int(20 * s)

    # Type sized off panel height so the layout always fits, not screen height.
    sp = col_h / 1080.0  # panel-relative scale
    f_name = _font(int(60 * sp), bold=True)
    f_owner_tag = _font(int(24 * sp), bold=True)
    f_score = _font(int(360 * sp), bold=True)
    f_diff = _font(int(72 * sp), bold=True)
    f_par = _font(int(32 * sp), bold=False)
    f_label = _font(int(24 * sp), bold=True)
    f_stat = _font(int(56 * sp), bold=True)

    # Vertical anchor points as fractions of panel height — guarantees fit.
    name_y_f   = 0.10
    score_y_f  = 0.36
    diff_y_f   = 0.66
    par_y_f    = 0.74
    stats_y_f  = 0.86

    for i, (b, p) in enumerate(breakdowns):
        col_x = side_margin + i * (col_w + gap)
        is_owner = p.get("is_owner")
        name_color = OWNER[:3] if is_owner else TEXT[:3]

        # Panel background: same gradient as the overlay card body.
        panel = _vgradient(col_w, col_h,
                           (BG_TOP[0], BG_TOP[1], BG_TOP[2], 255),
                           (BG_BOT[0], BG_BOT[1], BG_BOT[2], 255))
        panel.putalpha(_rounded_mask(col_w, col_h, radius))
        img.paste(panel, (col_x, col_top), panel)
        draw.rounded_rectangle(
            (col_x, col_top, col_x + col_w - 1, col_bot - 1),
            radius=radius, outline=BORDER[:3], width=1,
        )
        if is_owner:
            # Gold left-edge bar.
            bar_w = int(6 * s)
            draw.rounded_rectangle(
                (col_x + int(10 * s), col_top + int(28 * s),
                 col_x + int(10 * s) + bar_w, col_bot - int(28 * s)),
                radius=int(3 * s), fill=ACCENT[:3],
            )
            # YOU pill, top-right.
            tag = "YOU"
            tw = draw.textlength(tag, font=f_owner_tag)
            pill_pad_x = int(14 * s); pill_pad_y = int(8 * s)
            pill_h = int(40 * sp)
            pill_x1 = col_x + col_w - int(24 * s)
            pill_x0 = pill_x1 - int(tw) - 2 * pill_pad_x
            pill_y0 = col_top + int(24 * s)
            pill_y1 = pill_y0 + pill_h
            draw.rounded_rectangle(
                (pill_x0, pill_y0, pill_x1, pill_y1),
                radius=pill_h // 2, fill=ACCENT[:3],
            )
            tag_h = draw.textbbox((0, 0), tag, font=f_owner_tag)[3]
            draw.text(
                (pill_x0 + pill_pad_x, pill_y0 + (pill_h - tag_h) // 2 - 2),
                tag, font=f_owner_tag, fill=BG_BOT[:3],
            )

        cx = col_x + col_w // 2

        _centered(draw, p["name"].upper(), f_name, cx,
                  col_top + int(col_h * name_y_f), name_color)

        # Big score number — anchor of the panel.
        _centered(draw, str(b["strokes"]), f_score, cx,
                  col_top + int(col_h * score_y_f), ACCENT[:3])

        # Diff vs par — soft-red over par, gold under, muted "E".
        diff = b["diff"]
        if diff < 0:
            diff_str, diff_color = str(diff), UNDER_PAR[:3]
        elif diff > 0:
            diff_str, diff_color = f"+{diff}", OVER_PAR[:3]
        else:
            diff_str, diff_color = "E", MUTED[:3]
        _centered(draw, diff_str, f_diff, cx,
                  col_top + int(col_h * diff_y_f), diff_color)
        _centered(draw, f"par {b['par']}", f_par, cx,
                  col_top + int(col_h * par_y_f), MUTED[:3])

        # Stats row — five slots in one row across the bottom of the panel.
        stat_pairs = [
            ("BIRDIES", b["birdies"]),
            ("PARS",    b["pars"]),
            ("BOGEYS",  b["bogeys"]),
            ("DBL+",    b["doubles_plus"]),
            ("PUTTS",   b["putts"]),
        ]
        n_stats = len(stat_pairs)
        slot_w = col_w // n_stats
        stats_y = col_top + int(col_h * stats_y_f)
        for j, (lbl, val) in enumerate(stat_pairs):
            sx = col_x + slot_w * j + slot_w // 2
            _centered(draw, str(val), f_stat, sx, stats_y, TEXT[:3])
            _centered(draw, lbl, f_label, sx,
                      stats_y + int(col_h * 0.05), MUTED[:3])

    return img


def render_card_image(kind: str, canvas: dict, scores: dict | None,
                      round_name: str) -> Image.Image:
    """Dispatch a card kind to its renderer."""
    if kind == "title":
        course = scores.get("course") if scores else None
        names = [p["name"] for p in scores["players"]] if scores else []
        return render_title_card(canvas, course, round_name, names)
    if kind == "scorecard":
        return render_scorecard_card(canvas, scores)
    if kind == "summary":
        return render_summary_card(canvas, scores)
    raise ValueError(f"unknown card kind: {kind!r}")


AVAILABLE_CARDS = ("title", "scorecard", "summary")


# --- Encoding ---------------------------------------------------------

def _encode_card_segment(
    png_path: Path,
    seconds: float,
    canvas: dict,
    out_path: Path,
) -> None:
    """Encode one card PNG → canvas-spec mp4 segment with silent audio.

    The output matches every per-clip trim's codec/resolution/fps/colour
    tags so the concat demuxer can stream-copy them all together.
    """
    W = canvas["width"]
    H = canvas["height"]
    fps_str = canvas["fps_str"]
    pix_fmt = canvas["pix_fmt"]
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(png_path),
        "-f", "lavfi", "-t", f"{seconds:.3f}",
        "-i", "anullsrc=cl=stereo:r=48000",
        "-filter_complex",
        f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={fps_str},setsar=1,format={pix_fmt}[v];"
        f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", CRF, "-preset", PRESET,
        "-profile:v", canvas["profile"],
        "-color_primaries", canvas["primaries"],
        "-color_trc", canvas["transfer"],
        "-colorspace", canvas["matrix"],
        "-color_range", "tv",
        "-x264-params",
        f"colorprim={canvas['primaries']}:transfer={canvas['transfer']}"
        f":colormatrix={canvas['matrix']}",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _normalise_cards(cards: list[dict] | None, fallback: list[dict]) -> list[dict]:
    """Validate + clamp durations on a card list. Returns a fresh list."""
    if cards is None:
        cards = fallback
    out: list[dict] = []
    for c in cards:
        kind = c["kind"]
        if kind not in AVAILABLE_CARDS:
            raise ValueError(f"unknown card kind: {kind!r}")
        seconds = float(c.get("seconds") or DEFAULT_DURATIONS[kind])
        seconds = max(0.5, min(seconds, 60.0))
        out.append({"kind": kind, "seconds": seconds})
    return out


def render_final(
    round_dir: Path,
    start_cards: list[dict] | None = None,
    end_cards: list[dict] | None = None,
    progress=None,
    on_start=None,
    log=print,
    max_clips: int | None = None,
    out_name: str = "final.mp4",
) -> Path:
    """Build <round_dir>/<out_name> from approved trims, with custom cards.

    Two-phase pipeline:
      1. Encode each title/scorecard card into a short canvas-spec mp4.
      2. Run the ffmpeg concat demuxer with `-c copy` over the cards and
         all approved trims to byte-append them into final.mp4.

    The trims must already be encoded to the round's canvas spec — that's
    what `overlay.render_video` does at trim time. If a trim doesn't match
    canvas, `-c copy` will refuse and ffmpeg will error; the fix is to
    re-render the offending clip via the UI's "Re-render all clips" button.

    `progress`: optional callable invoked with progress dicts. The schema
                matches the old filter_complex pipeline so the UI keeps
                working unchanged: {elapsed_s, total_s, percent, eta_s,
                speed, fps, frame, segment_index, segment_total,
                segment_label, segment_kind, cpu_percent, rss_mb}.
    `on_start`: optional callable invoked once with {canvas, segments_total,
                total_s, pid} just after the concat ffmpeg launches.
    `max_clips`: optional cap on the number of clips, useful for testing.
    `out_name`: filename written under round_dir.
    """
    # Late import to avoid circular module dependency with batch_trim.
    from batch_trim import round_paths

    start_cards = _normalise_cards(start_cards, DEFAULT_START_CARDS)
    end_cards = _normalise_cards(end_cards, DEFAULT_END_CARDS)

    raw_dir, trims_dir, meta_dir = round_paths(round_dir)
    canvas = pick_canvas_for_round(raw_dir)
    infos = gather_approved(meta_dir, trims_dir)
    if not infos:
        raise RuntimeError("no approved clips with rendered trims")
    if max_clips is not None and max_clips > 0:
        infos = infos[:max_clips]
        log(f"[final] limiting to first {len(infos)} clips (max_clips={max_clips})")
    n_hdr = sum(1 for i in infos if i.is_hdr())
    log(f"[final] {len(infos)} approved clips ({n_hdr} HDR)  "
        f"canvas: {canvas['width']}x{canvas['height']} @ {canvas['fps_str']} "
        f"({canvas['pix_fmt']}, {canvas['primaries']}/{canvas['transfer']}/"
        f"{canvas['matrix']})")
    log(f"[final] start cards: {[c['kind'] for c in start_cards]}  "
        f"end cards: {[c['kind'] for c in end_cards]}")

    # Warn if any trim's params drifted from canvas (would break -c copy).
    mismatches: list[str] = []
    for inf in infos:
        if (inf.width, inf.height) != (canvas["width"], canvas["height"]):
            mismatches.append(
                f"{inf.path.name}: {inf.width}x{inf.height} (canvas "
                f"{canvas['width']}x{canvas['height']})")
        elif inf.pix_fmt != canvas["pix_fmt"]:
            mismatches.append(
                f"{inf.path.name}: pix_fmt={inf.pix_fmt} (canvas "
                f"{canvas['pix_fmt']})")
    if mismatches:
        log("[final] WARNING: trims don't match canvas — re-render them:")
        for m in mismatches[:5]:
            log(f"        {m}")
        if len(mismatches) > 5:
            log(f"        … and {len(mismatches) - 5} more")

    # Build the segment timeline (cards then clips then cards). Used both
    # for progress mapping and for assembling the concat list.
    segments: list[dict] = []
    t = 0.0
    for c in start_cards:
        segments.append({"kind": f"card:{c['kind']}", "label": c["kind"].title(),
                         "start": t, "end": t + c["seconds"], "index": len(segments),
                         "seconds": c["seconds"], "card_kind": c["kind"]})
        t += c["seconds"]
    for inf in infos:
        segments.append({"kind": "clip", "label": inf.path.stem,
                         "start": t, "end": t + inf.seconds, "index": len(segments),
                         "trim_path": inf.path, "seconds": inf.seconds})
        t += inf.seconds
    for c in end_cards:
        segments.append({"kind": f"card:{c['kind']}", "label": c["kind"].title(),
                         "start": t, "end": t + c["seconds"], "index": len(segments),
                         "seconds": c["seconds"], "card_kind": c["kind"]})
        t += c["seconds"]
    total_s = t

    scores_path = round_dir / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else None

    out_path = round_dir / out_name
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # --- Phase 1: encode each card to an mp4 segment that matches canvas.
        # The trims are already on disk at canvas spec; only the cards need
        # to be encoded fresh each finalise (the score/title content can
        # change between runs).
        n_cards = sum(1 for s in segments if s["kind"].startswith("card:"))
        log(f"[final] encoding {n_cards} card segment(s)…")

        if on_start is not None:
            on_start({
                "canvas": canvas,
                "segments_total": len(segments),
                "total_s": total_s,
                "pid": None,
            })

        cards_done = 0
        for seg in segments:
            if not seg["kind"].startswith("card:"):
                continue
            kind = seg["card_kind"]
            img = render_card_image(kind, canvas, scores, round_dir.name)
            png = td_path / f"card_{seg['index']:03d}_{kind}.png"
            img.save(png)
            seg_mp4 = td_path / f"card_{seg['index']:03d}_{kind}.mp4"
            _encode_card_segment(png, seg["seconds"], canvas, seg_mp4)
            seg["trim_path"] = seg_mp4
            cards_done += 1
            if progress is not None:
                # Cards are typically 3–7s each; report rough progress so the
                # UI doesn't sit silent during this phase.
                progress({
                    "elapsed_s": seg["end"],
                    "total_s": total_s,
                    "percent": min(0.95, seg["end"] / total_s),
                    "eta_s": None,
                    "speed": 0.0,
                    "fps": 0.0,
                    "frame": 0,
                    "segment_index": seg["index"] + 1,
                    "segment_total": len(segments),
                    "segment_label": seg["label"],
                    "segment_kind": seg["kind"],
                    "cpu_percent": None,
                    "rss_mb": None,
                })

        # --- Phase 2: concat-copy. Build a list file for the demuxer.
        list_path = td_path / "concat_list.txt"
        with list_path.open("w") as fh:
            for seg in segments:
                # `file '...'` — single-quote, escape any embedded quotes per
                # ffmpeg concat demuxer rules. Absolute paths are essential:
                # the demuxer resolves relative entries against the list
                # file's directory (a temp dir here), not the process cwd, so
                # relative trim paths silently fail to open and ffmpeg exits
                # rc=0 with a truncated output (only the title card lands).
                p = str(Path(seg["trim_path"]).resolve()).replace("'", "'\\''")
                fh.write(f"file '{p}'\n")
        log("[final] concat-copy…")

        cmd = [
            "ffmpeg", "-y", "-v", "error", "-nostats",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        # Update on_start with the real pid (cancel button uses this).
        if on_start is not None:
            on_start({
                "canvas": canvas,
                "segments_total": len(segments),
                "total_s": total_s,
                "pid": proc.pid,
            })

        stderr_lines: list[str] = []
        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line)
        threading.Thread(target=_drain_stderr, daemon=True).start()

        rc = proc.wait()
        if rc != 0:
            tail = "".join(stderr_lines)[-2000:]
            raise subprocess.CalledProcessError(rc, cmd, tail)

        # The concat demuxer logs "Impossible to open ..." for missing
        # segments but still exits rc=0 with a truncated output. Cross-check
        # the actual output duration against the planned timeline so any
        # silent skip surfaces as a hard error (not a 3-second `final.mp4`).
        actual_s = float(_ffprobe_video_stream(out_path)["duration"])
        if actual_s + 0.5 < total_s:
            tail = "".join(stderr_lines)[-2000:]
            raise RuntimeError(
                f"concat output is {actual_s:.2f}s but timeline is "
                f"{total_s:.2f}s — ffmpeg silently skipped segments. "
                f"stderr tail:\n{tail}"
            )

        # Snap to 100% on the final frame.
        if progress is not None:
            progress({
                "elapsed_s": total_s,
                "total_s": total_s,
                "percent": 1.0,
                "eta_s": 0.0,
                "speed": 0.0,
                "fps": 0.0,
                "frame": 0,
                "segment_index": len(segments),
                "segment_total": len(segments),
                "segment_label": segments[-1]["label"],
                "segment_kind": segments[-1]["kind"],
                "cpu_percent": None,
                "rss_mb": None,
            })

    log(f"[final] wrote {out_path}")
    return out_path


def main() -> int:
    import argparse
    from batch_trim import newest_round

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=Path,
                    help="round folder (default: newest under --clips-root)")
    ap.add_argument("--clips-root", type=Path, default=Path("clips"))
    ap.add_argument("--start", action="append", metavar="KIND[:SEC]",
                    help=f"start card (repeatable). KIND ∈ {AVAILABLE_CARDS}")
    ap.add_argument("--end", action="append", metavar="KIND[:SEC]",
                    help=f"end card (repeatable). KIND ∈ {AVAILABLE_CARDS}")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="cap clip count for fast iteration (default: all)")
    ap.add_argument("--out-name", default="final.mp4",
                    help="output filename inside round folder")
    args = ap.parse_args()

    def parse(spec: str) -> dict:
        if ":" in spec:
            k, s = spec.split(":", 1)
            return {"kind": k, "seconds": float(s)}
        return {"kind": spec, "seconds": DEFAULT_DURATIONS[spec]}

    start = [parse(s) for s in args.start] if args.start else None
    end = [parse(s) for s in args.end] if args.end else None

    round_dir = args.round or newest_round(args.clips_root)
    render_final(round_dir, start_cards=start, end_cards=end,
                 max_clips=args.max_clips, out_name=args.out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
