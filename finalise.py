# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Concatenate approved trims into a single export-ready video.

Output: clips/<round>/final.mp4 — a title card, every approved trim in
chronological order, then a scorecard card.

The canvas is sized to the highest-quality clip in the round (max longer
side x max shorter side, oriented landscape, max framerate, 10-bit if any
clip is 10-bit) so 4K source material passes through untouched. Smaller
clips are scaled up with Lanczos and padded to fit. Output is H.264 with
CRF 12 — meaningfully higher quality than the per-clip trims (CRF 18) so
the second encode pass doesn't become the quality floor of the round
(double-encoding at CRF 16 produced visible graininess).

Cards are rendered with Pillow at canvas resolution, then turned into
short video segments with silent audio inside the same ffmpeg call so
everything is concatenated in one pass.
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

from overlay import ACCENT, BORDER, MUTED, OWNER, TEXT, _font

# Default per-card durations. The HTTP API can override per-render.
DEFAULT_DURATIONS = {
    "title": 3.0,
    "scorecard": 7.0,
    "summary": 6.0,
}
DEFAULT_START_CARDS = [{"kind": "title", "seconds": DEFAULT_DURATIONS["title"]}]
DEFAULT_END_CARDS = [{"kind": "scorecard", "seconds": DEFAULT_DURATIONS["scorecard"]}]
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
    fps_pick = max(infos, key=lambda i: i.fps)
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
        "fps_str": fps_pick.fps_str,
        "fps": fps_pick.fps,
        **color,
    }


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


def render_title_card(
    canvas: dict,
    course: str | None,
    round_name: str,
    players: list[str],
) -> Image.Image:
    W, H = canvas["width"], canvas["height"]
    img = Image.new("RGB", (W, H), (14, 17, 22))
    draw = ImageDraw.Draw(img)

    # Sizes scale with canvas height so 1080p and 4K both look balanced.
    s = H / 1080.0
    f_course = _font(int(96 * s), bold=True)
    f_date = _font(int(40 * s), bold=False)
    f_players = _font(int(46 * s), bold=False)

    course_text = (course or "Round").upper()
    date_text = _round_label(round_name)
    players_text = "  ·  ".join(players) if players else ""

    _centered(draw, course_text, f_course, W // 2, int(H * 0.32), TEXT[:3])

    rule_w = int(W * 0.14)
    rule_h = max(2, int(4 * s))
    rule_y = int(H * 0.44)
    draw.rectangle(
        ((W - rule_w) // 2, rule_y, (W + rule_w) // 2, rule_y + rule_h),
        fill=ACCENT[:3],
    )

    _centered(draw, date_text, f_date, W // 2, int(H * 0.52), MUTED[:3])
    if players_text:
        _centered(draw, players_text, f_players, W // 2, int(H * 0.66),
                  TEXT[:3])
    return img


def render_scorecard_card(canvas: dict, scores: dict | None) -> Image.Image:
    """Full 18-hole scorecard. Falls back to a placeholder if no scores."""
    W, H = canvas["width"], canvas["height"]
    img = Image.new("RGB", (W, H), (14, 17, 22))
    draw = ImageDraw.Draw(img)
    s = H / 1080.0

    if not scores:
        f = _font(int(56 * s), bold=True)
        _centered(draw, "Scorecard unavailable", f, W // 2, H // 2, MUTED[:3])
        return img

    holes = scores["holes"]
    players = scores["players"]

    # Header
    f_title = _font(int(64 * s), bold=True)
    title = (scores.get("course") or "Scorecard").upper()
    _centered(draw, title, f_title, W // 2, int(H * 0.10), TEXT[:3])

    # Table layout: label col + 18 hole cols + total col, centered horizontally.
    side_margin = int(W * 0.04)
    grid_w = W - 2 * side_margin
    label_col_w = int(grid_w * 0.16)
    total_col_w = int(grid_w * 0.07)
    hole_cols_w = grid_w - label_col_w - total_col_w
    cell_w = hole_cols_w // len(holes)
    table_left = side_margin + (hole_cols_w - cell_w * len(holes)) // 2

    table_top = int(H * 0.22)
    table_bot = int(H * 0.92)
    rows = 2 + len(players)  # HOLE row, PAR row, then one row per player
    row_h = (table_bot - table_top) // rows

    f_label = _font(int(32 * s), bold=True)
    f_head = _font(int(30 * s), bold=True)
    f_par = _font(int(32 * s), bold=True)
    f_val = _font(int(34 * s), bold=True)

    def cell(text: str, font, x: int, y: int, w: int, h: int, color) -> None:
        b = draw.textbbox((0, 0), text, font=font)
        tw, th = b[2] - b[0], b[3] - b[1]
        draw.text((x + (w - tw) // 2, y + (h - th) // 2 - b[1]),
                  text, font=font, fill=color)

    total_x = table_left + label_col_w + cell_w * len(holes)

    # HOLE row
    y = table_top
    cell("HOLE", f_label, table_left, y, label_col_w, row_h, MUTED[:3])
    for i, hole in enumerate(holes):
        cell(str(hole["number"]), f_head,
             table_left + label_col_w + i * cell_w, y, cell_w, row_h,
             MUTED[:3])
    cell("TOT", f_label, total_x, y, total_col_w, row_h, MUTED[:3])

    # PAR row
    y += row_h
    cell("PAR", f_label, table_left, y, label_col_w, row_h, MUTED[:3])
    par_total = 0
    for i, hole in enumerate(holes):
        cell(str(hole["par"]), f_par,
             table_left + label_col_w + i * cell_w, y, cell_w, row_h,
             ACCENT[:3])
        par_total += hole["par"]
    cell(str(par_total), f_par, total_x, y, total_col_w, row_h, ACCENT[:3])

    # Divider under PAR
    line_y = y + row_h
    draw.line((table_left, line_y,
               total_x + total_col_w, line_y),
              fill=BORDER[:3], width=max(1, int(2 * s)))

    # Player rows
    for p in players:
        y += row_h
        name_color = OWNER[:3] if p.get("is_owner") else TEXT[:3]
        cell(p["name"], f_label, table_left, y, label_col_w, row_h,
             name_color)
        strokes_by_hole = {sc["hole"]: sc.get("strokes") for sc in p["scores"]}
        total = 0
        for i, hole in enumerate(holes):
            v = strokes_by_hole.get(hole["number"])
            txt = "—" if v is None else str(v)
            color = name_color
            if v is not None:
                diff = v - hole["par"]
                if diff < 0:
                    color = ACCENT[:3]      # under par
                elif diff > 1:
                    color = MUTED[:3]       # double-bogey or worse
                total += v
            cell(txt, f_val,
                 table_left + label_col_w + i * cell_w, y, cell_w, row_h,
                 color)
        cell(str(total), f_val, total_x, y, total_col_w, row_h, name_color)

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
    img = Image.new("RGB", (W, H), (14, 17, 22))
    draw = ImageDraw.Draw(img)
    s = H / 1080.0

    if not scores:
        f = _font(int(56 * s), bold=True)
        _centered(draw, "Summary unavailable", f, W // 2, H // 2, MUTED[:3])
        return img

    holes = scores["holes"]
    players = scores["players"]
    breakdowns = [(_player_breakdown(p, holes), p) for p in players]

    f_title = _font(int(56 * s), bold=True)
    _centered(draw, "ROUND SUMMARY", f_title, W // 2, int(H * 0.10), TEXT[:3])

    # Accent rule under the heading
    rule_w = int(W * 0.10); rule_h = max(2, int(3 * s))
    rule_y = int(H * 0.16)
    draw.rectangle(
        ((W - rule_w) // 2, rule_y, (W + rule_w) // 2, rule_y + rule_h),
        fill=ACCENT[:3],
    )

    # Layout: a column per player, side-by-side, vertically centered.
    n = len(players)
    side_margin = int(W * 0.06)
    col_w = (W - 2 * side_margin) // n
    col_top = int(H * 0.26)

    f_name = _font(int(48 * s), bold=True)
    f_score = _font(int(180 * s), bold=True)
    f_diff = _font(int(40 * s), bold=False)
    f_label = _font(int(22 * s), bold=False)
    f_stat = _font(int(36 * s), bold=True)

    for i, (b, p) in enumerate(breakdowns):
        cx = side_margin + col_w * i + col_w // 2
        name_color = OWNER[:3] if p.get("is_owner") else TEXT[:3]
        y = col_top
        _centered(draw, p["name"].upper(), f_name, cx, y, name_color)

        # Big score number
        y = int(H * 0.42)
        _centered(draw, str(b["strokes"]), f_score, cx, y, name_color)

        # Diff vs par
        diff = b["diff"]
        diff_str = f"+{diff}" if diff > 0 else (str(diff) if diff < 0 else "E")
        diff_color = ACCENT[:3] if diff <= 0 else MUTED[:3]
        y = int(H * 0.60)
        _centered(draw, f"{diff_str}  ·  par {b['par']}", f_diff, cx, y, diff_color)

        # Stat row labels + values
        labels = ["BIRDIES", "PARS", "BOGEYS", "DBL+", "PUTTS"]
        values = [b["birdies"], b["pars"], b["bogeys"], b["doubles_plus"], b["putts"]]
        # Lay out across the column.
        slot_w = col_w // len(labels)
        slot_left = side_margin + col_w * i
        for j, (lbl, val) in enumerate(zip(labels, values)):
            sx = slot_left + slot_w * j + slot_w // 2
            _centered(draw, str(val), f_stat, sx, int(H * 0.78), name_color)
            _centered(draw, lbl, f_label, sx, int(H * 0.84), MUTED[:3])

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

def _build_filter(canvas: dict, n_start: int, n_clips: int,
                  n_end: int) -> str:
    """Build the filter_complex string covering N start cards + M clips + K end cards.

    Each card contributes two consecutive inputs (PNG, anullsrc). Each clip
    is one input. Final input order:
        start cards: [png0, sil0, png1, sil1, ...]   (2*n_start inputs)
        clips:       [c0, c1, ..., c_{M-1}]          (n_clips inputs)
        end cards:   [png0, sil0, png1, sil1, ...]   (2*n_end inputs)

    No color conversion. Clips are scaled and padded into the canvas and
    keep their source color characteristics; the output is tagged once at
    the encoder level (see `render_final`) to match the canvas's pick. The
    card PNGs are RGB; ffmpeg auto-converts them into the canvas pix_fmt
    when they hit `format={pf}`.
    """
    W = canvas["width"]
    H = canvas["height"]
    fps = canvas["fps_str"]
    pf = canvas["pix_fmt"]
    parts: list[str] = []
    seg_labels: list[str] = []

    def card_vid(in_idx: int, label: str) -> None:
        parts.append(
            f"[{in_idx}:v]fps={fps},setsar=1,format={pf},setpts=PTS-STARTPTS[{label}]"
        )

    def silent_aud(in_idx: int, label: str) -> None:
        parts.append(
            f"[{in_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[{label}]"
        )

    def clip_vid(in_idx: int, label: str) -> None:
        # Scale to fit, pad to canvas, normalize fps + pix_fmt. Lanczos gives a
        # clean upscale for 1080p clips on a 4K canvas.
        parts.append(
            f"[{in_idx}:v]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={fps},setsar=1,format={pf},setpts=PTS-STARTPTS[{label}]"
        )

    def clip_aud(in_idx: int, label: str) -> None:
        parts.append(
            f"[{in_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[{label}]"
        )

    # Start cards
    cur = 0
    for i in range(n_start):
        v_lbl, a_lbl = f"vs{i}", f"as{i}"
        card_vid(cur, v_lbl); silent_aud(cur + 1, a_lbl)
        seg_labels += [v_lbl, a_lbl]
        cur += 2

    # Clips
    for i in range(n_clips):
        v_lbl, a_lbl = f"v{i}", f"a{i}"
        clip_vid(cur, v_lbl); clip_aud(cur, a_lbl)
        seg_labels += [v_lbl, a_lbl]
        cur += 1

    # End cards
    for i in range(n_end):
        v_lbl, a_lbl = f"ve{i}", f"ae{i}"
        card_vid(cur, v_lbl); silent_aud(cur + 1, a_lbl)
        seg_labels += [v_lbl, a_lbl]
        cur += 2

    n_segments = n_start + n_clips + n_end
    concat_inputs = "".join(f"[{l}]" for l in seg_labels)
    parts.append(
        f"{concat_inputs}concat=n={n_segments}:v=1:a=1[outv][outa]"
    )
    return ";".join(parts)


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


def _ffmpeg_resource(pid: int) -> dict | None:
    """Return {cpu_percent, rss_mb} for a running ffmpeg pid (or None)."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pcpu=,rss="],
            capture_output=True, text=True, timeout=2,
        )
        line = out.stdout.strip()
        if not line:
            return None
        cpu_str, rss_str = line.split()
        return {"cpu_percent": float(cpu_str), "rss_mb": int(rss_str) / 1024.0}
    except Exception:
        return None


def _segment_for(elapsed_s: float, segments: list[dict]) -> dict:
    """Find the segment containing elapsed_s; returns the last one if past end."""
    for seg in segments:
        if elapsed_s < seg["end"]:
            return seg
    return segments[-1]


def _emit_progress(event: dict, segments: list[dict], total_s: float,
                   pid: int, callback) -> None:
    if callback is None:
        return

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(event.get(key, default))
        except (TypeError, ValueError):
            return default

    elapsed_s = max(0.0, _f("out_time_us") / 1_000_000.0)
    speed_str = (event.get("speed") or "").rstrip("x").strip()
    try:
        speed = float(speed_str)
    except ValueError:
        speed = 0.0
    fps = _f("fps")
    frame = int(_f("frame"))
    seg = _segment_for(elapsed_s, segments)

    eta_s = (total_s - elapsed_s) / speed if speed > 0 else None
    percent = elapsed_s / total_s if total_s > 0 else 0.0

    res = _ffmpeg_resource(pid)
    callback({
        "elapsed_s": elapsed_s,
        "total_s": total_s,
        "percent": min(1.0, percent),
        "eta_s": eta_s,
        "speed": speed,
        "fps": fps,
        "frame": frame,
        "segment_index": seg["index"] + 1,   # 1-indexed for display
        "segment_total": len(segments),
        "segment_label": seg["label"],
        "segment_kind": seg["kind"],
        "cpu_percent": res["cpu_percent"] if res else None,
        "rss_mb": res["rss_mb"] if res else None,
    })


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
    """Encode <round_dir>/<out_name> from approved trims, with custom cards.

    `progress`: optional callable invoked once per ffmpeg progress tick with a
                dict {elapsed_s, total_s, percent, eta_s, speed, fps, frame,
                segment_*, cpu_percent, rss_mb}. Use this to surface a UI.
    `on_start`: optional callable invoked once with {canvas, segments,
                total_s, pid} just after ffmpeg launches — lets the caller
                stash the pid for cancellation.
    `max_clips`: optional cap on the number of clips concatenated, useful for
                fast iteration when tuning the encoder.
    `out_name`: filename written under round_dir (default: final.mp4).
    """
    # Late import to avoid circular module dependency with batch_trim.
    from batch_trim import round_paths

    start_cards = _normalise_cards(start_cards, DEFAULT_START_CARDS)
    end_cards = _normalise_cards(end_cards, DEFAULT_END_CARDS)

    raw_dir, trims_dir, meta_dir = round_paths(round_dir)
    infos = gather_approved(meta_dir, trims_dir)
    if not infos:
        raise RuntimeError("no approved clips with rendered trims")
    if max_clips is not None and max_clips > 0:
        infos = infos[:max_clips]
        log(f"[final] limiting to first {len(infos)} clips (max_clips={max_clips})")
    canvas = pick_canvas(infos)
    n_hdr = sum(1 for i in infos if i.is_hdr())
    log(f"[final] {len(infos)} approved clips ({n_hdr} HDR, passthrough)  "
        f"canvas: {canvas['width']}x{canvas['height']} @ {canvas['fps_str']} "
        f"({canvas['pix_fmt']}, {canvas['primaries']}/{canvas['transfer']}/"
        f"{canvas['matrix']})")
    log(f"[final] start cards: {[c['kind'] for c in start_cards]}  "
        f"end cards: {[c['kind'] for c in end_cards]}")

    # Build the segment timeline: cards then clips then cards. Each clip's
    # duration comes from ffprobe (real on-disk length, not nominal pre+post).
    segments: list[dict] = []
    t = 0.0
    for c in start_cards:
        segments.append({"kind": f"card:{c['kind']}", "label": c["kind"].title(),
                         "start": t, "end": t + c["seconds"], "index": len(segments)})
        t += c["seconds"]
    for inf in infos:
        segments.append({"kind": "clip", "label": inf.path.stem,
                         "start": t, "end": t + inf.seconds, "index": len(segments)})
        t += inf.seconds
    for c in end_cards:
        segments.append({"kind": f"card:{c['kind']}", "label": c["kind"].title(),
                         "start": t, "end": t + c["seconds"], "index": len(segments)})
        t += c["seconds"]
    total_s = t

    scores_path = round_dir / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else None

    out_path = round_dir / out_name
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        def png_for(prefix: str, idx: int, kind: str) -> Path:
            img = render_card_image(kind, canvas, scores, round_dir.name)
            p = td_path / f"{prefix}{idx}_{kind}.png"
            img.save(p)
            return p

        cmd: list[str] = [
            "ffmpeg", "-y", "-v", "error", "-nostats",
            "-progress", "pipe:1",
        ]

        for i, c in enumerate(start_cards):
            png = png_for("start", i, c["kind"])
            cmd += [
                "-loop", "1", "-t", f"{c['seconds']:.3f}", "-i", str(png),
                "-f", "lavfi", "-t", f"{c['seconds']:.3f}",
                "-i", "anullsrc=cl=stereo:r=48000",
            ]
        for inf in infos:
            cmd += ["-i", str(inf.path)]
        for i, c in enumerate(end_cards):
            png = png_for("end", i, c["kind"])
            cmd += [
                "-loop", "1", "-t", f"{c['seconds']:.3f}", "-i", str(png),
                "-f", "lavfi", "-t", f"{c['seconds']:.3f}",
                "-i", "anullsrc=cl=stereo:r=48000",
            ]
        cmd += [
            "-filter_complex",
            _build_filter(canvas, len(start_cards), len(infos), len(end_cards)),
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-profile:v", canvas["profile"],
            "-crf", CRF, "-preset", PRESET,
            # Tag the output with the canvas's color characteristics.
            # We don't tonemap — HLG sources stay HLG so HDR-capable
            # players show them at full brightness. Without these tags
            # the H.264 VUI ends up "unknown" and players fall back to
            # generic BT.709 decoding, which is what produced the
            # washed-out look.
            "-color_primaries", canvas["primaries"],
            "-color_trc", canvas["transfer"],
            "-colorspace", canvas["matrix"],
            "-color_range", "tv",
            "-x264-params",
            f"colorprim={canvas['primaries']}:transfer={canvas['transfer']}"
            f":colormatrix={canvas['matrix']}",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        log("[final] encoding…")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        if on_start is not None:
            on_start({
                "canvas": canvas, "segments_total": len(segments),
                "total_s": total_s, "pid": proc.pid,
            })

        # Drain stderr in a background thread (only used for error reporting).
        stderr_lines: list[str] = []
        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line)
        threading.Thread(target=_drain_stderr, daemon=True).start()

        # Parse stdout key=value progress events.
        event: dict[str, str] = {}
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            event[k] = v.strip()
            if k == "progress":
                _emit_progress(event, segments, total_s, proc.pid, progress)
                event = {}
        rc = proc.wait()
        if rc != 0:
            tail = "".join(stderr_lines)[-2000:]
            raise subprocess.CalledProcessError(rc, cmd, tail)
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
