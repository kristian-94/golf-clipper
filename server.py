# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi", "uvicorn", "pydantic",
#   "numpy", "scipy", "soundfile", "pillow",
#   "httpx",
# ]
# ///
"""Web UI for reviewing/adjusting golf clip trims.

Run:
    uv run server.py                         # newest round under clips/
    uv run server.py --round clips/18-april-2026
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from batch_trim import (
    detect_clip,
    ensure_recorded_at,
    list_videos,
    newest_round,
    render_from_sidecar,
    round_paths,
)
from correlate import apply_to_sidecars as correlate_round
from finalise import pick_canvas_for_round, render_final
from smartcaddy import fetch_and_save as smartcaddy_fetch, load_env

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI()
STATE: dict = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1)
RENDER_STATUS: dict[str, str] = {}  # stem -> "pending" | "rendering" | "done" | "error: ..."
FINAL_STATUS: dict = {"state": "idle"}  # state: idle|encoding|done|error|cancelled
FINAL_PID: dict = {}  # {"pid": int} while encoding, used by /cancel
# Aggregate status for the "Re-render all clips" batch job — feeds /progress.
# state: idle|running|done|error. Per-stem state remains in RENDER_STATUS.
BATCH_STATUS: dict = {"state": "idle"}
# PIDs of in-flight per-clip ffmpegs spawned by render workers. The
# /render-all/cancel endpoint SIGTERMs only these — using `pgrep -x ffmpeg`
# would also kill an unrelated finalise running in parallel.
RENDER_PIDS: set[int] = set()
# Wall-clock seconds for the last few completed renders. Used to extrapolate
# an ETA for the rest of the queue — clips are roughly the same length, so
# the last 3 successful renders give a stable enough average.
RENDER_DURATIONS: deque[float] = deque(maxlen=3)


def _ffmpeg_children_resource() -> dict:
    """Sum CPU% and RSS across all ffmpeg processes on the box.

    Used to give the progress UI a feel for the batch job's footprint —
    EXECUTOR may have up to two render workers active at once, each with
    its own ffmpeg child. We don't bother filtering to children of this
    pid (other ffmpeg invocations on the box are rare during a render).
    """
    try:
        out = __import__("subprocess").run(
            ["ps", "-axo", "pid,pcpu,rss,comm"],
            capture_output=True, text=True, timeout=2,
        )
    except Exception:
        return {"cpu_percent": None, "rss_mb": None, "n_procs": 0}
    cpu = 0.0
    rss_mb = 0.0
    n = 0
    for line in out.stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        _pid, pcpu, rss_kb, comm = parts
        if "ffmpeg" not in comm:
            continue
        try:
            cpu += float(pcpu)
            rss_mb += float(rss_kb) / 1024.0
            n += 1
        except ValueError:
            continue
    if n == 0:
        return {"cpu_percent": None, "rss_mb": None, "n_procs": 0}
    return {"cpu_percent": cpu, "rss_mb": rss_mb, "n_procs": n}


def setup_state(round_dir: Path) -> None:
    raw, trims, meta = round_paths(round_dir)
    STATE["round"] = round_dir
    STATE["raw"] = raw
    STATE["trims"] = trims
    STATE["meta"] = meta
    STATE["scores"] = round_dir / "scores.json"
    # Canvas is the round's canonical output spec — every per-clip render
    # encodes to it so finalise.py can byte-copy them together. Computed
    # once here from the raw clips; identical across renders unless the
    # raw set changes.
    STATE["canvas"] = pick_canvas_for_round(raw)
    cv = STATE["canvas"]
    print(f"canvas: {cv['width']}x{cv['height']} @ {cv['fps_str']} "
          f"({cv['pix_fmt']}, {cv['primaries']}/{cv['transfer']}/{cv['matrix']})",
          flush=True)


def initial_batch() -> None:
    """Detect any clip without a sidecar, then render every trim with overlays baked in.

    Two phases so correlation can run after all detections exist — that way the
    single render pass per clip can include the scorecard overlay if available.
    """
    raw, trims, meta = STATE["raw"], STATE["trims"], STATE["meta"]
    for src in list_videos(raw):
        if (meta / f"{src.stem}.json").exists():
            continue
        print(f"[batch] detect {src.name}", flush=True)
        try:
            detect_clip(src, meta)
        except Exception as e:
            print(f"[batch] ERROR {src.name}: {e}", flush=True)
    # Backfill recorded_at on any sidecar that pre-dates that field.
    n = ensure_recorded_at(meta, raw)
    if n:
        print(f"[batch] backfilled recorded_at on {n} sidecars", flush=True)
    # If we have a SmartCaddy scorecard, (re-)correlate now that all sidecars exist.
    if STATE["scores"].exists():
        try:
            summary = correlate_round(STATE["round"], mode=STATE.get("correlate_mode"),
                                      fetch_map=STATE.get("course_map", False))
            print(f"[batch] correlated {summary['correlated']} "
                  f"(uncorrelatable: {summary['uncorrelatable']})", flush=True)
        except Exception as e:
            print(f"[batch] correlate ERROR: {e}", flush=True)
    # Render any clip that doesn't yet have a trim. Skip the scorecard overlay
    # for the initial pass — a faster preview encode is enough for triage. The
    # overlay is baked on approve (see _kick_overlay_render).
    for meta_path in sorted(meta.glob("*.json")):
        stem = meta_path.stem
        trim_path = trims / f"{stem}.mp4"
        if trim_path.exists():
            # If the trim is unreadable (e.g. ffmpeg killed mid-encode by a
            # prior server restart) drop it and let the loop re-render. If
            # it's valid and the sidecar predates `has_overlay`, treat it as
            # overlay-baked (legacy renders always had one).
            import subprocess as _sp
            ok = _sp.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "format=duration", "-of", "csv=p=0",
                 str(trim_path)],
                capture_output=True,
            ).returncode == 0
            if ok:
                data = json.loads(meta_path.read_text())
                if "has_overlay" not in data:
                    data["has_overlay"] = True
                    meta_path.write_text(json.dumps(data, indent=2))
                # If the on-disk trim is approved but its overlay is stale
                # (the user reassigned hole/shot/player and the server
                # restarted before the queued re-render fired), re-queue it.
                # In-memory RENDER_STATUS is wiped on restart, so this is
                # the only thing keeping pending re-renders from being lost.
                if data.get("review") == "approved" and data.get("has_overlay") is False:
                    RENDER_STATUS[stem] = "pending"
                    # We're already running on EXECUTOR — `run_in_executor`
                    # would need an event loop (none here). Submit directly
                    # so the job lines up after this initial batch finishes.
                    EXECUTOR.submit(_render_job, stem)
                    print(f"[batch] re-queued stale overlay: {stem}", flush=True)
                continue
            print(f"[batch] corrupt trim {stem} — re-rendering", flush=True)
            trim_path.unlink()
        # Render with overlay if the clip was already approved before this
        # restart — otherwise approve→render won't fire and the clip would
        # ship overlay-less. Unreviewed/needs_fix clips get a fast preview.
        data_for_decision = json.loads(meta_path.read_text())
        with_overlay = data_for_decision.get("review") == "approved"
        label = "render" if with_overlay else "preview"
        print(f"[batch] {label} {stem}", flush=True)
        try:
            RENDER_STATUS[stem] = "rendering"
            status = render_from_sidecar(
                meta_path, raw, trims, STATE["canvas"], with_overlay=with_overlay,
            )
            RENDER_STATUS[stem] = "done" if status == "ok" else f"error: {status}"
        except Exception as e:
            RENDER_STATUS[stem] = f"error: {e}"
            print(f"[batch] render ERROR {stem}: {e}", flush=True)
    print("[batch] done", flush=True)
    # When started with --auto-finalise, kick the final concat right after
    # the batch settles so a headless run can produce final.mp4 unattended.
    if STATE.get("auto_finalise"):
        print("[batch] auto-finalise: kicking final encode", flush=True)
        EXECUTOR.submit(_final_job, None, None)


@app.on_event("startup")
async def _on_startup() -> None:
    asyncio.get_event_loop().run_in_executor(EXECUTOR, initial_batch)


# ---- HTML routes ----

_VIDEO_CHUNK = 1 << 20  # 1 MiB per Range slice


def _serve_video(path: Path, request: Request, media_type: str) -> Response:
    """Serve a media file with HTTP Range support so the browser can seek.

    FastAPI's `StaticFiles` mount doesn't honour the `Range` header — it
    always returns 200 with the full body. That makes scrubbing a 300 MB
    iPhone clip impossible (every seek waits for byte 0 to download). This
    handler reads the `Range` header, returns 206 Partial Content with the
    requested byte slice, and the browser can seek instantly.
    """
    if not path.exists():
        raise HTTPException(404)
    file_size = path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")

    if not range_header:
        # No Range header — stream the whole file but still advertise that we
        # support ranges, so the browser knows it can seek on the next request.
        def whole():
            with path.open("rb") as f:
                while chunk := f.read(_VIDEO_CHUNK):
                    yield chunk
        return StreamingResponse(
            whole(),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    # Parse `Range: bytes=START-END`. End is optional — open-ended means
    # "to EOF" which the browser uses to grab metadata at file start.
    try:
        units, _, spec = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(416, "invalid Range header")
    if start >= file_size or end >= file_size or start > end:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    length = end - start + 1

    def slice_iter():
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(_VIDEO_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        slice_iter(),
        status_code=206,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        },
    )


@app.get("/raw/{filename}")
async def serve_raw(filename: str, request: Request):
    # Prevent path traversal — only single filenames, no separators.
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise HTTPException(400, "bad filename")
    path = STATE["raw"] / filename
    media = "video/quicktime" if filename.lower().endswith(".mov") else "video/mp4"
    return _serve_video(path, request, media)


@app.get("/trims/{filename}")
async def serve_trim(filename: str, request: Request):
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise HTTPException(400, "bad filename")
    return _serve_video(STATE["trims"] / filename, request, "video/mp4")


@app.get("/web/{filename}")
async def serve_web_asset(filename: str):
    """Serve shared front-end assets (CSS/JS) — kept simple, no StaticFiles."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400)
    p = WEB_DIR / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/")
async def index_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/clip/{stem}")
async def compare_page(stem: str):
    return FileResponse(WEB_DIR / "compare.html")


@app.get("/assign")
async def assign_page():
    return FileResponse(WEB_DIR / "assign.html")


# ---- JSON API ----

@app.get("/api/system-status")
async def api_system_status():
    """Live snapshot for the status widget: render queue + machine load.

    Cheap to call — psutil's `cpu_percent(interval=0)` returns the cached
    value since the last call, and `disk_usage` is a single statvfs.
    Polled every couple of seconds from every page in the UI.
    """
    rendering = [s for s, st in RENDER_STATUS.items() if st == "rendering"]
    pending   = [s for s, st in RENDER_STATUS.items() if st == "pending"]
    errored   = [s for s, st in RENDER_STATUS.items() if isinstance(st, str) and st.startswith("error")]
    done      = sum(1 for st in RENDER_STATUS.values() if st == "done")
    try:
        du = shutil.disk_usage(STATE.get("round") or Path.cwd())
        disk_free_gb = du.free / 2**30
        disk_total_gb = du.total / 2**30
    except Exception:
        disk_free_gb = disk_total_gb = None
    avg_render_s = (
        sum(RENDER_DURATIONS) / len(RENDER_DURATIONS) if RENDER_DURATIONS else None
    )
    # ETA covers everything still ahead of us: in-flight renders + the queue.
    # max_workers=1 so they run serially; multiplying by avg is honest.
    remaining = len(pending) + len(rendering)
    eta_s = avg_render_s * remaining if (avg_render_s and remaining) else None
    return {
        "rendering": rendering,
        "pending": pending,
        "pending_count": len(pending),
        "errored": errored[:10],
        "done_count": done,
        "cpu_percent": psutil.cpu_percent(interval=0),
        "mem_percent": psutil.virtual_memory().percent,
        "disk_free_gb": disk_free_gb,
        "disk_total_gb": disk_total_gb,
        "avg_render_s": avg_render_s,
        "samples": len(RENDER_DURATIONS),
        "eta_s": eta_s,
    }


@app.get("/api/round")
async def api_round():
    info: dict = {
        "round": STATE["round"].name,
        "raw_count": len(list_videos(STATE["raw"])),
        "has_scores": STATE["scores"].exists(),
    }
    if STATE["scores"].exists():
        scores = json.loads(STATE["scores"].read_text())
        info["course"] = scores.get("course")
        info["players"] = [p["name"] for p in scores["players"]]
        info["assignment_mode"] = scores.get("assignment_mode", "timestamps")
        info["players_locked"] = bool(scores.get("players_locked", False))
        info["holes_locked"] = bool(scores.get("holes_locked", False))
    return info


class LockedUpdate(BaseModel):
    """Toggle a lock. The assign UI uses these to progressively hide
    controls as each dimension is finalised:
      * players_locked → hide per-clip player + shot dropdowns
      * holes_locked   → hide per-clip hole dropdown (clips show as read-only)
    """
    locked: bool


@app.post("/api/holes/locked")
async def api_set_holes_locked(update: LockedUpdate):
    """Lock the hole dropdowns once every clip is on the right hole.
    No-op beyond persisting the flag — the renumber pass that runs on
    every save uses (player, hole) groups and doesn't depend on this."""
    if not STATE["scores"].exists():
        raise HTTPException(409, "no scorecard")
    scores = json.loads(STATE["scores"].read_text())
    scores["holes_locked"] = bool(update.locked)
    STATE["scores"].write_text(json.dumps(scores, indent=2))
    print(f"[lock] holes_locked = {update.locked}", flush=True)
    return {"locked": update.locked}


@app.post("/api/players/locked")
async def api_set_players_locked(update: LockedUpdate):
    if not STATE["scores"].exists():
        raise HTTPException(409, "no scorecard")
    scores = json.loads(STATE["scores"].read_text())
    scores["players_locked"] = bool(update.locked)
    STATE["scores"].write_text(json.dumps(scores, indent=2))
    print(f"[lock] players_locked = {update.locked}", flush=True)

    # When locking, run a one-shot renumber across every hole / every player
    # so the chrono-derived shot indexes are correct from the start, even
    # for clips the user hasn't touched since flipping the toggle.
    touched: list[str] = []
    if update.locked:
        from correlate import renumber_player_chrono
        sidecars: dict[str, dict] = {}
        for p in STATE["meta"].glob("*.json"):
            try:
                sidecars[p.stem] = json.loads(p.read_text())
            except Exception:
                continue
        for player in scores.get("players", []):
            updates = renumber_player_chrono(scores, sidecars, player["name"])
            for s, upd in updates.items():
                path = STATE["meta"] / f"{s}.json"
                d = json.loads(path.read_text())
                d["shot_index"] = upd["shot_index"]
                d["shot_total"] = upd["shot_total"]
                d["has_overlay"] = False
                path.write_text(json.dumps(d, indent=2))
                touched.append(s)
                if d.get("review") == "approved":
                    _kick_overlay_render(s)
        print(f"[lock] initial renumber touched {len(touched)} clip(s)", flush=True)
    return {"locked": update.locked, "renumbered": touched}


@app.get("/api/scores")
async def api_scores():
    """Raw scores.json — used by the assignments review UI to render holes/strokes."""
    if not STATE["scores"].exists():
        raise HTTPException(404, "no scores.json for this round")
    return json.loads(STATE["scores"].read_text())


class PlayedThroughUpdate(BaseModel):
    """Per-player override for the leaderboard cutoff.

    `played_through=9` means the player completed up to (and including) hole 9
    and shouldn't appear on the leaderboard for any clip on hole 10+. Set to
    `null` to clear the override (player appears for every hole).
    """
    played_through: int | None


@app.post("/api/players/{name}/played-through")
async def api_set_played_through(name: str, update: PlayedThroughUpdate):
    """Update a player's played_through cutoff and refresh every sidecar.

    Walks every sidecar with a hole assignment, recomputes its `players` list
    against the new cutoff, and kicks an overlay re-render for any clip that
    was already approved (so on-disk trims pick up the new leaderboard).
    """
    if not STATE["scores"].exists():
        raise HTTPException(404, "no scorecard for this round")
    scores = json.loads(STATE["scores"].read_text())
    target = next((p for p in scores["players"] if p["name"] == name), None)
    if target is None:
        raise HTTPException(404, f"no player named {name!r}")
    if update.played_through is None:
        target.pop("played_through", None)
    else:
        if update.played_through < 1 or update.played_through > 18:
            raise HTTPException(400, "played_through must be between 1 and 18")
        target["played_through"] = update.played_through
    STATE["scores"].write_text(json.dumps(scores, indent=2))

    # Refresh every sidecar's `players` row using the updated cutoff.
    from correlate import players_through, leaderboard_order
    order = leaderboard_order(scores)
    refreshed: list[str] = []
    for meta_path in sorted(STATE["meta"].glob("*.json")):
        data = json.loads(meta_path.read_text())
        h = data.get("hole")
        if h is None:
            continue
        data["players"] = players_through(scores, h, order=order)
        # The on-disk trim's overlay is now stale.
        was_overlaid = data.get("has_overlay", False)
        data["has_overlay"] = False
        meta_path.write_text(json.dumps(data, indent=2))
        if was_overlaid and data.get("review") == "approved":
            refreshed.append(meta_path.stem)

    # Re-render approved clips so the baked overlay catches up.
    for stem in refreshed:
        _kick_overlay_render(stem)

    return {
        "name": name,
        "played_through": target.get("played_through"),
        "rerendering": refreshed,
    }


class AssignAllPlayerUpdate(BaseModel):
    player: str  # attribute every clip in the round to this player


@app.post("/api/players/assign-all")
async def api_assign_all_player(update: AssignAllPlayerUpdate):
    """Attribute every clip in the round to a single player.

    Bulk shortcut for rounds filmed on one phone where every clip is the
    same golfer. Sets `player` on every sidecar, re-derives shot
    indexes/totals from that player's stroke counts (so the per-hole shot
    tallies match the new owner), and re-renders any approved clip so the
    baked overlay catches up.
    """
    if not STATE["scores"].exists():
        raise HTTPException(404, "no scorecard for this round")
    scores = json.loads(STATE["scores"].read_text())
    valid = {p["name"] for p in scores.get("players", [])}
    if update.player not in valid:
        raise HTTPException(400, f"unknown player: {update.player!r}")

    changed = 0
    for meta_path in sorted(STATE["meta"].glob("*.json")):
        data = json.loads(meta_path.read_text())
        if data.get("player") == update.player:
            continue
        data["player"] = update.player
        data["has_overlay"] = False
        meta_path.write_text(json.dumps(data, indent=2))
        changed += 1

    # Re-derive shot numbering for the target player across every hole now
    # that they own all the clips.
    from correlate import renumber_player_chrono
    sidecars: dict[str, dict] = {}
    for p in STATE["meta"].glob("*.json"):
        try:
            sidecars[p.stem] = json.loads(p.read_text())
        except Exception:
            continue
    rerank = renumber_player_chrono(scores, sidecars, update.player)
    for stem, upd in rerank.items():
        path = STATE["meta"] / f"{stem}.json"
        d = json.loads(path.read_text())
        if (d.get("shot_index") == upd["shot_index"]
                and d.get("shot_total") == upd["shot_total"]):
            continue
        d["shot_index"] = upd["shot_index"]
        d["shot_total"] = upd["shot_total"]
        d["has_overlay"] = False
        path.write_text(json.dumps(d, indent=2))

    # Re-render approved clips whose baked overlay is now stale.
    for meta_path in sorted(STATE["meta"].glob("*.json")):
        d = json.loads(meta_path.read_text())
        if d.get("review") == "approved" and d.get("has_overlay") is False:
            _kick_overlay_render(meta_path.stem)

    return {"player": update.player, "changed": changed, "renumbered": len(rerank)}


def _owner_name() -> str | None:
    """Name of the round's owner (the user holding the phone), or None when
    no scorecard is loaded. Used as the default `player` for clips that
    haven't been manually attributed yet."""
    if not STATE.get("scores") or not STATE["scores"].exists():
        return None
    try:
        scores = json.loads(STATE["scores"].read_text())
    except Exception:
        return None
    for p in scores.get("players", []):
        if p.get("is_owner"):
            return p.get("name")
    players = scores.get("players") or []
    return players[0]["name"] if players else None


def _read_meta(stem: str) -> dict | None:
    p = STATE["meta"] / f"{stem}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    data["stem"] = stem
    data["render_status"] = RENDER_STATUS.get(stem)
    data["has_trim"] = (STATE["trims"] / f"{stem}.mp4").exists()
    data.setdefault("review", "unreviewed")
    # Default the active player to the round's owner. Lazy fallback (vs
    # rewriting every sidecar) so existing rounds pick this up automatically;
    # an explicit setting from /assign always wins over the default.
    if not data.get("player"):
        owner = _owner_name()
        if owner:
            data["player"] = owner
    return data


def _write_review(stem: str, review: str) -> dict:
    p = STATE["meta"] / f"{stem}.json"
    if not p.exists():
        raise HTTPException(404)
    data = json.loads(p.read_text())
    data["review"] = review
    p.write_text(json.dumps(data, indent=2))
    return data


@app.get("/api/clips")
async def api_clips():
    """Return one entry per sidecar, plus placeholders for un-detected raws.

    A sidecar's stem is no longer always the raw filename — splitting a clip
    creates a sub-clip sidecar with a suffixed stem (e.g. `IMG_9641_b`) that
    points at the same raw. Iterating sidecars lets every sub-clip appear in
    the grid; raws without a sidecar yet (mid-detection) still show up via
    the second pass below.
    """
    out = []
    seen_stems: set[str] = set()
    for meta_path in sorted(STATE["meta"].glob("*.json")):
        data = _read_meta(meta_path.stem)
        if data:
            seen_stems.add(meta_path.stem)
            out.append(data)
    for src in list_videos(STATE["raw"]):
        if src.stem in seen_stems:
            continue
        out.append({
            "stem": src.stem,
            "raw": src.name,
            "confidence": "pending",
            "flagged": False,
            "edited": False,
            "has_trim": False,
            "render_status": None,
        })
    # Chronological by recorded_at; filename sort breaks across devices
    # (e.g. Android `20260427_*.mp4` always sorts before iPhone `IMG_*.MOV`).
    # Sidecars without recorded_at fall back to filename order at the end.
    out.sort(key=lambda d: (d.get("recorded_at") is None, d.get("recorded_at") or d["stem"]))
    return out


@app.get("/api/clips/{stem}")
async def api_clip(stem: str):
    data = _read_meta(stem)
    if not data:
        raise HTTPException(404, "clip not processed yet")
    return data


class ClipUpdate(BaseModel):
    impact_s: float | None = None
    pre: float | None = None
    post: float | None = None


class ReviewUpdate(BaseModel):
    review: str  # "unreviewed" | "needs_fix" | "approved"


class RotateRequest(BaseModel):
    """Per-clip rotation override. Some raws come off the phone in the wrong
    orientation (e.g. EXIF says portrait but the lens was held landscape).
    `rotate` is degrees clockwise: 0, 90, 180, or 270."""
    rotate: int


class SplitRequest(BaseModel):
    """One raw video can contain multiple shots (e.g. several putts on the
    same green). Splitting clones the sidecar with a new impact_s; both clips
    share the raw, render to separate trims, and live as independent rows in
    the grid + assign UI."""
    impact_s: float


def _next_split_stem(raw_stem: str, existing_stems: set[str]) -> str | None:
    """Pick the next free letter suffix (`_b`, `_c`, …) for a sub-clip.

    Returns None if all 25 letter suffixes are taken — practically never,
    but the caller surfaces it as 409 rather than crashing. The original
    sidecar keeps the bare raw stem (`IMG_9641`) so existing trims/sidecars
    don't get renamed when the first split happens.
    """
    for letter in "bcdefghijklmnopqrstuvwxyz":
        candidate = f"{raw_stem}_{letter}"
        if candidate not in existing_stems:
            return candidate
    return None


class AssignUpdate(BaseModel):
    """Manual hole/shot reassignment from the review UI.

    `hole=None` clears the assignment (e.g. user marks the clip as "not a shot").
    `shot_index`/`shot_total` are optional — when omitted but `hole` is set,
    they're left as whatever the sidecar already had so the user can adjust
    hole and shot independently.
    """
    hole: int | None = None
    shot_index: int | None = None
    shot_total: int | None = None
    player: str | None = None  # name of the player whose shot this clip captures


def _kick_overlay_render(stem: str) -> None:
    """If the on-disk trim is a preview (no overlay baked in), re-render with
    the scorecard. No-op if the sidecar already records `has_overlay = true`,
    so approving an already-final clip doesn't churn ffmpeg."""
    meta_path = STATE["meta"] / f"{stem}.json"
    if not meta_path.exists():
        return
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return
    if data.get("has_overlay") is True:
        return
    RENDER_STATUS[stem] = "pending"
    asyncio.get_event_loop().run_in_executor(EXECUTOR, _render_job, stem)


@app.post("/api/clips/{stem}/review")
async def api_review_clip(stem: str, update: ReviewUpdate):
    if update.review not in ("unreviewed", "needs_fix", "approved"):
        raise HTTPException(400, "bad review value")
    _write_review(stem, update.review)
    if update.review == "approved":
        _kick_overlay_render(stem)
    return _read_meta(stem)


@app.post("/api/clips/{stem}/assign")
async def api_assign_clip(stem: str, update: AssignUpdate):
    """Manually adjust hole/shot/player for a clip (review-assignments UI).

    Touches ONLY the requested fields and the affected player's per-hole
    shot indexes. Does not walk forward across holes — that was the old
    cascade behaviour and it was clobbering the user's manual edits.
    """
    meta_path = STATE["meta"] / f"{stem}.json"
    if not meta_path.exists():
        raise HTTPException(404)
    if not STATE["scores"].exists():
        raise HTTPException(409, "no scorecard — assignment requires scores.json")

    data = json.loads(meta_path.read_text())
    scores = json.loads(STATE["scores"].read_text())
    par_by_hole = {h["number"]: h["par"] for h in scores["holes"]}

    from correlate import players_through, leaderboard_order

    sent = update.model_fields_set
    before = {
        "hole": data.get("hole"),
        "shot_index": data.get("shot_index"),
        "shot_total": data.get("shot_total"),
        "player": data.get("player"),
    }
    print(f"[assign] {stem} sent={list(sent)} body={update.model_dump()} before={before}",
          flush=True)
    old_player = data.get("player") or _owner_name()

    if "hole" in sent:
        if update.hole is None:
            for k in ("hole", "par", "players", "shot_index", "shot_total",
                      "correlate_warning", "correlate_error"):
                data.pop(k, None)
        else:
            if update.hole not in par_by_hole:
                raise HTTPException(400, f"hole {update.hole} not on this scorecard")
            data["hole"] = update.hole
            data["par"] = par_by_hole[update.hole]
            data["players"] = players_through(scores, update.hole, order=leaderboard_order(scores))
            if "shot_index" in sent and update.shot_index is not None:
                data["shot_index"] = update.shot_index
            if "shot_total" in sent and update.shot_total is not None:
                data["shot_total"] = update.shot_total
            data.pop("correlate_warning", None)
            data.pop("correlate_error", None)
    elif "shot_index" in sent or "shot_total" in sent:
        # Shot edit without hole change — apply directly. Hole stays put.
        if "shot_index" in sent and update.shot_index is not None:
            data["shot_index"] = update.shot_index
        if "shot_total" in sent and update.shot_total is not None:
            data["shot_total"] = update.shot_total

    if "player" in sent:
        valid = {p["name"] for p in scores.get("players", [])}
        if update.player and update.player not in valid:
            raise HTTPException(400, f"unknown player: {update.player!r}")
        if update.player:
            data["player"] = update.player
        else:
            data.pop("player", None)

    data["has_overlay"] = False
    meta_path.write_text(json.dumps(data, indent=2))

    after = {
        "hole": data.get("hole"),
        "shot_index": data.get("shot_index"),
        "shot_total": data.get("shot_total"),
        "player": data.get("player"),
    }
    print(f"[assign] {stem} after={after}", flush=True)

    if data.get("review") == "approved":
        _kick_overlay_render(stem)

    # Moving a clip to a different hole is the "shift everything after this
    # forward" signal — cascade chrono-later same-player clips through the
    # scorecard's stroke counts. This fires in BOTH locked and unlocked
    # mode: the user shouldn't have to lock players first just to get the
    # reflow. In locked mode we additionally renumber on player flips so the
    # derived shot indexes stay chrono-correct. A shot-only edit is treated
    # as a manual override and skips the renumber, so the user's pick sticks.
    side_changes: list[dict] = []
    hole_changed = (
        "hole" in sent and update.hole is not None
        and update.hole != before["hole"]
    )
    hole_or_player_changed = (
        ("hole" in sent and update.hole != before["hole"])
        or ("player" in sent and update.player != before["player"])
    )
    if hole_changed or (scores.get("players_locked") and hole_or_player_changed):
        affected: set[tuple[str, int]] = set()
        new_player = data.get("player") or _owner_name()
        new_hole = data.get("hole")
        if new_player and new_hole is not None:
            affected.add((new_player, new_hole))
        if before["player"] and before["hole"] is not None:
            affected.add((before["player"] or _owner_name(), before["hole"]))

        # Cascade only on a real hole change — not on shot edits or player
        # flips. The anchor's player keeps walking forward through holes
        # using their stroke counts as boundaries.
        cascade_triggered = hole_changed and new_player
        if cascade_triggered:
            # Renumber the new (player, hole) first so the anchor's
            # shot_index reflects its chrono position before cascade reads it.
            side_changes.extend(_renumber_in_place(scores, new_player, new_hole))
            cascade_changes, cascade_holes = _cascade_forward(scores, stem, new_player)
            side_changes.extend(cascade_changes)
            # Discard already-renumbered holes; renumber the rest the cascade
            # touched (old holes that lost a clip, new holes that gained one).
            for hole in cascade_holes:
                if (new_player, hole) in affected:
                    continue
                affected.add((new_player, hole))

        for player, hole in affected:
            side_changes.extend(_renumber_in_place(scores, player, hole))

    return {**(_read_meta(stem) or {}), "side_changes": side_changes}


def _cascade_forward(
    scores: dict, anchor_stem: str, player: str
) -> tuple[list[dict], set[int]]:
    """Walk same-player clips chronologically after `anchor_stem`, advancing
    holes/shots according to the player's stroke counts. The anchor's
    (hole, shot_index) seeds the cursor; subsequent clips fill the rest of
    the anchor's hole, then the next hole with strokes, etc.

    Returns (changes, holes_touched). `holes_touched` includes the OLD hole
    of every clip that moved, plus all the new holes the cursor visited —
    so the caller knows which (player, hole) groups to renumber after.
    """
    from correlate import cascade_from
    sidecars: dict[str, dict] = {}
    for p in STATE["meta"].glob("*.json"):
        try:
            sidecars[p.stem] = json.loads(p.read_text())
        except Exception:
            continue
    updates = cascade_from(scores, sidecars, anchor_stem)
    changes: list[dict] = []
    holes_touched: set[int] = set()
    for stem, upd in updates.items():
        path = STATE["meta"] / f"{stem}.json"
        on_disk = json.loads(path.read_text())
        old_hole = on_disk.get("hole")
        if (
            on_disk.get("hole") == upd["hole"]
            and on_disk.get("shot_index") == upd["shot_index"]
            and on_disk.get("shot_total") == upd["shot_total"]
        ):
            continue
        on_disk["hole"] = upd["hole"]
        on_disk["par"] = upd.get("par", on_disk.get("par"))
        if upd.get("players") is not None:
            on_disk["players"] = upd["players"]
        on_disk["shot_index"] = upd["shot_index"]
        on_disk["shot_total"] = upd["shot_total"]
        on_disk["has_overlay"] = False
        path.write_text(json.dumps(on_disk, indent=2))
        print(
            f"[cascade] {stem} hole {old_hole} -> {upd['hole']} "
            f"shot {upd['shot_index']}/{upd['shot_total']}",
            flush=True,
        )
        changes.append({
            "stem": stem, "hole": upd["hole"],
            "shot_index": upd["shot_index"], "shot_total": upd["shot_total"],
        })
        if old_hole is not None:
            holes_touched.add(old_hole)
        holes_touched.add(upd["hole"])
        if on_disk.get("review") == "approved":
            _kick_overlay_render(stem)
    print(f"[cascade] anchor={anchor_stem} player={player} touched {len(changes)} clip(s)", flush=True)
    return changes, holes_touched


def _renumber_in_place(scores: dict, player: str, hole: int) -> list[dict]:
    """Read every sidecar, rerank `player`'s clips at `hole` by chrono, and
    write the changes back. Returns the list of stems that actually changed.

    Lifted out of the cascade-style endpoint so api_assign_clip can call it
    inline when locked-players mode is on.
    """
    from correlate import renumber_player_chrono
    owner = _owner_name()
    sidecars: dict[str, dict] = {}
    for p in STATE["meta"].glob("*.json"):
        try:
            sidecars[p.stem] = json.loads(p.read_text())
        except Exception:
            continue
    rerank = renumber_player_chrono(scores, sidecars, player)
    changed: list[dict] = []
    for stem, upd in rerank.items():
        d = sidecars[stem]
        if d.get("hole") != hole:
            continue  # only this hole's bucket
        path = STATE["meta"] / f"{stem}.json"
        on_disk = json.loads(path.read_text())
        if (
            on_disk.get("shot_index") == upd["shot_index"]
            and on_disk.get("shot_total") == upd["shot_total"]
        ):
            continue
        old = (on_disk.get("shot_index"), on_disk.get("shot_total"))
        on_disk["shot_index"] = upd["shot_index"]
        on_disk["shot_total"] = upd["shot_total"]
        on_disk["has_overlay"] = False
        path.write_text(json.dumps(on_disk, indent=2))
        print(f"[renumber-inline] {stem} hole={hole} player={player} {old} -> ({upd['shot_index']}, {upd['shot_total']})", flush=True)
        changed.append({
            "stem": stem, "hole": hole,
            "shot_index": upd["shot_index"], "shot_total": upd["shot_total"],
        })
        if on_disk.get("review") == "approved":
            _kick_overlay_render(stem)
    return changed


@app.post("/api/clips/{stem}/renumber")
async def api_renumber_from(stem: str):
    """After a manual assignment, rerank ONLY same-player clips at the
    anchor's hole by recorded_at so shot indexes match chrono order with
    no duplicates.

    Scope by design:
      * touches one hole only — the anchor's hole
      * touches one player only — the anchor's player
      * never moves a clip to a different hole (that's the user's call)

    Logs every input clip and every output change.
    """
    if not STATE["scores"].exists():
        raise HTTPException(409, "no scorecard")
    scores = json.loads(STATE["scores"].read_text())
    sidecars: dict[str, dict] = {}
    for p in STATE["meta"].glob("*.json"):
        try:
            sidecars[p.stem] = json.loads(p.read_text())
        except Exception:
            continue
    if stem not in sidecars:
        raise HTTPException(404)

    anchor = sidecars[stem]
    anchor_hole = anchor.get("hole")
    if anchor_hole is None:
        print(f"[renumber] {stem} has no hole — skipping", flush=True)
        return {"updated": [], "count": 0, "reason": "no hole"}

    owner = _owner_name()
    anchor_player = anchor.get("player") or owner
    if not anchor_player:
        print(f"[renumber] {stem} no player & no owner — skipping", flush=True)
        return {"updated": [], "count": 0, "reason": "no player"}

    # Strokes for this player at this hole drives shot_total.
    player_obj = next(
        (p for p in scores.get("players", []) if p["name"] == anchor_player), None
    )
    strokes_for_hole = None
    for s in (player_obj.get("scores", []) if player_obj else []):
        if s.get("hole") == anchor_hole and s.get("strokes") is not None:
            strokes_for_hole = s["strokes"]
            break

    # Find every same-player clip at this hole, sort by recorded_at.
    members: list[tuple[str, dict]] = []
    for s, d in sidecars.items():
        if d.get("hole") != anchor_hole:
            continue
        if (d.get("player") or owner) != anchor_player:
            continue
        if not d.get("recorded_at"):
            print(f"[renumber] {s} missing recorded_at — leaving alone", flush=True)
            continue
        members.append((s, d))
    # Tiebreak by impact_s so split sub-clips (which inherit recorded_at
    # from the parent MOV) sort by their strike moment within the raw.
    members.sort(key=lambda kv: (kv[1]["recorded_at"], kv[1].get("impact_s") or 0.0))
    new_total = strokes_for_hole if strokes_for_hole is not None else len(members)
    print(
        f"[renumber] hole={anchor_hole} player={anchor_player} "
        f"strokes={strokes_for_hole} clips={len(members)} "
        f"order={[s for s, _ in members]}",
        flush=True,
    )

    changed: list[dict] = []
    for i, (s, d) in enumerate(members):
        new_shot = i + 1
        if d.get("shot_index") == new_shot and d.get("shot_total") == new_total:
            continue
        sidecar_path = STATE["meta"] / f"{s}.json"
        on_disk = json.loads(sidecar_path.read_text())
        old = (on_disk.get("shot_index"), on_disk.get("shot_total"))
        on_disk["shot_index"] = new_shot
        on_disk["shot_total"] = new_total
        on_disk["has_overlay"] = False
        sidecar_path.write_text(json.dumps(on_disk, indent=2))
        print(
            f"[renumber] {s} shot {old} -> ({new_shot}, {new_total})",
            flush=True,
        )
        changed.append({
            "stem": s, "hole": anchor_hole,
            "shot_index": new_shot, "shot_total": new_total,
        })
        if on_disk.get("review") == "approved":
            _kick_overlay_render(s)

    print(f"[renumber] done — {len(changed)} clip(s) changed", flush=True)
    return {"updated": changed, "count": len(changed)}


@app.post("/api/clips/{stem}/discard")
async def api_discard_clip(stem: str):
    meta_path = STATE["meta"] / f"{stem}.json"
    if not meta_path.exists():
        raise HTTPException(404)
    data = json.loads(meta_path.read_text())
    raw_name = data["raw"]
    raw_src = STATE["raw"] / raw_name
    trim_src = STATE["trims"] / f"{stem}.mp4"

    trash_root = STATE["round"] / "trash"
    (trash_root / "raw").mkdir(parents=True, exist_ok=True)
    (trash_root / "trims").mkdir(parents=True, exist_ok=True)
    (trash_root / "meta").mkdir(parents=True, exist_ok=True)

    # Move the trim + sidecar first so the raw-reference count below is accurate.
    if trim_src.exists():
        trim_src.rename(trash_root / "trims" / f"{stem}.mp4")
    meta_path.rename(trash_root / "meta" / f"{stem}.json")

    # Only move the raw to trash if no surviving sidecar still references it.
    # Sub-clips (created via /split) share a raw with the original — discarding
    # one shouldn't strand the others.
    still_referenced = False
    for p in STATE["meta"].glob("*.json"):
        try:
            if json.loads(p.read_text()).get("raw") == raw_name:
                still_referenced = True
                break
        except (json.JSONDecodeError, OSError):
            continue
    if not still_referenced and raw_src.exists():
        raw_src.rename(trash_root / "raw" / raw_name)

    RENDER_STATUS.pop(stem, None)
    return {"discarded": stem, "raw_moved": not still_referenced}


@app.post("/api/review/approve_unreviewed")
async def api_approve_unreviewed():
    newly_approved: list[str] = []
    for src in list_videos(STATE["raw"]):
        meta_path = STATE["meta"] / f"{src.stem}.json"
        if not meta_path.exists():
            continue
        data = json.loads(meta_path.read_text())
        if data.get("review", "unreviewed") == "unreviewed":
            data["review"] = "approved"
            meta_path.write_text(json.dumps(data, indent=2))
            newly_approved.append(src.stem)
    for stem in newly_approved:
        _kick_overlay_render(stem)
    return {"approved": len(newly_approved)}


@app.post("/api/clips/{stem}/rotate")
async def api_rotate_clip(stem: str, req: RotateRequest):
    """Set/clear per-clip rotation and re-render so the change takes effect.

    The trim is always stale after a rotation change, so we kick a fresh
    render — overlay-baked if previously approved, preview otherwise.
    """
    if req.rotate not in (0, 90, 180, 270):
        raise HTTPException(400, "rotate must be 0, 90, 180, or 270")
    meta_path = STATE["meta"] / f"{stem}.json"
    if not meta_path.exists():
        raise HTTPException(404)
    data = json.loads(meta_path.read_text())
    if req.rotate == 0:
        data.pop("rotate", None)
    else:
        data["rotate"] = req.rotate
    data["has_overlay"] = False
    meta_path.write_text(json.dumps(data, indent=2))
    RENDER_STATUS[stem] = "pending"
    asyncio.get_event_loop().run_in_executor(EXECUTOR, _render_job, stem)
    return _read_meta(stem)


@app.post("/api/clips/{stem}/split")
async def api_split_clip(stem: str, req: SplitRequest):
    """Create a sub-clip from this clip's raw video at a new impact moment.

    The new sidecar inherits hole/par/players/shot/gps/recorded_at from the
    parent (so a sub-clip on the same hole picks up the same overlay) but
    starts unreviewed and unrendered. We immediately queue a render so the
    new tile shows a video as soon as ffmpeg's done, no manual save needed.
    """
    src_meta = STATE["meta"] / f"{stem}.json"
    if not src_meta.exists():
        raise HTTPException(404)
    base = json.loads(src_meta.read_text())
    raw_name = base.get("raw")
    if not raw_name:
        raise HTTPException(409, "sidecar has no raw filename")

    raw_stem = Path(raw_name).stem
    existing = {p.stem for p in STATE["meta"].glob("*.json")}
    new_stem = _next_split_stem(raw_stem, existing)
    if new_stem is None:
        raise HTTPException(409, "too many splits for this raw")

    # Clone, then reset per-trim state — review starts fresh for the new shot,
    # detector confidence/flags belong to the original detection event, and
    # any cached render timestamps would be misleading.
    new_data = dict(base)
    new_data["impact_s"] = float(req.impact_s)
    new_data["edited"] = True
    new_data["review"] = "unreviewed"
    new_data["confidence"] = "manual"  # user-placed, not detected
    new_data["flagged"] = False
    new_data["reasons"] = []
    new_data["has_overlay"] = False
    for stale in ("trimmed_at", "correlate_warning", "correlate_error"):
        new_data.pop(stale, None)

    (STATE["meta"] / f"{new_stem}.json").write_text(json.dumps(new_data, indent=2))

    # Kick a preview render (no overlay) so the tile fills in fast. The
    # overlay gets baked when the user approves, matching the main flow.
    RENDER_STATUS[new_stem] = "pending"
    asyncio.get_event_loop().run_in_executor(EXECUTOR, _render_job, new_stem)

    return _read_meta(new_stem)


@app.get("/api/duplicate-splits")
async def api_duplicate_splits(gap: float = 4.0):
    """Find sub-clip pairs on the same raw whose impact moments are within
    `gap` seconds — likely accidental double-clicks of the split button.

    Returns a list of groups (one per raw) with their member clips sorted by
    impact, and the smallest gap inside the group. Only raws with at least
    one suspect pair appear.
    """
    by_raw: dict[str, list[dict]] = {}
    for meta_path in STATE["meta"].glob("*.json"):
        try:
            data = json.loads(meta_path.read_text())
        except Exception:
            continue
        raw = data.get("raw")
        impact = data.get("impact_s")
        if not raw or impact is None:
            continue
        by_raw.setdefault(raw, []).append({
            "stem": meta_path.stem,
            "impact_s": float(impact),
            "review": data.get("review", "unreviewed"),
        })

    groups = []
    for raw, members in by_raw.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: m["impact_s"])
        min_gap = min(
            members[i + 1]["impact_s"] - members[i]["impact_s"]
            for i in range(len(members) - 1)
        )
        if min_gap <= gap:
            groups.append({"raw": raw, "min_gap_s": min_gap, "clips": members})
    groups.sort(key=lambda g: g["min_gap_s"])
    return groups


@app.post("/api/clips/{stem}")
async def api_update_clip(stem: str, update: ClipUpdate):
    meta_path = STATE["meta"] / f"{stem}.json"
    if not meta_path.exists():
        raise HTTPException(404)
    data = json.loads(meta_path.read_text())
    if update.impact_s is not None:
        data["impact_s"] = update.impact_s
    if update.pre is not None:
        data["pre"] = update.pre
    if update.post is not None:
        data["post"] = update.post
    data["edited"] = True
    meta_path.write_text(json.dumps(data, indent=2))
    return _read_meta(stem)


def _batch_started(stem: str) -> None:
    """Called when a render job begins — only counts toward batch if queued."""
    if BATCH_STATUS.get("state") != "running":
        return
    if stem not in BATCH_STATUS.get("queued_stems", set()):
        return
    BATCH_STATUS["in_progress"].add(stem)
    BATCH_STATUS["current_label"] = stem


def _batch_finished(stem: str, ok: bool, msg: str = "") -> None:
    """Called when a render job ends. Updates counters; flips state when done."""
    if BATCH_STATUS.get("state") != "running":
        return
    if stem not in BATCH_STATUS.get("queued_stems", set()):
        return
    BATCH_STATUS["in_progress"].discard(stem)
    if ok:
        BATCH_STATUS["completed"] += 1
    else:
        BATCH_STATUS["errors"] += 1
        BATCH_STATUS["errors_list"].append({"stem": stem, "msg": msg})
    done = BATCH_STATUS["completed"] + BATCH_STATUS["errors"]
    if done >= BATCH_STATUS["total"]:
        if BATCH_STATUS.get("cancel_requested"):
            BATCH_STATUS["state"] = "cancelled"
        else:
            BATCH_STATUS["state"] = "error" if BATCH_STATUS["errors"] else "done"
        BATCH_STATUS["finished_at"] = __import__("time").time()


def _render_job(stem: str) -> None:
    """Single-pass trim+overlay+normalize from sidecar. Output matches canvas."""
    # Cancel-aware: queued workers check the flag before starting their ffmpeg
    # so a cancel drains the queue without spawning more encodes. In-flight
    # ffmpegs are killed by the /cancel endpoint via SIGTERM.
    if BATCH_STATUS.get("cancel_requested"):
        RENDER_STATUS[stem] = "cancelled"
        _batch_finished(stem, False, "cancelled")
        return
    try:
        RENDER_STATUS[stem] = "rendering"
        _batch_started(stem)
        meta_path = STATE["meta"] / f"{stem}.json"
        t0 = time.monotonic()
        status = render_from_sidecar(
            meta_path, STATE["raw"], STATE["trims"], STATE["canvas"],
            pid_set=RENDER_PIDS,
        )
        ok = status == "ok"
        if ok:
            RENDER_DURATIONS.append(time.monotonic() - t0)
        RENDER_STATUS[stem] = "done" if ok else f"error: {status}"
        _batch_finished(stem, ok, "" if ok else status)
    except Exception as e:
        RENDER_STATUS[stem] = f"error: {e}"
        _batch_finished(stem, False, str(e))


@app.post("/api/clips/{stem}/render")
async def api_render(stem: str):
    if not (STATE["meta"] / f"{stem}.json").exists():
        raise HTTPException(404)
    RENDER_STATUS[stem] = "pending"
    asyncio.get_event_loop().run_in_executor(EXECUTOR, _render_job, stem)
    return {"status": RENDER_STATUS[stem]}


@app.get("/api/clips/{stem}/status")
async def api_status(stem: str):
    return {"status": RENDER_STATUS.get(stem, "idle")}


@app.post("/api/render-all")
async def api_render_all():
    """Queue a re-render of every clip with a sidecar (overlay baked in if available)."""
    if BATCH_STATUS.get("state") == "running":
        raise HTTPException(409, "batch already running")
    stems: list[str] = []
    for meta_path in sorted(STATE["meta"].glob("*.json")):
        stems.append(meta_path.stem)
    # Initialise aggregate batch status BEFORE queuing — workers race the
    # main loop, so the queued_stems set must be visible when they start.
    BATCH_STATUS.clear()
    BATCH_STATUS.update({
        "state": "running",
        "started_at": __import__("time").time(),
        "total": len(stems),
        "completed": 0,
        "errors": 0,
        "in_progress": set(),
        "queued_stems": set(stems),
        "current_label": None,
        "errors_list": [],
        "canvas": STATE.get("canvas"),
    })
    for stem in stems:
        RENDER_STATUS[stem] = "pending"
        asyncio.get_event_loop().run_in_executor(EXECUTOR, _render_job, stem)
    return {"queued": len(stems), "stems": stems}


@app.post("/api/render-all/cancel")
async def api_render_all_cancel():
    """Stop the in-flight render-all batch.

    Sets the cancel flag (queued workers short-circuit before encoding) and
    SIGTERMs only the ffmpegs spawned by render workers — tracked in
    RENDER_PIDS — so a concurrent finalise encode isn't taken down with us.
    """
    if BATCH_STATUS.get("state") != "running":
        raise HTTPException(409, "no batch running")
    BATCH_STATUS["cancel_requested"] = True
    import os
    import signal as _signal
    killed = 0
    for pid in list(RENDER_PIDS):
        try:
            os.kill(pid, _signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
    return {"cancelling": True, "killed": killed}


@app.get("/api/render-all/status")
async def api_render_all_status():
    """Aggregate progress for the latest re-render-all run.

    Combines counters from BATCH_STATUS with a live ffmpeg CPU/RSS sample so
    the /progress page can show the same shape of card as the finalise job.
    """
    s: dict = {}
    for k, v in BATCH_STATUS.items():
        if isinstance(v, set):
            s[k] = sorted(v)
        else:
            s[k] = v
    state = s.get("state", "idle")
    if state == "running":
        elapsed = __import__("time").time() - s["started_at"]
        done = s["completed"] + s["errors"]
        s["elapsed_s"] = elapsed
        s["percent"] = (done / s["total"]) if s["total"] else 0.0
        # ETA: extrapolate from average per-clip time so far.
        if done > 0 and elapsed > 0:
            s["eta_s"] = elapsed / done * (s["total"] - done)
        else:
            s["eta_s"] = None
        s.update(_ffmpeg_children_resource())
    elif state in ("done", "error", "cancelled"):
        s["elapsed_s"] = s.get("finished_at", 0) - s.get("started_at", 0)
        s["percent"] = 1.0 if state != "cancelled" else (
            (s["completed"] + s["errors"]) / s["total"] if s.get("total") else 0.0
        )
        s["eta_s"] = 0.0
        s["cpu_percent"] = None
        s["rss_mb"] = None
        s["n_procs"] = 0
    return s


def _final_job(start_cards, end_cards) -> None:
    """Run inside the executor. Mutates FINAL_STATUS as ffmpeg progresses."""
    import os
    import signal

    def on_start(info: dict) -> None:
        # render_final calls on_start twice: first before encoding cards
        # (pid=None — there's no single ffmpeg yet, each card is its own
        # short subprocess), and then again with the real pid right before
        # the concat-copy phase. Only the second call has a cancellable pid.
        if info.get("pid"):
            FINAL_PID["pid"] = info["pid"]
        if "canvas" in info:
            FINAL_STATUS["canvas"] = info["canvas"]
            FINAL_STATUS["segments_total"] = info["segments_total"]
            FINAL_STATUS["total_s"] = info["total_s"]
            FINAL_STATUS.setdefault("started_at", __import__("time").time())

    def on_progress(p: dict) -> None:
        FINAL_STATUS.update(p)
        FINAL_STATUS["state"] = "encoding"

    try:
        # Reset transient progress fields from any previous run.
        for k in ("percent", "elapsed_s", "eta_s", "speed", "fps", "frame",
                  "segment_index", "segment_label", "segment_kind",
                  "cpu_percent", "rss_mb", "message"):
            FINAL_STATUS.pop(k, None)
        FINAL_STATUS["state"] = "encoding"
        out = render_final(
            STATE["round"],
            start_cards=start_cards, end_cards=end_cards,
            progress=on_progress, on_start=on_start,
        )
        FINAL_STATUS["state"] = "done"
        FINAL_STATUS["path"] = str(out)
        # Snap the bar to 100 on success.
        FINAL_STATUS["percent"] = 1.0
    except Exception as e:
        # Distinguish a user-triggered cancel from a real failure.
        if FINAL_STATUS.get("cancel_requested"):
            FINAL_STATUS["state"] = "cancelled"
            FINAL_STATUS["message"] = "cancelled by user"
        else:
            FINAL_STATUS["state"] = "error"
            FINAL_STATUS["message"] = str(e)
            print(f"[final] ERROR: {e}", flush=True)
    finally:
        FINAL_PID.pop("pid", None)
        FINAL_STATUS.pop("cancel_requested", None)


class CardSpec(BaseModel):
    kind: str
    seconds: float | None = None


class FinaliseRequest(BaseModel):
    start: list[CardSpec] | None = None
    end: list[CardSpec] | None = None


@app.post("/api/finalise")
async def api_finalise(req: FinaliseRequest | None = None):
    if FINAL_STATUS.get("state") == "encoding":
        raise HTTPException(409, "already encoding")
    # Refuse if any per-clip render is still in flight — otherwise finalise
    # will ffprobe a half-written .mp4 and fail. Approves can kick overlay
    # re-renders, so this race is easy to hit on a quick "approve all → final".
    pending = [s for s, st in RENDER_STATUS.items() if st in ("pending", "rendering")]
    if pending:
        raise HTTPException(
            409,
            f"{len(pending)} clip render(s) still in flight: {', '.join(sorted(pending)[:5])}"
            f"{'…' if len(pending) > 5 else ''}",
        )
    start = [c.model_dump() for c in req.start] if req and req.start is not None else None
    end = [c.model_dump() for c in req.end] if req and req.end is not None else None
    FINAL_STATUS["state"] = "encoding"
    FINAL_STATUS.pop("message", None)
    asyncio.get_event_loop().run_in_executor(EXECUTOR, _final_job, start, end)
    return {"state": FINAL_STATUS["state"]}


@app.post("/api/finalise/cancel")
async def api_finalise_cancel():
    pid = FINAL_PID.get("pid")
    if not pid or FINAL_STATUS.get("state") != "encoding":
        raise HTTPException(409, "no encoding in progress")
    import os
    import signal as _signal
    FINAL_STATUS["cancel_requested"] = True
    try:
        os.kill(pid, _signal.SIGTERM)
    except ProcessLookupError:
        pass
    return {"cancelling": True}


@app.get("/api/finalise/status")
async def api_finalise_status():
    out = STATE["round"] / "final.mp4"
    return {**FINAL_STATUS, "exists": out.exists()}


@app.get("/final.mp4")
async def serve_final():
    out = STATE["round"] / "final.mp4"
    if not out.exists():
        raise HTTPException(404)
    # `content_disposition_type="inline"` so browsers stream the video in
    # their built-in player rather than triggering a download. The filename
    # is still set for when the user does choose to save it.
    return FileResponse(
        out, media_type="video/mp4",
        filename=f"{STATE['round'].name}.mp4",
        content_disposition_type="inline",
    )


@app.get("/progress")
async def progress_page():
    return FileResponse(WEB_DIR / "progress.html")


def main() -> None:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=Path, help="round folder (default: newest under --clips-root)")
    ap.add_argument("--clips-root", type=Path, default=Path("clips"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--share-token", help="SmartCaddy round share token (no auth)")
    sc.add_argument("--round-id", help="SmartCaddy round id (uses PAT from .env)")
    ap.add_argument("--refetch-scores", action="store_true",
                    help="re-fetch scorecard even if scores.json already exists")
    ap.add_argument("--no-shot-times", action="store_true",
                    help="scorecard was entered after the round — ignore "
                         "score.created_at and assign clips to holes by "
                         "scorecard stroke count, then review in the UI. "
                         "Persisted into scores.json so the next run picks "
                         "the same mode automatically.")
    ap.add_argument("--auto-finalise", action="store_true",
                    help="after initial batch settles, automatically render "
                         "final.mp4 — useful for headless end-to-end runs")
    ap.add_argument("--course-map", action="store_true",
                    help="bake an OSM hole map into the overlay. Off by default. "
                         "Only renders when the round's GPS sits cleanly inside a "
                         "single course boundary (clustered/ambiguous areas get no "
                         "map rather than a wrong one).")
    args = ap.parse_args()

    round_dir = args.round or newest_round(args.clips_root)
    setup_state(round_dir)
    STATE["auto_finalise"] = bool(args.auto_finalise)
    STATE["course_map"] = bool(args.course_map)
    STATE["correlate_mode"] = "gaps" if args.no_shot_times else None  # None → read from scores.json
    print(f"serving round: {round_dir}", flush=True)

    # Map is opt-in. When off, drop any stale course pointer so a previously
    # fetched map doesn't keep baking into renders for this round.
    if not args.course_map:
        pointer = round_dir / "course.json"
        if pointer.exists():
            pointer.unlink()

    # SmartCaddy: fetch + correlate up front so the UI starts with hole info
    # for any sidecars that already exist. The post-batch pass in initial_batch
    # will catch newly-processed clips.
    if args.share_token or args.round_id or (args.refetch_scores and STATE["scores"].exists()):
        load_env(Path(__file__).parent / ".env")
        token = os.environ.get("SMARTCADDY_TOKEN")
        try:
            data = smartcaddy_fetch(
                round_dir,
                share_token=args.share_token,
                round_id=args.round_id,
                token=token,
            )
            print(f"[smartcaddy] {data['course']}: "
                  f"{len(data['players'])} players × {len(data['holes'])} holes", flush=True)
        except Exception as e:
            print(f"[smartcaddy] fetch failed: {e}", flush=True)
    if STATE["scores"].exists():
        try:
            ensure_recorded_at(STATE["meta"], STATE["raw"])
            summary = correlate_round(round_dir, mode=STATE["correlate_mode"],
                                      fetch_map=STATE["course_map"])
            print(f"[correlate] {summary['correlated']} clips correlated "
                  f"(uncorrelatable: {summary['uncorrelatable']})", flush=True)
        except Exception as e:
            print(f"[correlate] failed: {e}", flush=True)

    # Range-aware video serving is registered as routes (not StaticFiles mount)
    # so the browser can seek without redownloading from byte 0. StaticFiles
    # ignores the Range header and ships the whole file on every request,
    # which makes scrubbing through 300 MB iPhone .MOVs unusable.

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
