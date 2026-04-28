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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
EXECUTOR = ThreadPoolExecutor(max_workers=2)
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
            summary = correlate_round(STATE["round"])
            print(f"[batch] correlated {summary['correlated']} "
                  f"(uncorrelatable: {summary['uncorrelatable']})", flush=True)
        except Exception as e:
            print(f"[batch] correlate ERROR: {e}", flush=True)
    # Render any clip that doesn't yet have a trim. With correlation done above,
    # the overlay (if any) is baked in during this single pass.
    for meta_path in sorted(meta.glob("*.json")):
        stem = meta_path.stem
        if (trims / f"{stem}.mp4").exists():
            continue
        print(f"[batch] render {stem}", flush=True)
        try:
            RENDER_STATUS[stem] = "rendering"
            status = render_from_sidecar(meta_path, raw, trims, STATE["canvas"])
            RENDER_STATUS[stem] = "done" if status == "ok" else f"error: {status}"
        except Exception as e:
            RENDER_STATUS[stem] = f"error: {e}"
            print(f"[batch] render ERROR {stem}: {e}", flush=True)
    print("[batch] done", flush=True)


@app.on_event("startup")
async def _on_startup() -> None:
    asyncio.get_event_loop().run_in_executor(EXECUTOR, initial_batch)


# ---- HTML routes ----

@app.get("/")
async def index_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/clip/{stem}")
async def compare_page(stem: str):
    return FileResponse(WEB_DIR / "compare.html")


# ---- JSON API ----

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
    return info


def _read_meta(stem: str) -> dict | None:
    p = STATE["meta"] / f"{stem}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    data["stem"] = stem
    data["render_status"] = RENDER_STATUS.get(stem)
    data["has_trim"] = (STATE["trims"] / f"{stem}.mp4").exists()
    data.setdefault("review", "unreviewed")
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
    out = []
    for src in list_videos(STATE["raw"]):
        data = _read_meta(src.stem) or {
            "stem": src.stem,
            "raw": src.name,
            "confidence": "pending",
            "flagged": False,
            "edited": False,
            "has_trim": False,
            "render_status": None,
        }
        out.append(data)
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


@app.post("/api/clips/{stem}/review")
async def api_review_clip(stem: str, update: ReviewUpdate):
    if update.review not in ("unreviewed", "needs_fix", "approved"):
        raise HTTPException(400, "bad review value")
    _write_review(stem, update.review)
    return _read_meta(stem)


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

    if raw_src.exists():
        raw_src.rename(trash_root / "raw" / raw_name)
    if trim_src.exists():
        trim_src.rename(trash_root / "trims" / f"{stem}.mp4")
    meta_path.rename(trash_root / "meta" / f"{stem}.json")
    RENDER_STATUS.pop(stem, None)
    return {"discarded": stem}


@app.post("/api/review/approve_unreviewed")
async def api_approve_unreviewed():
    n = 0
    for src in list_videos(STATE["raw"]):
        meta_path = STATE["meta"] / f"{src.stem}.json"
        if not meta_path.exists():
            continue
        data = json.loads(meta_path.read_text())
        if data.get("review", "unreviewed") == "unreviewed":
            data["review"] = "approved"
            meta_path.write_text(json.dumps(data, indent=2))
            n += 1
    return {"approved": n}


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
        status = render_from_sidecar(
            meta_path, STATE["raw"], STATE["trims"], STATE["canvas"],
            pid_set=RENDER_PIDS,
        )
        ok = status == "ok"
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
    args = ap.parse_args()

    round_dir = args.round or newest_round(args.clips_root)
    setup_state(round_dir)
    print(f"serving round: {round_dir}", flush=True)

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
            summary = correlate_round(round_dir)
            print(f"[correlate] {summary['correlated']} clips correlated "
                  f"(uncorrelatable: {summary['uncorrelatable']})", flush=True)
        except Exception as e:
            print(f"[correlate] failed: {e}", flush=True)

    # Mount AFTER state is set so directories are known.
    app.mount("/raw", StaticFiles(directory=str(STATE["raw"])), name="raw")
    app.mount("/trims", StaticFiles(directory=str(STATE["trims"])), name="trims")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
