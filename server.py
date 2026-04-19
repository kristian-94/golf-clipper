# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi", "uvicorn", "pydantic",
#   "numpy", "scipy", "soundfile", "pillow",
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
from smartcaddy import fetch_and_save as smartcaddy_fetch, load_env

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI()
STATE: dict = {}
EXECUTOR = ThreadPoolExecutor(max_workers=2)
RENDER_STATUS: dict[str, str] = {}  # stem -> "pending" | "rendering" | "done" | "error: ..."


def setup_state(round_dir: Path) -> None:
    raw, trims, meta = round_paths(round_dir)
    STATE["round"] = round_dir
    STATE["raw"] = raw
    STATE["trims"] = trims
    STATE["meta"] = meta
    STATE["scores"] = round_dir / "scores.json"


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
            status = render_from_sidecar(meta_path, raw, trims)
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


def _render_job(stem: str) -> None:
    """Single-pass trim+overlay from sidecar. Overlay is baked in if hole info present."""
    try:
        RENDER_STATUS[stem] = "rendering"
        meta_path = STATE["meta"] / f"{stem}.json"
        status = render_from_sidecar(meta_path, STATE["raw"], STATE["trims"])
        RENDER_STATUS[stem] = "done" if status == "ok" else f"error: {status}"
    except Exception as e:
        RENDER_STATUS[stem] = f"error: {e}"


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
    queued = []
    for meta_path in sorted(STATE["meta"].glob("*.json")):
        stem = meta_path.stem
        RENDER_STATUS[stem] = "pending"
        asyncio.get_event_loop().run_in_executor(EXECUTOR, _render_job, stem)
        queued.append(stem)
    return {"queued": len(queued), "stems": queued}


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
