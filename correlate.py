# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "httpx"]
# ///
"""Assign each clip to a hole using SmartCaddy score timestamps.

Reads:  <round>/scores.json  (from smartcaddy.py)
        <round>/meta/*.json   (sidecars with recorded_at)
Writes: hole, par, players[], shot_index?, shot_total? back into each sidecar.

Hole boundary = max(score.created_at) across players for that hole — i.e. the
moment the slowest player logged the score, which is "after" everyone played
the hole. A clip with recorded_at <= boundary[h] (and > boundary[h-1])
belongs to hole h.

Usage:
    uv run correlate.py --round clips/18-april-2026
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from course_map import ACCURACY_THRESHOLD_M, fetch_course_geom


def _parse(ts: str) -> datetime:
    """Parse ISO-8601, including trailing Z, into an aware UTC datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def hole_boundaries(scores: dict) -> dict[int, datetime]:
    """For each hole, the latest createdAt across all players who logged a score."""
    out: dict[int, datetime] = {}
    for p in scores["players"]:
        for s in p["scores"]:
            t = _parse(s["created_at"])
            if t > out.get(s["hole"], datetime.min.replace(tzinfo=t.tzinfo)):
                out[s["hole"]] = t
    return out


def assign_hole(recorded_at: datetime, boundaries: dict[int, datetime]) -> int:
    """Smallest hole whose boundary >= recorded_at; falls back to last hole."""
    holes = sorted(boundaries)
    for h in holes:
        if recorded_at <= boundaries[h]:
            return h
    return holes[-1]


def players_through(scores: dict, hole: int) -> list[dict]:
    """Each player's running total **before** `hole` (i.e. holes 1..hole-1)."""
    return [
        {
            "name": p["name"],
            "is_owner": p.get("is_owner", False),
            "total": sum(s["strokes"] for s in p["scores"] if s["hole"] < hole),
        }
        for p in scores["players"]
    ]


def correlate(scores: dict, sidecars: dict[str, dict]) -> dict[str, dict]:
    """Return {stem: {hole, par, players, shot_index?, shot_total?}}.

    Stems with missing recorded_at are returned with hole=None and
    correlate_error set so callers can flag them.
    """
    boundaries = hole_boundaries(scores)
    par_by_hole = {h["number"]: h["par"] for h in scores["holes"]}

    # Phase 1: hole assignment
    assigned: dict[str, int] = {}
    out: dict[str, dict] = {}
    for stem, data in sidecars.items():
        ts = data.get("recorded_at")
        if not ts:
            out[stem] = {"hole": None, "correlate_error": "no recorded_at"}
            continue
        try:
            t = _parse(ts)
        except ValueError:
            out[stem] = {"hole": None, "correlate_error": f"bad recorded_at: {ts}"}
            continue
        h = assign_hole(t, boundaries)
        assigned[stem] = h
        out[stem] = {
            "hole": h,
            "par": par_by_hole.get(h),
            "players": players_through(scores, h),
        }

    # Phase 2: optional shot tally (only if owner's strokes for the hole == clip count)
    owner = next((p for p in scores["players"] if p.get("is_owner")), None)
    owner_strokes = (
        {s["hole"]: s["strokes"] for s in owner["scores"]} if owner else {}
    )
    by_hole: dict[int, list[tuple[str, datetime]]] = {}
    for stem, h in assigned.items():
        ts = sidecars[stem]["recorded_at"]
        by_hole.setdefault(h, []).append((stem, _parse(ts)))
    for h, items in by_hole.items():
        expected = owner_strokes.get(h)
        if expected and expected == len(items):
            items.sort(key=lambda x: x[1])
            for i, (stem, _) in enumerate(items, 1):
                out[stem]["shot_index"] = i
                out[stem]["shot_total"] = expected

    return out


def _seed_gps(sidecars: dict[str, dict]) -> tuple[float, float] | None:
    """Pick the most accurate GPS fix from any sidecar — used to seed the OSM query."""
    best: tuple[float, tuple[float, float]] | None = None
    for sc in sidecars.values():
        gps = sc.get("gps")
        acc = sc.get("gps_accuracy")
        if not gps or acc is None:
            continue
        if best is None or acc < best[0]:
            best = (acc, (gps[0], gps[1]))
    if best is None or best[0] > ACCURACY_THRESHOLD_M:
        return None
    return best[1]


def apply_to_sidecars(round_dir: Path, fetch_map: bool = True) -> dict:
    """Read scores.json + meta/*.json, write hole info back. Returns summary.

    When `fetch_map` is True (default), also fetches course geometry from OSM
    once and caches it as `<round>/course_geom.json` so the next render bakes
    a hole map into the overlay. Set False to skip — e.g. for a course that
    isn't well-mapped, the rest of the pipeline still works.
    """
    scores_path = round_dir / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} — run smartcaddy.py first")
    scores = json.loads(scores_path.read_text())

    meta_dir = round_dir / "meta"
    sidecars = {p.stem: json.loads(p.read_text()) for p in sorted(meta_dir.glob("*.json"))}

    correlations = correlate(scores, sidecars)

    n_ok = n_flagged = 0
    for stem, sc in sidecars.items():
        c = correlations[stem]
        # Replace any prior correlation block, leave other fields intact.
        for k in ("hole", "par", "players", "shot_index", "shot_total", "correlate_error"):
            sc.pop(k, None)
        sc.update(c)
        if c.get("hole") is None:
            n_flagged += 1
            sc["flagged"] = True
            reasons = sc.setdefault("reasons", [])
            msg = f"correlation: {c['correlate_error']}"
            if msg not in reasons:
                reasons.append(msg)
        else:
            n_ok += 1
        (meta_dir / f"{stem}.json").write_text(json.dumps(sc, indent=2))

    summary = {"correlated": n_ok, "uncorrelatable": n_flagged}

    if fetch_map:
        seed = _seed_gps(sidecars)
        if seed is None:
            print("[correlate] no clip with usable GPS — skipping course map fetch")
        else:
            geom = fetch_course_geom(round_dir, seed[0], seed[1])
            if geom:
                holes_in_osm = {
                    e["tags"].get("ref")
                    for e in geom["elements"]
                    if e.get("tags", {}).get("golf") == "hole"
                }
                print(f"[correlate] course geom cached — "
                      f"{len(holes_in_osm)} holes mapped in OSM")

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--no-map", action="store_true",
                    help="skip the OSM course-map fetch (use for unmapped courses)")
    args = ap.parse_args()
    summary = apply_to_sidecars(args.round, fetch_map=not args.no_map)
    print(f"correlated: {summary['correlated']}  uncorrelatable: {summary['uncorrelatable']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
