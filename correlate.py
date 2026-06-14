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
    """First hole (in play order) whose boundary >= recorded_at; falls back to
    the last hole played.

    Holes are walked in score-entry chronological order, not by hole number, so
    shotgun / back-nine-first rounds correlate correctly. (A clip taken during a
    hole has recorded_at <= that hole's score-entry time, so it maps to the
    earliest hole not yet scored when it was filmed.)"""
    holes = sorted(boundaries, key=lambda h: (boundaries[h], h))
    for h in holes:
        if recorded_at <= boundaries[h]:
            return h
    return holes[-1]


def leaderboard_order(scores: dict) -> str:
    """How the overlay leaderboard accumulates running totals.

    "time"   — sum the holes each player finished *before* the target hole was
               played (correct for shotgun / back-nine-first rounds).
    "number" — sum holes with a lower number. Used in gaps mode, where score
               timestamps are clustered and can't establish play order.
    """
    return "number" if scores.get("assignment_mode") == "gaps" else "time"


def players_through(scores: dict, hole: int, order: str = "number") -> list[dict]:
    """Each player's running total going **into** `hole`.

    `order` decides what "before this hole" means:
      * "number" — holes 1..hole-1 (default; assumes numeric play order).
      * "time"   — holes the player finished before they reached `hole`, keyed
                   on score-entry time. Correct when holes aren't played in
                   numeric order (back-nine-first, shotgun starts).

    `par_through` is the cumulative par of those same holes, so the overlay
    can render `total − par_through` as a ±par column on each row.

    A player with `played_through < hole` is omitted entirely — used when a
    playing partner left mid-round (e.g. only played the front 9), so their
    name doesn't appear on the leaderboard for later holes.
    """
    par_by_hole = {h["number"]: h["par"] for h in scores["holes"]}
    # In time order, work out which holes were played before the target hole
    # from the field-wide score-entry time per hole. Using the max across
    # players (hole_boundaries) is robust to a single mis-timed entry — far more
    # reliable than any one player's own timestamp, which can be logged late or
    # out of order.
    earlier_holes: set[int] | None = None
    if order == "time":
        boundaries = hole_boundaries(scores)
        ref = boundaries.get(hole)
        earlier_holes = {
            h for h, t in boundaries.items() if ref is not None and t < ref
        }
    out = []
    for p in scores["players"]:
        played_through = p.get("played_through")
        if played_through is not None and played_through < hole:
            continue
        total = 0
        par_through = 0
        for s in p["scores"]:
            if s.get("strokes") is None:
                continue
            if earlier_holes is not None:
                if s["hole"] not in earlier_holes:
                    continue
            elif s["hole"] >= hole:
                continue
            total += s["strokes"]
            par_through += par_by_hole.get(s["hole"], 0)
        out.append({
            "name": p["name"],
            "is_owner": p.get("is_owner", False),
            "total": total,
            "par_through": par_through,
        })
    return out


def correlate_by_gaps(
    scores: dict, sidecars: dict[str, dict]
) -> dict[str, dict]:
    """Assign hole/shot using clip recorded_at + scorecard stroke counts only.

    Use this when `score.created_at` is unreliable (e.g. user logged scores
    long after the round, so the timestamps cluster). We sort clips
    chronologically, walk holes 1..18 in order, and greedily eat
    `strokes_for_hole_h` clips per hole. The first segment defaults to the
    first hole that has strokes recorded.

    The result is a *first guess*; every assignment is flagged with
    `correlate_warning` so the review UI knows to surface it.
    """
    owner = next((p for p in scores["players"] if p.get("is_owner")), None)
    if not owner and scores["players"]:
        owner = scores["players"][0]
    owner_strokes: dict[int, int] = {}
    if owner:
        owner_strokes = {
            s["hole"]: s["strokes"]
            for s in owner["scores"]
            if s.get("strokes") is not None
        }
    par_by_hole = {h["number"]: h["par"] for h in scores["holes"]}
    holes_in_order = [h for h in sorted(par_by_hole) if owner_strokes.get(h)]

    # Chronological order. Clips without recorded_at land at the end and are
    # surfaced as errors so the user can fix the sidecar (uncommon).
    items = sorted(
        sidecars.items(),
        key=lambda kv: (
            kv[1].get("recorded_at") is None,
            kv[1].get("recorded_at") or "",
        ),
    )

    out: dict[str, dict] = {}
    queue = list(items)
    for h in holes_in_order:
        strokes = owner_strokes[h]
        for i in range(strokes):
            if not queue:
                break
            stem, data = queue.pop(0)
            if not data.get("recorded_at"):
                out[stem] = {"hole": None, "correlate_error": "no recorded_at"}
                continue
            out[stem] = {
                "hole": h,
                "par": par_by_hole.get(h),
                "players": players_through(scores, h),
                "shot_index": i + 1,
                "shot_total": strokes,
                "correlate_warning": "auto-assigned by stroke count — review",
            }
        if not queue:
            break

    # Anything past hole 18's stroke total spills into the last hole — better
    # than dropping the assignment entirely. The review UI is where these get
    # redistributed.
    if queue and holes_in_order:
        last = holes_in_order[-1]
        last_strokes = owner_strokes[last]
        for j, (stem, data) in enumerate(queue, last_strokes + 1):
            if not data.get("recorded_at"):
                out[stem] = {"hole": None, "correlate_error": "no recorded_at"}
                continue
            out[stem] = {
                "hole": last,
                "par": par_by_hole.get(last),
                "players": players_through(scores, last),
                "shot_index": j,
                "shot_total": j,
                "correlate_warning": "overflow past last hole — review",
            }

    return out


def cascade_from(
    scores: dict, sidecars: dict[str, dict], anchor_stem: str
) -> dict[str, dict]:
    """Re-walk same-player clips chronologically after `anchor_stem` so they
    pick up where the user's manual assignment left off.

    Use case: the user is reviewing /assign in chronological order. They fix
    a clip's hole (it should have been Hole 6, not Hole 5) — every later clip
    of the same player should also shift forward.

    Per-player attribution: each clip's `player` (default = round owner)
    determines whose stroke counts the cascade walks. Only clips matching
    the anchor's player are updated; others-player clips are left alone.

    Returns {stem: {hole, par, players, shot_index, shot_total}} for clips
    that should change. The anchor itself is excluded — the caller's manual
    setting stays authoritative.
    """
    anchor = sidecars.get(anchor_stem)
    if not anchor:
        return {}
    anchor_hole = anchor.get("hole")
    anchor_shot = anchor.get("shot_index")
    anchor_at = anchor.get("recorded_at")
    if not anchor_hole or not anchor_shot or not anchor_at:
        return {}

    owner = next((p for p in scores["players"] if p.get("is_owner")), None)
    if not owner and scores["players"]:
        owner = scores["players"][0]
    owner_name = owner["name"] if owner else None
    anchor_player = anchor.get("player") or owner_name
    if not anchor_player:
        return {}

    player_obj = next(
        (p for p in scores["players"] if p["name"] == anchor_player), None
    )
    if not player_obj:
        return {}
    strokes_by_hole = {
        s["hole"]: s["strokes"]
        for s in player_obj.get("scores", [])
        if s.get("strokes") is not None
    }
    par_by_hole = {h["number"]: h["par"] for h in scores["holes"]}
    holes_in_order = sorted(par_by_hole)

    after = sorted(
        (
            (stem, data)
            for stem, data in sidecars.items()
            if stem != anchor_stem
            and (data.get("player") or owner_name) == anchor_player
            and data.get("recorded_at")
            and data["recorded_at"] >= anchor_at
        ),
        key=lambda kv: kv[1]["recorded_at"],
    )

    cursor_hole: int | None = anchor_hole
    cursor_shot = anchor_shot + 1

    def advance_to_valid() -> None:
        """Bump (hole, shot) forward until shot ≤ strokes_for_hole. Returns
        with cursor_hole = None when there's nothing left for this player."""
        nonlocal cursor_hole, cursor_shot
        while cursor_hole is not None:
            if cursor_shot <= strokes_by_hole.get(cursor_hole, 0):
                return
            nexts = [h for h in holes_in_order if h > cursor_hole and strokes_by_hole.get(h)]
            if not nexts:
                cursor_hole = None
                return
            cursor_hole = nexts[0]
            cursor_shot = 1

    out: dict[str, dict] = {}
    for stem, _ in after:
        advance_to_valid()
        if cursor_hole is None:
            break
        out[stem] = {
            "hole": cursor_hole,
            "par": par_by_hole.get(cursor_hole),
            "players": players_through(scores, cursor_hole, order=leaderboard_order(scores)),
            "shot_index": cursor_shot,
            "shot_total": strokes_by_hole.get(cursor_hole, 0),
        }
        cursor_shot += 1
    return out


def renumber_player_chrono(
    scores: dict, sidecars: dict[str, dict], player_name: str
) -> dict[str, dict]:
    """Within each hole, rerank `player_name`'s clips by recorded_at and
    assign shot_index = chrono rank (1-based).

    Eliminates duplicate shot indexes (the rule "you can't have two 2nd
    shots") by deferring to the time of recording. Idempotent — runs after
    every manual assignment so the visible shot numbers always reflect the
    chronology even if the user manually edits them out of order.

    Returns {stem: {shot_index, shot_total}} for clips whose values change.
    """
    owner = next((p for p in scores["players"] if p.get("is_owner")), None)
    if not owner and scores["players"]:
        owner = scores["players"][0]
    owner_name = owner["name"] if owner else None

    player_obj = next(
        (p for p in scores["players"] if p["name"] == player_name), None
    )
    strokes_by_hole = {
        s["hole"]: s["strokes"]
        for s in (player_obj.get("scores", []) if player_obj else [])
        if s.get("strokes") is not None
    }

    by_hole: dict[int, list[tuple[str, dict]]] = {}
    for stem, data in sidecars.items():
        if data.get("hole") is None:
            continue
        if (data.get("player") or owner_name) != player_name:
            continue
        if not data.get("recorded_at"):
            continue
        by_hole.setdefault(data["hole"], []).append((stem, data))

    out: dict[str, dict] = {}
    for hole, clips in by_hole.items():
        # Split sub-clips share recorded_at (inherited from parent MOV
        # metadata). Tiebreak by impact_s — the strike moment within the
        # raw video — so sub-clips order correctly within their hole.
        clips.sort(key=lambda kv: (kv[1]["recorded_at"], kv[1].get("impact_s") or 0.0))
        total = strokes_by_hole.get(hole, len(clips))
        for i, (stem, data) in enumerate(clips):
            new_shot = i + 1
            if (
                data.get("shot_index") != new_shot
                or data.get("shot_total") != total
            ):
                out[stem] = {"shot_index": new_shot, "shot_total": total}
    return out


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
            "players": players_through(scores, h, order=leaderboard_order(scores)),
        }

    # Phase 2: per-player shot tally. Group a hole's clips by subject player
    # (sidecar `player`, default owner) and number them only when that player's
    # clip count equals their stroke count — so a mixed-camera hole numbers each
    # golfer's shots against their own score, not against the owner's.
    owner = next((p for p in scores["players"] if p.get("is_owner")), None)
    owner_name = (
        owner["name"] if owner
        else (scores["players"][0]["name"] if scores["players"] else None)
    )
    strokes_by_player = {
        p["name"]: {
            s["hole"]: s["strokes"]
            for s in p.get("scores", [])
            if s.get("strokes") is not None
        }
        for p in scores["players"]
    }
    groups: dict[tuple[int, str], list[tuple[str, datetime]]] = {}
    for stem, h in assigned.items():
        subject = sidecars[stem].get("player") or owner_name
        ts = sidecars[stem]["recorded_at"]
        groups.setdefault((h, subject), []).append((stem, _parse(ts)))
    for (h, subject), items in groups.items():
        expected = strokes_by_player.get(subject, {}).get(h)
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


def tag_players_from_devices(scores: dict, sidecars: dict[str, dict]) -> int:
    """Set each clip's subject `player` from its `device_model`, using the
    `device_subjects` map in scores.json (model → the player that phone films).

    Non-clobbering: only clips with no `player` yet are tagged, so manual
    re-assignments on the /assign page survive a re-correlate. Returns the
    number of clips newly tagged. A no-op when no map is configured.
    """
    device_subjects = scores.get("device_subjects") or {}
    if not device_subjects:
        return 0
    valid = {p["name"] for p in scores.get("players", [])}
    n = 0
    for sc in sidecars.values():
        if sc.get("player"):
            continue
        subject = device_subjects.get(sc.get("device_model"))
        if subject and subject in valid:
            sc["player"] = subject
            sc["player_source"] = "device"
            n += 1
    return n


def apply_to_sidecars(
    round_dir: Path, fetch_map: bool = True, mode: str | None = None
) -> dict:
    """Read scores.json + meta/*.json, write hole info back. Returns summary.

    `mode` selects the correlator:
      * "timestamps" — use score.created_at (default; needs SmartCaddy live logging)
      * "gaps"       — use clip recorded_at + scorecard strokes (manual scorecard)

    If `mode` is None we read it from scores.json's `assignment_mode` field,
    falling back to "timestamps". The chosen mode is persisted to scores.json
    so subsequent runs (server restarts) keep the same behaviour without
    needing the CLI flag.

    In "gaps" mode we only auto-assign clips that don't already have a hole;
    existing assignments are preserved but their `players` row is refreshed
    so leaderboard updates from a re-fetched scorecard still flow through.

    When `fetch_map` is True (default), also fetches course geometry from OSM
    once and caches it as `<round>/course_geom.json` so the next render bakes
    a hole map into the overlay. Set False to skip — e.g. for a course that
    isn't well-mapped, the rest of the pipeline still works.
    """
    scores_path = round_dir / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} — run smartcaddy.py first")
    scores = json.loads(scores_path.read_text())

    if mode is None:
        mode = scores.get("assignment_mode", "timestamps")
    if mode not in ("timestamps", "gaps"):
        raise ValueError(f"unknown correlate mode: {mode}")

    # Persist the chosen mode so the server boots into the same regime next time.
    if scores.get("assignment_mode") != mode:
        scores["assignment_mode"] = mode
        scores_path.write_text(json.dumps(scores, indent=2))

    meta_dir = round_dir / "meta"
    sidecars = {p.stem: json.loads(p.read_text()) for p in sorted(meta_dir.glob("*.json"))}

    # Phone-of-origin → subject player (only clips not already tagged).
    tag_players_from_devices(scores, sidecars)

    par_by_hole = {h["number"]: h["par"] for h in scores["holes"]}

    if mode == "gaps":
        # Only assign clips that don't yet have a hole. Anything the user
        # already touched (manually or by a prior auto-assign) is preserved.
        unassigned = {k: v for k, v in sidecars.items() if v.get("hole") is None}
        correlations = correlate_by_gaps(scores, unassigned)
        # For already-assigned clips, refresh `par` + `players` against the
        # current scorecard but leave hole/shot_index/shot_total intact.
        for stem, sc in sidecars.items():
            if stem in correlations:
                continue
            h = sc.get("hole")
            if h is None:
                continue
            correlations[stem] = {
                "hole": h,
                "par": par_by_hole.get(h),
                "players": players_through(scores, h),
                "shot_index": sc.get("shot_index"),
                "shot_total": sc.get("shot_total"),
                # preserve the existing warning if any
                **({"correlate_warning": sc["correlate_warning"]}
                   if sc.get("correlate_warning") else {}),
            }
    else:
        correlations = correlate(scores, sidecars)

    n_ok = n_flagged = 0
    for stem, sc in sidecars.items():
        c = correlations.get(stem) or {"hole": None, "correlate_error": "no correlation"}
        # Replace any prior correlation block, leave other fields intact.
        for k in (
            "hole", "par", "players", "shot_index", "shot_total",
            "correlate_error", "correlate_warning",
        ):
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
    ap.add_argument("--mode", choices=("timestamps", "gaps"),
                    help="how to assign clips to holes. `timestamps` uses "
                         "score.created_at (SmartCaddy live logging). `gaps` "
                         "uses clip recorded_at + scorecard strokes — pick "
                         "this when the scorecard was entered after the round.")
    args = ap.parse_args()
    summary = apply_to_sidecars(args.round, fetch_map=not args.no_map, mode=args.mode)
    print(f"correlated: {summary['correlated']}  uncorrelatable: {summary['uncorrelatable']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
