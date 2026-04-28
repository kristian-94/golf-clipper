# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""SmartCaddy API client + scorecard normalization.

Fetches a round (by shareToken or roundId) and writes a normalized
scores.json into the round folder. Used by correlate.py.

Usage:
    uv run smartcaddy.py --round clips/18-april-2026 --share-token TOKEN
    uv run smartcaddy.py --round clips/18-april-2026 --round-id ID
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

DEFAULT_API_BASE = "https://smartcaddy-backend-production.up.railway.app/api"
APP_VERSION = "2026.4.17"


def load_env(env_path: Path) -> None:
    """Minimal .env reader — populates os.environ for KEY=VALUE lines."""
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _client(token: str | None) -> httpx.Client:
    base = os.environ.get("SMARTCADDY_API_BASE", DEFAULT_API_BASE)
    headers = {"X-App-Version": APP_VERSION, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=base, headers=headers, timeout=15.0)


def fetch_by_share_token(share_token: str, token: str | None = None) -> dict:
    with _client(token) as c:
        r = c.get(f"/rounds/shared/{share_token}")
        r.raise_for_status()
        return r.json()


def fetch_by_round_id(round_id: str, token: str) -> dict:
    with _client(token) as c:
        r = c.get(f"/rounds/{round_id}")
        r.raise_for_status()
        return r.json()


def _player_name(ps: dict) -> str:
    if ps.get("guestName"):
        return ps["guestName"]
    user = ps.get("user") or {}
    name = user.get("username") or "player"
    return name[:1].upper() + name[1:]


def normalize(raw: dict) -> dict:
    """Reduce the SmartCaddy round payload to what the overlay needs.

    Output:
        {
          round_id, course, share_token, fetched_at,
          holes: [{number, par}, ...],
          players: [
            {session_id, name, is_owner,
             scores: [{hole, strokes, putts, created_at}, ...]},
            ...
          ]  # ordered: fewest total strokes first (leaderboard)
        }
    """
    hole_src = raw.get("roundHoles") or (raw.get("teeSelection") or {}).get("holes") or []
    pars = {h["holeNumber"]: h["par"] for h in hole_src}
    holes = [{"number": n, "par": pars[n]} for n in sorted(pars)]

    players: list[dict] = []
    for ps in raw.get("playerSessions", []):
        if not ps.get("isActive", True):
            continue
        scores = sorted(ps.get("scores", []), key=lambda s: s["holeNumber"])
        players.append({
            "session_id": ps["id"],
            "name": _player_name(ps),
            "is_owner": ps.get("userId") == raw.get("userId"),
            "scores": [
                {
                    "hole": s["holeNumber"],
                    "strokes": s["strokes"],
                    "putts": s.get("putts"),
                    "created_at": s["createdAt"],
                }
                for s in scores
            ],
        })

    players.sort(key=lambda p: sum(s["strokes"] for s in p["scores"]) or 10**9)

    return {
        "round_id": raw["id"],
        "course": raw.get("name") or (raw.get("course") or {}).get("name"),
        "share_token": raw.get("shareToken"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holes": holes,
        "players": players,
    }


def fetch_and_save(
    round_dir: Path,
    *,
    share_token: str | None = None,
    round_id: str | None = None,
    token: str | None = None,
) -> dict:
    """Fetch + normalize + persist to <round>/scores.json. Returns normalized dict."""
    if not (share_token or round_id):
        raise ValueError("share_token or round_id required")
    if round_id:
        if not token:
            raise ValueError("PAT required for round_id fetch")
        raw = fetch_by_round_id(round_id, token)
    else:
        raw = fetch_by_share_token(share_token, token)
    data = normalize(raw)
    out = round_dir / "scores.json"
    out.write_text(json.dumps(data, indent=2))
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=Path, required=True, help="round folder")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--share-token", help="SmartCaddy round share token")
    g.add_argument("--round-id", help="SmartCaddy round id (needs PAT)")
    args = ap.parse_args()

    load_env(Path(__file__).parent / ".env")
    token = os.environ.get("SMARTCADDY_TOKEN")

    try:
        data = fetch_and_save(
            args.round,
            share_token=args.share_token,
            round_id=args.round_id,
            token=token,
        )
    except httpx.HTTPStatusError as e:
        sys.exit(f"HTTP {e.response.status_code}: {e.response.text[:200]}")

    print(
        f"wrote {args.round / 'scores.json'}  "
        f"({len(data['players'])} players × {len(data['holes'])} holes, "
        f"course: {data['course']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
