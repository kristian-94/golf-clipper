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

# Caddly retired the REST backend; the scorecard now lives behind an MCP
# server (Streamable HTTP, Bearer PAT auth). Override with SMARTCADDY_MCP_URL.
MCP_URL = os.environ.get("SMARTCADDY_MCP_URL", "https://api.caddly.golf/mcp")
APP_VERSION = "2026.5.21"  # Caddly MCP rejects versions older than this (HTTP 426)


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


def _mcp_result(body: str) -> dict:
    """Extract the JSON-RPC result from an MCP Streamable-HTTP response.

    Responses come back as Server-Sent Events (`data: {...}` lines); a plain
    JSON body is tolerated too in case the server stops streaming.
    """
    payload = None
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = json.loads(line[5:].lstrip())
    if payload is None:
        payload = json.loads(body)
    if payload.get("error"):
        raise RuntimeError(f"MCP error: {payload['error']}")
    return payload["result"]


class _Mcp:
    """A single MCP session: the initialize handshake plus tool calls."""

    def __init__(self, token: str):
        self._c = httpx.Client(timeout=30.0, headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "X-App-Version": APP_VERSION,
            "Authorization": f"Bearer {token}",
        })
        r = self._c.post(MCP_URL, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "golf-clipper", "version": "1.0"}},
        })
        r.raise_for_status()
        sid = r.headers.get("mcp-session-id")
        if sid:
            self._c.headers["mcp-session-id"] = sid
        self._c.post(MCP_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    def tool(self, name: str, arguments: dict | None = None) -> dict:
        r = self._c.post(MCP_URL, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        r.raise_for_status()
        result = _mcp_result(r.text)
        return json.loads(result["content"][0]["text"])

    def close(self) -> None:
        self._c.close()


def fetch_by_share_token(share_token: str, token: str | None = None) -> dict:
    raise NotImplementedError(
        "Caddly retired the share-token REST endpoint. Fetch by --round-id "
        "against the MCP server instead."
    )


def _scorecard_to_raw(sc: dict, owner_username: str | None) -> dict:
    """Reshape an MCP get_scorecard payload into the REST-era round shape that
    normalize() expects (roundHoles + playerSessions)."""
    pars: dict[int, int] = {}
    sessions: list[dict] = []
    for p in sc.get("players", []):
        name = p.get("name") or "player"
        is_owner = bool(owner_username) and name.lower() == owner_username.lower()
        scores = []
        for h in p.get("holes", []):
            if h.get("par") is not None:
                pars.setdefault(h["hole"], h["par"])
            scores.append({
                "holeNumber": h["hole"],
                "strokes": h.get("strokes"),
                "putts": h.get("putts"),
                "createdAt": h.get("createdAt"),
            })
        sessions.append({
            "id": name,
            "isActive": True,
            "userId": owner_username if is_owner else None,
            "user": {"username": name},
            "scores": scores,
        })
    return {
        "id": sc.get("roundId"),
        "name": sc.get("course"),
        "shareToken": None,
        "userId": owner_username,
        "roundHoles": [{"holeNumber": n, "par": pars[n]} for n in sorted(pars)],
        "playerSessions": sessions,
    }


def fetch_by_round_id(round_id: str, token: str) -> dict:
    mcp = _Mcp(token)
    try:
        scorecard = mcp.tool("get_scorecard", {"roundId": round_id})
        try:
            owner = (mcp.tool("get_player_stats") or {}).get("username")
        except Exception:
            owner = None  # owner detection is best-effort; correlation still works
    finally:
        mcp.close()
    return _scorecard_to_raw(scorecard, owner)


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
    # Preserve user-set fields from the existing file: `assignment_mode` (set
    # by the gap correlator) and per-player `played_through` (set by the UI
    # when a partner left mid-round). Re-fetch should refresh strokes from
    # SmartCaddy without clobbering manual roster overrides.
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if "assignment_mode" in existing:
            data["assignment_mode"] = existing["assignment_mode"]
        if "players_locked" in existing:
            data["players_locked"] = existing["players_locked"]
        if "holes_locked" in existing:
            data["holes_locked"] = existing["holes_locked"]
        prior_played: dict[str, int] = {}
        for p in existing.get("players", []):
            if "played_through" in p and p.get("played_through") is not None:
                prior_played[p.get("name")] = p["played_through"]
        for p in data["players"]:
            if p["name"] in prior_played:
                p["played_through"] = prior_played[p["name"]]
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
