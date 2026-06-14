# Golf clipper

A personal tool for tidying up phone-recorded clips from a round of golf into
ready-to-edit shot videos.

## Why it exists

After every round Kristian ends up with ~60 raw clips of varying length. He
wants each one trimmed down to just the moments around the ball strike. Final
sequencing and any ball-tracer overlays happen in iMovie afterwards — **out
of scope here.**

## Workflow this tool supports

1. **Detect & batch** — for each raw clip, find the ball-strike moment from
   audio and trim a window around it. Each clip gets a sidecar with
   confidence + flags.
2. **Triage** — open the web UI, scan the grid of looping trimmed clips,
   click any that look wrong to mark them "needs fix," then bulk-approve
   the rest in one go.
3. **Fix** — for each marked clip, scrub the raw to the strike moment and
   tap Space. Save → re-render in background → auto-advance to the next.
4. **Discard** duds at any point — they get moved to a trash folder under
   the round, recoverable but out of the UI.

The detector's job is to be right *most* of the time so triage is fast.

## Folder model

One folder per round, named by date (e.g. `clips/18-april-2026/`). Each
round folder holds the raw clips, the rendered trims, the per-clip metadata
sidecars, and a trash subfolder. The server defaults to the newest round.

## Starting the app for a new round

When Kristian says "start the app" or "process the next round", just do all
of this — don't ask:

1. **Use the right Python.** `/opt/homebrew/opt/python@3.11/bin/python3.11`.
   Default `python3` is 3.13 and lacks Pillow/fastapi/etc. Don't try to
   `pip install` anything on 3.13 — wrong interpreter.
2. **Start via `python server.py`, not `uvicorn server:app`.** Uvicorn
   directly bypasses `main()`, leaves `STATE` empty, and every API call
   500s with `KeyError: 'raw'`.
3. **Always pass `--round-id` so the scorecard gets pulled** (powers
   the hole overlay + per-shot timing). The backend is now **Caddly**
   (the old SmartCaddy/Railway REST API is dead) — scores live behind the
   Caddly MCP server at `https://api.caddly.golf/mcp`, Bearer-auth with the
   PAT still stored in `.env` as `SMARTCADDY_TOKEN`. `smartcaddy.py` already
   targets this endpoint. To find today's round id:
   - The companion repo `~/projects/personal/green-jacket` mirrors every
     round to `scores/index.json`, keyed by date with its Caddly `roundId`.
     Read that file and pick the entry whose `date` matches the newest
     `clips/<date>/` folder — that's the fastest path.
   - Or call the Caddly MCP `get_round_history` tool directly (see
     green-jacket's CLAUDE.md for the MCP details) and match by date.
4. **If the scorecard was entered after the round, also pass
   `--no-shot-times`.** SmartCaddy stores `created_at` on each score; the
   default correlator uses that timestamp as a hole boundary. When
   Kristian taps in scores from a paper card after the round, every
   `created_at` clusters and the timestamp correlator collapses every
   clip onto one hole. `--no-shot-times` switches to the gap-based
   correlator: clips are walked chronologically, eaten greedily against
   the scorecard's per-hole stroke counts, and flagged for review in the
   "Review assignments" page in the web UI. The mode is persisted to
   `scores.json` so subsequent restarts don't need the flag.
5. **Startup takes ~60–90s before binding the port** when the OSM
   Overpass course-map fetch has to time out across all three mirrors.
   That's normal, not a hang — wait it out. Correlation still succeeds
   even when the map fetch fails.

Server lifecycle is on me — manage uvicorn via background bash tasks,
never tell Kristian to restart it.

## Preferences Kristian has voiced for this project

- **Triage speed beats fine-grained editing.** Picking the strike moment is
  the only meaningful decision per clip; pre/post stays at config.
- **Never make him wait.** Re-renders are fire-and-forget; the UI advances
  while ffmpeg works in the background.
- **One-click flow through the queue.** Save = save + render + approve +
  jump to the next problem clip, no detour through the grid.
- **Keep raw files untouched.** All outputs land in sibling folders inside
  the round directory.
