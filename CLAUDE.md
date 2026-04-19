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

## Preferences Kristian has voiced for this project

- **Triage speed beats fine-grained editing.** Picking the strike moment is
  the only meaningful decision per clip; pre/post stays at config.
- **Never make him wait.** Re-renders are fire-and-forget; the UI advances
  while ffmpeg works in the background.
- **One-click flow through the queue.** Save = save + render + approve +
  jump to the next problem clip, no detour through the grid.
- **Keep raw files untouched.** All outputs land in sibling folders inside
  the round directory.
