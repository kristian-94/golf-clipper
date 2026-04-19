# Golf clipper

A little tool that turns a phone full of golf clips into a tidy set of short
videos — one per shot — ready to drop into iMovie.

After a round of golf I usually come home with about 60 clips. Most are 10–20
seconds long but the only interesting moment in each is a single golf swing.
Trimming them by hand is tedious, so this does it for me.

![The review grid](docs/screenshot-grid.png)

## What it does

1. **Listens** to each clip and finds the loudest *thwack* — that's the moment
   the club hits the ball.
2. **Trims** the video down to a few seconds either side of that moment.
3. **Looks up the scorecard** from [SmartCaddy](https://smartcaddy.io) (the
   golf scoring app I help maintain), figures out which hole each clip belongs
   to based on the timestamps, and **bakes a little scorecard overlay** into
   the corner of the video — so when you watch the clip later you can see
   "Hole 7, Par 4" and the running totals for everyone in the group.
4. **Opens a web page** showing every trimmed clip on a grid. I scan through,
   click anything that doesn't look right (wrong moment caught, dud clip,
   etc.), then click "Approve all" on the rest. Bad clips can be sent to a
   trash folder. For the ones that need a tweak, there's a side-by-side editor
   where I scrub to the actual moment of impact and hit Save.

The raw video files are never modified — everything new lands in sibling
folders inside the round directory.

## How a round is organised

```
clips/
  18-april-2026/
    raw-18-april-2026/   ← the original clips off my phone (untouched)
    trims/               ← the short trimmed videos with overlays baked in
    meta/                ← one tiny JSON file per clip with detection info
    trash/               ← anywhere a discarded clip ends up (recoverable)
    scores.json          ← the scorecard data pulled from SmartCaddy
```

## What's where in the code

- **`impact_trim.py`** — the audio analysis that finds the ball-strike moment
- **`batch_trim.py`** — runs detection over every clip in a round, then
  renders the trimmed videos
- **`overlay.py`** — draws the scorecard card and runs the ffmpeg command that
  trims and overlays in a single pass
- **`smartcaddy.py`** — fetches a round's scores from the SmartCaddy API
- **`correlate.py`** — matches each clip to a hole using the score timestamps
- **`server.py`** + **`web/`** — the FastAPI web app for reviewing the clips

## Running it

You'll need [`uv`](https://docs.astral.sh/uv/) and `ffmpeg` installed.

```sh
# Detect + trim every clip in the newest round under clips/
uv run batch_trim.py

# (Optional) pull the scorecard from SmartCaddy and re-render with overlays
uv run smartcaddy.py --share-token <token> --round clips/18-april-2026
uv run correlate.py --round clips/18-april-2026
uv run batch_trim.py --round clips/18-april-2026 --re-render

# Open the review UI
uv run server.py
# → http://127.0.0.1:8000
```

## What it explicitly doesn't do

- **Final editing.** Sequencing the clips, adding ball-tracer effects, music,
  titles — all of that happens in iMovie afterwards.
- **Identify the player.** It assumes clips are of me. If a clip is actually
  of someone else in the group the scorecard overlay still works, but the
  shot counter (e.g. "SHOT 2/4") may be off. Future me problem.
