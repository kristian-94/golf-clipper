"""
Auto-trim golf clips around the ball-strike impact detected in audio.

Strategy:
  1. Extract mono audio with ffmpeg at 48 kHz.
  2. Compute a short-window energy envelope restricted to the 2-8 kHz band
     (where a clean ball strike has its strongest, sharpest transient).
  3. Score candidate impact times by: (a) HF envelope peak, (b) attack
     sharpness (rise over ~10 ms), (c) quietness of the 200 ms preceding it.
  4. Pick the best candidate, then trim the video from
     (impact - pre_seconds) to (impact + post_seconds) via ffmpeg.

Usage:
  uv run --with numpy --with scipy --with soundfile \
    python impact_trim.py INPUT.MOV [OUTPUT.MP4] \
        [--pre 1.5] [--post 4.0] [--dry-run] [--debug]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


SR = 48_000
HOP_MS = 5          # envelope hop
WIN_MS = 10         # envelope window (short → preserves transients)
ATTACK_MS = 10      # how fast the transient rises
PRE_QUIET_MS = 200  # quietness window before the strike


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True)


def extract_audio(src: Path, sr: int = SR) -> np.ndarray:
    """Decode src to mono float32 PCM at sr via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        run([
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-ac", "1", "-ar", str(sr), "-f", "wav",
            str(wav_path),
        ])
        data, got_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        assert got_sr == sr, f"ffmpeg gave {got_sr} Hz, wanted {sr}"
        return data
    finally:
        wav_path.unlink(missing_ok=True)


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def envelope(x: np.ndarray, sr: int, win_ms: float = WIN_MS, hop_ms: float = HOP_MS) -> tuple[np.ndarray, float]:
    """RMS envelope. Returns (env, hop_seconds)."""
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    # Simple framing
    n_frames = 1 + (len(x) - win) // hop
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32), hop / sr
    idx = np.arange(n_frames) * hop
    frames = np.stack([x[i : i + win] for i in idx])
    env = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return env.astype(np.float32), hop / sr


@dataclass
class Candidate:
    time_s: float
    amp: float
    attack: float       # rise ratio over ATTACK_MS
    pre_quiet: float    # ratio of peak to pre-window median
    score: float


# Thresholds for confidence assessment. Tuned from a 10-clip test set where
# real strikes had attack 7-70× and scores 20-160, while false positives
# (voice, soft putt) had attack ~1.1× and scores < 5.
MIN_STRIKE_ATTACK = 3.0
MIN_STRIKE_SCORE = 5.0

# Cluster warning: another candidate scoring ≥ CLUSTER_SCORE_RATIO of the
# top within ±CLUSTER_WINDOW_S suggests repetitive noise (bird, rattle)
# rather than an isolated ball strike.
CLUSTER_WINDOW_S = 4.0
CLUSTER_SCORE_RATIO = 0.3


@dataclass
class Assessment:
    confidence: str          # "strong" | "ok" | "weak"
    cluster: bool            # multiple strike-like candidates nearby
    reasons: list[str]

    @property
    def flagged(self) -> bool:
        return self.confidence == "weak" or self.cluster


def assess(cands: list[Candidate]) -> Assessment:
    reasons: list[str] = []
    if not cands:
        return Assessment("weak", False, ["no candidates"])
    top = cands[0]

    if top.attack < MIN_STRIKE_ATTACK:
        reasons.append(f"soft attack ({top.attack:.1f}× < {MIN_STRIKE_ATTACK}×)")
    if top.score < MIN_STRIKE_SCORE:
        reasons.append(f"low score ({top.score:.1f} < {MIN_STRIKE_SCORE})")

    cluster_peers = [
        c for c in cands[1:]
        if abs(c.time_s - top.time_s) <= CLUSTER_WINDOW_S
        and c.score >= top.score * CLUSTER_SCORE_RATIO
        and c.attack >= MIN_STRIKE_ATTACK
    ]
    cluster = len(cluster_peers) >= 1
    if cluster:
        peer_times = ", ".join(f"{c.time_s:.2f}s" for c in cluster_peers)
        reasons.append(
            f"{len(cluster_peers)} strike-like peer(s) within ±{CLUSTER_WINDOW_S}s "
            f"[{peer_times}] — repetitive noise?"
        )

    if top.attack < MIN_STRIKE_ATTACK or top.score < MIN_STRIKE_SCORE:
        confidence = "weak"
    elif top.score < 20:
        confidence = "ok"
    else:
        confidence = "strong"
    return Assessment(confidence, cluster, reasons)


def find_impacts(audio: np.ndarray, sr: int, top_k: int = 8) -> list[Candidate]:
    """Return the top_k ranked candidate impact times."""
    hf = bandpass(audio, sr, 2000, 8000)
    env, hop = envelope(np.abs(hf), sr)

    # Peak picking: local max with minimum separation
    min_gap = int(0.25 / hop)  # ≥250 ms between candidates
    peaks: list[int] = []
    if env.size:
        # Dynamic threshold: 3× running median (robust to loud voice)
        med = np.median(env)
        thresh = max(med * 3.0, env.max() * 0.15)
        i = 1
        while i < env.size - 1:
            if env[i] >= thresh and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
                if not peaks or (i - peaks[-1]) >= min_gap:
                    peaks.append(i)
                else:
                    # Keep the taller one in the gap
                    if env[i] > env[peaks[-1]]:
                        peaks[-1] = i
            i += 1

    attack_bins = max(1, int(ATTACK_MS / 1000 / hop))
    pre_bins = max(1, int(PRE_QUIET_MS / 1000 / hop))

    cands: list[Candidate] = []
    for p in peaks:
        amp = float(env[p])
        lo = max(0, p - attack_bins)
        attack = amp / (float(env[lo]) + 1e-6)
        pre_start = max(0, p - pre_bins - attack_bins)
        pre_end = max(pre_start + 1, p - attack_bins)
        pre_med = float(np.median(env[pre_start:pre_end])) + 1e-6
        pre_quiet = amp / pre_med
        # Score rewards sharp attack + silence before + raw loudness
        score = amp * attack * pre_quiet
        cands.append(Candidate(
            time_s=p * hop,
            amp=amp,
            attack=attack,
            pre_quiet=pre_quiet,
            score=score,
        ))

    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:top_k]


def trim_video(src: Path, dst: Path, start: float, duration: float) -> None:
    start = max(0.0, start)
    # Re-encode for frame-accurate trim. HEVC source → H.264 output for broad editor compatibility.
    run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path, nargs="?")
    ap.add_argument("--pre", type=float, default=1.5, help="seconds before impact")
    ap.add_argument("--post", type=float, default=4.0, help="seconds after impact")
    ap.add_argument("--dry-run", action="store_true", help="just print detection, don't trim")
    ap.add_argument("--debug", action="store_true", help="print top candidates as JSON")
    ap.add_argument("--skip-flagged", action="store_true",
                    help="don't write a trim if confidence is weak or a cluster is detected")
    args = ap.parse_args()

    src: Path = args.input
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 2

    audio = extract_audio(src)
    cands = find_impacts(audio, SR)
    if not cands:
        print("no impact-like transient detected", file=sys.stderr)
        return 1

    if args.debug:
        print(json.dumps(
            [c.__dict__ for c in cands],
            indent=2, default=float,
        ))

    best = cands[0]
    verdict = assess(cands)
    print(f"impact @ {best.time_s:.3f}s  "
          f"amp={best.amp:.3f} attack={best.attack:.1f}× preQuiet={best.pre_quiet:.1f}× "
          f"score={best.score:.3f}  [{verdict.confidence}]")
    for r in verdict.reasons:
        print(f"  ! {r}")

    if args.dry_run:
        return 0

    if args.skip_flagged and verdict.flagged:
        print("skipped (flagged)")
        return 0

    start = best.time_s - args.pre
    duration = args.pre + args.post
    dst = args.output or src.with_name(src.stem + "_shot.mp4")
    trim_video(src, dst, start, duration)
    print(f"wrote {dst}  ({start:.3f} → {start + duration:.3f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
