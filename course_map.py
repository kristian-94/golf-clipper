# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "httpx"]
# ///
"""Course geometry from OpenStreetMap — fetches once per round, renders per clip.

Optional layer over the rest of the pipeline:
  - if `course_geom.json` is missing in the round dir, nothing breaks
  - if a hole isn't tagged in OSM, that clip just gets no map
  - if a clip's GPS is missing or too imprecise, the map renders without a dot

To disable for a course that isn't well-mapped, just don't call `fetch_course_geom`
(or delete the cached `course_geom.json`).
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

# --- Config -------------------------------------------------------------

ACCURACY_THRESHOLD_M = 30.0  # iOS horizontal uncertainty above this → no dot
SEARCH_RADIUS_M = 1500       # Overpass radius around the seed GPS
ISO6709 = re.compile(r"([+-]\d+\.\d+)([+-]\d+\.\d+)")

# Polygon fill colors — bright enough to read on the dark scorecard panel
COLORS = {
    "fairway":      (164, 198, 130, 230),
    "green":        (120, 180, 100, 240),
    "tee":          (180, 200, 140, 230),
    "bunker":       (235, 220, 170, 230),
    "water_hazard": (130, 170, 215, 230),
    "rough":        (140, 170, 110, 160),
}
CENTERLINE = (240, 240, 240, 180)
DOT = (220, 50, 47, 255)
DOT_EDGE = (255, 255, 255, 255)
DRAW_ORDER = ["rough", "fairway", "tee", "bunker", "green", "water_hazard"]


# --- GPS extraction -----------------------------------------------------

def extract_gps(raw_path: Path) -> tuple[tuple[float, float] | None, float | None]:
    """Return ((lat, lon) or None, accuracy_m or None) from QuickTime metadata.

    The lat/lon may be `None` even when accuracy is set — callers should treat
    that as "we know GPS was attempted but the fix is unusable".
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries",
         "format_tags=com.apple.quicktime.location.ISO6709,"
         "com.apple.quicktime.location.accuracy.horizontal",
         "-of", "default=nw=1:nk=1", str(raw_path)],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    iso = next((l for l in lines if ISO6709.search(l)), None)
    acc: float | None = None
    for l in lines:
        try:
            acc = float(l)
            break
        except ValueError:
            continue
    if not iso:
        return (None, acc)
    m = ISO6709.search(iso)
    return ((float(m.group(1)), float(m.group(2))), acc)


# --- OSM fetch + cache --------------------------------------------------

def _overpass_query(lat: float, lon: float) -> str:
    return f"""
[out:json][timeout:25];
(
  way(around:{SEARCH_RADIUS_M},{lat},{lon})["golf"="hole"];
  way(around:{SEARCH_RADIUS_M},{lat},{lon})["golf"~"^(fairway|green|tee|bunker|water_hazard|rough)$"];
  way(around:{SEARCH_RADIUS_M},{lat},{lon})["leisure"="golf_course"];
  relation(around:{SEARCH_RADIUS_M},{lat},{lon})["leisure"="golf_course"];
);
out geom;
""".strip()


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Shared per-course cache lives at the repo root and is committed to git, so
# replaying any course we've seen before is offline. Each round folder gets a
# tiny `course.json` pointer with the slug.
COURSES_DIR = Path(__file__).resolve().parent / "courses"


def _course_name(geom: dict) -> str | None:
    """First leisure=golf_course element with a name tag wins."""
    for e in geom.get("elements", []):
        if e.get("tags", {}).get("leisure") == "golf_course":
            name = e["tags"].get("name")
            if name:
                return name
    return None


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    return re.sub(r"[\s_-]+", "-", s) or "unknown-course"


def _ring(elem: dict) -> list[tuple[float, float]]:
    """(lon, lat) ring of an OSM way's geometry."""
    return [(n["lon"], n["lat"]) for n in elem.get("geometry", [])]


def _centroid(elem: dict) -> tuple[float, float]:
    g = elem["geometry"]
    return (sum(p["lon"] for p in g) / len(g), sum(p["lat"] for p in g) / len(g))


def _point_in_ring(lon: float, lat: float, ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon. `ring` is a list of (lon, lat)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def isolate_to_course(geom: dict, seed_lat: float, seed_lon: float) -> dict | None:
    """Clip a multi-course Overpass result to the one course the seed sits in.

    A radius search near clustered courses (e.g. Georges River / Bankstown /
    Liverpool) returns every course's holes, with colliding `ref` numbers, so
    `_hole_features` would draw two courses at once. We keep only the single
    `leisure=golf_course` boundary polygon that contains the GPS seed plus the
    golf features inside it.

    Returns the clipped geom, or None when the seed isn't cleanly inside exactly
    one mappable boundary — in which case the caller renders no map at all
    rather than guess (the "only if the whole course is clean" rule).
    """
    boundaries = [
        e for e in geom.get("elements", [])
        if e.get("tags", {}).get("leisure") == "golf_course"
        and e.get("type") == "way" and len(e.get("geometry", [])) >= 3
    ]
    containing = [b for b in boundaries if _point_in_ring(seed_lon, seed_lat, _ring(b))]
    if len(containing) != 1:
        return None  # outside every boundary, or inside overlapping ones
    course = containing[0]
    ring = _ring(course)
    kept = [course]
    for e in geom["elements"]:
        if e is course or e.get("type") != "way" or not e.get("geometry"):
            continue
        if not e.get("tags", {}).get("golf"):
            continue
        lon, lat = _centroid(e)
        if _point_in_ring(lon, lat, ring):
            kept.append(e)
    return {**geom, "elements": kept}


def fetch_course_geom(round_dir: Path, seed_lat: float, seed_lon: float,
                      force: bool = False) -> dict | None:
    """Fetch OSM golf features near (seed_lat, seed_lon).

    Canonical cache: `<repo>/courses/<slug>.json` (committed). Each round dir
    gets a `course.json` pointer with the slug, so re-rendering the same round
    later — or playing the same course again — needs no network.
    Returns the geom dict, or None if every Overpass mirror fails.
    """
    pointer = round_dir / "course.json"

    # Already linked to a shared cache and we trust it? Use it.
    if pointer.exists() and not force:
        info = json.loads(pointer.read_text())
        cached = COURSES_DIR / f"{info['slug']}.json"
        if cached.exists():
            return json.loads(cached.read_text())

    import httpx  # lazy: render path doesn't need httpx
    payload = {"data": _overpass_query(seed_lat, seed_lon)}
    last_err: Exception | None = None
    data: dict | None = None
    for url in OVERPASS_ENDPOINTS:
        try:
            r = httpx.post(url, data=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            continue
    if data is None:
        print(f"[course_map] all Overpass mirrors failed: {last_err}")
        return None

    # Clip to the single course the GPS sits in. If we can't isolate one
    # cleanly, render no map rather than a multi-course mess.
    data = isolate_to_course(data, seed_lat, seed_lon)
    if data is None:
        print("[course_map] GPS not cleanly inside one course boundary — skipping map")
        return None

    name = _course_name(data) or f"course-{seed_lat:.4f}-{seed_lon:.4f}"
    slug = _slugify(name)
    COURSES_DIR.mkdir(exist_ok=True)
    (COURSES_DIR / f"{slug}.json").write_text(json.dumps(data))
    pointer.write_text(json.dumps({"slug": slug, "name": name}, indent=2))

    # Drop legacy round-local cache from the previous layout.
    legacy = round_dir / "course_geom.json"
    if legacy.exists():
        legacy.unlink()

    return data


def load_course_geom(round_dir: Path) -> dict | None:
    """Return cached course geom for this round, or None."""
    pointer = round_dir / "course.json"
    if pointer.exists():
        info = json.loads(pointer.read_text())
        cached = COURSES_DIR / f"{info['slug']}.json"
        if cached.exists():
            return json.loads(cached.read_text())
    # Legacy fallback: pre-shared-store rounds.
    legacy = round_dir / "course_geom.json"
    if legacy.exists():
        return json.loads(legacy.read_text())
    return None


# --- Projection ---------------------------------------------------------

def _project(lat: float, lon: float, bbox: tuple[float, float, float, float],
             size: tuple[int, int]) -> tuple[float, float]:
    """Equirectangular projection with cosine correction. Good enough at hole scale."""
    min_lat, min_lon, max_lat, max_lon = bbox
    w, h = size
    cos_lat = math.cos(math.radians((min_lat + max_lat) / 2))
    span_lon = (max_lon - min_lon) * cos_lat
    span_lat = max_lat - min_lat
    img_aspect = w / h
    geo_aspect = span_lon / span_lat if span_lat else img_aspect
    x = (lon - min_lon) / (max_lon - min_lon) * w
    y = (max_lat - lat) / (max_lat - min_lat) * h  # north up
    if geo_aspect > img_aspect:
        scale = img_aspect / geo_aspect
        y = h / 2 + (y - h / 2) * scale
    else:
        scale = geo_aspect / img_aspect
        x = w / 2 + (x - w / 2) * scale
    return (x, y)


# --- Render -------------------------------------------------------------

def _hole_features(geom: dict, hole: int) -> list[dict]:
    return [
        e for e in geom["elements"]
        if e["type"] == "way"
        and e.get("tags", {}).get("golf") == "hole"
        and e["tags"].get("ref") == str(hole)
    ]


def _bbox(features: list[dict], pad_pct: float = 0.20) -> tuple[float, float, float, float]:
    lats, lons = [], []
    for f in features:
        for n in f.get("geometry", []):
            lats.append(n["lat"])
            lons.append(n["lon"])
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    pad_lat = (max_lat - min_lat) * pad_pct
    pad_lon = (max_lon - min_lon) * pad_pct
    return (min_lat - pad_lat, min_lon - pad_lon, max_lat + pad_lat, max_lon + pad_lon)


def render_hole_map(
    geom: dict,
    hole: int,
    gps: tuple[float, float] | None = None,
    accuracy: float | None = None,
    size: tuple[int, int] = (270, 270),
) -> Image.Image | None:
    """Render a single hole on a transparent canvas, returning RGBA.

    Returns None if the hole has no centerline geometry in `geom` — caller
    should fall back to the plain scorecard.
    """
    holes = _hole_features(geom, hole)
    if not holes:
        return None
    bbox = _bbox(holes, pad_pct=0.20)

    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def in_bbox(f):
        for n in f.get("geometry", []):
            if bbox[0] <= n["lat"] <= bbox[2] and bbox[1] <= n["lon"] <= bbox[3]:
                return True
        return False

    feats = [
        e for e in geom["elements"]
        if e["type"] == "way" and e.get("tags", {}).get("golf") in DRAW_ORDER
        and in_bbox(e)
    ]
    feats.sort(key=lambda f: DRAW_ORDER.index(f["tags"]["golf"]))

    for f in feats:
        coords = [_project(n["lat"], n["lon"], bbox, size) for n in f["geometry"]]
        kind = f["tags"]["golf"]
        if len(coords) >= 3:
            draw.polygon(coords, fill=COLORS[kind])

    # Hole centerline drawn last so it sits on top of the polygons.
    for f in holes:
        coords = [_project(n["lat"], n["lon"], bbox, size) for n in f["geometry"]]
        if len(coords) >= 2:
            draw.line(coords, fill=CENTERLINE, width=2)

    # Clip GPS dot — only when accuracy is good enough.
    if gps is not None and (accuracy is None or accuracy <= ACCURACY_THRESHOLD_M):
        if bbox[0] <= gps[0] <= bbox[2] and bbox[1] <= gps[1] <= bbox[3]:
            px, py = _project(gps[0], gps[1], bbox, size)
            r = 6
            draw.ellipse((px - r, py - r, px + r, py + r),
                         fill=DOT, outline=DOT_EDGE, width=2)

    # Trim transparent margin so the visual edge of the map is the actual
    # hole geometry — otherwise the bbox padding shows up as dead space next
    # to the scoreboard.
    content = img.getbbox()
    if content:
        img = img.crop(content)

    return img
