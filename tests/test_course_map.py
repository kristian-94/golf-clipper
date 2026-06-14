"""Tests for OSM course-map isolation.

Run:
    /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/

An Overpass radius search near clustered courses returns several courses' holes
with colliding `ref` numbers. `isolate_to_course` must clip the result to the
single course the GPS seed sits in, so the overlay never draws two courses at
once — and must refuse (return None) when it can't pick one cleanly, so the
caller falls back to no map rather than a wrong one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from course_map import isolate_to_course  # noqa: E402


def _way(tags: dict, pts: list[tuple[float, float]]) -> dict:
    """OSM way from (lon, lat) points."""
    return {"type": "way",
            "geometry": [{"lat": la, "lon": lo} for lo, la in pts],
            "tags": tags}


def _two_course_geom() -> dict:
    """Two non-overlapping courses, each with a hole numbered 1 (the collision)."""
    course_a = _way({"leisure": "golf_course", "name": "Course A"},
                    [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    hole_a = _way({"golf": "hole", "ref": "1"}, [(0.4, 0.5), (0.6, 0.5)])
    fair_a = _way({"golf": "fairway"}, [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    course_b = _way({"leisure": "golf_course", "name": "Course B"},
                    [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)])
    hole_b = _way({"golf": "hole", "ref": "1"}, [(10.4, 10.5), (10.6, 10.5)])
    fair_b = _way({"golf": "fairway"}, [(10.4, 10.4), (10.6, 10.4), (10.6, 10.6)])
    return {"elements": [course_a, hole_a, fair_a, course_b, hole_b, fair_b]}


def test_isolates_to_the_course_the_seed_is_in():
    geom = _two_course_geom()
    out = isolate_to_course(geom, seed_lat=0.5, seed_lon=0.5)
    assert out is not None
    names = [e["tags"]["name"] for e in out["elements"]
             if e["tags"].get("leisure") == "golf_course"]
    assert names == ["Course A"], "kept the wrong / multiple courses"
    # The colliding ref=1 must resolve to Course A's hole only.
    holes = [e for e in out["elements"] if e["tags"].get("golf") == "hole"]
    assert len(holes) == 1
    lon = holes[0]["geometry"][0]["lon"]
    assert lon < 1, "kept the other course's hole 1"
    # Course B's features are gone entirely.
    assert all(e["geometry"][0]["lon"] < 1 for e in out["elements"])


def test_returns_none_when_seed_is_outside_every_boundary():
    geom = _two_course_geom()
    assert isolate_to_course(geom, seed_lat=5.0, seed_lon=5.0) is None


def test_returns_none_when_no_boundary_polygon_present():
    """Holes but no golf_course boundary → can't verify cleanliness → no map."""
    geom = {"elements": [_way({"golf": "hole", "ref": "1"}, [(0.4, 0.5), (0.6, 0.5)])]}
    assert isolate_to_course(geom, seed_lat=0.5, seed_lon=0.5) is None


def test_single_clean_course_passes_through():
    """A normal isolated course (one boundary, seed inside) keeps its holes."""
    course = _way({"leisure": "golf_course", "name": "Solo GC"},
                  [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    holes = [_way({"golf": "hole", "ref": str(n)}, [(0.5, 0.5 + n * 0.01), (1.5, 0.5)])
             for n in range(1, 19)]
    out = isolate_to_course({"elements": [course, *holes]}, seed_lat=1.0, seed_lon=1.0)
    assert out is not None
    refs = sorted(int(e["tags"]["ref"]) for e in out["elements"]
                  if e["tags"].get("golf") == "hole")
    assert refs == list(range(1, 19))
