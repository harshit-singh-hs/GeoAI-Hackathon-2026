"""Venues and geospatial context from OpenStreetMap.

Live Overpass API first; if it is overloaded or slow we fall back to a local index built from the
Geofabrik extract (scripts/build_osm_index.py). Both paths return the same Feature dicts:

  {"id", "cls", "name", "lat", "lon", "bbox": [minlat, maxlat, minlon, maxlon], "area", "tags", "geom"}
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import time

from ..activities import Activity
from ..osm_classes import KEEP_TAGS, ROAD_CLASSES, context_classes
from .http import FetchError, fetch_json, record
from .routing import haversine_km

OVERPASS = os.environ.get(
    "GEOAI_OVERPASS", "https://overpass-api.de/api/interpreter,https://overpass.private.coffee/api/interpreter"
).split(",")
OSM_MODE = os.environ.get("GEOAI_OSM_MODE", "auto")  # auto | live | local
INDEX_PATH = os.environ.get("GEOAI_OSM_INDEX", "data/osm/osm_index.sqlite")
OVERPASS_BUDGET_S = float(os.environ.get("GEOAI_OVERPASS_TIMEOUT", "60"))  # background completion limit
OVERPASS_WAIT_S = float(os.environ.get("GEOAI_OVERPASS_WAIT", "12"))  # how long a user request waits for it
BREAKER_S = 180  # after an Overpass failure, go straight to the local index for this long

# Context search radii (m) around each candidate venue.
R_ROAD, R_INDUSTRY, R_WATER, R_GREEN, R_POWER, R_HOSPITAL = 1500, 2500, 800, 600, 300, 4000

_breaker_until = 0.0
_slow_until = 0.0  # after a timeout we keep querying Overpass in the background but don't block on it


# ---------------------------------------------------------------- helpers

def _match(tags: dict, act: Activity) -> tuple[float, str] | None:
    best = None
    for s in act.selectors:
        if all(k in tags and re.search(rx, tags[k]) for k, rx in s.tags.items()):
            if best is None or s.weight > best[0]:
                best = (s.weight, s.label)
    return best


def _feature_from_overpass(e: dict) -> dict | None:
    tags = e.get("tags", {})
    if "center" in e:
        lat, lon = e["center"]["lat"], e["center"]["lon"]
    elif "lat" in e:
        lat, lon = e["lat"], e["lon"]
    elif e.get("geometry"):
        g = e["geometry"]
        lat, lon = g[len(g) // 2]["lat"], g[len(g) // 2]["lon"]
    else:
        return None
    b = e.get("bounds")
    bbox = [b["minlat"], b["maxlat"], b["minlon"], b["maxlon"]] if b else [lat, lat, lon, lon]
    # Overpass gives bounds, not polygons: approximate area as 60% of the bbox (good enough for ranking).
    area = 0.6 * _bbox_area(bbox) if b and e["type"] != "node" else 0.0
    geom = [[p["lat"], p["lon"]] for p in e["geometry"]] if e.get("geometry") else None
    return {"id": f"{e['type'][0]}{e['id']}", "cls": None, "name": tags.get("name:en") or tags.get("name"),
            "lat": lat, "lon": lon, "bbox": bbox, "area": area,
            "tags": {k: v for k, v in tags.items() if k in KEEP_TAGS}, "geom": geom}


def _bbox_area(bb) -> float:
    h = (bb[1] - bb[0]) * 110540
    w = (bb[3] - bb[2]) * 111320 * math.cos(math.radians((bb[0] + bb[1]) / 2))
    return abs(h * w)


def _deg_box(lat, lon, radius_m):
    dlat = radius_m / 110540
    dlon = radius_m / (111320 * max(0.1, math.cos(math.radians(lat))))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def _trip_breaker(task: asyncio.Task) -> None:
    global _breaker_until
    if not task.cancelled() and task.exception() is not None:
        _breaker_until = time.time() + BREAKER_S


async def _overpass(query: str, label: str):
    """Live Overpass with a short wait. If it is slow, answer from the offline index now and let the live request
    finish in the background so its result lands in the cache for the next similar question."""
    if OSM_MODE == "local" or time.time() < _breaker_until:
        return None
    task = asyncio.create_task(fetch_json(OVERPASS, source=f"Overpass API ({label})", data={"data": query},
                                          ttl=7 * 86400, timeout=OVERPASS_BUDGET_S, retries=0))
    task.add_done_callback(_trip_breaker)
    global _slow_until
    wait = OVERPASS_BUDGET_S if OSM_MODE == "live" else (1.0 if time.time() < _slow_until else OVERPASS_WAIT_S)
    try:
        return await asyncio.wait_for(asyncio.shield(task), wait)
    except asyncio.TimeoutError:
        if OSM_MODE == "auto" and wait > 1.0:
            _slow_until = time.time() + BREAKER_S
        record(f"Overpass API ({label})", OVERPASS[0], cached=False, ok=False,
               note=f"no answer within {wait:.0f}s; still running in background to warm the cache")
        return None
    except FetchError:
        if OSM_MODE == "live":
            raise
        return None


_db: sqlite3.Connection | None = None


def _local() -> sqlite3.Connection | None:
    global _db
    if _db is None and os.path.exists(INDEX_PATH):
        _db = sqlite3.connect(f"file:{INDEX_PATH}?mode=ro", uri=True, check_same_thread=False)
    return _db


def _local_query(lat, lon, radius_m, classes: tuple[str, ...]) -> list[dict]:
    db = _local()
    if db is None:
        return []
    s, n, w, e = _deg_box(lat, lon, radius_m)
    rows = db.execute(
        f"""SELECT f.osm, f.cls, f.name, f.lat, f.lon, f.area, f.tags, f.geom, r.minlat, r.maxlat, r.minlon, r.maxlon
            FROM feat_rt r JOIN feat f ON f.id = r.id
            WHERE r.maxlat >= ? AND r.minlat <= ? AND r.maxlon >= ? AND r.minlon <= ?
              AND f.cls IN ({",".join("?" * len(classes))})""",
        (s, n, w, e, *classes)).fetchall()
    return [{"id": r[0], "cls": r[1], "name": r[2], "lat": r[3], "lon": r[4], "area": r[5], "tags": json.loads(r[6]),
             "geom": json.loads(r[7]) if r[7] else None, "bbox": [r[8], r[9], r[10], r[11]]} for r in rows]


def _same_name(a: str | None, b: str | None) -> bool:
    """'Visapur' vs 'Visapur Fort', 'Tennis Court' vs 'Tennis Courts' count as the same place nearby."""
    if not a or not b:
        return False
    a, b = re.sub(r"[^a-z0-9 ]", "", a.lower()).strip(), re.sub(r"[^a-z0-9 ]", "", b.lower()).strip()
    return a == b or a.startswith(b) or b.startswith(a)


def _dedupe_and_pick(venues: list[dict], origin, radius_km: float, n: int) -> list[dict]:
    """Rank by suitability, size and distance; drop near-duplicates; keep spatially diverse picks."""
    for v in venues:
        v["dist_km"] = round(haversine_km(origin[0], origin[1], v["lat"], v["lon"]), 2)
        size = min(1.0, math.log10(max(v["area"], 1)) / 5)  # 100 000 m2 -> 1.0
        private = 0.5 if v["tags"].get("access") in ("private", "no", "customers") else 1.0
        v["prelim"] = v["suit"] * (0.6 + 0.4 * size) * (1 - 0.5 * v["dist_km"] / max(radius_km, 1)) * private
    venues.sort(key=lambda v: -v["prelim"])
    picked: list[dict] = []
    for v in venues:
        if v["dist_km"] > radius_km * 1.05:
            continue
        if any(haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]) < 0.35 or
               (_same_name(v["name"], p["name"]) and haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]) < 2)
               for p in picked):
            continue
        picked.append(v)
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------- public API

async def find_venues(act: Activity, lat: float, lon: float, radius_km: float, n: int = 10) -> tuple[list[dict], str]:
    radius_m = int(radius_km * 1000)
    clauses = "".join(
        f'nwr(around:{radius_m},{lat:.5f},{lon:.5f})' + "".join(f'["{k}"~"{rx}"]' for k, rx in s.tags.items())
        + '["name"];'
        for s in act.selectors)
    q = f"[out:json][timeout:50];({clauses});out tags center bb qt 400;"
    res = await _overpass(q, f"{act.id} venues")
    source = "overpass"
    if res is not None:
        feats = [f for f in (_feature_from_overpass(e) for e in res.get("elements", [])) if f and f["name"]]
    else:
        feats = [f for f in _local_query(lat, lon, radius_m, ("venue",)) if f["name"]]
        source = "local"
        record("OSM offline index (Geofabrik extract)", f"local:{INDEX_PATH}", cached=True,
               note="Overpass unavailable; used Geofabrik extract dated in README")
    out = []
    for f in feats:
        m = _match(f["tags"] | ({"name": f["name"]} if f["name"] else {}), act)
        if m:
            f["cls"], f["suit"], f["kind"] = "venue", m[0], m[1]
            out.append(f)
    return _dedupe_and_pick(out, (lat, lon), radius_km, n), source


async def find_context(points: list[tuple[float, float]]) -> tuple[list[dict], str]:
    """Roads, industry, water, green cover, power lines and hospitals around the candidate venues."""
    if not points:
        return [], "none"
    parts = {"roads": [], "ind": [], "env": [], "pl": [], "hosp": []}
    rx_road = "|".join(ROAD_CLASSES)
    for lat, lon in points:
        p = f"{lat:.5f},{lon:.5f}"
        parts["roads"].append(f'way(around:{R_ROAD},{p})["highway"~"^({rx_road})$"];')
        parts["ind"].append(f'nwr(around:{R_INDUSTRY},{p})["landuse"~"^(industrial|quarry|landfill)$"];'
                            f'nwr(around:{R_INDUSTRY},{p})["man_made"~"^(works|wastewater_plant)$"];'
                            f'nwr(around:{R_INDUSTRY},{p})["power"="plant"];')
        parts["env"].append(f'wr(around:{R_WATER},{p})["natural"="water"];wr(around:{R_WATER},{p})["landuse"="reservoir"];'
                            f'wr(around:{R_GREEN},{p})["leisure"~"^(park|garden|nature_reserve)$"];'
                            f'wr(around:{R_GREEN},{p})["landuse"~"^(forest|grass|meadow|recreation_ground|village_green)$"];'
                            f'wr(around:{R_GREEN},{p})["natural"~"^(wood|scrub|grassland)$"];')
        parts["pl"].append(f'way(around:{R_POWER},{p})["power"="line"];')
        parts["hosp"].append(f'nwr(around:{R_HOSPITAL},{p})["amenity"="hospital"];')
    q = (f"[out:json][timeout:50];"
         f"({''.join(parts['roads'])});out tags geom qt;"
         f"({''.join(parts['ind'])});out tags center bb qt;"
         f"({''.join(parts['env'])});out tags center bb qt;"
         f"({''.join(parts['pl'])});out tags bb qt;"
         f"({''.join(parts['hosp'])});out tags center qt 60;")
    res = await _overpass(q, "context")
    feats: list[dict] = []
    if res is not None:
        for e in res.get("elements", []):
            f = _feature_from_overpass(e)
            if not f:
                continue
            for cls in context_classes(e.get("tags", {})):
                feats.append({**f, "cls": cls})
        return feats, "overpass"
    seen = set()
    radii = {"road": R_ROAD, "industrial": R_INDUSTRY, "water": R_WATER + 1000, "green": R_GREEN + 1000,
             "power_line": R_POWER, "hospital": R_HOSPITAL}
    for lat, lon in points:
        for cls, radius in radii.items():
            for f in _local_query(lat, lon, radius, (cls,)):
                key = (f["id"], f["cls"])
                if key not in seen:
                    seen.add(key)
                    feats.append(f)
    record("OSM offline index (Geofabrik extract)", f"local:{INDEX_PATH}", cached=True, note="context fallback")
    return feats, "local"
