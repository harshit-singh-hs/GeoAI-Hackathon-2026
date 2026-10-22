"""Build the offline OSM fallback index from a Geofabrik extract.

The assistant queries Overpass live first. Public Overpass servers are frequently overloaded, so this index
(built from the openly downloadable Geofabrik extract, © OpenStreetMap contributors, ODbL) is used when Overpass
does not answer in time. Every response says which of the two was used.

Usage:  python scripts/build_osm_index.py data/osm/india-latest.osm.pbf data/osm/osm_index.sqlite
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import time

import osmium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.activities import ACTIVITIES  # noqa: E402
from app.osm_classes import KEEP_TAGS, context_classes  # noqa: E402

COMPILED = [{k: re.compile(v) for k, v in s.tags.items()} for a in ACTIVITIES.values() for s in a.selectors]
FILTER_KEYS = sorted({k for c in COMPILED for k in c} | {
    "highway", "landuse", "man_made", "power", "natural", "amenity", "waterway", "leisure"})


def is_venue(tags: dict) -> bool:
    if "name" not in tags and "name:en" not in tags:
        return False  # unnamed venues are not useful to recommend
    return any(all(k in tags and rx.search(tags[k]) for k, rx in c.items()) for c in COMPILED)


def ring_area_m2(coords: list[tuple[float, float]]) -> float:
    """Planar shoelace area on an equirectangular projection around the ring's mean latitude."""
    if len(coords) < 3:
        return 0.0
    lat0 = math.radians(sum(c[1] for c in coords) / len(coords))
    kx, ky = 111320.0 * math.cos(lat0), 110540.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:] + coords[:1]):
        s += (x1 * kx) * (y2 * ky) - (x2 * kx) * (y1 * ky)
    return abs(s) / 2


def simplify(coords: list[tuple[float, float]], step_m: float = 60) -> list[list[float]]:
    """Keep vertices at least ~step_m apart (plenty for distance-to-road at 100 m resolution)."""
    out = [coords[0]]
    for c in coords[1:-1]:
        dx = (c[0] - out[-1][0]) * 111320 * math.cos(math.radians(c[1]))
        dy = (c[1] - out[-1][1]) * 110540
        if dx * dx + dy * dy >= step_m * step_m:
            out.append(c)
    if len(coords) > 1:
        out.append(coords[-1])
    return [[round(lat, 5), round(lon, 5)] for lon, lat in out]


def main(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.remove(dst)
    db = sqlite3.connect(dst)
    db.executescript("""
        CREATE TABLE feat (id INTEGER PRIMARY KEY, osm TEXT, cls TEXT, name TEXT, lat REAL, lon REAL,
                           area REAL, tags TEXT, geom TEXT);
        CREATE VIRTUAL TABLE feat_rt USING rtree(id, minlat, maxlat, minlon, maxlon);
    """)
    rows, rts = [], []
    n = 0
    t0 = time.time()

    def add(osm, cls, tags, lat, lon, bbox, area=0.0, geom=None):
        nonlocal n
        n += 1
        name = tags.get("name:en") or tags.get("name")
        kept = {k: v for k, v in tags.items() if k in KEEP_TAGS}
        rows.append((n, osm, cls, name, lat, lon, area, json.dumps(kept, ensure_ascii=False),
                     json.dumps(geom) if geom else None))
        rts.append((n, bbox[0], bbox[1], bbox[2], bbox[3]))
        if len(rows) >= 50000:
            flush()

    def flush():
        db.executemany("INSERT INTO feat VALUES (?,?,?,?,?,?,?,?,?)", rows)
        db.executemany("INSERT INTO feat_rt VALUES (?,?,?,?,?)", rts)
        db.commit()
        rows.clear()
        rts.clear()
        print(f"  {n:,} features, {time.time() - t0:.0f}s", flush=True)

    fp = (osmium.FileProcessor(src)
          .with_locations("flex_mem")
          .with_areas(osmium.filter.KeyFilter(*FILTER_KEYS))
          .with_filter(osmium.filter.KeyFilter(*FILTER_KEYS)))

    for obj in fp:
        tags = dict(obj.tags)
        classes = context_classes(tags)
        venue = is_venue(tags)
        if not classes and not venue:
            continue
        if venue:
            classes = ["venue", *classes]
        try:
            if obj.is_node():
                lat, lon = obj.location.lat, obj.location.lon
                for cls in classes:
                    if cls not in ("green", "water"):  # area-only classes; a bare node says nothing about cover
                        add(f"n{obj.id}", cls, tags, lat, lon, (lat, lat, lon, lon))
            elif obj.is_way():
                # Closed ways with area tags come back again as Area objects; here keep only linear features.
                if obj.is_closed() and not any(c in ("road", "power_line") for c in classes):
                    continue
                coords = [(nd.lon, nd.lat) for nd in obj.nodes if nd.location.valid()]
                if len(coords) < 2:
                    continue
                lats, lons = [c[1] for c in coords], [c[0] for c in coords]
                bbox = (min(lats), max(lats), min(lons), max(lons))
                mid = coords[len(coords) // 2]
                for cls in classes:
                    if cls in ("road", "power_line") or (cls == "venue" and tags.get("highway") == "cycleway"):
                        add(f"w{obj.id}", cls, tags, mid[1], mid[0], bbox, geom=simplify(coords))
            elif obj.is_area():
                outer = [[(nd.lon, nd.lat) for nd in ring] for ring in obj.outer_rings()]
                outer = [r for r in outer if len(r) >= 3]
                if not outer:
                    continue
                pts = [p for r in outer for p in r]
                lats, lons = [p[1] for p in pts], [p[0] for p in pts]
                bbox = (min(lats), max(lats), min(lons), max(lons))
                area = sum(ring_area_m2(r) for r in outer)
                big = max(outer, key=len)
                clat, clon = sum(p[1] for p in big) / len(big), sum(p[0] for p in big) / len(big)
                osm = f"{'w' if obj.from_way() else 'r'}{obj.orig_id()}"
                for cls in classes:
                    if cls in ("road", "power_line"):
                        continue
                    add(osm, cls, tags, clat, clon, bbox, area)
        except osmium.InvalidLocationError:
            continue
    flush()
    db.execute("CREATE INDEX feat_cls ON feat(cls)")
    db.commit()
    print(f"done: {n:,} features in {time.time() - t0:.0f}s -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
