"""Travel time origin -> many destinations via OSRM table service (free-flow, no live traffic)."""
from __future__ import annotations

import math

from .http import FetchError, fetch_json

OSRM = {
    "driving": ["https://router.project-osrm.org/table/v1/driving/",
                "https://routing.openstreetmap.de/routed-car/table/v1/driving/"],
    "cycling": ["https://routing.openstreetmap.de/routed-bike/table/v1/driving/"],
    "walking": ["https://routing.openstreetmap.de/routed-foot/table/v1/driving/"],
}
# Fallback average speeds (km/h) and road detour factor when OSRM is unreachable.
FALLBACK_SPEED = {"driving": 25.0, "cycling": 14.0, "walking": 4.5}
DETOUR = 1.35


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


async def travel(origin: tuple[float, float], dests: list[tuple[float, float]], mode: str = "driving") -> list[dict]:
    """Returns [{'minutes', 'km', 'estimated': bool}] per destination (free-flow times)."""
    if not dests:
        return []
    coords = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in [origin, *dests])
    urls = [u + coords for u in OSRM.get(mode, OSRM["driving"])]
    try:
        res = await fetch_json(urls, source=f"OSRM ({mode})", params={"sources": "0", "annotations": "duration,distance"},
                               ttl=7 * 86400, timeout=15)
        if res.get("code") != "Ok":
            raise FetchError(res.get("code"))
        durs, dists = res["durations"][0][1:], res["distances"][0][1:]
        out = []
        for (lat, lon), du, di in zip(dests, durs, dists):
            if du is None:
                out.append(_estimate(origin, (lat, lon), mode))
            else:
                out.append({"minutes": round(du / 60, 1), "km": round(di / 1000, 1), "estimated": False})
        return out
    except FetchError:
        return [_estimate(origin, d, mode) for d in dests]


def _estimate(origin, dest, mode) -> dict:
    km = haversine_km(*origin, *dest) * DETOUR
    return {"minutes": round(km / FALLBACK_SPEED.get(mode, 25.0) * 60, 1), "km": round(km, 1), "estimated": True}
