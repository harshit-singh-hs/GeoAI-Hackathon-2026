"""Place name -> coordinates. Nominatim first (biased to India, then global), Photon as fallback."""
from __future__ import annotations

import os

from .http import FetchError, fetch_json

NOMINATIM = "https://nominatim.openstreetmap.org/search"
PHOTON = "https://photon.komoot.io/api/"
COUNTRY_BIAS = os.environ.get("GEOAI_COUNTRY_BIAS", "in")

# Nominatim place types that describe a whole settlement rather than a neighbourhood.
CITY_TYPES = {"city", "town", "municipality", "county", "state_district", "district", "administrative", "state"}


def _from_nominatim(r: dict) -> dict:
    bb = [float(x) for x in r.get("boundingbox", [])] or None  # [south, north, west, east]
    return {
        "name": r.get("name") or r["display_name"].split(",")[0],
        "display_name": r["display_name"],
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "kind": r.get("addresstype") or r.get("type"),
        "is_city": (r.get("addresstype") or r.get("type")) in CITY_TYPES,
        "bbox": bb,
        "osm_key": r.get("category"),
        "osm_value": r.get("type"),
        "osm_id": f"{r.get('osm_type', 'x')[0]}{r.get('osm_id')}",
        "source": "Nominatim (OpenStreetMap)",
    }


def _from_photon(f: dict) -> dict:
    p = f["properties"]
    lon, lat = f["geometry"]["coordinates"]
    ext = p.get("extent")  # [west, north, east, south]
    parts = [p.get(k) for k in ("name", "district", "city", "state", "country") if p.get(k)]
    return {
        "name": p.get("name") or parts[0],
        "display_name": ", ".join(dict.fromkeys(parts)),
        "lat": lat,
        "lon": lon,
        "kind": p.get("type") or p.get("osm_value"),
        "is_city": (p.get("type") or p.get("osm_value")) in CITY_TYPES,
        "bbox": [ext[3], ext[1], ext[0], ext[2]] if ext else None,
        "osm_key": p.get("osm_key"),
        "osm_value": p.get("osm_value"),
        "osm_id": f"{(p.get('osm_type') or 'x')[0].lower()}{p.get('osm_id')}",
        "source": "Photon (komoot, OpenStreetMap data)",
    }


async def geocode(text: str) -> dict | None:
    """Return the best match for `text`, or None if nothing plausible was found."""
    text = text.strip()
    if not text:
        return None
    base = {"q": text, "format": "jsonv2", "limit": 1, "accept-language": "en", "addressdetails": 0}
    attempts = ([{**base, "countrycodes": COUNTRY_BIAS}] if COUNTRY_BIAS else []) + [base]
    for params in attempts:
        try:
            res = await fetch_json(NOMINATIM, source="Nominatim", params=params, ttl=30 * 86400, timeout=12)
        except FetchError:
            break  # Nominatim down -> try Photon
        if res:
            return _from_nominatim(res[0])
    try:
        res = await fetch_json(PHOTON, source="Photon", params={"q": text, "limit": 1, "lang": "en"}, ttl=30 * 86400, timeout=12)
    except FetchError:
        return None
    feats = res.get("features") or []
    return _from_photon(feats[0]) if feats else None
