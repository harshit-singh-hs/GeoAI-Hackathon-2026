"""OSM tag -> context class mapping, shared by the live Overpass client and the offline index builder."""
from __future__ import annotations

KEEP_TAGS = {
    "name", "name:en", "leisure", "landuse", "natural", "tourism", "historic", "sport", "amenity", "man_made",
    "power", "lit", "surface", "access", "golf", "club", "boundary", "highway", "ref", "operator", "fee",
    "opening_hours", "wheelchair", "water", "waterway", "industrial", "product", "healthcare",
}

ROAD_CLASSES = {"motorway": 1.0, "trunk": 0.85, "primary": 0.6}
# Relative emission weight of non-road sources (rough, for ranking exposure only).
# landuse=industrial in Indian OSM data often covers warehouses and light industry, so it is weighted modestly.
INDUSTRY_WEIGHTS = {
    ("power", "plant"): 1.0, ("landuse", "industrial"): 0.35, ("man_made", "works"): 0.6,
    ("landuse", "quarry"): 0.5, ("landuse", "landfill"): 0.6, ("man_made", "wastewater_plant"): 0.25,
}
GREEN = {("leisure", "park"), ("leisure", "garden"), ("leisure", "nature_reserve"), ("landuse", "forest"),
         ("landuse", "grass"), ("landuse", "meadow"), ("landuse", "recreation_ground"), ("landuse", "village_green"),
         ("natural", "wood"), ("natural", "scrub"), ("natural", "grassland")}
WATER = {("natural", "water"), ("landuse", "reservoir"), ("waterway", "riverbank")}


def industry_weight(tags: dict) -> float:
    return max((w for (k, v), w in INDUSTRY_WEIGHTS.items() if tags.get(k) == v), default=0.0)


def context_classes(tags: dict) -> list[str]:
    out = []
    if tags.get("highway") in ROAD_CLASSES:
        out.append("road")
    if industry_weight(tags) > 0:
        out.append("industrial")
    if any(tags.get(k) == v for k, v in GREEN):
        out.append("green")
    if any(tags.get(k) == v for k, v in WATER):
        out.append("water")
    if tags.get("power") == "line":
        out.append("power_line")
    if tags.get("amenity") == "hospital" or tags.get("healthcare") == "hospital":
        out.append("hospital")
    return out
