"""Hourly weather + air quality from Open-Meteo (one multi-point request each)."""
from __future__ import annotations

import os

from .http import FetchError, fetch_json

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MAX_WEATHER_DAYS = 16
MAX_AQ_DAYS = 5  # CAMS forecast horizon is ~5 days; beyond that values are null
WX_TTL = float(os.environ.get("GEOAI_WX_TTL", "1800"))
# Forecast models are gridded: points closer than the grid spacing get identical values. Requesting one point per
# cell (not per venue) keeps us well inside Open-Meteo's free quota. CAMS air quality is ~0.4 deg globally.
WX_CELL, AQ_CELL = 0.05, 0.25

HOURLY_WX = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation_probability",
    "precipitation", "weather_code", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "uv_index", "is_day", "cape",
]
HOURLY_AQ = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide", "us_aqi", "dust"]

WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _as_list(res) -> list[dict]:
    return res if isinstance(res, list) else [res]


def _cells(points: list[tuple[float, float]], size: float) -> tuple[list[tuple[float, float]], list[int]]:
    """Snap points to grid-cell centres; return the unique cells and, per input point, its cell index."""
    cells, index, out = {}, [], []
    for lat, lon in points:
        key = (round(round(lat / size) * size, 4), round(round(lon / size) * size, 4))
        if key not in cells:
            cells[key] = len(out)
            out.append(key)
        index.append(cells[key])
    return out, index


def _days_bucket(days: int) -> int:
    return 3 if days <= 3 else 7 if days <= 7 else MAX_WEATHER_DAYS


def _coords(points: list[tuple[float, float]]) -> dict:
    return {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
    }


async def forecast(points: list[tuple[float, float]], days: int) -> list[dict]:
    """Per point: {'tz', 'utc_offset', 'elevation', 'hourly': {time: {...}}, 'daily': {date: {...}}}."""
    cells, idx = _cells(points, WX_CELL)
    params = {
        **_coords(cells),
        "hourly": ",".join(HOURLY_WX),
        "daily": "sunrise,sunset,uv_index_max",
        "timezone": "auto",
        "forecast_days": _days_bucket(max(1, min(days, MAX_WEATHER_DAYS))),
        "wind_speed_unit": "kmh",
    }
    res = await fetch_json(FORECAST_URL, source="Open-Meteo Forecast", params=params, ttl=WX_TTL, timeout=20)
    out = []
    for loc in _as_list(res):
        h = loc["hourly"]
        hourly = {t: {k: h[k][i] for k in HOURLY_WX} for i, t in enumerate(h["time"])}
        d = loc["daily"]
        daily = {t: {"sunrise": d["sunrise"][i], "sunset": d["sunset"][i], "uv_index_max": d["uv_index_max"][i]}
                 for i, t in enumerate(d["time"])}
        out.append({"tz": loc["timezone"], "utc_offset": loc["utc_offset_seconds"], "elevation": loc.get("elevation"),
                    "hourly": hourly, "daily": daily})
    return [out[i] for i in idx]


async def air_quality(points: list[tuple[float, float]], days: int) -> list[dict] | None:
    """Per point: {time: {pm2_5, ..., us_aqi}}. Returns None if the AQ service is unavailable."""
    cells, idx = _cells(points, AQ_CELL)
    params = {**_coords(cells), "hourly": ",".join(HOURLY_AQ), "timezone": "auto", "forecast_days": 6}
    try:
        res = await fetch_json(AQ_URL, source="Open-Meteo Air Quality (CAMS)", params=params, ttl=max(WX_TTL, 3600), timeout=20)
    except FetchError:
        return None
    out = []
    for loc in _as_list(res):
        h = loc["hourly"]
        out.append({t: {k: h[k][i] for k in HOURLY_AQ} for i, t in enumerate(h["time"])})
    return [out[i] for i in idx]
