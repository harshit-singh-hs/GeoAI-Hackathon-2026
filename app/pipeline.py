"""Question -> grounded recommendation. Orchestrates parsing, live data fetching, scoring and narration."""
from __future__ import annotations

import asyncio
import math
import re
import time
import uuid
from collections import Counter, OrderedDict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from timezonefinder import TimezoneFinder

from .activities import ACTIVITIES, PROFILES, apply_profiles
from .data import weather as wxapi
from .data.geocode import geocode
from .data.http import FetchError, start_trace
from .data.osm import _match, _same_name, find_context, find_venues
from .data.routing import haversine_km, travel
from .parse import ParsedQuery, parse
from .scoring import HourScore, place_context, score_hour, site_bonus, traffic_level
from .timewin import resolve

_tzf = TimezoneFinder()
RESULTS: OrderedDict[str, dict] = OrderedDict()  # recent results, for the map page
MAX_RESULTS = 300
SUPPORTED = ", ".join(a.name for a in ACTIVITIES.values())


def _remember(res: dict) -> None:
    RESULTS[res["id"]] = res
    while len(RESULTS) > MAX_RESULTS:
        RESULTS.popitem(last=False)


def _base(query: str, pq: ParsedQuery | None) -> dict:
    return {"id": uuid.uuid4().hex[:12], "query": query, "status": "ok",
            "understood": pq.to_dict() if pq else None, "answer": "", "answer_source": "template"}


def _category(reason: str) -> str:
    return re.split(r"[:(\d]| and ", reason, maxsplit=1)[0].strip().rstrip(",") or reason


def _hm(t: datetime) -> str:
    return t.strftime("%a %d %b, %H:%M")


async def run(query: str, llm=None, lat: float | None = None, lon: float | None = None,
              now_utc: datetime | None = None) -> dict:
    t_start = time.time()
    trace = start_trace()
    query = (query or "").strip()[:500]
    pq = await parse(query, llm) if query else ParsedQuery(out_of_scope=True)
    res = _base(query, pq)

    def finish(r: dict) -> dict:
        r["sources"] = [vars(s) for s in trace.records]
        r["timing_ms"] = round((time.time() - t_start) * 1000)
        _remember(r)
        return r

    # ---------------------------------------------------------------- scope and clarification
    if pq.out_of_scope:
        res.update(status="out_of_scope", answer=(
            "I can help you pick a place and time for outdoor activities using live weather, air-quality and map "
            f"data. That question looks outside that scope. Try something like: \"Where can I go running tomorrow "
            f"evening near Koregaon Park?\" Supported activities: {SUPPORTED}."))
        return finish(res)
    if pq.activity is None:
        if pq.activity_text:
            res.update(status="unsupported_activity", answer=(
                f"I don't have a dedicated model for **{pq.activity_text}** yet, so I'd rather not guess its weather "
                f"limits. I can plan: {SUPPORTED}. If one of these is close enough, ask again with it."))
            return finish(res)
        if re.search(r"\b(outdoors?|outside|go out|step out)\b", query.lower()):
            pq.activity, pq.activity_text = "walking", "being outdoors"  # stated assumption, shown in 'understood'
        else:
            res.update(status="clarify", answer=(
                f"Which activity do you have in mind? I can plan: {SUPPORTED}."))
            return finish(res)

    act0 = ACTIVITIES[pq.activity]
    act, lim = apply_profiles(act0, pq.profiles)

    if pq.location:
        geo = await geocode(pq.location)
        if geo is None:
            res.update(status="clarify", answer=(
                f"I couldn't find **{pq.location}** on OpenStreetMap. Could you give a nearby landmark, "
                "neighbourhood or city name?"))
            return finish(res)
    elif lat is not None and lon is not None:
        geo = {"name": "your location", "display_name": f"{lat:.4f}, {lon:.4f}", "lat": lat, "lon": lon,
               "is_city": False, "source": "coordinates supplied by the caller"}
    else:
        res.update(status="clarify", answer=(
            f"Which area are you in? Tell me a neighbourhood or city (for example \"{act.name.split(' (')[0].lower()} "
            "near Powai\") and I'll check live weather, air quality and nearby places for you."))
        return finish(res)

    tzname = _tzf.timezone_at(lat=geo["lat"], lng=geo["lon"]) or "Asia/Kolkata"
    tz = ZoneInfo(tzname)
    now = (now_utc or datetime.now(timezone.utc)).astimezone(tz).replace(tzinfo=None)
    win = resolve(query, now)
    res["understood"].update(resolved_location=geo["display_name"], timezone=tzname, window=win.label,
                             now_local=now.isoformat(timespec="minutes"),
                             window_days=[d.isoformat() for d in win.days], window_hours=list(win.hours),
                             activity_name=act.name, profile_labels=[PROFILES[p].label for p in pq.profiles])
    if "past_time" in win.notes:
        res.update(status="clarify", answer="That time is in the past; I can only plan using forecasts. Which upcoming day did you mean?")
        return finish(res)
    if not win.days:
        res.update(status="beyond_forecast", answer=(
            f"Reliable hourly forecasts only reach {wxapi.MAX_WEATHER_DAYS} days ahead (air quality about "
            f"{wxapi.MAX_AQ_DAYS} days), so I can't responsibly recommend a time that far out. Ask me again closer "
            "to the date."))
        return finish(res)

    radius = pq.radius_km or act.default_radius_km * (1.6 if geo.get("is_city") else 1.0)
    radius = max(1.0, min(radius, 60.0))

    # ---------------------------------------------------------------- venues
    osm_src = "n/a"
    if pq.commute or (act.id == "cycling" and pq.intent == "go_no_go"):
        venues = [_pseudo_venue(geo, "your route / area")]
    else:
        venues, osm_src = await find_venues(act, geo["lat"], geo["lon"], radius)
        if len(venues) < 3 and not pq.radius_km and radius < 60:
            radius = min(60.0, radius * 2)
            more, osm_src = await find_venues(act, geo["lat"], geo["lon"], radius)
            seen = {v["id"] for v in venues}
            venues += [v for v in more if v["id"] not in seen]
        named = _named_venue(geo, act)
        if not named and pq.location:
            # The geocoder may return a feature inside the venue (e.g. a tomb in a garden): match by name instead.
            named = next((v for v in venues if _same_name(v["name"], pq.location) or
                          (len(pq.location) > 5 and pq.location.lower().rstrip("s") in v["name"].lower())), None)
            if named:
                named["requested"] = True
        if named:
            venues = [named] + [v for v in venues if not (_same_name(v["name"], named["name"]) or
                                                          haversine_km(v["lat"], v["lon"], named["lat"], named["lon"]) < 0.3)]
            venues = venues[:10]
        if not venues:
            venues = [_pseudo_venue(geo, f"{geo['name']} (no mapped {act.name.lower()} venue found within {radius:.0f} km)")]
    res["understood"]["radius_km"] = round(radius, 1)

    # ---------------------------------------------------------------- live data, in parallel
    points = [(v["lat"], v["lon"]) for v in venues]
    days = (max(win.days) - now.date()).days + 2
    try:
        wx, aq, trav, (ctx, ctx_src) = await asyncio.gather(
            wxapi.forecast(points, days), wxapi.air_quality(points, min(days, 7)),
            travel((geo["lat"], geo["lon"]), points, act.travel_mode), find_context(points))
    except FetchError as e:
        res.update(status="no_data", answer=(
            "I couldn't reach the live weather service just now, and I won't guess the weather. "
            f"Please try again in a minute. ({e})"))
        return finish(res)

    # ---------------------------------------------------------------- score place x hour
    slots = win.slots(now)
    scored = []
    for i, v in enumerate(venues):
        pc = place_context(v, ctx)
        bonus, site_why = site_bonus(act, v, pc)
        tmin = None if v.get("pseudo") else trav[i]["minutes"]
        sun = {}
        for d, s in wx[i]["daily"].items():
            sun[d] = {"sunrise": datetime.fromisoformat(s["sunrise"]), "sunset": datetime.fromisoformat(s["sunset"])}
        hours: dict[datetime, HourScore] = {}
        needed = sorted({t + timedelta(hours=k) for t in slots for k in range(act.duration_h)})
        for t in needed:
            key = t.strftime("%Y-%m-%dT%H:00")
            if key not in wx[i]["hourly"]:
                continue
            hours[t] = score_hour(act, lim, v, pc, t, wx[i]["hourly"][key], (aq[i].get(key) if aq else None),
                                  sun.get(t.date().isoformat(), {}), tmin)
        scored.append({"venue": v, "pc": pc, "bonus": bonus, "site_why": site_why, "travel": trav[i] if not v.get("pseudo") else None,
                       "hours": hours, "sun": sun})

    starts = _candidate_starts(slots, win, scored[0]["sun"] if scored else {}, act)
    slot_rows = []
    for s in scored:
        for t in starts:
            block = [s["hours"].get(t + timedelta(hours=k)) for k in range(act.duration_h)]
            if not block or block[0] is None:
                continue
            block = [b for b in block if b is not None]
            vals = [b.score for b in block]
            veto = next((b.veto for b in block if b.veto), None)
            raw = max(0.0, min(100.0, 0.6 * min(vals) + 0.4 * sum(vals) / len(vals) + s["bonus"]))
            slot_rows.append({"s": s, "start": t, "end": t + timedelta(hours=len(block)),
                              "score": round(min(raw, 15.0) if veto else raw, 1), "raw": raw, "veto": veto, "block": block})
    if not slot_rows:
        res.update(status="no_data", answer="The forecast doesn't cover the hours you asked about yet. Try a nearer time.")
        return finish(res)

    # Best distinct venues, each at its own best start time. A venue the user named always comes first.
    slot_rows.sort(key=lambda r: (r["veto"] is None, r["raw"]), reverse=True)  # least-bad still ranks sensibly
    recs, used = [], set()
    named_rows = [r for r in slot_rows if r["s"]["venue"].get("requested")]
    if named_rows:
        recs.append(named_rows[0])
        used.add(named_rows[0]["s"]["venue"]["id"])
    for r in slot_rows:
        vid = r["s"]["venue"]["id"]
        if vid in used:
            continue
        used.add(vid)
        recs.append(r)
        if len(recs) == 3:
            break

    res.update(
        status="ok",
        recommendations=[_rec_dict(k + 1, r, act, win) for k, r in enumerate(recs)],
        by_day=_by_day(slot_rows),
        avoid=_avoid_windows(scored, slots, act, recs[0]),
        downwind_insight=_downwind_insight(recs, slots),
        exact_hour_check=_exact_check(scored, win, recs) if win.exact_hour is not None else None,
        advice=_advice(act, pq, recs, scored),
        data_notes=_data_notes(aq, wx, slots, osm_src, ctx_src, trav, venues),
        geojson=_geojson(geo, recs, scored),
    )
    res["understood"]["origin"] = {"name": geo["display_name"], "lat": geo["lat"], "lon": geo["lon"]}

    from .narrate import narrate  # local import: narrate depends on result shape defined here
    await narrate(res, llm)
    return finish(res)


# ---------------------------------------------------------------------------- helpers

VENUE_KEYS = {"leisure", "natural", "tourism", "historic", "boundary", "golf"}  # always count as a destination
VENUE_LANDUSE = {"recreation_ground", "forest", "grass", "meadow", "village_green"}


def _named_venue(geo: dict, act) -> dict | None:
    """If the place the user named is itself a venue (a park, a fort, a beach...), evaluate it directly."""
    if geo.get("is_city") or not geo.get("osm_key"):
        return None
    tags = {geo["osm_key"]: geo.get("osm_value") or "", "name": geo["name"]}
    m = _match(tags, act)
    if not m and not (geo["osm_key"] in VENUE_KEYS or (geo["osm_key"] == "landuse" and geo.get("osm_value") in VENUE_LANDUSE)):
        return None  # a neighbourhood, road or building: search around it instead
    suit, kind = (m if m else (0.7, (geo.get("osm_value") or "place").replace("_", " ")))
    bb = geo.get("bbox") or [geo["lat"]] * 2 + [geo["lon"]] * 2
    h = (bb[1] - bb[0]) * 110540
    w = (bb[3] - bb[2]) * 111320 * math.cos(math.radians(geo["lat"]))
    return {"id": geo.get("osm_id") or "named", "cls": "venue", "name": geo["name"], "lat": geo["lat"], "lon": geo["lon"],
            "bbox": bb, "area": abs(h * w) * 0.6, "tags": tags, "geom": None, "suit": suit, "kind": kind,
            "dist_km": 0.0, "requested": True}


def _pseudo_venue(geo: dict, label: str) -> dict:
    return {"id": "origin", "cls": "venue", "name": label, "lat": geo["lat"], "lon": geo["lon"],
            "bbox": [geo["lat"]] * 2 + [geo["lon"]] * 2, "area": 0, "tags": {}, "geom": None,
            "suit": 0.6, "kind": "area", "dist_km": 0.0, "pseudo": True}


def _candidate_starts(slots: list[datetime], win, sun: dict, act) -> list[datetime]:
    if not win.sun_event:
        if win.exact_hour is not None:
            return slots
        lo, hi = act.start_hours
        sensible = [t for t in slots if lo <= t.hour <= hi]
        return sensible or slots
    out = []
    for t in slots:
        s = sun.get(t.date().isoformat(), {}).get(win.sun_event)
        if s is None:
            continue
        lo, hi = (s - timedelta(hours=2), s + timedelta(minutes=10)) if win.sun_event == "sunset" else \
                 (s - timedelta(minutes=40), s + timedelta(hours=1, minutes=30))
        if lo <= t <= hi:
            out.append(t)
    return out or slots


def _rec_dict(rank: int, r: dict, act, win) -> dict:
    s, v = r["s"], r["s"]["venue"]
    first = r["block"][0]
    worst = min(r["block"], key=lambda b: b.score)
    cautions = list(dict.fromkeys([x for b in r["block"] for x in b.reasons]))[:5]
    why = list(dict.fromkeys(first.positives + s["site_why"]))[:5]
    return {
        "rank": rank, "place": v["name"], "kind": v.get("kind"), "osm_id": v["id"], "lat": v["lat"], "lon": v["lon"],
        "distance_km": v.get("dist_km"),
        "travel": ({"minutes_free_flow": s["travel"]["minutes"], "km": s["travel"]["km"],
                    "estimated": s["travel"]["estimated"], "mode": act.travel_mode} if s["travel"] else None),
        "start": r["start"].isoformat(timespec="minutes"), "end": r["end"].isoformat(timespec="minutes"),
        "when": f"{r['start']:%a %d %b}, {r['start']:%H:%M}–{r['end']:%H:%M}",
        "score": r["score"], "verdict": _verdict(r["score"], r["veto"]), "veto": r["veto"],
        "conditions_at_start": first.metrics, "worst_hour": {"time": worst.time.strftime("%H:%M"), **worst.metrics},
        "why": why, "cautions": cautions, "downwind": first.downwind[:2],
        "context": s["pc"].summary(),
    }


def _verdict(score: float, veto: str | None) -> str:
    if veto:
        return "avoid"
    return "great" if score >= 80 else "good" if score >= 65 else "okay" if score >= 50 else "poor"


def _by_day(rows: list[dict]) -> list[dict]:
    """Per day: the best slot, plus a day score that also rewards having several good options."""
    best: dict = {}
    per_day: dict = {}
    for r in rows:
        d = r["start"].date()
        per_day.setdefault(d, []).append(r["score"])
        if d not in best or r["score"] > best[d]["score"]:
            best[d] = r
    out = []
    for d in sorted(best):
        r = best[d]
        top = sorted(per_day[d], reverse=True)[:6]
        day_score = round(0.6 * r["score"] + 0.4 * sum(top) / len(top), 1)
        good_hours = sum(1 for x in per_day[d] if x >= 65)
        out.append({"day": d.strftime("%A %d %b"), "day_score": day_score, "good_slots": good_hours,
                    "best_score": r["score"], "verdict": _verdict(r["score"], r["veto"]),
                    "best_place": r["s"]["venue"]["name"], "best_time": f"{r['start']:%H:%M}–{r['end']:%H:%M}",
                    "main_issue": (r["veto"] or (r["block"][0].reasons[0] if r["block"][0].reasons else None))})
    return out


def _avoid_windows(scored: list[dict], slots: list[datetime], act, top: dict) -> list[dict]:
    flagged: list[tuple[datetime, str, str]] = []
    for t in slots:
        if top["veto"] is None and top["start"] <= t < top["end"]:
            continue  # never tell people to avoid the slot we recommend
        hs = [s["hours"][t] for s in scored if t in s["hours"]]
        if not hs:
            continue
        vetoes = [h.veto for h in hs if h.veto]
        if len(vetoes) >= max(1, 0.6 * len(hs)):
            cat = Counter(_category(v) for v in vetoes).most_common(1)[0][0]
            example = next(v for v in vetoes if _category(v) == cat)
            flagged.append((t, cat, example))
        elif sum(h.score for h in hs) / len(hs) < 35:
            reasons = [h.reasons[0] for h in hs if h.reasons]
            if reasons:
                cat = Counter(_category(r) for r in reasons).most_common(1)[0][0]
                flagged.append((t, cat, next(r for r in reasons if _category(r) == cat)))
    out: list[dict] = []
    for t, cat, ex in flagged:
        if out and out[-1]["_cat"] == cat and out[-1]["_end"] == t:
            out[-1]["_end"] = t + timedelta(hours=1)
            out[-1]["examples"].append(ex)
        else:
            out.append({"_cat": cat, "_start": t, "_end": t + timedelta(hours=1), "examples": [ex]})
    return [{"from": _hm(w["_start"]), "to": w["_end"].strftime("%H:%M"), "reason": w["_cat"],
             "detail": w["examples"][0] if w["examples"][0] == w["examples"][-1] else f"{w['examples'][0]} … {w['examples'][-1]}"}
            for w in out]


def _downwind_insight(recs: list[dict], slots: list[datetime]) -> str | None:
    """The place-and-hour story: same venue, different hour, different exposure because the wind shifts."""
    for r in recs:
        hs = [h for t, h in r["s"]["hours"].items() if t in set(slots)]
        if len(hs) < 2:
            continue
        worst = max(hs, key=lambda h: h.metrics["downwind_exposure"])
        best_h = r["block"][0]
        wx_, bx_ = worst.metrics, best_h.metrics
        if not worst.downwind or wx_["downwind_exposure"] < 0.3 or wx_["downwind_exposure"] - bx_["downwind_exposure"] < 0.2:
            continue
        name = r["s"]["venue"]["name"]
        if wx_["wind_from"] != bx_["wind_from"]:
            return (f"At {worst.time:%a %H:%M}, {name} is downwind of {worst.downwind[0].split(':')[0]} "
                    f"(wind from the {wx_['wind_from']}, {wx_['wind_kmh']:.0f} km/h). The wind "
                    f"comes from the {bx_['wind_from']} at {best_h.time:%H:%M}, so the recommended slot avoids that plume.")
        if "road" in worst.downwind[0].lower() or traffic_level(worst.time) > traffic_level(best_h.time):
            return (f"At {worst.time:%a %H:%M}, {name} gets exhaust from {worst.downwind[0].split(':')[0]} "
                    f"(wind from the {wx_['wind_from']}, peak traffic). At {best_h.time:%H:%M} traffic is lighter.")
    return None


def _exact_check(scored, win, recs) -> dict | None:
    target_hours = [t for t in (recs[0]["s"]["hours"] if recs else {}) if t.hour == win.exact_hour]
    if not target_hours:
        return None
    t = min(target_hours)
    h = recs[0]["s"]["hours"][t]
    return {"time": _hm(t), "place": recs[0]["s"]["venue"]["name"], "score": h.score,
            "verdict": _verdict(h.score, h.veto), "veto": h.veto, "issues": h.reasons[:3], "conditions": h.metrics}


def _advice(act, pq, recs, scored) -> dict:
    carry, watch = list(act.carry), list(act.watch)
    ms = [b.metrics for r in recs[:1] for b in r["block"]]
    if ms:
        if max(m["feels_like_c"] or 0 for m in ms) >= 32:
            carry.append("extra water / electrolytes (feels like {:.0f}°C)".format(max(m["feels_like_c"] for m in ms)))
        if max(m["uv_index"] or 0 for m in ms) >= 6:
            carry.append("sunscreen SPF 30+ and a hat (UV {:.0f})".format(max(m["uv_index"] for m in ms)))
        if max(m["rain_chance_pct"] or 0 for m in ms) >= 30:
            carry.append("umbrella or rain jacket (rain chance up to {}%)".format(max(m["rain_chance_pct"] for m in ms)))
        aqis = [m["us_aqi"] for m in ms if m["us_aqi"] is not None]
        if aqis and max(aqis) > 100:
            carry.append(f"an N95 mask if you feel irritation (US AQI up to {max(aqis)})")
    for p in pq.profiles:
        carry.extend(PROFILES[p].advice)
    if recs and recs[0]["s"]["sun"]:
        d = recs[0]["start"].date().isoformat()
        ss = recs[0]["s"]["sun"].get(d, {}).get("sunset")
        if ss and act.daylight != "any":
            watch.append(f"sunset is at {ss:%H:%M}")
    if recs and pq.profiles and recs[0]["s"]["pc"].hospital:
        h = recs[0]["s"]["pc"].hospital
        watch.append(f"nearest hospital mapped in OSM: {h[0]} (~{h[1] / 1000:.1f} km)")
    return {"carry": list(dict.fromkeys(carry))[:9], "watch_for": list(dict.fromkeys(watch))[:7]}


def _data_notes(aq, wx, slots, osm_src, ctx_src, trav, venues) -> list[str]:
    notes = []
    if aq is None:
        notes.append("Air-quality service was unreachable; air quality was not scored.")
    elif slots and any(aq[0].get(t.strftime("%Y-%m-%dT%H:00"), {}).get("us_aqi") is None for t in slots):
        notes.append(f"Air-quality forecasts only extend ~{wxapi.MAX_AQ_DAYS} days; later hours are scored without AQI.")
    if "local" in (osm_src, ctx_src):
        notes.append("Overpass API was unavailable, so map features came from the offline OpenStreetMap extract (Geofabrik).")
    if any(t.get("estimated") for t in trav):
        notes.append("Some travel times are straight-line estimates (routing service unavailable).")
    notes.append("Travel times are free-flow (OSRM) with a time-of-day congestion multiplier; no live traffic feed is used.")
    notes.append("Downwind exposure is a geometric estimate from OSM roads/industry and forecast wind, not a measurement.")
    return notes


def _geojson(geo: dict, recs: list[dict], scored: list[dict]) -> dict:
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [geo["lon"], geo["lat"]]},
              "properties": {"role": "origin", "name": geo["display_name"]}}]
    rec_ids = {r["s"]["venue"]["id"]: k + 1 for k, r in enumerate(recs)}
    for s in scored:
        v = s["venue"]
        if v.get("pseudo"):
            continue
        rank = rec_ids.get(v["id"])
        best = max((r for r in recs if r["s"] is s), key=lambda r: r["score"], default=None)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [v["lon"], v["lat"]]},
                      "properties": {"role": "recommended" if rank else "candidate", "rank": rank, "name": v["name"],
                                     "kind": v.get("kind"), "score": best["score"] if best else None,
                                     "when": f"{best['start']:%a %H:%M}" if best else None}})
    return {"type": "FeatureCollection", "features": feats}
