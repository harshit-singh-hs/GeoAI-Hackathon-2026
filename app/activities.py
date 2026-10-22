"""Activity and personal-profile knowledge.

Thresholds here are rules of thumb (sports-medicine heat guidance, US-EPA AQI activity advice, WHO UV guidance),
kept deliberately simple and transparent so every recommendation can be traced to a rule and a measured value.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Selector:
    """OSM tag filter: every key must match its regex (Overpass-style). `weight` = how well such a venue suits."""
    tags: dict[str, str]
    weight: float
    label: str


@dataclass(frozen=True)
class Activity:
    id: str
    name: str
    synonyms: tuple[str, ...]
    selectors: tuple[Selector, ...]
    default_radius_km: float
    duration_h: int
    feels_ok: tuple[float, float]  # comfortable apparent-temperature band, °C
    feels_max: float  # above this apparent temperature: avoid
    feels_min: float  # below this: avoid
    exertion: float  # 0..1, scales air-quality and heat sensitivity
    rain_sens: float  # 0..1
    wind_max: float  # sustained km/h above which play/activity suffers badly
    gust_max: float
    uv_sens: float  # 0..1 (exposure duration x skin exposure)
    daylight: str  # "required" | "preferred" | "any"
    lit_ok_at_night: bool = False  # venue tagged lit=yes may be used after dark
    visibility_matters: bool = False
    golden_hour_bonus: bool = False
    likes_water: float = 0.0
    likes_green: float = 0.5
    travel_mode: str = "driving"
    preferred_hours: tuple[int, ...] = ()
    start_hours: tuple[int, int] = (5, 21)  # sensible start times when the user gives no exact time
    carry: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()


PARK = {"leisure": "^park$"}
REC = {"landuse": "^recreation_ground$"}
GARDEN = {"leisure": "^garden$"}
RESERVE = {"leisure": "^nature_reserve$"}
BEACH = {"natural": "^beach$"}


def S(tags, w, label):
    return Selector(tags, w, label)


ACTIVITIES: dict[str, Activity] = {a.id: a for a in [
    Activity(
        "running", "Running / jogging",
        ("run", "running", "jog", "jogging", "marathon", "sprint", "5k", "10k"),
        (S({"leisure": "^track$"}, 1.0, "running track"), S(PARK, 0.9, "park"), S(REC, 0.75, "recreation ground"),
         S(RESERVE, 0.8, "nature reserve"), S(BEACH, 0.75, "beach"), S(GARDEN, 0.6, "garden"),
         S({"leisure": "^stadium$"}, 0.6, "stadium")),
        6, 1, (10, 25), 35, 2, 1.0, 0.6, 35, 50, 0.6, "preferred", lit_ok_at_night=True, likes_green=0.8,
        start_hours=(5, 21),
        preferred_hours=(5, 6, 7, 8, 17, 18, 19),
        carry=("water bottle", "light breathable clothing", "phone with emergency contact"),
        watch=("heat and humidity during the run", "uneven surfaces"),
    ),
    Activity(
        "cycling", "Cycling (leisure or commute)",
        ("cycle", "cycling", "bike", "biking", "bicycle", "ride to work", "cycle to work"),
        (S({"highway": "^cycleway$", "name": "."}, 1.0, "cycleway"), S(PARK, 0.6, "park"), S(RESERVE, 0.7, "nature reserve"),
         S({"leisure": "^track$", "sport": "cycling"}, 1.0, "cycling track")),
        10, 1, (10, 27), 36, 2, 0.9, 0.8, 25, 40, 0.7, "preferred", lit_ok_at_night=True, likes_green=0.6,
        start_hours=(5, 20),
        travel_mode="cycling", preferred_hours=(6, 7, 8, 17, 18),
        carry=("helmet", "front and rear lights", "water", "puncture kit"),
        watch=("wet roads reduce braking", "traffic at junctions", "crosswinds on open stretches"),
    ),
    Activity(
        "golf", "Golf",
        ("golf", "tee time", "driving range", "18 holes", "9 holes"),
        (S({"leisure": "^golf_course$"}, 1.0, "golf course"), S({"golf": "^driving_range$"}, 0.6, "driving range")),
        30, 4, (15, 29), 37, 8, 0.45, 0.9, 25, 40, 1.0, "required", start_hours=(6, 15),
        likes_green=1.0, preferred_hours=(6, 7, 8, 15, 16),
        carry=("sunscreen SPF 30+", "hat", "at least 1.5 L water", "umbrella"),
        watch=("lightning: leave the course at the first thunder", "midday UV over 4 hours"),
    ),
    Activity(
        "cricket", "Cricket (match or practice)",
        ("cricket", "nets", "net practice", "batting", "bowling"),
        (S({"leisure": "^pitch$", "sport": "cricket"}, 1.0, "cricket pitch"),
         S({"leisure": "^(stadium|sports_centre)$", "sport": "cricket"}, 0.85, "cricket ground"),
         S(REC, 0.6, "recreation ground"), S({"leisure": "^playground$"}, 0.3, "playground"), S(PARK, 0.4, "park")),
        10, 3, (15, 30), 38, 8, 0.65, 1.0, 35, 50, 1.0, "required", lit_ok_at_night=True, start_hours=(6, 18),
        preferred_hours=(7, 8, 15, 16, 17),
        carry=("water and electrolytes", "caps and sunscreen", "first-aid kit"),
        watch=("wet outfield after rain is slippery", "heat during long innings"),
    ),
    Activity(
        "football", "Outdoor football / five-a-side",
        ("football", "soccer", "five-a-side", "5-a-side", "futsal", "kickabout"),
        (S({"leisure": "^pitch$", "sport": "soccer"}, 1.0, "football pitch"),
         S({"leisure": "^(stadium|sports_centre)$", "sport": "soccer"}, 0.85, "football ground"),
         S(REC, 0.55, "recreation ground"), S({"leisure": "^pitch$", "sport": "multi"}, 0.7, "multi-sport pitch")),
        8, 2, (12, 28), 36, 5, 1.0, 0.5, 35, 55, 0.8, "preferred", lit_ok_at_night=True, start_hours=(6, 21),
        preferred_hours=(6, 7, 17, 18, 19, 20),
        carry=("water and electrolytes", "studs suited to the surface", "shin guards"),
        watch=("heat stress during sprints", "slippery turf after rain"),
    ),
    Activity(
        "tennis", "Outdoor tennis",
        ("tennis",),
        (S({"leisure": "^pitch$", "sport": "tennis"}, 1.0, "tennis court"),
         S({"leisure": "^sports_centre$", "sport": "tennis"}, 0.9, "tennis centre"),
         S({"club": "^sport$", "sport": "tennis"}, 0.8, "tennis club")),
        8, 2, (14, 29), 37, 6, 0.9, 1.0, 18, 30, 0.9, "preferred", lit_ok_at_night=True, start_hours=(6, 21),
        preferred_hours=(6, 7, 8, 17, 18, 19),
        carry=("water", "spare grips", "cap and sunscreen"),
        watch=("gusty wind affects ball flight", "wet courts are dangerous"),
    ),
    Activity(
        "hiking", "Hiking / nature walk",
        ("hike", "hiking", "trek", "trekking", "trail", "nature walk", "fort"),
        (S({"historic": "^fort$"}, 0.9, "fort"), S({"natural": "^peak$", "name": "."}, 0.85, "peak"),
         S(RESERVE, 0.9, "nature reserve"), S({"boundary": "^(national_park|protected_area)$"}, 0.8, "protected area"),
         S({"tourism": "^viewpoint$"}, 0.7, "viewpoint"), S({"natural": "^waterfall$"}, 0.6, "waterfall"),
         S({"natural": "^wood$", "name": "."}, 0.6, "woodland")),
        25, 4, (8, 26), 35, 0, 0.8, 0.9, 40, 60, 0.9, "required", visibility_matters=True, start_hours=(5, 13),
        likes_green=1.0, likes_water=0.3, preferred_hours=(6, 7, 8),
        carry=("2+ L water", "trail shoes with grip", "rain jacket", "torch", "basic first aid", "offline map"),
        watch=("slippery rock and mud after rain", "return before sunset", "phone signal can drop"),
    ),
    Activity(
        "walking", "Walk / stroll in a park",
        ("walk", "walking", "stroll", "morning walk", "evening walk", "brisk walk"),
        (S(PARK, 1.0, "park"), S(GARDEN, 0.9, "garden"), S(RESERVE, 0.8, "nature reserve"),
         S(BEACH, 0.8, "beach"), S(REC, 0.6, "recreation ground"), S({"leisure": "^track$"}, 0.7, "track")),
        5, 1, (12, 29), 38, 4, 0.4, 0.6, 40, 60, 0.6, "preferred", lit_ok_at_night=True, likes_green=1.0,
        start_hours=(5, 20),
        likes_water=0.3, preferred_hours=(6, 7, 8, 17, 18),
        carry=("water", "comfortable shoes", "cap"),
        watch=("uneven footpaths", "heat around midday"),
    ),
    Activity(
        "picnic", "Picnic / park outing",
        ("picnic", "park outing", "outing", "day out", "family outing"),
        (S({"tourism": "^picnic_site$"}, 1.0, "picnic site"), S(PARK, 0.9, "park"), S(GARDEN, 0.85, "garden"),
         S(BEACH, 0.6, "beach"), S(REC, 0.6, "recreation ground")),
        15, 3, (17, 30), 37, 10, 0.2, 1.0, 30, 45, 0.8, "required", likes_green=1.0, likes_water=0.5,
        start_hours=(7, 16),
        preferred_hours=(8, 9, 10, 16, 17),
        carry=("mat or sheet", "shade or umbrella", "drinking water", "mosquito repellent", "trash bags"),
        watch=("food spoils fast in heat", "mosquitoes near water around dusk"),
    ),
    Activity(
        "event", "Outdoor event / gathering",
        ("event", "party", "birthday", "wedding", "reception", "gathering", "fair", "festival", "get-together", "function"),
        (S({"amenity": "^events_venue$"}, 0.95, "events venue"), S(REC, 0.8, "recreation ground"),
         S(PARK, 0.7, "park"), S(GARDEN, 0.7, "garden")),
        12, 4, (18, 30), 38, 12, 0.2, 1.0, 25, 40, 0.6, "any", likes_green=0.7, start_hours=(8, 20),
        preferred_hours=(16, 17, 18, 19, 20),
        carry=("rain-proof canopy/tent", "tie-downs for decor", "drinking water", "mosquito repellent", "power backup"),
        watch=("gusts lifting tents and decor", "sudden showers", "guests in direct sun"),
    ),
    Activity(
        "photography", "Outdoor photography",
        ("photo", "photos", "photography", "photoshoot", "photo shoot", "sunset", "sunrise", "golden hour", "pictures", "shoot"),
        (S({"tourism": "^viewpoint$"}, 1.0, "viewpoint"), S(BEACH, 0.9, "beach"), S({"man_made": "^pier$"}, 0.85, "pier"),
         S({"historic": "^(fort|monument|castle|ruins)$"}, 0.8, "heritage site"),
         S({"natural": "^water$", "name": "."}, 0.75, "lake / water body"), S({"tourism": "^attraction$"}, 0.6, "attraction"),
         S(PARK, 0.5, "park")),
        12, 1, (5, 35), 40, 0, 0.2, 0.7, 40, 60, 0.3, "any", visibility_matters=True, golden_hour_bonus=True,
        start_hours=(5, 19),
        likes_water=0.8, preferred_hours=(6, 17, 18),
        carry=("lens cloth (humidity fogs lenses)", "rain cover for the camera", "spare batteries", "tripod"),
        watch=("haze cuts contrast", "low cloud can block the sun at the horizon"),
    ),
]}


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    keywords: tuple[str, ...]
    heat_delta: float = 0.0  # added to feels_max (negative = tighter)
    cold_delta: float = 0.0  # added to feels_min
    aqi_factor: float = 1.0  # multiplies AQ penalties
    aqi_limit: int = 200  # US AQI above which we say avoid
    uv_factor: float = 1.0
    max_travel_min: float = 60
    daylight: bool = False  # prefer daylight hours (safety) for this profile
    advice: tuple[str, ...] = ()


PROFILES: dict[str, Profile] = {p.id: p for p in [
    Profile("asthma", "asthma / respiratory condition",
            ("asthma", "asthmatic", "copd", "bronchitis", "breathing problem", "lung", "wheez", "allerg"),
            cold_delta=4, aqi_factor=1.8, aqi_limit=150, daylight=True,
            advice=("carry your reliever inhaler", "avoid hours downwind of busy roads or industry",
                    "warm up gradually; cold dry air can trigger symptoms")),
    Profile("senior", "senior citizen",
            ("senior", "elderly", "old people", "aged", "parents", "grandparent", "grandparents", "grandfather",
             "grandmother", "grandpa", "grandma", "retired"),
            heat_delta=-4, cold_delta=3, aqi_factor=1.5, aqi_limit=150, uv_factor=1.2, max_travel_min=30, daylight=True,
            advice=("prefer flat, well-lit paths with benches", "carry water and any regular medication",
                    "avoid the hottest hours")),
    Profile("child", "children",
            ("kid", "kids", "child", "children", "son", "daughter", "toddler", "baby", "infant", "school"),
            heat_delta=-3, aqi_factor=1.6, aqi_limit=150, uv_factor=1.4, max_travel_min=40, daylight=True,
            advice=("children dehydrate faster: plan water breaks every 20 minutes", "apply sunscreen and use caps")),
    Profile("pregnancy", "pregnancy",
            ("pregnant", "pregnancy", "expecting", "trimester"),
            heat_delta=-5, aqi_factor=1.6, aqi_limit=150, uv_factor=1.2, max_travel_min=30, daylight=True,
            advice=("avoid overheating and strenuous exertion", "stay close to shade and toilets",
                    "keep water and a snack handy")),
    Profile("heart", "heart condition / high blood pressure",
            ("heart", "cardiac", "blood pressure", "hypertension", "bp patient"),
            heat_delta=-4, aqi_factor=1.7, aqi_limit=150,
            advice=("keep exertion moderate", "carry prescribed medication", "stop if you feel chest discomfort")),
]}


def apply_profiles(act: Activity, profiles: list[str]) -> tuple[Activity, dict]:
    """Tighten an activity's limits for the user's profiles. Returns (adjusted activity, combined limits)."""
    ps = [PROFILES[p] for p in profiles if p in PROFILES]
    lim = {
        "aqi_factor": max([p.aqi_factor for p in ps], default=1.0),
        "aqi_limit": min([p.aqi_limit for p in ps], default=200),
        "daylight": any(p.daylight for p in ps),
        "labels": [p.label for p in ps],
        "uv_factor": max([p.uv_factor for p in ps], default=1.0),
        "max_travel_min": min([p.max_travel_min for p in ps], default=60),
    }
    adj = replace(
        act,
        feels_max=act.feels_max + min([p.heat_delta for p in ps], default=0),
        feels_min=act.feels_min + max([p.cold_delta for p in ps], default=0),
        feels_ok=(act.feels_ok[0], min(act.feels_ok[1], act.feels_max + min([p.heat_delta for p in ps], default=0) - 3)),
    )
    return adj, lim
