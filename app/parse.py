"""Free-form question -> structured request.

Two extractors run on every query: the SLM (few-shot or fine-tuned JSON extraction) and a deterministic rule
extractor. SLM output is validated against the query text (a location must literally appear in the question,
a radius must be a number the user typed, activities/profiles must come from our closed lists), and the rules
fill any gaps. If the SLM is unavailable the rules alone still produce a usable request.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .activities import ACTIVITIES, PROFILES

INTENTS = ("where", "when", "go_no_go", "compare", "advice")


@dataclass
class ParsedQuery:
    activity: str | None = None
    activity_text: str | None = None  # what the user called it (kept for unsupported activities)
    location: str | None = None
    radius_km: float | None = None
    time_text: str | None = None
    profiles: list[str] = field(default_factory=list)
    intent: str = "where"
    commute: bool = False
    out_of_scope: bool = False
    parser: str = "rules"

    def to_dict(self) -> dict:
        return asdict(self)


# Outdoor things people ask about that we do not model specifically.
UNSUPPORTED_OUTDOOR = ("swim", "swimming", "kite", "skate", "skating", "skateboard", "surf", "kayak", "boating",
                       "fishing", "camping", "paragliding", "badminton", "basketball", "volleyball", "horse riding",
                       "stargazing", "birdwatching", "bird watching", "yoga", "rock climbing", "skiing")
INDOOR_OR_OFFTOPIC = ("movie", "restaurant", "recipe", "stock", "flight", "hotel", "code", "python", "capital of",
                      "who is", "translate", "homework", "gym", "indoor", "mall", "shopping")

TIME_WORDS = (r"today|tonight|tomorrow|tmrw|weekend|morning|afternoon|evening|night|sunrise|sunset|noon|"
              r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun|"
              r"next|this|after|before|at \d|by \d|\d+ ?(?:am|pm)|now|week|days?|hours?|golden hour|dawn|dusk|lunch")
STOP = (r"(?=\s*(?:[,?.!;]|$|\bfor\b|\bwith\b|\bwithin\b|\btomorrow\b|\btoday\b|\btonight\b|\bthis\b|\bnext\b|"
        r"\bon\b|\bat\b|\bduring\b|\bin the\b|\bbetween\b|\bbefore\b|\bafter\b|\bover\b|\bwhere\b|\bwhen\b|"
        r"\bany\b|\bis\b|\bare\b|\bshould\b|\bwhat\b|\bif\b|\bmorning\b|\bevening\b|\bafternoon\b|\bnight\b|"
        r"\bsaturday\b|\bsunday\b|\bmonday\b|\btuesday\b|\bwednesday\b|\bthursday\b|\bfriday\b|\bweekend\b))")


def _find_activity(t: str) -> tuple[str | None, str | None]:
    best = None
    for aid, a in ACTIVITIES.items():
        for syn in a.synonyms:
            m = re.search(rf"\b{re.escape(syn)}\b", t)
            if m:
                # Sun events only decide the activity when nothing else does ("run at sunset" is running).
                weak = syn in ("sunset", "sunrise", "golden hour", "shoot", "outing", "day out", "function")
                key = (0 if weak else 1, len(syn))
                if best is None or key > best[0]:
                    best = (key, aid, m.group(0))
    if best:
        return best[1], best[2]
    for w in UNSUPPORTED_OUTDOOR:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return None, w
    return None, None


NOT_PLACES = (r"^(the|my|our|a|an|me|work|home|office|mind|general|time|advance|future|rain|sun|shade|summer|winter|"
              r"monsoon|it|this|that|there|here|all|kids?|family|friends|morning|evening|night|afternoon|weekend|"
              r"order|case|between|total|person|public|groups?|team|club|park|parks|garden|india|january|february|march|april|"
              r"may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|"
              r"saturday|sunday|am|pm|i)\b")


def _trim_location(cand: str) -> str:
    """Cut a candidate at the first time/stop word ("thane tomorrow" -> "thane")."""
    words = cand.split()
    for i, w in enumerate(words):
        if i and re.fullmatch(rf"(?:{TIME_WORDS}|for|with|within|on|at|is|are|and|or|to|please|pls)", w.lower().strip(",.?!")):
            return " ".join(words[:i])
    return cand


def _find_location(q: str) -> str | None:
    pats = [
        rf"\b(?:km|kms|kilometers?|kilometres?|minutes?|mins?)\s+(?:of|from|around)\s+([A-Za-z][\w .'-]+?){STOP}",
        r"\b(?:near|nearby|around|close to|in|from|at|towards|to)\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*(?:,\s*[A-Z][\w.'-]*)?)\b",
        rf"\b(?:near|nearby|around|close to)\s+([a-z][\w .'-]+?){STOP}",
        rf"\b(?:in|at|from)\s+([a-z][\w.'-]+(?:\s+[a-z][\w.'-]+)?){STOP}",
    ]
    for p in pats:
        for m in re.finditer(p, q):
            cand = _trim_location(m.group(1).strip(" ,.'"))
            if cand and not re.fullmatch(TIME_WORDS, cand.lower()) and not re.match(NOT_PLACES, cand.lower()):
                return cand
    return None


def _find_profiles(t: str) -> list[str]:
    out = []
    for pid, p in PROFILES.items():
        if any(re.search(rf"\b{re.escape(k)}", t) for k in p.keywords):
            out.append(pid)
    for m in re.finditer(r"\b(\d{1,3})[ -]?(?:years?|yrs?|yo)\b(?:[ -]?old)?|\baged? (\d{1,3})\b", t):
        age = int(m.group(1) or m.group(2))
        if age >= 60 and "senior" not in out:
            out.append("senior")
        elif age <= 12 and "child" not in out:
            out.append("child")
    return out


def _find_intent(t: str) -> str:
    if re.search(r"\b(or|vs\.?|versus|better)\b", t) and re.search(r"\b(saturday|sunday|today|tomorrow|morning|evening|weekday|\w+day)\b.*\b(or|vs)\b", t):
        return "compare"
    if re.search(r"^(is|are|can|should|will|would)\b|\b(okay|ok|safe|good idea)\b.*\?", t) and not re.search(r"\bwhere\b", t):
        return "go_no_go"
    if re.search(r"\b(what should|keep in mind|tips|prepare|precautions?)\b", t):
        return "advice"
    if re.search(r"\b(when|what time|best time|which time)\b", t) and not re.search(r"\bwhere\b|\bplace|spot", t):
        return "when"
    return "where"


def _time_text(t: str) -> str | None:
    hits = re.findall(rf"(?:\b(?:{TIME_WORDS})\b[\w ]*?)+", t)
    hits = [h.strip() for h in hits if h.strip()]
    return " ".join(hits) if hits else None


def parse_rules(query: str) -> ParsedQuery:
    t = query.lower()
    aid, atext = _find_activity(t)
    pq = ParsedQuery(activity=aid, activity_text=atext, location=_find_location(query), profiles=_find_profiles(t),
                     intent=_find_intent(t), time_text=_time_text(t))
    if m := re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kms|kilomet(?:er|re)s?)\b", t):
        pq.radius_km = float(m.group(1))
    elif m := re.search(r"(\d+)\s*(?:min|mins|minutes)\b", t):
        pq.radius_km = round(int(m.group(1)) * 0.5, 1)  # ~30 km/h city driving
    pq.commute = bool(re.search(r"\b(to work|to office|to college|to school|commute)\b", t))
    if aid is None and atext is None and (any(w in t for w in INDOOR_OR_OFFTOPIC) or not re.search(
            r"\b(outdoor|outside|out|go|walk|play|park|weather|place|spot)\b", t)):
        pq.out_of_scope = True
    return pq


# ------------------------------------------------------------------ SLM extraction

PARSE_SYSTEM = """You extract structured fields from a question about an outdoor activity. Reply with JSON only.
Fields:
- activity: one of [""" + ", ".join(ACTIVITIES) + """] or null if none fits
- activity_text: the activity words used in the question, or null
- location: the place name exactly as written in the question, or null if no place is named
- radius_km: number if the user gave a distance, else null
- time_text: the time words exactly as written (e.g. "tomorrow evening", "this weekend", "at 8 AM"), or null
- profiles: list from [""" + ", ".join(PROFILES) + """] that the question mentions (e.g. kids -> child, parents/elderly -> senior)
- intent: one of [where, when, go_no_go, compare, advice]
- commute: true if travelling to work/college
- out_of_scope: true if the question is not about doing something outdoors
Never invent a place or a time that is not in the question."""

FEW_SHOT = [
    ("Can I take my 6 year old to play football near Andheri East this Sunday morning?",
     {"activity": "football", "activity_text": "play football", "location": "Andheri East", "radius_km": None,
      "time_text": "this Sunday morning", "profiles": ["child"], "intent": "go_no_go", "commute": False, "out_of_scope": False}),
    ("best picnic spots within 20 km of Hebbal tomorrow?",
     {"activity": "picnic", "activity_text": "picnic", "location": "Hebbal", "radius_km": 20, "time_text": "tomorrow",
      "profiles": [], "intent": "where", "commute": False, "out_of_scope": False}),
    ("what's a good recipe for biryani",
     {"activity": None, "activity_text": None, "location": None, "radius_km": None, "time_text": None, "profiles": [],
      "intent": "advice", "commute": False, "out_of_scope": True}),
]


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*}", "}", m.group(0).replace("None", "null").replace("True", "true").replace("False", "false")))
        except json.JSONDecodeError:
            return None


def parse_messages(query: str, few_shot: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": PARSE_SYSTEM}]
    for q, a in (FEW_SHOT if few_shot else []):
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
    msgs.append({"role": "user", "content": query})
    return msgs


def merge(query: str, rules: ParsedQuery, slm: dict | None) -> ParsedQuery:
    """Validate SLM fields against the question text and closed vocabularies; fall back to rules per field."""
    if not slm:
        return rules
    ql = query.lower()
    out = ParsedQuery(**rules.to_dict())
    out.parser = "slm+rules"
    act = slm.get("activity")
    if act in ACTIVITIES:
        # Trust the SLM's activity unless the rules found an explicit, different keyword.
        if rules.activity is None or rules.activity == act:
            out.activity = act
        out.activity_text = slm.get("activity_text") or rules.activity_text
    loc = slm.get("location")
    if isinstance(loc, str) and loc.strip() and loc.strip().lower() in ql:
        out.location = loc.strip()
    r = slm.get("radius_km")
    if isinstance(r, (int, float)) and r > 0 and re.search(rf"\b{int(r)}\b", ql):
        out.radius_km = float(r)
    tt = slm.get("time_text")
    if isinstance(tt, str) and tt.strip() and all(w in ql for w in tt.lower().split()):
        out.time_text = tt.strip()
    profs = [p for p in slm.get("profiles") or [] if p in PROFILES]
    out.profiles = sorted(set(rules.profiles) | set(profs))
    if slm.get("intent") in INTENTS:
        out.intent = slm["intent"]
    out.commute = rules.commute or bool(slm.get("commute"))
    # Out-of-scope needs both to agree, unless the SLM found a valid activity.
    out.out_of_scope = out.activity is None and (rules.out_of_scope and rules.activity_text is None or bool(slm.get("out_of_scope")))
    return out


async def parse(query: str, llm=None) -> ParsedQuery:
    rules = parse_rules(query)
    if llm is None or not llm.ready:
        return rules
    raw = await llm.chat(parse_messages(query, few_shot=not llm.finetuned), max_new_tokens=160, temperature=0.0)
    return merge(query, rules, _extract_json(raw or ""))
