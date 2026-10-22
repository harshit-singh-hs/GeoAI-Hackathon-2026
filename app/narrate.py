"""Turn the scored result into the user-facing answer.

The SLM writes the answer from a compact FACT SHEET only. A grounding check then verifies that every number and
every place-like name in the answer appears in the fact sheet; if not, we retry once and otherwise fall back to a
deterministic template. Either way the answer can only contain facts fetched for this query.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

NARRATE_SYSTEM = """You are an outdoor-activity planning assistant. Write the reply to the user's question using ONLY the FACTS.
Rules:
- First line: restate the DIRECT ANSWER in bold, in plain words (keep its place, time and yes/no/not-recommended decision exactly), then one sentence on the key conditions.
- If the DIRECT ANSWER starts with NOT RECOMMENDED or NO, the reply must start with "**Not recommended" or "**No", then give the least-bad option.
- Write days the way the FACTS do (e.g. "Sat 24 Oct"); say "today" or "tomorrow" only if that matches NOW.
- Describe air quality with the label given after "=" (e.g. "unhealthy"), never with your own judgement.
- Then up to 3 options as a short numbered list: place, distance, day and time, score, and the single most important reason.
- Then one line each for **Carry:**, **Watch for:** and **Avoid:**, picked from the FACTS.
- Do not copy the FACTS line by line; summarise. Copy numbers exactly. Never add a place, number or weather fact that is not in the FACTS.
- Under 170 words. Markdown."""

EXAMPLE_FACTS = """QUESTION: tennis after 5 pm today near Adyar?
ACTIVITY: Outdoor tennis
AREA: Adyar, Chennai (searched within 8 km). TIME ASKED: today after work.
NOW: Tue 14 Oct, 15:10 local time.
QUESTION TYPE: where
DIRECT ANSWER: GO TO Riverside Courts on Tue 14 Oct, 18:00–20:00 (verdict good, score 74/100).
OPTIONS (best first):
1. Riverside Courts (tennis court, 1.4 km away (~5 min driving without traffic)) on Tue 14 Oct, 18:00–20:00: verdict good (score 74/100).
   Conditions at start: feels like 30°C, rain chance 10%, wind from SE 14 km/h, US AQI 61 = moderate (PM2.5 18 µg/m³), partly cloudy.
   Good: after dark (venue is tagged as lit).
   Watch: warm: feels like 30°C.
2. Lakeview Tennis Club (tennis club, 3.2 km away (~11 min driving without traffic)) on Tue 14 Oct, 17:00–19:00: verdict okay (score 61/100).
   Conditions at start: feels like 33°C, rain chance 10%, wind from SE 18 km/h, US AQI 64 = moderate (PM2.5 19 µg/m³), partly cloudy.
   Watch: warm: feels like 33°C; gusty at times (29 km/h).
AVOID: Tue 14 Oct, 17:00–18:00: warm: feels like 33°C
CARRY: water; spare grips; cap and sunscreen
WATCH FOR: gusty wind affects ball flight; wet courts are dangerous; sunset is at 17:58"""

EXAMPLE_ANSWER = """**Play at Riverside Courts on Tue 14 Oct, 18:00–20:00.** It's warm (feels like 30°C) but rain is unlikely (10%) and air quality is moderate (US AQI 61); the courts are tagged as lit for after sunset.

1. **Riverside Courts**: 1.4 km (~5 min), 18:00–20:00, score 74. Lit courts, so the 17:58 sunset is no problem.
2. **Lakeview Tennis Club**: 3.2 km (~11 min), 17:00–19:00, score 61. Hotter at the start (feels like 33°C) and gusts up to 29 km/h.

**Carry:** water, spare grips, cap and sunscreen.
**Watch for:** gusty wind affecting ball flight.
**Avoid:** 17:00–18:00, when it still feels like 33°C."""


def aqi_label(aqi: float | None) -> str:
    if aqi is None:
        return "not available"
    return ("good" if aqi <= 50 else "moderate" if aqi <= 100 else "unhealthy for sensitive groups" if aqi <= 150
            else "unhealthy" if aqi <= 200 else "very unhealthy" if aqi <= 300 else "hazardous")


def direct_answer(res: dict) -> str:
    """The headline decision, computed deterministically so the SLM only has to phrase it."""
    recs, u = res["recommendations"], res["understood"]
    top = recs[0]
    e = res.get("exact_hour_check")
    if u.get("intent") == "go_no_go" and e:
        if e["verdict"] in ("great", "good", "okay"):
            return f"YES, {e['time']} is {e['verdict']} for this at {e['place']} (score {e['score']:.0f}/100)."
        why = e.get("veto") or (e["issues"][0] if e.get("issues") else "conditions are poor")
        alt = f" Better: {top['place']} on {top['when']} (score {top['score']:.0f}/100)." if top["verdict"] != "avoid" else ""
        return f"NO, {e['time']} is {e['verdict']}: {why}.{alt}"
    if top["verdict"] == "poor":
        issue = top["cautions"][0] if top["cautions"] else "several conditions are unfavourable"
        return (f"NOT RECOMMENDED: conditions are poor everywhere in this window. Least-bad option: {top['place']} on "
                f"{top['when']} (score {top['score']:.0f}/100), main issue: {issue}.")
    if top["verdict"] == "avoid":
        reasons = list(dict.fromkeys(a["reason"] for a in res.get("avoid", [])))[:3] or [top["veto"]]
        return (f"NOT RECOMMENDED in this time window: every option has a problem ({'; '.join(reasons)}). "
                f"Least-bad option: {top['place']} on {top['when']}, where the problem is: {top['veto']}.")
    if u.get("intent") == "compare" and len(res.get("by_day", [])) > 1:
        d = sorted(res["by_day"], key=lambda x: -x["day_score"])
        return (f"{d[0]['day'].split()[0].upper()} is better (day score {d[0]['day_score']:.0f} vs {d[1]['day_score']:.0f}). "
                f"Best: {d[0]['best_place']} at {d[0]['best_time']}.")
    return f"GO TO {top['place']} on {top['when']} (verdict {top['verdict']}, score {top['score']:.0f}/100)."


def _cond(m: dict) -> str:
    parts = [f"feels like {m['feels_like_c']:.0f}°C" if m.get("feels_like_c") is not None else None,
             f"rain chance {m['rain_chance_pct']}%",
             f"wind from {m['wind_from']} {m['wind_kmh']:.0f} km/h",
             f"UV {m['uv_index']:.0f}" if m.get("uv_index") else None,
             (f"US AQI {m['us_aqi']} = {aqi_label(m['us_aqi'])}" + (f" (PM2.5 {m['pm2_5']:.0f} µg/m³)" if m.get("pm2_5") is not None else ""))
             if m.get("us_aqi") is not None else "AQI not available",
             m.get("sky")]
    return ", ".join(p for p in parts if p)


def fact_sheet(res: dict) -> str:
    u = res["understood"]
    L = [f"QUESTION: {res['query']}",
         f"ACTIVITY: {u['activity_name']}" + (f". PERSON: {', '.join(u['profile_labels'])}" if u.get("profile_labels") else ""),
         f"AREA: {u['resolved_location']} (searched within {u['radius_km']} km). TIME ASKED: {u['window']}.",
         f"NOW: {datetime.fromisoformat(u['now_local']):%a %d %b, %H:%M} local time.",
         f"QUESTION TYPE: {u['intent']}",
         f"DIRECT ANSWER: {direct_answer(res)}"]
    if res.get("exact_hour_check"):
        e = res["exact_hour_check"]
        L.append(f"AT THE REQUESTED TIME ({e['time']}) at {e['place']}: verdict {e['verdict']} (score {e['score']:.0f}/100); "
                 f"{_cond(e['conditions'])}" + (f"; problem: {e['veto']}" if e.get("veto") else "")
                 + (f"; issues: {'; '.join(e['issues'])}" if e.get("issues") else ""))
    L.append("OPTIONS (best first):")
    for r in res["recommendations"]:
        trav = ""
        if r.get("travel"):
            trav = f", {r['distance_km']} km away (~{r['travel']['minutes_free_flow']:.0f} min {r['travel']['mode']} without traffic)"
        L.append(f"{r['rank']}. {r['place']} ({r['kind']}{trav}) on {r['when']}: verdict {r['verdict']} (score {r['score']:.0f}/100).")
        L.append(f"   Conditions at start: {_cond(r['conditions_at_start'])}.")
        if r["why"]:
            L.append(f"   Good: {'; '.join(r['why'][:3])}.")
        if r["cautions"]:
            L.append(f"   Watch: {'; '.join(r['cautions'][:3])}.")
        if r.get("veto"):
            L.append(f"   Problem: {r['veto']}.")
    if res.get("by_day") and len(res["by_day"]) > 1:
        L.append("BY DAY: " + " | ".join(f"{d['day']}: day score {d['day_score']:.0f}, {d['good_slots']} good start times, best {d['best_time']} at {d['best_place']} (score {d['best_score']:.0f}, {d['verdict']})"
                                         + (f", main issue: {d['main_issue']}" if d.get("main_issue") else "") for d in res["by_day"]))
    if res.get("downwind_insight"):
        L.append(f"WIND AND POLLUTION: {res['downwind_insight']}")
    if res.get("avoid"):
        L.append("AVOID: " + " | ".join(f"{a['from']}–{a['to']}: {a['detail']}" for a in res["avoid"][:4]))
    L.append("CARRY: " + "; ".join(res["advice"]["carry"]))
    L.append("WATCH FOR: " + "; ".join(res["advice"]["watch_for"]))
    for n in res.get("data_notes", [])[:2]:
        L.append(f"NOTE: {n}")
    return "\n".join(L)


# ---------------------------------------------------------------- grounding check

_COMMON = {"carry", "watch", "avoid", "watch for", "best", "option", "options", "note", "us aqi", "uv", "pm2.5",
           "great", "good", "okay", "poor", "yes", "no", "rain", "wind", "monday", "tuesday", "wednesday", "thursday",
           "friday", "saturday", "sunday", "today", "tomorrow", "aqi", "the", "if", "at", "for", "it", "a"}


def _numbers(text: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", text)}


def _allowed_numbers(facts: str) -> set[str]:
    nums = _numbers(facts)
    for h, m in re.findall(r"\b(\d{1,2}):(\d{2})\b", facts):  # 17:00 may be written as 5 PM / 5:00 pm
        h12 = int(h) % 12 or 12
        nums |= {str(h12), str(int(h))}
    for n in list(nums):
        try:
            f = float(n)
            nums |= {str(round(f)), f"{f:.1f}".rstrip("0").rstrip(".")}
        except ValueError:
            pass
    return nums | {"1", "2", "3", "0"}


DAY_WORDS = r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
ABBR = {"monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu", "friday": "fri", "saturday": "sat",
        "sunday": "sun"}
LABEL_RANK = {"good": 0, "clean": 0, "fresh": 0, "low": 0, "moderate": 1, "fair": 1, "acceptable": 1, "decent": 1,
              "okay": 1, "ok": 1}


MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
WEEKDAY = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _allowed_day_words(facts: str) -> set[str]:
    """Day words that correctly name a date appearing in the facts ('today' only if that date is NOW's date)."""
    m = re.search(r"NOW: \w{3} (\d{1,2}) (\w{3})", facts)
    if not m:
        return set()
    year = datetime.now().year
    now = datetime(year, MONTHS[m.group(2).lower()], int(m.group(1))).date()
    words = set()
    body = "\n".join(line for line in facts.splitlines() if not line.startswith("NOW:"))
    for d, mon in re.findall(r"\b(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", body):
        day = datetime(year, MONTHS[mon.lower()], int(d)).date()
        words.add(WEEKDAY[day.weekday()])
        if day == now:
            words |= {"today", "tonight"}
        elif day == now + timedelta(days=1):
            words.add("tomorrow")
    return words


def check_grounding(answer: str, facts: str) -> list[str]:
    """Return the list of unsupported tokens (empty = grounded)."""
    bad = sorted(_numbers(answer) - _allowed_numbers(facts), key=len)
    fl = facts.lower()
    ok_days = _allowed_day_words(facts)
    for w in set(re.findall(DAY_WORDS, answer.lower())):
        if w not in ok_days:
            bad.append(f"day '{w}'")
    # Air-quality adjectives must not be milder than the label computed from the number.
    for adj, num in re.findall(r"\b(\w+)\s+air(?: quality)?\b[^.\n]{0,25}?US AQI (\d+)", answer, re.I) + \
            re.findall(r"air quality (?:is |was )?(\w+)[^.\n]{0,25}?US AQI (\d+)", answer, re.I):
        a = int(num)
        if adj.lower() in LABEL_RANK and LABEL_RANK[adj.lower()] < (0 if a <= 50 else 1 if a <= 100 else 2):
            bad.append(f"'{adj} air quality' for US AQI {a}")
    for phrase in re.findall(r"\b(?:[A-Z][a-z'.-]+)(?:\s+(?:[A-Z][a-z'.-]+|of|de|la))+\b", answer):
        p = re.sub(r"^(The|A|An|At|In|On|Near|Try|Visit|Head|Go|Choose|Pick|Play|Run|Hike|Walk|Cycle|Plan|Book|Yes|No|Best|Option)\s+", "", phrase).strip()
        if len(p.split()) >= 2 and p.lower() not in fl and p.lower() not in _COMMON:
            if not all(w.lower() in fl or w.lower() in _COMMON for w in p.split()):
                bad.append(p)
    return bad


# ---------------------------------------------------------------- template (fallback and reference)

def template_answer(res: dict) -> str:
    recs = res["recommendations"]
    u = res["understood"]
    top = recs[0]
    intent = u.get("intent")
    act = u["activity_name"].split(" (")[0].split(" /")[0].lower()
    lines = []
    e = res.get("exact_hour_check")
    if intent == "go_no_go" and e:
        if e["verdict"] in ("great", "good", "okay"):
            lines.append(f"**Yes, {e['time']} works for {act}** ({e['verdict']}, score {e['score']:.0f}/100): {_cond(e['conditions'])}.")
        else:
            why = e.get("veto") or (e["issues"][0] if e.get("issues") else "conditions are poor")
            lines.append(f"**Not ideal at {e['time']}**: {why}. Better: **{top['when']}** at {top['place']} (score {top['score']:.0f}/100).")
    elif intent == "compare" and len(res.get("by_day", [])) > 1:
        d = sorted(res["by_day"], key=lambda x: -x["day_score"])
        lines.append(f"**{d[0]['day'].split()[0]} is the better choice**: best {d[0]['best_time']} at {d[0]['best_place']} "
                     f"(day score {d[0]['day_score']:.0f}, {d[0]['good_slots']} good start times) vs {d[1]['day'].split()[0]} "
                     f"(day score {d[1]['day_score']:.0f}, {d[1]['good_slots']} good start times"
                     + (f", {d[1]['main_issue']}" if d[1].get("main_issue") else "") + ").")
    elif top["verdict"] == "avoid":
        lines.append(f"**I wouldn't plan {act} in this window**: {top['veto']}. Least-bad option below; consider another day.")
    else:
        lines.append(f"**Best bet: {top['place']} on {top['when']}** ({top['verdict']}, score {top['score']:.0f}/100).")
    lines.append("")
    for r in recs:
        trav = f", {r['distance_km']} km (~{r['travel']['minutes_free_flow']:.0f} min)" if r.get("travel") else ""
        reason = r["why"][0] if r["why"] else (r["cautions"][0] if r["cautions"] else "")
        lines.append(f"{r['rank']}. **{r['place']}** ({r['kind']}{trav}): {r['when']}, score {r['score']:.0f}. "
                     f"{_cond(r['conditions_at_start'])}." + (f" {reason[0].upper() + reason[1:]}." if reason else ""))
    if res.get("downwind_insight"):
        lines += ["", f"🌬️ {res['downwind_insight']}"]
    lines += ["", "**Carry:** " + "; ".join(res["advice"]["carry"][:6]) + ".",
              "**Watch for:** " + "; ".join(res["advice"]["watch_for"][:5]) + "."]
    if res.get("avoid"):
        lines.append("**Avoid:** " + "; ".join(f"{a['from']}–{a['to']} ({a['detail']})" for a in res["avoid"][:3]) + ".")
    return "\n".join(lines)


def check_decision(answer: str, res: dict) -> str | None:
    """The opening of the answer must carry the same decision as DIRECT ANSWER."""
    head = answer.strip().split("\n")[0].lower()
    top = res["recommendations"][0]
    da = direct_answer(res)
    if da.startswith("NOT RECOMMENDED"):  # also covers 'poor everywhere'
        return None if re.search(r"not recommended|wouldn.t|avoid|not (a )?good|no\b|unsafe|isn.t", head) else "lead must say not recommended"
    if da.startswith("NO,"):
        return None if re.search(r"\bno\b|not ideal|not recommended|avoid|isn.t|wouldn.t", head) else "lead must say no"
    if da.startswith("YES,"):
        return None if re.search(r"\byes\b|good|fine|okay|go ahead|works", head) else "lead must say yes"
    if res["understood"].get("intent") == "compare" and len(res.get("by_day", [])) > 1:
        day = da.split()[0].lower()
        return None if day in head else f"lead must name {day.title()}"
    return None if top["place"].lower() in head else f"lead must name {top['place']}"


async def narrate(res: dict, llm) -> None:
    facts = fact_sheet(res)
    res["fact_sheet"] = facts
    res["answer"] = template_answer(res)
    res["answer_source"] = "template"
    res["grounding"] = {"checked": False}
    if llm is None or not llm.ready:
        return
    msgs = [{"role": "system", "content": NARRATE_SYSTEM},
            {"role": "user", "content": f"FACTS:\n{EXAMPLE_FACTS}\n\nWrite the reply."},
            {"role": "assistant", "content": EXAMPLE_ANSWER},
            {"role": "user", "content": f"FACTS:\n{facts}\n\nWrite the reply."}]
    for attempt in range(2):
        out = await llm.chat(msgs, max_new_tokens=360, temperature=0.0 if attempt == 0 else 0.2)
        if not out:
            return
        bad = check_grounding(out, facts)
        wrong = check_decision(out, res)
        if wrong:
            bad.append(wrong)
        if not bad:
            res.update(answer=out.strip(), answer_source="slm", grounding={"checked": True, "passed": True, "attempts": attempt + 1})
            return
        res["grounding"] = {"checked": True, "passed": False, "unsupported": bad[:10], "attempts": attempt + 1}
        msgs = msgs[:4] + [{"role": "assistant", "content": out},
                           {"role": "user", "content": f"These are not in the FACTS: {', '.join(bad[:8])}. Rewrite using only the FACTS."}]
