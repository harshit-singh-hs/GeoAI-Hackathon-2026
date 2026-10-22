"""Turn vague time phrases ("this weekend", "tomorrow evening", "at 8 AM", "next two days") into concrete hours.

Deterministic on purpose: the SLM extracts the phrase, this code decides what it means, so the window is always
explainable. All times are local to the queried place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3,
            "thu": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "a couple of": 2,
            "couple of": 2, "few": 3, "a few": 3}
# Part of day -> (start hour, end hour exclusive). "sunrise"/"sunset" are refined with real sun times later.
PARTS = [
    (r"early morning|dawn|daybreak", (5, 8)), (r"late morning", (9, 12)), (r"\bmorning\b", (5, 11)),
    (r"\bnoon\b|midday|lunch ?time|\blunch\b", (11, 14)), (r"late afternoon", (15, 18)), (r"afternoon", (12, 17)),
    (r"after work|after office|post work", (17, 21)), (r"sunset|golden hour|dusk", (16, 20)),
    (r"sunrise", (5, 9)), (r"\bevening\b", (16, 21)), (r"tonight|\bnight\b", (19, 24)),
]
DEFAULT_HOURS = (5, 21)


@dataclass
class TimeWindow:
    days: list[date]
    hours: tuple[int, int]
    label: str
    exact_hour: int | None = None
    sun_event: str | None = None  # "sunrise" | "sunset" when the user asked for it
    notes: list[str] = field(default_factory=list)

    def slots(self, now: datetime) -> list[datetime]:
        """All candidate local hours in the window that are not in the past."""
        out = []
        cur_hour = now.replace(minute=0, second=0, microsecond=0)
        for d in self.days:
            for h in range(self.hours[0], self.hours[1]):
                t = datetime(d.year, d.month, d.day, h)
                if t >= cur_hour:
                    out.append(t)
        return out


def _parse_clock(text: str) -> int | None:
    m = re.search(r"\b(?:at|by|around|@)?\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", text)
    if m:
        h = int(m.group(1)) % 12
        return h + 12 if m.group(3).startswith("p") else h
    m = re.search(r"\b(?:at|by|around|@)\s*(\d{1,2})(?::(\d{2}))?\b(?!\s*(?:km|kms|kilometers?|min|mins|minutes|days?|hours?|hrs?))", text)
    if m and 0 <= int(m.group(1)) <= 23:
        h = int(m.group(1))
        return h if h >= 5 else h + 12  # bare "at 3" is far more likely 15:00 than 03:00
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    return int(m.group(1)) if m else None


SLANG = [(r"\b(evng|evn|eve|evening's)\b", "evening"), (r"\b(mrng|mrg|morn|mornin)\b", "morning"),
         (r"\b(tmrw|tmr|tmrow|tomoro|tomorow|2mrw|2moro)\b", "tomorrow"), (r"\b(tdy|tday|2day)\b", "today"),
         (r"\b(aftn|arvo|aft)\b", "afternoon"), (r"\b(wknd|wkend|w/e)\b", "weekend"), (r"\b(nite|nyt)\b", "night"),
         (r"\b(tonite|2nite)\b", "tonight")]


def normalise(text: str) -> str:
    t = text.lower()
    for rx, rep in SLANG:
        t = re.sub(rx, rep, t)
    return t


def resolve(text: str, now: datetime, max_days: int = 16) -> TimeWindow:
    t = " " + normalise(text) + " "
    today = now.date()
    notes: list[str] = []
    days: list[date] | None = None
    label = ""

    if re.search(r"\byesterday\b|\blast (week|night|weekend)\b", t):
        notes.append("past_time")

    if re.search(r"day after tomorrow", t):
        days, label = [today + timedelta(days=2)], "day after tomorrow"
    elif re.search(r"\b(tomorrow|tmrw|tmr|tomorow)\b", t):
        days, label = [today + timedelta(days=1)], "tomorrow"
    elif m := re.search(r"\b(next|coming|following)\s+(\d+|one|two|three|four|five|six|seven|couple of|few)\s+days?\b", t):
        n = int(m.group(2)) if m.group(2).isdigit() else NUMWORDS[m.group(2)]
        days, label = [today + timedelta(days=i) for i in range(n + 1 if n <= 3 else n)], f"next {n} days"
    elif re.search(r"\bnext weekend\b", t):
        sat = today + timedelta(days=(5 - today.weekday()) % 7 + 7 if today.weekday() < 5 else (12 - today.weekday()))
        days, label = [sat, sat + timedelta(days=1)], "next weekend"
    elif re.search(r"\b(this |the |coming )?weekend\b", t):
        wd = today.weekday()
        if wd == 5:
            days = [today, today + timedelta(days=1)]
        elif wd == 6:
            days = [today]
        else:
            sat = today + timedelta(days=5 - wd)
            days = [sat, sat + timedelta(days=1)]
        label = "this weekend"
    elif re.search(r"\b(this|next|coming) week\b", t):
        start = today if "this" in t else today + timedelta(days=7 - today.weekday())
        days, label = [start + timedelta(days=i) for i in range(7 if "this" not in t else 7 - today.weekday())], "week"
    else:
        found = []
        for name, wd in WEEKDAYS.items():
            if re.search(rf"\b{name}\b", t):
                delta = (wd - today.weekday()) % 7
                if re.search(rf"\bnext\s+{name}\b", t) and delta == 0:
                    delta = 7
                d = today + timedelta(days=delta)
                if d not in found:
                    found.append(d)
        if found:
            days, label = sorted(found), " / ".join(d.strftime("%A") for d in sorted(found))
        elif re.search(r"\b(today|tonight|this (morning|afternoon|evening)|now|right now|currently|after work)\b", t):
            days, label = [today], "today"
        else:
            m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
                          r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
            if m:
                day = int(m.group(1) or m.group(4))
                mon = (m.group(2) or m.group(3))[:3]
                month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(mon) + 1
                try:
                    d = date(today.year, month, day)
                    if d < today:
                        d = date(today.year + 1, month, day)
                    days, label = [d], d.strftime("%d %b")
                except ValueError:
                    pass

    hours = DEFAULT_HOURS
    sun_event = "sunset" if re.search(r"sunset|golden hour|dusk", t) else "sunrise" if "sunrise" in t else None
    part_label = ""
    for rx, rng in PARTS:
        if re.search(rx, t):
            hours, part_label = rng, re.search(rx, t).group(0).strip()
            break
    exact = _parse_clock(t)
    if exact is not None and exact < 12 and hours[0] >= 12 and not re.search(r"\d\s*a\.?m", t):
        exact += 12  # "at 6 after work" -> 18:00
    if exact is not None:
        hours = (max(0, exact - 2), min(24, exact + 3))
        part_label = f"around {exact:02d}:00"
    if re.search(r"\b(right now|now|currently|at the moment)\b", t) and exact is None and not part_label:
        hours = (now.hour, min(24, now.hour + 4))
        part_label = "now"

    if days is None:
        if exact is not None or part_label:
            # "at 8 AM" / "evening" with no day: the next occurrence of that time.
            first = today if now.hour < hours[1] - 1 else today + timedelta(days=1)
            days, label = [first], ("today" if first == today else "tomorrow")
        else:
            days, label = [today, today + timedelta(days=1)], "next 48 hours"
            notes.append("no_time_given")

    horizon = today + timedelta(days=max_days - 1)
    if any(d > horizon for d in days):
        notes.append("beyond_forecast")
        days = [d for d in days if d <= horizon]
    return TimeWindow(days=days, hours=hours, label=f"{label} {part_label}".strip(), exact_hour=exact,
                      sun_event=sun_event, notes=notes)
