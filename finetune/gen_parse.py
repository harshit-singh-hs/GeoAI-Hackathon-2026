"""Synthetic training data for the query-parsing task.

Questions are assembled from templates (activity phrasings x places x time phrases x distances x health
profiles x question types) with realistic noise (lowercase, slang, missing fields, out-of-scope and unsupported
requests). Labels are known by construction, so no external model is involved. The rulebook's example
questions are excluded from the training set.

Usage: python finetune/gen_parse.py finetune/data/parse.jsonl [n]
"""
from __future__ import annotations

import json
import random
import sys

ACT = {  # id -> list of (verb phrase, noun phrase)
    "running": [("go running", "a run"), ("jog", "jogging"), ("go for a run", "a morning run"), ("run a 5k", "a 5k run"),
                ("do my long run", "a long run"), ("go jogging", "a jog"), ("do sprint training", "sprint training")],
    "cycling": [("go cycling", "a bike ride"), ("ride my bike", "cycling"), ("cycle", "a cycling ride"),
                ("go for a ride", "a long ride"), ("cycle to work", "my cycle commute"), ("bike to office", "a bike commute")],
    "golf": [("play golf", "golf"), ("play 18 holes", "a round of golf"), ("hit the driving range", "golf practice"),
             ("tee off", "a golf round")],
    "cricket": [("play cricket", "a cricket match"), ("do net practice", "cricket nets"), ("play gully cricket", "a cricket game"),
                ("practice batting", "cricket practice")],
    "football": [("play football", "a football match"), ("play five-a-side", "five-a-side"), ("kick a ball around", "a kickabout"),
                 ("play soccer", "a 5-a-side game")],
    "tennis": [("play tennis", "tennis"), ("hit some tennis balls", "a tennis match"), ("play a tennis match", "tennis practice")],
    "hiking": [("go hiking", "a hike"), ("go trekking", "a trek"), ("do a fort trek", "a nature walk"), ("hike", "a trail walk")],
    "walking": [("go for a walk", "a walk"), ("take a stroll", "a stroll"), ("go for a morning walk", "a morning walk"),
                ("walk", "an evening walk"), ("do a brisk walk", "a brisk walk")],
    "picnic": [("have a picnic", "a picnic"), ("go for a family picnic", "a park outing"), ("plan a picnic", "a family day out")],
    "event": [("host an outdoor birthday party", "an outdoor birthday party"), ("hold a wedding reception outdoors", "a wedding reception"),
              ("organise a get-together", "an outdoor get-together"), ("set up a small fair", "an outdoor event")],
    "photography": [("take photos", "photography"), ("do a photoshoot", "a photoshoot"), ("shoot the sunset", "sunset photos"),
                    ("take sunrise pictures", "sunrise photography"), ("do a pre-wedding shoot", "a pre-wedding shoot")],
}
UNSUPPORTED = ["go swimming", "fly kites", "go skating", "go kayaking", "go fishing", "go camping", "play badminton outside",
               "go paragliding", "do outdoor yoga", "go birdwatching", "go stargazing", "play basketball"]
OUT_OF_SCOPE = ["what's the capital of france", "write a python function to sort a list", "best biryani recipe",
                "is the stock market open tomorrow", "recommend a good movie tonight", "book me a cab to the airport",
                "translate 'good morning' to hindi", "who won the cricket world cup in 2011", "how do I reset my wifi router",
                "suggest a good laptop under 60000", "explain quantum computing simply", "what time does the mall close",
                "any good indoor gyms near me", "how to make cold coffee"]
PLACES = [
    "Koramangala", "Indiranagar", "HSR Layout", "Jayanagar", "Malleshwaram", "Hebbal", "Electronic City", "Yelahanka",
    "Bandra West", "Andheri East", "Juhu", "Colaba", "Worli", "Dadar", "Thane", "Navi Mumbai", "Borivali", "Chembur",
    "Kothrud", "Baner", "Viman Nagar", "Aundh", "Wakad", "Kharadi", "Hadapsar", "Shivajinagar",
    "Salt Lake", "Park Street", "New Town", "Ballygunge", "Howrah", "Dum Dum",
    "Connaught Place", "Hauz Khas", "Dwarka", "Saket", "Rohini", "Vasant Kunj", "Noida Sector 62", "Gurugram", "Faridabad",
    "Banjara Hills", "Gachibowli", "Jubilee Hills", "Kukatpally", "Secunderabad", "Madhapur",
    "Adyar", "T Nagar", "Velachery", "Besant Nagar", "Mylapore", "Tambaram",
    "Satellite, Ahmedabad", "Navrangpura", "Bopal", "Malviya Nagar, Jaipur", "Vaishali Nagar", "Gomti Nagar", "Hazratganj",
    "Sector 17, Chandigarh", "Panchkula", "Mohali", "Bhopal", "Indore", "Nagpur", "Nashik", "Aurangabad", "Mysuru",
    "Coimbatore", "Kochi", "Thiruvananthapuram", "Kozhikode", "Panaji", "Margao", "Shimla", "Dehradun", "Rishikesh",
    "Mussoorie", "Ooty", "Munnar", "Mahabaleshwar", "Alibaug", "Puducherry", "Visakhapatnam", "Vijayawada",
    "Bhubaneswar", "Guwahati", "Shillong", "Patna", "Ranchi", "Surat", "Vadodara", "Rajkot", "Udaipur", "Jodhpur",
    "Amritsar", "Ludhiana", "Kanpur", "Varanasi", "Agra", "Mangaluru", "Hubballi", "Belagavi", "Raipur", "Jabalpur",
    "Ulsoor Lake", "Sankey Tank", "Necklace Road", "Sukhna Lake", "Kankaria Lake", "Elliot's Beach", "Carter Road",
    "Aarey Colony", "Nehru Park", "Sunder Nursery", "Rabindra Sarobar", "Victoria Memorial", "Sinhagad", "Pashan Lake",
    "KBR Park", "Guindy National Park", "Rock Garden", "Lake Pichola", "Nandi Hills", "Matheran", "Kasauli", "Lansdowne",
]
TIMES = ["tomorrow evening", "tomorrow morning", "this evening", "tonight", "today after work", "this weekend",
         "on Saturday", "on Sunday morning", "next Saturday", "in the next two days", "over the next 3 days",
         "at 6 AM tomorrow", "at 7:30 am", "around 5 pm today", "this Sunday afternoon", "tmrw morning", "day after tomorrow",
         "on 28 Oct", "early tomorrow morning", "at sunrise tomorrow", "at sunset today", "right now", "after office today",
         "this week", "next weekend", "on Friday evening", "by noon tomorrow", "tomorrow afternoon", "tmrw evng",
         "later today", "this Saturday at 4 pm", "on Monday morning"]
PROFILES = {
    "asthma": ["I have asthma, ", "I'm asthmatic, so ", "with my mild asthma, ", "my son has asthma, "],
    "senior": ["my parents want to know: ", "for my 72-year-old dad, ", "with my grandma, ", "I'm 68, ", "for elderly parents, "],
    "child": ["with my 6 year old, ", "with two kids, ", "for my toddler and me, ", "with a group of school kids, "],
    "pregnancy": ["I'm pregnant, ", "my wife is 6 months pregnant, ", "I'm in my third trimester, "],
    "heart": ["my father has a heart condition, ", "I have high blood pressure, ", "as a cardiac patient, "],
}
EXCLUDE = {  # the rulebook's example questions are never used for training
    "where can i go running tomorrow evening near koregaon park?",
    "best time this weekend for a kids' cricket match within 15 km of whitefield?",
    "i have mild asthma, where's good for a morning walk near powai in the next two days?",
    "is it okay to cycle to work at 8 am?",
    "planning an outdoor birthday party on sunday afternoon in pune, what should i keep in mind?",
    "good place and time for golf on saturday within 40 km of andheri?",
    "where can i take sunset photos this evening around bandra?",
    "is saturday or sunday better for a hike near lonavala?",
    "when is it safe for my parents to walk outdoors tomorrow in bengaluru?",
    "outdoor tennis after work today near hinjewadi, any good spots?",
}


def label(activity=None, activity_text=None, location=None, radius_km=None, time_text=None, profiles=(), intent="where",
          commute=False, out_of_scope=False) -> dict:
    return {"activity": activity, "activity_text": activity_text, "location": location, "radius_km": radius_km,
            "time_text": time_text, "profiles": sorted(profiles), "intent": intent, "commute": commute,
            "out_of_scope": out_of_scope}


def _strip_on(t: str | None) -> str | None:
    return t[3:] if t and t.startswith("on ") else t


def one(rng: random.Random) -> tuple[str, dict]:
    r = rng.random()
    if r < 0.06:
        q = rng.choice(OUT_OF_SCOPE)
        return q, label(intent="advice", out_of_scope=True)
    if r < 0.12:
        u, loc, t = rng.choice(UNSUPPORTED), rng.choice(PLACES), rng.choice(TIMES)
        q = rng.choice([f"can I {u} near {loc} {t}?", f"is it a good day to {u} in {loc} {t}?", f"where can I {u} {t} around {loc}"])
        return q, label(activity_text=u.replace("go ", "").replace("do ", "").replace("play ", ""), location=loc,
                        time_text=_strip_on(t), intent="go_no_go" if q.startswith(("can", "is")) else "where")
    if r < 0.16:
        loc, t = rng.choice(PLACES), rng.choice(TIMES)
        q = rng.choice([f"where should I go {t} near {loc}?", f"anything fun to do outdoors {t} in {loc}?", f"plans {t} around {loc}?"])
        return q, label(location=loc, time_text=_strip_on(t))

    act = rng.choice(list(ACT))
    verb, noun = rng.choice(ACT[act])
    loc = rng.choice(PLACES) if rng.random() > 0.08 else None
    t = rng.choice(TIMES) if rng.random() > 0.12 else None
    radius = rng.choice([3, 5, 8, 10, 15, 20, 25, 30, 40, 50]) if rng.random() < 0.25 and loc else None
    prof = rng.choice(list(PROFILES)) if rng.random() < 0.3 else None
    pre = rng.choice(PROFILES[prof]) if prof else ""
    commute = "work" in verb or "office" in verb
    where = (f"within {radius} km of {loc}" if radius else f"near {loc}" if rng.random() < 0.6 else f"around {loc}"
             if rng.random() < 0.5 else f"in {loc}") if loc else ""
    tt = t or ""
    intent = rng.choice(["where", "where", "when", "go_no_go", "compare", "advice"])
    if commute and intent in ("where", "compare"):
        intent = "go_no_go"
    if intent == "where":
        q = rng.choice([f"{pre}where can I {verb} {tt} {where}?", f"{pre}best place to {verb} {where} {tt}?",
                        f"{pre}suggest a good spot for {noun} {where} {tt}", f"{pre}any good places for {noun} {where} {tt}?"])
    elif intent == "when":
        q = rng.choice([f"{pre}what's the best time {tt} to {verb} {where}?", f"{pre}when should I {verb} {tt} {where}?",
                        f"{pre}best time for {noun} {tt} {where}?"])
    elif intent == "go_no_go":
        q = rng.choice([f"{pre}is it okay to {verb} {tt} {where}?", f"{pre}can I {verb} {tt} {where}?",
                        f"{pre}should I {verb} {where} {tt}?", f"{pre}is it safe to {verb} {where} {tt}?"])
    elif intent == "compare":
        pair = rng.choice(["Saturday or Sunday", "today or tomorrow", "morning or evening tomorrow", "Friday or Saturday"])
        t = pair
        q = rng.choice([f"{pre}is {pair} better for {noun} {where}?", f"{pre}{noun} {where}: {pair}?",
                        f"{pre}which is better for {noun} {where}, {pair}?"])
    else:
        q = rng.choice([f"{pre}planning {noun} {tt} {where}, what should I keep in mind?", f"{pre}any tips for {noun} {where} {tt}?",
                        f"{pre}what precautions for {noun} {tt} {where}?"])
    q = " ".join(q.split()).replace(" ?", "?").replace(" ,", ",")
    q = q[0].upper() + q[1:]
    # Noise: casual typing.
    if rng.random() < 0.25:
        q = q.lower()
    if rng.random() < 0.2:
        q = q.rstrip("?")
    if rng.random() < 0.08:
        q = "hey, " + q[0].lower() + q[1:]
    loc_label = loc
    if loc and loc not in q:  # lowercased by noise
        loc_label = loc.lower() if loc.lower() in q else loc
    t_label = _strip_on(t)
    if t_label and t_label not in q:
        t_label = t_label.lower() if t_label.lower() in q else t_label
    return q, label(activity=act, activity_text=noun if noun.lower() in q.lower() else verb, location=loc_label,
                    radius_km=radius, time_text=t_label, profiles=[prof] if prof else [], intent=intent, commute=commute)


def main(path: str, n: int = 2400, seed: int = 7) -> None:
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n:
        q, y = one(rng)
        if q.lower() in seen or q.lower().rstrip("?") + "?" in EXCLUDE:
            continue
        seen.add(q.lower())
        rows.append({"task": "parse", "query": q, "label": y})
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} parse examples -> {path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2400)
