"""Run a set of varied questions against a running endpoint and summarise status, grounding and latency.

Usage: python scripts/eval_queries.py [base_url] [out.jsonl]
"""
import json
import sys
import time

import httpx

QUERIES = [
    # the rulebook's examples
    "Where can I go running tomorrow evening near Koregaon Park?",
    "Best time this weekend for a kids' cricket match within 15 km of Whitefield?",
    "I have mild asthma, where's good for a morning walk near Powai in the next two days?",
    "Is it okay to cycle to work at 8 AM?",
    "Planning an outdoor birthday party on Sunday afternoon in Pune, what should I keep in mind?",
    "Good place and time for golf on Saturday within 40 km of Andheri?",
    "Where can I take sunset photos this evening around Bandra?",
    "Is Saturday or Sunday better for a hike near Lonavala?",
    "When is it safe for my parents to walk outdoors tomorrow in Bengaluru?",
    "Outdoor tennis after work today near Hinjewadi, any good spots?",
    # other cities / phrasings
    "five-a-side football tonight near Salt Lake, Kolkata?",
    "picnic spots near Hyderabad this sunday for a family with a toddler",
    "I'm 7 months pregnant, can I go for a walk in Lodhi Garden tomorrow morning?",
    "golf near gurgaon on saturday morning",
    "best place to jog near Anna Nagar Chennai early morning tomorrow",
    "cycling ride around Chandigarh this weekend",
    "wedding reception outdoors in Jaipur next Saturday evening, any tips?",
    "sunrise photography near Marine Drive tomorrow",
    "nature walk near Sanjay Gandhi National Park tomorrow morning",
    "cricket nets practice near Shivaji Park after 4 pm today",
    "my dad has a heart condition, is a walk at Cubbon Park ok at noon tomorrow?",
    "tennis in Ahmedabad next 3 days",
    # messy / out of scope
    "run tmrw evng near kp pune",
    "where should I go this weekend",
    "football",
    "can i go swimming in thane tomorrow",
    "whats the capital of france",
    "hike near Manali on 25 December",
    "is it a good day to be outside in Indore?",
    "Is it okay to cycle to work at 8 AM near Baner, Pune?",
]


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    out = open(sys.argv[2] if len(sys.argv) > 2 else "outputs/eval.jsonl", "w")
    n = ok = slm = fail = 0
    times = []
    with httpx.Client(timeout=180) as c:
        for q in QUERIES:
            t = time.time()
            try:
                r = c.post(f"{base}/ask", json={"query": q, "verbose": True}).json()
            except Exception as e:
                print(f"ERROR {q}: {e}")
                fail += 1
                continue
            dt = time.time() - t
            n += 1
            times.append(dt)
            ok += r.get("status") == "ok"
            slm += r.get("answer_source") == "slm"
            out.write(json.dumps({"query": q, "seconds": round(dt, 1), "response": r}) + "\n")
            out.flush()
            top = (r.get("recommendations") or [{}])[0]
            print(f"{dt:5.1f}s {r.get('status', '?'):18s} {r.get('answer_source', '?'):8s} "
                  f"{str((r.get('grounding') or {}).get('passed')):5s} | {q[:58]:58s} | "
                  f"{top.get('place', '-')[:28]} {top.get('when', '')}")
    times.sort()
    print(f"\n{n} answered, {ok} ok, {slm} SLM-written, {fail} errors; "
          f"median {times[len(times) // 2]:.1f}s, max {times[-1]:.1f}s")


if __name__ == "__main__":
    main()
