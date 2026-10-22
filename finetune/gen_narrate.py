"""Training data for the answer-writing task.

Each example is (FACT SHEET -> answer). Fact sheets come from real pipeline runs on synthetic questions (live
Open-Meteo / OSRM / geocoding, offline OSM index so we don't load Overpass). To cover conditions that late
October doesn't offer (monsoon rain, thunderstorms, winter fog, heatwaves, smog), a weather scenario is applied to
the fetched values before scoring; the fact sheet stays internally consistent because it is computed from them.
Targets come from the deterministic reference writer (app.narrate.template_answer) and must pass the same
grounding and decision checks the live system applies.

Usage: GEOAI_OSM_MODE=local python finetune/gen_narrate.py finetune/data/parse.jsonl finetune/data/narrate.jsonl [n]
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.narrate import NARRATE_SYSTEM, check_decision, check_grounding, template_answer  # noqa: E402
from app.pipeline import run  # noqa: E402

SCENARIOS = [("none", 0.28), ("pleasant", 0.12), ("monsoon", 0.12), ("thunder", 0.10), ("heatwave", 0.08),
             ("cold_fog", 0.08), ("smog", 0.08), ("clean", 0.07), ("windy", 0.07)]


def make_transform(name: str, rng: random.Random):
    def tf(wx, aq):
        wx, aq = copy.deepcopy(wx), copy.deepcopy(aq)
        for loc in wx:
            for t, h in loc["hourly"].items():
                hour = int(t[11:13])
                if name == "pleasant":
                    h["temperature_2m"] -= 6
                    h["apparent_temperature"] -= 7
                elif name == "monsoon":
                    h["precipitation_probability"] = rng.randint(45, 100)
                    h["precipitation"] = round(rng.choice([0, 0.2, 0.8, 1.5, 3.0, 6.0]), 1)
                    h["weather_code"] = rng.choice([61, 63, 80, 81, 65]) if h["precipitation"] > 0 else 3
                    h["cloud_cover"] = rng.randint(80, 100)
                    h["apparent_temperature"] -= 3
                    h["visibility"] = rng.choice([3000, 6000, 10000])
                elif name == "thunder" and 13 <= hour <= 19:
                    h["cape"] = rng.randint(1200, 3200)
                    h["precipitation_probability"] = rng.randint(50, 95)
                    if rng.random() < 0.35:
                        h["weather_code"] = 95
                        h["precipitation"] = 4.0
                elif name == "heatwave":
                    h["apparent_temperature"] += rng.uniform(5, 9)
                    h["temperature_2m"] += 5
                    h["uv_index"] = (h["uv_index"] or 0) + (2 if h["is_day"] else 0)
                elif name == "cold_fog":
                    h["apparent_temperature"] -= rng.uniform(12, 17)
                    h["temperature_2m"] -= 12
                    if hour <= 9:
                        h["weather_code"], h["visibility"] = 45, rng.choice([300, 800, 1500])
                elif name == "windy":
                    h["wind_speed_10m"] *= 2.8
                    h["wind_gusts_10m"] *= 2.6
        if aq:
            k = {"smog": 1.9, "clean": 0.35, "pleasant": 0.6}.get(name)
            if k:
                for loc in aq:
                    for v in loc.values():
                        if v.get("us_aqi") is not None:
                            v["us_aqi"] = int(v["us_aqi"] * k)
                        if v.get("pm2_5") is not None:
                            v["pm2_5"] = v["pm2_5"] * k
        return wx, aq
    return tf


async def main(parse_path: str, out_path: str, n: int, concurrency: int = 3, seed: int = 11) -> None:
    rng = random.Random(seed)
    qs = [json.loads(l) for l in open(parse_path)]
    qs = [q for q in qs if q["label"].get("activity") and q["label"].get("location")]
    rng.shuffle(qs)
    sem = asyncio.Semaphore(concurrency)
    done, kept, rejected = 0, 0, 0
    out = open(out_path, "w")

    async def one(item):
        nonlocal done, kept, rejected
        scen = rng.choices([s for s, _ in SCENARIOS], [w for _, w in SCENARIOS])[0]
        async with sem:
            try:
                res = await run(item["query"], None, transform=None if scen == "none" else make_transform(scen, random.Random(rng.random())))
            except Exception as e:  # a flaky upstream call: skip this example
                print("skip:", item["query"][:50], type(e).__name__, str(e)[:80])
                return
        done += 1
        if res.get("status") != "ok":
            return
        answer = template_answer(res, seed=rng.randint(0, 10**6))
        facts = res["fact_sheet"]
        if check_grounding(answer, facts) or check_decision(answer, facts):
            rejected += 1
            return
        kept += 1
        out.write(json.dumps({"task": "narrate", "scenario": scen, "query": item["query"], "messages": [
            {"role": "system", "content": NARRATE_SYSTEM},
            {"role": "user", "content": f"FACTS:\n{facts}\n\nWrite the reply."},
            {"role": "assistant", "content": answer}]}, ensure_ascii=False) + "\n")
        out.flush()
        if kept % 25 == 0:
            print(f"kept {kept} (ran {done}, rejected {rejected})", flush=True)

    await asyncio.gather(*(one(q) for q in qs[: int(n * 1.4)]))
    print(f"done: kept {kept}, ran {done}, rejected {rejected} -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 700,
                     seed=int(sys.argv[4]) if len(sys.argv) > 4 else 11))
