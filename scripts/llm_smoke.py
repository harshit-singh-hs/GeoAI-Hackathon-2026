"""Smoke-test an SLM on parsing + narration: speed, JSON validity, grounding pass rate."""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.llm import LLM
from app.parse import parse_messages, _extract_json, merge, parse_rules
from app.pipeline import run

QUERIES = ["Where can I go running tomorrow evening near Koregaon Park?",
           "Best time this weekend for a kids' cricket match within 15 km of Whitefield?",
           "I have mild asthma, where's good for a morning walk near Powai in the next two days?",
           "Is it okay to cycle to work at 8 AM?",
           "Planning an outdoor birthday party on Sunday afternoon in Pune, what should I keep in mind?",
           "Where can I take sunset photos this evening around Bandra?",
           "can i go swimming in thane tomorrow", "my grandpa wants a stroll near lalbagh tmrw morning",
           "whats the capital of france"]

async def main():
    llm = LLM(); t = time.time(); llm.load(); print("load", llm.info(), f"{time.time()-t:.1f}s")
    for q in QUERIES:
        t = time.time(); raw = await llm.chat(parse_messages(q), max_new_tokens=160)
        js = _extract_json(raw or "")
        print(f"{time.time()-t:.2f}s", q[:50], "->", json.dumps(js) if js else f"BAD: {raw!r}")
    for q in QUERIES[:3]:
        t = time.time(); r = await run(q, llm)
        print("=" * 90, f"\n{q}  [{time.time()-t:.1f}s] source={r['answer_source']} grounding={r.get('grounding')}\n{r['answer']}")

asyncio.run(main())
