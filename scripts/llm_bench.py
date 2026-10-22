"""Compare SLMs on the narration step for the same live results (same facts for every model)."""
import asyncio, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.llm import LLM
from app.pipeline import run
from app.narrate import narrate

QS = ["Where can I go running tomorrow evening near Koregaon Park?", "Is Saturday or Sunday better for a hike near Lonavala?",
      "Where can I take sunset photos this evening around Bandra?", "Good place and time for golf on Saturday within 40 km of Andheri?"]

async def main():
    llm = LLM(); llm.load()
    for q in QS:
        r = await run(q, None)  # facts without the SLM, then narrate with it
        t = time.time(); await narrate(r, llm)
        print("=" * 90, f"\n{q}  [narrate {time.time()-t:.1f}s] source={r['answer_source']} grounding={r.get('grounding')}\n{r['answer']}")

asyncio.run(main())
