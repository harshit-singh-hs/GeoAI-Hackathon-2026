"""HTTP API.

  POST /ask         {"query": "...", "lat": optional, "lon": optional}  -> grounded recommendation (JSON)
  GET  /ask?q=...   same, convenient from a browser or curl
  GET  /health      model + data-source status
  GET  /activities  supported activities and personal profiles
  GET  /map/{id}    interactive map for a previous answer
  GET  /            web UI
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import pipeline
from .activities import ACTIVITIES, PROFILES
from .data.osm import INDEX_PATH, OSM_MODE, OVERPASS
from .llm import LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"
MAX_CONCURRENT = int(os.environ.get("GEOAI_MAX_CONCURRENT", "6"))

llm = LLM()
_sem = asyncio.Semaphore(MAX_CONCURRENT)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("GEOAI_NO_LLM") != "1":
        await asyncio.to_thread(llm.load)
    yield


app = FastAPI(title="GeoAI Outdoor Activity Assistant", version="0.1.0", lifespan=lifespan,
              description="Recommends where and when to go outdoors, grounded in live OSM, weather and air-quality data.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["Where can I go running tomorrow evening near Koregaon Park?"])
    lat: float | None = Field(None, ge=-90, le=90, description="Optional caller location, used when the question names no place")
    lon: float | None = Field(None, ge=-180, le=180)
    verbose: bool = Field(False, description="Include the fact sheet and full per-option details")


def _public(res: dict, verbose: bool) -> dict:
    if verbose:
        return res
    out = {k: v for k, v in res.items() if k not in ("fact_sheet",)}
    return out


async def _ask(query: str, lat: float | None, lon: float | None, verbose: bool) -> dict:
    async with _sem:
        try:
            res = await asyncio.wait_for(pipeline.run(query, llm if llm.ready else None, lat, lon), timeout=120)
        except asyncio.TimeoutError:
            raise HTTPException(504, "The live data sources took too long to respond. Please retry.")
    res["map_url"] = f"/map/{res['id']}" if res.get("geojson") else None
    res["model"] = {"name": llm.model_id, "adapter": llm.adapter, "used_for": ["query parsing", "answer writing"],
                    "loaded": llm.ready}
    res["attribution"] = ("Map data © OpenStreetMap contributors (ODbL). Weather: Open-Meteo.com (CC BY 4.0). "
                          "Air quality: CAMS via Open-Meteo. Routing: OSRM.")
    return _public(res, verbose)


@app.post("/ask")
async def ask_post(body: AskBody):
    return await _ask(body.query, body.lat, body.lon, body.verbose)


@app.get("/ask")
async def ask_get(q: str = Query(..., min_length=1, max_length=500), lat: float | None = None, lon: float | None = None,
                  verbose: bool = False):
    return await _ask(q, lat, lon, verbose)


@app.get("/health")
async def health():
    return {"status": "ok", "llm": llm.info(), "osm_mode": OSM_MODE, "overpass": OVERPASS,
            "offline_osm_index": os.path.exists(INDEX_PATH)}


@app.get("/activities")
async def activities():
    return {"activities": [{"id": a.id, "name": a.name, "examples": list(a.synonyms[:4])} for a in ACTIVITIES.values()],
            "profiles": [{"id": p.id, "label": p.label} for p in PROFILES.values()]}


@app.get("/result/{rid}")
async def result(rid: str):
    if rid not in pipeline.RESULTS:
        raise HTTPException(404, "unknown or expired result id")
    return pipeline.RESULTS[rid]


@app.get("/map/{rid}", response_class=HTMLResponse)
async def map_page(rid: str):
    if rid not in pipeline.RESULTS:
        raise HTTPException(404, "unknown or expired result id")
    return HTMLResponse((STATIC / "map.html").read_text().replace("__RID__", rid))


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
