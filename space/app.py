"""Stable public entry point for the GeoAI outdoor-activity assistant.

The assistant itself (model + data pipeline) runs on the author's GPU workstation and is exposed through a
Cloudflare tunnel. This Space only forwards requests to it, so the public URL never changes even if the tunnel
does; the current tunnel address is kept in the TARGET_URL Space variable.
"""
import os

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="GeoAI Outdoor Planner (gateway)")
client = httpx.AsyncClient(timeout=180, follow_redirects=True)


@app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def proxy(path: str, request: Request):
    target = os.environ.get("TARGET_URL", "").rstrip("/")
    if not target:
        return JSONResponse({"error": "backend not configured"}, status_code=503)
    try:
        r = await client.request(request.method, f"{target}/{path}", params=request.query_params,
                                 content=await request.body(),
                                 headers={"content-type": request.headers.get("content-type", "application/json")})
    except httpx.HTTPError as e:
        return JSONResponse({"error": "The assistant backend is temporarily unreachable; please retry in a minute.",
                             "detail": type(e).__name__}, status_code=503)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"),
                    headers={"access-control-allow-origin": "*"})
