"""Shared async HTTP client.

Every external call goes through `fetch_json` so that we get, in one place:
  * a polite, identifiable User-Agent (required by Nominatim / Overpass usage policies),
  * an on-disk TTL cache (respecting rate limits, and surviving restarts),
  * retries and mirror failover,
  * a per-request trace of every source we touched, which the API returns as evidence.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx

USER_AGENT = os.environ.get(
    "GEOAI_USER_AGENT",
    "GeoAI-Outdoor-Assistant/0.1 (+https://github.com/harshit-singh-hs/GeoAI-Hackathon-2026)",
)
CACHE_PATH = os.environ.get("GEOAI_CACHE", "cache/http_cache.sqlite")


class FetchError(RuntimeError):
    pass


@dataclass
class SourceRecord:
    source: str
    url: str
    fetched_at: str
    cached: bool
    ok: bool = True
    note: str = ""


@dataclass
class Trace:
    records: list[SourceRecord] = field(default_factory=list)

    def add(self, rec: SourceRecord) -> None:
        self.records.append(rec)


_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("geoai_trace", default=None)


def start_trace() -> Trace:
    t = Trace()
    _trace.set(t)
    return t


def record(source: str, url: str, cached: bool, ok: bool = True, note: str = "") -> None:
    t = _trace.get()
    if t is not None:
        t.add(SourceRecord(source, url, datetime.now(timezone.utc).isoformat(timespec="seconds"), cached, ok, note))


class _DiskCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, ts REAL, v TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float):
        with self._lock:
            row = self._db.execute("SELECT ts, v FROM c WHERE k=?", (key,)).fetchone()
        if row and time.time() - row[0] < ttl:
            return json.loads(row[1])
        return None

    def set(self, key: str, value) -> None:
        with self._lock:
            self._db.execute("REPLACE INTO c VALUES (?,?,?)", (key, time.time(), json.dumps(value)))
            self._db.commit()


_cache = _DiskCache(CACHE_PATH)
_client: httpx.AsyncClient | None = None
# Per-host minimum spacing between requests (seconds); Nominatim policy is max 1 req/s.
_HOST_SPACING = {"nominatim.openstreetmap.org": 1.1}
_host_locks: dict[str, asyncio.Lock] = {}
_host_last: dict[str, float] = {}


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    return _client


def _cache_key(url: str, params: dict | None, data: dict | None) -> str:
    raw = url + "?" + urlencode(sorted((params or {}).items()), doseq=True) + "#" + json.dumps(data, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


async def _throttle(url: str) -> None:
    host = httpx.URL(url).host
    spacing = _HOST_SPACING.get(host)
    if not spacing:
        return
    lock = _host_locks.setdefault(host, asyncio.Lock())
    async with lock:
        wait = _host_last.get(host, 0) + spacing - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _host_last[host] = time.monotonic()


async def fetch_json(
    urls: str | list[str],
    *,
    source: str,
    params: dict | None = None,
    data: dict | None = None,
    ttl: float = 3600,
    timeout: float = 30.0,
    retries: int = 1,
):
    """GET (or POST form `data`) JSON from the first URL/mirror that answers.

    `urls` may be a list of mirrors; they are tried in order. Results are cached on disk for `ttl` seconds.
    """
    mirrors = [urls] if isinstance(urls, str) else list(urls)
    key = _cache_key(mirrors[0], params, data)
    hit = _cache.get(key, ttl)
    shown_url = mirrors[0] + ("?" + urlencode(params, doseq=True) if params else "")
    if hit is not None:
        record(source, shown_url, cached=True)
        return hit

    last_err: Exception | None = None
    for url in mirrors:
        for attempt in range(retries + 1):
            try:
                await _throttle(url)
                if data is not None:
                    r = await client().post(url, params=params, data=data, timeout=timeout)
                else:
                    r = await client().get(url, params=params, timeout=timeout)
                if r.status_code in (429, 502, 503, 504):
                    raise FetchError(f"{url} -> HTTP {r.status_code}")
                r.raise_for_status()
                out = r.json()
                _cache.set(key, out)
                record(source, url + ("?" + urlencode(params, doseq=True) if params else ""), cached=False)
                return out
            except (httpx.HTTPError, FetchError, ValueError) as e:
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
    record(source, shown_url, cached=False, ok=False, note=str(last_err)[:200])
    raise FetchError(f"{source}: all endpoints failed ({last_err})")
