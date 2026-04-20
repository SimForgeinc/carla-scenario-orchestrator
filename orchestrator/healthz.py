"""Minimal aiohttp liveness probe for CARLA workers.

Exposes two routes on the configured port (default 8081):

  GET /healthz  -> 200 if a heartbeat landed in the last 30s, else 503
  GET /ready    -> 200 once the worker has confirmed CARLA RPC is ready
"""

from __future__ import annotations

import time
from typing import Any

from aiohttp import web

STALE_AFTER_S = 30


async def _healthz(request: web.Request) -> web.Response:
    state: Any = request.app["state"]
    now = time.time()
    age = now - (state.last_heartbeat_ts or 0.0)
    body = {
        "status": "ok" if age < STALE_AFTER_S else "stale",
        "worker_id": state.worker_id,
        "last_heartbeat": state.last_heartbeat_ts,
        "age_seconds": round(age, 3),
    }
    status = 200 if age < STALE_AFTER_S else 503
    return web.json_response(body, status=status)


async def _ready(request: web.Request) -> web.Response:
    state: Any = request.app["state"]
    if state.ready:
        return web.json_response({"ready": True, "worker_id": state.worker_id})
    return web.json_response({"ready": False}, status=503)


async def start_healthz_server(port: int, state: Any) -> web.AppRunner:
    app = web.Application()
    app["state"] = state
    app.add_routes([web.get("/healthz", _healthz), web.get("/ready", _ready)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    return runner
