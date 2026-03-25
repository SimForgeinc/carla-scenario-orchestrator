# CARLA Scenario Orchestrator

Multi-GPU simulation backend for SimCloud. Manages 7 persistent CARLA instances
across A100 GPUs, dispatches simulation jobs via Temporal workflows, and streams
real-time actor positions to the editor frontend.

## Quick Reference

```bash
# Check health
curl http://127.0.0.1:18421/api/health

# Check slot capacity
curl http://127.0.0.1:18421/api/capacity

# Restart orchestrator
sudo systemctl restart carla-scenario-orchestrator.service

# View logs
sudo journalctl -u carla-scenario-orchestrator.service -f

# Temporal Web UI
http://216.151.21.122:8080

# Run tests
cd /home/ubuntu/carla-scenario-orchestrator
.venv/bin/python -m pytest tests/ -v
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams and data flow.

Summary: 7 GPU slots, each running a persistent CARLA Docker container and a
persistent Temporal worker process. Jobs are scheduled to the slot that already
has the requested map loaded (avoiding 5-14s map switch penalty). The Traffic
Manager is created fresh by each simulation run — never pre-warmed at startup.

```
User clicks Simulate
  → Orchestrator picks best slot (map-aware, LRU)
  → Temporal dispatches to slot's worker process
  → Worker reuses persistent CARLA client
  → Simulation runs in sync mode (world.tick() loop)
  → Frames stream to browser via WebSocket
  → Artifacts upload to S3, local files deleted
  → Slot released for next job
```

## Repo Layout

```
orchestrator/               Python backend
  app.py                    FastAPI endpoints
  service.py                Job lifecycle, slot dispatch
  scheduler.py              Map-aware LRU slot scheduling
  worker_pool.py            Temporal persistent worker pool
  carla_metadata.py         Distributed metadata queries
  artifact_storage.py       S3-first artifact upload
  runtime_backend.py        Docker container lifecycle
  config.py                 Settings from .env
  models.py                 Pydantic models
  store.py                  In-memory job store
  carla_runner/             CARLA simulation logic
    simulation_service.py   Core simulation loop
    models.py               Request/response models
    dataset_repository.py   Map metadata
  llm/                      Bedrock scenario generation
monitor/                    Svelte ops dashboard
scripts/                    Benchmark and utility scripts
tests/                      Unit tests (27 tests)
```

## GPU Allocation

```
GPU 0: Excluded (reserved for other use)
GPUs 1-7: 7 execution slots (all serve simulations + metadata queries)
```

No dedicated metadata GPU — metadata queries are routed to whichever slot
already has the requested map loaded.

## Port Plan

| Slot | GPU | CARLA RPC | Traffic Manager |
|------|-----|-----------|-----------------|
| 0    | 1   | 18467     | 19467           |
| 1    | 2   | 18504     | 19504           |
| 2    | 3   | 18541     | 19541           |
| 3    | 4   | 18578     | 19578           |
| 4    | 5   | 18615     | 19615           |
| 5    | 6   | 18652     | 19652           |
| 6    | 7   | 18689     | 19689           |

Orchestrator API: `18421`
Temporal server: `7233`
Temporal UI: `8080`

## API Endpoints

### Core
- `GET /api/health` — orchestrator + slot health
- `GET /api/capacity` — slot status, maps, busy/free counts

### Jobs
- `POST /api/jobs` — submit simulation
- `GET /api/jobs` — list all jobs
- `GET /api/jobs/{id}` — job detail with events
- `POST /api/jobs/{id}/cancel` — cancel job
- `GET /api/jobs/{id}/log` — debug log
- `GET /api/jobs/{id}/diagnostics` — run diagnostics
- `WS /api/jobs/{id}/stream` — real-time event stream

### Map & Metadata
- `GET /api/carla/status` — CARLA connection status
- `POST /api/carla/map/load` — switch map
- `GET /api/map/runtime` — road segments, waypoints
- `GET /api/map/generated` — generated map with runtime data
- `GET /api/map/xodr` — OpenDRIVE export
- `GET /api/maps/supported` — available map list
- `GET /api/actors/blueprints` — vehicle/walker blueprints

### Operations
- `POST /api/slots/preload` — pre-load a map on idle slot
- `POST /api/jobs/{id}/events` — internal: workers push events
- `GET /metrics` — Prometheus metrics

### LLM
- `POST /api/llm/generate` — Bedrock scenario generation
- `POST /api/llm/scene-assistant` — scene assistant chat

## Environment

Key variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_GPU_DEVICES` | `1,2,3,4,5,6,7` | GPU device IDs |
| `ORCH_METADATA_SLOT_INDEX` | `-1` | `-1` = no dedicated metadata slot |
| `ORCH_CARLA_IMAGE` | `carlasim/carla:0.9.16-vwpoc` | CARLA Docker image |
| `ORCH_STORAGE_BUCKET` | — | S3 bucket for artifacts |
| `ORCH_CARLA_RPC_PORT_BASE` | `18467` | First slot's CARLA port |
| `ORCH_PORT_STRIDE` | `37` | Port increment per slot |

## Infrastructure

### Temporal
Self-hosted Docker Compose at `/home/ubuntu/temporal-docker/`:
- Temporal v1.29.1, Python SDK v1.24.0
- PostgreSQL + Elasticsearch persistence
- Web UI at port 8080

### Systemd
- `carla-scenario-orchestrator.service` — main backend
- `carla-scenario-monitor.service` — ops dashboard (port 3001)

### Docker Containers
- `carla-orch-slot-{0-5}` — CARLA instances
- `carla-orch-metadata` — CARLA instance (now slot 6, execution)
- `temporal`, `temporal-postgresql`, `temporal-elasticsearch`, `temporal-ui`

## Testing

```bash
# All tests (27 total: 12 scheduler + 15 worker_pool)
.venv/bin/python -m pytest tests/test_scheduler.py tests/test_worker_pool.py -v

# Throughput benchmark (50-1000 jobs)
.venv/bin/python scripts/benchmark_throughput.py --jobs 100 --map VW_Poc --concurrent 20

# Movement verification benchmark
.venv/bin/python /tmp/bench_movement.py
```

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Needs matching CARLA Python wheel (carla==0.9.16)
uvicorn orchestrator.app:app --host 0.0.0.0 --port 18421
```
