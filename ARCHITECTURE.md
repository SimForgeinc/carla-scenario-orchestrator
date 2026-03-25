# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Ubuntu Server (216.151.21.122)  —  8x A100 GPUs            │
│                                                              │
│  ┌─── Docker Containers ─────────────────────────────────┐  │
│  │  carla-orch-slot-0  (GPU 1) ── CarlaUE4 process       │  │
│  │  carla-orch-slot-1  (GPU 2) ── CarlaUE4 process       │  │
│  │  carla-orch-slot-2  (GPU 3) ── CarlaUE4 process       │  │
│  │  carla-orch-slot-3  (GPU 4) ── CarlaUE4 process       │  │
│  │  carla-orch-slot-4  (GPU 5) ── CarlaUE4 process       │  │
│  │  carla-orch-slot-5  (GPU 6) ── CarlaUE4 process       │  │
│  │  carla-orch-metadata(GPU 7) ── CarlaUE4 (slot 6)      │  │
│  │                                                        │  │
│  │  temporal            ── Workflow orchestration server   │  │
│  │  temporal-postgresql  ── Temporal's database            │  │
│  │  temporal-ui          ── Web UI (port 8080)            │  │
│  │  temporal-elasticsearch ── Workflow search              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─── Orchestrator (systemd, port 18421) ────────────────┐  │
│  │  Main: uvicorn (FastAPI)                               │  │
│  │    ├── Worker process slot 0 (Temporal, persistent)    │  │
│  │    ├── Worker process slot 1 (Temporal, persistent)    │  │
│  │    ├── Worker process slot 2 (Temporal, persistent)    │  │
│  │    ├── Worker process slot 3 (Temporal, persistent)    │  │
│  │    ├── Worker process slot 4 (Temporal, persistent)    │  │
│  │    ├── Worker process slot 5 (Temporal, persistent)    │  │
│  │    ├── Worker process slot 6 (Temporal, persistent)    │  │
│  │    └── Health check thread (every 30s)                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  GPU 0: Not in orchestrator pool (reserved for other use)    │
└─────────────────────────────────────────────────────────────┘
```

## Job Lifecycle

When a user clicks "Simulate" in the editor:

```
Browser (SvelteKit editor)
     │
     │  POST /api/jobs  { map_name, actors, duration, ... }
     ▼
Orchestrator (FastAPI)
     │
     ├── 1. SCHEDULER: Pick best GPU slot
     │   ├── Prefer slot with matching map loaded (zero switch time)
     │   ├── Among matches, pick least-recently-used (even distribution)
     │   └── Fall back to any free slot if no map match
     │
     ├── 2. TEMPORAL DISPATCH
     │   ├── Start SimulationWorkflow on slot's task queue
     │   └── Temporal routes to the correct worker process
     │
     ▼
Worker Process (one per slot, persistent)
     │
     ├── 3a. Reuse persistent CARLA client (no reconnect overhead)
     │
     ├── 3b. Check/load map if needed (0ms if cached, 5-14s if switching)
     │
     ├── 3c. Set synchronous mode + fixed delta (0.05s = 20 FPS physics)
     │
     ├── 3d. Create Traffic Manager (4.2s first time, instant after)
     │   └── TM is a CARLA-internal process that drives cars via autopilot
     │
     ├── 3e. Spawn actors (vehicles, walkers) at road positions
     │   └── set_autopilot(True) → TM controls steering/throttle/brake
     │
     ├── 3f. SIMULATION LOOP (core)
     │   │   for each tick:
     │   │     world.tick()        → physics advances 0.05s
     │   │     read positions      → x, y, z, speed per actor
     │   │     send frame event    → batched HTTP POST to orchestrator
     │   │                           → WebSocket to browser → dots on map
     │   └── repeat for duration_seconds / fixed_delta_seconds ticks
     │
     ├── 3g. Cleanup: destroy actors, restore async mode, stop recorder
     │   └── CARLA client stays connected for next job
     │
     └── 4. Return result to Temporal → Orchestrator
              ├── Upload artifacts to S3
              ├── Delete local files
              ├── Release slot (scheduler marks free)
              └── WebSocket sends "stream_complete"
```

## Persistent Worker Pool

Each GPU slot runs a **persistent worker process** that stays alive across jobs:

```
Worker process lifecycle:
  startup → connect to CARLA (once) → connect to Temporal → wait for jobs
       ↓
  job arrives → run simulation → cleanup → wait for next job
       ↓
  job arrives → run simulation → cleanup → wait for next job
       ↓
  ... (forever, until orchestrator restarts)
```

Benefits over the old subprocess-per-job model:
- No Python startup overhead (~1.1s saved per job)
- No CARLA reconnection overhead (~0.5s saved per job)
- Temporal handles retries, heartbeats, and crash recovery

Workers are spawned with `multiprocessing.get_context('spawn')` to avoid
Temporal's Rust runtime fork assertion.

## Traffic Manager (TM)

The TM is a **separate process inside the CARLA Docker container** that plans
routes and sends throttle/steer/brake commands to vehicles with autopilot enabled.

Critical constraint: **only the simulation_service should create the TM.**

```
CORRECT:
  Worker startup: connect to CARLA (no TM)
  Simulation:     tm = client.get_trafficmanager(port)  ← creates TM
                  tm.set_synchronous_mode(True)
                  vehicle.set_autopilot(True, port)
                  world.tick() → car moves ✓

WRONG (causes frozen cars):
  Worker startup: client.get_trafficmanager(port)  ← creates TM early, binds port
  Simulation:     client.get_trafficmanager(port)  ← gets stale reference
                  vehicle.set_autopilot(True, port)
                  world.tick() → car reports speed but position frozen ✗
```

The TM persists in CARLA's memory once created. First job per slot pays a ~4.2s
cold-start penalty; all subsequent jobs reuse the existing TM instantly.

## Map-Aware Scheduling

The scheduler tracks which map each slot has loaded:

```
Slot 0: VW_Poc        ← user requests VW_Poc → routed here (0ms)
Slot 1: Town10HD_Opt  ← user requests Town10HD → routed here (0ms)
Slot 2: VW_Poc        ← user requests VW_Poc → routed here if slot 0 busy
Slot 3: VW_Poc
Slot 4: Town10HD_Opt
Slot 5: VW_Poc
Slot 6: Town10HD_Opt
```

Two-pass algorithm:
1. Find all free slots with matching map → pick LRU among them
2. If no match, pick any free slot (will load map, 5-14s penalty)

LRU (Least Recently Used) selection ensures even distribution when multiple
slots are free — tracks `last_released_time` per slot.

## Distributed Metadata

There is **no dedicated metadata GPU**. All 7 slots serve both simulations
and metadata queries (map data, blueprints, XODR export).

```
Editor requests map data for VW_Poc
  → _resolve_metadata_slot("VW_Poc")
  → finds idle slot with VW_Poc loaded
  → creates temporary carla.Client to that slot's port
  → executes query (instant, no map switch)
  → returns result
```

Priority: idle + map match > any idle > busy + map match > fallback.

Metadata responses are cached aggressively:
- Runtime map data: cached per map name
- XODR: cached per map name
- Generated map: cached per map name
- Blueprints: cached globally (same across all maps)

## Health Monitoring

A background thread checks worker health every 30 seconds:

```
Worker alive? → yes → reset failure counter, mark ready
             → no  → increment failure counter
                      ├── failures < 3: restart worker (exponential backoff)
                      ├── failure 2+: also restart Docker container
                      └── failures >= 3: mark slot UNHEALTHY
                          (scheduler stops routing jobs to it)
```

Backoff: 30s, 60s, 120s, max 5 minutes between restart attempts.

Recovery: if a previously unhealthy worker comes back alive, it's
automatically marked ready and re-enters the scheduling pool.

## Event Streaming

Simulation frames flow from CARLA to the user's browser:

```
CARLA world.tick()
  → Worker reads actor positions
  → EventPusher batches 20 frames
  → HTTP POST /api/jobs/{id}/events (with 1-retry on failure)
  → Orchestrator stores in JobStore
  → WebSocket /api/jobs/{id}/stream polls and sends to browser
  → Frontend renders actor dots on the map
```

## Artifact Storage

S3-first: all artifacts are uploaded to S3 immediately after job completion,
then the local directory is deleted.

```
S3 layout:
  {prefix}/{source_run_id}/executions/{job_id}/{run_id}/
    ├── manifest.json
    ├── recording.mp4
    ├── scenario.log
    ├── run.log
    └── (other files)
```

Individual JPEG frames from `ego_camera_frames/` are skipped during upload.

## Temporal

Self-hosted via Docker Compose at `/home/ubuntu/temporal-docker/`:
- Temporal server v1.29.1 (port 7233)
- Temporal Web UI (port 8080)
- PostgreSQL persistence
- Elasticsearch for workflow search

Python SDK: temporalio v1.24.0

Workflows:
- `SimulationWorkflow`: wraps `run_simulation_activity` with 10-min timeout,
  60s heartbeat, and 1 retry on failure
- `PreloadMapWorkflow`: wraps `preload_map_activity` with 2-min timeout

Worker config per slot:
- `max_concurrent_activities=1` (one CARLA instance = one simulation at a time)
- `max_cached_workflows=2`
- `graceful_shutdown_timeout=30s`

## Prometheus Metrics

Exposed at `GET /metrics` via `prometheus-fastapi-instrumentator`:
- `carla_jobs_total{status}` — counter of completed jobs
- `carla_job_duration_seconds` — histogram of job wall-clock duration
- `carla_worker_restarts_total{slot}` — counter of worker process restarts
- `carla_slot_busy{slot}` — gauge, 1 when slot is running a job

## Benchmarks (2026-03-25)

### Throughput (100 jobs, VW_Poc map, 7 slots)

```
Success rate:   100% (100/100)
Total time:     366s
Throughput:     16.4 jobs/min
Latency:        p50=37s  p95=131s  min=3.1s

Per-slot distribution:
  slot 0:  9 jobs
  slot 1: 12 jobs
  slot 2: 17 jobs
  slot 3: 13 jobs
  slot 4: 15 jobs
  slot 5: 17 jobs
  slot 6: 14 jobs  (former metadata GPU)
```

### Movement Verification (50 jobs)

```
Succeeded:      48/50 (2 timed out from queue wait)
Cars moved:     48/48 — ALL cars moved on ALL 7 slots
Avg distance:   23.9m per 5s simulation
Avg end speed:  5.8 m/s (20.9 km/h)
Zero stuck:     0 cars frozen
```

### Per-Job Timing

```
Cold start (first job per slot, includes TM init):  ~10s
Warm (subsequent jobs, TM reused):                   ~3.1s
Map switch penalty (if slot has wrong map):           5-14s
```

### Historical Comparison

| Config | Slots | Success | Throughput | Min Latency |
|--------|-------|---------|------------|-------------|
| Before optimizations | 5 (1 zombie) | 99% | 9.7/min | 3.5s |
| After + slot 1 restored | 6 | 100% | 27.2/min | 3.7s |
| After + no metadata GPU | 7 | 100% | 16.4/min | 3.1s |
