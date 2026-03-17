# CARLA Scenario Orchestrator

`carla-scenario-orchestrator` is the control plane for multi-tenant CARLA execution on a shared GPU host.

The design goal is one user request per isolated CARLA runtime:

- one API process receives jobs
- one GPU slot is reserved per active job
- one CARLA Docker container is launched per job
- one scenario runner subprocess connects to that CARLA instance and executes the scenario
- artifacts are written under `runs/<job_id>/`
- generated artifacts can also be pushed to S3 for durable storage

This repo reuses the proven scenario execution worker from `carla-scenario-tool-server`, but removes the singleton service model that only allowed one active simulation at a time.

## API

- `GET /api/health`
- `GET /api/capacity`
- `GET /api/maps/supported`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs`
- `POST /api/jobs/{job_id}/cancel`
- `POST /api/simulation/run`

`POST /api/simulation/run` is a compatibility endpoint that accepts the existing `SimulationRunRequest` payload and returns a job identifier.

## Environment

See [`.env.example`](/Users/maikyon/Documents/Programming/carla-scenario-orchestrator/.env.example).

The important variables are:

- `ORCH_GPU_DEVICES=0,1,2,3,4,5,6,7`
- `ORCH_CARLA_IMAGE=carlasim/carla:0.9.16-phase3`
- `ORCH_CARLA_RPC_PORT_BASE=2000`
- `ORCH_TRAFFIC_MANAGER_PORT_BASE=8000`
- `ORCH_PORT_STRIDE=100`
- `ORCH_STORAGE_BUCKET=simcloud-assets-public-test`
- `ORCH_STORAGE_REGION=us-east-1`
- `ORCH_STORAGE_PREFIX=runs`

With the defaults, GPU slot `0` uses CARLA RPC `2000` and TM `8000`, slot `1` uses `2100` and `8100`, and so on.

When `ORCH_STORAGE_BUCKET` is set, the orchestrator uploads `manifest.json`, `recording.mp4`, `scenario.log`, and `run.log` to:

```text
runs/<source_run_id>/executions/<job_id>/<backend_run_id>/
```

Set `source_run_id` on the submitted simulation request so uploaded artifacts line up with the originating SimCloud run.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn orchestrator.app:app --host 0.0.0.0 --port 8002
```

You still need a matching CARLA Python wheel installed in the same environment for the runner subprocess.

## Test

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
