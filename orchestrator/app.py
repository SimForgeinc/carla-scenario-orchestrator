from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .models import CancelJobResponse, CompatibilityRunResponse, JobListResponse, JobRecord, JobSubmissionResponse
from .service import OrchestratorService
from .carla_runner.models import SimulationRunRequest


settings = Settings.load()
service = OrchestratorService(settings)
app = FastAPI(title="CARLA Scenario Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return service.health()


@app.get("/api/capacity")
async def capacity():
    return service.capacity()


@app.get("/api/maps/supported")
async def supported_maps():
    return {"maps": service.supported_maps()}


@app.get("/api/jobs", response_model=JobListResponse)
async def list_jobs():
    return service.list_jobs()


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
async def get_job(job_id: str):
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/api/jobs", response_model=JobSubmissionResponse)
async def submit_job(request: SimulationRunRequest):
    return service.submit_job(request)


@app.post("/api/jobs/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(job_id: str):
    try:
        return service.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.post("/api/simulation/run", response_model=CompatibilityRunResponse)
async def compatibility_run(request: SimulationRunRequest):
    return service.submit_compatibility_job(request)

