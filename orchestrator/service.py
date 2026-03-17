from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from .config import Settings
from .models import (
    CancelJobResponse,
    CompatibilityRunResponse,
    HealthResponse,
    JobArtifacts,
    JobListResponse,
    JobRecord,
    JobState,
    JobSubmissionResponse,
    RuntimeLaunchSpec,
)
from .runtime_backend import DockerRuntimeBackend, RuntimeBackend
from .scheduler import GpuScheduler
from .store import JobStore
from .carla_runner.dataset_repository import list_supported_maps
from .carla_runner.models import SimulationRunRequest, SimulationStreamMessage


class OrchestratorService:
    def __init__(
        self,
        settings: Settings,
        scheduler: GpuScheduler | None = None,
        store: JobStore | None = None,
        runtime_backend: RuntimeBackend | None = None,
    ) -> None:
        self.settings = settings
        self.scheduler = scheduler or GpuScheduler(settings)
        self.store = store or JobStore()
        self.runtime_backend = runtime_backend or DockerRuntimeBackend(settings)
        self._cancel_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit_job(self, request: SimulationRunRequest) -> JobSubmissionResponse:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.settings.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        artifacts = JobArtifacts(
            output_dir=str(job_dir),
            request_file=str(job_dir / "request.json"),
            runtime_settings_file=str(job_dir / "runtime_settings.json"),
        )
        job = self.store.create(job_id, request, artifacts)
        cancel_event = threading.Event()
        worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        with self._lock:
            self._cancel_events[job_id] = cancel_event
            self._threads[job_id] = worker
        self.store.update_queue_positions()
        worker.start()
        current = self.store.get(job_id)
        assert current is not None
        return JobSubmissionResponse(job_id=job_id, state=current.state, queue_position=current.queue_position)

    def submit_compatibility_job(self, request: SimulationRunRequest) -> CompatibilityRunResponse:
        response = self.submit_job(request)
        return CompatibilityRunResponse(
            status="accepted",
            job_id=response.job_id,
            state=response.state,
            queue_position=response.queue_position,
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.store.get(job_id)

    def list_jobs(self) -> JobListResponse:
        return JobListResponse(items=self.store.list())

    def cancel_job(self, job_id: str) -> CancelJobResponse:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        with self._lock:
            cancel_event = self._cancel_events.get(job_id)
        if cancel_event is not None:
            cancel_event.set()
        if job.state == JobState.queued:
            self.store.update(job_id, state=JobState.cancelled, error="Job cancelled before start.")
            self.store.update_queue_positions()
        current = self.store.get(job_id)
        assert current is not None
        return CancelJobResponse(job_id=job_id, state=current.state)

    def supported_maps(self) -> list[str]:
        return sorted(list_supported_maps())

    def capacity(self):
        return self.scheduler.snapshot()

    def health(self) -> HealthResponse:
        capacity = self.scheduler.snapshot()
        return HealthResponse(
            total_slots=capacity.total_slots,
            busy_slots=capacity.busy_slots,
            queued_jobs=self.store.queued_count(),
        )

    def _run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        with self._lock:
            cancel_event = self._cancel_events[job_id]
        try:
            lease = self.scheduler.acquire(job_id, cancel_event)
        except RuntimeError as exc:
            current = self.store.get(job_id)
            if current is not None and current.state == JobState.cancelled:
                return
            self.store.update(job_id, state=JobState.cancelled, error=str(exc))
            return

        self.store.update_queue_positions()
        self.store.update(job_id, state=JobState.starting, gpu=lease.to_model(), queue_position=0)

        runtime_spec = self._write_runtime_files(job, lease.to_model())
        self.store.update(job_id, container_name=f"{self.settings.carla_container_prefix}-{job_id}".lower())

        def on_event(payload: SimulationStreamMessage) -> None:
            self.store.append_event(job_id, payload)
            current = self.store.get(job_id)
            if current is None:
                return
            updates = {}
            if current.state == JobState.starting:
                updates["state"] = JobState.running
            if payload.error and current.state != JobState.cancelled:
                updates["error"] = payload.error
            if payload.recording is not None:
                updates["run_id"] = payload.recording.run_id
            if updates:
                self.store.update(job_id, **updates)

        try:
            result = self.runtime_backend.run_job(runtime_spec, on_event, cancel_event)
            updates = {
                "state": result.state,
                "error": result.error,
                "run_id": result.run_id,
                "artifacts": job.artifacts.model_copy(
                    update={
                        "manifest_path": result.manifest_path,
                        "recording_path": result.recording_path,
                        "scenario_log_path": result.scenario_log_path,
                        "debug_log_path": result.debug_log_path,
                    }
                ),
            }
            self.store.update(job_id, **updates)
        except Exception as exc:  # noqa: BLE001
            final_state = JobState.cancelled if cancel_event.is_set() else JobState.failed
            self.store.update(job_id, state=final_state, error=str(exc))
        finally:
            self.scheduler.release(job_id)
            self.store.update_queue_positions()

    def _write_runtime_files(self, job: JobRecord, gpu) -> RuntimeLaunchSpec:
        job_dir = Path(job.artifacts.output_dir)
        request_file = Path(job.artifacts.request_file)
        runtime_settings_file = Path(job.artifacts.runtime_settings_file)
        request_file.write_text(job.request.model_dump_json(indent=2), encoding="utf-8")
        runtime_settings = {
            "carla_host": "127.0.0.1",
            "carla_port": gpu.carla_rpc_port,
            "carla_timeout": self.settings.carla_timeout_seconds,
            "tm_port": gpu.traffic_manager_port,
            "output_root": str(job_dir),
        }
        runtime_settings_file.write_text(json.dumps(runtime_settings, indent=2), encoding="utf-8")
        return RuntimeLaunchSpec(
            job_id=job.job_id,
            request_file=str(request_file),
            runtime_settings_file=str(runtime_settings_file),
            output_dir=str(job_dir),
            gpu=gpu,
        )

