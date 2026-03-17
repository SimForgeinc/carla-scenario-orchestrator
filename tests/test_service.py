from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.models import JobState, RuntimeExecutionResult, RuntimeLaunchSpec
from orchestrator.service import OrchestratorService
from orchestrator.carla_runner.models import (
    ActorDraft,
    ActorRoadAnchor,
    RecordingInfo,
    SelectedRoad,
    SimulationRunRequest,
    SimulationStreamMessage,
)


class FakeRuntimeBackend:
    def __init__(self) -> None:
        self.started_specs: list[RuntimeLaunchSpec] = []

    def run_job(self, spec, on_event, cancel_event):
        self.started_specs.append(spec)
        on_event(
            SimulationStreamMessage(
                frame=1,
                timestamp=time.time(),
                actors=[],
            )
        )
        on_event(
            SimulationStreamMessage(
                frame=2,
                timestamp=time.time(),
                actors=[],
                recording=RecordingInfo(
                    run_id=f"run-{spec.job_id}",
                    label="Test",
                    created_at="2026-03-16T00:00:00Z",
                ),
            )
        )
        if cancel_event.is_set():
            return RuntimeExecutionResult(state=JobState.cancelled, error="Job cancelled.")
        return RuntimeExecutionResult(
            state=JobState.succeeded,
            run_id=f"run-{spec.job_id}",
            manifest_path=str(Path(spec.output_dir) / "run" / "manifest.json"),
            recording_path=str(Path(spec.output_dir) / "run" / "recording.mp4"),
        )


def sample_request() -> SimulationRunRequest:
    return SimulationRunRequest(
        map_name="Town10HD_Opt",
        selected_roads=[SelectedRoad(id="10", name="Road 10")],
        actors=[
            ActorDraft(
                id="ego-1",
                label="Ego",
                kind="vehicle",
                role="ego",
                blueprint="vehicle.tesla.model3",
                spawn=ActorRoadAnchor(road_id="10"),
            )
        ],
        duration_seconds=1.0,
        topdown_recording=False,
    )


class ServiceTests(unittest.TestCase):
    def make_service(self, gpu_devices=("0", "1")) -> OrchestratorService:
        temp_dir = Path(tempfile.mkdtemp(prefix="carla-orchestrator-tests-"))
        settings = Settings(
            repo_root=temp_dir,
            jobs_root=temp_dir / "runs",
            gpu_devices=gpu_devices,
            carla_image="carla:test",
            carla_container_prefix="carla-orch",
            carla_startup_timeout_seconds=5,
            carla_rpc_port_base=2000,
            traffic_manager_port_base=8000,
            port_stride=100,
            carla_timeout_seconds=20,
            python_executable="python3",
            docker_network_mode="host",
            carla_start_command_template="./CarlaUE4.sh -carla-rpc-port={rpc_port}",
            utility_backend_base=None,
        )
        settings.jobs_root.mkdir(parents=True, exist_ok=True)
        return OrchestratorService(settings=settings, runtime_backend=FakeRuntimeBackend())

    def test_submit_job_runs_to_completion(self) -> None:
        service = self.make_service()
        response = service.submit_job(sample_request())
        deadline = time.time() + 3
        while time.time() < deadline:
            job = service.get_job(response.job_id)
            if job is not None and job.state == JobState.succeeded:
                break
            time.sleep(0.05)
        job = service.get_job(response.job_id)
        assert job is not None
        self.assertEqual(job.state, JobState.succeeded)
        self.assertEqual(job.gpu.device_id if job.gpu else None, "0")
        self.assertEqual(len(job.events), 2)
        self.assertTrue(job.artifacts.recording_path.endswith("recording.mp4"))

    def test_cancel_queued_job_marks_it_cancelled(self) -> None:
        blocker = threading.Event()

        class BlockingRuntimeBackend(FakeRuntimeBackend):
            def run_job(self, spec, on_event, cancel_event):
                blocker.wait(timeout=1)
                return super().run_job(spec, on_event, cancel_event)

        service = self.make_service(gpu_devices=("0",))
        service = OrchestratorService(settings=service.settings, runtime_backend=BlockingRuntimeBackend())
        first = service.submit_job(sample_request())
        second = service.submit_job(sample_request())
        time.sleep(0.1)
        cancelled = service.cancel_job(second.job_id)
        self.assertEqual(cancelled.state, JobState.cancelled)
        blocker.set()
        deadline = time.time() + 2
        while time.time() < deadline:
            job = service.get_job(first.job_id)
            if job is not None and job.state == JobState.succeeded:
                break
            time.sleep(0.05)
        queued_job = service.get_job(second.job_id)
        assert queued_job is not None
        self.assertEqual(queued_job.state, JobState.cancelled)


if __name__ == "__main__":
    unittest.main()
