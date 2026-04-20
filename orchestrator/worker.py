"""Single-file CARLA worker process.

Runs as PID 1 in the container image. Launches CARLA as a subprocess,
waits for its RPC port to accept connections, then claims jobs from the
control plane, runs them via ``carla_runner._simulation_worker``, streams
per-tick events back to the control plane, uploads artifacts to S3, and
marks completion/failure through the control plane.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from botocore.exceptions import ClientError
from pydantic import ValidationError

log = logging.getLogger("orchestrator.worker")


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"{name} must be set")
    return value or ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


POOL = _env("POOL", "on-prem")
GPU_INDEX = _env("GPU_INDEX", "0")
GPU_CLASS = _env("GPU_CLASS", "a100_40")
SIMFORGE_ENV = _env("SIMFORGE_ENV", "dev")
WORKER_ID = _env("WORKER_ID", f"{POOL}-{SIMFORGE_ENV}-gpu{GPU_INDEX}-{uuid.uuid4().hex[:6]}")
WORKER_VERSION = _env("WORKER_VERSION", "cp-http-20260417")
CONTROL_PLANE_URL = _env("CONTROL_PLANE_URL", required=True).rstrip("/")
INTERNAL_TOKEN = _env("CP_INTERNAL_WORKER_TOKEN", _env("CP_INTERNAL_ARTIFACTS_TOKEN", ""), required=True)
HEARTBEAT_INTERVAL_S = 10
CARLA_RPC_HOST = "127.0.0.1"
CARLA_RPC_PORT = int(_env("CARLA_RPC_PORT", "20467"))
CARLA_BOOT_TIMEOUT_S = 90
EVENT_BATCH_MAX = max(1, int(_env("CP_EVENT_BATCH_MAX", "64")))
HEALTHZ_PORT = int(_env("HEALTHZ_PORT", "8081"))


def _normalize_map_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.replace("\\", "/").split("/")[-1].strip()
    return normalized[:-5] if normalized.endswith(".xodr") else normalized


class WorkerState:
    def __init__(self) -> None:
        self.worker_id: str = WORKER_ID
        self.state: str = "starting"
        self.job_id: str | None = None
        self.current_map: str | None = None
        self.last_heartbeat_ts: float = 0.0
        self.ready: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "pool": POOL,
            "gpu_class": GPU_CLASS,
            "gpu_index": GPU_INDEX,
            "state": self.state,
            "job_id": self.job_id,
            "current_map": self.current_map,
            "last_heartbeat": self.last_heartbeat_ts,
            "ready": self.ready,
            "ts": _now_iso(),
        }


STATE = WorkerState()


def _refresh_current_map_name(timeout: float = 5.0) -> str | None:
    try:
        import carla

        client = carla.Client(CARLA_RPC_HOST, CARLA_RPC_PORT)
        client.set_timeout(timeout)
        current_map = client.get_world().get_map().name
        normalized = _normalize_map_name(current_map)
        STATE.current_map = normalized or None
        return STATE.current_map
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to refresh current map name: %s", exc)
        return STATE.current_map


def _headers() -> dict[str, str]:
    return {
        "X-Internal-Token": INTERNAL_TOKEN,
        "Content-Type": "application/json",
    }


async def _post_json(
    session: aiohttp.ClientSession,
    path: str,
    body: dict[str, Any],
    *,
    ok_statuses: tuple[int, ...] = (200,),
) -> tuple[int, dict[str, Any]]:
    url = f"{CONTROL_PLANE_URL}{path}"
    async with session.post(url, headers=_headers(), json=body) as resp:
        status = resp.status
        text = await resp.text()
        if status not in ok_statuses:
            raise RuntimeError(f"POST {path} failed: {status} {text[:300]}")
        if not text:
            return status, {}
        try:
            return status, json.loads(text)
        except json.JSONDecodeError:
            return status, {}


async def _heartbeat(session: aiohttp.ClientSession) -> list[str]:
    if STATE.current_map is None and STATE.ready:
        _refresh_current_map_name(timeout=5.0)
    STATE.last_heartbeat_ts = time.time()
    _, payload = await _post_json(
        session,
        "/cp/internal/workers/heartbeat",
        {
            "worker_id": WORKER_ID,
            "pool": POOL,
            "gpu_id": GPU_INDEX,
            "gpu_class": GPU_CLASS,
            "status": STATE.state,
            "simforge_env": SIMFORGE_ENV,
            "current_job_id": STATE.job_id,
            "hostname": os.environ.get("HOSTNAME", "gpu-host"),
            "worker_version": WORKER_VERSION,
            "metadata": {
                "transport": "cp-http",
                "current_map": STATE.current_map,
            },
        },
    )
    return list(payload.get("cancel_requested_job_ids", []))


async def _claim_job(session: aiohttp.ClientSession) -> dict[str, Any] | None:
    status, payload = await _post_json(
        session,
        "/cp/internal/jobs/claim",
        {
            "worker_id": WORKER_ID,
            "pool": POOL,
            "gpu_id": GPU_INDEX,
            "gpu_class": GPU_CLASS,
            "simforge_env": SIMFORGE_ENV,
            "hostname": os.environ.get("HOSTNAME", "gpu-host"),
            "worker_version": WORKER_VERSION,
            "metadata": {
                "transport": "cp-http",
                "current_map": STATE.current_map,
            },
        },
        ok_statuses=(200, 204),
    )
    if status == 204:
        return None
    return payload


async def _append_event(session: aiohttp.ClientSession, job_id: str, payload: dict[str, Any]) -> None:
    await _append_events(session, job_id, [payload])


async def _append_events(
    session: aiohttp.ClientSession,
    job_id: str,
    payloads: list[dict[str, Any]],
) -> None:
    if not payloads:
        return
    await _post_json(
        session,
        f"/cp/internal/jobs/{job_id}/events",
        {
            "worker_id": WORKER_ID,
            "events": [
                {"type": str(payload.get("type") or "update"), "payload": dict(payload)}
                for payload in payloads
            ],
        },
    )


async def _complete_job(
    session: aiohttp.ClientSession,
    job_id: str,
    *,
    artifacts: list[dict[str, Any]],
    simulation_id: str | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> None:
    await _post_json(
        session,
        f"/cp/internal/jobs/{job_id}/complete",
        {
            "worker_id": WORKER_ID,
            "artifacts": artifacts,
            "simulation_id": simulation_id,
            "execution_metadata": execution_metadata or {},
        },
    )


async def _fail_job(
    session: aiohttp.ClientSession,
    job_id: str,
    *,
    error: str,
    execution_metadata: dict[str, Any] | None = None,
) -> None:
    await _post_json(
        session,
        f"/cp/internal/jobs/{job_id}/fail",
        {
            "worker_id": WORKER_ID,
            "error": error,
            "execution_metadata": execution_metadata or {},
        },
    )


def _launch_carla() -> subprocess.Popen[bytes]:
    binary = _env(
        "CARLA_SERVER_BIN",
        "/workspace/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping",
    )
    if not Path(binary).exists():
        raise RuntimeError(f"CARLA server binary not found at {binary}")
    args = [
        binary,
        "-RenderOffScreen",
        "-vulkan",
        "-nosound",
        f"-carla-rpc-port={CARLA_RPC_PORT}",
        f"-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={GPU_INDEX}",
    ]
    log.info("Launching CARLA: %s", " ".join(args))
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


async def _wait_for_carla_rpc(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2.0)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            log.info("CARLA RPC accepting connections on %s:%d", host, port)
            return
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(1.0)
    raise RuntimeError(f"CARLA RPC did not accept connections on {host}:{port} within {timeout}s")


def _run_simulation_blocking(
    spec: dict[str, Any],
    out_queue: "queue.Queue[Any]",
    stop_event: threading.Event,
    pause_event: threading.Event,
) -> dict[str, Any]:
    from .carla_runner.simulation_service import _simulation_worker

    request_payload = spec.get("request") or spec
    settings = spec.get("settings") or {
        "output_root": os.environ.get("ORCH_JOBS_ROOT", "/tmp/runs"),
        "carla_host": CARLA_RPC_HOST,
        "carla_port": CARLA_RPC_PORT,
        "tm_port": int(os.environ.get("CARLA_TM_PORT", "8000")),
        "carla_timeout": float(os.environ.get("ORCH_CARLA_TIMEOUT", "20")),
    }
    output_root = Path(settings["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    _simulation_worker(request_payload, settings, out_queue, stop_event, pause_event)
    run_dirs = [candidate for candidate in output_root.iterdir() if candidate.is_dir()]
    latest_run_dir = max(run_dirs, key=lambda candidate: candidate.stat().st_mtime, default=None)
    return {
        "output_root": str(output_root),
        "run_id": latest_run_dir.name if latest_run_dir is not None else None,
        "run_dir": str(latest_run_dir) if latest_run_dir is not None else None,
    }


async def _drain_queue(
    session: aiohttp.ClientSession,
    job_id: str,
    out_queue: "queue.Queue[Any]",
    done: threading.Event,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    last: dict[str, Any] = {}
    while True:
        try:
            first_msg = await loop.run_in_executor(None, out_queue.get, True, 0.5)
        except queue.Empty:
            if done.is_set() and out_queue.empty():
                return last
            continue

        batch = [first_msg]
        while len(batch) < EVENT_BATCH_MAX:
            try:
                batch.append(out_queue.get_nowait())
            except queue.Empty:
                break

        payloads: list[dict[str, Any]] = []
        for msg in batch:
            if hasattr(msg, "model_dump"):
                payload = msg.model_dump(mode="json")
            elif isinstance(msg, dict):
                payload = msg
            else:
                payload = {"type": "log", "message": str(msg)}
            payloads.append(payload)
            if payload.get("type") in {"completed", "failed", "cancelled", "terminal"}:
                last = payload

        await _append_events(session, job_id, payloads)
        if done.is_set() and out_queue.empty():
            return last


async def _maybe_upload_artifacts(
    job_id: str,
    spec: dict[str, Any],
    result: dict[str, Any],
    last_event: dict[str, Any],
) -> list[dict[str, Any]]:
    request_payload = spec.get("request") or spec
    try:
        from .carla_runner.models import SimulationRunRequest

        request = SimulationRunRequest.model_validate(request_payload)
        if not request.topdown_recording and not request.sensors and not request.gt_sensors:
            return []
    except (ValidationError, TypeError):
        log.warning("artifact spec validation failed for %s", job_id, exc_info=True)
        request = None

    bucket = os.environ.get("ORCH_STORAGE_BUCKET") or os.environ.get("S3_BUCKET")
    if not bucket:
        return []
    output_root = result.get("output_root") or os.environ.get("ORCH_JOBS_ROOT", "/tmp/runs")
    run_id = last_event.get("run_id") or result.get("run_id") or spec.get("run_id")
    run_dir = Path(result.get("run_dir") or (Path(output_root) / run_id if run_id else output_root))
    try:
        from .artifact_storage import S3ArtifactStorage
        from .config import Settings
        from .models import JobArtifacts, JobRecord, JobState

        settings = Settings.load()
        storage = S3ArtifactStorage(settings)
        now = datetime.now(timezone.utc)
        output_root_path = Path(output_root)
        request_file = output_root_path / f"{job_id}.request.json"
        runtime_file = output_root_path / f"{job_id}.runtime.json"
        request_file.parent.mkdir(parents=True, exist_ok=True)
        request_file.write_text(json.dumps(request_payload), encoding="utf-8")
        runtime_file.write_text(json.dumps(spec.get("settings") or {}), encoding="utf-8")
        manifest_path = run_dir / "manifest.json"
        recording_path = run_dir / "recording.mp4"
        scenario_log_path = run_dir / "scenario.log"
        debug_log_path = run_dir / "debug.log"
        job = JobRecord(
            job_id=job_id,
            state=JobState.succeeded,
            created_at=now,
            updated_at=now,
            request=request,
            artifacts=JobArtifacts(
                output_dir=str(output_root_path),
                request_file=str(request_file),
                runtime_settings_file=str(runtime_file),
                manifest_path=str(manifest_path) if manifest_path.exists() else None,
                recording_path=str(recording_path) if recording_path.exists() else None,
                scenario_log_path=str(scenario_log_path) if scenario_log_path.exists() else None,
                debug_log_path=str(debug_log_path) if debug_log_path.exists() else None,
            ),
            run_id=run_id,
        )
        uploaded = await asyncio.to_thread(storage.upload_job_artifacts, job)
        return [item.model_dump(mode="json") for item in uploaded]
    except (ClientError, OSError, asyncio.TimeoutError, ValueError) as exc:
        log.exception("artifact upload failed for %s", job_id)
        return []


async def _run_one(session: aiohttp.ClientSession, claim: dict[str, Any]) -> None:
    job_id = claim.get("id") or "unknown-job"
    spec = claim.get("request_payload") or {}
    simulation_id = claim.get("simulation_id")

    STATE.state = "busy"
    STATE.job_id = job_id

    out_queue: "queue.Queue[Any]" = queue.Queue()
    stop_event = threading.Event()
    pause_event = threading.Event()
    done = threading.Event()
    result_box: dict[str, Any] = {}
    err_box: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            result_box["result"] = _run_simulation_blocking(spec, out_queue, stop_event, pause_event)
        except BaseException as exc:
            err_box["err"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_run, name=f"sim-{job_id}", daemon=True)
    thread.start()
    last = await _drain_queue(session, job_id, out_queue, done)
    await asyncio.get_running_loop().run_in_executor(None, thread.join)

    try:
        if "err" in err_box:
            await _fail_job(session, job_id, error=repr(err_box["err"]))
        else:
            artifacts = await _maybe_upload_artifacts(job_id, spec, result_box.get("result") or {}, last)
            await _complete_job(
                session,
                job_id,
                artifacts=artifacts,
                simulation_id=simulation_id,
                execution_metadata={"final_state": last.get("state") or "succeeded"},
            )
    finally:
        requested_map = _normalize_map_name(str(spec.get("map_name") or ""))
        if requested_map:
            STATE.current_map = requested_map
        _refresh_current_map_name(timeout=2.0)
        STATE.state = "idle"
        STATE.job_id = None


async def heartbeat_loop(session: aiohttp.ClientSession, shutdown: asyncio.Event) -> None:
    while not shutdown.is_set():
        try:
            cancel_ids = await _heartbeat(session)
            if STATE.job_id and STATE.job_id in cancel_ids:
                log.warning("cancel requested for %s but worker-side cancellation is not wired yet", STATE.job_id)
        except Exception as exc:
            log.warning("heartbeat failed: %s", exc)
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=HEARTBEAT_INTERVAL_S)
        except asyncio.TimeoutError:
            continue


async def _async_main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    carla_proc: subprocess.Popen[bytes] | None = None
    if os.environ.get("SKIP_CARLA_BOOT", "").lower() not in {"1", "true", "yes"}:
        carla_proc = _launch_carla()
        try:
            await _wait_for_carla_rpc(CARLA_RPC_HOST, CARLA_RPC_PORT, CARLA_BOOT_TIMEOUT_S)
        except Exception as exc:
            log.error("CARLA boot failed: %s", exc)
            if carla_proc and carla_proc.poll() is None:
                carla_proc.terminate()
            return 2
        _refresh_current_map_name()

    timeout = aiohttp.ClientTimeout(total=120, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _on_sigterm() -> None:
            STATE.state = "draining"
            shutdown.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _on_sigterm)
            except NotImplementedError:
                pass

        STATE.state = "idle"
        STATE.ready = True

        hb_task = asyncio.create_task(heartbeat_loop(session, shutdown))

        from .healthz import start_healthz_server
        healthz_runner = await start_healthz_server(HEALTHZ_PORT, STATE)

        try:
            while not shutdown.is_set():
                if carla_proc is not None and carla_proc.poll() is not None:
                    log.error("CARLA subprocess exited with code %s; shutting worker down", carla_proc.returncode)
                    shutdown.set()
                    break
                try:
                    claim = await _claim_job(session)
                except Exception as exc:
                    log.warning("claim failed: %s", exc)
                    await asyncio.sleep(2.0)
                    continue
                if claim is None:
                    await asyncio.sleep(2.0)
                    continue
                await _run_one(session, claim)
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except (asyncio.CancelledError, Exception):
                pass
            if healthz_runner is not None:
                await healthz_runner.cleanup()
            if carla_proc is not None and carla_proc.poll() is None:
                carla_proc.terminate()
                try:
                    carla_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    carla_proc.kill()
    return 0


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
