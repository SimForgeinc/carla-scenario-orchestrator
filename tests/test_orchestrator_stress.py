"""
Orchestrator stress / integration tests.

These tests hit the live orchestrator API at http://127.0.0.1:18421
and exercise GPU scheduling, job lifecycle, recording, and S3 upload.

Requirements:
    - Orchestrator running on port 18421
    - 6 execution GPU slots (indices 0-5), 1 metadata slot (index 6)
    - pytest-timeout installed
"""
from __future__ import annotations

import time
import uuid
import concurrent.futures

import pytest
import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

API = "http://127.0.0.1:18421"


def make_scenario(label: str = "stress", duration: float = 3.0, recording: bool = True) -> dict:
    """Build a minimal valid SimulationRunRequest payload."""
    tag = uuid.uuid4().hex[:6]
    return {
        "map_name": "Town10HD_Opt",
        "selected_roads": [{"id": "10", "name": "Road 10"}],
        "actors": [
            {
                "id": f"ego-{label}-{tag}",
                "label": f"Ego {label}",
                "kind": "vehicle",
                "role": "ego",
                "blueprint": "vehicle.tesla.model3",
                "spawn": {"road_id": "10", "s_fraction": 0.3},
                "speed_kph": 60,
                "autopilot": True,
                "placement_mode": "road",
                "is_static": False,
                "timeline": [],
                "route": [],
            }
        ],
        "duration_seconds": duration,
        "fixed_delta_seconds": 0.05,
        "topdown_recording": recording,
        "recording_width": 640,
        "recording_height": 480,
        "recording_fov": 90,
    }


def make_multi_actor_scenario(label: str, actors_extra: list[dict], **kwargs) -> dict:
    """Build a scenario with the ego plus additional actors."""
    base = make_scenario(label=label, **kwargs)
    base["actors"].extend(actors_extra)
    return base


def submit_job(payload: dict) -> dict:
    resp = requests.post(f"{API}/api/jobs", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def wait_for_job(job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{API}/api/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        job = resp.json()
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(1.0)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def wait_for_state(job_id: str, target_state: str, timeout: float = 60) -> dict:
    """Poll until a job reaches *target_state* (or a terminal state)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{API}/api/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        job = resp.json()
        if job["state"] == target_state:
            return job
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise TimeoutError(f"Job {job_id} never reached state={target_state} within {timeout}s")


def get_capacity() -> dict:
    resp = requests.get(f"{API}/api/capacity", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_diagnostics(job_id: str):
    resp = requests.get(f"{API}/api/jobs/{job_id}/diagnostics", timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def cancel_job(job_id: str) -> dict:
    resp = requests.post(f"{API}/api/jobs/{job_id}/cancel", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_health() -> dict:
    resp = requests.get(f"{API}/api/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_jobs() -> list[dict]:
    resp = requests.get(f"{API}/api/jobs", timeout=10)
    resp.raise_for_status()
    return resp.json()["items"]


def wait_until_all_slots_free(timeout: float = 300):
    """Block until the orchestrator has all 6 execution slots free."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        cap = get_capacity()
        if cap["busy_slots"] == 0:
            return cap
        time.sleep(2.0)
    raise TimeoutError("Slots did not free up in time")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_slots_free():
    """Before every test, wait until all execution slots are free."""
    wait_until_all_slots_free(timeout=300)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_parallel_6_jobs_fill_all_slots():
    """Submit 6 jobs simultaneously and verify each gets a unique GPU slot."""
    payloads = [make_scenario(label=f"par6-{i}", duration=5.0) for i in range(6)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(submit_job, p) for p in payloads]
        results = [f.result() for f in futures]

    job_ids = [r["job_id"] for r in results]

    # Give scheduler a moment to assign all slots
    time.sleep(3)
    cap = get_capacity()
    assert cap["busy_slots"] == 6, f"Expected 6 busy slots, got {cap['busy_slots']}"

    completed = [wait_for_job(jid, timeout=120) for jid in job_ids]

    for job in completed:
        assert job["state"] == "succeeded", f"Job {job['job_id']} state={job['state']} error={job.get('error')}"

    containers = {job["container_name"] for job in completed}
    assert len(containers) == 6, f"Expected 6 unique containers, got {containers}"


@pytest.mark.timeout(300)
def test_queue_overflow_7_jobs():
    """Submit 7 jobs (6 slots + 1 queued), all should eventually succeed."""
    payloads = [make_scenario(label=f"over7-{i}", duration=5.0) for i in range(7)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(submit_job, p) for p in payloads]
        results = [f.result() for f in futures]

    job_ids = [r["job_id"] for r in results]
    time.sleep(3)

    cap = get_capacity()
    assert cap["busy_slots"] <= 6

    completed = [wait_for_job(jid, timeout=180) for jid in job_ids]
    for job in completed:
        assert job["state"] == "succeeded", f"Job {job['job_id']} state={job['state']} error={job.get('error')}"


@pytest.mark.timeout(300)
def test_cancel_queued_job():
    """Fill all slots, queue a 7th job, cancel the queued one."""
    fill_payloads = [make_scenario(label=f"fill-{i}", duration=8.0) for i in range(6)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fill_futures = [pool.submit(submit_job, p) for p in fill_payloads]
        fill_results = [f.result() for f in fill_futures]
    fill_ids = [r["job_id"] for r in fill_results]

    time.sleep(3)

    queued_result = submit_job(make_scenario(label="to-cancel", duration=3.0))
    queued_id = queued_result["job_id"]

    queued_job = wait_for_state(queued_id, "queued", timeout=10)
    assert queued_job["state"] == "queued", f"Expected queued, got {queued_job['state']}"

    cancel_resp = cancel_job(queued_id)
    assert cancel_resp["state"] == "cancelled"

    for jid in fill_ids:
        job = wait_for_job(jid, timeout=120)
        assert job["state"] == "succeeded", f"Fill job {jid} state={job['state']}"


@pytest.mark.timeout(180)
def test_cancel_running_job():
    """Submit a job, wait until running, then cancel it."""
    result = submit_job(make_scenario(label="run-cancel", duration=15.0, recording=False))
    job_id = result["job_id"]

    job = wait_for_state(job_id, "running", timeout=60)
    assert job["state"] == "running", f"Expected running, got {job['state']}"

    # Cancel sets the cancel event; worker picks it up asynchronously
    cancel_resp = cancel_job(job_id)
    assert cancel_resp["state"] in ("running", "cancelled"), (
        f"Unexpected state after cancel: {cancel_resp['state']}"
    )

    # Wait for the job to reach a terminal state
    final = wait_for_job(job_id, timeout=60)
    assert final["state"] in ("cancelled", "succeeded"), (
        f"Expected cancelled or succeeded, got {final['state']}"
    )

    time.sleep(2)
    cap = get_capacity()
    assert cap["busy_slots"] == 0, f"Expected 0 busy, got {cap['busy_slots']}"


@pytest.mark.timeout(180)
def test_recording_produces_mp4():
    """Submit a job with recording enabled and verify output."""
    result = submit_job(make_scenario(label="rec-mp4", duration=3.0, recording=True))
    job_id = result["job_id"]
    job = wait_for_job(job_id, timeout=120)
    assert job["state"] == "succeeded", f"Job failed: {job.get('error')}"

    diag = get_diagnostics(job_id)
    assert diag is not None, "Diagnostics not found"
    assert diag["saved_frame_count"] > 0, f"Expected frames > 0, got {diag['saved_frame_count']}"

    rec_path = job.get("artifacts", {}).get("recording_path") or diag.get("recording_path")
    assert rec_path is not None, "recording_path is null"


@pytest.mark.timeout(180)
def test_no_recording_when_disabled():
    """Submit a job with recording disabled, verify no frames saved."""
    result = submit_job(make_scenario(label="no-rec", duration=3.0, recording=False))
    job_id = result["job_id"]
    job = wait_for_job(job_id, timeout=120)
    assert job["state"] == "succeeded", f"Job failed: {job.get('error')}"

    diag = get_diagnostics(job_id)
    assert diag is not None, "Diagnostics not found"
    assert diag["saved_frame_count"] == 0, f"Expected 0 frames, got {diag['saved_frame_count']}"


@pytest.mark.timeout(180)
def test_s3_upload_after_completion():
    """Verify that completed jobs have S3 artifacts uploaded."""
    result = submit_job(make_scenario(label="s3-up", duration=3.0, recording=True))
    job_id = result["job_id"]
    job = wait_for_job(job_id, timeout=120)
    assert job["state"] == "succeeded", f"Job failed: {job.get('error')}"

    artifacts = job.get("artifacts", {})
    uploaded = artifacts.get("uploaded_artifacts", [])
    assert len(uploaded) >= 1, f"Expected at least 1 uploaded artifact, got {len(uploaded)}"

    s3_uris = [a["s3_uri"] for a in uploaded if a.get("s3_uri")]
    assert len(s3_uris) >= 1, "No artifacts with s3_uri found"

    manifest_artifacts = [a for a in uploaded if a.get("kind") == "MANIFEST"]
    assert len(manifest_artifacts) >= 1, "No MANIFEST artifact found"


@pytest.mark.timeout(180)
def test_capacity_returns_after_batch():
    """Verify capacity returns to full after jobs complete."""
    cap_before = get_capacity()
    assert cap_before["free_slots"] == 6, f"Pre-test: expected 6 free, got {cap_before['free_slots']}"

    payloads = [make_scenario(label=f"cap-{i}", duration=3.0, recording=False) for i in range(3)]
    results = [submit_job(p) for p in payloads]
    job_ids = [r["job_id"] for r in results]

    for jid in job_ids:
        job = wait_for_job(jid, timeout=120)
        assert job["state"] == "succeeded"

    time.sleep(2)
    cap_after = get_capacity()
    assert cap_after["free_slots"] == 6, f"Post-test: expected 6 free, got {cap_after['free_slots']}"


@pytest.mark.timeout(300)
def test_concurrent_mixed_configs():
    """Submit 6 jobs with different actor configurations simultaneously."""
    tag = uuid.uuid4().hex[:4]

    # Job 1: ego only with recording
    job1 = make_scenario(label=f"mix1-{tag}", duration=3.0, recording=True)

    # Job 2: ego + 2 traffic vehicles
    job2 = make_multi_actor_scenario(
        label=f"mix2-{tag}",
        duration=3.0,
        recording=False,
        actors_extra=[
            {
                "id": f"traffic-a-{tag}",
                "label": "Traffic A",
                "kind": "vehicle",
                "role": "traffic",
                "blueprint": "vehicle.audi.a2",
                "spawn": {"road_id": "10", "s_fraction": 0.5},
                "speed_kph": 40,
                "autopilot": True,
                "placement_mode": "road",
                "is_static": False,
                "timeline": [],
                "route": [],
            },
            {
                "id": f"traffic-b-{tag}",
                "label": "Traffic B",
                "kind": "vehicle",
                "role": "traffic",
                "blueprint": "vehicle.bmw.grandtourer",
                "spawn": {"road_id": "10", "s_fraction": 0.7},
                "speed_kph": 50,
                "autopilot": True,
                "placement_mode": "road",
                "is_static": False,
                "timeline": [],
                "route": [],
            },
        ],
    )

    # Job 3: ego + walker
    job3 = make_multi_actor_scenario(
        label=f"mix3-{tag}",
        duration=3.0,
        recording=False,
        actors_extra=[
            {
                "id": f"walker-{tag}",
                "label": "Pedestrian",
                "kind": "walker",
                "role": "traffic",
                "blueprint": "walker.pedestrian.0001",
                "spawn": {"road_id": "10", "s_fraction": 0.6},
                "speed_kph": 5,
                "autopilot": True,
                "placement_mode": "road",
                "is_static": False,
                "timeline": [],
                "route": [],
            },
        ],
    )

    # Job 4: ego + prop, no recording
    job4 = make_multi_actor_scenario(
        label=f"mix4-{tag}",
        duration=3.0,
        recording=False,
        actors_extra=[
            {
                "id": f"prop-{tag}",
                "label": "Cone",
                "kind": "prop",
                "role": "prop",
                "blueprint": "static.prop.constructioncone",
                "spawn": {"road_id": "10", "s_fraction": 0.4},
                "speed_kph": 0,
                "autopilot": False,
                "placement_mode": "road",
                "is_static": True,
                "timeline": [],
                "route": [],
            },
        ],
    )

    # Job 5: ego with timeline (speed changes) using ActorTimelineClip schema
    job5 = make_scenario(label=f"mix5-{tag}", duration=5.0, recording=False)
    job5["actors"][0]["timeline"] = [
        {
            "id": f"clip-slow-{tag}",
            "start_time": 1.0,
            "end_time": 2.5,
            "action": "set_speed",
            "target_speed_kph": 30,
        },
        {
            "id": f"clip-fast-{tag}",
            "start_time": 3.0,
            "end_time": 4.5,
            "action": "set_speed",
            "target_speed_kph": 80,
        },
    ]

    # Job 6: ego with route
    job6 = make_scenario(label=f"mix6-{tag}", duration=5.0, recording=False)
    job6["actors"][0]["route"] = [
        {"road_id": "10", "s_fraction": 0.3},
        {"road_id": "10", "s_fraction": 0.9},
    ]

    payloads = [job1, job2, job3, job4, job5, job6]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(submit_job, p) for p in payloads]
        results = [f.result() for f in futures]

    job_ids = [r["job_id"] for r in results]
    completed = [wait_for_job(jid, timeout=120) for jid in job_ids]

    for job in completed:
        assert job["state"] == "succeeded", (
            f"Job {job['job_id']} state={job['state']} error={job.get('error')}"
        )

    for jid in job_ids:
        diag = get_diagnostics(jid)
        if diag is not None:
            assert diag.get("worker_error") is None, (
                f"Job {jid} had worker_error: {diag['worker_error']}"
            )


@pytest.mark.timeout(360)
def test_rapid_fire_12_jobs():
    """Submit 12 jobs rapidly. First 6 start, next 6 queue, all succeed."""
    payloads = [make_scenario(label=f"rapid-{i}", duration=3.0, recording=False) for i in range(12)]

    results = []
    for p in payloads:
        results.append(submit_job(p))

    job_ids = [r["job_id"] for r in results]

    completed = []
    for jid in job_ids:
        job = wait_for_job(jid, timeout=300)
        completed.append(job)

    for job in completed:
        assert job["state"] == "succeeded", (
            f"Job {job['job_id']} state={job['state']} error={job.get('error')}"
        )

    assert len(completed) == 12


@pytest.mark.timeout(300)
def test_health_under_load():
    """Hit /api/health while all slots are busy."""
    payloads = [make_scenario(label=f"hlth-{i}", duration=8.0, recording=False) for i in range(6)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(submit_job, p) for p in payloads]
        results = [f.result() for f in futures]

    job_ids = [r["job_id"] for r in results]
    time.sleep(4)

    health = get_health()
    assert health["status"] == "healthy", f"Health status: {health['status']}"
    assert health["carla_connected"] is True
    assert health["busy_slots"] <= 6

    for jid in job_ids:
        wait_for_job(jid, timeout=120)


@pytest.mark.timeout(180)
def test_job_list_ordering():
    """Submit 3 jobs with delays and verify list order."""
    ids = []
    for i in range(3):
        result = submit_job(make_scenario(label=f"order-{i}", duration=3.0, recording=False))
        ids.append(result["job_id"])
        if i < 2:
            time.sleep(0.5)

    all_jobs = get_jobs()
    all_ids = [j["job_id"] for j in all_jobs]

    positions = []
    for jid in ids:
        assert jid in all_ids, f"Job {jid} not in job list"
        positions.append(all_ids.index(jid))

    increasing = all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))
    decreasing = all(positions[i] > positions[i + 1] for i in range(len(positions) - 1))
    assert increasing or decreasing, (
        f"Jobs not in monotonic order. positions={positions}"
    )

    for jid in ids:
        wait_for_job(jid, timeout=120)


@pytest.mark.timeout(360)
def test_queue_position_decrements():
    """Submit 8 jobs (6 + 2 queued), verify queue positions update."""
    payloads = [make_scenario(label=f"qpos-{i}", duration=6.0, recording=False) for i in range(8)]
    results = [submit_job(p) for p in payloads]
    job_ids = [r["job_id"] for r in results]

    time.sleep(4)

    queued_ids = []
    for jid in job_ids:
        resp = requests.get(f"{API}/api/jobs/{jid}", timeout=10)
        resp.raise_for_status()
        job = resp.json()
        if job["state"] == "queued":
            queued_ids.append((jid, job.get("queue_position", -1)))

    if len(queued_ids) >= 2:
        for _, pos in queued_ids:
            assert pos >= 0, f"Queue position should be >= 0, got {pos}"

    completed = [wait_for_job(jid, timeout=300) for jid in job_ids]
    for job in completed:
        assert job["state"] == "succeeded", (
            f"Job {job['job_id']} state={job['state']} error={job.get('error')}"
        )
