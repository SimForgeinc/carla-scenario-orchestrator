"""Error-resilience tests for the CARLA Scenario Orchestrator.

Validates that invalid/edge-case payloads are handled gracefully
without crashing the service. Covers Pydantic validation (422),
runtime error handling, and service stability after errors.

Run with:
    .venv/bin/python -m pytest tests/test_error_resilience.py -v --tb=short
"""
from __future__ import annotations

import time
import uuid

import pytest
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API = "http://127.0.0.1:18421"
MAP = "Town10HD_Opt"
ROAD = "10"
VEHICLE_BP = "vehicle.tesla.model3"
DEFAULT_DURATION = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_payload(**overrides) -> dict:
    """Return a minimal valid SimulationRunRequest payload with overrides."""
    payload = {
        "map_name": MAP,
        "selected_roads": [{"id": ROAD, "name": "Road 10"}],
        "actors": [
            {
                "id": str(uuid.uuid4()),
                "label": "ego",
                "kind": "vehicle",
                "role": "ego",
                "blueprint": VEHICLE_BP,
                "spawn": {"road_id": ROAD, "s_fraction": 0.3},
                "speed_kph": 60.0,
            }
        ],
        "duration_seconds": DEFAULT_DURATION,
    }
    payload.update(overrides)
    return payload


def _submit(payload: dict, expect_status: int | None = None) -> requests.Response:
    """POST /api/jobs and optionally assert status code."""
    resp = requests.post(f"{API}/api/jobs", json=payload, timeout=10)
    if expect_status is not None:
        assert resp.status_code == expect_status, (
            f"Expected {expect_status}, got {resp.status_code}: {resp.text[:300]}"
        )
    return resp


def _submit_and_wait(payload: dict, timeout: float = 60) -> dict:
    """Submit a job and poll until terminal state. Returns job record."""
    resp = requests.post(f"{API}/api/jobs", json=payload, timeout=10)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        time.sleep(1.0)
        r = requests.get(f"{API}/api/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        job = r.json()
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
    pytest.fail(
        f"Job {job_id} timed out after {timeout}s "
        f"(last state: {job.get('state', '?')})"
    )


def _health_ok() -> bool:
    r = requests.get(f"{API}/api/health", timeout=5)
    return r.status_code == 200 and r.json().get("ok", False)


def _get_capacity() -> dict:
    r = requests.get(f"{API}/api/capacity", timeout=5)
    r.raise_for_status()
    return r.json()


# ===========================================================================
# SECTION 1: Pydantic validation tests (should return 422)
# ===========================================================================

class TestPydanticValidation:
    """Payloads that violate model constraints should be rejected with 422."""

    def test_missing_map_name(self):
        """1. POST /api/jobs without map_name -> 422."""
        payload = _base_payload()
        del payload["map_name"]
        _submit(payload, expect_status=422)

    def test_invalid_actor_kind(self):
        """2. kind='helicopter' is not in Literal['vehicle','walker','prop'] -> 422."""
        payload = _base_payload()
        payload["actors"][0]["kind"] = "helicopter"
        _submit(payload, expect_status=422)

    def test_negative_duration(self):
        """3. duration_seconds=-5 violates ge=1.0 -> 422."""
        payload = _base_payload(duration_seconds=-5)
        _submit(payload, expect_status=422)

    def test_duration_too_long(self):
        """4. duration_seconds=9999 violates le=TIMELINE_DURATION_LIMIT_SECONDS -> 422."""
        payload = _base_payload(duration_seconds=9999)
        _submit(payload, expect_status=422)

    def test_invalid_recording_width(self):
        """5. recording_width=0 violates ge=320 -> 422."""
        payload = _base_payload(recording_width=0)
        _submit(payload, expect_status=422)

    def test_empty_actors_accepted(self):
        """6. actors=[] should be accepted (not all scenarios need actors)."""
        payload = _base_payload(actors=[])
        resp = _submit(payload)
        assert resp.status_code in (200, 201), (
            f"Empty actors should be accepted, got {resp.status_code}: {resp.text[:300]}"
        )

    def test_empty_blueprint_accepted(self):
        """7. blueprint='' should be accepted at API level (fails at spawn)."""
        payload = _base_payload()
        payload["actors"][0]["blueprint"] = ""
        resp = _submit(payload)
        # Accepted by Pydantic (no min_length on blueprint) -> 200/201
        # OR rejected if there is a validator -> 422
        # Either is fine; it must NOT be a 500
        assert resp.status_code != 500, (
            f"Empty blueprint caused a server crash: {resp.text[:300]}"
        )

    def test_speed_out_of_range(self):
        """8. speed_kph=500 violates le=240 -> 422."""
        payload = _base_payload()
        payload["actors"][0]["speed_kph"] = 500
        _submit(payload, expect_status=422)

    def test_s_fraction_out_of_range(self):
        """9. s_fraction=2.0 violates le=1.0 -> 422."""
        payload = _base_payload()
        payload["actors"][0]["spawn"]["s_fraction"] = 2.0
        _submit(payload, expect_status=422)

    def test_invalid_timeline_action(self):
        """10. action='fly_to_moon' is not a valid TimelineAction -> 422."""
        payload = _base_payload()
        payload["actors"][0]["timeline"] = [
            {
                "id": "clip1",
                "start_time": 0.0,
                "action": "fly_to_moon",
            }
        ]
        _submit(payload, expect_status=422)


# ===========================================================================
# SECTION 2: Runtime error handling
# ===========================================================================

class TestRuntimeErrors:
    """Payloads accepted by Pydantic but expected to fail during CARLA execution."""

    def test_nonexistent_road_id(self):
        """11. road_id='99999' -> job completes with worker_error or skipped_actors."""
        payload = _base_payload()
        payload["actors"][0]["spawn"]["road_id"] = "99999"
        job = _submit_and_wait(payload, timeout=90)
        # The job should not crash the service; it either succeeds with skipped actors
        # or fails with a meaningful error.
        assert job["state"] in ("succeeded", "failed"), (
            f"Unexpected state: {job['state']}"
        )
        if job["state"] == "failed":
            assert job.get("error"), "Failed job should have an error message"

    def test_nonexistent_blueprint(self):
        """12. blueprint='vehicle.does.not.exist' -> job fails with meaningful error."""
        payload = _base_payload()
        payload["actors"][0]["blueprint"] = "vehicle.does.not.exist"
        job = _submit_and_wait(payload, timeout=90)
        assert job["state"] in ("succeeded", "failed"), (
            f"Unexpected state: {job['state']}"
        )
        if job["state"] == "failed":
            assert job.get("error"), "Failed job should have an error message"

    def test_nonexistent_map(self):
        """13. map_name='TownDoesNotExist' -> job fails gracefully."""
        payload = _base_payload(map_name="TownDoesNotExist")
        job = _submit_and_wait(payload, timeout=90)
        assert job["state"] == "failed", (
            f"Non-existent map should fail, got {job['state']}"
        )
        assert job.get("error"), "Failed job should have an error message"

    def test_chase_actor_invalid_target(self):
        """14. chase_actor targeting nonexistent actor -> job completes without crash."""
        payload = _base_payload()
        payload["actors"][0]["timeline"] = [
            {
                "id": "chase-clip",
                "start_time": 0.0,
                "end_time": 2.0,
                "action": "chase_actor",
                "target_actor_id": "nonexistent_actor_xyz",
            }
        ]
        job = _submit_and_wait(payload, timeout=90)
        # Should not crash; either succeeds (target not found, vehicle brakes)
        # or fails gracefully
        assert job["state"] in ("succeeded", "failed"), (
            f"Unexpected state: {job['state']}"
        )

    def test_duplicate_actor_ids(self):
        """15. Two actors with the same id -> should not crash the service."""
        actor_id = "duplicate-id-test"
        payload = _base_payload()
        payload["actors"] = [
            {
                "id": actor_id,
                "label": "car_a",
                "kind": "vehicle",
                "role": "ego",
                "blueprint": VEHICLE_BP,
                "spawn": {"road_id": ROAD, "s_fraction": 0.2},
                "speed_kph": 40.0,
            },
            {
                "id": actor_id,
                "label": "car_b",
                "kind": "vehicle",
                "role": "traffic",
                "blueprint": VEHICLE_BP,
                "spawn": {"road_id": ROAD, "s_fraction": 0.8},
                "speed_kph": 40.0,
            },
        ]
        job = _submit_and_wait(payload, timeout=90)
        # Must not crash; any terminal state is acceptable
        assert job["state"] in ("succeeded", "failed"), (
            f"Unexpected state: {job['state']}"
        )


# ===========================================================================
# SECTION 3: Service stability after errors
# ===========================================================================

class TestServiceStability:
    """Ensure the service stays healthy after error scenarios."""

    def test_health_after_bad_payload(self):
        """16. Submit bad payload -> 422 -> /api/health still healthy."""
        payload = _base_payload(duration_seconds=-1)
        _submit(payload, expect_status=422)
        assert _health_ok(), "Service should remain healthy after a 422 rejection"

    def test_run_after_failed_run(self):
        """17. Submit bad runtime payload -> fails -> submit good payload -> succeeds."""
        # First: bad payload that passes validation but fails at runtime
        bad_payload = _base_payload(map_name="TownDoesNotExist")
        bad_job = _submit_and_wait(bad_payload, timeout=90)
        assert bad_job["state"] == "failed", "Bad map job should fail"

        # Then: good payload should succeed
        good_payload = _base_payload()
        good_job = _submit_and_wait(good_payload, timeout=90)
        assert good_job["state"] == "succeeded", (
            f"Good job after failed job should succeed, got {good_job['state']}: "
            f"{good_job.get('error', '')}"
        )

    def test_capacity_after_error(self):
        """18. Submit bad payload -> check capacity slots not leaked."""
        cap_before = _get_capacity()
        free_before = cap_before["free_slots"]

        # Submit a payload that will be rejected with 422
        bad_payload = _base_payload(duration_seconds=-1)
        _submit(bad_payload, expect_status=422)

        cap_after = _get_capacity()
        free_after = cap_after["free_slots"]
        assert free_after == free_before, (
            f"Capacity slot leaked after 422: "
            f"free_before={free_before}, free_after={free_after}"
        )
