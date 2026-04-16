"""Multi-map behavioral integration tests.

Tests vehicle spawning, lane changes, stops, and recordings across
multiple CARLA maps (Town01, Town03, Town10HD_Opt).

Each map is loaded once and all tests for that map run before loading the next.
Requires a live CARLA instance.
"""
from __future__ import annotations

import math
import time
import uuid
from typing import Any

import pytest
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API = "http://127.0.0.1:18421"
VEHICLE_BP = "vehicle.tesla.model3"
DEFAULT_DURATION = 4.0
DEFAULT_DELTA = 0.05


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _uid() -> str:
    return uuid.uuid4().hex[:8]


def load_map(map_name: str, timeout: float = 60) -> None:
    """Load a CARLA map, waiting until ready."""
    resp = requests.post(
        f"{API}/api/carla/map/load",
        json={"map_name": map_name},
        timeout=timeout,
    )
    resp.raise_for_status()
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = requests.get(f"{API}/api/carla/status", timeout=10).json()
        current = status.get("normalized_map_name") or ""
        if current == map_name or current.endswith(map_name):
            return
        time.sleep(1)
    pytest.fail(f"Map {map_name} did not load within {timeout}s")


def get_runtime_map() -> dict:
    """Get the runtime map data for the currently loaded map."""
    resp = requests.get(f"{API}/api/map/runtime", timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_multi_lane_road(runtime_map: dict) -> tuple[dict, int] | None:
    """Find a non-intersection road with driving_right >= 2 or driving_left >= 2.
    Returns (road, lane_id) where lane_id is the outermost driving lane in the
    direction that has >=2 lanes."""
    for road in runtime_map.get("road_summaries", []):
        if road.get("is_intersection"):
            continue
        for sec in road.get("section_summaries", []):
            dr = sec.get("driving_right", 0)
            dl = sec.get("driving_left", 0)
            if dr >= 2:
                # rightmost lane: -dr (outermost), lane_change_left goes to -(dr-1)
                return road, -dr
            if dl >= 2:
                return road, dl
    return None


def find_driving_road(runtime_map: dict) -> dict | None:
    """Find a non-intersection road with at least 1 driving lane."""
    for road in runtime_map.get("road_summaries", []):
        if road.get("is_intersection"):
            continue
        for sec in road.get("section_summaries", []):
            if sec.get("total_driving", 0) >= 1:
                return road
    return None


def submit_and_wait(payload: dict, timeout: float = 60) -> tuple[dict, dict]:
    """POST a job, poll until succeeded/failed, return (job_record, diagnostics)."""
    resp = requests.post(f"{API}/api/jobs", json=payload, timeout=10)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        r = requests.get(f"{API}/api/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        job = r.json()
        if job["state"] in ("succeeded", "failed"):
            diag = requests.get(
                f"{API}/api/jobs/{job_id}/diagnostics", timeout=10
            ).json()
            if job["state"] == "failed":
                error = (
                    job.get("error") or diag.get("worker_error") or "unknown"
                )
                pytest.fail(f"Job {job_id} failed: {error}")
            return job, diag
    pytest.fail(
        f"Job {job_id} timed out after {timeout}s (last state: {job.get('state', '?')})"
    )


def get_actor_states(events: list[dict], actor_label: str) -> list[dict]:
    """Extract per-frame states for a specific actor label."""
    states = []
    for ev in events:
        for a in ev.get("payload", {}).get("actors", []):
            if a.get("label") == actor_label:
                states.append(a)
    return states


def base_payload(
    map_name: str,
    road_id: str,
    *,
    actors: list[dict] | None = None,
    duration: float = DEFAULT_DURATION,
    recording: bool = False,
) -> dict:
    return {
        "map_name": map_name,
        "selected_roads": [{"id": road_id, "name": f"Road {road_id}"}],
        "actors": actors or [],
        "duration_seconds": duration,
        "fixed_delta_seconds": DEFAULT_DELTA,
        "topdown_recording": recording,
    }


def ego_actor(
    road_id: str,
    *,
    label: str = "Ego",
    autopilot: bool = True,
    speed_kph: float = 60.0,
    timeline: list[dict] | None = None,
    s_fraction: float = 0.3,
    lane_id: int | None = None,
) -> dict:
    return {
        "id": f"ego-{_uid()}",
        "label": label,
        "kind": "vehicle",
        "role": "ego",
        "blueprint": VEHICLE_BP,
        "spawn": {"road_id": road_id, "s_fraction": s_fraction, "lane_id": lane_id},
        "autopilot": autopilot,
        "is_static": False,
        "speed_kph": speed_kph,
        "placement_mode": "road",
        "timeline": timeline or [],
    }


# ---------------------------------------------------------------------------
# Map context - loaded once per map
# ---------------------------------------------------------------------------

class MapContext:
    def __init__(self, map_name: str):
        self.map_name = map_name
        self.runtime_map: dict = {}
        self.driving_road: dict | None = None
        self.multi_lane_road: dict | None = None
        self.multi_lane_id: int | None = None

    def setup(self) -> None:
        load_map(self.map_name)
        self.runtime_map = get_runtime_map()
        self.driving_road = find_driving_road(self.runtime_map)
        result = find_multi_lane_road(self.runtime_map)
        if result:
            self.multi_lane_road, self.multi_lane_id = result
        assert self.driving_road is not None, (
            f"No driving road found in {self.map_name}"
        )


_map_contexts: dict[str, MapContext] = {}


def _get_or_load_map(map_name: str) -> MapContext:
    if map_name not in _map_contexts:
        ctx = MapContext(map_name)
        ctx.setup()
        _map_contexts[map_name] = ctx
    return _map_contexts[map_name]


# ---------------------------------------------------------------------------
# Parametrized helper to generate 5 tests per map
# ---------------------------------------------------------------------------

def _run_vehicle_spawns_and_drives(map_name: str) -> None:
    ctx = _get_or_load_map(map_name)
    road = ctx.driving_road
    assert road is not None
    payload = base_payload(
        map_name, road["id"],
        actors=[ego_actor(road["id"], autopilot=True, speed_kph=80)],
    )
    job, diag = submit_and_wait(payload)
    states = get_actor_states(job.get("events", []), "Ego")
    speeds = [s.get("speed_mps", 0) for s in states]
    assert len(speeds) > 0, f"No frames recorded for Ego in {map_name}"
    assert max(speeds) > 0, f"Ego never moved in {map_name}, speeds: {speeds[:10]}"


def _run_lane_change(map_name: str) -> None:
    ctx = _get_or_load_map(map_name)
    road = ctx.multi_lane_road
    lane_id = ctx.multi_lane_id
    if road is None or lane_id is None:
        pytest.skip(f"No multi-lane road found in {map_name}")

    timeline = [
        {
            "id": f"lc-{_uid()}",
            "start_time": 1.5,
            "action": "lane_change_left",
            "enabled": True,
        }
    ]
    payload = base_payload(
        map_name, road["id"],
        actors=[
            ego_actor(
                road["id"],
                autopilot=True,
                speed_kph=60,
                timeline=timeline,
                lane_id=lane_id,
                s_fraction=0.15,
            )
        ],
        duration=6.0,
    )
    job, diag = submit_and_wait(payload, timeout=75)
    states = get_actor_states(job.get("events", []), "Ego")
    lane_ids = [s["lane_id"] for s in states if s.get("lane_id") is not None]
    assert len(lane_ids) >= 2, f"Not enough frames with lane_id in {map_name}"
    unique_lanes = set(lane_ids)
    assert len(unique_lanes) > 1, (
        f"Lane never changed in {map_name} (stayed {lane_ids[0]}), "
        f"road={road['id']}, lane_ids={lane_ids[:20]}"
    )


def _run_stop(map_name: str) -> None:
    ctx = _get_or_load_map(map_name)
    road = ctx.driving_road
    assert road is not None
    timeline = [
        {
            "id": f"speed-{_uid()}",
            "start_time": 0.5,
            "action": "set_speed",
            "target_speed_kph": 60,
            "enabled": True,
        },
        {
            "id": f"stop-{_uid()}",
            "start_time": 2.5,
            "action": "stop",
            "enabled": True,
        },
    ]
    payload = base_payload(
        map_name, road["id"],
        actors=[ego_actor(road["id"], autopilot=True, speed_kph=60, timeline=timeline)],
        duration=6.0,
    )
    job, diag = submit_and_wait(payload, timeout=75)
    states = get_actor_states(job.get("events", []), "Ego")
    speeds = [s.get("speed_mps", 0) for s in states]
    assert len(speeds) > 0, f"No frames in {map_name}"
    n = max(1, len(speeds) // 4)
    tail = speeds[-n:]
    assert any(s < 1.0 for s in tail), (
        f"Vehicle never stopped in {map_name}. tail speeds: {tail}"
    )


def _run_recording(map_name: str) -> None:
    ctx = _get_or_load_map(map_name)
    road = ctx.driving_road
    assert road is not None
    payload = base_payload(
        map_name, road["id"],
        actors=[ego_actor(road["id"], autopilot=True, speed_kph=40)],
        recording=True,
        duration=3.0,
    )
    job, diag = submit_and_wait(payload, timeout=75)
    assert diag.get("saved_frame_count", 0) > 0, (
        f"No frames saved in {map_name}: {diag.get('saved_frame_count')}"
    )


def _run_no_errors(map_name: str) -> None:
    ctx = _get_or_load_map(map_name)
    road = ctx.driving_road
    assert road is not None
    payload = base_payload(
        map_name, road["id"],
        actors=[ego_actor(road["id"], autopilot=True, speed_kph=40)],
    )
    job, diag = submit_and_wait(payload)
    assert diag.get("worker_error") is None, (
        f"worker_error in {map_name}: {diag['worker_error']}"
    )
    assert len(diag.get("skipped_actors", [])) == 0, (
        f"skipped_actors in {map_name}: {diag['skipped_actors']}"
    )


# ---------------------------------------------------------------------------
# Town01 tests
# ---------------------------------------------------------------------------


class TestTown01:
    MAP = "Town01"

    @pytest.fixture(autouse=True, scope="class")
    def _load_map(self):
        _get_or_load_map(self.MAP)

    def test_Town01_vehicle_spawns_and_drives(self):
        _run_vehicle_spawns_and_drives(self.MAP)

    def test_Town01_lane_change(self):
        _run_lane_change(self.MAP)

    def test_Town01_stop(self):
        _run_stop(self.MAP)

    def test_Town01_recording(self):
        _run_recording(self.MAP)

    def test_Town01_no_errors(self):
        _run_no_errors(self.MAP)


# ---------------------------------------------------------------------------
# Town03 tests
# ---------------------------------------------------------------------------


class TestTown03:
    MAP = "Town03"

    @pytest.fixture(autouse=True, scope="class")
    def _load_map(self):
        _get_or_load_map(self.MAP)

    def test_Town03_vehicle_spawns_and_drives(self):
        _run_vehicle_spawns_and_drives(self.MAP)

    def test_Town03_lane_change(self):
        _run_lane_change(self.MAP)

    def test_Town03_stop(self):
        _run_stop(self.MAP)

    def test_Town03_recording(self):
        _run_recording(self.MAP)

    def test_Town03_no_errors(self):
        _run_no_errors(self.MAP)


# ---------------------------------------------------------------------------
# Town10HD_Opt tests
# ---------------------------------------------------------------------------


class TestTown10HDOpt:
    MAP = "Town10HD_Opt"

    @pytest.fixture(autouse=True, scope="class")
    def _load_map(self):
        _get_or_load_map(self.MAP)

    def test_Town10HD_Opt_vehicle_spawns_and_drives(self):
        _run_vehicle_spawns_and_drives(self.MAP)

    def test_Town10HD_Opt_lane_change(self):
        _run_lane_change(self.MAP)

    def test_Town10HD_Opt_stop(self):
        _run_stop(self.MAP)

    def test_Town10HD_Opt_recording(self):
        _run_recording(self.MAP)

    def test_Town10HD_Opt_no_errors(self):
        _run_no_errors(self.MAP)
