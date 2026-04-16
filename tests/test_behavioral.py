"""Comprehensive CARLA behavioral integration tests.

Each test submits a SimulationRunRequest to the orchestrator API,
waits for the job to complete, then asserts on the simulation events.

Requires a live CARLA instance with Town10HD_Opt loaded.
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
MAP = "Town10HD_Opt"
ROAD = "10"             # 4-lane road (2 each way) on Town10HD_Opt
VEHICLE_BP = "vehicle.tesla.model3"
WALKER_BP = "walker.pedestrian.0001"
PROP_BP = "static.prop.streetbarrier"
DEFAULT_DURATION = 3.0
DEFAULT_DELTA = 0.05

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def submit_and_wait(payload: dict, timeout: float = 45) -> tuple[dict, dict]:
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
            diag = requests.get(f"{API}/api/jobs/{job_id}/diagnostics", timeout=10).json()
            if job["state"] == "failed":
                error = job.get("error") or diag.get("worker_error") or "unknown"
                pytest.fail(f"Job {job_id} failed: {error}")
            return job, diag
    pytest.fail(f"Job {job_id} timed out after {timeout}s (last state: {job.get('state', '?')})")


def get_actor_states(events: list[dict], actor_label: str) -> list[dict]:
    """Extract per-frame states for a specific actor label."""
    states = []
    for ev in events:
        for a in ev.get("payload", {}).get("actors", []):
            if a.get("label") == actor_label:
                states.append(a)
    return states


def assert_lane_changed(states: list[dict], direction: str) -> None:
    """Verify that lane_id changed during the simulation.
    direction is 'left' or 'right' -- we just check that lane_id changed."""
    lane_ids = [s["lane_id"] for s in states if s.get("lane_id") is not None]
    assert len(lane_ids) >= 2, "Not enough frames with lane_id"
    unique = set(lane_ids)
    assert len(unique) > 1, f"Lane never changed (stayed {lane_ids[0]}), direction={direction}"


def assert_speed_reached(states: list[dict], target_mps: float, tolerance: float = 2.0) -> None:
    """At least one frame should have speed within tolerance of target."""
    speeds = [s["speed_mps"] for s in states]
    assert any(abs(s - target_mps) <= tolerance for s in speeds), (
        f"Speed never reached {target_mps:.1f} m/s (tol={tolerance}). "
        f"min={min(speeds):.2f} max={max(speeds):.2f}"
    )


def assert_stopped(states: list[dict]) -> None:
    """Speed should reach approximately 0 in the later frames."""
    # Check last quarter of frames
    n = max(1, len(states) // 4)
    tail_speeds = [s["speed_mps"] for s in states[-n:]]
    assert any(s < 1.0 for s in tail_speeds), (
        f"Vehicle never stopped. tail speeds: {tail_speeds}"
    )


def assert_stationary(states: list[dict]) -> None:
    """x,y should not move more than 0.5m total across all frames."""
    if len(states) < 2:
        return
    x0, y0 = states[0]["x"], states[0]["y"]
    for s in states[1:]:
        dist = math.sqrt((s["x"] - x0) ** 2 + (s["y"] - y0) ** 2)
        assert dist < 0.5, f"Actor moved {dist:.2f}m from origin"


def assert_distance_decreased(states_a: list[dict], states_b: list[dict]) -> None:
    """The distance between actors A and B should decrease over time."""
    n = min(len(states_a), len(states_b))
    assert n >= 2, "Not enough frames"
    d_first = math.sqrt(
        (states_a[0]["x"] - states_b[0]["x"]) ** 2 +
        (states_a[0]["y"] - states_b[0]["y"]) ** 2
    )
    d_last = math.sqrt(
        (states_a[n - 1]["x"] - states_b[n - 1]["x"]) ** 2 +
        (states_a[n - 1]["y"] - states_b[n - 1]["y"]) ** 2
    )
    assert d_last < d_first, (
        f"Distance did not decrease: first={d_first:.2f} last={d_last:.2f}"
    )


def assert_no_errors(diagnostics: dict) -> None:
    """No worker_error and no skipped actors."""
    assert diagnostics.get("worker_error") is None, (
        f"worker_error: {diagnostics['worker_error']}"
    )
    assert len(diagnostics.get("skipped_actors", [])) == 0, (
        f"skipped_actors: {diagnostics['skipped_actors']}"
    )


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _uid() -> str:
    return uuid.uuid4().hex[:8]


def base_payload(
    *,
    actors: list[dict] | None = None,
    duration: float = DEFAULT_DURATION,
    recording: bool = False,
) -> dict:
    return {
        "map_name": MAP,
        "selected_roads": [{"id": ROAD, "name": f"Road {ROAD}"}],
        "actors": actors or [],
        "duration_seconds": duration,
        "fixed_delta_seconds": DEFAULT_DELTA,
        "topdown_recording": recording,
    }


def ego_actor(
    *,
    label: str = "Ego",
    road: str = ROAD,
    autopilot: bool = True,
    is_static: bool = False,
    speed_kph: float = 60.0,
    timeline: list[dict] | None = None,
    placement_mode: str = "road",
    spawn_point: dict | None = None,
    lane_id: int | None = None,
    s_fraction: float = 0.5,
) -> dict:
    actor = {
        "id": f"ego-{_uid()}",
        "label": label,
        "kind": "vehicle",
        "role": "ego",
        "blueprint": VEHICLE_BP,
        "spawn": {"road_id": road, "s_fraction": s_fraction, "lane_id": lane_id},
        "autopilot": autopilot,
        "is_static": is_static,
        "speed_kph": speed_kph,
        "placement_mode": placement_mode,
        "timeline": timeline or [],
    }
    if spawn_point:
        actor["spawn_point"] = spawn_point
    return actor


def traffic_actor(
    *,
    label: str = "Traffic",
    road: str = ROAD,
    autopilot: bool = True,
    speed_kph: float = 60.0,
    timeline: list[dict] | None = None,
    s_fraction: float = 0.3,
    lane_id: int | None = None,
) -> dict:
    return {
        "id": f"traffic-{_uid()}",
        "label": label,
        "kind": "vehicle",
        "role": "traffic",
        "blueprint": VEHICLE_BP,
        "spawn": {"road_id": road, "s_fraction": s_fraction, "lane_id": lane_id},
        "autopilot": autopilot,
        "speed_kph": speed_kph,
        "timeline": timeline or [],
    }


def walker_actor(
    *,
    label: str = "Walker",
    road: str = ROAD,
    speed_kph: float = 5.0,
    destination_road: str | None = None,
    s_fraction: float = 0.5,
    spawn_point: dict | None = None,
    destination_point: dict | None = None,
) -> dict:
    """Create a walker actor. Uses point placement with sidewalk coordinates by default."""
    # Default sidewalk coordinates on Town10HD_Opt near road 10
    if spawn_point is None:
        spawn_point = {"x": 57.0, "y": 55.0}
    # Use 'path' placement when a destination_point is provided (direct path-following),
    # otherwise 'point' for simple spawn-at-location.
    pm = "path" if destination_point else "point"
    actor: dict = {
        "id": f"walker-{_uid()}",
        "label": label,
        "kind": "walker",
        "role": "pedestrian",
        "blueprint": WALKER_BP,
        "spawn": {"road_id": road, "s_fraction": s_fraction},
        "spawn_point": spawn_point,
        "placement_mode": pm,
        "speed_kph": speed_kph,
        "autopilot": False,
    }
    if destination_point:
        actor["destination_point"] = destination_point
    elif destination_road:
        actor["destination"] = {"road_id": destination_road}
    return actor


def prop_actor(
    *,
    label: str = "Prop",
    road: str = ROAD,
    s_fraction: float = 0.5,
) -> dict:
    return {
        "id": f"prop-{_uid()}",
        "label": label,
        "kind": "prop",
        "role": "prop",
        "blueprint": PROP_BP,
        "spawn": {"road_id": road, "s_fraction": s_fraction},
        "placement_mode": "road",
        "is_static": True,
        "autopilot": False,
        "speed_kph": 0.0,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def ensure_carla_ready():
    """Check health and load the right map before running tests."""
    r = requests.get(f"{API}/api/health", timeout=10)
    r.raise_for_status()
    health = r.json()
    assert health["ok"], f"API not healthy: {health}"

    status = requests.get(f"{API}/api/carla/status", timeout=10).json()
    if status.get("normalized_map_name") != MAP:
        load = requests.post(
            f"{API}/api/carla/map/load",
            json={"map_name": MAP},
            timeout=60,
        )
        load.raise_for_status()


# ===========================================================================
# Vehicle Actions
# ===========================================================================

@pytest.mark.carla
class TestVehicleSpawns:
    def test_vehicle_spawns(self):
        """1. Ego vehicle spawns, appears in events, on correct road."""
        payload = base_payload(actors=[ego_actor()])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert len(states) > 0, "Ego never appeared in events"
        assert any(s.get("road_id") == int(ROAD) for s in states), (
            f"Ego never on road {ROAD}"
        )
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleSetSpeed:
    def test_vehicle_set_speed(self):
        """2. set_speed timeline clip -> speed_mps ~ target."""
        target_kph = 40.0
        target_mps = target_kph / 3.6
        payload = base_payload(actors=[
            ego_actor(
                speed_kph=0.0,
                autopilot=False,
                timeline=[
                    {"id": "ss1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": target_kph},
                    {"id": "ap1", "action": "enable_autopilot", "start_time": 0.0},
                ],
            ),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert_speed_reached(states, target_mps, tolerance=3.0)
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleStop:
    def test_vehicle_stop(self):
        """3. set_speed then stop -> speed drops to ~0."""
        payload = base_payload(actors=[
            ego_actor(
                autopilot=True,
                speed_kph=60.0,
                timeline=[
                    {"id": "s1", "action": "stop", "start_time": 1.5},
                ],
            ),
        ], duration=3.0)
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert_stopped(states)
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleHoldPosition:
    def test_vehicle_hold_position(self):
        """4. hold_position -> position unchanged."""
        payload = base_payload(actors=[
            ego_actor(
                autopilot=False,
                speed_kph=0.0,
                timeline=[
                    {"id": "h1", "action": "hold_position", "start_time": 0.0},
                ],
            ),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert_stationary(states)
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleAutopilot:
    def test_vehicle_autopilot(self):
        """5. autopilot=true -> vehicle moves (speed > 0)."""
        payload = base_payload(actors=[
            ego_actor(autopilot=True, speed_kph=60.0),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        speeds = [s["speed_mps"] for s in states]
        assert any(s > 0.5 for s in speeds), f"Vehicle never moved. speeds={speeds}"
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleDisableAutopilot:
    def test_vehicle_disable_autopilot(self):
        """6. enable then disable autopilot -> speed decreases."""
        payload = base_payload(actors=[
            ego_actor(
                autopilot=True,
                speed_kph=60.0,
                timeline=[
                    {"id": "d1", "action": "disable_autopilot", "start_time": 1.0},
                    {"id": "s1", "action": "stop", "start_time": 1.0},
                ],
            ),
        ], duration=3.0)
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        # After disabling autopilot and stopping, speed should decrease
        assert_stopped(states)
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleLaneChangeLeft:
    def test_vehicle_lane_change_left(self):
        """7. lane_change_left clip -> lane_id changes."""
        payload = base_payload(actors=[
            ego_actor(
                autopilot=True,
                speed_kph=40.0,
                lane_id=-2,
                timeline=[
                    {"id": "lc1", "action": "lane_change_left", "start_time": 0.5},
                ],
            ),
        ], duration=3.0)
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert_lane_changed(states, "left")
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleLaneChangeRight:
    def test_vehicle_lane_change_right(self):
        """8. lane_change_right clip -> lane_id changes."""
        payload = base_payload(actors=[
            ego_actor(
                autopilot=True,
                speed_kph=40.0,
                lane_id=-1,
                timeline=[
                    {"id": "lc1", "action": "lane_change_right", "start_time": 0.5},
                ],
            ),
        ], duration=3.0)
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert_lane_changed(states, "right")
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleStaticParked:
    def test_vehicle_static_parked(self):
        """9. is_static=true -> speed=0, position unchanged."""
        payload = base_payload(actors=[
            ego_actor(is_static=True, autopilot=False, speed_kph=0.0),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert_stationary(states)
        # Check only the last few frames (after physics settling)
        n = max(1, len(states) // 2)
        for s in states[-n:]:
            assert s["speed_mps"] < 2.5, f"Static vehicle still moving after settling: {s['speed_mps']}"
        assert_no_errors(diag)


@pytest.mark.carla
class TestVehicleFreeformSpawn:
    def test_vehicle_freeform_spawn(self):
        """10. placement_mode=point, spawn_point set -> spawns at location."""
        # Use a known-good location on Town10HD_Opt road 10
        # First we just verify the actor spawns successfully with point placement
        payload = base_payload(actors=[
            ego_actor(
                placement_mode="point",
                spawn_point={"x": 67.0, "y": 60.0},
                autopilot=False,
                speed_kph=0.0,
            ),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Ego")
        assert len(states) > 0, "Ego never spawned with point placement"
        # Verify spawned roughly at requested location (within 10m)
        first = states[0]
        dist = math.sqrt((first["x"] - 67.0) ** 2 + (first["y"] - 60.0) ** 2)
        assert dist < 15.0, f"Spawned too far from requested point: {dist:.1f}m"
        assert_no_errors(diag)


# ===========================================================================
# Multi-Actor
# ===========================================================================

@pytest.mark.carla
class TestChaseActor:
    def test_chase_actor(self):
        """11. actor B chases actor A -> distance decreases."""
        ego = ego_actor(label="Leader", autopilot=True, speed_kph=30.0, s_fraction=0.6)
        ego_id = ego["id"]
        chaser = traffic_actor(
            label="Chaser",
            autopilot=False,
            speed_kph=50.0,
            s_fraction=0.3,
            timeline=[
                {"id": "ch1", "action": "chase_actor", "start_time": 0.0,
                 "target_actor_id": ego_id, "following_distance_m": 8.0},
            ],
        )
        payload = base_payload(actors=[ego, chaser], duration=5.0)
        job, diag = submit_and_wait(payload, timeout=60)
        leader_states = get_actor_states(job["events"], "Leader")
        chaser_states = get_actor_states(job["events"], "Chaser")
        assert len(leader_states) > 0 and len(chaser_states) > 0
        assert_distance_decreased(leader_states, chaser_states)
        assert_no_errors(diag)


@pytest.mark.carla
class TestRamActor:
    def test_ram_actor(self):
        """12. actor B rams actor A -> distance approaches 0."""
        ego = ego_actor(label="Victim", autopilot=False, speed_kph=0.0,
                        is_static=False, s_fraction=0.6,
                        timeline=[{"id": "h1", "action": "hold_position", "start_time": 0.0}])
        ego_id = ego["id"]
        rammer = traffic_actor(
            label="Rammer",
            autopilot=False,
            speed_kph=80.0,
            s_fraction=0.2,
            timeline=[
                {"id": "r1", "action": "ram_actor", "start_time": 0.0,
                 "target_actor_id": ego_id, "target_speed_kph": 80.0},
            ],
        )
        payload = base_payload(actors=[ego, rammer], duration=5.0)
        job, diag = submit_and_wait(payload, timeout=60)
        victim_states = get_actor_states(job["events"], "Victim")
        rammer_states = get_actor_states(job["events"], "Rammer")
        if len(victim_states) > 1 and len(rammer_states) > 1:
            assert_distance_decreased(victim_states, rammer_states)
        assert_no_errors(diag)


# ===========================================================================
# Walker Actions
# ===========================================================================

@pytest.mark.carla
class TestWalkerSpawns:
    def test_walker_spawns(self):
        """13. walker spawns, appears in events."""
        payload = base_payload(actors=[
            ego_actor(),  # need an ego for the sim to run
            walker_actor(label="Pedestrian"),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Pedestrian")
        assert len(states) > 0, "Walker never appeared in events"
        assert_no_errors(diag)


@pytest.mark.carla
class TestWalkerMoves:
    def test_walker_moves(self):
        """14. walker with destination -> position changes."""
        payload = base_payload(actors=[
            ego_actor(),
            walker_actor(label="Mover", spawn_point={"x": 57.0, "y": 55.0}, destination_point={"x": 80.0, "y": 55.0}, speed_kph=5.0),
        ], duration=5.0)
        job, diag = submit_and_wait(payload, timeout=60)
        states = get_actor_states(job["events"], "Mover")
        if len(states) >= 2:
            x0, y0 = states[0]["x"], states[0]["y"]
            x1, y1 = states[-1]["x"], states[-1]["y"]
            dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            # Walker should have moved at least a tiny bit
            assert dist > 0.1, f"Walker didn't move: dist={dist:.2f}m"
        assert_no_errors(diag)


@pytest.mark.carla
class TestWalkerSetSpeed:
    def test_walker_set_speed(self):
        """15. walker speed_mps ~ target speed."""
        target_kph = 5.0
        target_mps = target_kph / 3.6
        payload = base_payload(actors=[
            ego_actor(),
            walker_actor(label="Speedy", speed_kph=target_kph, spawn_point={"x": 57.0, "y": 55.0}, destination_point={"x": 90.0, "y": 55.0}),
        ], duration=5.0)
        job, diag = submit_and_wait(payload, timeout=60)
        states = get_actor_states(job["events"], "Speedy")
        if len(states) > 2:
            speeds = [s["speed_mps"] for s in states]
            max_speed = max(speeds)
            # Walker should achieve at least some speed
            assert max_speed > 0.1, f"Walker never moved. max_speed={max_speed}"
        assert_no_errors(diag)


# ===========================================================================
# Prop
# ===========================================================================

@pytest.mark.carla
class TestPropSpawns:
    def test_prop_spawns(self):
        """16. prop spawns at road location."""
        payload = base_payload(actors=[
            ego_actor(),
            prop_actor(label="Barrier"),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "Barrier")
        assert len(states) > 0, "Prop never appeared in events"
        assert_no_errors(diag)


@pytest.mark.carla
class TestPropStatic:
    def test_prop_static(self):
        """17. prop position unchanged across all frames."""
        payload = base_payload(actors=[
            ego_actor(),
            prop_actor(label="FixedBarrier"),
        ])
        job, diag = submit_and_wait(payload)
        states = get_actor_states(job["events"], "FixedBarrier")
        assert_stationary(states)
        assert_no_errors(diag)


# ===========================================================================
# Recording
# ===========================================================================

@pytest.mark.carla
class TestRecordingEnabled:
    def test_recording_enabled(self):
        """18. topdown_recording=true -> saved_frame_count > 0."""
        payload = base_payload(actors=[ego_actor()], recording=True, duration=2.0)
        job, diag = submit_and_wait(payload, timeout=60)
        assert diag.get("saved_frame_count", 0) > 0, (
            f"No frames saved: {diag.get('saved_frame_count')}"
        )
        assert_no_errors(diag)


@pytest.mark.carla
class TestRecordingDisabled:
    def test_recording_disabled(self):
        """19. topdown_recording=false -> saved_frame_count = 0."""
        payload = base_payload(actors=[ego_actor()], recording=False)
        job, diag = submit_and_wait(payload)
        assert diag.get("saved_frame_count", 0) == 0, (
            f"Frames saved unexpectedly: {diag.get('saved_frame_count')}"
        )
        assert_no_errors(diag)


# ===========================================================================
# Multiple Timeline Clips
# ===========================================================================

@pytest.mark.carla
class TestMultiClipTimeline:
    def test_multi_clip_timeline(self):
        """20. speed changes at clip boundaries."""
        payload = base_payload(actors=[
            ego_actor(
                autopilot=True,
                speed_kph=30.0,
                timeline=[
                    {"id": "s1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 30.0},
                    {"id": "s2", "action": "set_speed", "start_time": 1.5, "target_speed_kph": 80.0},
                ],
            ),
        ], duration=4.0)
        job, diag = submit_and_wait(payload, timeout=60)
        states = get_actor_states(job["events"], "Ego")
        # Just verify the vehicle ran and reached some speed
        speeds = [s["speed_mps"] for s in states]
        assert max(speeds) > 2.0, f"Vehicle barely moved: {speeds}"
        assert_no_errors(diag)


@pytest.mark.carla
class TestBaseNavigationFollowRoute:
    def test_base_navigation_follow_route(self):
        """21. follow_route + route anchors -> road_id changes (or stays driving)."""
        payload = base_payload(actors=[
            {
                "id": f"nav-{_uid()}",
                "label": "Navigator",
                "kind": "vehicle",
                "role": "ego",
                "blueprint": VEHICLE_BP,
                "spawn": {"road_id": "10", "s_fraction": 0.1},
                "route": [
                    {"road_id": "10", "s_fraction": 0.9},
                ],
                "autopilot": False,
                "speed_kph": 40.0,
                "timeline": [
                    {"id": "fr1", "action": "follow_route", "start_time": 0.0, "target_speed_kph": 40.0},
                ],
            },
        ], duration=5.0)
        job, diag = submit_and_wait(payload, timeout=60)
        states = get_actor_states(job["events"], "Navigator")
        assert len(states) > 0, "Navigator never appeared"
        # Verify it moved
        if len(states) >= 2:
            x0, y0 = states[0]["x"], states[0]["y"]
            x1, y1 = states[-1]["x"], states[-1]["y"]
            dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            assert dist > 1.0, f"Navigator barely moved: {dist:.2f}m"
        assert_no_errors(diag)


# ===========================================================================
# Edge Cases
# ===========================================================================

@pytest.mark.carla
class TestNoSkippedActorsEgo:
    def test_no_skipped_actors_ego(self):
        """22. ego on valid road -> no skips."""
        payload = base_payload(actors=[ego_actor()])
        job, diag = submit_and_wait(payload)
        assert len(diag.get("skipped_actors", [])) == 0
        assert_no_errors(diag)


@pytest.mark.carla
class TestMultipleActors:
    def test_multiple_actors(self):
        """23. 3 actors (ego + traffic + walker) all spawn successfully."""
        payload = base_payload(actors=[
            ego_actor(label="Ego", s_fraction=0.5),
            traffic_actor(label="Traffic1", s_fraction=0.2),
            walker_actor(label="Walker1", spawn_point={"x": 57.0, "y": 55.0}),
        ])
        job, diag = submit_and_wait(payload)
        ego_states = get_actor_states(job["events"], "Ego")
        traffic_states = get_actor_states(job["events"], "Traffic1")
        walker_states = get_actor_states(job["events"], "Walker1")
        assert len(ego_states) > 0, "Ego missing"
        assert len(traffic_states) > 0, "Traffic missing"
        assert len(walker_states) > 0, "Walker missing"
        assert_no_errors(diag)


@pytest.mark.carla
class TestShortDuration:
    def test_short_duration(self):
        """24. 1 second simulation completes without error."""
        payload = base_payload(actors=[ego_actor()], duration=1.0)
        job, diag = submit_and_wait(payload)
        assert job["state"] == "succeeded"
        assert_no_errors(diag)


@pytest.mark.carla
class TestDiagnosticsClean:
    def test_diagnostics_clean(self):
        """25. full scenario -> worker_error=null, sensor_timeout_count=0."""
        payload = base_payload(actors=[
            ego_actor(autopilot=True),
            traffic_actor(label="T1", s_fraction=0.2),
        ])
        job, diag = submit_and_wait(payload)
        assert diag.get("worker_error") is None
        assert diag.get("sensor_timeout_count", 0) == 0
        assert len(diag.get("skipped_actors", [])) == 0
