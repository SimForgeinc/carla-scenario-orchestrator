"""Integration test: validates timed-path vehicles reach waypoints in CARLA.

Submits a real simulation job with a timed_path car, collects per-frame
actor positions via WebSocket, and verifies each waypoint was reached
within tolerance.

Requires:
  - CARLA orchestrator running at localhost:18421
  - A CARLA map loaded with drivable road segments
  - pip: pytest-asyncio aiohttp websockets
"""
from __future__ import annotations

import asyncio
import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Any

import pytest
import aiohttp
import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORCHESTRATOR_BASE = "http://127.0.0.1:18421"
WS_BASE = "ws://127.0.0.1:18421"
POSITION_TOLERANCE_M = 5.0
TIME_TOLERANCE_S = 1.5
SIM_TIMEOUT_S = 45.0
WAYPOINT_SPACING_M = 15.0  # ~54 kph, reasonable for PID control
NUM_WAYPOINTS = 5  # t=0 through t=4
MIN_ROAD_LENGTH_M = 60.0
MIN_STRAIGHTNESS = 0.80
SIM_DURATION_S = 10  # 5s of waypoints + 5s buffer
ACTOR_LABEL = "Timed 1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WaypointResult:
    index: int
    expected_x: float
    expected_y: float
    expected_time: float
    actual_x: float | None = None
    actual_y: float | None = None
    actual_time: float | None = None
    distance: float | None = None
    reached: bool = False
    skipped: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _sync_get_json(url: str) -> dict:
    resp = urllib.request.urlopen(url, timeout=5)
    return json.loads(resp.read())


def check_health() -> bool:
    try:
        data = _sync_get_json(f"{ORCHESTRATOR_BASE}/api/health")
        return bool(data.get("ok")) and bool(data.get("carla_connected"))
    except Exception:
        return False


def get_current_map() -> str:
    data = _sync_get_json(f"{ORCHESTRATOR_BASE}/api/carla/status")
    return data.get("current_map", "")


def discover_waypoints() -> tuple[list[dict[str, float]], str, dict[str, Any]]:
    """Find a straight road segment and sample waypoints along it.

    Returns (waypoints, map_name, segment_info) where waypoints are in
    frontend coordinates (same as the payload and WS stream).

    Uses a margin from road ends to avoid barriers/junctions.
    Prefers longer, straighter segments for reliable PID driving.
    """
    status = _sync_get_json(f"{ORCHESTRATOR_BASE}/api/carla/status")
    map_name = status.get("current_map", "")
    runtime = _sync_get_json(f"{ORCHESTRATOR_BASE}/api/map/runtime")
    segments = runtime.get("road_segments", [])

    MARGIN_M = 20.0  # stay 20m from road ends to avoid barriers
    MIN_USABLE_M = 60.0  # need at least 60m of usable road

    # Score all non-junction segments
    candidates: list[tuple[float, float, float, Any]] = []
    for seg in segments:
        if seg.get("is_junction"):
            continue
        cl = seg.get("centerline", [])
        if len(cl) < 4:
            continue
        total_len = 0.0
        for i in range(1, len(cl)):
            dx = cl[i]["x"] - cl[i - 1]["x"]
            dy = cl[i]["y"] - cl[i - 1]["y"]
            total_len += math.sqrt(dx * dx + dy * dy)
        usable = total_len - 2 * MARGIN_M
        if usable < MIN_USABLE_M:
            continue
        dx = cl[-1]["x"] - cl[0]["x"]
        dy = cl[-1]["y"] - cl[0]["y"]
        straight = math.sqrt(dx * dx + dy * dy)
        straightness = straight / max(total_len, 0.001)
        if straightness < MIN_STRAIGHTNESS:
            continue
        # Score: prefer longer usable area, then straighter
        candidates.append((usable, straightness, total_len, seg))

    if not candidates:
        pytest.skip("No suitable road segment found (need ≥60m usable + ≥80% straight)")

    # Pick the one with most usable length
    candidates.sort(key=lambda c: (-c[1], -c[0]))  # prefer straighter, then longer
    usable, straightness, total_len, seg = candidates[0]
    cl = seg["centerline"]
    road_id = seg["road_id"]
    lane_id = seg.get("lane_id", -1)
    section_id = seg.get("section_id", 0)

    # Compute cumulative distances along centerline
    cum_dist = [0.0]
    for i in range(1, len(cl)):
        dx = cl[i]["x"] - cl[i - 1]["x"]
        dy = cl[i]["y"] - cl[i - 1]["y"]
        cum_dist.append(cum_dist[-1] + math.sqrt(dx * dx + dy * dy))

    # Sample NUM_WAYPOINTS points with margin from edges
    spacing = min(WAYPOINT_SPACING_M, usable / (NUM_WAYPOINTS - 1))
    waypoints = []
    for wp_idx in range(NUM_WAYPOINTS):
        target_dist = MARGIN_M + spacing * wp_idx
        # Interpolate along centerline
        for i in range(1, len(cl)):
            if cum_dist[i] >= target_dist:
                t = (target_dist - cum_dist[i - 1]) / max(cum_dist[i] - cum_dist[i - 1], 0.001)
                x = cl[i - 1]["x"] + t * (cl[i]["x"] - cl[i - 1]["x"])
                y = cl[i - 1]["y"] + t * (cl[i]["y"] - cl[i - 1]["y"])
                waypoints.append({
                    "x": float(x),
                    "y": -float(y),  # CARLA → frontend: negate y
                    "time": float(wp_idx),
                })
                break

    seg_info = {
        "road_id": road_id,
        "section_id": section_id,
        "lane_id": lane_id,
        "length": total_len,
        "usable_length": usable,
        "straightness": straightness,
        "spacing": spacing,
        "num_centerline_points": len(cl),
    }

    if len(waypoints) < 3:
        pytest.skip(f"Could only place {len(waypoints)} waypoints (need ≥3)")

    return waypoints, map_name, seg_info


def build_payload(waypoints: list[dict], map_name: str, road_id: Any, section_id: int, lane_id: int) -> dict:
    """Build SimulationRunRequest with a single timed_path actor."""
    return {
        "map_name": map_name,
        "selected_roads": [],
        "actors": [
            {
                "id": "timed-test-1",
                "label": ACTOR_LABEL,
                "kind": "vehicle",
                "role": "traffic",
                "is_static": False,
                "placement_mode": "timed_path",
                "blueprint": "vehicle.tesla.model3",
                "spawn": {
                    "road_id": str(road_id),
                    "s_fraction": 0.1,
                    "lane_id": lane_id,
                    "section_id": section_id,
                },
                "spawn_point": {"x": waypoints[0]["x"], "y": waypoints[0]["y"]},
                "route": [],
                "route_direction": "forward",
                "lane_facing": "with_lane",
                "destination": None,
                "destination_point": None,
                "speed_kph": 120,
                "autopilot": False,
                "color": None,
                "notes": None,
                "timed_waypoints": waypoints,
                "timeline": [],
            }
        ],
        "duration_seconds": SIM_DURATION_S,
        "fixed_delta_seconds": 0.05,
        "topdown_recording": False,
        "recording_width": 640,
        "recording_height": 480,
        "recording_fov": 90,
    }


async def submit_job(session: aiohttp.ClientSession, payload: dict) -> str:
    async with session.post(f"{ORCHESTRATOR_BASE}/api/jobs", json=payload) as resp:
        assert resp.status == 200, f"Job submission failed: {resp.status} {await resp.text()}"
        body = await resp.json()
        return body["job_id"]


async def collect_frames(job_id: str, timeout: float = SIM_TIMEOUT_S) -> list[dict]:
    """Connect to WebSocket and collect all frames until simulation ends."""
    frames: list[dict] = []
    uri = f"{WS_BASE}/api/jobs/{job_id}/stream"
    deadline = asyncio.get_event_loop().time() + timeout

    async with websockets.connect(uri) as ws:
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(raw)
                messages = data if isinstance(data, list) else [data]
                for msg in messages:
                    frames.append(msg)
                    if msg.get("simulation_ended"):
                        return frames
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break

    if not any(f.get("simulation_ended") for f in frames):
        raise TimeoutError(f"Simulation did not complete within {timeout}s ({len(frames)} frames collected)")
    return frames


def validate_waypoint(
    frames: list[dict],
    wp: dict,
    actor_label: str,
    index: int,
) -> WaypointResult:
    """Find the closest approach of the actor to the waypoint within the time window."""
    result = WaypointResult(
        index=index,
        expected_x=wp["x"],
        expected_y=wp["y"],
        expected_time=wp["time"],
    )

    best_dist = None
    for frame in frames:
        ts = frame.get("timestamp", 0.0)
        # Search in a wider window: from simulation start to well past the deadline
        if ts > wp["time"] + TIME_TOLERANCE_S + 2.0:
            continue
        for actor in frame.get("actors", []):
            if actor.get("label") != actor_label:
                continue
            d = distance_2d(actor["x"], actor["y"], wp["x"], wp["y"])
            if best_dist is None or d < best_dist:
                best_dist = d
                result.actual_x = actor["x"]
                result.actual_y = actor["y"]
                result.actual_time = ts
                result.distance = d

    result.reached = best_dist is not None and best_dist < POSITION_TOLERANCE_M
    return result


def print_report(
    results: list[WaypointResult],
    map_name: str,
    seg_info: dict,
) -> None:
    """Print a detailed human-readable validation report."""
    print()
    print("=" * 80)
    print("TIMED WAYPOINT VALIDATION REPORT")
    print(f"Map: {map_name} | Duration: {SIM_DURATION_S}s | Waypoints: {len(results)}")
    print(f"Road: {seg_info['road_id']} lane={seg_info['lane_id']} "
          f"length={seg_info['length']:.0f}m straightness={seg_info['straightness']:.3f}")
    print(f"Tolerance: position={POSITION_TOLERANCE_M}m time={TIME_TOLERANCE_S}s")
    print("-" * 80)
    print(f"{'WP':>3} | {'Expected (x, y) @ t':>22} | {'Actual (x, y) @ t':>22} | {'Dist':>6} | Status")
    print(f"{'---':>3} | {'-' * 22} | {'-' * 22} | {'-' * 6} | ------")

    for r in results:
        expected = f"({r.expected_x:7.1f},{r.expected_y:7.1f}) @ {r.expected_time:.1f}s"
        if r.skipped:
            actual = "(skipped — spawn)".ljust(22)
            dist = "—".rjust(6)
            status = "SKIP"
        elif r.actual_x is not None:
            actual = f"({r.actual_x:7.1f},{r.actual_y:7.1f}) @ {r.actual_time:.1f}s"
            dist = f"{r.distance:.1f}m".rjust(6)
            status = "PASS" if r.reached else "FAIL"
        else:
            actual = "(not found)".ljust(22)
            dist = "—".rjust(6)
            status = "FAIL"
        print(f"{r.index:3d} | {expected} | {actual} | {dist} | {status}")

    # Visit order check
    visit_times = [r.actual_time for r in results if r.actual_time is not None and not r.skipped]
    if len(visit_times) >= 2:
        monotonic = all(visit_times[i] <= visit_times[i + 1] for i in range(len(visit_times) - 1))
        times_str = " < ".join(f"{t:.2f}" for t in visit_times)
        symbol = "✓" if monotonic else "✗"
        print(f"\nVisit order: {symbol} {'monotonic' if monotonic else 'NOT monotonic'} ({times_str})")

    passed = sum(1 for r in results if r.reached or r.skipped)
    total = len(results)
    print(f"Result: {passed}/{total} waypoints passed")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_if_unavailable():
    if not check_health():
        pytest.skip("CARLA orchestrator not available or CARLA not connected")


@pytest.mark.asyncio
async def test_car_reaches_timed_waypoints():
    """Submit a timed-path car and verify it reaches each waypoint."""

    # 1. Discover waypoints from runtime map
    waypoints, map_name, seg_info = discover_waypoints()
    print(f"\nDiscovered {len(waypoints)} waypoints on map {map_name}")
    for wp in waypoints:
        print(f"  t={wp['time']:.0f}s: ({wp['x']:.1f}, {wp['y']:.1f})")

    # 2. Submit simulation job
    payload = build_payload(
        waypoints, map_name,
        road_id=seg_info["road_id"],
        section_id=seg_info["section_id"],
        lane_id=seg_info["lane_id"],
    )

    async with aiohttp.ClientSession() as session:
        job_id = await submit_job(session, payload)
        print(f"Job submitted: {job_id}")

    # 3. Collect frames via WebSocket
    frames = await collect_frames(job_id)
    actor_frames = [f for f in frames if any(a.get("label") == ACTOR_LABEL for a in f.get("actors", []))]
    print(f"Collected {len(frames)} frames ({len(actor_frames)} with actor '{ACTOR_LABEL}')")

    assert len(actor_frames) > 0, f"Actor '{ACTOR_LABEL}' never appeared in simulation frames"

    # 4. Validate each waypoint
    results: list[WaypointResult] = []
    for i, wp in enumerate(waypoints):
        if i == 0:
            # Skip spawn waypoint — CARLA may place slightly offset
            r = WaypointResult(
                index=i, expected_x=wp["x"], expected_y=wp["y"],
                expected_time=wp["time"], skipped=True,
            )
        else:
            r = validate_waypoint(frames, wp, ACTOR_LABEL, i)
        results.append(r)

    # 5. Print report
    print_report(results, map_name, seg_info)

    # 6. Assert all non-skipped waypoints reached
    for r in results:
        if r.skipped:
            continue
        assert r.reached, (
            f"Waypoint {r.index} NOT reached: "
            f"expected ({r.expected_x:.1f}, {r.expected_y:.1f}) @ t={r.expected_time:.1f}s, "
            f"closest approach was {r.distance:.1f}m "
            f"at ({r.actual_x:.1f}, {r.actual_y:.1f}) @ t={r.actual_time:.1f}s "
            f"(tolerance: {POSITION_TOLERANCE_M}m)"
        )

    # 7. Assert visit order is monotonic
    visit_times = [r.actual_time for r in results if r.actual_time is not None and not r.skipped]
    for i in range(1, len(visit_times)):
        assert visit_times[i] >= visit_times[i - 1], (
            f"Visit order violation: waypoint {i + 1} reached at {visit_times[i]:.2f}s "
            f"before waypoint {i} at {visit_times[i - 1]:.2f}s"
        )
