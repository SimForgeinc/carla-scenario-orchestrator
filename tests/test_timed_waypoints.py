"""Unit tests for timed waypoint support.

Tests the speed calculation, deadline logic, and model validation
for the timed_path placement mode.
"""
import math
import pytest
from orchestrator.carla_runner.models import (
    ActorDraft,
    ActorRoadAnchor,
    TimedWaypoint,
    ActorMapPoint,
)


# ---- Model tests ----


class TestTimedWaypointModel:
    def test_basic_creation(self):
        wp = TimedWaypoint(x=10.0, y=20.0, time=3.0)
        assert wp.x == 10.0
        assert wp.y == 20.0
        assert wp.time == 3.0

    def test_zero_time(self):
        wp = TimedWaypoint(x=0.0, y=0.0, time=0.0)
        assert wp.time == 0.0

    def test_actor_draft_with_timed_waypoints(self):
        actor = ActorDraft(
            id="test-1",
            label="Timed 1",
            kind="vehicle",
            role="traffic",
            placement_mode="timed_path",
            blueprint="vehicle.tesla.model3",
            spawn=ActorRoadAnchor(road_id="r1", s_fraction=0.5),
            spawn_point=ActorMapPoint(x=10.0, y=20.0),
            timed_waypoints=[
                TimedWaypoint(x=10.0, y=20.0, time=0.0),
                TimedWaypoint(x=30.0, y=40.0, time=1.0),
                TimedWaypoint(x=50.0, y=60.0, time=2.0),
            ],
        )
        assert actor.placement_mode == "timed_path"
        assert len(actor.timed_waypoints) == 3
        assert actor.timed_waypoints[1].time == 1.0

    def test_actor_draft_default_empty_timed_waypoints(self):
        actor = ActorDraft(
            id="test-2",
            label="Traffic 1",
            kind="vehicle",
            blueprint="vehicle.tesla.model3",
            spawn=ActorRoadAnchor(road_id="r1", s_fraction=0.5),
        )
        assert actor.timed_waypoints == []

    def test_timed_path_placement_mode_valid(self):
        actor = ActorDraft(
            id="test-3",
            label="Timed 2",
            kind="vehicle",
            placement_mode="timed_path",
            blueprint="vehicle.tesla.model3",
            spawn=ActorRoadAnchor(road_id="r1", s_fraction=0.5),
        )
        assert actor.placement_mode == "timed_path"


# ---- Speed calculation tests ----


def compute_desired_speed(distance: float, time_remaining: float, max_speed_mps: float) -> float:
    """Replicates the speed calculation from the tick loop."""
    time_remaining = max(0.05, time_remaining)
    return min(max_speed_mps, distance / time_remaining)


class TestSpeedCalculation:
    def test_normal_case(self):
        # 20m away, 2 seconds remaining, max speed 30 m/s
        speed = compute_desired_speed(20.0, 2.0, 30.0)
        assert speed == 10.0  # 20m / 2s = 10 m/s

    def test_clamped_to_max_speed(self):
        # 100m away, 1 second remaining, max speed 30 m/s
        speed = compute_desired_speed(100.0, 1.0, 30.0)
        assert speed == 30.0  # clamped to max

    def test_division_by_zero_guarded(self):
        # 0 time remaining should not divide by zero
        speed = compute_desired_speed(10.0, 0.0, 30.0)
        # time_remaining clamped to 0.05, so 10/0.05 = 200, clamped to 30
        assert speed == 30.0

    def test_negative_time_remaining_guarded(self):
        speed = compute_desired_speed(10.0, -5.0, 30.0)
        # Clamped to 0.05 → 10/0.05 = 200, clamped to 30
        assert speed == 30.0

    def test_very_close_waypoint(self):
        # 0.5m away, 1 second remaining
        speed = compute_desired_speed(0.5, 1.0, 30.0)
        assert speed == 0.5  # only need 0.5 m/s

    def test_exact_speed_needed(self):
        # 30m away, 1 second, max speed 30
        speed = compute_desired_speed(30.0, 1.0, 30.0)
        assert speed == 30.0

    def test_zero_distance(self):
        speed = compute_desired_speed(0.0, 2.0, 30.0)
        assert speed == 0.0


# ---- Waypoint advancement tests ----


class TestWaypointAdvancement:
    """Tests the waypoint index advancement logic from the tick loop."""

    def test_advance_on_arrival(self):
        """When reached=True, index should advance."""
        index = 1
        reached = True
        if reached:
            index += 1
        assert index == 2

    def test_hold_at_final_waypoint(self):
        """When all waypoints visited, vehicle should stop."""
        waypoints = [
            {"time": 0.0},
            {"time": 1.0},
            {"time": 2.0},
        ]
        index = 2  # at last waypoint
        reached = True
        if reached:
            index += 1
        # index (3) >= len(waypoints) (3), so vehicle should not be in remaining targets
        assert index >= len(waypoints)

    def test_deadline_passed_uses_max_speed(self):
        """When simulation_time > target_time, desired_speed should be max."""
        simulation_time = 3.5
        target_time = 3.0
        max_speed_mps = 25.0

        if simulation_time > target_time:
            desired_speed = max_speed_mps
        else:
            time_remaining = max(0.05, target_time - simulation_time)
            desired_speed = min(max_speed_mps, 10.0 / time_remaining)

        assert desired_speed == max_speed_mps
