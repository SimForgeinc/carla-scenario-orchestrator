"""Unit tests for _evaluate_timeline from simulation_service.

These tests run offline -- they do NOT require a live CARLA instance.
"""
from __future__ import annotations

import pytest

from orchestrator.carla_runner.models import ActorDraft, ActorRoadAnchor, ActorTimelineClip
from orchestrator.carla_runner.simulation_service import (
    TimelineActorState,
    TimelineDirective,
    _evaluate_timeline,
    _sorted_timeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actor(
    *,
    speed_kph: float = 60.0,
    autopilot: bool = True,
    is_static: bool = False,
    timeline: list[dict] | None = None,
) -> ActorDraft:
    clips = []
    if timeline:
        for t in timeline:
            clips.append(ActorTimelineClip(**t))
    return ActorDraft(
        id="test-actor",
        label="Test",
        kind="vehicle",
        role="ego",
        blueprint="vehicle.tesla.model3",
        spawn=ActorRoadAnchor(road_id="10"),
        speed_kph=speed_kph,
        autopilot=autopilot,
        is_static=is_static,
        timeline=clips,
    )


def _fresh_state() -> TimelineActorState:
    return TimelineActorState()


# ---------------------------------------------------------------------------
# Baseline (no timeline clips)
# ---------------------------------------------------------------------------

class TestBaselineDefaults:
    def test_default_speed_from_actor(self):
        actor = _make_actor(speed_kph=72.0)
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.target_speed_mps == pytest.approx(72.0 / 3.6, abs=0.01)

    def test_default_autopilot_enabled(self):
        actor = _make_actor(autopilot=True)
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.autopilot_enabled is True

    def test_default_autopilot_disabled(self):
        actor = _make_actor(autopilot=False)
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.autopilot_enabled is False

    def test_static_actor_holds_position(self):
        actor = _make_actor(is_static=True)
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.hold_position is True

    def test_non_static_no_hold(self):
        actor = _make_actor(is_static=False)
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.hold_position is False

    def test_no_route_follow_by_default(self):
        actor = _make_actor()
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.route_follow_enabled is False

    def test_no_chase_by_default(self):
        actor = _make_actor()
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.chase_target_actor_id is None

    def test_no_ram_by_default(self):
        actor = _make_actor()
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.0)
        assert d.ram_target_actor_id is None


# ---------------------------------------------------------------------------
# set_speed action
# ---------------------------------------------------------------------------

class TestSetSpeed:
    def test_set_speed_changes_target(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 30.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.target_speed_mps == pytest.approx(30.0 / 3.6, abs=0.01)

    def test_set_speed_clears_hold(self):
        actor = _make_actor(is_static=True, timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 20.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.hold_position is False

    def test_set_speed_without_target_kph_keeps_actor_default(self):
        actor = _make_actor(speed_kph=80.0, timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        # target_speed_kph is None so the set_speed branch is skipped entirely
        assert d.target_speed_mps == pytest.approx(80.0 / 3.6, abs=0.01)

    def test_set_speed_not_active_before_start(self):
        actor = _make_actor(speed_kph=50.0, timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 2.0, "target_speed_kph": 10.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.0)
        assert d.target_speed_mps == pytest.approx(50.0 / 3.6, abs=0.01)


# ---------------------------------------------------------------------------
# stop / hold_position
# ---------------------------------------------------------------------------

class TestStopAndHold:
    def test_stop_sets_zero_speed_and_hold(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "stop", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.target_speed_mps == 0.0
        assert d.hold_position is True

    def test_hold_position_sets_zero_speed_and_hold(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "hold_position", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.target_speed_mps == 0.0
        assert d.hold_position is True

    def test_stop_clears_chase(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "chase_actor", "start_time": 0.0, "target_actor_id": "other"},
            {"id": "c2", "action": "stop", "start_time": 1.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.5)
        assert d.chase_target_actor_id is None
        assert d.hold_position is True


# ---------------------------------------------------------------------------
# enable_autopilot / disable_autopilot
# ---------------------------------------------------------------------------

class TestAutopilot:
    def test_enable_autopilot(self):
        actor = _make_actor(autopilot=False, timeline=[
            {"id": "c1", "action": "enable_autopilot", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.autopilot_enabled is True

    def test_disable_autopilot(self):
        actor = _make_actor(autopilot=True, timeline=[
            {"id": "c1", "action": "disable_autopilot", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.autopilot_enabled is False

    def test_enable_then_disable(self):
        actor = _make_actor(autopilot=False, timeline=[
            {"id": "c1", "action": "enable_autopilot", "start_time": 0.0},
            {"id": "c2", "action": "disable_autopilot", "start_time": 1.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.5)
        assert d.autopilot_enabled is False

    def test_enable_clears_direct_control(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "chase_actor", "start_time": 0.0, "target_actor_id": "x"},
            {"id": "c2", "action": "enable_autopilot", "start_time": 1.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.5)
        assert d.chase_target_actor_id is None
        assert d.autopilot_enabled is True


# ---------------------------------------------------------------------------
# lane_change_left / lane_change_right
# ---------------------------------------------------------------------------

class TestLaneChange:
    def test_lane_change_left(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "lane_change_left", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.lane_change_direction == "left"

    def test_lane_change_right(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "lane_change_right", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.lane_change_direction == "right"

    def test_lane_change_not_reapplied(self):
        """Once a lane change clip has been applied, it must not set direction again."""
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "lane_change_left", "start_time": 0.0},
        ])
        state = _fresh_state()
        # First evaluation applies it
        d1 = _evaluate_timeline(actor, state, simulation_time=0.5)
        assert d1.lane_change_direction == "left"
        assert "c1" in state.applied_clips
        # Second evaluation -- same state -- should NOT set direction
        d2 = _evaluate_timeline(actor, state, simulation_time=1.0)
        assert d2.lane_change_direction is None

    def test_two_lane_changes_both_apply(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "lane_change_left", "start_time": 0.0},
            {"id": "c2", "action": "lane_change_right", "start_time": 1.0},
        ])
        state = _fresh_state()
        d1 = _evaluate_timeline(actor, state, simulation_time=0.5)
        assert d1.lane_change_direction == "left"
        d2 = _evaluate_timeline(actor, state, simulation_time=1.5)
        assert d2.lane_change_direction == "right"


# ---------------------------------------------------------------------------
# follow_route
# ---------------------------------------------------------------------------

class TestFollowRoute:
    def test_follow_route_enables_route_and_disables_autopilot(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "follow_route", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.route_follow_enabled is True
        assert d.autopilot_enabled is False
        assert d.hold_position is False

    def test_follow_route_with_speed(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "follow_route", "start_time": 0.0, "target_speed_kph": 40.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.target_speed_mps == pytest.approx(40.0 / 3.6, abs=0.01)

    def test_follow_route_clears_chase(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "chase_actor", "start_time": 0.0, "target_actor_id": "t"},
            {"id": "c2", "action": "follow_route", "start_time": 1.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.5)
        assert d.chase_target_actor_id is None
        assert d.route_follow_enabled is True


# ---------------------------------------------------------------------------
# chase_actor / ram_actor
# ---------------------------------------------------------------------------

class TestChaseAndRam:
    def test_chase_actor_sets_target(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "chase_actor", "start_time": 0.0, "target_actor_id": "victim", "following_distance_m": 12.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.chase_target_actor_id == "victim"
        assert d.follow_distance_m == 12.0
        assert d.autopilot_enabled is False

    def test_chase_clamps_follow_distance(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "chase_actor", "start_time": 0.0, "target_actor_id": "v", "following_distance_m": 2.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.follow_distance_m == 6.0  # clamped to min 6.0

    def test_chase_default_follow_distance(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "chase_actor", "start_time": 0.0, "target_actor_id": "v"},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.follow_distance_m == 10.0  # default

    def test_ram_actor_sets_target(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "ram_actor", "start_time": 0.0, "target_actor_id": "victim"},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.ram_target_actor_id == "victim"
        assert d.follow_distance_m == 0.0
        assert d.autopilot_enabled is False

    def test_ram_with_speed(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "ram_actor", "start_time": 0.0, "target_actor_id": "v", "target_speed_kph": 120.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.target_speed_mps == pytest.approx(120.0 / 3.6, abs=0.01)

    def test_chase_clears_route_follow(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "follow_route", "start_time": 0.0},
            {"id": "c2", "action": "chase_actor", "start_time": 1.0, "target_actor_id": "t"},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.5)
        assert d.route_follow_enabled is False
        assert d.chase_target_actor_id == "t"


# ---------------------------------------------------------------------------
# turn instructions
# ---------------------------------------------------------------------------

class TestTurnInstructions:
    def test_turn_left_sets_route_instruction(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "turn_left_at_next_intersection", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.route_instruction == "Left"

    def test_turn_right_sets_route_instruction(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "turn_right_at_next_intersection", "start_time": 0.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        assert d.route_instruction == "Right"

    def test_turn_not_reapplied(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "turn_left_at_next_intersection", "start_time": 0.0},
        ])
        state = _fresh_state()
        d1 = _evaluate_timeline(actor, state, simulation_time=0.5)
        assert d1.route_instruction == "Left"
        d2 = _evaluate_timeline(actor, state, simulation_time=1.0)
        assert d2.route_instruction is None


# ---------------------------------------------------------------------------
# Clip ordering and enabled flag
# ---------------------------------------------------------------------------

class TestClipOrdering:
    def test_later_clip_overrides_earlier(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 20.0},
            {"id": "c2", "action": "set_speed", "start_time": 1.0, "target_speed_kph": 80.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.5)
        assert d.target_speed_mps == pytest.approx(80.0 / 3.6, abs=0.01)

    def test_only_first_clip_active_before_second(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 20.0},
            {"id": "c2", "action": "set_speed", "start_time": 2.0, "target_speed_kph": 80.0},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.0)
        assert d.target_speed_mps == pytest.approx(20.0 / 3.6, abs=0.01)

    def test_disabled_clip_skipped(self):
        actor = _make_actor(speed_kph=50.0, timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 10.0, "enabled": False},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=0.5)
        # Clip is disabled, so actor default speed should remain
        assert d.target_speed_mps == pytest.approx(50.0 / 3.6, abs=0.01)

    def test_clips_sorted_by_start_time(self):
        """Clips added out of order should still be processed in start_time order."""
        actor = _make_actor(timeline=[
            {"id": "c2", "action": "set_speed", "start_time": 2.0, "target_speed_kph": 80.0},
            {"id": "c1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 20.0},
        ])
        # At t=1.0 only c1 should be active
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=1.0)
        assert d.target_speed_mps == pytest.approx(20.0 / 3.6, abs=0.01)


# ---------------------------------------------------------------------------
# _sorted_timeline helper
# ---------------------------------------------------------------------------

class TestSortedTimeline:
    def test_sorted_by_start_time(self):
        actor = _make_actor(timeline=[
            {"id": "b", "action": "stop", "start_time": 2.0},
            {"id": "a", "action": "stop", "start_time": 1.0},
        ])
        result = _sorted_timeline(actor)
        assert [c.id for c in result] == ["a", "b"]

    def test_stable_sort_by_id(self):
        actor = _make_actor(timeline=[
            {"id": "b", "action": "stop", "start_time": 0.0},
            {"id": "a", "action": "stop", "start_time": 0.0},
        ])
        result = _sorted_timeline(actor)
        assert [c.id for c in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# Complex multi-clip scenarios
# ---------------------------------------------------------------------------

class TestComplexScenarios:
    def test_speed_then_stop_then_resume(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "set_speed", "start_time": 0.0, "target_speed_kph": 60.0},
            {"id": "c2", "action": "stop", "start_time": 2.0},
            {"id": "c3", "action": "set_speed", "start_time": 4.0, "target_speed_kph": 40.0},
        ])
        state = _fresh_state()
        d1 = _evaluate_timeline(actor, state, simulation_time=1.0)
        assert d1.target_speed_mps == pytest.approx(60.0 / 3.6, abs=0.01)
        assert d1.hold_position is False

        d2 = _evaluate_timeline(actor, state, simulation_time=3.0)
        assert d2.target_speed_mps == 0.0
        assert d2.hold_position is True

        d3 = _evaluate_timeline(actor, state, simulation_time=5.0)
        assert d3.target_speed_mps == pytest.approx(40.0 / 3.6, abs=0.01)
        assert d3.hold_position is False

    def test_chase_then_ram_overrides(self):
        actor = _make_actor(timeline=[
            {"id": "c1", "action": "chase_actor", "start_time": 0.0, "target_actor_id": "a"},
            {"id": "c2", "action": "ram_actor", "start_time": 2.0, "target_actor_id": "a"},
        ])
        d = _evaluate_timeline(actor, _fresh_state(), simulation_time=3.0)
        assert d.chase_target_actor_id is None
        assert d.ram_target_actor_id == "a"
        assert d.follow_distance_m == 0.0
