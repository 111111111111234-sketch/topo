"""Focused tests for the VLM GOAT agent control loop."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.agents.goat_agent_new import (
    GoatAgent,
    GoalManager,
    LocalVisualServo,
    MemoryWriter,
    NavPhase,
    NavigationPlanner,
    NewGoatConfig,
    PerceptionPacket,
    RecoveryManager,
    ServoState,
    StopVerifier,
    StructurePlanner,
)
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.perception_report import PerceptionReport


def test_visual_servo_breaks_left_right_alignment_cycle():
    state = ServoState(active_anchor_id="obj_1", entry_step=1)
    servo = LocalVisualServo(NewGoatConfig(), state)
    evidence = {
        "goal_visible": True,
        "effective_bbox_area": 0.08,
        "range_bin": "near",
    }

    first = servo.act({**evidence, "bearing": "right"}, step=2)
    second = servo.act({**evidence, "bearing": "left"}, step=3)
    third = servo.act({**evidence, "bearing": "right"}, step=4)

    assert first.action == "turn_right"
    assert second.action == "turn_left"
    assert third.action == "move_forward"
    assert third.reason == "servo_alignment_flip_advance"
    assert state.forward_action_count == 1
    assert state.alignment_flip_count == 2
    assert state.alignment_break_count == 1


def test_visual_servo_keeps_turning_when_bearing_does_not_cross_center():
    state = ServoState(active_anchor_id="obj_1", entry_step=1)
    servo = LocalVisualServo(NewGoatConfig(), state)

    first = servo.act({"bearing": "left"}, step=2)
    second = servo.act({"bearing": "left"}, step=3)

    assert first.action == "turn_left"
    assert second.action == "turn_left"
    assert state.forward_action_count == 0


def test_visual_servo_does_not_advance_for_far_or_tiny_target():
    for evidence in (
        {"goal_visible": True, "effective_bbox_area": 0.08, "range_bin": "far"},
        {"goal_visible": True, "effective_bbox_area": 0.02, "range_bin": "near"},
    ):
        state = ServoState(active_anchor_id="obj_1", entry_step=1)
        servo = LocalVisualServo(NewGoatConfig(), state)
        actions = [
            servo.act({**evidence, "bearing": bearing}, step=step).action
            for step, bearing in enumerate(("right", "left", "right"), start=2)
        ]
        assert actions == ["turn_right", "turn_left", "turn_right"]
        assert state.alignment_break_count == 0


def test_waypoints_are_fixed_samples_and_form_loop_closures():
    cfg = NewGoatConfig()
    topo = DynamicTopoMap()
    writer = MemoryWriter(cfg)
    embed = np.ones(4, dtype=np.float32)

    first = writer._write_waypoint(topo, np.array([0.0, 0.0, 0.0]), 0.0, embed)
    reused = writer._write_waypoint(topo, np.array([0.25, 0.0, 0.0]), 0.0, embed)
    second = writer._write_waypoint(topo, np.array([0.50, 0.0, 0.0]), 0.0, embed)

    assert reused == first
    assert second != first
    assert np.allclose(topo.get_node(first).position, [0.0, 0.0, 0.0])
    assert np.allclose(topo.get_node(second).position, [0.50, 0.0, 0.0])
    assert topo.graph.has_edge(first, second)

    loop = writer._write_waypoint(topo, np.array([0.10, 0.0, 0.0]), 0.0, embed)
    assert loop == first
    assert writer.prev_vp_id == second
    assert topo.graph.has_edge(second, first)
    assert np.allclose(topo.get_node(first).position, [0.0, 0.0, 0.0])


def test_anchor_reach_uses_fixed_position_not_waypoint_id():
    cfg = NewGoatConfig()
    topo = DynamicTopoMap()
    anchor_id = topo.add_node(
        NodeType.OBJECT,
        position=np.array([0.0, 0.0, 0.0]),
        label="wardrobe",
        attributes={
            "semantic_role": "object_anchor",
            "anchor_waypoint_id": "way_1",
            "anchor_waypoint_position": [0.0, 0.0, 0.0],
        },
    )
    anchor = topo.get_node(anchor_id)
    planner = NavigationPlanner(cfg)

    assert not planner.at_anchor_waypoint(
        anchor, np.array([2.0, 0.0, 0.0]), cur_vp_id="way_1",
    )
    assert planner.at_anchor_waypoint(
        anchor, np.array([0.5, 0.0, 0.0]), cur_vp_id="different_waypoint",
    )


def _visual_stop_fixture(bbox):
    cfg = NewGoatConfig()
    state = ServoState(
        active_anchor_id="obj_1",
        best_bbox_area=0.36,
        best_stop_pose=np.zeros(3, dtype=np.float32),
        visual_advance_steps=2,
        forward_action_count=3,
        approach_travel_distance=1.0,
        confirm_buffer=[True, True, True],
        stop_buffer=[True, True, True],
        plateau_forward_count=2,
        aligned_at_peak=True,
    )
    report = PerceptionReport(
        objects=[ObjectObservation(
            label="wardrobe",
            bbox=bbox,
            confidence=0.9,
            bearing="center",
            range_bin="near",
            visible=True,
        )],
        goal_visible=True,
        source="vlm",
        is_full=True,
    )
    packet = PerceptionPacket(
        report=report,
        source="vlm",
        fresh_vlm=True,
        vlm_mode="confirm",
        step_id=10,
    )
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="wardrobe")
    evidence = {
        "bbox_plateau": True,
        "retreating": False,
        "range_bin": "near",
    }
    return StopVerifier(cfg, state), packet, goal, evidence


def test_visual_stop_does_not_require_gt_instance_distance():
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.5, 0.4])

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert decision.should_stop
    assert decision.reason == "visual_confirmed_stop"


def test_visual_stop_rejects_insufficient_current_bbox():
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.3, 0.3])

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert not decision.should_stop
    assert decision.reason == "layer3_fresh_vlm_not_confirmed"


def test_structure_planner_respects_blacklist_expiry():
    topo = DynamicTopoMap()
    room_id = topo.add_node(
        NodeType.ROOM,
        position=np.zeros(3),
        label="bedroom",
        confidence=0.9,
        attributes={"blacklisted_until": 5},
    )
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="wardrobe", room_prior=["bedroom"])
    planner = StructurePlanner()

    topo._current_step = 5
    assert planner.select(topo, goal, np.zeros(3)).node_id is None

    topo._current_step = 6
    assert planner.select(topo, goal, np.zeros(3)).node_id == room_id


def test_frontier_navigation_remains_global_search_with_structure_prior():
    agent = GoatAgent(ConfTopoConfig())
    agent.set_new_goal(GoalNode(target_object="wardrobe", room_prior=["bedroom"]))
    agent._position = np.zeros(3, dtype=np.float32)
    agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0]),
        label="bedroom",
        confidence=0.9,
        attributes={"semantic_role": "room_summary"},
    )
    frontier_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([2.0, 0.0, 0.0]),
        confidence=0.5,
    )

    decision = agent.plan()

    assert decision.target_node_id == frontier_id
    assert decision.target_type == "frontier"
    assert agent.nav_phase == NavPhase.GLOBAL_SEARCH


def test_recovery_clears_stale_no_progress_after_real_movement():
    recovery = RecoveryManager(NewGoatConfig())
    recovery.recovery_reason = "no_progress"
    recovery.recent_positions = [np.zeros(3, dtype=np.float32)]

    triggered = recovery.note_position(np.array([0.25, 0.0, 0.0]))

    assert not triggered
    assert recovery.recovery_reason == ""
    assert len(recovery.recent_positions) == 1


def test_agent_ignores_gt_instance_distance_observation():
    agent = GoatAgent(ConfTopoConfig())
    agent.observe({
        "position": [0.0, 0.0, 0.0],
        "heading": 0.0,
        "instance_distance": 0.1,
    })

    assert not hasattr(agent, "_instance_distance")
