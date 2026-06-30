"""Tests for goat_agent_1 P0 trace telemetry and P1 mandatory VISUAL_APPROACH."""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent_1 import (
    ConfTopoGOATAgent,
    LOCAL_SPIN_RESELECT_STEPS,
    SimpleNavPhase,
)
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import NodeType
from conftopo.core.instruction_graph import GoalNode, InstructionGraph


def _make_agent() -> ConfTopoGOATAgent:
    config = ConfTopoConfig()
    config.perception.backend = "clip_groundingdino"
    agent = ConfTopoGOATAgent(config)
    goal = GoalNode(target_object="rack", target_embedding=np.zeros(512, dtype=np.float32))
    agent.set_goal(InstructionGraph(goal_type="object_goal", goal_nodes=[goal]))
    agent.observe({
        "rgb": np.ones((64, 64, 3), dtype=np.uint8) * 128,
        "rgb_embed": np.random.randn(512).astype(np.float32),
        "position": np.zeros(3, dtype=np.float32),
        "heading": 0.0,
    })
    agent.update_memory()
    return agent


def test_step_telemetry_includes_required_fields():
    agent = _make_agent()
    plan = agent.plan()
    out = agent.act(plan)
    debug = out["sticky_debug"]
    for key in (
        "nav_phase",
        "phase_reason",
        "proposal_type",
        "proposal_source",
        "proposal_score",
        "stop_reason",
        "stop_bbox_area",
        "stop_centered",
        "stop_close",
        "active_anchor_id",
        "anchor_distance",
        "approach_steps",
    ):
        assert key in debug, f"missing telemetry field: {key}"


def test_scan_complete_enters_visual_approach_not_verify_stop():
    agent = _make_agent()
    agent._nav_phase = SimpleNavPhase.SCAN_TRACK
    agent._scan_turns_remaining = 1
    agent._scan_after_anchor_steps = 3
    agent._goal_local_step = 20
    agent._position = np.zeros(3, dtype=np.float32)
    anchor_id = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        label="rack",
        attributes={
            "semantic_role": "object_anchor",
            "anchor_waypoint_position": [0.0, 0.0, 0.0],
        },
    )
    agent._active_anchor_id = anchor_id
    agent._last_reached_anchor_id = anchor_id
    agent._cur_vlm_report = {
        "fresh": True,
        "goal_visible": True,
        "stop_candidate": True,
        "objects": [{
            "label": "rack",
            "bbox": [0.4, 0.4, 0.6, 0.7],
            "range_bin": "near",
            "visibility": "visible",
            "confidence": 0.9,
        }],
    }
    plan = {"target_node_id": None, "target_position": None, "candidate_ids": [], "scores": [], "sticky_debug": {}}
    agent.act(plan)
    assert agent._nav_phase == SimpleNavPhase.VISUAL_APPROACH


def test_cannot_stop_without_approach_requirements():
    agent = _make_agent()
    agent._nav_phase = SimpleNavPhase.VERIFY_STOP
    agent._goal_local_step = 20
    agent._last_reached_anchor_id = "x"
    agent._scan_after_anchor_steps = 4
    agent._approach_forward_count = 0
    agent._approach_travel_distance = 0.0
    agent._position = np.zeros(3, dtype=np.float32)
    anchor_id = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        label="rack",
        attributes={"semantic_role": "object_anchor", "anchor_waypoint_position": [0, 0, 0]},
    )
    agent._active_anchor_id = anchor_id
    agent._cur_vlm_report = {
        "fresh": True,
        "goal_visible": True,
        "stop_candidate": True,
        "objects": [{
            "label": "rack",
            "bbox": [0.4, 0.4, 0.6, 0.7],
            "range_bin": "near",
            "visibility": "visible",
            "confidence": 0.9,
        }],
    }
    agent._vlm_confirm_buffer = [True, True]
    decision = agent._evaluate_stop()
    assert not decision.can_stop
    assert decision.reason in {"approach_not_enough", "need_visual_approach"}


def test_approach_counters_preserved_across_scan_bounce():
    agent = _make_agent()
    agent._nav_phase = SimpleNavPhase.VISUAL_APPROACH
    agent._approach_forward_count = 2
    agent._approach_travel_distance = 0.6
    agent._active_anchor_id = "anchor_x"
    agent._last_reached_anchor_id = "anchor_x"

    agent._enter_phase(SimpleNavPhase.SCAN_TRACK, "not_centered")
    assert agent._approach_forward_count == 2
    assert agent._approach_travel_distance == 0.6

    agent._enter_phase(SimpleNavPhase.VISUAL_APPROACH, "bbox_too_small")
    assert agent._approach_forward_count == 2
    assert agent._approach_travel_distance == 0.6
    assert agent._approach_steps == 0
    assert agent._approach_lost_count == 0


def test_scan_complete_resets_approach_counters():
    agent = _make_agent()
    agent._nav_phase = SimpleNavPhase.SCAN_TRACK
    agent._approach_forward_count = 1
    agent._approach_travel_distance = 0.3

    agent._enter_phase(SimpleNavPhase.VISUAL_APPROACH, "scan_complete_approach")
    assert agent._approach_forward_count == 0
    assert agent._approach_travel_distance == 0.0


def test_label_matches_substring_and_canonical():
    agent = _make_agent()
    assert agent._label_matches_current_goal("wooden rack", {"canonical_label": "rack"})
    assert not agent._label_matches_current_goal("heater", {"canonical_label": "heater"})

    bed_goal = GoalNode(target_object="bed", target_embedding=np.zeros(512, dtype=np.float32))
    agent.set_new_goal(bed_goal)
    assert agent._label_matches_current_goal("bed frame", {"raw_label": "bed"})


def test_scan_turns_decrement_without_plan_reset():
    agent = _make_agent()
    wp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    anchor_id = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        label="rack",
        confidence=0.9,
        attributes={
            "semantic_role": "object_anchor",
            "anchor_confirmed": True,
            "anchor_waypoint_id": wp_id,
            "anchor_waypoint_position": [0.0, 0.0, 0.0],
            "promote_confirm_count": 2,
            "seen_count": 2,
        },
    )
    wp = agent.topo_map.get_node(wp_id)
    wp.attributes["goal_stop_object_id"] = anchor_id
    wp.attributes["goal_stop_object_label"] = "rack"
    agent._cur_vp_id = wp_id
    agent._position = np.zeros(3, dtype=np.float32)

    agent._begin_anchor_scan_session(anchor_id, "test")
    assert agent._scan_turns_remaining == 4

    agent.on_navigation_event(wp_id, "target_reached")
    assert agent._scan_turns_remaining == 4

    agent._nav_phase = SimpleNavPhase.SCAN_TRACK
    agent._scan_turns_remaining = 2
    agent._active_anchor_id = anchor_id
    agent._last_reached_anchor_id = anchor_id
    plan = agent.plan()
    assert plan.get("target_type") == "memory_exploit_current"
    assert agent._scan_turns_remaining == 2


def test_new_goal_blocks_previous_anchor_and_clears_waypoint_link():
    agent = _make_agent()
    rack_anchor = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        label="rack",
        attributes={
            "semantic_role": "object_anchor",
            "anchor_waypoint_position": [1.0, 0.0, 1.0],
        },
    )
    wp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        attributes={"goal_stop_object_id": rack_anchor, "goal_stop_object_label": "rack"},
    )

    bed_goal = GoalNode(target_object="bed", target_embedding=np.zeros(512, dtype=np.float32))
    agent.set_new_goal(bed_goal)

    assert agent._is_blocked_target(rack_anchor)
    wp = agent.topo_map.get_node(wp_id)
    assert "goal_stop_object_id" not in wp.attributes

    agent.on_navigation_event(wp_id, "target_reached")
    assert agent._nav_phase != SimpleNavPhase.SCAN_TRACK or agent._scan_turns_remaining == 0


def test_weak_vlm_hypothesis_does_not_trigger_local_scan():
    agent = _make_agent()
    wp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        label="rack",
        confidence=0.9,
        attributes={
            "semantic_role": "vlm_hypothesis",
            "anchor_waypoint_id": wp_id,
            "bbox": [0.75, 0.0, 0.99, 0.25],
            "seen_count": 1,
        },
    )
    agent._cur_vp_id = wp_id
    agent._position = np.zeros(3, dtype=np.float32)
    plan = agent.plan()
    assert plan.get("target_type") != "memory_exploit_current"
    assert agent._nav_phase != SimpleNavPhase.SCAN_TRACK


def test_recover_prefers_frontier_navigation():
    agent = _make_agent()
    agent.topo_map.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        attributes={"semantic_role": "frontier"},
    )
    agent._nav_phase = SimpleNavPhase.RECOVER
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_vp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    plan = agent.plan()
    out = agent.act(plan)
    assert plan.get("target_type") == "frontier_explore"
    assert out.get("action") == "navigate"
    assert out.get("target_node_id") is not None


def test_without_high_confidence_frontier_beats_visited():
    agent = _make_agent()
    agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
    )
    frontier_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        attributes={"semantic_role": "frontier"},
    )
    agent._position = np.zeros(3, dtype=np.float32)
    plan = agent.plan()
    assert plan.get("target_type") == "frontier_explore"
    assert plan.get("target_node_id") == frontier_id


def test_hypothesis_requires_distinct_observation_viewpoint():
    agent = _make_agent()
    wp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        label="rack",
        confidence=0.5,
        attributes={
            "semantic_role": "vlm_hypothesis",
            "anchor_waypoint_id": wp_id,
            "seen_count": 2,
        },
    )
    agent.topo_map.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([2.5, 0.0, 0.0], dtype=np.float32),
        attributes={"semantic_role": "frontier"},
    )
    agent._cur_vp_id = wp_id
    agent._position = np.zeros(3, dtype=np.float32)
    plan = agent.plan()
    assert plan.get("target_type") != "hypothesis_verify"


def test_idle_spin_releases_sticky_target():
    agent = _make_agent()
    frontier_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        attributes={"semantic_role": "frontier"},
    )
    stuck_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.2, 0.0, 0.1], dtype=np.float32),
    )
    agent._sticky_target_id = stuck_id
    agent._sticky_last_distance = 0.5
    agent._sticky_last_heading = 0.0
    agent._position = np.zeros(3, dtype=np.float32)
    agent._spin_idle_steps = LOCAL_SPIN_RESELECT_STEPS
    plan = agent.plan()
    assert plan.get("target_type") == "frontier_explore"
    assert agent._sticky_target_id != stuck_id
    assert agent._is_blocked_target(stuck_id)
