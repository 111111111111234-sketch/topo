"""Tests for clean ETPNav generic navigation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from conftopo.agents.goat_agent_etpnav_clean import (
    CleanETPGoatConfig,
    CleanETPPlanner,
    ConfTopoGOATAgentCleanETPNav,
    ETPBBoxGrowthStopVerifier,
    classify_goal_evidence,
)
from conftopo.agents.goat_agent_new import GoalManager, NavPhase, NavTarget, ServoState, StopDecision
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.instruction_graph import GoalNode, GoalProposal


def test_goal_switch_resets_local_servo_state():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.set_new_goal(GoalNode(target_object="rack"))
    agent.servo_state.forward_action_count = 5
    agent.servo_state.active_anchor_id = "obj_old"
    agent.servo_state.confirm_buffer = [True, True]
    agent.nav_phase = NavPhase.LOCAL_VISUAL_APPROACH
    agent.set_new_goal(GoalNode(target_object="bed"))
    assert agent.nav_phase == NavPhase.GLOBAL_SEARCH
    assert agent.servo_state.active_anchor_id is None
    assert agent.servo_state.forward_action_count == 0
    assert agent.servo_state.confirm_buffer == []
    assert agent._last_selected_goal_proposal is None


def test_servo_blocked_during_goal_cooldown():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.new_config.goal_servo_cooldown_steps = 5
    agent.set_new_goal(GoalNode(target_object="bed"))
    assert agent._servo_allowed_for_current_goal() is False


def _goal_manager(target: str) -> GoalManager:
    gm = GoalManager()
    gm.current_goal = GoalNode(target_object=target)
    return gm


def test_clean_planner_prefers_resolved_object_anchor_over_ghost():
    planner = CleanETPPlanner(CleanETPGoatConfig())
    topo = DynamicTopoMap()
    topo.step()
    goal = _goal_manager("bed")
    obj_id = topo.add_node(
        NodeType.OBJECT,
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        label="bed",
        attributes={
            "semantic_role": "object_anchor",
            "anchor_waypoint_id": None,
            "multi_view_count": 2,
        },
    )
    proposals = [
        GoalProposal(
            goal_id="bed",
            candidate_node_id="way_1",
            candidate_type="waypoint_candidate",
            score=8.0,
            source="ghost_candidate",
            status="active",
        ),
        GoalProposal(
            goal_id="bed",
            candidate_node_id=obj_id,
            candidate_type="object_approach",
            score=2.0,
            source="object_memory",
            status="active",
        ),
    ]
    best = planner.select_best_proposal(proposals, topo, goal, np.zeros(3), None)
    assert best is not None
    assert best.candidate_node_id == obj_id
    assert best.candidate_type == "object_anchor"


def test_wrong_label_anchor_blocks_track_ready():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.goal_manager.current_goal = GoalNode(target_object="bed")
    agent._goal_servo_unlock_step = 0
    agent.topo_map.step()
    oid = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        confidence=0.9,
        label="rack",
        attributes={"semantic_role": "object_anchor"},
    )
    agent.servo_state.active_anchor_id = oid
    agent.servo_state.track_buffer = [
        {"visible": True, "bbox_valid": True},
        {"visible": True, "bbox_valid": True},
    ]
    assert agent._track_ready_for_approach() is False


def test_far_visible_does_not_allow_stop():
    cfg = CleanETPGoatConfig()
    verifier = ETPBBoxGrowthStopVerifier(cfg, ServoState())
    goal = _goal_manager("bed")

    @dataclass
    class Obj:
        label: str = "bed"
        bbox: tuple = (0.45, 0.45, 0.48, 0.48)
        bearing: str = "center"
        range_bin: str = "far"
        confidence: float = 0.9

    packet = SimpleNamespace(
        fresh_vlm=True,
        cached_vlm=False,
        step_id=1,
        report=SimpleNamespace(
            goal_visible=True,
            goal_match_confidence=0.9,
            target_direction="center",
            target_visibility="clear",
            stop_candidate=False,
            objects=[Obj()],
        ),
    )
    assert classify_goal_evidence(packet, goal) == "far_visible"
    stop = verifier.can_stop(packet, goal, np.zeros(3), {})
    assert stop.should_stop is False
    assert stop.reason == "far_visible_no_stop"


def test_approach_transitions_to_verify_not_stop():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.set_new_goal(GoalNode(target_object="bed"))
    agent._goal_servo_unlock_step = 0
    agent.nav_phase = NavPhase.LOCAL_VISUAL_APPROACH
    agent.topo_map.step()
    oid = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        confidence=0.9,
        label="bed",
        attributes={"semantic_role": "object_anchor"},
    )
    agent.servo_state.active_anchor_id = oid
    agent.servo_state.track_buffer = [
        {"visible": True, "bbox_valid": True},
        {"visible": True, "bbox_valid": True},
    ]
    agent._last_packet = SimpleNamespace(
        fresh_vlm=True,
        cached_vlm=False,
        step_id=agent.topo_map.current_step,
        report=SimpleNamespace(
            goal_visible=True,
            goal_match_confidence=0.95,
            target_direction="center",
            target_visibility="clear",
            stop_candidate=True,
            objects=[],
        ),
    )
    servo_evidence = {"range_bin": "close", "bbox_has_meaningful_growth": True}
    servo_action = MagicMock(action="hold", next_phase=NavPhase.LOCAL_VISUAL_APPROACH, reason="hold")
    stop_decision = StopDecision(True, True, True, True, "etp_bbox_growth_stop")

    with patch.object(agent.local_servo, "update_evidence", return_value=servo_evidence):
        with patch.object(agent.local_servo, "act", return_value=servo_action):
            with patch.object(
                type(agent.stop_verifier),
                "is_near_stop_evidence",
                return_value=True,
            ):
                with patch.object(
                    type(agent.stop_verifier),
                    "is_verify_candidate",
                    return_value=True,
                ):
                    with patch.object(
                        type(agent.stop_verifier),
                        "can_stop",
                        return_value=stop_decision,
                    ):
                        decision = agent._plan_servo_phase(agent._last_nav_target, [], [])

    assert agent.nav_phase == NavPhase.STOP_VERIFY
    assert decision is not None
    assert decision.action != "stop"


def test_goal_evidence_buffer_records_outside_track_phase():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.set_new_goal(GoalNode(target_object="rack"))
    agent.nav_phase = NavPhase.GLOBAL_SEARCH
    agent.topo_map.step()

    @dataclass
    class Obj:
        label: str = "rack"
        bbox: tuple = (0.4, 0.4, 0.5, 0.5)
        bearing: str = "center"
        range_bin: str = "medium"
        confidence: float = 0.8

    agent._last_packet = SimpleNamespace(
        fresh_vlm=True,
        cached_vlm=False,
        step_id=agent.topo_map.current_step,
        report=SimpleNamespace(
            goal_visible=True,
            goal_match_confidence=0.8,
            target_direction="center",
            target_visibility="visible",
            stop_candidate=False,
            objects=[Obj()],
        ),
    )
    agent._update_goal_evidence_buffer()
    assert len(agent.servo_state.track_buffer) == 1
    assert agent.servo_state.track_buffer[0]["visible"] is True
    assert agent.servo_state.track_buffer[0]["evidence_type"] == "far_visible"


def test_far_from_anchor_forces_route_phase():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.set_new_goal(GoalNode(target_object="rack"))
    agent.topo_map.step()
    agent._position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    wp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([8.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        label="wp",
    )
    oid = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.array([8.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        label="rack",
        attributes={
            "semantic_role": "object_anchor",
            "anchor_waypoint_id": wp_id,
            "anchor_waypoint_position": [8.0, 0.0, 0.0],
        },
    )
    agent.nav_phase = NavPhase.LOCAL_VISUAL_APPROACH
    agent.servo_state.active_anchor_id = oid
    nav_target = NavTarget(
        target_position=np.array([8.0, 0.0, 0.0], dtype=np.float32),
        target_node_id=oid,
        target_type="object_anchor",
        reason="route_to_object_anchor_waypoint",
        expected_phase_after_reach=NavPhase.TRACK_TARGET,
    )
    decision = agent._enforce_object_anchor_route(nav_target, [], [])
    assert decision is not None
    assert decision.action == "navigate"
    assert agent.nav_phase == NavPhase.ROUTE_TO_OBJECT_ANCHOR


def test_within_track_radius_skips_forced_route():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.set_new_goal(GoalNode(target_object="heater"))
    agent.topo_map.step()
    agent._position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    wp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.8, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        label="wp",
    )
    oid = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.array([0.9, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        label="heater",
        attributes={
            "semantic_role": "object_anchor",
            "anchor_waypoint_id": wp_id,
            "anchor_waypoint_position": [0.8, 0.0, 0.0],
        },
    )
    agent.nav_phase = NavPhase.TRACK_TARGET
    agent.servo_state.active_anchor_id = oid
    nav_target = NavTarget(
        target_position=np.array([0.8, 0.0, 0.0], dtype=np.float32),
        target_node_id=oid,
        target_type="object_anchor",
        reason="route_to_object_anchor_waypoint",
        expected_phase_after_reach=NavPhase.TRACK_TARGET,
    )
    assert agent._enforce_object_anchor_route(nav_target, [], []) is None
    assert agent.nav_phase == NavPhase.TRACK_TARGET


def test_track_ready_without_centered():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.goal_manager.current_goal = GoalNode(target_object="heater")
    agent._goal_servo_unlock_step = 0
    agent.nav_phase = NavPhase.TRACK_TARGET
    agent.topo_map.step()
    oid = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.zeros(3, dtype=np.float32),
        confidence=0.9,
        label="heater",
        attributes={"semantic_role": "object_anchor"},
    )
    agent.servo_state.active_anchor_id = oid
    agent.servo_state.track_buffer = [
        {"visible": True, "bbox_valid": True, "centered": False},
        {"visible": True, "bbox_valid": True, "centered": False},
    ]
    assert agent._track_ready_for_approach() is True


def test_stop_verify_cooldown_blocks_reentry():
    agent = ConfTopoGOATAgentCleanETPNav()
    agent.set_new_goal(GoalNode(target_object="bed"))
    agent._goal_servo_unlock_step = 0
    agent.nav_phase = NavPhase.LOCAL_VISUAL_APPROACH
    agent.topo_map.step()
    agent._stop_verify_cooldown_until = agent.topo_map.current_step + 3
    assert agent._stop_verify_entry_allowed(True, True, {}) is False
