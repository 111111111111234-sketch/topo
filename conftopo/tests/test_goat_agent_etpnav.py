"""Unit tests for ETPNav GOAT agent navigation fixes."""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent_etpnav import (
    ConfTopoGOATAgentETPNav,
    ETPBBoxGrowthStopVerifier,
    ETPGoatConfig,
    ETPGraphNavigationPlanner,
    ETPTaskTelemetry,
    _telemetry_phase_key,
    _telemetry_proposal_source,
    _telemetry_stop_block_reason,
)
from conftopo.agents.goat_agent_new import GoalManager, NavPhase, ServoState
from conftopo.core.dynamic_topo_map import NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.core.instruction_graph import GoalProposal
from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.perception_report import PerceptionReport
from conftopo.agents.goat_agent_new import PerceptionPacket


def _packet(*, fresh: bool, cached: bool, goal_visible: bool, objects=None, confidence: float = 0.8):
    report = PerceptionReport(
        goal_visible=goal_visible,
        goal_match_confidence=confidence,
        target_visibility="clear",
        target_direction="center",
        objects=list(objects or []),
    )
    return PerceptionPacket(
        report=report,
        source="vlm" if fresh else "cached_vlm",
        fresh_vlm=fresh,
        cached_vlm=cached,
        vlm_mode="confirm",
    )


def test_etp_stop_verifier_accepts_fresh_confirm_near_goal():
    cfg = ETPGoatConfig()
    state = ServoState()
    state.confirm_buffer = [True, True, True]
    state.centered_buffer = [True, True, True]
    state.close_buffer = [True]
    state.forward_action_count = 3
    state.approach_travel_distance = 1.0
    verifier = ETPBBoxGrowthStopVerifier(cfg, state)
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater")
    packet = _packet(
        fresh=True,
        cached=False,
        goal_visible=True,
        objects=[ObjectObservation(label="heater", bbox=[0.2, 0.2, 0.7, 0.7], confidence=0.9, range_bin="close", bearing="center")],
    )
    decision = verifier.can_stop(
        packet,
        goal,
        np.zeros(3, dtype=np.float32),
        {
            "bbox_has_meaningful_growth": False,
            "bbox_plateau": True,
            "retreating": False,
            "relative_progress": "closer",
            "range_bin": "close",
        },
    )
    assert decision.should_stop is True
    assert decision.reason == "etp_bbox_growth_stop"


def test_etp_stop_verifier_rejects_loose_cached_confirm():
    cfg = ETPGoatConfig()
    state = ServoState()
    state.confirm_buffer = [True, True]
    state.centered_buffer = [True, True]
    state.forward_action_count = 3
    state.approach_travel_distance = 1.0
    verifier = ETPBBoxGrowthStopVerifier(cfg, state)
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater")
    packet = _packet(
        fresh=False,
        cached=True,
        goal_visible=True,
        objects=[ObjectObservation(label="heater", bbox=[0.2, 0.2, 0.7, 0.7], confidence=0.9, range_bin="close", bearing="center")],
    )
    packet.vlm_cache_age = 5
    packet.vlm_position_delta = 0.0
    packet.vlm_heading_delta = 0.0
    decision = verifier.can_stop(
        packet,
        goal,
        np.zeros(3, dtype=np.float32),
        {
            "bbox_has_meaningful_growth": False,
            "bbox_plateau": True,
            "retreating": False,
            "relative_progress": "closer",
            "range_bin": "close",
        },
    )
    assert decision.should_stop is False
    assert decision.reason == "no_fresh_target"


def test_etp_stop_verifier_no_growth_close_ok():
    cfg = ETPGoatConfig()
    state = ServoState()
    state.confirm_buffer = [True, True, True]
    state.centered_buffer = [True, True, True]
    state.close_buffer = [True, False, True]
    state.forward_action_count = 3
    state.approach_travel_distance = 0.8
    verifier = ETPBBoxGrowthStopVerifier(cfg, state)
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater")
    packet = _packet(
        fresh=True,
        cached=False,
        goal_visible=True,
        objects=[ObjectObservation(label="heater", bbox=[0.2, 0.2, 0.7, 0.7], confidence=0.9, range_bin="close", bearing="center")],
    )
    decision = verifier.can_stop(
        packet,
        goal,
        np.zeros(3, dtype=np.float32),
        {
            "bbox_has_meaningful_growth": False,
            "bbox_plateau": False,
            "retreating": False,
            "relative_progress": "closer",
            "range_bin": "close",
        },
    )
    assert decision.should_stop is True


def test_etp_planner_prefers_object_memory_during_reroute_window():
    cfg = ETPGoatConfig()
    planner = ETPGraphNavigationPlanner(cfg)
    planner.prefer_goal_anchor_reroute = True
    proposals = [
        GoalProposal(
            goal_id="g0",
            candidate_node_id="way_1",
            candidate_type="waypoint_candidate",
            score=5.0,
            source="ghost_candidate",
            status="active",
        ),
        GoalProposal(
            goal_id="g0",
            candidate_node_id="obj_1",
            candidate_type="object_anchor",
            score=1.0,
            source="object_memory",
            status="active",
        ),
    ]
    best = planner.select_best_proposal(proposals)
    assert best is not None
    assert best.candidate_node_id == "obj_1"


def test_etp_track_timeout_sets_reroute_window():
    agent = ConfTopoGOATAgentETPNav()
    agent.topo_map.step()
    agent._transition(NavPhase.TRACK_TARGET, "test")
    agent.phase_enter_step = 0
    agent.new_config.track_timeout_steps = 1
    agent.topo_map.step()
    decision = agent._plan_track_phase(
        agent._last_nav_target,
        [],
        [],
    )
    assert decision.reason == "etp_track_timeout"
    assert agent._etp_prefer_goal_anchor_until > agent.topo_map.current_step


def test_etp_track_buffer_accepts_cached_confirm():
    agent = ConfTopoGOATAgentETPNav()
    agent.goal_manager.current_goal = GoalNode(target_object="heater")
    agent._transition(NavPhase.TRACK_TARGET, "test")
    agent._last_packet = _packet(
        fresh=False,
        cached=True,
        goal_visible=True,
        objects=[ObjectObservation(label="heater", bbox=[0.3, 0.3, 0.6, 0.6], confidence=0.9)],
        confidence=0.9,
    )
    agent._last_packet.step_id = agent.topo_map.current_step
    agent._update_track_buffer()
    assert len(agent.servo_state.track_buffer) == 1
    assert agent.servo_state.track_buffer[0]["visible"] is True


def test_etp_task_telemetry_mappings():
    assert _telemetry_phase_key(NavPhase.GLOBAL_SEARCH) == "SEARCH"
    assert _telemetry_phase_key(NavPhase.TRACK_TARGET) == "TRACK"
    assert _telemetry_phase_key(NavPhase.LOCAL_VISUAL_APPROACH) == "APPROACH"
    assert _telemetry_phase_key(NavPhase.STOP_VERIFY) == "STOP_VERIFY"
    assert _telemetry_phase_key(NavPhase.STOP) is None

    ghost = GoalProposal(
        goal_id="g0",
        candidate_node_id="cand_1",
        candidate_type="waypoint_candidate",
        target_position=np.zeros(3, dtype=np.float32),
        source="ghost_candidate",
    )
    assert _telemetry_proposal_source(ghost) == "ghost"

    assert _telemetry_stop_block_reason("no_fresh_target") == "no_fresh_target"
    assert _telemetry_stop_block_reason("target_not_centered") == "not_centered"
    assert _telemetry_stop_block_reason("need_growth_or_strong_multiframe") == "no_growth"
    assert _telemetry_stop_block_reason("not_approaching", {"retreating": True}) == "retreating"

    tel = ETPTaskTelemetry()
    tel.record_phase(NavPhase.TRACK_TARGET)
    tel.record_action("move_forward")
    tel.record_action("turn_left")
    tel.record_proposal(ghost)
    tel.record_stop_block("no_fresh_target")
    tel.record_track_buffer_sample(
        {"visible": True, "centered": True, "bbox_valid": True},
        fresh=True,
        cached=False,
    )
    snap = tel.snapshot()
    assert snap["phase_counts"]["TRACK"] == 1
    assert snap["action_counts"]["forward"] == 1
    assert snap["action_counts"]["turn_left"] == 1
    assert snap["proposal_source_counts"]["ghost"] == 1
    assert snap["stop_block_reasons"]["no_fresh_target"] == 1
    assert snap["track_buffer_stats"]["visible_count"] == 1
    assert snap["track_buffer_stats"]["fresh_count"] == 1


def test_etp_ghost_arrival_scan_starts_on_candidate_reached():
    agent = ConfTopoGOATAgentETPNav()
    agent.new_config.ghost_arrival_scan_turns = 3
    node_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_CANDIDATE,
        position=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        confidence=0.5,
        label="",
        attributes={"semantic_role": "ghost_candidate"},
    )
    out = agent.on_navigation_event(node_id, "target_reached")
    assert out["action"] == "candidate_promoted"
    assert agent._ghost_scan_turns_left == 3


def test_etp_explore_stuck_expands_ghost_rays():
    agent = ConfTopoGOATAgentETPNav()
    agent.new_config.explore_stuck_steps = 2
    agent._transition(NavPhase.GLOBAL_SEARCH, "test")
    agent._explore_last_position = np.zeros(3, dtype=np.float32)
    agent._position = np.zeros(3, dtype=np.float32)
    agent._last_packet = _packet(fresh=False, cached=False, goal_visible=False)
    agent._last_memory = type("M", (), {
        "goal_anchor_written": False,
        "object_merge_count_this_step": 0,
        "written_node_ids": [],
    })()
    agent._update_explore_stuck_state()
    agent._update_explore_stuck_state()
    assert agent.memory_writer.ghost_escalation_level == 1
    assert agent._explore_pending_turn is True


def test_etp_agent_resets_task_telemetry_on_new_goal():
    agent = ConfTopoGOATAgentETPNav()
    agent.task_telemetry.record_phase(NavPhase.TRACK_TARGET)
    agent.set_new_goal(GoalNode(target_object="bed"))
    assert agent.task_telemetry.snapshot()["phase_counts"]["TRACK"] == 0
