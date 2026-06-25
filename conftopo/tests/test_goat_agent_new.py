"""Focused tests for the VLM GOAT agent control loop."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.agents import GoatAgent
from conftopo.agents.goat_agent_new import (
    GoalManager,
    GoalProposal,
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
    StructureTarget,
    _bbox_growth_metrics,
)
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import FakeVLMBackend
from conftopo.perception.vlm_perceiver import VLMPerceiver
from conftopo.perception.vlm_prompts import build_user_prompt, parse_vlm_json


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


def _visual_stop_fixture(bbox, range_bin="near"):
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
            range_bin=range_bin,
            visible=True,
        )],
        goal_visible=True,
        goal_match_confidence=0.9,
        target_direction="center",
        target_visibility="clear",
        apparent_scale="large",
        relative_progress="unchanged",
        stop_candidate=True,
        recommended_action="stop_candidate",
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
        "range_bin": range_bin,
    }
    return StopVerifier(cfg, state), packet, goal, evidence


def test_visual_stop_does_not_require_gt_instance_distance():
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.5, 0.4])

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert decision.should_stop
    assert decision.reason == "visual_confirmed_stop"


def test_visual_stop_accepts_bbox_growth_without_absolute_threshold():
    cfg = NewGoatConfig()
    state = ServoState(
        active_anchor_id="obj_1",
        best_bbox_area=0.08,
        servo_entry_bbox_baseline=0.04,
        best_stop_pose=np.zeros(3, dtype=np.float32),
        visual_advance_steps=2,
        forward_action_count=2,
        approach_travel_distance=0.6,
        confirm_buffer=[True, True, True],
        stop_buffer=[True, True, True],
        plateau_forward_count=2,
        aligned_at_peak=True,
        bbox_history=[0.04, 0.05, 0.07, 0.08],
    )
    report = PerceptionReport(
        objects=[ObjectObservation(
            label="wardrobe",
            bbox=[0.0, 0.0, 0.35, 0.25],
            confidence=0.9,
            bearing="center",
            range_bin="near",
            visible=True,
        )],
        goal_visible=True,
        goal_match_confidence=0.9,
        target_direction="center",
        target_visibility="clear",
        apparent_scale="medium",
        relative_progress="unchanged",
        stop_candidate=True,
        recommended_action="stop_candidate",
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
    growth = _bbox_growth_metrics(state, cfg)
    evidence = {
        "bbox_plateau": True,
        "retreating": False,
        "range_bin": "near",
        "bbox_has_meaningful_growth": growth["effective_growth"] >= cfg.bbox_min_growth_ratio,
    }
    decision = StopVerifier(cfg, state).can_stop(packet, goal, np.zeros(3), evidence)
    assert decision.should_stop
    assert decision.layer2


def test_visual_stop_rejects_insufficient_current_bbox():
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.3, 0.3])

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert not decision.should_stop
    assert decision.reason == "layer3_fresh_vlm_not_confirmed"


def test_visual_stop_requires_vlm_stop_candidate():
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.5, 0.4])
    packet.report.stop_candidate = False

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert not decision.should_stop
    assert decision.reason == "layer3_fresh_vlm_not_confirmed"


def test_visual_stop_rejects_short_approach_without_stop_candidate():
    """If approach_travel_distance is very short, stop_candidate must be True."""
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.5, 0.4])
    verifier.state.approach_travel_distance = 0.5
    packet.report.stop_candidate = False
    evidence["bbox_plateau"] = True

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert not decision.should_stop


def test_visual_stop_rejects_tiny_fresh_bbox():
    """A very small fresh bbox should block STOP even if other conditions pass."""
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.15, 0.15])

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert not decision.should_stop


def test_visual_stop_rejects_target_getting_farther():
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.5, 0.4])
    evidence["relative_progress"] = "farther"

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert not decision.should_stop
    assert decision.reason == "layer3_target_getting_farther"


def test_visual_stop_does_not_use_range_as_a_hard_veto():
    for range_bin in ("mid", "far", "unknown"):
        verifier, packet, goal, evidence = _visual_stop_fixture(
            [0.0, 0.0, 0.5, 0.4],
            range_bin=range_bin,
        )

        decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

        assert decision.should_stop
        assert decision.reason == "visual_confirmed_stop"


def test_vlm_parser_preserves_close_range_bins():
    for range_bin in ("close", "very_near", "near"):
        parsed = parse_vlm_json(
            '{"room":{"label":"bedroom","confidence":0.9},'
            '"objects":[{"label":"wardrobe","bbox":[0.2,0.1,0.8,0.9],'
            f'"visible":true,"bearing":"center","range":"{range_bin}",'
            '"confidence":0.9}],"goal_visible":true}'
        )

        assert parsed["objects"][0]["range"] == range_bin


def test_vlm_parser_preserves_relative_control_fields():
    parsed = parse_vlm_json(
        '{"room":{"label":"bedroom","confidence":0.9},'
        '"objects":[{"label":"wardrobe","bbox":[0.2,0.1,0.8,0.9],'
        '"visible":true,"bearing":"center","range":"mid","confidence":0.9}],'
        '"goal_visible":true,"goal_match_confidence":0.88,'
        '"target_direction":"center","target_visibility":"clear",'
        '"apparent_scale":"large","relative_progress":"closer",'
        '"stop_candidate":true,"recommended_action":"hold_and_verify"}'
    )

    assert parsed["goal_match_confidence"] == 0.88
    assert parsed["target_direction"] == "center"
    assert parsed["target_visibility"] == "clear"
    assert parsed["relative_progress"] == "closer"
    assert parsed["stop_candidate"] is True
    assert parsed["recommended_action"] == "hold_and_verify"


def test_confirm_prompt_describes_previous_frame_and_action_history():
    prompt = build_user_prompt(
        "wardrobe",
        mode="confirm",
        has_previous_image=True,
        action_history=["turn_left", "move_forward"],
    )

    assert "PREVIOUS reference image" in prompt
    assert "turn_left -> move_forward" in prompt


def test_confirm_without_previous_frame_forces_uncertain_progress():
    backend = FakeVLMBackend({"relative_progress": "closer"})
    report = VLMPerceiver(backend).perceive(
        np.zeros((8, 8, 3), dtype=np.uint8),
        "wardrobe",
        mode="confirm",
    )

    assert report.relative_progress == "uncertain"


def test_visual_stop_requires_actual_approach_distance():
    verifier, packet, goal, evidence = _visual_stop_fixture([0.0, 0.0, 0.5, 0.4])
    verifier.state.approach_travel_distance = 0.75

    decision = verifier.can_stop(packet, goal, np.zeros(3), evidence)

    assert not decision.should_stop
    assert decision.reason == "layer2_approach_incomplete"


def test_target_bearing_observations_triangulate_object_and_approach():
    cfg = NewGoatConfig()
    topo = DynamicTopoMap()
    writer = MemoryWriter(cfg)
    node_id = topo.add_node(
        NodeType.OBJECT,
        position=np.array([0.0, 0.0, 0.0]),
        label="wardrobe",
        confidence=0.8,
        attributes={"semantic_role": "object_anchor"},
    )
    node = topo.get_node(node_id)
    centered = ObjectObservation(
        label="wardrobe",
        bbox=[0.4, 0.2, 0.6, 0.8],
        confidence=0.9,
        bearing="center",
        range_bin="near",
        visible=True,
    )

    writer._record_target_geometry(
        topo,
        node,
        centered,
        np.array([0.0, 0.0, 0.0]),
        -np.pi / 2.0,
        "way_1",
    )
    assert "estimated_object_position" not in node.attributes

    writer._record_target_geometry(
        topo,
        node,
        centered,
        np.array([1.0, 0.0, -1.0]),
        np.pi,
        "way_2",
    )

    assert np.allclose(
        node.attributes["estimated_object_position"],
        [1.0, 0.0, 0.0],
        atol=1e-4,
    )
    assert node.attributes["best_approach_source"] == "bearing_triangulation"
    assert any(np.allclose(
        node.attributes["best_approach_position"], candidate,
    ) for candidate in ([0.0, 0.0, 0.0], [1.0, 0.0, -1.0]))


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


def test_materialize_goal_perception_from_clip_and_top_level():
    from conftopo.perception.perception_report import PerceptionReport
    from conftopo.agents.goat_agent_new import PerceptionManager

    pm = PerceptionManager(ConfTopoConfig(), NewGoatConfig())
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater")
    report = PerceptionReport(
        objects=[ObjectObservation(label="table", bbox=None, confidence=0.6, source="vlm")],
        goal_visible=False,
        goal_match_confidence=0.0,
        target_visibility="not_visible",
        target_direction="center",
        apparent_scale="medium",
        source="vlm",
        is_full=True,
    )
    pm._materialize_goal_perception(report, goal, clip_goal_sim=0.42)
    heater_objs = [o for o in report.objects if o.label == "heater"]
    assert heater_objs
    assert report.goal_visible
    assert report.goal_match_confidence >= 0.28


def test_materialize_goal_perception_relabels_alias_match():
    from conftopo.perception.perception_report import PerceptionReport
    from conftopo.agents.goat_agent_new import PerceptionManager

    pm = PerceptionManager(ConfTopoConfig(), NewGoatConfig())
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="bed")
    report = PerceptionReport(
        objects=[ObjectObservation(label="mattress", bbox=[0.1, 0.2, 0.5, 0.7], confidence=0.55, source="vlm")],
        goal_visible=False,
        goal_match_confidence=0.0,
        source="vlm",
        is_full=True,
    )
    pm._materialize_goal_perception(report, goal, clip_goal_sim=0.0)
    assert any(o.label == "bed" for o in report.objects)
    assert report.goal_visible


def test_materialize_goal_perception_promotes_weak_alias_match():
    from conftopo.perception.perception_report import PerceptionReport
    from conftopo.agents.goat_agent_new import PerceptionManager

    pm = PerceptionManager(ConfTopoConfig(), NewGoatConfig())
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="bed")
    report = PerceptionReport(
        objects=[ObjectObservation(label="mattress", bbox=[0.1, 0.2, 0.4, 0.6], confidence=0.27, source="vlm")],
        goal_visible=False,
        goal_match_confidence=0.0,
        source="vlm",
        is_full=True,
    )
    pm._materialize_goal_perception(report, goal, clip_goal_sim=0.0)
    assert any(o.label == "bed" for o in report.objects)
    assert report.goal_visible


def test_materialize_goal_perception_from_room_prior_clip():
    from conftopo.perception.perception_report import PerceptionReport
    from conftopo.agents.goat_agent_new import PerceptionManager

    pm = PerceptionManager(ConfTopoConfig(), NewGoatConfig())
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater", room_prior=["hallway"])
    report = PerceptionReport(
        objects=[ObjectObservation(label="table", bbox=None, confidence=0.5, source="vlm")],
        room_label="hallway",
        room_confidence=0.8,
        goal_visible=False,
        goal_match_confidence=0.0,
        target_visibility="not_visible",
        target_direction="center",
        apparent_scale="medium",
        source="vlm",
        is_full=True,
    )
    pm._materialize_goal_perception(report, goal, clip_goal_sim=0.24)
    assert any(o.label == "heater" for o in report.objects)
    assert report.goal_visible


def test_frontier_prefers_room_prior_when_no_goal_anchor():
    planner = NavigationPlanner(NewGoatConfig())
    topo = DynamicTopoMap()
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater", room_prior=["hallway"])
    agent_pos = np.zeros(3, dtype=np.float32)
    topo.add_node(
        NodeType.ROOM,
        position=np.array([8.0, 0.0, 0.0]),
        label="hallway",
        confidence=0.9,
        attributes={"semantic_role": "room_summary"},
    )
    near_room = topo.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([7.0, 0.0, 0.0]),
        confidence=0.5,
        attributes={"anchor_waypoint_id": "way_0"},
    )
    far_room = topo.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([0.0, 0.0, 8.0]),
        confidence=0.5,
        attributes={"anchor_waypoint_id": "way_0"},
    )
    proposals = planner._collect_frontier_proposals(
        topo, goal, StructureTarget(), agent_pos,
    )
    by_id = {p.candidate_node_id: p.score for p in proposals}
    assert by_id[near_room] > by_id[far_room]


def test_semantic_memory_room_beats_unrelated_frontier():
    """Room matching room_prior should outscore a distant unrelated frontier."""
    planner = NavigationPlanner(NewGoatConfig())
    topo = DynamicTopoMap()
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="bed", room_prior=["bedroom"])
    agent_pos = np.zeros(3, dtype=np.float32)
    bedroom_id = topo.add_node(
        NodeType.ROOM,
        position=np.array([6.0, 0.0, 0.0]),
        label="bedroom",
        confidence=0.9,
        attributes={"cross_goal_preserved": True, "summary_type": "room_region"},
    )
    far_frontier = topo.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([0.0, 0.0, 12.0]),
        confidence=0.5,
        attributes={"anchor_waypoint_id": "way_0"},
    )
    proposals = planner.collect_goal_proposals(
        topo, goal, StructureTarget(), NavPhase.GLOBAL_SEARCH, agent_pos,
    )
    scored = planner.score_goal_proposals(proposals, goal, topo, agent_pos)
    best = planner.select_best_proposal(scored)
    assert best is not None
    assert best.source == "semantic_memory"
    assert best.candidate_node_id == bedroom_id


def test_semantic_memory_skips_too_close_node():
    """A semantic node the agent is already standing on should not be proposed."""
    planner = NavigationPlanner(NewGoatConfig())
    topo = DynamicTopoMap()
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater", room_prior=["hallway"])
    agent_pos = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    nearby_room = topo.add_node(
        NodeType.ROOM,
        position=np.array([3.1, 0.0, 0.0]),
        label="hallway",
        confidence=0.9,
        attributes={"cross_goal_preserved": True, "summary_type": "room_region"},
    )
    proposals = planner.collect_goal_proposals(
        topo, goal, StructureTarget(), NavPhase.GLOBAL_SEARCH, agent_pos,
    )
    sem_proposals = [p for p in proposals if p.source == "semantic_memory"]
    assert len(sem_proposals) == 0


def test_semantic_memory_excludes_environment_objects():
    """environment_object nodes should NOT appear as semantic_memory proposals."""
    planner = NavigationPlanner(NewGoatConfig())
    topo = DynamicTopoMap()
    goal = GoalManager()
    goal.current_goal = GoalNode(target_object="heater", room_prior=["hallway"])
    agent_pos = np.zeros(3, dtype=np.float32)
    topo.upsert_object_observation(
        label="nightstand",
        bbox=None,
        confidence=0.8,
        position=np.array([5.0, 0.0, 0.0]),
        source="vlm",
        spatial_attrs={"semantic_role": "environment_object", "cross_goal_preserved": True},
    )
    proposals = planner.collect_goal_proposals(
        topo, goal, StructureTarget(), NavPhase.GLOBAL_SEARCH, agent_pos,
    )
    sem_proposals = [p for p in proposals if p.source == "semantic_memory"]
    assert all(p.candidate_type != "environment_object" for p in sem_proposals)


def test_cross_goal_region_room_survives_resync():
    topo = DynamicTopoMap()
    room_id = topo.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0]),
        label="hallway",
        confidence=0.65,
        node_id="region::hallway::0",
        attributes={
            "summary_type": "room_region",
            "region_source": "waypoint_cluster",
            "base_label": "hallway",
            "cross_goal_preserved": True,
        },
    )
    for idx in range(4):
        wp = f"way_{idx}"
        topo.add_node(
            NodeType.WAYPOINT_VISITED,
            position=np.array([float(idx), 0.0, 0.0]),
            confidence=0.8,
            node_id=wp,
        )
    topo._sync_structure_rooms_from_waypoints()
    assert topo.has_node(room_id)


def test_object_memory_wins_over_higher_scored_frontier():
    """Once a goal anchor exists, frontier must not reclaim control."""
    planner = NavigationPlanner(NewGoatConfig())
    frontier = GoalProposal(
        goal_id="goal_0",
        candidate_node_id="way_1",
        candidate_type="frontier",
        target_position=np.array([5.0, 0.0, 0.0]),
        score=0.95,
        source="frontier",
    )
    anchor = GoalProposal(
        goal_id="goal_0",
        candidate_node_id="obj_1",
        candidate_type="object_anchor",
        target_position=np.array([1.0, 0.0, 0.0]),
        score=0.55,
        source="object_memory",
    )
    best = planner.select_best_proposal([frontier, anchor])
    assert best is not None
    assert best.source == "object_memory"
    assert best.candidate_node_id == "obj_1"


def test_frontier_navigation_remains_global_search_with_structure_prior():
    """When a bedroom room and a frontier exist, the agent should navigate to
    the bedroom (semantic_memory) for a wardrobe goal, or to a nearby frontier.
    Either way it stays in GLOBAL_SEARCH."""
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
    agent.topo_map.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([2.0, 0.0, 0.0]),
        confidence=0.5,
    )

    decision = agent.plan()

    assert decision.target_type in ("frontier", "room_summary", "room")
    assert agent.nav_phase in (NavPhase.GLOBAL_SEARCH, NavPhase.ROUTE_TO_STRUCTURE)


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
