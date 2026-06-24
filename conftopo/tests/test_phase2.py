"""Phase 2 integration tests: LightPerceiver + ETPNav Agent + GOAT Agent."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.perception.light_perceiver import LightPerceiver, cosine_sim
from conftopo.agents.etpnav_agent import ConfTopoETPNavAgent
from conftopo.agents.goat_agent_new import ConfTopoGOATAgent
from conftopo.agents.goat_agent import ConfTopoGOATAgent as MainGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import InstructionGraph, SubGoal, GoalNode, Relation
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType
from conftopo.core.hypothesis_pool import Hypothesis
from conftopo.core.rule_scorer import compute_semantic_bias
from conftopo.perception.heavy_perceiver import ObjectObservation


def test_cosine_sim():
    """Test cosine similarity utility."""
    a = np.array([1, 0, 0], dtype=np.float32)
    b = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    sims = cosine_sim(a, b)
    assert abs(sims[0] - 1.0) < 1e-5
    assert abs(sims[1]) < 1e-5
    print("  ✓ cosine_sim")


def test_light_perceiver_basic():
    """Test LightPerceiver with synthetic embeddings."""
    D = 512
    # Create synthetic room embeddings
    room_labels = ["kitchen", "bedroom", "bathroom"]
    room_embeds = np.random.randn(3, D).astype(np.float32)
    room_embeds /= np.linalg.norm(room_embeds, axis=1, keepdims=True)

    perceiver = LightPerceiver(
        room_labels=room_labels,
        room_text_embeds=room_embeds,
    )

    # Set goal
    perceiver.set_goal_labels(
        labels=["sink", "table"],
        embeddings=np.random.randn(2, D).astype(np.float32),
    )

    # Perceive single view
    visual = np.random.randn(D).astype(np.float32)
    result = perceiver.perceive(visual)
    assert "room_label" in result
    assert "room_scores" in result
    assert len(result["room_scores"]) == 3
    assert "goal_scores" in result
    assert len(result["goal_scores"]) == 2
    print(f"  ✓ perceive single view: room={result['room_label']}")

    # Perceive multi-view
    visual_pano = np.random.randn(12, D).astype(np.float32)
    result = perceiver.perceive(visual_pano)
    assert "room_label" in result
    print(f"  ✓ perceive multi-view: room={result['room_label']}")

    # Pano perceive
    pano_result = perceiver.perceive_pano(visual_pano)
    assert "per_view" in pano_result
    assert len(pano_result["per_view"]) == 12
    print(f"  ✓ perceive_pano: room={pano_result.get('room_label', '?')}")

    # Quick methods
    room, conf = perceiver.classify_room(visual)
    assert room in room_labels
    print(f"  ✓ classify_room: {room} ({conf:.3f})")

    goal, sim = perceiver.match_goal(visual)
    assert goal in ["sink", "table"]
    print(f"  ✓ match_goal: {goal} ({sim:.3f})")


def test_etpnav_agent_alpha_zero():
    """Test that alpha=0 produces zero bias (退化安全)."""
    config = ConfTopoConfig()
    config.planning.alpha = 0.0
    agent = ConfTopoETPNavAgent(config)

    ig = InstructionGraph(goal_type="route", sub_goals=[
        SubGoal(id=0, action="go_forward", landmark="kitchen"),
    ])
    agent.set_instruction_graph(ig)

    # Simulate some graph updates
    agent.on_graph_update(
        cur_vp="0", cur_pos=np.zeros(3),
        cur_embeds=np.random.randn(768).astype(np.float32),
        cand_vps=["0_0", "0_1"],
        cand_pos=[np.array([2, 0, 0]), np.array([0, 0, 2])],
    )

    bias = agent.get_semantic_bias(["0", "0_0", "0_1"], np.zeros(3))
    assert np.allclose(bias, 0.0), f"alpha=0 should give zero bias, got {bias}"
    print("  ✓ alpha=0 → zero bias (退化安全)")


def test_etpnav_agent_alpha_positive():
    """Test that alpha>0 produces non-trivial bias."""
    config = ConfTopoConfig()
    config.planning.alpha = 0.3
    agent = ConfTopoETPNavAgent(config)

    D = 768
    embed = np.random.randn(D).astype(np.float32)
    ig = InstructionGraph(goal_type="route", sub_goals=[
        SubGoal(id=0, action="go_forward", landmark="kitchen",
                landmark_embedding=np.random.randn(D).astype(np.float32)),
    ])
    agent.set_instruction_graph(ig)

    agent.on_graph_update(
        cur_vp="0", cur_pos=np.zeros(3), cur_embeds=embed,
        cand_vps=["0_0", "0_1", "0_2"],
        cand_pos=[np.array([2, 0, 0]), np.array([0, 0, 2]), np.array([-2, 0, 0])],
    )

    bias = agent.get_semantic_bias(
        [None, "0", "0_0", "0_1", "0_2"],
        np.zeros(3),
    )
    assert bias.shape[0] == 5
    assert bias[0] == 0.0  # None node → 0
    # At least some non-zero biases
    assert not np.allclose(bias, 0.0), "alpha>0 should produce non-zero bias"
    print(f"  ✓ alpha=0.3 → bias range [{bias.min():.4f}, {bias.max():.4f}]")


def test_etpnav_agent_multi_step():
    """Test multi-step graph mirroring."""
    config = ConfTopoConfig()
    config.planning.alpha = 0.5
    agent = ConfTopoETPNavAgent(config)

    ig = InstructionGraph(goal_type="route", sub_goals=[
        SubGoal(id=0, action="go_forward", landmark="door"),
    ])
    agent.set_instruction_graph(ig)

    # Step 1
    agent.on_graph_update(
        cur_vp="0", cur_pos=np.array([0, 0, 0]),
        cur_embeds=np.random.randn(768).astype(np.float32),
        cand_vps=["0_0"], cand_pos=[np.array([2, 0, 0])],
    )
    # Step 2: move to ghost
    agent.on_visit_ghost("0_0", np.array([2, 0, 0]))
    agent.on_graph_update(
        cur_vp="1", cur_pos=np.array([2, 0, 0]),
        cur_embeds=np.random.randn(768).astype(np.float32),
        cand_vps=["1_0"], cand_pos=[np.array([4, 0, 0])],
        prev_vp="0",
    )

    assert agent.topo_map.num_nodes >= 3  # at least 2 visited + 1 frontier
    print(f"  ✓ multi-step: {agent.topo_map.num_nodes} nodes in topo map")


def test_goat_agent_basic():
    """Test GOAT agent observe → update → plan cycle."""
    config = ConfTopoConfig()
    agent = ConfTopoGOATAgent(config)

    D = 512
    goal = GoalNode(
        target_object="sink",
        target_embedding=np.random.randn(D).astype(np.float32),
        room_prior=["kitchen", "bathroom"],
    )
    agent.set_new_goal(goal)

    # Step 1: initial observation
    obs = {
        "position": [0, 0, 0],
        "heading": 0.0,
        "rgb_embed": np.random.randn(D).astype(np.float32),
    }
    action = agent.step(obs)
    stats = agent.memory_stats
    assert stats["visited_waypoints"] >= 1
    assert stats["frontiers"] >= 1  # initial frontiers generated
    print(f"  ✓ step 1: {stats}")

    # Step 2: move
    obs2 = {
        "position": [2, 0, 0],
        "heading": 0.0,
        "rgb_embed": np.random.randn(D).astype(np.float32),
    }
    action2 = agent.step(obs2)
    stats2 = agent.memory_stats
    assert stats2["visited_waypoints"] >= 2
    print(f"  ✓ step 2: {stats2}")


def test_goal_relation_near_scores_matching_object_higher():
    """near(reference) boosts target objects that are near existing reference nodes."""
    topo = DynamicTopoMap()
    chair_near = topo.add_node(NodeType.OBJECT, [0, 0, 0], label="chair", confidence=0.7)
    table = topo.add_node(NodeType.OBJECT, [1.0, 0, 0], label="table", confidence=0.7)
    chair_far = topo.add_node(NodeType.OBJECT, [8.0, 0, 0], label="chair", confidence=0.7)
    topo.add_edge(chair_near, table, EdgeType.OBSERVED_AT)

    graph = InstructionGraph(
        goal_type="object_goal",
        goal_nodes=[GoalNode(target_object="chair", relations=[Relation("near", "table")])],
    )
    scores = compute_semantic_bias(
        graph,
        topo,
        [chair_near, chair_far],
        agent_position=np.zeros(3, dtype=np.float32),
        normalize=False,
    )
    assert scores[0] > scores[1]
    assert graph.total_goals == 1
    assert len(topo.get_nodes_by_type(NodeType.OBJECT)) == 3
    print("  ✓ relation scoring: near(table)")


def test_goal_relation_reference_context_scores_room_and_waypoint():
    """Relation references in summaries / waypoint views provide search context."""
    topo = DynamicTopoMap()
    room_with_ref = topo.add_node(
        NodeType.ROOM,
        [0, 0, 0],
        label="dining room",
        attributes={"summary_type": "room_region", "contains_labels": ["table"]},
    )
    room_without_ref = topo.add_node(
        NodeType.ROOM,
        [1, 0, 0],
        label="bedroom",
        attributes={"summary_type": "room_region", "contains_labels": ["bed"]},
    )
    wp_with_ref = topo.add_node(
        NodeType.WAYPOINT_FRONTIER,
        [0, 0, 1],
        attributes={"view_object_labels": ["table"]},
    )
    wp_without_ref = topo.add_node(
        NodeType.WAYPOINT_FRONTIER,
        [1, 0, 1],
        attributes={"view_object_labels": ["bed"]},
    )
    graph = InstructionGraph(
        goal_type="object_goal",
        goal_nodes=[GoalNode(target_object="chair", relations=[Relation("near", "table")])],
    )
    scores = compute_semantic_bias(
        graph,
        topo,
        [room_with_ref, room_without_ref, wp_with_ref, wp_without_ref],
        agent_position=np.zeros(3, dtype=np.float32),
        normalize=False,
    )
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
    print("  ✓ relation scoring: context candidates")


def test_main_goat_runtime_priors_do_not_mutate_goalgraph():
    """Category priors are runtime context, not written back into the input GoalNode."""
    agent = MainGOATAgent(ConfTopoConfig())
    goal = GoalNode(target_object="sink")
    agent.set_new_goal(goal)
    assert goal.room_prior == []
    assert goal.landmarks == []
    assert {"kitchen", "bathroom"} <= set(agent._effective_room_priors(agent.instruction_graph.get_current_goal()))
    heavy_labels = set(agent._heavy_labels())
    assert "sink" in heavy_labels
    assert agent._build_vlm_context(agent.instruction_graph.get_current_goal(), "explore")
    print("  ✓ runtime priors do not mutate GoalGraph")


def test_main_goat_goal_proposals_wrap_candidates():
    """Planner boundary uses GoalProposal as runtime hypothesis over map nodes."""
    agent = MainGOATAgent(ConfTopoConfig())
    agent.set_new_goal(GoalNode(target_object="chair", relations=[Relation("near", "table")]))
    obj_id = agent.topo_map.add_node(NodeType.OBJECT, [0, 0, 0], label="chair")
    table_id = agent.topo_map.add_node(NodeType.OBJECT, [1, 0, 0], label="table")
    agent.topo_map.add_edge(obj_id, table_id, EdgeType.OBSERVED_AT)
    obj = agent.topo_map.get_node(obj_id)

    proposals = agent.generate_goal_proposals([obj], np.array([1.0], dtype=np.float32))
    scored = agent.score_goal_proposals(proposals)
    best = agent.select_best_proposal(scored)

    assert best is not None
    assert best.goal_id == "goal_0"
    assert best.candidate_node_id == obj_id
    assert best.candidate_type == "object"
    assert best.target_position is not None
    assert best.source == "object_memory"
    print("  ✓ GoalProposal wraps planner candidates")


def test_main_goat_hypothesis_promotes_only_after_confirmed_observation():
    """Weak hypotheses become ObjectNode only after heavy/VLM confirmation."""
    class FakeHeavy:
        def perceive(self, *args, **kwargs):
            return [
                ObjectObservation(
                    label="chair",
                    bbox=[0.1, 0.2, 0.4, 0.8],
                    confidence=0.85,
                    source="heavy",
                    step_id=2,
                )
            ]

    agent = MainGOATAgent(ConfTopoConfig())
    agent.set_new_goal(GoalNode(target_object="chair"))
    cur_vp = agent.topo_map.add_node(NodeType.WAYPOINT_VISITED, [0, 0, 0], label="wp")
    agent._cur_vp_id = cur_vp
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    agent.set_heavy_perceiver(FakeHeavy())

    first = agent.hypothesis_pool.add_or_update(Hypothesis(
        id="",
        goal_id="goal_0",
        kind="object",
        label="chair",
        source="clip",
        anchor_node_id=cur_vp,
        position=np.zeros(3, dtype=np.float32),
        score=0.6,
        confidence=0.6,
        first_seen_step=1,
        last_seen_step=1,
    ))
    second = agent.hypothesis_pool.add_or_update(Hypothesis(
        id="",
        goal_id="goal_0",
        kind="object",
        label="chair",
        source="clip",
        anchor_node_id=cur_vp,
        position=np.zeros(3, dtype=np.float32),
        score=0.62,
        confidence=0.62,
        first_seen_step=2,
        last_seen_step=2,
    ))
    assert first.id == second.id
    assert second.status == "needs_verify"
    assert len(agent.topo_map.get_nodes_by_type(NodeType.OBJECT)) == 0

    room_label = agent._add_heavy_object_nodes(cur_vp, None)
    assert room_label is None or isinstance(room_label, str)
    objects = agent.topo_map.get_nodes_by_type(NodeType.OBJECT)
    assert len(objects) == 1
    promoted = agent.hypothesis_pool.to_debug_list(goal_id="goal_0")
    assert promoted[0]["status"] == "promoted"
    assert promoted[0]["promoted_node_id"] == objects[0].node_id
    print("  ✓ hypothesis promotes only after confirmed observation")


def test_goat_agent_multi_goal():
    """Test GOAT agent multi-goal: memory NOT cleared between goals."""
    config = ConfTopoConfig()
    agent = ConfTopoGOATAgent(config)

    D = 512
    goal1 = GoalNode(target_object="sofa", target_embedding=np.random.randn(D).astype(np.float32))
    agent.set_new_goal(goal1)

    # Explore for goal 1
    for i in range(5):
        obs = {"position": [i * 2, 0, 0], "heading": 0.0, "rgb_embed": np.random.randn(D).astype(np.float32)}
        agent.step(obs)

    nodes_after_goal1 = agent.topo_map.num_nodes
    print(f"  After goal 1: {nodes_after_goal1} nodes")

    # Switch to goal 2 — memory preserved!
    goal2 = GoalNode(target_object="table", target_embedding=np.random.randn(D).astype(np.float32))
    agent.set_new_goal(goal2)  # NOT calling reset()

    assert agent.topo_map.num_nodes == nodes_after_goal1, "Memory should be preserved!"

    # Continue exploring
    for i in range(3):
        obs = {"position": [10 + i * 2, 0, 0], "heading": 0.0, "rgb_embed": np.random.randn(D).astype(np.float32)}
        agent.step(obs)

    nodes_after_goal2 = agent.topo_map.num_nodes
    assert nodes_after_goal2 > nodes_after_goal1, "Should have more nodes from continued exploration"
    print(f"  After goal 2: {nodes_after_goal2} nodes (accumulated, not reset)")
    print(f"  ✓ multi-goal memory reuse: {nodes_after_goal1} → {nodes_after_goal2} nodes")


def test_goat_agent_semantic_node_creation():
    """High-similarity CLIP creates hints/landmarks but not formal object nodes."""
    config = ConfTopoConfig()
    config.perception.object_threshold = 0.1
    config.perception.room_threshold = 0.1
    config.perception.landmark_threshold = 0.1
    agent = MainGOATAgent(config)

    D = 512
    embed = np.zeros(D, dtype=np.float32)
    embed[0] = 1.0
    goal = GoalNode(
        target_object="sink",
        target_embedding=embed.copy(),
        room_prior=["kitchen"],
        landmarks=["door"],
        landmark_embeddings=embed[np.newaxis, :].copy(),
    )
    agent.set_new_goal(goal)
    agent.perceiver.room_labels = ["kitchen"]
    agent.perceiver.room_text_embeds = embed[np.newaxis, :].copy()

    action = agent.step({
        "position": [0, 0, 0],
        "heading": 0.0,
        "rgb_embed": embed.copy(),
    })
    stats = agent.memory_stats
    assert stats["objects"] == 0
    assert stats["landmarks"] >= 1
    visited = agent.topo_map.get_visited()
    assert visited, "expected at least one visited waypoint"
    assert visited[0].attributes.get("view_room_label") == "kitchen"
    assert visited[0].attributes.get("clip_goal_hypotheses")
    assert stats["rooms"] == 0, "clip room nodes should live on waypoint context only"
    assert action.get("target_node_id") is not None
    assert len(action.get("candidate_ids", [])) > 0
    clip_props = [p for p in action.get("goal_proposals", []) if p.get("source") == "clip"]
    assert clip_props
    assert all(p.get("requires_verification") is True for p in clip_props)
    assert all(p.get("can_stop") is False for p in clip_props)
    assert all(float(p.get("score", 0.0)) <= 0.65 for p in clip_props)
    assert action.get("hypotheses")
    print(f"  ✓ semantic nodes: {stats}")



def test_phase2_auto_threshold():
    """Auto threshold is stable and honors the configured floor."""
    from conftopo.acceptance.phase2 import auto_threshold
    assert abs(auto_threshold([0.10, 0.20], min_threshold=0.045, ratio=0.85) - 0.17) < 1e-6
    assert abs(auto_threshold([0.01], min_threshold=0.045, ratio=0.85) - 0.045) < 1e-6
    assert abs(auto_threshold([], min_threshold=0.045, ratio=0.85) - 0.045) < 1e-6
    print("  ✓ auto threshold")


def test_goat_agent_sticky_target_and_frontier_consume():
    """Sticky target prevents target flicker and consumed frontiers are skipped."""
    config = ConfTopoConfig()
    config.planning.sticky_target_enabled = True
    config.planning.sticky_release_after_no_progress = 100
    agent = ConfTopoGOATAgent(config)
    D = 512
    embed = np.zeros(D, dtype=np.float32)
    embed[0] = 1.0
    goal = GoalNode(target_object="sink", target_embedding=embed.copy())
    agent.set_new_goal(goal)
    first = agent.step({"position": [0, 0, 0], "heading": 0.0, "rgb_embed": embed.copy()})
    second = agent.step({"position": [0.05, 0, 0], "heading": 0.0, "rgb_embed": embed.copy()})
    assert first.get("target_node_id") == second.get("target_node_id")
    frontier = agent.topo_map.get_frontiers()[0]
    agent._consumed_frontier_ids.add(frontier.node_id)
    frontier.attributes["consumed"] = True
    agent._clear_sticky("test_consume")
    plan = agent.plan()
    assert frontier.node_id not in plan.get("candidate_ids", [])
    print("  ✓ sticky target + frontier consume")


if __name__ == "__main__":
    print("=== Phase 2 Integration Tests ===\n")

    print("[1] Cosine Similarity")
    test_cosine_sim()

    print("\n[2] LightPerceiver")
    test_light_perceiver_basic()

    print("\n[3] ETPNav Agent alpha=0 (退化安全)")
    test_etpnav_agent_alpha_zero()

    print("\n[4] ETPNav Agent alpha>0")
    test_etpnav_agent_alpha_positive()

    print("\n[5] ETPNav Agent multi-step")
    test_etpnav_agent_multi_step()

    print("\n[6] GOAT Agent basic cycle")
    test_goat_agent_basic()

    print("\n[7] GOAT Agent multi-goal memory reuse")
    test_goat_agent_multi_goal()

    print("\n[8] GOAT Agent semantic node creation")
    test_goat_agent_semantic_node_creation()

    print("\n[9] Auto threshold")
    test_phase2_auto_threshold()

    print("\n[10] Sticky target/frontier consume")
    test_goat_agent_sticky_target_and_frontier_consume()

    print("\n=== All Phase 2 tests passed! ===")



def test_goat_frontier_target_reached_consumed():
    """Reached frontier is consumed and not selected again."""
    config = ConfTopoConfig()
    agent = ConfTopoGOATAgent(config)
    D = 512
    embed = np.zeros(D, dtype=np.float32)
    embed[0] = 1.0
    goal = GoalNode(target_object="sink", target_embedding=embed.copy())
    agent.set_new_goal(goal)
    agent.step({"position": [0, 0, 0], "heading": 0.0, "rgb_embed": embed.copy()})

    frontier = next(iter(agent.topo_map.get_frontiers()))
    event = agent.on_navigation_event(frontier.node_id, "target_reached")
    assert event["action"] == "consumed_frontier"
    assert frontier.attributes["consumed"] is True

    out = agent.plan()
    assert frontier.node_id not in (out.get("candidate_ids") or [])
    skipped = out.get("sticky_debug", {}).get("skipped_candidates", [])
    assert any(row["node_id"] == frontier.node_id and row["reason"] == "consumed" for row in skipped)
    print("  ✓ frontier target_reached -> consumed and skipped")


def test_goat_no_progress_blocks_sticky_target():
    """Sticky no-progress releases and blocks a target for a short TTL."""
    config = ConfTopoConfig()
    config.planning.sticky_release_after_no_progress = 1
    config.planning.sticky_min_progress = 0.05
    agent = ConfTopoGOATAgent(config)
    agent._position = np.zeros(3, dtype=np.float32)
    wid = agent.topo_map.add_node(NodeType.WAYPOINT_VISITED, position=np.array([2, 0, 0], dtype=np.float32))
    node = agent.topo_map.get_node(wid)
    agent._sticky_target_id = wid
    agent._sticky_last_distance = 2.0
    agent._sticky_no_progress_steps = 0

    result = agent._sticky_plan_if_valid([node], [wid], np.array([1.0], dtype=np.float32))
    assert result is None
    assert agent._is_blocked_target(wid)
    assert node.attributes["blocked_reason"] == "no_progress"
    assert agent._sticky_target_id is None
    assert agent._sticky_release_reason == "no_progress"
    print("  ✓ no-progress -> sticky release and target blocked")


def test_goat_candidate_filter_skips_consumed_blocked_too_close():
    """Candidate filter reports consumed, blocked, and too-close skip reasons."""
    config = ConfTopoConfig()
    config.planning.target_too_close_radius = 0.5
    agent = ConfTopoGOATAgent(config)
    agent._position = np.zeros(3, dtype=np.float32)

    consumed_id = agent.topo_map.add_node(NodeType.WAYPOINT_FRONTIER, position=np.array([2, 0, 0], dtype=np.float32))
    consumed = agent.topo_map.get_node(consumed_id)
    consumed.attributes["consumed"] = True
    agent._consumed_frontier_ids.add(consumed_id)

    blocked_id = agent.topo_map.add_node(NodeType.WAYPOINT_VISITED, position=np.array([3, 0, 0], dtype=np.float32))
    blocked = agent.topo_map.get_node(blocked_id)
    agent._block_target(blocked_id, "unit_test")

    close_id = agent.topo_map.add_node(NodeType.OBJECT, position=np.array([0.1, 0, 0], dtype=np.float32))
    close = agent.topo_map.get_node(close_id)

    assert agent._candidate_skip_reason(consumed) == "consumed"
    assert agent._candidate_skip_reason(blocked) == "blocked"
    assert agent._candidate_skip_reason(close) == "too_close"
    print("  ✓ candidate filter skips consumed/blocked/too_close")


if __name__ == "__main__":
    print("\n[11] Navigation target lifecycle")
    test_goat_frontier_target_reached_consumed()
    test_goat_no_progress_blocks_sticky_target()
    test_goat_candidate_filter_skips_consumed_blocked_too_close()
    print("\n=== Navigation stability tests passed! ===")
