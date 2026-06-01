"""Phase 2 integration tests: LightPerceiver + ETPNav Agent + GOAT Agent."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.perception.light_perceiver import LightPerceiver, cosine_sim
from conftopo.agents.etpnav_agent import ConfTopoETPNavAgent
from conftopo.agents.goat_agent import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import InstructionGraph, SubGoal, GoalNode
from conftopo.core.dynamic_topo_map import NodeType


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

    print("\n=== All Phase 2 tests passed! ===")
