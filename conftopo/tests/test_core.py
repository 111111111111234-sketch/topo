"""Unit tests for ConfTopo Core modules."""

import numpy as np
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, EdgeType, SemanticNode
from conftopo.core.instruction_graph import InstructionGraph, SubGoal, GoalNode, Relation
from conftopo.core.confidence import (
    ConfidenceFactors,
    compute_semantic_confidence,
    compute_topo_confidence,
    update_on_observation,
    temporal_decay,
)
from conftopo.core.rule_scorer import compute_semantic_bias, cosine_similarity


def test_dynamic_topo_map_basic():
    """Test basic node/edge operations."""
    topo = DynamicTopoMap()

    # Add nodes
    n1 = topo.add_node(NodeType.WAYPOINT_VISITED, position=[0, 0, 0], label="start")
    n2 = topo.add_node(NodeType.WAYPOINT_FRONTIER, position=[3, 0, 0], label="frontier_1")
    n3 = topo.add_node(NodeType.OBJECT, position=[1, 1, 0], label="chair", confidence=0.7)

    assert topo.num_nodes == 3
    assert topo.get_node(n1).label == "start"
    assert topo.get_node(n2).node_type == NodeType.WAYPOINT_FRONTIER
    assert topo.get_node(n3).confidence == 0.7

    # Add edges
    topo.add_edge(n1, n2, EdgeType.NAVIGABLE)
    topo.add_edge(n1, n3, EdgeType.OBSERVED_AT)

    neighbors = topo.get_neighbors(n1)
    assert len(neighbors) == 2

    nav_neighbors = topo.get_neighbors(n1, EdgeType.NAVIGABLE)
    assert len(nav_neighbors) == 1
    assert nav_neighbors[0] == n2

    print("  [PASS] basic node/edge operations")


def test_dynamic_topo_map_queries():
    """Test spatial queries."""
    topo = DynamicTopoMap()

    topo.add_node(NodeType.WAYPOINT_VISITED, position=[0, 0, 0])
    topo.add_node(NodeType.WAYPOINT_VISITED, position=[5, 0, 0])
    topo.add_node(NodeType.WAYPOINT_FRONTIER, position=[10, 0, 0])

    assert topo.has_nearby_visited(np.array([0.5, 0, 0]), radius=1.0)
    assert not topo.has_nearby_visited(np.array([3, 0, 0]), radius=1.0)

    frontiers = topo.get_frontiers()
    assert len(frontiers) == 1

    visited = topo.get_visited()
    assert len(visited) == 2

    nearest = topo.find_nearest_node(np.array([4, 0, 0]))
    assert np.linalg.norm(nearest.position - np.array([5, 0, 0])) < 0.01

    within = topo.find_nodes_within_radius(np.array([0, 0, 0]), radius=6.0)
    assert len(within) == 2

    print("  [PASS] spatial queries")


def test_dynamic_topo_map_confidence():
    """Test confidence operations."""
    topo = DynamicTopoMap()

    n1 = topo.add_node(NodeType.OBJECT, position=[0, 0, 0], confidence=0.6)
    topo.update_confidence(n1, 0.2)
    assert abs(topo.get_node(n1).confidence - 0.8) < 1e-5

    topo.update_confidence(n1, 0.5)  # should clamp to 1.0
    assert topo.get_node(n1).confidence == 1.0

    topo.step()
    topo.step()
    topo.decay_all_confidences()
    assert topo.get_node(n1).confidence < 1.0

    print("  [PASS] confidence operations")


def test_dynamic_topo_map_memory_management():
    """Test promote, merge, prune."""
    topo = DynamicTopoMap()
    topo.merge_radius = 1.5

    # Promote frontier to visited
    n1 = topo.add_node(NodeType.WAYPOINT_FRONTIER, position=[0, 0, 0], confidence=0.3)
    topo.promote_frontier_to_visited(n1)
    assert topo.get_node(n1).node_type == NodeType.WAYPOINT_VISITED
    assert topo.get_node(n1).confidence > 0.3

    # Merge nearby nodes
    n2 = topo.add_node(NodeType.OBJECT, position=[5, 0, 0], confidence=0.8, label="chair")
    n3 = topo.add_node(NodeType.OBJECT, position=[5.5, 0, 0], confidence=0.4, label="chair")
    topo.merge_nearby_nodes(NodeType.OBJECT)
    assert topo.num_nodes == 2  # n3 merged into n2

    # Prune low confidence
    n4 = topo.add_node(NodeType.LANDMARK, position=[10, 0, 0], confidence=0.05)
    topo.prune_low_confidence()
    assert not topo.has_node(n4)
    assert topo.has_node(n1)  # visited waypoints never pruned

    print("  [PASS] memory management (promote/merge/prune)")


def test_dynamic_topo_map_serialization():
    """Test serialize/deserialize."""
    topo = DynamicTopoMap()
    n1 = topo.add_node(NodeType.WAYPOINT_VISITED, position=[1, 2, 3], label="wp1", confidence=0.9)
    n2 = topo.add_node(NodeType.OBJECT, position=[4, 5, 6], label="table", confidence=0.7)
    topo.add_edge(n1, n2, EdgeType.OBSERVED_AT)

    data = topo.to_dict()
    topo2 = DynamicTopoMap.from_dict(data)

    assert topo2.num_nodes == 2
    assert topo2.get_node(n1).label == "wp1"
    assert abs(topo2.get_node(n2).confidence - 0.7) < 1e-5
    assert len(topo2.get_neighbors(n1)) == 1

    print("  [PASS] serialization")


def test_instruction_graph_route():
    """Test route-mode InstructionGraph."""
    ig = InstructionGraph(goal_type="route")
    ig.sub_goals = [
        SubGoal(id=0, action="go_forward", landmark="kitchen island", spatial_relation="towards"),
        SubGoal(id=1, action="turn_left", landmark="hallway", spatial_relation="at"),
        SubGoal(id=2, action="stop", landmark="bedroom door", spatial_relation="at"),
    ]

    assert ig.total_goals == 3
    assert ig.get_current_goal().landmark == "kitchen island"
    assert not ig.is_complete()

    ig.advance()
    assert ig.get_current_goal().landmark == "hallway"
    assert ig.completed_goals == 1

    ig.advance()
    ig.advance()
    assert ig.is_complete()

    print("  [PASS] instruction graph (route mode)")


def test_instruction_graph_object_goal():
    """Test object-goal-mode InstructionGraph."""
    ig = InstructionGraph(goal_type="object_goal")
    ig.goal_nodes = [
        GoalNode(target_object="sofa", room_prior=["living room"], goal_type="category"),
        GoalNode(target_object="sink", room_prior=["kitchen", "bathroom"], goal_type="category"),
        GoalNode(
            target_object="white chair",
            attributes=["white", "wooden"],
            room_prior=["bedroom"],
            landmarks=["window"],
            relations=[Relation("near", "window")],
            goal_type="description",
        ),
    ]

    assert ig.total_goals == 3
    assert ig.get_current_goal().target_object == "sofa"

    ig.advance()
    assert ig.get_current_goal().target_object == "sink"

    # Multi-goal switch (GOAT-style)
    ig.set_current_goal_by_index(2)
    assert ig.get_current_goal().target_object == "white chair"
    assert ig.get_current_goal().goal_type == "description"

    print("  [PASS] instruction graph (object_goal mode)")


def test_instruction_graph_serialization():
    """Test JSON save/load."""
    ig = InstructionGraph(goal_type="object_goal")
    ig.goal_nodes = [
        GoalNode(target_object="table", room_prior=["dining room"], goal_type="category"),
        GoalNode(
            target_object="lamp",
            attributes=["tall"],
            relations=[Relation("on", "desk")],
            goal_type="description",
        ),
    ]
    ig.advance()  # mark first as completed

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    ig.save(path)

    ig2 = InstructionGraph.load(path)
    assert ig2.goal_type == "object_goal"
    assert ig2.total_goals == 2
    assert ig2.goal_nodes[0].status == "completed"
    assert ig2.goal_nodes[1].target_object == "lamp"
    assert ig2.goal_nodes[1].relations[0].reference == "desk"

    os.unlink(path)
    print("  [PASS] instruction graph serialization")


def test_confidence_system():
    """Test confidence computation."""
    factors = ConfidenceFactors(
        detection_score=0.8,
        multi_view_count=2,
        task_relevance=0.6,
        time_decay=0.9,
    )
    c = compute_semantic_confidence(factors)
    assert 0.0 <= c <= 1.0
    assert c > 0.3  # should be reasonably high

    factors_topo = ConfidenceFactors(
        navigability=0.9,
        execution_success=0.8,
        collision_history=0.0,
        backtrack_frequency=0.0,
        time_decay=1.0,
    )
    c_topo = compute_topo_confidence(factors_topo)
    assert c_topo > 0.5

    # Update on consistent observation should increase confidence
    c_new = update_on_observation(0.5, 0.8, is_consistent=True)
    assert c_new > 0.5

    # Update on inconsistent observation should decrease
    c_bad = update_on_observation(0.5, 0.8, is_consistent=False)
    assert c_bad < 0.5

    # Temporal decay
    c_decayed = temporal_decay(1.0, steps_elapsed=10, decay_rate=0.95)
    assert c_decayed < 1.0
    assert c_decayed > 0.5

    print("  [PASS] confidence system")


def test_rule_scorer():
    """Test semantic bias computation."""
    topo = DynamicTopoMap()

    # Create a simple scenario
    n1 = topo.add_node(NodeType.WAYPOINT_VISITED, position=[0, 0, 0])
    n2 = topo.add_node(NodeType.WAYPOINT_FRONTIER, position=[5, 0, 0])
    n3 = topo.add_node(NodeType.WAYPOINT_FRONTIER, position=[0, 5, 0])

    # Object near n2
    obj = topo.add_node(
        NodeType.OBJECT, position=[5, 1, 0],
        embedding=np.random.randn(512).astype(np.float32),
        label="sofa",
    )
    topo.add_edge(n2, obj, EdgeType.OBSERVED_AT)

    # Goal: find sofa
    ig = InstructionGraph(goal_type="object_goal")
    target_embed = np.random.randn(512).astype(np.float32)
    ig.goal_nodes = [
        GoalNode(target_object="sofa", target_embedding=target_embed, room_prior=["living room"]),
    ]

    scores = compute_semantic_bias(
        goal_graph=ig,
        topo_map=topo,
        candidate_node_ids=[n1, n2, n3],
        agent_position=np.array([0, 0, 0]),
        normalize=True,
    )

    assert scores.shape == (3,)
    # n2 should score highest (has object nearby + is frontier)
    assert scores[1] > scores[0]  # frontier with object > visited
    assert abs(scores.mean()) < 1e-5  # normalized

    print("  [PASS] rule scorer")


if __name__ == "__main__":
    print("Running ConfTopo Core unit tests...\n")
    test_dynamic_topo_map_basic()
    test_dynamic_topo_map_queries()
    test_dynamic_topo_map_confidence()
    test_dynamic_topo_map_memory_management()
    test_dynamic_topo_map_serialization()
    test_instruction_graph_route()
    test_instruction_graph_object_goal()
    test_instruction_graph_serialization()
    test_confidence_system()
    test_rule_scorer()
    print("\nAll tests passed!")
