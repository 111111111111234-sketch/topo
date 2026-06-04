"""Phase 2 remaining work tests: pathfinder helpers, target lifecycle, SOON interface."""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.adapters.soon_adapter import SOONConfTopoAdapter
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.navigation import CollisionLikeTracker, relative_to_world, world_to_relative


ROOT = Path(__file__).resolve().parents[2]


def test_relative_world_conversion():
    origin = np.array([10.0, 0.5, -2.0], dtype=np.float32)
    rel = np.array([1.5, 0.0, -3.0], dtype=np.float32)
    world = relative_to_world(rel, origin)
    assert np.allclose(world, [11.5, 0.5, -5.0])
    assert np.allclose(world_to_relative(world, origin), rel)
    print("  ok relative/world conversion")


def test_collision_like_tracker():
    tracker = CollisionLikeTracker(move_epsilon=0.03, trigger_steps=3)
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.01, 0.0, 0.0]),
        np.array([0.015, 0.0, 0.0]),
        np.array([0.02, 0.0, 0.0]),
    ]
    outputs = [tracker.update("move_forward", pos) for pos in positions]
    assert outputs[-1]["collision_like"] is True
    assert outputs[-1]["stuck_steps"] >= 3
    reset = tracker.update("turn_left", np.array([0.02, 0.0, 0.0]))
    assert reset["collision_like"] is False
    assert reset["stuck_steps"] == 0
    print("  ok collision-like tracker")


def test_candidate_waypoint_lifecycle():
    topo = DynamicTopoMap()
    cid = topo.add_candidate_waypoint(np.array([1.0, 0.0, 0.0]), label="ghost")
    node = topo.get_node(cid)
    assert node.node_type == NodeType.WAYPOINT_CANDIDATE
    topo.promote_frontier_to_visited(cid)
    node = topo.get_node(cid)
    assert node.node_type == NodeType.WAYPOINT_VISITED
    assert node.attributes["promoted_from"] == "candidate"
    fid = topo.add_node(NodeType.WAYPOINT_FRONTIER, np.array([2.0, 0.0, 0.0]))
    topo.consume_node(fid, "target_reached")
    frontier = topo.get_node(fid)
    assert frontier.attributes["consumed"] is True
    assert frontier.attributes["consume_reason"] == "target_reached"
    oid = topo.add_node(NodeType.OBJECT, np.array([3.0, 0.0, 0.0]))
    topo.block_node(oid, "unreachable", until_step=7)
    obj = topo.get_node(oid)
    assert obj.attributes["blocked"] is True
    assert obj.attributes["blocked_reason"] == "unreachable"
    assert obj.attributes["blocked_until_step"] == 7
    print("  ok candidate/consume/block lifecycle")


def test_soon_adapter_goal_graph_interface():
    graph_path = ROOT / "data/goal_graphs/soon/val_unseen_house_goal_graphs.json"
    if not graph_path.exists():
        print("  skip SOON adapter: fixture missing")
        return
    graphs = json.load(open(graph_path))
    key = next(iter(graphs))
    adapter = SOONConfTopoAdapter(ROOT / "data/datasets/soon", ROOT / "data/goal_graphs/soon")
    ig = adapter.load_instruction_graph(key=key)
    goal = ig.get_current_goal()
    assert goal is not None
    assert isinstance(goal.target_object, str)
    assert isinstance(goal.attributes, list)
    assert isinstance(goal.room_prior, list)
    assert isinstance(goal.landmarks, list)
    if goal.target_embedding is not None:
        assert hasattr(goal.target_embedding, "shape")
    print("  ok SOON GoalNode/InstructionGraph interface")


if __name__ == "__main__":
    test_relative_world_conversion()
    test_collision_like_tracker()
    test_candidate_waypoint_lifecycle()
    test_soon_adapter_goal_graph_interface()
    print("Phase 2 remaining tests passed")
