"""Rule-based semantic scorer for navigation decision (Phase 2)."""

from typing import List, Optional, Union
import numpy as np

from conftopo.core.instruction_graph import InstructionGraph, SubGoal, GoalNode
from conftopo.core.dynamic_topo_map import DynamicTopoMap, SemanticNode, NodeType


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_semantic_bias(
    goal_graph: InstructionGraph,
    topo_map: DynamicTopoMap,
    candidate_node_ids: List[str],
    agent_position: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute semantic bias scores for candidate nodes.

    Returns z-normalized scores (mean=0, std=1) if normalize=True.
    """
    current_goal = goal_graph.get_current_goal()
    if current_goal is None:
        return np.zeros(len(candidate_node_ids), dtype=np.float32)

    scores = []
    for nid in candidate_node_ids:
        node = topo_map.get_node(nid)
        if node is None:
            scores.append(0.0)
            continue
        score = _score_node(current_goal, node, topo_map, agent_position)
        scores.append(score)

    scores = np.array(scores, dtype=np.float32)

    if normalize and len(scores) > 1:
        std = scores.std()
        if std > 1e-6:
            scores = (scores - scores.mean()) / std
        else:
            scores = scores - scores.mean()

    return scores


def _score_node(
    goal: Union[SubGoal, GoalNode],
    node: SemanticNode,
    topo_map: DynamicTopoMap,
    agent_pos: np.ndarray,
) -> float:
    """Score a single node against the current goal."""
    score = 0.0

    if isinstance(goal, GoalNode):
        score += _score_object_goal(goal, node, topo_map)
    else:
        score += _score_route_goal(goal, node, topo_map)

    # Frontier exploration bonus
    if node.node_type == NodeType.WAYPOINT_FRONTIER:
        score += 0.2

    # Distance penalty (prefer closer nodes when scores are similar)
    dist = np.linalg.norm(node.position - agent_pos)
    score -= 0.05 * min(dist, 20.0) / 20.0

    # Visited penalty (avoid revisiting)
    if node.visit_count > 1:
        score -= 0.1 * min(node.visit_count, 5) / 5.0

    # Confidence bonus (prefer high-confidence nodes)
    score += 0.1 * node.confidence

    return score


def _score_object_goal(goal: GoalNode, node: SemanticNode, topo_map: DynamicTopoMap) -> float:
    """Score a node for object-goal task (GOAT/SOON)."""
    score = 0.0

    # Target object match
    if node.node_type == NodeType.OBJECT and goal.target_embedding is not None:
        sim = cosine_similarity(goal.target_embedding, node.embedding)
        score += 0.4 * sim

    # Room prior match
    if node.node_type == NodeType.ROOM and goal.room_prior:
        if node.label in goal.room_prior:
            score += 0.3
        elif node.embedding is not None and goal.room_prior_embeddings is not None:
            sims = [cosine_similarity(node.embedding, rpe)
                    for rpe in goal.room_prior_embeddings]
            score += 0.3 * max(sims) if sims else 0.0

    # Room/region summary: weak match if summary contains target object label
    if (node.node_type == NodeType.ROOM
            and node.attributes.get("summary_type") == "room_region"):
        contains = node.attributes.get("contains_labels", [])
        target = getattr(goal, "target_object", None)
        if target and target in contains:
            score += 0.25

    # Landmark proximity
    if goal.landmark_embeddings is not None and node.embedding is not None:
        sims = [cosine_similarity(node.embedding, le)
                for le in goal.landmark_embeddings]
        if sims:
            score += 0.2 * max(sims)

    # Nodes that have related semantic nodes nearby get a bonus (skip folded)
    if node.node_type in (NodeType.WAYPOINT_VISITED, NodeType.WAYPOINT_FRONTIER):
        neighbors = topo_map.get_neighbors(node.node_id)
        for neighbor_id in neighbors:
            neighbor = topo_map.get_node(neighbor_id)
            if neighbor and neighbor.node_type == NodeType.OBJECT:
                if neighbor.attributes.get("folded"):
                    continue
                if goal.target_embedding is not None and neighbor.embedding is not None:
                    sim = cosine_similarity(goal.target_embedding, neighbor.embedding)
                    if sim > 0.5:
                        score += 0.2 * sim

    return score


def _score_route_goal(goal: SubGoal, node: SemanticNode, topo_map: DynamicTopoMap) -> float:
    """Score a node for route instruction task (R2R)."""
    score = 0.0

    # Landmark alignment
    if goal.landmark_embedding is not None and node.embedding is not None:
        sim = cosine_similarity(goal.landmark_embedding, node.embedding)
        score += 0.4 * sim

    # Room match
    if goal.implied_room and node.node_type == NodeType.ROOM:
        if node.label == goal.implied_room:
            score += 0.3

    # Nodes near landmarks matching the goal get a bonus
    if node.node_type in (NodeType.WAYPOINT_VISITED, NodeType.WAYPOINT_FRONTIER):
        neighbors = topo_map.get_neighbors(node.node_id)
        for neighbor_id in neighbors:
            neighbor = topo_map.get_node(neighbor_id)
            if neighbor and neighbor.node_type == NodeType.LANDMARK:
                if goal.landmark_embedding is not None and neighbor.embedding is not None:
                    sim = cosine_similarity(goal.landmark_embedding, neighbor.embedding)
                    if sim > 0.5:
                        score += 0.2 * sim

    return score
