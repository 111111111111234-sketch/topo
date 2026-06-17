"""Rule-based semantic scorer for navigation decision (Phase 2)."""

from typing import Dict, List, Optional, Union
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
    current_node_id: Optional[str] = None,
    explored_rooms_no_target: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Compute semantic bias scores for candidate nodes.

    Local topo style: semantic relevance plus a small Euclidean distance
    penalty.  The remote graph-distance penalty over-preferred nearby known
    structure nodes and under-explored semantically promising objects.
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
        score = _score_node(current_goal, node, topo_map, agent_position,
                            explored_rooms_no_target=explored_rooms_no_target)
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
    explored_rooms_no_target: Optional[dict] = None,
) -> float:
    """Score a single node against the current goal."""
    score = 0.0

    if isinstance(goal, GoalNode):
        score += _score_object_goal(goal, node, topo_map)
    else:
        score += _score_route_goal(goal, node, topo_map)

    if node.node_type == NodeType.WAYPOINT_FRONTIER:
        score += 0.2

    dist = np.linalg.norm(node.position - agent_pos)
    score -= 0.05 * min(dist, 20.0) / 20.0

    if node.visit_count > 1:
        score -= 0.1 * min(node.visit_count, 5) / 5.0

    score += 0.1 * node.confidence

    if explored_rooms_no_target:
        room_id = node.attributes.get("room_id")
        if room_id and room_id in explored_rooms_no_target:
            score -= 0.6
        elif node.node_type == NodeType.ROOM and node.node_id in explored_rooms_no_target:
            score -= 0.6

    return score


def _score_object_goal(goal: GoalNode, node: SemanticNode, topo_map: DynamicTopoMap) -> float:
    """Score a node for object-goal task."""
    score = 0.0

    if node.node_type == NodeType.OBJECT and goal.target_embedding is not None:
        sim = cosine_similarity(goal.target_embedding, node.embedding)
        score += 0.4 * sim

    if node.node_type == NodeType.ROOM and goal.room_prior:
        if node.label in goal.room_prior:
            score += 0.3
        elif node.embedding is not None and goal.room_prior_embeddings is not None:
            sims = [cosine_similarity(node.embedding, rpe)
                    for rpe in goal.room_prior_embeddings]
            score += 0.3 * max(sims) if sims else 0.0

    if (node.node_type == NodeType.ROOM
            and node.attributes.get("summary_type") == "room_region"):
        contains = [str(c).lower() for c in node.attributes.get("contains_labels", [])]
        target = getattr(goal, "target_object", None)
        if target:
            target_parts = {target.lower()}
            if " and " in target:
                target_parts.update(part.strip().lower() for part in target.split(" and ") if part.strip())
            if " or " in target:
                target_parts.update(part.strip().lower() for part in target.split(" or ") if part.strip())
            if target_parts & set(contains):
                score += 0.25

    if goal.landmark_embeddings is not None and node.embedding is not None:
        sims = [cosine_similarity(node.embedding, le)
                for le in goal.landmark_embeddings]
        if sims:
            score += 0.2 * max(sims)

    landmark_labels = {
        str(l).strip().lower()
        for l in (goal.landmarks or [])
        if str(l).strip()
    }
    if landmark_labels:
        node_label = (node.label or "").strip().lower()
        if node.node_type in (NodeType.LANDMARK, NodeType.OBJECT) and node_label in landmark_labels:
            score += 0.35 if node.node_type == NodeType.LANDMARK else 0.25

    if node.node_type in (NodeType.WAYPOINT_VISITED, NodeType.WAYPOINT_FRONTIER):
        view_labels = {
            str(x).strip().lower()
            for x in node.attributes.get("view_object_labels", []) or []
            if str(x).strip()
        }
        view_room = node.attributes.get("view_room_label")
        if view_room and str(view_room).strip().lower() not in ("", "unknown"):
            view_labels.add(str(view_room).strip().lower())
        if goal.room_prior and view_labels & {r.lower() for r in goal.room_prior}:
            score += 0.2
        if landmark_labels and view_labels & landmark_labels:
            score += 0.15
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
