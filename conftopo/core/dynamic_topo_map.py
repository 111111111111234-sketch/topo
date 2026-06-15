"""DynamicTopoMap: confidence-aware semantic topological memory graph."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import numpy as np
import networkx as nx
import math

from conftopo.core.confidence import ConfidenceFactors, compute_semantic_confidence
from conftopo.core.landmark_roles import (
    can_promote_object_to_landmark,
    classify_landmark_role,
    is_structural_label,
)
from conftopo.core.region_rooms import (
    normalize_region_label,
    region_instances_from_waypoints,
    room_for_labeled_waypoint,
)


class NodeType(Enum):
    WAYPOINT_VISITED = "waypoint_visited"
    WAYPOINT_FRONTIER = "waypoint_frontier"
    WAYPOINT_CANDIDATE = "waypoint_candidate"
    LANDMARK = "landmark"
    OBJECT = "object"
    ROOM = "room"


class EdgeType(Enum):
    NAVIGABLE = "navigable"
    OBSERVED_AT = "observed_at"
    BELONGS_TO = "belongs_to"
    VISIBLE_FROM = "visible_from"
    ADJACENT_TO = "adjacent_to"  # traversable link in semantic-structural layer

_PORTAL_LABEL_PRIORITY = (
    "door", "hallway", "entrance", "corridor", "arch", "gate", "opening",
)

@dataclass
class SemanticNode:
    """A node in the semantic topological map."""
    node_id: str
    node_type: NodeType
    position: np.ndarray  # 3D position (x, y, z)
    embedding: Optional[np.ndarray] = None  # semantic embedding (CLIP)
    confidence: float = 0.5 # [0,1]
    label: str = ""
    step_id: int = 0  # when this node was created/last updated
    visit_count: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


def _embedding_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _heading_to_direction_label(heading_rad: float) -> str:
    """Convert relative heading to 8-direction label."""
    h = _normalize_angle(heading_rad)
    if abs(h) < 0.3927:          # pi/8
        return "forward"
    if h > 0.3927 and h < 1.1781:  # pi/8 to 3pi/8
        return "forward-left"
    if h > 1.1781 and h < 1.9635:  # 3pi/8 to 5pi/8
        return "left"
    if h > 1.9635 and h < 2.7489:  # 5pi/8 to 7pi/8
        return "backward-left"
    if abs(h) > 2.7489:             # beyond 7pi/8
        return "backward"
    if h < -0.3927 and h > -1.1781:
        return "forward-right"
    if h < -1.1781 and h > -1.9635:
        return "right"
    if h < -1.9635 and h > -2.7489:
        return "backward-right"
    return "forward"


class DynamicTopoMap:
    """Confidence-aware semantic topological memory.

    A unified graph containing waypoints, frontiers, objects, landmarks,
    and rooms as nodes, with typed edges connecting them. Each node carries
    a multi-factor confidence score that decays over time and updates with
    new observations.

    map 会随着 agent 的移动和观测在线更新：
    observe -> add/update nodes -> update confidence -> merge/prune -> plan
    """

    def __init__(self, config=None):
        self.graph = nx.Graph()
        self._nodes: Dict[str, SemanticNode] = {}
        self._node_counter = 0
        self._current_step = 0


        # Config
        if config is not None:
            self.confidence_decay = config.confidence_decay
            self.near_radius = config.near_radius
            self.far_radius = config.far_radius
            self.prune_threshold = config.prune_threshold
            self.max_nodes = config.max_nodes
            self.merge_radius = config.merge_radius
            self.summary_radius = getattr(config, "summary_radius", 5.0)
            self.summary_min_distance = getattr(config, "summary_min_distance", 6.0)
            self.summary_low_detail_threshold = getattr(config, "summary_low_detail_threshold", 0.35)
            self.summary_mid_detail_threshold = getattr(config, "summary_mid_detail_threshold", 0.65)
            self.summary_max_observations = getattr(config, "summary_max_observations", 20)
            self.fold_distance = getattr(config, "fold_distance", 3.0)
            self.far_prune_distance = getattr(config, "far_prune_distance", 10.0)
            self.far_prune_threshold = getattr(config, "far_prune_threshold", 0.18)
            self.mid_prune_distance = getattr(config, "mid_prune_distance", 6.0)
            self.mid_prune_threshold = getattr(config, "mid_prune_threshold", 0.12)
            self.room_level_min_distance = getattr(config, "room_level_min_distance", 8.0)
            self.room_level_confidence_max = getattr(config, "room_level_confidence_max", 0.55)
            self.room_level_detail_max = getattr(config, "room_level_detail_max", 0.45)
            self.room_link_max_distance = getattr(config, "room_link_max_distance", 12.0)
            self.waypoint_compress_enabled = getattr(config, "waypoint_compress_enabled", True)
            self.waypoint_compress_distance = getattr(config, "waypoint_compress_distance", 5.0)
            self.waypoint_compress_keep_near = getattr(config, "waypoint_compress_keep_near", 3.0)
            self.waypoint_compress_collinear_deg = getattr(config, "waypoint_compress_collinear_deg", 20.0)
        else:
            # 默认配置
            self.confidence_decay = 0.95
            self.near_radius = 3.0
            self.far_radius = 10.0
            self.prune_threshold = 0.1
            self.max_nodes = 500
            self.merge_radius = 1.0
            self.summary_radius = 5.0
            self.summary_min_distance = 6.0
            self.summary_low_detail_threshold = 0.35
            self.summary_mid_detail_threshold = 0.65
            self.summary_max_observations = 20
            self.fold_distance = 3.0
            self.far_prune_distance = 10.0
            self.far_prune_threshold = 0.18
            self.mid_prune_distance = 6.0
            self.mid_prune_threshold = 0.12
            self.room_level_min_distance = 4.5
            self.room_level_confidence_max = 0.55
            self.room_level_detail_max = 0.45
            self.room_link_max_distance = 12.0
            self.waypoint_compress_enabled = True
            self.waypoint_compress_distance = 5.0
            self.waypoint_compress_keep_near = 3.0
            self.waypoint_compress_collinear_deg = 20.0

        self.object_history_keep_recent = 2
        self.landmark_history_keep_recent = 2
        self._last_granularity_debug: Dict[str, int] = {}

        self._last_visibility_debug: Dict[str, int] = {}
    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def current_step(self) -> int:
        return self._current_step

    def step(self):
        """Advance internal step counter."""
        self._current_step += 1

    def _generate_id(self, prefix: str = "n") -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    # ==================== Node Operations ====================

    def add_node(
        self,
        node_type: NodeType,
        position: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        confidence: float = 0.5,
        label: str = "",
        node_id: Optional[str] = None,
        attributes: Optional[Dict] = None,
    ) -> str:
        """Add a node to the map. Returns node_id."""
        if node_id is None:
            prefix = node_type.value[:3]
            node_id = self._generate_id(prefix)

        node = SemanticNode(
            node_id=node_id,
            node_type=node_type,
            position=np.array(position, dtype=np.float32),
            embedding=embedding,
            confidence=confidence,
            label=label,
            step_id=self._current_step,
            visit_count=1 if node_type == NodeType.WAYPOINT_VISITED else 0,
            attributes=attributes or {},
        )
        self._nodes[node_id] = node
        self.graph.add_node(node_id, node_type=node_type.value)
        return node_id

    def get_node(self, node_id: str) -> Optional[SemanticNode]:
        return self._nodes.get(node_id)

    def get_object_anchor(self, node_id: str) -> Optional[dict]:
        node = self._nodes.get(node_id)
        if node is None:
            return None

        attrs = node.attributes
        if not attrs.get("is_semantic_anchor"):
            return None

        position = attrs.get("anchor_position")
        if position is None:
            position = node.position.copy()

        return {
            "position": position,
            "waypoint_id": attrs.get("anchor_waypoint_id"),
            "waypoint_position": attrs.get("anchor_waypoint_position"),
            "room_id": attrs.get("anchor_room_id"),
            "label": node.label,
            "confidence": float(node.confidence),
        }

    def remove_node(self, node_id: str):
        if node_id in self._nodes:
            del self._nodes[node_id]
            self.graph.remove_node(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    # ==================== Edge Operations ====================
    # edge 由 两个 node a/b 连接
    def add_edge(
        self,
        node_a: str,
        node_b: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ):
        """Add a typed edge between two nodes."""
        if node_a in self._nodes and node_b in self._nodes:
            self.graph.add_edge(
                node_a, node_b,
                edge_type=edge_type.value,
                weight=weight,
            )
            if edge_type == EdgeType.NAVIGABLE:
                self._enrich_navigable_edge(node_a, node_b)
    #找与 node 直接相连的 node
    def _enrich_navigable_edge(self, node_a: str, node_b: str) -> None:
        """Add structured attributes to a NAVIGABLE edge (bidirectional)."""
        if not self.graph.has_edge(node_a, node_b):
            return
        node_a_obj = self._nodes.get(node_a)
        node_b_obj = self._nodes.get(node_b)
        if node_a_obj is None or node_b_obj is None:
            return
        pos_a = node_a_obj.position
        pos_b = node_b_obj.position
        dx = float(pos_b[0] - pos_a[0])
        dz = float(pos_b[2] - pos_a[2])
        dist = float(np.linalg.norm([dx, dz]))
        heading_ab = _normalize_angle(math.atan2(dx, dz))
        heading_ba = _normalize_angle(math.atan2(-dx, -dz))
        label_ab = _heading_to_direction_label(heading_ab)
        label_ba = _heading_to_direction_label(heading_ba)

        edge_data = self.graph.edges[node_a, node_b]
        edge_data["distance_m"] = round(dist, 2)
        edge_data["traversability"] = max(1.0, float(edge_data.get("traversability", 1.0)))
        edge_data["visited_count"] = int(edge_data.get("visited_count", 0)) + 1
        edge_data["evidence"] = list(set(edge_data.get("evidence", []) + ["odometry"]))
        edge_data["directions"] = {
            node_a: {"to": node_b, "direction_label": label_ab, "heading_delta": round(heading_ab, 4)},
            node_b: {"to": node_a, "direction_label": label_ba, "heading_delta": round(heading_ba, 4)},
        }
        edge_data["description_type"] = "navigable_transition"

    def reduce_edge_traversability(self, node_a: str, node_b: str, penalty: float = 0.3) -> None:
        """Reduce traversability of the edge between *node_a* and *node_b*."""
        if self.graph.has_edge(node_a, node_b):
            edge = self.graph.edges[node_a, node_b]
            edge["traversability"] = max(0.1, float(edge.get("traversability", 1.0)) - penalty)

    def _infer_passage_type(self, portal_node: Optional[SemanticNode], portal_label: str) -> str:
        """Infer passage type from portal node attributes and label."""
        if portal_node is None:
            return "passage"
        attrs = portal_node.attributes
        label = (portal_label or "").lower()
        if attrs.get("synthetic_portal"):
            return "synthetic_portal"
        if "door" in label or "doorway" in label:
            return "door"
        if "arch" in label:
            return "arch"
        if "corridor" in label or "hallway" in label:
            return "corridor"
        return "passage"

    def _enrich_adjacent_edge(
        self,
        node_a: str,
        node_b: str,
        room_a_label: str,
        room_b_label: str,
        portal_label: str = "",
        portal_node: Optional[SemanticNode] = None,
    ) -> None:
        """Add structured attributes to an ADJACENT_TO edge."""
        if not self.graph.has_edge(node_a, node_b):
            return
        passage_type = self._infer_passage_type(portal_node, portal_label)
        edge_data = self.graph.edges[node_a, node_b]
        edge_data["connected_rooms"] = [room_a_label, room_b_label]
        edge_data["via_landmark"] = portal_label or "passage"
        edge_data["passage_type"] = passage_type
        edge_data["confidence"] = float(edge_data.get("confidence", 0.6))
        edge_data["evidence"] = list(set(edge_data.get("evidence", []) + ["room_transition"]))
        edge_data["transitions"] = {
            node_a: {"from_room": room_a_label, "to_room": room_b_label, "via_landmark": portal_label or "passage", "passage_type": passage_type},
            node_b: {"from_room": room_b_label, "to_room": room_a_label, "via_landmark": portal_label or "passage", "passage_type": passage_type},
        }


    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get neighbor node IDs, optionally filtered by edge type."""
        if node_id not in self.graph:
            return []
        neighbors = []
        for neighbor in self.graph.neighbors(node_id):
            if edge_type is None:
                neighbors.append(neighbor)
            else:
                edge_data = self.graph.edges[node_id, neighbor]
                if edge_data.get("edge_type") == edge_type.value:
                    neighbors.append(neighbor)
        return neighbors

    # ==================== Query Operations ====================

    def get_nodes_by_type(self, node_type: NodeType) -> List[SemanticNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_frontiers(self) -> List[SemanticNode]:
        return self.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER)

    def get_candidates(self) -> List[SemanticNode]:
        return self.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE)

    def get_visited(self) -> List[SemanticNode]:
        return self.get_nodes_by_type(NodeType.WAYPOINT_VISITED)

    #判断 附近是否有 visited node ,用于 frontier 生成
    def has_nearby_visited(self, position: np.ndarray, radius: float = 1.0) -> bool:
        """Check if there's a visited node near the given position."""
        pos = np.array(position, dtype=np.float32)
        for node in self.get_visited():
            if np.linalg.norm(node.position - pos) < radius:
                return True
        return False

    def find_nearest_node(
        self, position: np.ndarray, node_type: Optional[NodeType] = None
    ) -> Optional[SemanticNode]:
        """Find the nearest node to a position."""
        pos = np.array(position, dtype=np.float32)
        best_node = None
        best_dist = float("inf")
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            dist = np.linalg.norm(node.position - pos)
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node

    def find_nodes_within_radius(
        self, position: np.ndarray, radius: float, node_type: Optional[NodeType] = None
    ) -> List[SemanticNode]:
        """Find all nodes within radius of a position."""
        pos = np.array(position, dtype=np.float32)
        results = []
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            if np.linalg.norm(node.position - pos) < radius:
                results.append(node)
        return results

    def get_node_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Get stacked embeddings for a list of nodes."""
        embeddings = []
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None and node.embedding is not None:
                embeddings.append(node.embedding)
            else:
                embeddings.append(np.zeros(512, dtype=np.float32))
        return np.stack(embeddings, axis=0) if embeddings else np.empty((0, 512))

    def get_node_confidences(self, node_ids: List[str]) -> np.ndarray:
        """Get confidence values for a list of nodes."""
        return np.array(
            [self._nodes[nid].confidence for nid in node_ids if nid in self._nodes],
            dtype=np.float32,
        )

    # ==================== Confidence Operations ====================

    def update_confidence(self, node_id: str, delta: float):
        """Update a node's confidence, clamped to [0, 1]."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.confidence = max(0.0, min(1.0, node.confidence + delta))
            node.step_id = self._current_step

    def set_confidence(self, node_id: str, value: float):
        if node_id in self._nodes:
            self._nodes[node_id].confidence = max(0.0, min(1.0, value))
            self._nodes[node_id].step_id = self._current_step

    def decay_all_confidences(self):
        """Apply one step of decay to stale node confidences."""
        for node in self._nodes.values():
            if node.step_id < self._current_step:
                node.confidence *= self.confidence_decay

    # ==================== Object Observation Operations ====================

    def upsert_object_observation(
        self,
        *,
        label: str,
        bbox: List[float],
        confidence: float,
        position: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        viewpoint_id: Optional[str] = None,
        view_heading: float = 0.0,
        room_context: Optional[str] = None,
        target_relevance: float = 0.0,
        room_prior_score: float = 0.0,
        source: str = "heavy",
    ) -> Tuple[str, bool]:
        """Create or update an object node from an object-level observation.

        Returns (node_id, merged_existing).
        """
        label = str(label)
        pos = np.array(position, dtype=np.float32)
        emb = np.array(embedding, dtype=np.float32) if embedding is not None else None
        matches = self.find_nodes_within_radius(pos, self.merge_radius * 2.0, NodeType.OBJECT)
        match = self._best_object_match(matches, label, bbox, emb, view_heading, pos, room_context)

        if match is None:
            attrs = self._new_object_attributes(
                bbox=bbox,
                confidence=confidence,
                viewpoint_id=viewpoint_id,
                view_heading=view_heading,
                room_context=room_context,
                source=source,
                target_relevance=target_relevance,
                room_prior_score=room_prior_score,
            )
            node_id = self.add_node(
                NodeType.OBJECT,
                position=pos,
                embedding=emb,
                confidence=compute_semantic_confidence(ConfidenceFactors(
                    detection_score=confidence,
                    multi_view_count=1,
                    task_relevance=target_relevance,
                    room_prior_score=room_prior_score,
                )),
                label=label,
                attributes=attrs,
            )
            if self._should_add_observation_edge(node_id, viewpoint_id, room_context):
                self.add_edge(viewpoint_id, node_id, EdgeType.OBSERVED_AT)
            self._nodes[node_id].attributes["best_approach_position"] = self._compute_best_approach(node_id).tolist()
            return node_id, False

        self._merge_object_observation(
            match,
            bbox=bbox,
            confidence=confidence,
            position=pos,
            embedding=emb,
            viewpoint_id=viewpoint_id,
            view_heading=view_heading,
            room_context=room_context,
            source=source,
            target_relevance=target_relevance,
            room_prior_score=room_prior_score,
        )
        if self._should_add_observation_edge(match.node_id, viewpoint_id, room_context):
            self.add_edge(viewpoint_id, match.node_id, EdgeType.OBSERVED_AT)
        match.attributes["best_approach_position"] = self._compute_best_approach(match.node_id).tolist()
        return match.node_id, True

    def _best_object_match(
        self,
        candidates: List[SemanticNode],
        label: str,
        bbox: List[float],
        embedding: Optional[np.ndarray],
        view_heading: float,
        position: np.ndarray,
        room_context: Optional[str] = None,
    ) -> Optional[SemanticNode]:
        best = None
        best_score = -1.0
        for node in candidates:
            if node.label != label:
                continue
            if node.attributes.get("folded"):
                continue
            if not self._room_context_compatible(node, room_context):
                continue
            last_bbox = node.attributes.get("last_bbox") or bbox
            bbox_score = _bbox_iou(last_bbox, bbox)
            heading = float(node.attributes.get("last_view_heading", view_heading))
            heading_score = 1.0 if abs(_angle_delta(heading, view_heading)) <= 0.45 else 0.0
            emb_score = _embedding_similarity(node.embedding, embedding)
            dist = float(np.linalg.norm(node.position - position))
            score = max(bbox_score, heading_score * 0.8, emb_score)
            if dist < self.merge_radius:
                score = max(score, 0.5)
            if score > best_score and score >= 0.45:
                best = node
                best_score = score
        return best

    def _room_context_compatible(self, node: SemanticNode, room_context: Optional[str]) -> bool:
        if room_context is None:
            return True
        current = str(room_context).strip()
        if not current or current == "unknown":
            return True
        known_rooms = node.attributes.get("room_contexts")
        if known_rooms is None:
            previous = node.attributes.get("room_context")
            known_rooms = [previous] if previous is not None else []
        known = {str(room).strip() for room in known_rooms if room is not None and str(room).strip() and str(room).strip() != "unknown"}
        return not known or current in known

    def _should_add_observation_edge(
        self,
        object_id: str,
        viewpoint_id: Optional[str],
        room_context: Optional[str],
    ) -> bool:
        if viewpoint_id is None:
            return False
        obj = self._nodes.get(object_id)
        viewpoint = self._nodes.get(viewpoint_id)
        if obj is None or viewpoint is None:
            return False
        if not self._room_context_compatible(obj, room_context):
            return False
        dist = float(np.linalg.norm(obj.position - viewpoint.position))
        if room_context is not None and str(room_context).strip() and str(room_context).strip() != "unknown":
            return dist <= self.far_radius
        return dist <= max(self.near_radius * 1.5, self.merge_radius * 3.0)

    def _new_object_attributes(
        self,
        *,
        bbox: List[float],
        confidence: float,
        viewpoint_id: Optional[str],
        view_heading: float,
        room_context: Optional[str],
        source: str,
        target_relevance: float,
        room_prior_score: float,
    ) -> Dict[str, Any]:
        obs = {
            "bbox": [float(v) for v in bbox],
            "confidence": float(confidence),
            "viewpoint_id": viewpoint_id,
            "view_heading": float(view_heading),
            "step_id": self._current_step,
            "source": source,
        }
        return {
            "bbox_observations": [obs],
            "detection_scores": [float(confidence)],
            "viewpoints": [viewpoint_id] if viewpoint_id is not None else [],
            "first_seen_step": self._current_step,
            "last_seen_step": self._current_step,
            "multi_view_count": 1,
            "room_context": room_context,
            "room_contexts": [room_context] if room_context is not None else [],
            "granularity": "object",
            "last_bbox": [float(v) for v in bbox],
            "last_view_heading": float(view_heading),
            "target_relevance": float(target_relevance),
            "room_prior_score": float(room_prior_score),
            "redundancy_penalty": 0.0,
            "conflict_penalty": 0.0,
            "source": source,
        }

    def _merge_object_observation(
        self,
        node: SemanticNode,
        *,
        bbox: List[float],
        confidence: float,
        position: np.ndarray,
        embedding: Optional[np.ndarray],
        viewpoint_id: Optional[str],
        view_heading: float,
        room_context: Optional[str],
        source: str,
        target_relevance: float,
        room_prior_score: float,
    ) -> None:
        attrs = node.attributes
        obs = {
            "bbox": [float(v) for v in bbox],
            "confidence": float(confidence),
            "viewpoint_id": viewpoint_id,
            "view_heading": float(view_heading),
            "step_id": self._current_step,
            "source": source,
        }
        attrs.setdefault("bbox_observations", []).append(obs)
        attrs.setdefault("detection_scores", []).append(float(confidence))
        if viewpoint_id is not None and viewpoint_id not in attrs.setdefault("viewpoints", []):
            attrs["viewpoints"].append(viewpoint_id)
        attrs["last_seen_step"] = self._current_step
        attrs["last_bbox"] = [float(v) for v in bbox]
        attrs["last_view_heading"] = float(view_heading)
        attrs["multi_view_count"] = max(1, len(attrs.get("viewpoints", [])), len(attrs.get("bbox_observations", [])))
        if room_context is not None:
            attrs["room_context"] = room_context
            room_contexts = attrs.setdefault("room_contexts", [])
            if room_context not in room_contexts:
                room_contexts.append(room_context)
        attrs["target_relevance"] = max(float(attrs.get("target_relevance", 0.0)), float(target_relevance))
        attrs["room_prior_score"] = max(float(attrs.get("room_prior_score", 0.0)), float(room_prior_score))
        low_scores = [s for s in attrs.get("detection_scores", []) if float(s) < 0.35]
        attrs["redundancy_penalty"] = max(0.0, (len(low_scores) - 1) / 5.0)
        attrs["conflict_penalty"] = self._nearby_conflict_penalty(node, position)
        node.position = (node.position * max(1, node.visit_count) + position) / (max(1, node.visit_count) + 1)
        node.visit_count += 1
        if embedding is not None:
            if node.embedding is None:
                node.embedding = embedding
            elif confidence >= max(attrs.get("detection_scores", [confidence])):
                node.embedding = embedding
            else:
                node.embedding = (0.8 * node.embedding + 0.2 * embedding).astype(np.float32)
        node.step_id = self._current_step
        node.confidence = compute_semantic_confidence(ConfidenceFactors(
            detection_score=max(float(s) for s in attrs.get("detection_scores", [confidence])),
            multi_view_count=int(attrs["multi_view_count"]),
            task_relevance=float(attrs.get("target_relevance", 0.0)),
            room_prior_score=float(attrs.get("room_prior_score", 0.0)),
            redundancy_penalty=float(attrs.get("redundancy_penalty", 0.0)),
            conflict_penalty=float(attrs.get("conflict_penalty", 0.0)),
        ))

    def _compute_best_approach(self, object_node_id: str) -> np.ndarray:
        """Pick the best approach position for an OBJECT node.

        Selects the closest observed viewpoint (WAYPOINT_VISITED) to the
        object.  Falls back to the object position itself when no valid
        viewpoint is recorded.
        """
        node = self._nodes.get(object_node_id)
        if node is None:
            return np.zeros(3, dtype=np.float32)
        obj_pos = node.position
        viewpoint_ids = node.attributes.get("viewpoints", [])
        best_pos = obj_pos.copy()
        best_dist = float("inf")
        for vp_id in viewpoint_ids:
            vp = self._nodes.get(vp_id)
            if vp is None:
                continue
            d = float(np.linalg.norm(vp.position - obj_pos))
            if d < best_dist:
                best_dist = d
                best_pos = vp.position.copy()
        return best_pos

    def _nearby_conflict_penalty(self, node: SemanticNode, position: np.ndarray) -> float:
        conflicts = 0
        for other in self.find_nodes_within_radius(position, self.merge_radius, NodeType.OBJECT):
            if other.node_id != node.node_id and other.label != node.label:
                conflicts += 1
        return min(1.0, conflicts / 3.0)

    # ==================== Memory Management ====================

    def add_candidate_waypoint(
        self,
        position: np.ndarray,
        label: str = "",
        confidence: float = 0.35,
        source: str = "ghost",
    ) -> str:
        """Add an ETPNav-style candidate/ghost waypoint."""
        return self.add_node(
            NodeType.WAYPOINT_CANDIDATE,
            position,
            confidence=confidence,
            label=label,
            attributes={
                "source": source,
                "consumed": False,
                "blocked": False,
                "state": "candidate",
            },
        )

    def _set_node_type(self, node: SemanticNode, node_type: NodeType) -> None:
        node.node_type = node_type
        if node.node_id in self.graph.nodes:
            self.graph.nodes[node.node_id]["node_type"] = node_type.value

    def _merge_object_into_landmark(self, landmark: SemanticNode, obj: SemanticNode) -> None:
        landmark.confidence = max(float(landmark.confidence), float(obj.confidence))
        absorbed = landmark.attributes.setdefault("absorbed_object_ids", [])
        if obj.node_id not in absorbed:
            absorbed.append(obj.node_id)
        labels = landmark.attributes.setdefault("absorbed_labels", [])
        if obj.label and obj.label not in labels:
            labels.append(obj.label)
        landmark.position = (
            (landmark.position * max(1, landmark.visit_count))
            + obj.position
        ) / (max(1, landmark.visit_count) + 1)
        landmark.visit_count = max(int(landmark.visit_count), int(obj.visit_count))
        landmark.step_id = max(int(landmark.step_id), int(obj.step_id))
        if obj.embedding is not None and landmark.embedding is None:
            landmark.embedding = obj.embedding

    def _promote_object_to_landmark_node(self, node: SemanticNode) -> None:
        attrs = node.attributes
        viewpoints = list(attrs.get("viewpoints", []))
        observations = []
        for vp in viewpoints[: self.landmark_history_keep_recent]:
            observations.append({
                "viewpoint_id": vp,
                "confidence": float(attrs.get("history_best_confidence", node.confidence)),
                "step_id": int(attrs.get("last_seen_step", node.step_id)),
                "source": "promoted_object",
            })
        if not observations:
            observations.append({
                "viewpoint_id": None,
                "confidence": float(node.confidence),
                "step_id": int(node.step_id),
                "source": "promoted_object",
            })
        attrs["landmark_source"] = "promoted_object"
        attrs["promoted_from_object"] = True
        attrs["observations"] = observations
        attrs["viewpoints"] = [obs["viewpoint_id"] for obs in observations if obs.get("viewpoint_id")]
        attrs["granularity"] = "landmark"
        attrs["folded"] = False
        # Tag with structural/semantic role so downstream consumers (skeleton,
        # persistence, planner) can treat structural anchors differently from
        # room-interior props.
        attrs["landmark_role"] = classify_landmark_role(node.label, attrs)
        self._set_node_type(node, NodeType.LANDMARK)

    def _fuse_object_to_landmark(self, node: SemanticNode) -> Optional[str]:
        """Mid-distance fusion: object -> navigation landmark anchor."""
        if node.node_type != NodeType.OBJECT:
            return None
        if node.attributes.get("granularity") != "landmark":
            return None
        if node.attributes.get("fused_into_landmark_id"):
            return node.attributes["fused_into_landmark_id"]

        attrs = node.attributes
        candidates = self.find_nodes_within_radius(
            node.position, self.merge_radius * 3.0, NodeType.LANDMARK,
        )
        matched = next(
            (
                lm for lm in candidates
                if lm.label == node.label
                and lm.attributes.get("landmark_source") != "environment"
            ),
            None,
        )
        if matched is not None:
            self._merge_object_into_landmark(matched, node)
            attrs["fused_into_landmark_id"] = matched.node_id
            attrs["folded"] = True
            attrs["folded_reason"] = "fused_into_landmark"
            self.add_edge(node.node_id, matched.node_id, EdgeType.BELONGS_TO)
            return matched.node_id

        has_bbox = bool(attrs.get("bbox_observations"))
        has_heavy = attrs.get("source") in ("groundingdino", "heavy")
        target_relevance = float(attrs.get("target_relevance", 0.0))
        multi_view = int(
            attrs.get("multi_view_count")
            or attrs.get("history_view_count")
            or len(attrs.get("viewpoints", []))
            or max(1, int(node.visit_count))
        )
        allowed, role, reason = can_promote_object_to_landmark(
            label=node.label,
            confidence=float(node.confidence),
            multi_view_count=multi_view,
            target_relevance=target_relevance,
            source=str(attrs.get("source", "")),
            has_bbox=has_bbox,
        )
        attrs["promotion_check"] = {
            "allowed": allowed,
            "role": role,
            "reason": reason,
            "confidence": float(node.confidence),
            "multi_view": multi_view,
            "target_relevance": target_relevance,
        }
        if allowed:
            self._promote_object_to_landmark_node(node)
            # promote_check wrote landmark_role; double-check using the
            # final attrs in case label changed.
            attrs["landmark_role"] = role
            return node.node_id
        return None

    def promote_frontier_to_visited(self, node_id: str):
        """When agent visits a frontier/candidate, promote it to visited waypoint."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
                promoted_from = "candidate" if node.node_type == NodeType.WAYPOINT_CANDIDATE else "frontier"
                node.node_type = NodeType.WAYPOINT_VISITED
                node.visit_count += 1
                node.confidence = min(1.0, node.confidence + 0.3)
                node.step_id = self._current_step
                node.attributes["promoted_from"] = promoted_from
                node.attributes["state"] = "visited"
                node.attributes["consumed"] = False
                node.attributes["blocked"] = False
                self.graph.nodes[node_id]["node_type"] = NodeType.WAYPOINT_VISITED.value

    def consume_node(self, node_id: str, reason: str):
        """Mark a candidate/frontier target as consumed so planning will skip it."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.attributes["consumed"] = True
        node.attributes["consume_reason"] = reason
        node.attributes["state"] = "consumed"
        node.step_id = self._current_step
        if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
            node.confidence = max(0.0, node.confidence - 0.25)

    def block_node(self, node_id: str, reason: str, until_step: Optional[int] = None):
        """Temporarily block a problematic target from target selection."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.attributes["blocked"] = True
        node.attributes["blocked_reason"] = reason
        node.attributes["blocked_until_step"] = until_step
        node.attributes["state"] = "blocked"
        node.step_id = self._current_step
        node.confidence = max(0.0, node.confidence - 0.15)

    def merge_nearby_nodes(self, node_type: Optional[NodeType] = None):
        """Merge nodes of same type that are too close together."""
        nodes = list(self._nodes.values())
        if node_type is not None:
            nodes = [n for n in nodes if n.node_type == node_type]

        merged: Set[str] = set()
        for i, node_a in enumerate(nodes):
            if node_a.node_id in merged:
                continue
            for node_b in nodes[i + 1:]:
                if node_b.node_id in merged:
                    continue
                if node_a.node_type != node_b.node_type:
                    continue
                dist = np.linalg.norm(node_a.position - node_b.position)
                if dist < self.merge_radius:
                    # Keep the higher confidence one
                    if node_b.confidence > node_a.confidence:
                        node_a, node_b = node_b, node_a
                    node_a.confidence = min(1.0, node_a.confidence + 0.1)
                    node_a.visit_count += node_b.visit_count
                    # Transfer edges
                    for neighbor in list(self.graph.neighbors(node_b.node_id)):
                        if neighbor != node_a.node_id:
                            edge_data = self.graph.edges[node_b.node_id, neighbor]
                            self.graph.add_edge(node_a.node_id, neighbor, **edge_data)
                    self.remove_node(node_b.node_id)
                    merged.add(node_b.node_id)

    def _is_persistent_structure_node(self, node: SemanticNode) -> bool:
        """Bottom-layer spatial anchors that should survive long-range navigation.

        Only ROOM summaries and **structural** landmarks (door / corridor /
        stair / ...) are unconditionally persistent. Semantic landmarks
        (chair, tv, ...) survive only when they remain task-relevant; this
        keeps the long-term skeleton focused on room-to-room anchors.
        """
        if node.node_type == NodeType.ROOM:
            return node.attributes.get("summary_type") == "room_region"
        if node.node_type != NodeType.LANDMARK:
            return False
        source = node.attributes.get("landmark_source", "")
        if source == "goal_hint":
            return True
        role = classify_landmark_role(node.label, node.attributes)
        if role == "structural":
            return source in ("promoted_object", "environment", "goal_hint") or bool(
                node.attributes.get("promoted_from_object")
            )
        # semantic landmark: only persist while task-relevant
        return float(node.attributes.get("target_relevance", 0.0)) > 0.0

    def prune_low_confidence(self, agent_pos: Optional[np.ndarray] = None):
        """Remove nodes with confidence below distance-aware thresholds."""
        to_remove = []
        pos = np.array(agent_pos, dtype=np.float32) if agent_pos is not None else None
        for node in self._nodes.values():
            if node.node_type == NodeType.WAYPOINT_VISITED:
                continue
            if self._is_persistent_structure_node(node):
                continue
            if node.attributes.get("is_semantic_anchor"):
                continue
            if float(node.attributes.get("target_relevance", 0.0)) > 0:
                continue
            if pos is not None and node.node_type in (NodeType.OBJECT, NodeType.LANDMARK):
                dist = float(np.linalg.norm(node.position - pos))
                if dist > self.far_prune_distance:
                    threshold = self.far_prune_threshold
                elif dist > self.mid_prune_distance:
                    threshold = self.mid_prune_threshold
                else:
                    threshold = self.prune_threshold
            else:
                threshold = self.prune_threshold
            if node.confidence < threshold:
                to_remove.append(node.node_id)
        for nid in to_remove:
            self.remove_node(nid)

    def adaptive_granularity(self, agent_pos: np.ndarray):
        """Compress distant semantic details into room/region summaries."""
        pos = np.array(agent_pos, dtype=np.float32)
        debug = {
            "object_nodes": 0,
            "far_object_candidates": 0,
            "object_room_level_updates": 0,
            "object_detail_kept": 0,
            "object_landmark_updates": 0,
            "landmark_nodes": 0,
            "landmark_room_level_updates": 0,
            "summary_updates": 0,
            "folded_anchor_marks": 0,
        }
        for node in list(self._nodes.values()):
            dist = float(np.linalg.norm(node.position - pos))
            if node.node_type == NodeType.OBJECT:
                debug["object_nodes"] += 1
                detail_score = self._semantic_detail_score(node, pos)
                node.attributes["detail_score"] = detail_score
                # Staleness decay: mid-range objects not seen in 3+ steps get reduced detail_score
                if dist > self.near_radius and dist <= self.room_level_min_distance:
                    last_seen = int(node.attributes.get("last_seen_step", node.step_id))
                    steps_since = self._current_step - last_seen
                    if steps_since > 2:
                        decay = max(0.35, 1.0 - 0.10 * (steps_since - 2))
                        detail_score *= decay
                if dist <= self.near_radius:
                    node.attributes["granularity"] = "object"
                    debug["object_detail_kept"] += 1
                    continue
                if dist > self.room_level_min_distance:
                    debug["far_object_candidates"] += 1
                    node.attributes["granularity"] = "room_level"
                    self._compress_object_history(node, "far_low_confidence")
                    summary_id = self._add_node_to_room_summary(node, "far_low_confidence")
                    debug["object_room_level_updates"] += 1
                    if summary_id is not None:
                        node.attributes["fused_into_summary_id"] = summary_id
                        debug["summary_updates"] += 1
                    self._mark_node_folded_anchor(node, "room_level", summary_id)
                    debug["folded_anchor_marks"] += 1
                elif detail_score >= self.summary_mid_detail_threshold:
                    node.attributes["granularity"] = "object"
                    debug["object_detail_kept"] += 1
                else:
                    node.attributes["granularity"] = "landmark"
                    self._compress_object_history(node, "mid_or_far_object")
                    self._fuse_object_to_landmark(node)
                    debug["object_landmark_updates"] += 1
            elif node.node_type == NodeType.LANDMARK and dist > self.near_radius:
                debug["landmark_nodes"] += 1
                detail_score = self._semantic_detail_score(node, pos)
                node.attributes["detail_score"] = detail_score
                if dist >= self.room_level_min_distance and detail_score < self.room_level_detail_max:
                    node.attributes["granularity"] = "room_level"
                    debug["landmark_room_level_updates"] += 1
                else:
                    node.attributes["granularity"] = "landmark"
                self._compress_landmark_history(node, "far_landmark")
                if dist >= self.summary_min_distance or detail_score < self.summary_mid_detail_threshold:
                    summary_id = self._add_node_to_room_summary(node, "far_landmark")
                    if summary_id is not None:
                        node.attributes["fused_into_summary_id"] = summary_id
                        debug["summary_updates"] += 1
        self._last_granularity_debug = debug
        self._update_node_visibility(pos)
        self._maintain_spatial_structure_graph()
        if getattr(self, "waypoint_compress_enabled", True):
            self.compress_distant_waypoints(pos)

    def _waypoint_nav_neighbors(self, waypoint_id: str) -> List[str]:
        neighbors = []
        for neighbor in self.get_neighbors(waypoint_id, EdgeType.NAVIGABLE):
            node = self._nodes.get(neighbor)
            if node is not None and node.node_type == NodeType.WAYPOINT_VISITED:
                neighbors.append(neighbor)
        return neighbors

    def _entrance_waypoint_ids(self, rooms: List[SemanticNode]) -> Set[str]:
        """Waypoints at navigable room-boundary crossings."""
        waypoints = {
            node.node_id: node
            for node in self.get_nodes_by_type(NodeType.WAYPOINT_VISITED)
        }
        entrances: Set[str] = set()
        for node_a, node_b, data in self.graph.edges(data=True):
            if data.get("edge_type") != EdgeType.NAVIGABLE.value:
                continue
            if node_a not in waypoints or node_b not in waypoints:
                continue
            room_a = self._room_for_position(waypoints[node_a].position, rooms)
            room_b = self._room_for_position(waypoints[node_b].position, rooms)
            if room_a is None or room_b is None or room_a.node_id == room_b.node_id:
                continue
            entrances.add(node_a)
            entrances.add(node_b)
        return entrances

    def _is_collinear_waypoint(
        self,
        prev_node: SemanticNode,
        mid_node: SemanticNode,
        next_node: SemanticNode,
        max_angle_deg: float,
    ) -> bool:
        vec_a = prev_node.position - mid_node.position
        vec_b = next_node.position - mid_node.position
        planar_a = vec_a[[0, 2]]
        planar_b = vec_b[[0, 2]]
        norm_a = float(np.linalg.norm(planar_a))
        norm_b = float(np.linalg.norm(planar_b))
        if norm_a < 1e-4 or norm_b < 1e-4:
            return True
        cosine = float(np.dot(planar_a, planar_b) / (norm_a * norm_b))
        cosine = float(np.clip(cosine, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        return angle >= (180.0 - max_angle_deg)

    def _redirect_waypoint_edges(self, waypoint_id: str, keep_id: str) -> None:
        """Move non-navigable edges from a waypoint onto a kept anchor."""
        for neighbor in list(self.graph.neighbors(waypoint_id)):
            if neighbor == keep_id:
                continue
            edge_data = dict(self.graph.edges[waypoint_id, neighbor])
            edge_type_value = edge_data.get("edge_type", EdgeType.OBSERVED_AT.value)
            weight = float(edge_data.get("weight", 1.0))
            if self.graph.has_edge(keep_id, neighbor):
                existing = self.graph.edges[keep_id, neighbor]
                if existing.get("edge_type") == edge_type_value:
                    continue
            edge_type = EdgeType(edge_type_value)
            self.add_edge(keep_id, neighbor, edge_type, weight=weight)

    def _bridge_waypoints(self, node_a: str, node_b: str, via_weight: float) -> None:
        if node_a == node_b:
            return
        if self.graph.has_edge(node_a, node_b):
            data = self.graph.edges[node_a, node_b]
            if data.get("edge_type") == EdgeType.NAVIGABLE.value:
                data["weight"] = float(data.get("weight", via_weight)) + float(via_weight)
            return
        self.add_edge(node_a, node_b, EdgeType.NAVIGABLE, weight=float(via_weight))

    def _collapse_corridor_waypoint(self, waypoint_id: str, prev_id: str, next_id: str) -> None:
        waypoint = self._nodes.get(waypoint_id)
        prev_node = self._nodes.get(prev_id)
        next_node = self._nodes.get(next_id)
        if waypoint is None or prev_node is None or next_node is None:
            return

        weight_prev = float(self.graph.edges[waypoint_id, prev_id].get(
            "weight",
            np.linalg.norm(waypoint.position - prev_node.position),
        ))
        weight_next = float(self.graph.edges[waypoint_id, next_id].get(
            "weight",
            np.linalg.norm(waypoint.position - next_node.position),
        ))

        anchor_id = prev_id
        anchor = prev_node
        anchor.attributes["compressed_waypoint_count"] = int(
            anchor.attributes.get("compressed_waypoint_count", 0)
        ) + 1 + int(waypoint.attributes.get("compressed_waypoint_count", 0))
        if anchor.attributes.get("waypoint_role") != "entrance":
            anchor.attributes["waypoint_role"] = "corridor_anchor"

        self._redirect_waypoint_edges(waypoint_id, anchor_id)
        self._bridge_waypoints(prev_id, next_id, weight_prev + weight_next)
        self.remove_node(waypoint_id)

    def _mark_waypoint_roles(self, entrance_ids: Set[str]) -> None:
        for waypoint in self.get_nodes_by_type(NodeType.WAYPOINT_VISITED):
            if waypoint.node_id in entrance_ids:
                waypoint.attributes["waypoint_role"] = "entrance"
            elif int(waypoint.attributes.get("compressed_waypoint_count", 0)) > 0:
                if waypoint.attributes.get("waypoint_role") != "entrance":
                    waypoint.attributes["waypoint_role"] = "corridor_anchor"

    def compress_distant_waypoints(self, agent_pos: np.ndarray) -> int:
        """Collapse far collinear corridor waypoints; keep entrances and junctions."""
        pos = np.array(agent_pos, dtype=np.float32)
        compress_dist = float(self.waypoint_compress_distance)
        keep_near = float(self.waypoint_compress_keep_near)
        collinear_deg = float(self.waypoint_compress_collinear_deg)
        rooms = self._room_region_summaries()
        entrance_ids = self._entrance_waypoint_ids(rooms)
        self._mark_waypoint_roles(entrance_ids)

        removed = 0
        max_passes = max(8, len(self.get_visited()))
        for _ in range(max_passes):
            removed_this_pass = 0
            for waypoint in list(self.get_nodes_by_type(NodeType.WAYPOINT_VISITED)):
                waypoint_id = waypoint.node_id
                if waypoint_id in entrance_ids:
                    continue
                dist = float(np.linalg.norm(waypoint.position - pos))
                if dist <= keep_near:
                    continue
                if dist < compress_dist:
                    continue
                neighbors = self._waypoint_nav_neighbors(waypoint_id)
                if len(neighbors) != 2:
                    continue
                prev_id, next_id = neighbors
                prev_node = self._nodes.get(prev_id)
                next_node = self._nodes.get(next_id)
                if prev_node is None or next_node is None:
                    continue
                if not self._is_collinear_waypoint(prev_node, waypoint, next_node, collinear_deg):
                    continue
                self._collapse_corridor_waypoint(waypoint_id, prev_id, next_id)
                removed += 1
                removed_this_pass += 1
            if removed_this_pass == 0:
                break

        self._mark_waypoint_roles(entrance_ids)
        return removed

    def _room_region_summaries(self) -> List[SemanticNode]:
        return [
            room for room in self.get_nodes_by_type(NodeType.ROOM)
            if room.attributes.get("summary_type") == "room_region"
        ]

    def _room_for_position(
        self,
        position: np.ndarray,
        rooms: Optional[List[SemanticNode]] = None,
    ) -> Optional[SemanticNode]:
        rooms = rooms if rooms is not None else self._room_region_summaries()
        if not rooms:
            return None
        pos = np.array(position, dtype=np.float32)
        expanded = self.summary_radius * 1.8
        candidates = [
            room for room in rooms
            if float(np.linalg.norm(room.position - pos)) <= expanded
        ]
        pool = candidates if candidates else rooms
        return min(pool, key=lambda room: float(np.linalg.norm(room.position - pos)))

    def _waypoint_region_label(self, waypoint: SemanticNode) -> Optional[str]:
        return normalize_region_label(waypoint.attributes.get("view_room_label"))

    def _topo_dict_snapshot(self) -> Tuple[List[dict], List[dict]]:
        nodes = [
            {
                "id": node.node_id,
                "type": node.node_type.value,
                "position": node.position.tolist(),
                "label": node.label,
                "attributes": dict(node.attributes),
            }
            for node in self._nodes.values()
        ]
        edges = [
            {
                "source": node_a,
                "target": node_b,
                "type": data.get("edge_type", ""),
            }
            for node_a, node_b, data in self.graph.edges(data=True)
        ]
        return nodes, edges

    # ------------------------------------------------------------------
    # Double-layer views (Phase 2)
    # ------------------------------------------------------------------
    # We do NOT split the underlying graph. Instead, we expose read-only
    # views that select node/edge subsets relevant to each layer:
    #
    #   * navigation layer  -> waypoints (visited / frontier / candidate)
    #                          + NAVIGABLE edges
    #   * structure layer   -> rooms + portals + structural landmarks
    #                          + ADJACENT_TO / BELONGS_TO edges between them
    #
    # Consumers (planner, viz, debug) should prefer these views over
    # touching ``self.graph`` directly so that future storage refactors do
    # not ripple through the codebase.

    _NAV_NODE_TYPES = (
        NodeType.WAYPOINT_VISITED,
        NodeType.WAYPOINT_FRONTIER,
        NodeType.WAYPOINT_CANDIDATE,
    )

    def _is_structural_landmark_node(self, node: SemanticNode) -> bool:
        if node.node_type != NodeType.LANDMARK:
            return False
        attrs = node.attributes
        if attrs.get("synthetic_portal") or attrs.get("structure_role") == "portal":
            return True
        if attrs.get("landmark_source") == "environment" or attrs.get("source") == "environment":
            return False
        return classify_landmark_role(node.label, attrs) == "structural"

    def navigation_layer_nodes(self) -> List[SemanticNode]:
        """Return the navigation layer node set (waypoints only)."""
        return [
            node for node in self._nodes.values()
            if node.node_type in self._NAV_NODE_TYPES
        ]

    def navigation_layer_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return NAVIGABLE edges between waypoint nodes."""
        nav_ids = {node.node_id for node in self.navigation_layer_nodes()}
        edges: List[Tuple[str, str, Dict[str, Any]]] = []
        for node_a, node_b, data in self.graph.edges(data=True):
            if data.get("edge_type") != EdgeType.NAVIGABLE.value:
                continue
            if node_a in nav_ids and node_b in nav_ids:
                edges.append((node_a, node_b, dict(data)))
        return edges

    def structure_layer_nodes(self) -> List[SemanticNode]:
        """Rooms + portals + structural landmarks that form the skeleton."""
        out: List[SemanticNode] = []
        for node in self._nodes.values():
            if node.node_type == NodeType.ROOM and node.attributes.get("summary_type") == "room_region":
                out.append(node)
            elif self._is_structural_landmark_node(node):
                out.append(node)
        return out

    def structure_layer_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """ADJACENT_TO / BELONGS_TO edges within the structure layer."""
        struct_ids = {node.node_id for node in self.structure_layer_nodes()}
        wanted = (EdgeType.ADJACENT_TO.value, EdgeType.BELONGS_TO.value)
        edges: List[Tuple[str, str, Dict[str, Any]]] = []
        for node_a, node_b, data in self.graph.edges(data=True):
            if data.get("edge_type") not in wanted:
                continue
            if node_a in struct_ids and node_b in struct_ids:
                edges.append((node_a, node_b, dict(data)))
        return edges

    def cross_layer_waypoint_room_edges(self) -> List[Tuple[str, str]]:
        """Cross-layer BELONGS_TO edges binding waypoints to rooms."""
        rooms = {
            node.node_id for node in self._nodes.values()
            if node.node_type == NodeType.ROOM
            and node.attributes.get("summary_type") == "room_region"
        }
        waypoints = {node.node_id for node in self.navigation_layer_nodes()}
        out: List[Tuple[str, str]] = []
        for node_a, node_b, data in self.graph.edges(data=True):
            if data.get("edge_type") != EdgeType.BELONGS_TO.value:
                continue
            if node_a in waypoints and node_b in rooms:
                out.append((node_a, node_b))
            elif node_b in waypoints and node_a in rooms:
                out.append((node_b, node_a))
        return out

    def get_navigation_view(self) -> Dict[str, Any]:
        """Snapshot of the navigation layer (waypoints + navigable edges)."""
        nodes = self.navigation_layer_nodes()
        edges = self.navigation_layer_edges()
        return {
            "layer": "navigation",
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type.value,
                    "position": n.position.tolist(),
                    "label": n.label,
                    "attributes": dict(n.attributes),
                }
                for n in nodes
            ],
            "edges": [
                {"source": a, "target": b, "type": data.get("edge_type", ""),
                 "weight": float(data.get("weight", 1.0))}
                for a, b, data in edges
            ],
        }

    def get_structure_view(self) -> Dict[str, Any]:
        """Snapshot of the structural skeleton layer."""
        nodes = self.structure_layer_nodes()
        edges = self.structure_layer_edges()
        cross = self.cross_layer_waypoint_room_edges()
        return {
            "layer": "structure",
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type.value,
                    "position": n.position.tolist(),
                    "label": n.label,
                    "attributes": dict(n.attributes),
                }
                for n in nodes
            ],
            "edges": [
                {"source": a, "target": b, "type": data.get("edge_type", ""),
                 "weight": float(data.get("weight", 1.0))}
                for a, b, data in edges
            ],
            "waypoint_room_bindings": [
                {"waypoint_id": wp, "room_id": room} for wp, room in cross
            ],
        }

    def _room_for_transition(
        self,
        position: np.ndarray,
        rooms: Optional[List[SemanticNode]] = None,
        waypoint: Optional[SemanticNode] = None,
    ) -> Optional[SemanticNode]:
        """Assign a waypoint to the nearest matching region instance."""
        rooms = rooms if rooms is not None else self._sync_structure_rooms_from_waypoints()
        if not rooms:
            return None
        if waypoint is not None:
            nodes, edges = self._topo_dict_snapshot()
            wp_dict = {
                "id": waypoint.node_id,
                "position": waypoint.position.tolist(),
                "attributes": dict(waypoint.attributes),
            }
            room_dicts = [
                {
                    "id": room.node_id,
                    "label": room.label,
                    "position": room.position.tolist(),
                    "attributes": dict(room.attributes),
                }
                for room in rooms
            ]
            matched = room_for_labeled_waypoint(wp_dict, room_dicts, nodes=nodes, edges=edges)
            if matched is not None:
                return self._nodes.get(matched["id"])
        if waypoint is not None and self._waypoint_region_label(waypoint):
            return None
        pos = np.array(position, dtype=np.float32)
        strict = [
            room for room in rooms
            if float(np.linalg.norm(room.position - pos)) <= self.summary_radius
        ]
        if len(strict) == 1:
            return strict[0]
        if strict:
            return min(strict, key=lambda room: float(np.linalg.norm(room.position - pos)))
        return None

    def _sync_structure_rooms_from_waypoints(self) -> List[SemanticNode]:
        """Region rooms: one instance per spatial cluster within each region label."""
        waypoints = self.get_nodes_by_type(NodeType.WAYPOINT_VISITED)
        if not waypoints:
            return self._room_region_summaries()

        nodes, edges = self._topo_dict_snapshot()
        wp_dicts = [
            {
                "id": waypoint.node_id,
                "position": waypoint.position.tolist(),
                "attributes": dict(waypoint.attributes),
            }
            for waypoint in waypoints
        ]
        instances = region_instances_from_waypoints(wp_dicts, nodes=nodes, edges=edges)
        if not instances:
            return self._room_region_summaries()

        by_base: Dict[str, SemanticNode] = {}
        for room in self._room_region_summaries():
            key = normalize_region_label(
                room.attributes.get("base_label") or room.label
            ) or str(room.label or "").strip().lower()
            if key and key not in by_base:
                by_base[key] = room

        result: List[SemanticNode] = []
        keep_ids: set = set()
        for idx, inst in enumerate(instances):
            base_label = inst["base_label"]
            display = inst["label"]
            centroid = inst["centroid"]
            room_id = f"region::{base_label}::{idx}"
            existing = self._nodes.get(room_id)
            if existing is None:
                room_id = self.add_node(
                    NodeType.ROOM,
                    position=centroid,
                    confidence=0.65,
                    label=display,
                    node_id=room_id,
                    attributes={
                        "summary_type": "room_region",
                        "region_source": "waypoint_cluster",
                        "base_label": base_label,
                        "region_instance": inst["instance_suffix"],
                        "waypoint_ids": inst["waypoint_ids"],
                        "contains_labels": [],
                        "contains_node_ids": [],
                        "summary_observations": [],
                        "source_granularities": [],
                    },
                )
                room = self._nodes[room_id]
            else:
                room = existing
                room.position = centroid
                room.label = display
                room.attributes["base_label"] = base_label
                room.attributes["region_instance"] = inst["instance_suffix"]
                room.attributes["waypoint_ids"] = inst["waypoint_ids"]
                room.attributes["region_source"] = "waypoint_cluster"
            keep_ids.add(room.node_id)
            result.append(room)

        for room in list(self._room_region_summaries()):
            if room.node_id not in keep_ids:
                src = room.attributes.get("region_source")
                if src == "waypoint_cluster" or str(room.node_id).startswith("region::"):
                    self.remove_node(room.node_id)

        self._merge_alias_region_rooms(result)
        return result

    def _merge_alias_region_rooms(self, canonical: List[SemanticNode]) -> None:
        """Drop duplicate summaries that alias to the same canonical region."""
        keep_ids = {room.node_id for room in canonical}
        canonical_labels = {
            normalize_region_label(room.label) or str(room.label or "").strip().lower()
            for room in canonical
        }
        for room in list(self._room_region_summaries()):
            if room.node_id in keep_ids:
                continue
            norm = normalize_region_label(room.label) or str(room.label or "").strip().lower()
            if norm in canonical_labels:
                self.remove_node(room.node_id)

    def _portal_label_score(self, label: str) -> int:
        lowered = str(label or "").strip().lower()
        for idx, token in enumerate(_PORTAL_LABEL_PRIORITY):
            if token in lowered:
                return idx
        return len(_PORTAL_LABEL_PRIORITY)

    def _pick_portal_landmark(
        self,
        midpoint: np.ndarray,
        nav_landmarks: List[SemanticNode],
        max_dist: float,
        room_a: Optional[SemanticNode] = None,
        room_b: Optional[SemanticNode] = None,
    ) -> Optional[SemanticNode]:
        pos = np.array(midpoint, dtype=np.float32)
        candidates = [
            landmark for landmark in nav_landmarks
            if float(np.linalg.norm(landmark.position - pos)) <= max_dist
        ]
        if not candidates:
            return None

        def _betweenness(landmark: SemanticNode) -> float:
            if room_a is None or room_b is None:
                return 0.0
            ap = room_a.position[[0, 2]]
            bp = room_b.position[[0, 2]]
            lp = landmark.position[[0, 2]]
            ab = bp - ap
            denom = float(np.dot(ab, ab))
            if denom < 1e-6:
                return float(np.linalg.norm(lp - ap))
            t = float(np.clip(np.dot(lp - ap, ab) / denom, 0.0, 1.0))
            proj = ap + t * ab
            return float(np.linalg.norm(lp - proj))

        return min(
            candidates,
            key=lambda landmark: (
                len(landmark.attributes.get("structure_room_pairs", [])),
                self._portal_label_score(landmark.label),
                _betweenness(landmark),
                float(np.linalg.norm(landmark.position - pos)),
                -float(landmark.confidence),
            ),
        )

    def _filter_summary_contains_near_room(self, room: SemanticNode) -> None:
        """Keep only object labels observed near the room anchor."""
        attrs = room.attributes
        room_pos = room.position
        near_labels: List[str] = []
        label_counts: Dict[str, int] = {}
        for obs in attrs.get("summary_observations", []):
            pos = np.array(obs.get("position", room_pos), dtype=np.float32)
            if float(np.linalg.norm(pos - room_pos)) > self.summary_radius:
                continue
            label = str(obs.get("label") or "").strip()
            if not label:
                continue
            key = label.lower()
            label_counts[key] = label_counts.get(key, 0) + 1
            if key not in {v.lower() for v in near_labels}:
                near_labels.append(label)
        if near_labels:
            attrs["contains_labels"] = near_labels[:8]
            attrs["label_counts"] = label_counts

    def _dedupe_room_summaries(self, rooms: Optional[List[SemanticNode]] = None) -> None:
        rooms = rooms if rooms is not None else self._room_region_summaries()
        for room in rooms:
            self._filter_summary_contains_near_room(room)
            attrs = room.attributes
            labels = attrs.get("contains_labels", [])
            unique_labels: List[str] = []
            seen = set()
            for label in labels:
                key = str(label or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                unique_labels.append(str(label).strip())
            attrs["contains_labels"] = unique_labels[:8]
            label_counts = attrs.get("label_counts", {})
            if not label_counts:
                label_counts = {}
                for label in labels:
                    key = str(label or "").strip().lower()
                    if key:
                        label_counts[key] = label_counts.get(key, 0) + 1
            if label_counts:
                attrs["label_counts"] = label_counts

    def _edge_exists(self, node_a: str, node_b: str, edge_type: EdgeType) -> bool:
        if node_a not in self.graph or node_b not in self.graph:
            return False
        if not self.graph.has_edge(node_a, node_b):
            return False
        data = self.graph.edges[node_a, node_b]
        return data.get("edge_type") == edge_type.value

    def _add_or_strengthen_edge(
        self,
        node_a: str,
        node_b: str,
        edge_type: EdgeType,
        weight: float,
    ) -> None:
        if node_a == node_b or node_a not in self._nodes or node_b not in self._nodes:
            return
        if self._edge_exists(node_a, node_b, edge_type):
            if self.graph.has_edge(node_a, node_b):
                prev = float(self.graph.edges[node_a, node_b].get("weight", weight))
                self.graph.edges[node_a, node_b]["weight"] = min(prev, float(weight))
            return
        self.add_edge(node_a, node_b, edge_type, weight=float(weight))

    def _nav_landmarks(self) -> List[SemanticNode]:
        """Landmarks eligible for the long-term spatial structure skeleton.

        Only **structural** landmarks (door / corridor / stair / window /
        synthetic portals) participate in room-portal-room skeleton building.
        Semantic landmarks (sofa / tv / vase) stay in the local semantic
        layer and never become portals or hubs.
        """
        result = []
        for node in self.get_nodes_by_type(NodeType.LANDMARK):
            attrs = node.attributes
            if attrs.get("landmark_source") == "environment" or attrs.get("source") == "environment":
                continue
            if attrs.get("synthetic_portal") or attrs.get("structure_role") == "portal":
                result.append(node)
                continue
            role = classify_landmark_role(node.label, attrs)
            attrs["landmark_role"] = role
            if role != "structural":
                continue
            result.append(node)
        return result

    def _nearest_nav_landmark(
        self,
        position: np.ndarray,
        landmarks: List[SemanticNode],
        max_dist: float,
    ) -> Optional[SemanticNode]:
        pos = np.array(position, dtype=np.float32)
        best = None
        best_dist = float(max_dist)
        for landmark in landmarks:
            dist = float(np.linalg.norm(landmark.position - pos))
            if dist < best_dist:
                best = landmark
                best_dist = dist
        return best

    def _clear_structure_anchors(self, landmarks: List[SemanticNode]) -> None:
        for landmark in landmarks:
            landmark.attributes.pop("structure_anchor", None)
            landmark.attributes.pop("structure_role", None)

    def _landmarks_for_room(
        self,
        room: SemanticNode,
        nav_landmarks: List[SemanticNode],
    ) -> List[SemanticNode]:
        members = [
            landmark for landmark in nav_landmarks
            if self._edge_exists(landmark.node_id, room.node_id, EdgeType.BELONGS_TO)
        ]
        if members:
            return members
        return [
            landmark for landmark in nav_landmarks
            if float(np.linalg.norm(landmark.position - room.position)) <= self.summary_radius
        ]

    def _prune_structure_adjacent_edges(self) -> None:
        """Drop stale room/landmark adjacent edges before rebuilding traversable links."""
        to_remove = []
        for node_a, node_b, data in self.graph.edges(data=True):
            if data.get("edge_type") != EdgeType.ADJACENT_TO.value:
                continue
            na = self._nodes.get(node_a)
            nb = self._nodes.get(node_b)
            if na is None or nb is None:
                continue
            if na.node_type in (NodeType.ROOM, NodeType.LANDMARK) and nb.node_type in (
                NodeType.ROOM, NodeType.LANDMARK,
            ):
                to_remove.append((node_a, node_b))
        for node_a, node_b in to_remove:
            if self.graph.has_edge(node_a, node_b):
                self.graph.remove_edge(node_a, node_b)

    def _traversable_room_transitions(
        self,
        rooms: List[SemanticNode],
    ) -> List[Tuple[SemanticNode, SemanticNode, SemanticNode, SemanticNode, float]]:
        """Room pairs actually crossed by navigable waypoint segments."""
        waypoints = {
            node.node_id: node
            for node in self.get_nodes_by_type(NodeType.WAYPOINT_VISITED)
        }
        transitions = []
        seen_pairs = set()
        for node_a, node_b, data in self.graph.edges(data=True):
            if data.get("edge_type") != EdgeType.NAVIGABLE.value:
                continue
            if node_a not in waypoints or node_b not in waypoints:
                continue
            wp_a = waypoints[node_a]
            wp_b = waypoints[node_b]
            room_a = self._room_for_transition(wp_a.position, rooms, waypoint=wp_a)
            room_b = self._room_for_transition(wp_b.position, rooms, waypoint=wp_b)
            if room_a is None or room_b is None or room_a.node_id == room_b.node_id:
                continue
            pair_key = tuple(sorted((room_a.node_id, room_b.node_id)))
            seg_dist = float(np.linalg.norm(wp_a.position - wp_b.position))
            if pair_key in seen_pairs:
                for idx, item in enumerate(transitions):
                    if tuple(sorted((item[0].node_id, item[1].node_id))) == pair_key:
                        if seg_dist < item[4]:
                            transitions[idx] = (room_a, room_b, wp_a, wp_b, seg_dist)
                        break
                continue
            seen_pairs.add(pair_key)
            transitions.append((room_a, room_b, wp_a, wp_b, seg_dist))
        return transitions

    def _prune_synthetic_portals(self) -> None:
        stale = [
            node.node_id for node in self.get_nodes_by_type(NodeType.LANDMARK)
            if node.attributes.get("synthetic_portal")
        ]
        for node_id in stale:
            self.remove_node(node_id)

    def _pair_portal_node_id(self, room_a_id: str, room_b_id: str) -> str:
        pair = tuple(sorted((room_a_id, room_b_id)))
        return f"portal::{pair[0]}::{pair[1]}"

    def _maintain_spatial_structure_graph(self) -> None:
        """Rebuild one traversable room→portal→room chain per crossed room pair."""
        rooms = self._sync_structure_rooms_from_waypoints()
        if len(rooms) < 1:
            return

        nav_landmarks = self._nav_landmarks()
        self._clear_structure_anchors(nav_landmarks)
        self._prune_synthetic_portals()
        self._prune_structure_adjacent_edges()

        portal_max = max(self.merge_radius * 5.0, self.summary_radius * 1.2)
        transitions = sorted(
            self._traversable_room_transitions(rooms),
            key=lambda item: tuple(sorted((item[0].node_id, item[1].node_id))),
        )
        for room_a, room_b, wp_a, wp_b, seg_dist in transitions:
            midpoint = (wp_a.position + wp_b.position) * 0.5
            ref_portal = self._pick_portal_landmark(
                midpoint, nav_landmarks, portal_max, room_a=room_a, room_b=room_b,
            )
            pair_key = tuple(sorted((room_a.node_id, room_b.node_id)))
            portal_id = self._pair_portal_node_id(room_a.node_id, room_b.node_id)
            if not self.has_node(portal_id):
                self.add_node(
                    NodeType.LANDMARK,
                    position=midpoint.copy(),
                    confidence=float(ref_portal.confidence) if ref_portal else 0.55,
                    label=str(ref_portal.label) if ref_portal else "passage",
                    node_id=portal_id,
                    attributes={
                        "structure_anchor": True,
                        "structure_role": "portal",
                        "synthetic_portal": True,
                        "landmark_role": "structural",
                        "structure_room_pairs": [list(pair_key)],
                        "portal_ref": ref_portal.node_id if ref_portal else None,
                        "structure_pair_labels": [
                            str(room_a.label or room_a.node_id),
                            str(room_b.label or room_b.node_id),
                        ],
                    },
                )
            portal = self._nodes[portal_id]
            portal.position = midpoint.copy()
            dist_a = float(np.linalg.norm(room_a.position - portal.position))
            dist_b = float(np.linalg.norm(portal.position - room_b.position))
            self._add_or_strengthen_edge(
                room_a.node_id, portal.node_id, EdgeType.ADJACENT_TO, dist_a,
            )
            self._add_or_strengthen_edge(
                portal.node_id, room_b.node_id, EdgeType.ADJACENT_TO, dist_b,
            )
            self._enrich_adjacent_edge(
                room_a.node_id, portal.node_id,
                str(room_a.label or ""), str(room_b.label or ""),
                portal_label=str(portal.label or ""), portal_node=portal,
            )
            self._enrich_adjacent_edge(
                portal.node_id, room_b.node_id,
                str(room_a.label or ""), str(room_b.label or ""),
                portal_label=str(portal.label or ""), portal_node=portal,
            )

        for landmark in nav_landmarks:
            if landmark.attributes.get("structure_role") == "portal":
                continue
            room = self._room_for_position(landmark.position, rooms)
            if room is None:
                continue
            if self._edge_exists(landmark.node_id, room.node_id, EdgeType.ADJACENT_TO):
                continue
            if not self._edge_exists(landmark.node_id, room.node_id, EdgeType.BELONGS_TO):
                self.add_edge(landmark.node_id, room.node_id, EdgeType.BELONGS_TO)

        for room in rooms:
            members = self._landmarks_for_room(room, nav_landmarks)
            if not members:
                continue
            hub = max(members, key=lambda landmark: float(landmark.confidence))
            if hub.attributes.get("structure_role") != "portal":
                hub.attributes["structure_role"] = "hub"
                hub.attributes["structure_anchor"] = False
            room.attributes["hub_landmark_id"] = hub.node_id
            if self._edge_exists(hub.node_id, room.node_id, EdgeType.ADJACENT_TO):
                continue
            if not self._edge_exists(hub.node_id, room.node_id, EdgeType.BELONGS_TO):
                self.add_edge(hub.node_id, room.node_id, EdgeType.BELONGS_TO)

        self._dedupe_room_summaries(rooms)

    def _update_node_visibility(self, agent_pos: np.ndarray):
        """Mark nodes as folded/unfolded based on distance and granularity."""
        for node in list(self._nodes.values()):
            if node.node_type not in (NodeType.OBJECT, NodeType.LANDMARK):
                continue
            dist = float(np.linalg.norm(node.position - agent_pos))
            attrs = node.attributes
            # Unfold conditions
            if dist <= self.near_radius:
                self._mark_node_active_detail(node)
                continue
            if attrs.get("recovered_from_summary"):
                self._mark_node_active_detail(node)
                continue
            if float(attrs.get("target_relevance", 0.0)) > 0 and dist <= self.far_radius:
                self._mark_node_active_detail(node)
                continue
            # Fold conditions
            granularity = attrs.get("granularity", "")
            if granularity == "room_level" and dist > self.near_radius:
                self._mark_node_folded_anchor(node, "room_level")
            elif attrs.get("history_compressed") and dist > self.fold_distance:
                summary_id = self._find_folded_summary_id(node)
                if summary_id is not None:
                    self._mark_node_folded_anchor(node, "summary_member", summary_id)
                else:
                    self._mark_node_folded_anchor(node, "far_compressed")
            else:
                self._mark_node_active_detail(node)

    def _mark_node_active_detail(self, node: SemanticNode) -> None:
        attrs = node.attributes
        attrs["folded"] = False
        attrs["folded_detail"] = False
        attrs["is_active_detail"] = True

    def _mark_node_folded_anchor(
        self,
        node: SemanticNode,
        reason: str,
        summary_id: Optional[str] = None,
    ) -> None:
        attrs = node.attributes
        attrs["folded"] = True
        attrs["folded_detail"] = True
        attrs["folded_reason"] = reason
        attrs["is_active_detail"] = False
        attrs["is_semantic_anchor"] = True

        attrs["anchor_position"] = node.position.tolist()
        nearest_visited_wp = self.find_nearest_node(
            node.position,
            NodeType.WAYPOINT_VISITED,
        )
        attrs["anchor_waypoint_id"] = nearest_visited_wp.node_id if nearest_visited_wp else None
        attrs["anchor_waypoint_position"] = (
            nearest_visited_wp.position.tolist() if nearest_visited_wp else None
        )

        anchor_room_id = summary_id or attrs.get("fused_into_summary_id") or self._find_folded_summary_id(node)
        attrs["anchor_room_id"] = anchor_room_id
        if anchor_room_id is not None:
            attrs["folded_summary_id"] = anchor_room_id

    def _find_folded_summary_id(self, node: SemanticNode) -> Optional[str]:
        """Find the room_region summary this node belongs to, if any."""
        for neighbor_id in self.get_neighbors(node.node_id):
            neighbor = self._nodes.get(neighbor_id)
            if neighbor is None:
                continue
            if (neighbor.node_type == NodeType.ROOM
                    and neighbor.attributes.get("summary_type") == "room_region"):
                return neighbor.node_id
        return None

    def _semantic_detail_score(self, node: SemanticNode, agent_pos: np.ndarray) -> float:
        dist = float(np.linalg.norm(node.position - agent_pos))
        if dist <= self.near_radius:
            distance_score = 1.0
        elif dist >= self.far_radius:
            distance_score = 0.0
        else:
            span = max(1e-6, self.far_radius - self.near_radius)
            distance_score = 1.0 - (dist - self.near_radius) / span

        last_seen = node.attributes.get("last_seen_step", node.step_id)
        staleness = max(0, self._current_step - int(last_seen))
        recency_score = float(self.confidence_decay ** staleness)
        multi_view_count = int(node.attributes.get("multi_view_count", max(1, node.visit_count)))
        multi_view_score = min(1.0, multi_view_count / 3.0)
        relevance_score = max(
            float(node.attributes.get("target_relevance", 0.0)),
            float(node.attributes.get("room_prior_score", 0.0)),
        )
        score = (
            0.35 * distance_score
            + 0.25 * float(node.confidence)
            + 0.20 * recency_score
            + 0.15 * multi_view_score
            + 0.05 * relevance_score
        )
        return float(max(0.0, min(1.0, score)))

    def _add_node_to_room_summary(self, node: SemanticNode, reason: str) -> Optional[str]:
        if not node.label:
            return None
        summary = self._find_or_create_room_summary(node)
        self._update_room_summary(summary, node, reason)
        self.add_edge(node.node_id, summary.node_id, EdgeType.BELONGS_TO)
        return summary.node_id

    def _find_or_create_room_summary(self, node: SemanticNode) -> SemanticNode:
        room_label = self._summary_room_label(node)
        label_key = normalize_region_label(room_label) or str(room_label or "").strip().lower()
        candidates = []
        for room in self.get_nodes_by_type(NodeType.ROOM):
            if room.attributes.get("summary_type") != "room_region":
                continue
            room_key = normalize_region_label(room.label) or str(room.label or "").strip().lower()
            if room_key == label_key:
                candidates.append(room)
        if candidates:
            return min(candidates, key=lambda room: float(np.linalg.norm(room.position - node.position)))

        node_id = self.add_node(
            NodeType.ROOM,
            position=node.position.copy(),
            embedding=node.embedding,
            confidence=max(0.3, min(1.0, node.confidence)),
            label=room_label,
            attributes={
                "summary_type": "room_region",
                "contains_labels": [],
                "contains_node_ids": [],
                "summary_observations": [],
                "source_granularities": [],
                "created_from_node_id": node.node_id,
            },
        )
        return self._nodes[node_id]

    def _summary_room_label(self, node: SemanticNode) -> str:
        room_context = node.attributes.get("room_context")
        if room_context is not None and str(room_context).strip() and str(room_context).strip() != "unknown":
            normalized = normalize_region_label(str(room_context).strip())
            return normalized if normalized else str(room_context).strip()
        contexts = node.attributes.get("room_contexts", [])
        for value in contexts:
            if value is not None and str(value).strip() and str(value).strip() != "unknown":
                normalized = normalize_region_label(str(value).strip())
                return normalized if normalized else str(value).strip()
        return "region"

    def _update_room_summary(self, summary: SemanticNode, node: SemanticNode, reason: str) -> None:
        attrs = summary.attributes
        attrs["summary_type"] = "room_region"
        labels = attrs.setdefault("contains_labels", [])
        if node.label and node.label not in labels:
            labels.append(node.label)
        source_ids = attrs.setdefault("contains_node_ids", [])
        new_member = node.node_id not in source_ids
        if new_member:
            source_ids.append(node.node_id)
        granularities = attrs.setdefault("source_granularities", [])
        granularity = node.attributes.get("granularity", node.node_type.value)
        if granularity not in granularities:
            granularities.append(granularity)

        if new_member:
            observations = attrs.setdefault("summary_observations", [])
            observations.append({
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "label": node.label,
                "confidence": float(node.confidence),
                "granularity": granularity,
                "reason": reason,
                "step_id": self._current_step,
                "position": node.position.tolist(),
            })
            if len(observations) > self.summary_max_observations:
                del observations[: len(observations) - self.summary_max_observations]

            member_count = len(source_ids)
            if member_count <= 1:
                summary.position = node.position.copy()
            else:
                summary.position = (
                    (summary.position * (member_count - 1)) + node.position
                ) / member_count

        attrs["last_summary_update_step"] = self._current_step
        self._absorb_object_into_room_summary(node, summary)
        if attrs.get("summary_dirty"):
            attrs["summary_text"] = self._format_room_summary_text(summary)
            attrs["summary_dirty"] = False
        summary.confidence = max(float(summary.confidence), min(1.0, float(node.confidence) + 0.05))
        summary.step_id = self._current_step

    def _normalize_semantic_label(self, label: str) -> str:
        label = (label or "").strip().lower()
        if label.endswith("s") and len(label) > 3:
            label = label[:-1]
        return label

    def _absorb_object_into_room_summary(
        self,
        obj_node: SemanticNode,
        room_node: SemanticNode,
    ) -> None:
        if obj_node.node_type != NodeType.OBJECT:
            return
        label = self._normalize_semantic_label(obj_node.label)
        if not label:
            return

        summary = room_node.attributes.setdefault("semantic_summary", {})
        folded_ids = summary.setdefault("folded_object_ids", [])
        is_new_folded_id = obj_node.node_id not in folded_ids
        if is_new_folded_id:
            folded_ids.append(obj_node.node_id)

        contains = summary.setdefault("contains_labels", {})
        if is_new_folded_id:
            contains[label] = contains.get(label, 0) + 1
        else:
            contains.setdefault(label, 1)

        label_conf = summary.setdefault("label_confidence", {})
        old_conf = float(label_conf.get(label, 0.0))
        label_conf[label] = max(old_conf, float(obj_node.confidence))

        representatives = summary.setdefault("representative_objects", {})
        old_rep = representatives.get(label)
        pos = obj_node.position.copy()
        if hasattr(pos, "tolist"):
            pos = pos.tolist()

        if old_rep is None or obj_node.confidence > float(old_rep.get("confidence", 0.0)):
            representatives[label] = {
                "node_id": obj_node.node_id,
                "position": pos,
                "confidence": float(obj_node.confidence),
            }

        room_node.attributes["summary_dirty"] = True

    def _format_room_summary_text(self, room_node: SemanticNode) -> str:
        summary = room_node.attributes.get("semantic_summary", {})
        contains = summary.get("contains_labels", {})

        if not contains:
            return f"{room_node.label} has no stable folded object summary."

        top_items = sorted(contains.items(), key=lambda x: -x[1])[:5]
        item_text = ", ".join([f"{cnt} {label}" for label, cnt in top_items])
        return f"{room_node.label} contains {item_text}."

    def find_nearby_room_summary(
        self,
        position: np.ndarray,
        label: Optional[str] = None,
        radius: Optional[float] = None,
    ) -> Optional[SemanticNode]:
        pos = np.array(position, dtype=np.float32)
        radius = self.summary_radius if radius is None else float(radius)
        best = None
        best_dist = float("inf")
        for room in self.get_nodes_by_type(NodeType.ROOM):
            if room.attributes.get("summary_type") != "room_region":
                continue
            if label is not None and room.attributes.get("contains_labels") and label not in room.attributes.get("contains_labels", []):
                continue
            dist = float(np.linalg.norm(room.position - pos))
            if dist <= radius and dist < best_dist:
                best = room
                best_dist = dist
        return best

    def assign_waypoint_to_room(
        self,
        waypoint_id: str,
        view_room_label: Optional[str] = None,
    ) -> Optional[str]:
        """Bind a visited waypoint to its room summary explicitly.

        Returns the room summary id if a binding was created/updated, else
        ``None``. The binding lives both as ``waypoint.attributes["room_id"]``
        (cheap reads) and as a ``BELONGS_TO`` edge (so debug / viz code that
        traverses edges sees a consistent picture).
        """
        wp = self._nodes.get(waypoint_id)
        if wp is None or wp.node_type != NodeType.WAYPOINT_VISITED:
            return None

        rooms = self._room_region_summaries()
        if not rooms:
            wp.attributes.pop("room_id", None)
            wp.attributes.pop("room_label", None)
            return None

        label_key = None
        if view_room_label is None:
            view_room_label = wp.attributes.get("view_room_label")
        if view_room_label and str(view_room_label).strip().lower() != "unknown":
            label_key = normalize_region_label(str(view_room_label)) or str(view_room_label).strip().lower()

        labelled: List[SemanticNode] = []
        if label_key:
            for room in rooms:
                room_key = normalize_region_label(room.label) or str(room.label or "").strip().lower()
                if room_key == label_key:
                    labelled.append(room)

        candidate_pool = labelled or rooms
        room = min(
            candidate_pool,
            key=lambda r: float(np.linalg.norm(r.position - wp.position)),
        )
        dist = float(np.linalg.norm(room.position - wp.position))
        # Be lenient: a waypoint should bind even slightly outside the
        # summary_radius if it's the nearest matching room.
        if dist > self.summary_radius * 2.0 and not labelled:
            wp.attributes.pop("room_id", None)
            wp.attributes.pop("room_label", None)
            return None

        prev_room_id = wp.attributes.get("room_id")
        wp.attributes["room_id"] = room.node_id
        wp.attributes["room_label"] = room.label
        wp.attributes["room_distance"] = dist

        if prev_room_id and prev_room_id != room.node_id and self.graph.has_edge(waypoint_id, prev_room_id):
            edge = self.graph.edges[waypoint_id, prev_room_id]
            if edge.get("edge_type") == EdgeType.BELONGS_TO.value:
                self.graph.remove_edge(waypoint_id, prev_room_id)
        self.add_edge(waypoint_id, room.node_id, EdgeType.BELONGS_TO)
        return room.node_id

    def mark_recovered_from_summary(
        self,
        object_id: str,
        position: np.ndarray,
        label: Optional[str] = None,
    ) -> Optional[str]:
        summary = self.find_nearby_room_summary(position, label=label)
        if summary is None:
            return None
        node = self._nodes.get(object_id)
        if node is None:
            return None
        node.attributes["recovered_from_summary"] = True
        node.attributes["summary_node_id"] = summary.node_id
        node.attributes["recovered_step"] = self._current_step
        self.add_edge(object_id, summary.node_id, EdgeType.BELONGS_TO)
        return summary.node_id

    def _compress_object_history(self, node: SemanticNode, reason: str) -> None:
        attrs = node.attributes
        observations = list(attrs.get("bbox_observations", []))
        if len(observations) <= self.object_history_keep_recent:
            return

        scores = [float(obs.get("confidence", 0.0)) for obs in observations]
        best_idx = int(np.argmax(scores)) if scores else 0
        keep_indices = {best_idx}
        keep_indices.update(range(max(0, len(observations) - self.object_history_keep_recent), len(observations)))
        kept = [observations[idx] for idx in sorted(keep_indices)]
        attrs["bbox_observations"] = kept
        attrs["detection_scores"] = [float(obs.get("confidence", 0.0)) for obs in kept]
        attrs["viewpoints"] = [
            obs.get("viewpoint_id")
            for obs in kept
            if obs.get("viewpoint_id") is not None
        ]
        attrs["history_compressed"] = True
        attrs["history_compression_reason"] = reason
        attrs["history_original_observation_count"] = max(
            int(attrs.get("history_original_observation_count", 0)),
            len(observations),
        )
        attrs["history_kept_observation_count"] = len(kept)
        attrs["history_best_confidence"] = max(scores) if scores else 0.0
        attrs["history_mean_confidence"] = float(np.mean(scores)) if scores else 0.0
        attrs["multi_view_count"] = max(
            int(attrs.get("multi_view_count", 1)),
            int(attrs["history_original_observation_count"]),
        )

    def _compress_landmark_history(self, node: SemanticNode, reason: str) -> None:
        attrs = node.attributes
        observations = list(attrs.get("observations", []))
        if len(observations) > self.landmark_history_keep_recent:
            scores = [float(obs.get("confidence", node.confidence)) for obs in observations]
            best_idx = int(np.argmax(scores)) if scores else 0
            keep_indices = {best_idx}
            keep_indices.update(range(max(0, len(observations) - self.landmark_history_keep_recent), len(observations)))
            kept = [observations[idx] for idx in sorted(keep_indices)]
            attrs["observations"] = kept
            attrs["history_best_confidence"] = max(scores) if scores else float(node.confidence)
            attrs["history_mean_confidence"] = float(np.mean(scores)) if scores else float(node.confidence)
            attrs["history_original_observation_count"] = max(
                int(attrs.get("history_original_observation_count", 0)),
                len(observations),
            )
            attrs["history_kept_observation_count"] = len(kept)
        else:
            attrs.setdefault("history_original_observation_count", len(observations))
            attrs.setdefault("history_kept_observation_count", len(observations))

        viewpoints = list(attrs.get("viewpoints", []))
        if len(viewpoints) > self.landmark_history_keep_recent:
            attrs["viewpoints"] = viewpoints[-self.landmark_history_keep_recent:]
            attrs["history_original_viewpoint_count"] = max(
                int(attrs.get("history_original_viewpoint_count", 0)),
                len(viewpoints),
            )
            attrs["history_kept_viewpoint_count"] = len(attrs["viewpoints"])

        attrs["history_compressed"] = True
        attrs["history_compression_reason"] = reason

    # ==================== Shortest Path ====================

    def shortest_path(self, source: str, target: str, edge_type: EdgeType = EdgeType.NAVIGABLE) -> Optional[List[str]]:
        """Find shortest path between two nodes using specified edge type."""
        sub_graph = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == edge_type.value:
                sub_graph.add_edge(u, v, weight=data.get("weight", 1.0))
        try:
            return nx.shortest_path(sub_graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def path_distance(self, source: str, target: str) -> float:
        """Euclidean distance between two nodes."""
        if source in self._nodes and target in self._nodes:
            return float(np.linalg.norm(
                self._nodes[source].position - self._nodes[target].position
            ))
        return float("inf")

    # ==================== Serialization ====================

    def to_dict(self) -> dict:
        """Serialize map state."""
        nodes_data = {}
        for nid, node in self._nodes.items():
            nodes_data[nid] = {
                "node_type": node.node_type.value,
                "position": node.position.tolist(),
                "confidence": node.confidence,
                "label": node.label,
                "step_id": node.step_id,
                "visit_count": node.visit_count,
                "attributes": node.attributes,
            }
        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            edges_data.append({"u": u, "v": v, **data})
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "current_step": self._current_step,
            "node_counter": self._node_counter,
        }

    @classmethod
    def from_dict(cls, data: dict, config=None) -> "DynamicTopoMap":
        """Deserialize map state."""
        topo = cls(config=config)
        topo._current_step = data.get("current_step", 0)
        topo._node_counter = data.get("node_counter", 0)
        for nid, ndata in data.get("nodes", {}).items():
            topo._nodes[nid] = SemanticNode(
                node_id=nid,
                node_type=NodeType(ndata["node_type"]),
                position=np.array(ndata["position"], dtype=np.float32),
                confidence=ndata["confidence"],
                label=ndata.get("label", ""),
                step_id=ndata.get("step_id", 0),
                visit_count=ndata.get("visit_count", 0),
                attributes=ndata.get("attributes", {}),
            )
            topo.graph.add_node(nid, node_type=ndata["node_type"])
        for edata in data.get("edges", []):
            u, v = edata.pop("u"), edata.pop("v")
            topo.graph.add_edge(u, v, **edata)
        return topo

    def reset(self):
        """Clear all nodes and edges (new episode)."""
        self._nodes.clear()
        self.graph.clear()
        self._node_counter = 0
        self._current_step = 0


# ==================== Edge Description (debug / LLM only) ====================

def format_edge_description(edge_attrs: dict, from_node_id=None) -> str:
    dist = edge_attrs.get('distance_m', 0.0)
    direction = 'forward'
    if from_node_id is not None:
        direction = edge_attrs.get('directions', {}).get(from_node_id, {}).get('direction_label', 'forward')
    return 'Move ' + direction + ' for ' + str(round(dist, 1)) + 'm.'


def format_portal_description(edge_attrs: dict, from_room_id=None) -> str:
    pt = edge_attrs.get('passage_type', 'passage')
    if from_room_id is not None:
        transitions = edge_attrs.get('transitions', {})
        tr = transitions.get(from_room_id)
        if tr:
            return 'Go through ' + pt + ' from ' + str(tr.get('from_room', 'one area')) + ' to ' + str(tr.get('to_room', 'another area')) + '.'
    rooms = edge_attrs.get('connected_rooms', ['one area', 'another area'])
    return 'Go through ' + pt + ' between ' + str(rooms[0]) + ' and ' + str(rooms[1]) + '.'
