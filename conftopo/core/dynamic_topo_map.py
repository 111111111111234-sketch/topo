"""DynamicTopoMap: confidence-aware semantic topological memory graph."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import numpy as np
import networkx as nx

from conftopo.core.confidence import ConfidenceFactors, compute_semantic_confidence


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
        else:
            # 默认配置
            self.confidence_decay = 0.95
            self.near_radius = 3.0
            self.far_radius = 10.0
            self.prune_threshold = 0.1
            self.max_nodes = 500
            self.merge_radius = 1.0

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
    #找与 node 直接相连的 node
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
        """Apply time decay to all node confidences."""
        for node in self._nodes.values():
            steps_since_update = self._current_step - node.step_id
            if steps_since_update > 0:
                decay = self.confidence_decay ** steps_since_update
                node.confidence *= decay

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
        match = self._best_object_match(matches, label, bbox, emb, view_heading, pos)

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
            if viewpoint_id is not None:
                self.add_edge(viewpoint_id, node_id, EdgeType.OBSERVED_AT)
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
        if viewpoint_id is not None:
            self.add_edge(viewpoint_id, match.node_id, EdgeType.OBSERVED_AT)
        return match.node_id, True

    def _best_object_match(
        self,
        candidates: List[SemanticNode],
        label: str,
        bbox: List[float],
        embedding: Optional[np.ndarray],
        view_heading: float,
        position: np.ndarray,
    ) -> Optional[SemanticNode]:
        best = None
        best_score = -1.0
        for node in candidates:
            if node.label != label:
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

    def prune_low_confidence(self):
        """Remove nodes with confidence below threshold (except visited waypoints)."""
        to_remove = []
        for node in self._nodes.values():
            if node.node_type == NodeType.WAYPOINT_VISITED:
                continue  # never prune visited waypoints
            if node.confidence < self.prune_threshold:
                to_remove.append(node.node_id)
        for nid in to_remove:
            self.remove_node(nid)

    def adaptive_granularity(self, agent_pos: np.ndarray):
        """Merge fine-grained nodes far from agent into coarser representations."""
        pos = np.array(agent_pos, dtype=np.float32)
        for node in list(self._nodes.values()):
            dist = np.linalg.norm(node.position - pos)
            if (
                dist > self.far_radius
                and node.node_type == NodeType.OBJECT
                and node.confidence < 0.5
            ):
                node.attributes["granularity"] = "room_level"
            elif node.node_type == NodeType.OBJECT and dist <= self.near_radius:
                node.attributes["granularity"] = "object"
            elif node.node_type == NodeType.OBJECT:
                node.attributes["granularity"] = "landmark"

    # ==================== Shortest Path ====================

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes (navigable edges only)."""
        nav_graph = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == EdgeType.NAVIGABLE.value:
                nav_graph.add_edge(u, v, weight=data.get("weight", 1.0))
        try:
            return nx.shortest_path(nav_graph, source, target, weight="weight")
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
