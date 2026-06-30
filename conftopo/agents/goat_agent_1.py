"""Single GOAT agent with an explicit perception-memory-planning loop.

This file is intentionally a clean first pass.  It keeps GOAT as one agent,
but separates the internal responsibilities so the step loop reads like the
method description:

observe -> CLIP -> optional VLM -> memory -> proposals -> planner -> state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType, SemanticNode
from conftopo.core.instruction_graph import GoalNode, GoalProposal, InstructionGraph
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import Qwen3VLBackend
from conftopo.perception.vlm_perceiver import VLMPerceiver


class GOATState(str, Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    APPROACH = "APPROACH"
    VERIFY_STOP = "VERIFY_STOP"
    STOP = "STOP"


@dataclass
class AgentPose:
    position: np.ndarray
    heading: float = 0.0


@dataclass
class ClipHint:
    room_label: str = "unknown"
    room_confidence: float = 0.0
    goal_scores: List[Any] = field(default_factory=list)
    landmark_scores: List[Any] = field(default_factory=list)
    best_goal_sim: float = 0.0
    best_landmark_sim: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def goal_candidate(self) -> bool:
        return self.best_goal_sim >= 0.35


@dataclass
class NavigationTarget:
    node_id: Optional[str]
    position: Optional[np.ndarray]
    target_type: str = "none"
    proposal: Optional[GoalProposal] = None


@dataclass
class AgentState:
    mode: GOATState = GOATState.SEARCH
    transition_reason: str = "init"
    verify_hits: int = 0
    verify_attempts: int = 0
    verify_cooldown_until_step: int = -1
    verify_window: List[bool] = field(default_factory=list)
    target_visible_window: List[bool] = field(default_factory=list)
    target_area_window: List[float] = field(default_factory=list)
    verify_area_window: List[float] = field(default_factory=list)
    target_area_trend: float = 0.0
    track_hits: int = 0
    target_lost_steps: int = 0
    state_steps: int = 0
    last_goal_seen_step: int = -1
    active_object_node_id: Optional[str] = None
    stuck_steps: int = 0
    stuck_recoveries: int = 0
    stuck_target_node_id: Optional[str] = None
    last_motion_delta: float = 0.0
    last_stuck_reason: str = ""


@dataclass
class Hypothesis:
    hypothesis_id: str
    label: str
    kind: str
    confidence: float
    position: np.ndarray
    anchor_waypoint_id: Optional[str]
    source: str
    first_seen_step: int
    last_seen_step: int
    seen_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decayed_confidence(self, current_step: int, decay: float) -> float:
        age = max(0, current_step - self.last_seen_step)
        return float(self.confidence * (decay ** age))


@dataclass
class ObjectMemoryEntry:
    object_id: str
    node_id: str
    label: str
    confidence: float
    attributes: Dict[str, Any]
    bbox_history: List[Any]
    anchor_waypoint_id: str
    anchor_waypoint_position: np.ndarray
    estimated_object_position: np.ndarray
    observed_from: List[np.ndarray]
    first_seen_step: int
    last_seen_step: int
    seen_count: int = 1
    failed_count: int = 0
    unreachable_count: int = 0

    @property
    def latest_bbox(self) -> Any:
        return self.bbox_history[-1] if self.bbox_history else None


class GoalManager:
    """Owns the current GOAT goal and GoalGraph bridge."""

    def __init__(self, agent: "GOATAgent"):
        self.agent = agent

    def current_goal(self) -> Optional[GoalNode]:
        if self.agent.instruction_graph is None:
            return None
        goal = self.agent.instruction_graph.get_current_goal()
        return goal if isinstance(goal, GoalNode) else None

    def set_goal(self, goal: GoalNode) -> None:
        if self.agent.instruction_graph is None:
            self.agent.instruction_graph = InstructionGraph(
                goal_type="object_goal",
                goal_nodes=[goal],
            )
        else:
            self.agent.instruction_graph.set_current_goal(goal)


class PerceptionModule:
    """Light CLIP every step, VLM only when useful."""

    def __init__(self, config: ConfTopoConfig):
        self.config = config
        self.clip = LightPerceiver(room_labels=config.perception.room_labels)
        self.vlm: Optional[VLMPerceiver] = None
        if config.perception.backend == "vlm":
            self.vlm = VLMPerceiver(
                Qwen3VLBackend(
                    api_base=config.perception.vlm_api_base,
                    model=config.perception.vlm_model,
                    timeout=config.perception.vlm_timeout,
                )
            )
        self.last_vlm_step: int = -1
        self.last_vlm_reason: str = ""

    def configure_goal(self, goal: GoalNode) -> None:
        self.clip.set_goal_labels(
            labels=[goal.target_object],
            embeddings=(
                goal.target_embedding[np.newaxis, :]
                if goal.target_embedding is not None
                else None
            ),
        )
        if goal.landmarks:
            self.clip.set_landmark_labels(
                labels=goal.landmarks,
                embeddings=goal.landmark_embeddings,
            )

    def run_clip(self, rgb_embed: Optional[np.ndarray], goal: Optional[GoalNode]) -> ClipHint:
        if rgb_embed is None:
            return ClipHint()
        report = self.clip.perceive(rgb_embed)
        return ClipHint(
            room_label=str(report.get("room_label", "unknown")),
            room_confidence=float(report.get("room_confidence", 0.0)),
            goal_scores=list(report.get("goal_scores", [])),
            landmark_scores=list(report.get("landmark_scores", [])),
            best_goal_sim=float(report.get("best_goal_sim", 0.0)),
            best_landmark_sim=float(report.get("best_landmark_sim", 0.0)),
            raw=report,
        )

    def should_trigger_vlm(
        self,
        *,
        clip_hint: ClipHint,
        goal: Optional[GoalNode],
        state: AgentState,
        step_id: int,
    ) -> bool:
        if self.vlm is None or goal is None:
            self.last_vlm_reason = "no_vlm"
            return False
        if state.mode in (GOATState.TRACK, GOATState.APPROACH, GOATState.VERIFY_STOP):
            self.last_vlm_reason = "state_refresh"
            return step_id - self.last_vlm_step >= 2
        if clip_hint.goal_candidate:
            self.last_vlm_reason = "clip_goal_candidate"
            return True

        if state.mode == GOATState.SEARCH:
            if self.last_vlm_step < 0:
                self.last_vlm_reason = "initial_search_scan"
                return True

            if step_id <= 30 and step_id - self.last_vlm_step >= 5:
                self.last_vlm_reason = "early_search_scan"
                return True

            if step_id - self.last_vlm_step >= 8:
                self.last_vlm_reason = "periodic_search"
                return True

        self.last_vlm_reason = "skip"
        return False

    def run_vlm(
        self,
        rgb: Optional[np.ndarray],
        goal: Optional[GoalNode],
        *,
        step_id: int,
        state: AgentState,
        clip_hint: ClipHint,
    ) -> Optional[Dict[str, Any]]:
        if self.vlm is None or rgb is None or goal is None:
            return None
        goal_text = _goal_text(goal)
        mode = "confirm" if state.mode != GOATState.SEARCH else "explore"
        context = (
            f"state={state.mode.value}; "
            f"clip_room={clip_hint.room_label}; "
            f"clip_goal_sim={clip_hint.best_goal_sim:.3f}"
        )
        try:
            report = self.vlm.perceive(
                np.asarray(rgb),
                goal_text,
                step_id=step_id,
                context=context,
                mode=mode,
            )
        except Exception as exc:
            return {
                "fresh": False,
                "step": step_id,
                "mode": mode,
                "trigger_reason": f"vlm_error:{type(exc).__name__}",
                "goal_visible": False,
                "stop_candidate": False,
                "objects": [],
                "error": str(exc),
            }

        self.last_vlm_step = step_id
        return _vlm_report_to_dict(report, reason=self.last_vlm_reason, mode=mode)


class HypothesisPool:
    """Short-term weak evidence from CLIP or non-confirmed VLM observations."""

    def __init__(
        self,
        *,
        max_size: int = 30,
        ttl_steps: int = 12,
        merge_radius: float = 2.0,
        decay: float = 0.85,
        min_confidence: float = 0.12,
    ):
        self.max_size = max_size
        self.ttl_steps = ttl_steps
        self.merge_radius = merge_radius
        self.decay = decay
        self.min_confidence = min_confidence
        self._items: Dict[str, Hypothesis] = {}
        self._counter = 0

    def clear(self) -> None:
        self._items.clear()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[Hypothesis]:
        return list(self._items.values())

    def add_or_update(
        self,
        *,
        label: str,
        kind: str,
        confidence: float,
        pose: AgentPose,
        anchor_waypoint_id: Optional[str],
        source: str,
        step_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Hypothesis]:
        label = str(label).strip()
        if not label or confidence < self.min_confidence:
            return None

        matched = self._find_match(label, kind, pose.position)
        if matched is not None:
            old_weight = max(1, matched.seen_count)
            new_conf = max(float(confidence), matched.decayed_confidence(step_id, self.decay))
            matched.position = (
                matched.position * old_weight + pose.position
            ) / float(old_weight + 1)
            repeat_bonus = min(0.2, 0.03 * matched.seen_count)
            matched.confidence = min(1.0, new_conf + repeat_bonus)
            matched.anchor_waypoint_id = anchor_waypoint_id or matched.anchor_waypoint_id
            matched.source = source
            matched.last_seen_step = step_id
            matched.seen_count += 1
            matched.metadata.update(metadata or {})
            return matched

        self._counter += 1
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{self._counter}",
            label=label,
            kind=kind,
            confidence=float(confidence),
            position=pose.position.copy(),
            anchor_waypoint_id=anchor_waypoint_id,
            source=source,
            first_seen_step=step_id,
            last_seen_step=step_id,
            metadata=dict(metadata or {}),
        )
        self._items[hypothesis.hypothesis_id] = hypothesis
        self._trim(step_id)
        return hypothesis

    def decay_and_prune(self, current_step: int) -> None:
        for hyp in self._items.values():
            if current_step > hyp.last_seen_step:
                hyp.confidence = float(hyp.confidence * self.decay)
        self._trim(current_step)

    def active_for_goal(
        self,
        goal: Optional[GoalNode],
        current_step: int,
    ) -> List[Hypothesis]:
        if goal is None:
            return []
        goal_label = str(goal.target_object or "").lower().strip()
        room_priors = {str(room).lower().strip() for room in goal.room_prior or []}
        landmarks = {str(label).lower().strip() for label in goal.landmarks or []}
        out = []
        for hyp in self._items.values():
            if current_step - hyp.last_seen_step > self.ttl_steps:
                continue
            if hyp.confidence < self.min_confidence:
                continue
            label = hyp.label.lower().strip()
            if hyp.kind == "object" and label == goal_label:
                out.append(hyp)
            elif hyp.kind == "room" and label in room_priors:
                out.append(hyp)
            elif hyp.kind == "landmark" and label in landmarks:
                out.append(hyp)
        out.sort(key=lambda hyp: (hyp.confidence, hyp.seen_count), reverse=True)
        return out

    def suppress_confirmed(self, *, label: str, position: np.ndarray, step_id: int) -> None:
        label_key = str(label).lower().strip()
        for hyp_id, hyp in list(self._items.items()):
            if hyp.kind != "object" or hyp.label.lower().strip() != label_key:
                continue
            if _distance(hyp.position, position) <= self.merge_radius * 1.5:
                self._items.pop(hyp_id, None)
        self._trim(step_id)

    def to_debug(self, current_step: int) -> List[Dict[str, Any]]:
        return [
            {
                "id": hyp.hypothesis_id,
                "label": hyp.label,
                "kind": hyp.kind,
                "confidence": round(float(hyp.confidence), 4),
                "age": int(current_step - hyp.last_seen_step),
                "seen_count": hyp.seen_count,
                "source": hyp.source,
                "anchor_waypoint_id": hyp.anchor_waypoint_id,
            }
            for hyp in sorted(
                self._items.values(),
                key=lambda item: item.confidence,
                reverse=True,
            )[:8]
        ]

    def _find_match(
        self,
        label: str,
        kind: str,
        position: np.ndarray,
    ) -> Optional[Hypothesis]:
        label_key = label.lower().strip()
        best: Optional[Hypothesis] = None
        best_dist = float("inf")
        for hyp in self._items.values():
            if hyp.label.lower().strip() != label_key or hyp.kind != kind:
                continue
            dist = _distance(hyp.position, position)
            if dist < self.merge_radius and dist < best_dist:
                best = hyp
                best_dist = dist
        return best

    def _trim(self, current_step: int) -> None:
        stale_ids = [
            hyp_id
            for hyp_id, hyp in self._items.items()
            if (
                current_step - hyp.last_seen_step > self.ttl_steps
                or hyp.confidence < self.min_confidence
            )
        ]
        for hyp_id in stale_ids:
            self._items.pop(hyp_id, None)

        if len(self._items) <= self.max_size:
            return
        keep = sorted(
            self._items.values(),
            key=lambda hyp: (hyp.confidence, hyp.last_seen_step, hyp.seen_count),
            reverse=True,
        )[:self.max_size]
        self._items = {hyp.hypothesis_id: hyp for hyp in keep}


class ObjectMemoryPool:
    """Confirmed object memory backed by OBJECT nodes in the TopoMap."""

    def __init__(
        self,
        topo_map: DynamicTopoMap,
        *,
        merge_radius: float = 2.0,
        max_bbox_history: int = 8,
        max_observed_from: int = 8,
    ):
        self.topo_map = topo_map
        self.merge_radius = merge_radius
        self.max_bbox_history = max_bbox_history
        self.max_observed_from = max_observed_from
        self._entries: Dict[str, ObjectMemoryEntry] = {}

    def clear_runtime(self) -> None:
        """Keep long-term object entries across GOAT goals."""
        return

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[ObjectMemoryEntry]:
        return list(self._entries.values())

    def get(self, object_id: str) -> Optional[ObjectMemoryEntry]:
        return self._entries.get(object_id)

    def goal_entries(self, goal: Optional[GoalNode]) -> List[ObjectMemoryEntry]:
        if goal is None:
            return []
        goal_label = str(goal.target_object or "").lower().strip()
        out = [
            entry
            for entry in self._entries.values()
            if entry.label.lower().strip() == goal_label
        ]
        out.sort(key=lambda entry: (entry.confidence, entry.seen_count), reverse=True)
        return out

    def upsert_confirmed(
        self,
        *,
        label: str,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
    ) -> ObjectMemoryEntry:
        estimated_position = _estimate_object_position(pose, obj)
        node = self._find_matching_node(label, estimated_position)
        if node is None:
            node_id = self._create_object_node(
                label,
                confidence,
                obj,
                pose,
                waypoint_id,
                estimated_position,
            )
        else:
            node_id = node.node_id
            self._update_object_node(
                node,
                confidence,
                obj,
                pose,
                waypoint_id,
                estimated_position,
            )

        node = self.topo_map.get_node(node_id)
        if node is None:
            raise RuntimeError(f"missing object node after upsert: {node_id}")

        entry = self._entries.get(node_id)
        bbox = obj.get("bbox")
        attrs = dict(obj)
        if entry is None:
            entry = ObjectMemoryEntry(
                object_id=node_id,
                node_id=node_id,
                label=label,
                confidence=float(node.confidence),
                attributes=attrs,
                bbox_history=[bbox] if bbox is not None else [],
                anchor_waypoint_id=waypoint_id,
                anchor_waypoint_position=pose.position.copy(),
                estimated_object_position=node.position.copy(),
                observed_from=[pose.position.copy()],
                first_seen_step=self.topo_map.current_step,
                last_seen_step=self.topo_map.current_step,
            )
            self._entries[node_id] = entry
        else:
            entry.confidence = float(node.confidence)
            entry.attributes.update(attrs)
            if bbox is not None:
                entry.bbox_history.append(bbox)
                entry.bbox_history = entry.bbox_history[-self.max_bbox_history:]
            entry.anchor_waypoint_id = waypoint_id
            entry.anchor_waypoint_position = pose.position.copy()
            entry.estimated_object_position = node.position.copy()
            entry.observed_from.append(pose.position.copy())
            entry.observed_from = entry.observed_from[-self.max_observed_from:]
            entry.last_seen_step = self.topo_map.current_step
            entry.seen_count += 1

        self._sync_node_from_entry(node, entry)
        return entry

    def mark_navigation_failed(
        self,
        object_id: str,
        *,
        unreachable: bool = False,
    ) -> None:
        entry = self._entries.get(object_id)
        if entry is None:
            return

        entry.failed_count += 1
        if unreachable:
            entry.unreachable_count += 1
        entry.attributes["failed_count"] = entry.failed_count
        entry.attributes["unreachable_count"] = entry.unreachable_count

        node = self.topo_map.get_node(entry.node_id)
        if node is not None:
            node.attributes["failed_count"] = entry.failed_count
            node.attributes["unreachable_count"] = entry.unreachable_count
            node.attributes["last_failed_step"] = self.topo_map.current_step
            node.attributes["memory_state"] = (
                "unreachable" if unreachable else "navigation_failed"
            )

    def to_debug(self) -> List[Dict[str, Any]]:
        return [
            {
                "object_id": entry.object_id,
                "label": entry.label,
                "confidence": round(float(entry.confidence), 4),
                "seen_count": entry.seen_count,
                "anchor_waypoint_id": entry.anchor_waypoint_id,
                "estimated_object_position": entry.estimated_object_position.tolist(),
                "last_seen_step": entry.last_seen_step,
                "latest_bbox": entry.latest_bbox,
                "failed_count": entry.failed_count,
                "unreachable_count": entry.unreachable_count,
            }
            for entry in sorted(
                self._entries.values(),
                key=lambda item: (item.confidence, item.seen_count),
                reverse=True,
            )[:8]
        ]

    def _find_matching_node(
        self,
        label: str,
        position: np.ndarray,
    ) -> Optional[SemanticNode]:
        nearby = self.topo_map.find_nodes_within_radius(
            position,
            radius=self.merge_radius,
            node_type=NodeType.OBJECT,
        )
        label_key = label.lower().strip()
        best = None
        best_dist = float("inf")
        for node in nearby:
            if node.attributes.get("semantic_role") != "object_anchor":
                continue
            if node.label.lower().strip() != label_key:
                continue
            dist = _distance(node.position, position)
            if dist < best_dist:
                best = node
                best_dist = dist
        return best

    def _create_object_node(
        self,
        label: str,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        estimated_position: np.ndarray,
    ) -> str:
        attrs = self._node_attrs(
            confidence,
            obj,
            pose,
            waypoint_id,
            estimated_position,
            seen_count=1,
        )
        node_id = self.topo_map.add_node(
            NodeType.OBJECT,
            position=estimated_position.copy(),
            confidence=confidence,
            label=label,
            attributes=attrs,
        )
        self.topo_map.add_edge(waypoint_id, node_id, EdgeType.ANCHORED_TO)
        return node_id

    def _update_object_node(
        self,
        node: SemanticNode,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        estimated_position: np.ndarray,
    ) -> None:
        seen_count = int(node.attributes.get("seen_count", 1)) + 1
        repeat_bonus = min(0.15, 0.02 * seen_count)
        old_weight = max(1, seen_count - 1)
        node.position = (
            node.position * old_weight + estimated_position
        ) / float(old_weight + 1)
        node.confidence = min(1.0, max(float(node.confidence), confidence) + repeat_bonus)
        node.step_id = self.topo_map.current_step
        node.attributes.update(
            self._node_attrs(
                confidence,
                obj,
                pose,
                waypoint_id,
                node.position,
                seen_count=seen_count,
            )
        )

    def _node_attrs(
        self,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        estimated_position: np.ndarray,
        *,
        seen_count: int,
    ) -> Dict[str, Any]:
        bbox = obj.get("bbox")
        return {
            "semantic_role": "object_anchor",
            "memory_state": "confirmed",
            "source": "vlm",
            "confidence": float(confidence),
            "bbox": bbox,
            "bbox_history": [bbox] if bbox is not None else [],
            "range_bin": obj.get("range_bin", "unknown"),
            "visibility": obj.get("visibility", "unknown"),
            "attributes": dict(obj),
            "anchor_waypoint_id": waypoint_id,
            "anchor_waypoint_position": pose.position.copy().tolist(),
            "estimated_object_position": estimated_position.copy().tolist(),
            "observed_from": pose.position.copy().tolist(),
            "observed_from_history": [pose.position.copy().tolist()],
            "last_seen_step": self.topo_map.current_step,
            "seen_count": seen_count,
            "failed_count": int(obj.get("failed_count", 0) or 0),
            "unreachable_count": int(obj.get("unreachable_count", 0) or 0),
        }

    def _sync_node_from_entry(
        self,
        node: SemanticNode,
        entry: ObjectMemoryEntry,
    ) -> None:
        node.attributes["bbox_history"] = list(entry.bbox_history)
        node.attributes["observed_from_history"] = [
            pos.tolist() for pos in entry.observed_from
        ]
        node.attributes["anchor_waypoint_id"] = entry.anchor_waypoint_id
        node.attributes["anchor_waypoint_position"] = (
            entry.anchor_waypoint_position.tolist()
        )
        node.attributes["estimated_object_position"] = (
            entry.estimated_object_position.tolist()
        )
        node.attributes["seen_count"] = entry.seen_count
        node.attributes["last_seen_step"] = entry.last_seen_step
        node.attributes["failed_count"] = entry.failed_count
        node.attributes["unreachable_count"] = entry.unreachable_count
        node.attributes["memory_state"] = "confirmed"
        node.position = entry.estimated_object_position.copy()
        node.confidence = max(float(node.confidence), float(entry.confidence))


class MemoryManager:
    """Owns short-term hypotheses and the shared DynamicTopoMap."""

    def __init__(self, topo_map: DynamicTopoMap, config: ConfTopoConfig):
        self.topo_map = topo_map
        self.config = config
        self.hypothesis_pool = HypothesisPool()
        self.object_memory_pool = ObjectMemoryPool(topo_map)
        self.structure_layer: Dict[str, Any] = {}
        self.navigation_layer: Dict[str, Any] = {}
        self.current_waypoint_id: Optional[str] = None

    def reset_runtime(self) -> None:
        self.current_waypoint_id = None
        self.hypothesis_pool.clear()
        self.object_memory_pool.clear_runtime()

    def mark_object_navigation_failed(
        self,
        object_id: str,
        *,
        unreachable: bool = False,
    ) -> None:
        self.object_memory_pool.mark_navigation_failed(
            object_id,
            unreachable=unreachable,
        )

    def mark_navigation_target_failed(
        self,
        node_id: Optional[str],
        *,
        target_type: str,
        unreachable: bool = False,
        reason: str = "navigation_failed",
    ) -> bool:
        if node_id is None:
            return False

        if target_type == "object":
            self.mark_object_navigation_failed(node_id, unreachable=unreachable)
            entry = self.object_memory_pool.get(node_id)
            self.navigation_layer["last_failed_target"] = {
                "node_id": node_id,
                "target_type": target_type,
                "failed_count": entry.failed_count if entry is not None else None,
                "unreachable_count": (
                    entry.unreachable_count if entry is not None else None
                ),
                "reason": reason,
                "step": self.topo_map.current_step,
            }
            return True

        node = self.topo_map.get_node(node_id)
        if node is None:
            return False

        failed_count = int(node.attributes.get("failed_count", 0) or 0) + 1
        unreachable_count = int(node.attributes.get("unreachable_count", 0) or 0)
        if unreachable:
            unreachable_count += 1

        node.attributes["failed_count"] = failed_count
        node.attributes["unreachable_count"] = unreachable_count
        node.attributes["last_failed_step"] = self.topo_map.current_step
        node.attributes["failure_reason"] = reason
        node.confidence = max(0.05, float(node.confidence) * 0.5)

        if target_type == "frontier":
            node.attributes["blocked"] = True
            node.attributes["consumed"] = True
        else:
            node.attributes["blocked"] = bool(unreachable)

        self.navigation_layer["last_failed_target"] = {
            "node_id": node_id,
            "target_type": target_type,
            "failed_count": failed_count,
            "unreachable_count": unreachable_count,
            "reason": reason,
            "step": self.topo_map.current_step,
        }
        return True

    def update_step_memory(
        self,
        *,
        clip_hint: ClipHint,
        vlm_report: Optional[Dict[str, Any]],
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        goal: Optional[GoalNode],
    ) -> None:
        self.topo_map.step()
        waypoint_id = self._add_or_update_waypoint(pose, rgb_embed)
        self._update_room(clip_hint, pose, rgb_embed, waypoint_id)
        self.update_hypothesis(clip_hint, pose, waypoint_id)
        if vlm_report is not None:
            self.update_weak_vlm_hypotheses(vlm_report, pose, goal, waypoint_id)
        if vlm_report is not None:
            self.update_confirmed_memory(vlm_report, pose, goal, waypoint_id)
        self._generate_frontiers(pose, rgb_embed, waypoint_id)
        self.hypothesis_pool.decay_and_prune(self.topo_map.current_step)
        self.topo_map.decay_all_confidences()

    def update_hypothesis(
        self,
        clip_hint: ClipHint,
        pose: AgentPose,
        waypoint_id: Optional[str],
    ) -> None:
        label = _best_label(clip_hint.goal_scores)
        if label is not None and clip_hint.best_goal_sim >= 0.25:
            self.hypothesis_pool.add_or_update(
                label=label,
                kind="object",
                confidence=clip_hint.best_goal_sim,
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="clip_goal",
                step_id=self.topo_map.current_step,
                metadata={"scores": list(clip_hint.goal_scores[:3])},
            )

        room_label = str(clip_hint.room_label or "").strip()
        if room_label and room_label != "unknown" and clip_hint.room_confidence >= 0.35:
            self.hypothesis_pool.add_or_update(
                label=room_label,
                kind="room",
                confidence=clip_hint.room_confidence,
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="clip_room",
                step_id=self.topo_map.current_step,
            )

        landmark = _best_label(clip_hint.landmark_scores)
        if landmark is not None and clip_hint.best_landmark_sim >= 0.25:
            self.hypothesis_pool.add_or_update(
                label=landmark,
                kind="landmark",
                confidence=clip_hint.best_landmark_sim,
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="clip_landmark",
                step_id=self.topo_map.current_step,
                metadata={"scores": list(clip_hint.landmark_scores[:3])},
            )

    def update_weak_vlm_hypotheses(
        self,
        vlm_report: Dict[str, Any],
        pose: AgentPose,
        goal: Optional[GoalNode],
        waypoint_id: str,
    ) -> None:
        goal_label = str(getattr(goal, "target_object", "") or "").lower().strip()
        confirmed_goal_visible = bool(vlm_report.get("goal_visible", False))
        for obj in vlm_report.get("objects", []) or []:
            label = str(obj.get("label", "")).strip()
            if not label:
                continue
            if confirmed_goal_visible and label.lower().strip() == goal_label:
                continue
            self.hypothesis_pool.add_or_update(
                label=label,
                kind="object",
                confidence=float(obj.get("confidence", 0.0) or 0.0),
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="weak_vlm",
                step_id=self.topo_map.current_step,
                metadata={
                    "bbox": obj.get("bbox"),
                    "range_bin": obj.get("range_bin", "unknown"),
                    "visibility": obj.get("visibility", "unknown"),
                },
            )

    def update_confirmed_memory(
        self,
        vlm_report: Dict[str, Any],
        pose: AgentPose,
        goal: Optional[GoalNode],
        waypoint_id: str,
    ) -> None:
        if not bool(vlm_report.get("goal_visible", False)):
            return
        objects = vlm_report.get("objects", []) or []
        goal_label = str(getattr(goal, "target_object", "") or "").lower().strip()
        candidates = [
            dict(obj)
            for obj in objects
            if str(obj.get("label", "")).lower().strip() == goal_label
        ]
        if not candidates:
            return

        best = max(candidates, key=lambda obj: float(obj.get("confidence", 0.0) or 0.0))
        label = str(best.get("label") or getattr(goal, "target_object", "object"))
        confidence = max(
            float(best.get("confidence", 0.0) or 0.0),
            float(vlm_report.get("goal_match_confidence", 0.0) or 0.0),
            0.55,
        )
        entry = self.object_memory_pool.upsert_confirmed(
            label=label,
            confidence=confidence,
            obj=best,
            pose=pose,
            waypoint_id=waypoint_id,
        )
        self.hypothesis_pool.suppress_confirmed(
            label=label,
            position=pose.position,
            step_id=self.topo_map.current_step,
        )
        self.structure_layer["last_confirmed_object_id"] = entry.object_id

    def _add_or_update_waypoint(
        self,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
    ) -> str:
        nearest = self.topo_map.find_nearest_node(
            pose.position,
            node_type=NodeType.WAYPOINT_VISITED,
        )
        if nearest is not None and np.linalg.norm(nearest.position - pose.position) < 0.6:
            nearest.visit_count += 1
            nearest.confidence = min(1.0, nearest.confidence + 0.05)
            nearest.step_id = self.topo_map.current_step
            node_id = nearest.node_id
        else:
            node_id = self.topo_map.add_node(
                NodeType.WAYPOINT_VISITED,
                position=pose.position.copy(),
                embedding=rgb_embed,
                confidence=0.9,
            )
        if self.current_waypoint_id is not None and self.current_waypoint_id != node_id:
            self.topo_map.add_edge(self.current_waypoint_id, node_id, EdgeType.NAVIGABLE)
        self.current_waypoint_id = node_id
        return node_id

    def _update_room(
        self,
        clip_hint: ClipHint,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        waypoint_id: str,
    ) -> None:
        if clip_hint.room_label == "unknown" or clip_hint.room_confidence < 0.2:
            return
        waypoint = self.topo_map.get_node(waypoint_id)
        if waypoint is not None:
            waypoint.attributes["room_label"] = clip_hint.room_label
            waypoint.attributes["room_confidence"] = clip_hint.room_confidence

        nearby = self.topo_map.find_nodes_within_radius(
            pose.position,
            radius=5.0,
            node_type=NodeType.ROOM,
        )
        room = next((node for node in nearby if node.label == clip_hint.room_label), None)
        if room is None:
            room_id = self.topo_map.add_node(
                NodeType.ROOM,
                position=pose.position.copy(),
                embedding=rgb_embed,
                confidence=clip_hint.room_confidence,
                label=clip_hint.room_label,
            )
            self.topo_map.add_edge(waypoint_id, room_id, EdgeType.BELONGS_TO)
        else:
            room.confidence = max(room.confidence, clip_hint.room_confidence)
            room.step_id = self.topo_map.current_step

    def _generate_frontiers(
        self,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        waypoint_id: str,
    ) -> None:
        for delta in (-0.7, 0.0, 0.7):
            heading = pose.heading + delta
            position = pose.position + np.array(
                [-np.sin(heading) * 2.5, 0.0, -np.cos(heading) * 2.5],
                dtype=np.float32,
            )
            nearest = self.topo_map.find_nearest_node(
                position,
                node_type=NodeType.WAYPOINT_FRONTIER,
            )
            if nearest is not None and np.linalg.norm(nearest.position - position) < 1.2:
                continue
            frontier_id = self.topo_map.add_node(
                NodeType.WAYPOINT_FRONTIER,
                position=position,
                embedding=rgb_embed,
                confidence=0.45,
                attributes={"source": "frontier", "created_from": waypoint_id},
            )
            self.topo_map.add_edge(waypoint_id, frontier_id, EdgeType.NAVIGABLE)


class ProposalBuilder:
    """GoalGraph + TopoMap -> executable proposals over existing nodes."""

    def __init__(self):
        self.last_debug: Dict[str, Any] = {}

    def build(
        self,
        *,
        goal: Optional[GoalNode],
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[GoalProposal]:
        if goal is None:
            self.last_debug = {
                "proposal_count": 0,
                "proposal_counts": {},
                "top_proposals": [],
            }
            return []

        goal_id = _goal_id(goal)
        object_proposals = self._object_proposals(goal_id, goal, memory, pose)
        has_confirmed_goal_object = bool(object_proposals)
        room_proposals = self._room_proposals(goal_id, goal, memory, pose)
        hypothesis_proposals = self._hypothesis_proposals(
            goal_id,
            goal,
            memory,
            pose,
            suppress_goal_hypotheses=has_confirmed_goal_object,
        )
        frontier_proposals = self._frontier_proposals(goal_id, goal, memory, pose)

        proposals = (
            object_proposals
            + room_proposals
            + hypothesis_proposals
            + frontier_proposals
        )
        proposals = self._deduplicate(proposals)
        proposals.sort(key=lambda proposal: proposal.score, reverse=True)

        self.last_debug = {
            "proposal_count": len(proposals),
            "proposal_counts": {
                "object": len(object_proposals),
                "room": len(room_proposals),
                "hypothesis": len(hypothesis_proposals),
                "frontier": len(frontier_proposals),
                "deduplicated": len(proposals),
            },
            "top_proposals": [self._proposal_debug(p) for p in proposals[:6]],
        }
        return proposals

    def _object_proposals(
        self,
        goal_id: str,
        goal: GoalNode,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[GoalProposal]:
        proposals: List[GoalProposal] = []
        goal_label = str(goal.target_object).lower().strip()
        for entry in memory.object_memory_pool.entries():
            if entry.label.lower().strip() != goal_label:
                continue
            node = memory.topo_map.get_node(entry.node_id)
            if node is None:
                continue
            target_position = np.asarray(
                entry.estimated_object_position,
                dtype=np.float32,
            )
            dist = _distance(pose.position, target_position)
            task_score = self._task_score_object(goal, entry)
            confidence = max(float(entry.confidence), float(node.confidence))
            history = min(1.0, entry.seen_count / 5.0)
            history_bonus = min(0.2, 0.03 * entry.seen_count)
            failed_count = max(
                int(entry.failed_count),
                int(node.attributes.get("failed_count", 0) or 0),
            )
            unreachable_count = max(
                int(entry.unreachable_count),
                int(node.attributes.get("unreachable_count", 0) or 0),
            )
            risk_penalty = min(
                1.0,
                0.25 * failed_count + 0.75 * unreachable_count,
            )
            score = (
                0.45 * task_score
                + 0.35 * confidence
                + 0.15 * history
                - 0.05 * dist
                - risk_penalty
            )
            risk_reason = (
                f"; failed={failed_count}; unreachable={unreachable_count}"
                if failed_count or unreachable_count
                else ""
            )
            reason = (
                f"confirmed_object:{entry.label}; "
                f"task={task_score:.2f}; conf={confidence:.2f}; "
                f"seen={entry.seen_count}; risk={risk_penalty:.2f}{risk_reason}"
            )
            proposals.append(
                self._with_reason(GoalProposal(
                    goal_id=goal_id,
                    candidate_node_id=node.node_id,
                    candidate_type="object",
                    anchor_node_id=entry.anchor_waypoint_id,
                    target_position=target_position,
                    score=score,
                    semantic_score=task_score,
                    task_score=task_score,
                    history_bonus=history_bonus,
                    distance_cost=dist,
                    reachability_score=max(0.0, 1.0 - risk_penalty),
                    risk_penalty=risk_penalty,
                    negative_evidence=float(failed_count + unreachable_count),
                    source="object_memory",
                    can_stop=True,
                    requires_verification=True,
                    evidence_refs=[entry.object_id],
                ), reason)
            )
        return proposals

    def _room_proposals(
        self,
        goal_id: str,
        goal: GoalNode,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[GoalProposal]:
        proposals: List[GoalProposal] = []
        room_priors = {str(room).lower().strip() for room in goal.room_prior or []}
        for node in memory.topo_map.get_nodes_by_type(NodeType.ROOM):
            room_label = str(node.label).lower().strip()
            if room_priors and room_label not in room_priors:
                continue
            dist = _distance(pose.position, node.position)
            task_score = 1.0 if room_label in room_priors else 0.3
            confidence = float(node.confidence)
            score = 0.50 * task_score + 0.30 * confidence - 0.04 * dist
            reason = (
                f"room_prior:{room_label}; task={task_score:.2f}; "
                f"conf={confidence:.2f}"
            )
            proposals.append(
                self._with_reason(GoalProposal(
                    goal_id=goal_id,
                    candidate_node_id=node.node_id,
                    candidate_type="room",
                    target_position=node.position.copy(),
                    score=score,
                    room_score=task_score,
                    task_score=task_score,
                    distance_cost=dist,
                    source="structure_layer",
                    can_stop=False,
                    requires_verification=False,
                    evidence_refs=[node.node_id],
                ), reason)
            )
        return proposals

    def _hypothesis_proposals(
        self,
        goal_id: str,
        goal: GoalNode,
        memory: MemoryManager,
        pose: AgentPose,
        *,
        suppress_goal_hypotheses: bool,
    ) -> List[GoalProposal]:
        if suppress_goal_hypotheses:
            return []

        proposals: List[GoalProposal] = []
        for hyp in memory.hypothesis_pool.active_for_goal(
            goal,
            memory.topo_map.current_step,
        ):
            if hyp.anchor_waypoint_id is None:
                continue
            anchor = memory.topo_map.get_node(hyp.anchor_waypoint_id)
            if anchor is None:
                continue
            dist = _distance(pose.position, anchor.position)
            task_score = self._task_score_hypothesis(goal, hyp)
            history = min(1.0, hyp.seen_count / 3.0)
            history_bonus = min(0.15, 0.04 * hyp.seen_count)
            score = (
                0.45 * task_score
                + 0.30 * hyp.confidence
                + 0.10 * history
                - 0.04 * dist
            )
            reason = (
                f"weak_{hyp.kind}:{hyp.label}; task={task_score:.2f}; "
                f"conf={hyp.confidence:.2f}; seen={hyp.seen_count}"
            )
            proposals.append(
                self._with_reason(GoalProposal(
                    goal_id=goal_id,
                    candidate_node_id=anchor.node_id,
                    candidate_type="hypothesis",
                    anchor_node_id=anchor.node_id,
                    target_position=anchor.position.copy(),
                    score=score,
                    semantic_score=task_score,
                    task_score=task_score,
                    history_bonus=history_bonus,
                    distance_cost=dist,
                    source="hypothesis_pool",
                    can_stop=False,
                    requires_verification=True,
                    evidence_refs=[hyp.hypothesis_id],
                ), reason)
            )
        return proposals

    def _frontier_proposals(
        self,
        goal_id: str,
        goal: GoalNode,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[GoalProposal]:
        proposals: List[GoalProposal] = []
        for node in memory.topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER):
            if node.attributes.get("consumed", False):
                continue
            if node.attributes.get("blocked", False):
                continue
            dist = _distance(pose.position, node.position)
            frontier_value = float(node.confidence)
            semantic_value, semantic_reason = self._frontier_task_value(goal, memory, node)
            blocked_penalty = 0.30 if node.attributes.get("blocked", False) else 0.0
            failed_count = float(node.attributes.get("failed_count", 0) or 0)
            unreachable_count = float(node.attributes.get("unreachable_count", 0) or 0)
            failed_penalty = 0.20 * failed_count
            risk_penalty = min(1.0, blocked_penalty + failed_penalty + 0.35 * unreachable_count)
            score = (
                0.35 * frontier_value
                + 0.30 * semantic_value
                - 0.03 * dist
                - risk_penalty
            )
            reason = (
                f"frontier; value={frontier_value:.2f}; "
                f"semantic={semantic_value:.2f}; risk={risk_penalty:.2f}; "
                f"{semantic_reason}"
            )
            proposals.append(
                self._with_reason(GoalProposal(
                    goal_id=goal_id,
                    candidate_node_id=node.node_id,
                    candidate_type="frontier",
                    target_position=node.position.copy(),
                    score=score,
                    frontier_value=frontier_value,
                    distance_cost=dist,
                    reachability_score=max(0.0, 1.0 - risk_penalty),
                    risk_penalty=risk_penalty,
                    negative_evidence=failed_count + unreachable_count,
                    source="navigation_layer",
                    can_stop=False,
                    requires_verification=False,
                    evidence_refs=[node.node_id],
                ), reason)
            )
        return proposals

    def _deduplicate(self, proposals: List[GoalProposal]) -> List[GoalProposal]:
        best: Dict[tuple, GoalProposal] = {}
        for proposal in proposals:
            key = (
                proposal.goal_id,
                proposal.candidate_type,
                proposal.candidate_node_id,
            )
            if key not in best or proposal.score > best[key].score:
                best[key] = proposal
        return list(best.values())

    def _task_score_object(
        self,
        goal: GoalNode,
        entry: ObjectMemoryEntry,
    ) -> float:
        score = 0.0
        if entry.label.lower().strip() == str(goal.target_object).lower().strip():
            score += 0.8

        attrs = entry.attributes or {}
        attr_text = str(attrs).lower()
        for attr in [str(a).lower() for a in goal.attributes or []]:
            if attr in attr_text:
                score += 0.1

        return min(1.0, score)

    def _frontier_task_value(
        self,
        goal: GoalNode,
        memory: MemoryManager,
        node: SemanticNode,
    ) -> tuple:
        value = float(node.attributes.get("semantic_value", 0.0) or 0.0)
        reasons: List[str] = []

        room_priors = {str(room).lower().strip() for room in goal.room_prior or []}
        landmark_priors = {str(label).lower().strip() for label in goal.landmarks or []}

        room_label = str(node.attributes.get("room_label", "")).lower().strip()
        created_from = node.attributes.get("created_from")
        if not room_label and created_from:
            parent = memory.topo_map.get_node(str(created_from))
            if parent is not None:
                room_label = str(parent.attributes.get("room_label", "")).lower().strip()

        if room_label and room_label in room_priors:
            value += 0.45
            reasons.append(f"room_prior={room_label}")

        if landmark_priors:
            nearby_landmarks = memory.topo_map.find_nodes_within_radius(
                node.position,
                radius=5.0,
                node_type=NodeType.LANDMARK,
            )
            matched = [
                landmark.label
                for landmark in nearby_landmarks
                if str(landmark.label).lower().strip() in landmark_priors
            ]
            if matched:
                value += 0.30
                reasons.append(f"near_landmark={matched[0]}")

        if not node.attributes.get("visited", False):
            value += 0.10
            reasons.append("unexplored")

        if node.attributes.get("blocked", False):
            reasons.append("blocked")
        failed_count = int(node.attributes.get("failed_count", 0) or 0)
        if failed_count:
            reasons.append(f"failed={failed_count}")

        return min(1.0, value), ",".join(reasons) or "generic"

    def _task_score_hypothesis(self, goal: GoalNode, hyp: Hypothesis) -> float:
        label = hyp.label.lower().strip()
        target = str(goal.target_object).lower().strip()

        if hyp.kind == "object" and label == target:
            return 0.8

        if hyp.kind == "room":
            room_priors = {str(room).lower().strip() for room in goal.room_prior or []}
            return 0.7 if label in room_priors else 0.2

        if hyp.kind == "landmark":
            landmarks = {str(landmark).lower().strip() for landmark in goal.landmarks or []}
            return 0.6 if label in landmarks else 0.2

        return 0.1

    @staticmethod
    def _with_reason(proposal: GoalProposal, reason: str) -> GoalProposal:
        setattr(proposal, "reason", reason)
        return proposal

    @staticmethod
    def _proposal_debug(proposal: GoalProposal) -> Dict[str, Any]:
        return {
            "node_id": proposal.candidate_node_id,
            "type": proposal.candidate_type,
            "score": round(float(proposal.score), 4),
            "source": proposal.source,
            "can_stop": proposal.can_stop,
            "requires_verification": proposal.requires_verification,
            "distance_cost": round(float(proposal.distance_cost), 4),
            "reachability_score": round(float(proposal.reachability_score), 4),
            "risk_penalty": round(float(proposal.risk_penalty), 4),
            "negative_evidence": round(float(proposal.negative_evidence), 4),
            "evidence_refs": list(proposal.evidence_refs),
            "reason": str(getattr(proposal, "reason", "")),
        }


class Planner:
    """Select the next navigation target from proposals."""

    def select_target(
        self,
        *,
        proposals: List[GoalProposal],
        memory: MemoryManager,
        pose: AgentPose,
        state: AgentState,
    ) -> NavigationTarget:
        if not proposals:
            return NavigationTarget(node_id=None, position=None, target_type="idle")

        def total_score(proposal: GoalProposal) -> float:
            type_bonus = {
                "object": 0.30,
                "room": 0.15,
                "hypothesis": 0.20,
                "frontier": 0.00,
            }.get(proposal.candidate_type, 0.0)
            return (
                proposal.score
                + type_bonus
                - 0.05 * proposal.risk_penalty
                - 0.10 * proposal.negative_evidence
            )

        object_proposals = [p for p in proposals if p.candidate_type == "object"]
        if object_proposals:
            best_object = max(object_proposals, key=total_score)
            fallback_proposals = [
                p for p in proposals if p.candidate_type != "object"
            ]
            if not fallback_proposals:
                best = best_object
            else:
                best_fallback = max(fallback_proposals, key=total_score)
                object_score = total_score(best_object)
                fallback_score = total_score(best_fallback)
                object_is_usable = (
                    best_object.reachability_score >= 0.35
                    and best_object.risk_penalty < 0.50
                    and object_score >= fallback_score - 0.10
                )
                best = best_object if object_is_usable else best_fallback
        else:
            hypothesis_proposals = [
                p for p in proposals if p.candidate_type == "hypothesis"
            ]
            best = (
                max(hypothesis_proposals, key=total_score)
                if hypothesis_proposals
                else max(proposals, key=total_score)
            )
        return NavigationTarget(
            node_id=best.candidate_node_id,
            position=best.target_position,
            target_type=best.candidate_type,
            proposal=best,
        )


class Controller:
    """Converts selected targets and state decisions to environment actions."""

    def search_action(self, nav_target: NavigationTarget) -> Dict[str, Any]:
        if nav_target.position is None:
            return {"action": "turn_left", "reason": "search_no_target"}
        return {
            "action": "navigate",
            "target_position": nav_target.position,
            "target_node_id": nav_target.node_id,
            "target_type": nav_target.target_type,
        }

    def approach_action(
        self,
        vlm_report: Optional[Dict[str, Any]],
        nav_target: NavigationTarget,
        goal: Optional[GoalNode] = None,
    ) -> Dict[str, Any]:
        bbox = _target_bbox(vlm_report, goal)
        if bbox is not None:
            center = 0.5 * (float(bbox[0]) + float(bbox[2]))
            if center < 0.35:
                return {"action": "turn_left", "reason": "approach_center_target"}
            if center > 0.65:
                return {"action": "turn_right", "reason": "approach_center_target"}
            return {"action": "move_forward", "reason": "approach_visible_target"}
        return self.search_action(nav_target)

    def verify_action(
        self,
        vlm_report: Optional[Dict[str, Any]] = None,
        goal: Optional[GoalNode] = None,
    ) -> Dict[str, Any]:
        fresh = bool(vlm_report and vlm_report.get("fresh", False))
        bbox = _target_bbox(vlm_report, goal)
        if bbox is None:
            return {"action": "turn_left", "reason": "verify_scan_no_target"}
        if not fresh:
            return {"action": "move_forward", "reason": "verify_wait_for_fresh"}
        center = _bbox_center_x(bbox)
        if center is None:
            return {"action": "turn_left", "reason": "verify_scan_no_bbox"}
        if center < 0.35:
            return {"action": "turn_left", "reason": "verify_center_target"}
        if center > 0.65:
            return {"action": "turn_right", "reason": "verify_center_target"}
        return {"action": "move_forward", "reason": "verify_hold_centered"}

    def stop_action(self) -> Dict[str, Any]:
        return {"action": "stop", "reason": "verified_goal"}


class StopVerifier:
    """Multi-frame stop gate.  SEARCH/TRACK cannot stop directly."""

    def observe(
        self,
        state: AgentState,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        if not vlm_report or not bool(vlm_report.get("fresh", False)):
            return False
        state.verify_attempts += 1
        best = _target_object(vlm_report, goal)
        if best is not None:
            state.verify_area_window.append(_bbox_area(best.get("bbox")))
            state.verify_area_window = state.verify_area_window[-4:]
        hit = self._single_frame_stop_hit(vlm_report, goal, state)
        state.verify_window.append(hit)
        state.verify_window = state.verify_window[-4:]
        state.verify_hits = sum(1 for item in state.verify_window if item)
        return state.verify_hits >= 2

    def should_enter_verify(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        best = _target_object(vlm_report, goal)
        if best is None:
            return False
        bbox_area = _bbox_area(best.get("bbox"))
        range_bin = str(best.get("range_bin", "")).lower()
        return _range_area_close_enough_for_verify(range_bin, bbox_area)

    def _single_frame_stop_hit(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
        state: AgentState,
    ) -> bool:
        if not vlm_report or not bool(vlm_report.get("goal_visible", False)):
            return False
        best = _target_object(vlm_report, goal)
        if best is None:
            return False
        bbox = best.get("bbox")
        bbox_area = _bbox_area(bbox)
        range_bin = str(best.get("range_bin", "")).lower()
        bbox_confidence = str(best.get("bbox_confidence", "")).lower()
        visibility = str(best.get("visibility", "")).lower()
        center = _bbox_center_x(bbox)
        centered = center is None or 0.15 <= center <= 0.90
        area_growing = _area_trend(state.verify_area_window) >= 0.015
        area_holding_close = (
            len(state.verify_area_window) >= 2
            and min(state.verify_area_window[-2:]) >= 0.08
            and state.verify_area_window[-1] >= 0.9 * state.verify_area_window[-2]
        )
        close_enough = _range_area_close_enough_for_stop(range_bin, bbox_area)
        strong_visual_stop = (
            close_enough
            and bbox_confidence != "low"
            and visibility not in {"occluded", "not_visible"}
            and centered
            and (area_growing or area_holding_close)
        )
        vlm_stop_candidate = (
            bool(vlm_report.get("stop_candidate", False))
            and close_enough
            and (area_growing or area_holding_close)
        )
        return (
            vlm_stop_candidate
            or strong_visual_stop
        )


class GOATAgent(ConfTopoBaseAgent):
    """Monolithic GOAT agent with modular internals."""

    def __init__(self, config: Optional[ConfTopoConfig] = None):
        super().__init__(config)
        self.goal_manager = GoalManager(self)
        self.perception = PerceptionModule(self.config)
        self.memory = MemoryManager(self.topo_map, self.config)
        self.proposal_builder = ProposalBuilder()
        self.planner = Planner()
        self.controller = Controller()
        self.stop_verifier = StopVerifier()
        self.state = AgentState()

        self.step_id: int = 0
        self._origin_position: Optional[np.ndarray] = None
        self._pose: Optional[AgentPose] = None
        self._rgb: Optional[np.ndarray] = None
        self._rgb_embed: Optional[np.ndarray] = None
        self._last_clip_hint: ClipHint = ClipHint()
        self._fresh_vlm_report: Optional[Dict[str, Any]] = None
        self._cached_vlm_report: Optional[Dict[str, Any]] = None
        self._cached_vlm_step: int = -1
        self._vlm_for_control: Optional[Dict[str, Any]] = None
        self._last_proposals: List[GoalProposal] = []
        self._last_nav_target: NavigationTarget = NavigationTarget(None, None)
        self._last_pose_for_stuck: Optional[np.ndarray] = None
        self._stuck_monitor_target_id: Optional[str] = None
        self._stuck_marked_target_id: Optional[str] = None

    def reset(self) -> None:
        super().reset()
        self.memory = MemoryManager(self.topo_map, self.config)
        self.state = AgentState(transition_reason="reset")
        self.step_id = 0
        self._origin_position = None
        self._pose = None
        self._rgb = None
        self._rgb_embed = None
        self._last_clip_hint = ClipHint()
        self._fresh_vlm_report = None
        self._cached_vlm_report = None
        self._cached_vlm_step = -1
        self._vlm_for_control = None
        self._last_proposals = []
        self._last_nav_target = NavigationTarget(None, None)
        self._last_pose_for_stuck = None
        self._stuck_monitor_target_id = None
        self._stuck_marked_target_id = None

    def reset_keep_memory(self) -> None:
        super().reset_keep_memory()
        self.state = AgentState(transition_reason="new_goal")
        self.memory.reset_runtime()
        self._fresh_vlm_report = None
        self._cached_vlm_report = None
        self._cached_vlm_step = -1
        self._vlm_for_control = None
        self._last_proposals = []
        self._last_nav_target = NavigationTarget(None, None)
        self._last_pose_for_stuck = None
        self._stuck_monitor_target_id = None
        self._stuck_marked_target_id = None

    def set_new_goal(self, goal: GoalNode) -> None:
        self.reset_keep_memory()
        self.goal_manager.set_goal(goal)
        self.perception.configure_goal(goal)

    def observe(self, obs: Dict[str, Any]) -> None:
        raw_position = np.asarray(obs["position"], dtype=np.float32)
        if self._origin_position is None:
            self._origin_position = raw_position.copy()
        position = raw_position - self._origin_position
        position[1] = 0.0
        self._pose = AgentPose(
            position=position,
            heading=float(obs.get("heading", 0.0)),
        )
        self._rgb = obs.get("rgb")
        if "rgb_embed" in obs:
            self._rgb_embed = np.asarray(obs["rgb_embed"], dtype=np.float32)
        elif "pano_rgb_embed" in obs:
            self._rgb_embed = np.asarray(obs["pano_rgb_embed"], dtype=np.float32)
        else:
            self._rgb_embed = None

    def update_memory(self) -> None:
        if self._pose is None:
            return
        self.memory.update_step_memory(
            clip_hint=self._last_clip_hint,
            vlm_report=self._fresh_vlm_report,
            pose=self._pose,
            rgb_embed=self._rgb_embed,
            goal=self.goal_manager.current_goal(),
        )

    def plan(self) -> NavigationTarget:
        if self._pose is None:
            return NavigationTarget(None, None, "idle")
        self._last_proposals = self.proposal_builder.build(
            goal=self.goal_manager.current_goal(),
            memory=self.memory,
            pose=self._pose,
        )
        self._last_nav_target = self.planner.select_target(
            proposals=self._last_proposals,
            memory=self.memory,
            pose=self._pose,
            state=self.state,
        )
        return self._last_nav_target

    def act(self, plan_output: NavigationTarget) -> Dict[str, Any]:
        return self.act_with_state_machine(
            nav_target=plan_output,
            vlm_report=self._vlm_for_control,
        )

    def report_navigation_failed(
        self,
        target_node_id: Optional[str] = None,
        *,
        unreachable: bool = False,
    ) -> bool:
        """Let the external executor report that the selected object target failed."""
        nav_target = self._last_nav_target
        proposal = nav_target.proposal
        if proposal is None or proposal.candidate_type != "object":
            return False
        if target_node_id is not None and target_node_id != proposal.candidate_node_id:
            return False

        object_ids = [str(ref) for ref in proposal.evidence_refs if ref]
        if not object_ids:
            object_ids = [proposal.candidate_node_id]
        for object_id in object_ids:
            self.memory.mark_object_navigation_failed(
                object_id,
                unreachable=unreachable,
            )
        return True

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.step_id += 1
        self._step_count += 1

        self.observe(obs)
        goal = self.goal_manager.current_goal()

        self._last_clip_hint = self.perception.run_clip(self._rgb_embed, goal)
        need_vlm = self.perception.should_trigger_vlm(
            clip_hint=self._last_clip_hint,
            goal=goal,
            state=self.state,
            step_id=self.step_id,
        )
        self._fresh_vlm_report = None
        if need_vlm:
            self._fresh_vlm_report = self.perception.run_vlm(
                self._rgb,
                goal,
                step_id=self.step_id,
                state=self.state,
                clip_hint=self._last_clip_hint,
            )
            if self._fresh_vlm_report and self._fresh_vlm_report.get("fresh", False):
                self._cached_vlm_report = self._fresh_vlm_report
                self._cached_vlm_step = self.step_id

        self._vlm_for_control = self._fresh_vlm_report
        if (
            self._vlm_for_control is None
            and self._cached_vlm_report is not None
            and self.step_id - self._cached_vlm_step <= 3
        ):
            self._vlm_for_control = dict(self._cached_vlm_report)
            self._vlm_for_control["fresh"] = False
            self._vlm_for_control["cached_from_step"] = self._cached_vlm_step

        self.update_memory()
        nav_target = self.plan()
        action = self.act_with_state_machine(
            nav_target=nav_target,
            vlm_report=self._vlm_for_control,
        )
        self._update_stuck_monitor(nav_target, action)
        action["debug"] = self._debug_snapshot(need_vlm)
        return action

    def act_with_state_machine(
        self,
        *,
        nav_target: NavigationTarget,
        vlm_report: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        goal = self.goal_manager.current_goal()
        target = _target_object(vlm_report, goal)
        target_visible = target is not None
        self._observe_target_visibility(target, nav_target)

        if self.state.mode == GOATState.SEARCH:
            if target_visible:
                self._enter_state(GOATState.TRACK, "target_object_visible")
                return self.controller.approach_action(vlm_report, nav_target, goal)
            return self.controller.search_action(nav_target)

        elif self.state.mode == GOATState.TRACK:
            if self._target_lost(max_lost_steps=4):
                self._enter_state(GOATState.SEARCH, "target_lost_in_track")
                return self.controller.search_action(nav_target)
            if (
                target_visible
                and self.step_id >= self.state.verify_cooldown_until_step
                and self._should_enter_verify(vlm_report, goal)
            ):
                self._enter_state(GOATState.VERIFY_STOP, "area_trend_verify_candidate")
                return self.controller.verify_action(vlm_report, goal)
            if self._tracking_stable(vlm_report) and self.state.track_hits >= 2:
                self._enter_state(GOATState.APPROACH, "bbox_or_bearing_stable")
                return self.controller.approach_action(vlm_report, nav_target, goal)
            return self.controller.approach_action(vlm_report, nav_target, goal)

        elif self.state.mode == GOATState.APPROACH:
            if self._target_lost(max_lost_steps=3):
                self._enter_state(GOATState.TRACK, "target_lost_during_approach")
                return self.controller.search_action(nav_target)
            if (
                target_visible
                and self.step_id >= self.state.verify_cooldown_until_step
                and self._should_enter_verify(vlm_report, goal)
            ):
                self._enter_state(GOATState.VERIFY_STOP, "near_and_large_bbox")
                return self.controller.verify_action(vlm_report, goal)
            return self.controller.approach_action(vlm_report, nav_target, goal)

        elif self.state.mode == GOATState.VERIFY_STOP:
            if self.stop_verifier.observe(self.state, vlm_report, goal):
                self._enter_state(GOATState.STOP, "multi_frame_verified")
                return self.controller.stop_action()
            if self.state.verify_attempts >= 6 and self.state.verify_hits == 0:
                self.state.verify_cooldown_until_step = self.step_id + 3
                self._enter_state(GOATState.APPROACH, "verify_failed_no_hits")
                return self.controller.approach_action(vlm_report, nav_target, goal)
            if self._target_lost(max_lost_steps=3):
                self.state.verify_cooldown_until_step = self.step_id + 3
                self._enter_state(GOATState.TRACK, "target_lost_during_verify")
                return self.controller.search_action(nav_target)
            return self.controller.verify_action(vlm_report, goal)

        return self.controller.stop_action()

    def _enter_state(self, mode: GOATState, reason: str) -> None:
        if self.state.mode == mode:
            return
        self.state.mode = mode
        self.state.transition_reason = reason
        self.state.state_steps = 0
        if mode == GOATState.SEARCH:
            self.state.track_hits = 0
            self.state.target_visible_window.clear()
            self.state.active_object_node_id = None
        if mode == GOATState.TRACK:
            self.state.track_hits = 1 if self.state.target_visible_window[-1:] == [True] else 0
        if mode == GOATState.VERIFY_STOP:
            self.state.verify_attempts = 0
            self.state.verify_area_window.clear()
        else:
            self.state.verify_window.clear()
            self.state.verify_hits = 0
            self.state.verify_attempts = 0

    def _observe_target_visibility(
        self,
        target: Optional[Dict[str, Any]],
        nav_target: NavigationTarget,
    ) -> None:
        target_visible = target is not None
        self.state.state_steps += 1
        self.state.target_visible_window.append(target_visible)
        self.state.target_visible_window = self.state.target_visible_window[-5:]
        if target_visible:
            area = _bbox_area(target.get("bbox"))
            if area > 0.0:
                self.state.target_area_window.append(area)
                self.state.target_area_window = self.state.target_area_window[-5:]
                self.state.target_area_trend = _area_trend(
                    self.state.target_area_window
                )
            self.state.last_goal_seen_step = self.step_id
            self.state.target_lost_steps = 0
            self.state.track_hits += 1
            if nav_target.target_type == "object":
                self.state.active_object_node_id = nav_target.node_id
        else:
            self.state.target_lost_steps += 1
            self.state.track_hits = 0

    def _target_lost(self, *, max_lost_steps: int) -> bool:
        if self.state.last_goal_seen_step < 0:
            return False
        return self.state.target_lost_steps >= max_lost_steps

    def _tracking_stable(self, vlm_report: Optional[Dict[str, Any]]) -> bool:
        best = _target_object(vlm_report, self.goal_manager.current_goal())
        if best is None:
            return False
        bbox = best.get("bbox")
        center = _bbox_center_x(bbox)
        if center is None:
            return bool(vlm_report and vlm_report.get("goal_visible", False))
        return 0.25 <= center <= 0.75

    def _should_enter_verify(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        best = _target_object(vlm_report, goal)
        if best is None or goal is None:
            return False

        bbox = best.get("bbox")
        bbox_area = _bbox_area(bbox)
        center = _bbox_center_x(bbox)
        range_bin = str(best.get("range_bin", "")).lower()
        visibility = str(best.get("visibility", "")).lower()
        bbox_confidence = str(best.get("bbox_confidence", "")).lower()
        area_growing = self.state.target_area_trend >= 0.015
        area_grew_over_window = (
            len(self.state.target_area_window) >= 3
            and self.state.target_area_window[-1]
            >= 1.20 * max(0.01, self.state.target_area_window[0])
        )
        mature_memory = False
        goal_label = str(goal.target_object or "").lower().strip()
        for entry in self.memory.object_memory_pool.entries():
            if entry.label.lower().strip() == goal_label and entry.seen_count >= 3:
                mature_memory = True
                break
        if (
            not _range_area_close_enough_for_verify(range_bin, bbox_area)
            or center is None
            or not 0.15 <= center <= 0.90
            or visibility in {"occluded", "not_visible"}
            or bbox_confidence == "low"
            or not (area_growing or area_grew_over_window or mature_memory)
        ):
            return False
        return True

    def _update_stuck_monitor(
        self,
        nav_target: NavigationTarget,
        action: Dict[str, Any],
    ) -> None:
        if self._pose is None:
            return

        current_position = self._pose.position.copy()
        if self._last_pose_for_stuck is None:
            self._last_pose_for_stuck = current_position
            return

        motion_delta = _distance(current_position, self._last_pose_for_stuck)
        self.state.last_motion_delta = motion_delta
        self._last_pose_for_stuck = current_position

        action_name = str(action.get("action", ""))
        target_id = nav_target.node_id
        target_type = nav_target.target_type
        can_be_stuck = (
            self.state.mode in (GOATState.SEARCH, GOATState.TRACK, GOATState.APPROACH)
            and action_name in {"navigate", "move_forward"}
            and target_id is not None
            and target_type in {"frontier", "room", "hypothesis", "object"}
            and not bool(self._vlm_for_control and self._vlm_for_control.get("goal_visible", False))
        )

        if not can_be_stuck:
            self._reset_stuck_monitor(clear_reason=False)
            return

        if target_id != self._stuck_monitor_target_id:
            self._stuck_monitor_target_id = target_id
            self._stuck_marked_target_id = None
            self.state.stuck_steps = 0
            self.state.stuck_target_node_id = target_id

        if motion_delta < 0.03:
            self.state.stuck_steps += 1
        else:
            self.state.stuck_steps = 0
            self.state.last_stuck_reason = ""

        if self.state.stuck_steps < 8:
            return

        self.state.stuck_recoveries += 1
        self.state.last_stuck_reason = (
            f"stuck_on_{target_type}:{target_id}; "
            f"delta={motion_delta:.3f}; steps={self.state.stuck_steps}"
        )
        if target_type == "object":
            self.memory.navigation_layer["last_failed_target"] = {
                "node_id": target_id,
                "target_type": target_type,
                "failed_count": None,
                "unreachable_count": None,
                "reason": "object_anchor_scan_no_progress",
                "step": self.topo_map.current_step,
            }
        else:
            self.memory.mark_navigation_target_failed(
                target_id,
                target_type=target_type,
                unreachable=True,
                reason="stuck_no_progress",
            )
        self.state.stuck_steps = 0

        action.clear()
        reason = (
            "object_anchor_scan_no_progress"
            if target_type == "object"
            else "stuck_recovery"
        )
        action.update(
            {
                "action": "turn_left",
                "reason": reason,
                "blocked_target_node_id": target_id,
                "blocked_target_type": target_type,
            }
        )

    def _reset_stuck_monitor(self, *, clear_reason: bool = True) -> None:
        self.state.stuck_steps = 0
        self.state.stuck_target_node_id = None
        self._stuck_monitor_target_id = None
        self._stuck_marked_target_id = None
        if clear_reason:
            self.state.last_stuck_reason = ""

    def _debug_snapshot(self, need_vlm: bool) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "state": self.state.mode.value,
            "transition_reason": self.state.transition_reason,
            "state_steps": self.state.state_steps,
            "clip_room": self._last_clip_hint.room_label,
            "clip_goal_sim": self._last_clip_hint.best_goal_sim,
            "need_vlm": need_vlm,
            "vlm_reason": self.perception.last_vlm_reason,
            "fresh_vlm": bool(self._fresh_vlm_report),
            "cached_vlm_step": self._cached_vlm_step,
            "using_cached_vlm": bool(
                self._vlm_for_control is not None
                and self._fresh_vlm_report is None
            ),
            "vlm_goal_visible": bool(
                self._vlm_for_control and self._vlm_for_control.get("goal_visible", False)
            ),
            "target_visible_window": list(self.state.target_visible_window),
            "target_area_window": [
                round(float(area), 4) for area in self.state.target_area_window
            ],
            "target_area_trend": round(float(self.state.target_area_trend), 4),
            "verify_area_window": [
                round(float(area), 4) for area in self.state.verify_area_window
            ],
            "track_hits": self.state.track_hits,
            "target_lost_steps": self.state.target_lost_steps,
            "last_goal_seen_step": self.state.last_goal_seen_step,
            "active_object_node_id": self.state.active_object_node_id,
            "hypothesis_count": len(self.memory.hypothesis_pool),
            "active_hypotheses": self.memory.hypothesis_pool.to_debug(
                self.topo_map.current_step
            ),
            "object_memory_count": len(self.memory.object_memory_pool),
            "object_memory": self.memory.object_memory_pool.to_debug(),
            "proposal_count": len(self._last_proposals),
            "proposal_debug": dict(self.proposal_builder.last_debug),
            "selected_proposal": (
                self.proposal_builder._proposal_debug(self._last_nav_target.proposal)
                if self._last_nav_target.proposal is not None
                else None
            ),
            "target_node_id": self._last_nav_target.node_id,
            "target_type": self._last_nav_target.target_type,
            "verify_hits": self.state.verify_hits,
            "verify_attempts": self.state.verify_attempts,
            "verify_cooldown_until_step": self.state.verify_cooldown_until_step,
            "stuck_steps": self.state.stuck_steps,
            "stuck_recoveries": self.state.stuck_recoveries,
            "stuck_target_node_id": self.state.stuck_target_node_id,
            "last_motion_delta": self.state.last_motion_delta,
            "last_stuck_reason": self.state.last_stuck_reason,
            "last_failed_target": dict(
                self.memory.navigation_layer.get("last_failed_target", {})
            ),
        }


def _goal_text(goal: GoalNode) -> str:
    parts = [
        str(goal.description or "").strip(),
        " ".join(str(attr) for attr in goal.attributes or []),
        str(goal.target_object or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _goal_id(goal: GoalNode) -> str:
    return str(goal.image_cache_key or goal.goal_image_id or goal.target_object)


def _best_label(scores: List[Any]) -> Optional[str]:
    if not scores:
        return None
    return str(scores[0][0])


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _estimate_object_position(pose: AgentPose, obj: Dict[str, Any]) -> np.ndarray:
    range_bin = str(obj.get("range_bin", "unknown")).lower().strip()
    forward_distance = _range_bin_distance(range_bin)

    bbox_center = _bbox_center_x(obj.get("bbox"))
    if bbox_center is None:
        bearing = str(obj.get("bearing", "")).lower().strip()
        if bearing == "left":
            lateral_fraction = -0.45
        elif bearing == "right":
            lateral_fraction = 0.45
        else:
            lateral_fraction = 0.0
    else:
        lateral_fraction = float(np.clip((bbox_center - 0.5) * 1.4, -0.75, 0.75))

    heading = float(pose.heading)
    forward = np.array(
        [-np.sin(heading), 0.0, -np.cos(heading)],
        dtype=np.float32,
    )
    right = np.array(
        [np.cos(heading), 0.0, -np.sin(heading)],
        dtype=np.float32,
    )
    lateral_distance = lateral_fraction * max(0.4, forward_distance * 0.8)
    estimate = (
        pose.position
        + forward * forward_distance
        + right * lateral_distance
    ).astype(np.float32)
    estimate[1] = 0.0
    return estimate


def _range_bin_distance(range_bin: str) -> float:
    if range_bin in {"very_near", "close"}:
        return 0.6
    if range_bin == "near":
        return 1.35
    if range_bin in {"mid", "medium"}:
        return 2.2
    if range_bin == "far":
        return 3.2
    return 1.5


def _range_area_close_enough_for_verify(range_bin: str, bbox_area: float) -> bool:
    if range_bin in {"very_near", "close"}:
        return bbox_area >= 0.05
    if range_bin == "near":
        return bbox_area >= 0.14
    return False


def _range_area_close_enough_for_stop(range_bin: str, bbox_area: float) -> bool:
    if range_bin in {"very_near", "close"}:
        return bbox_area >= 0.06
    if range_bin == "near":
        return bbox_area >= 0.20
    return False


def _vlm_report_to_dict(report: PerceptionReport, *, reason: str, mode: str) -> Dict[str, Any]:
    return {
        "fresh": True,
        "step": int(report.step_id),
        "mode": mode,
        "trigger_reason": reason,
        "room": report.room_label,
        "room_confidence": float(report.room_confidence),
        "goal_visible": bool(report.goal_visible),
        "goal_match_confidence": float(report.goal_match_confidence),
        "target_direction": report.target_direction,
        "target_visibility": report.target_visibility,
        "apparent_scale": report.apparent_scale,
        "stop_candidate": bool(report.stop_candidate),
        "recommended_action": report.recommended_action,
        "goal_reason": report.goal_reason,
        "objects": [obj.to_dict() for obj in report.objects],
        "landmarks": list(report.portals),
        "raw": dict(report.raw),
    }


def _target_object(
    vlm_report: Optional[Dict[str, Any]],
    goal: Optional[GoalNode],
) -> Optional[Dict[str, Any]]:
    if not vlm_report or goal is None:
        return None
    goal_label = str(goal.target_object or "").lower().strip()
    objects = [dict(obj) for obj in vlm_report.get("objects", []) if obj]
    matched = [
        obj
        for obj in objects
        if str(obj.get("label", "")).lower().strip() == goal_label
    ]
    if matched:
        return max(matched, key=lambda obj: float(obj.get("confidence", 0.0) or 0.0))

    if bool(vlm_report.get("goal_visible", False)) and objects:
        goal_match_confidence = float(vlm_report.get("goal_match_confidence", 0.0) or 0.0)
        visible_objects = [
            obj
            for obj in objects
            if str(obj.get("visibility", "")).lower() not in {"occluded", "not_visible"}
        ]
        candidates = visible_objects or objects
        if len(candidates) == 1 or goal_match_confidence >= 0.5:
            return max(candidates, key=lambda obj: float(obj.get("confidence", 0.0) or 0.0))
    return None


def _target_bbox(
    vlm_report: Optional[Dict[str, Any]],
    goal: Optional[GoalNode],
) -> Optional[Any]:
    target = _target_object(vlm_report, goal)
    return None if target is None else target.get("bbox")


def _bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return 0.0
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return max(0.0, min(1.0, (x2 - x1) * (y2 - y1)))


def _bbox_center_x(bbox: Any) -> Optional[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, _y1, x2, _y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return 0.5 * (x1 + x2)


def _area_trend(areas: List[float]) -> float:
    valid = [float(area) for area in areas if area > 0.0]
    if len(valid) < 2:
        return 0.0
    return float(valid[-1] - valid[0])


ConfTopoGOATAgent = GOATAgent


__all__ = [
    "AgentPose",
    "AgentState",
    "ClipHint",
    "Controller",
    "GOATAgent",
    "GOATState",
    "GoalManager",
    "Hypothesis",
    "HypothesisPool",
    "MemoryManager",
    "NavigationTarget",
    "ObjectMemoryEntry",
    "ObjectMemoryPool",
    "PerceptionModule",
    "Planner",
    "ProposalBuilder",
    "StopVerifier",
    "ConfTopoGOATAgent",
]
