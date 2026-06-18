"""ConfTopo-GOAT Agent (modular scheduler).

Drop-in replacement for the monolithic ``goat_agent.py``.  Import
``ConfTopoGOATAgent`` from here — it is the same class as ``GoatAgent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import math
import numpy as np

from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType, SemanticNode
from conftopo.core.instruction_graph import GoalNode, InstructionGraph
from conftopo.core.landmark_roles import is_structural_label
from conftopo.perception.clip_gdino_report_builder import ClipGdinoReportBuilder
from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.perception_trigger import PerceptionTrigger, TriggerState
from conftopo.perception.vlm_perceiver import VLMPerceiver


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


_CENTER_BEARINGS = frozenset({"center", "front", "front-center", "unknown"})


def _split_compound_label(label: Optional[str]) -> Set[str]:
    if not label:
        return set()
    text = str(label).lower().strip()
    parts = {text}
    for sep in (" and ", " or ", ","):
        if sep in text:
            parts.update(p.strip() for p in text.split(sep) if p.strip())
    return parts


def _label_matches(label: Optional[str], candidates: Set[str]) -> bool:
    if not label or not candidates:
        return False
    lab = str(label).lower().strip()
    return lab in candidates or any(c in lab or lab in c for c in candidates)


def _bbox_area(bbox: Sequence[float] | None) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if w > 1.0 or h > 1.0:
        w /= 256.0
        h /= 256.0
    return float(max(0.0, w) * max(0.0, h))


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


OBJECT_CATEGORY_ROOM_PRIORS: Dict[str, List[str]] = {
    "plush toy": ["bedroom", "living room", "closet"],
    "stuffed": ["bedroom", "living room", "closet"],
    "toy": ["bedroom", "living room", "closet"],
    "wardrobe": ["bedroom"],
    "bed": ["bedroom"],
    "pillow": ["bedroom"],
    "sofa": ["living room"],
    "couch": ["living room"],
    "tv": ["living room"],
    "towel": ["bathroom"],
    "toilet": ["bathroom"],
    "sink": ["bathroom", "kitchen"],
    "fridge": ["kitchen"],
    "refrigerator": ["kitchen"],
    "oven": ["kitchen"],
    "microwave": ["kitchen"],
    "pan": ["kitchen"],
    "plate": ["kitchen", "dining room"],
}

OBJECT_CATEGORY_LANDMARK_PRIORS: Dict[str, List[str]] = {
    "plush toy": ["bed", "sofa", "couch", "shelf", "cabinet", "table", "dresser", "nightstand", "desk", "chair"],
    "stuffed": ["bed", "sofa", "couch", "shelf", "cabinet", "table", "dresser", "nightstand", "desk", "chair"],
    "toy": ["bed", "sofa", "couch", "shelf", "cabinet", "table", "dresser", "nightstand", "desk"],
    "wardrobe": ["bed", "cabinet", "dresser"],
    "pillow": ["bed", "sofa", "couch"],
    "tv": ["sofa", "cabinet", "table"],
    "towel": ["sink", "toilet", "rack"],
    "plate": ["table", "counter", "sink"],
}

DEFAULT_HEAVY_OBJECT_VOCABULARY: List[str] = [
    "rack", "shelf", "cabinet", "counter", "table", "chair", "sofa", "bed",
    "desk", "door", "window", "sink", "toilet", "fridge", "oven", "microwave",
]

DEFAULT_STRUCTURAL_HEAVY_VOCABULARY: List[str] = [
    "door", "doorway", "window", "arch", "archway", "counter", "gate",
]


def _category_priors(target_object: Optional[str], table: Dict[str, List[str]]) -> List[str]:
    if not target_object:
        return []
    text = str(target_object).lower()
    for key, vals in table.items():
        if key in text:
            return list(vals)
    return []


class NavPhase(str, Enum):
    GLOBAL_SEARCH = "GLOBAL_SEARCH"
    ROUTE_TO_STRUCTURE = "ROUTE_TO_STRUCTURE"
    ROUTE_TO_OBJECT_ANCHOR = "ROUTE_TO_OBJECT_ANCHOR"
    LOCAL_VISUAL_APPROACH = "LOCAL_VISUAL_APPROACH"
    PEAK_RETURN = "PEAK_RETURN"
    STOP_VERIFY = "STOP_VERIFY"
    RECOVERY = "RECOVERY"
    STOP = "STOP"


@dataclass
class NewGoatConfig:
    vlm_cache_max_age: int = 3
    vlm_cache_max_position_delta: float = 0.35
    vlm_cache_max_heading_delta: float = 0.5
    fresh_vlm_stop_max_age: int = 4
    min_frontier_distance: float = 1.0
    frontier_step_size: float = 2.5
    frontier_merge_radius: float = 1.0
    waypoint_merge_radius: float = 0.45
    min_move_for_frontier: float = 0.5
    servo_entry_evidence: int = 2
    stop_buffer_size: int = 3
    anchor_blacklist_steps: int = 40
    object_write_throttle_steps: int = 4
    no_progress_window: int = 8
    no_progress_distance: float = 0.12
    phase_timeout_steps: int = 40
    local_approach_timeout_steps: int = 48
    bbox_min_approach: float = 0.04
    bbox_min_stop: float = 0.12
    bbox_peak_ratio: float = 0.80
    stop_pose_radius: float = 0.55
    room_writeback_confidence: float = 0.7
    min_forward_before_stop: int = 3
    min_approach_distance: float = 1.0
    bbox_plateau_window: int = 3
    bbox_plateau_growth: float = 0.02
    bbox_stop_min: float = 0.45
    min_forward_after_plateau: int = 5
    servo_lost_count_max: int = 5
    anchor_min_confidence: float = 0.3
    anchor_stale_age: int = 20
    anchor_single_frame_min_confidence: float = 0.5
    anchor_servo_enter_distance: float = 2.0
    anchor_waypoint_reach_distance: float = 0.55
    success_stop_distance: float = 1.0
    stop_retreat_margin: float = 0.3
    anchor_at_zone_distance: float = 0.6


@dataclass
class PerceptionPacket:
    report: PerceptionReport
    source: str
    fresh_vlm: bool = False
    cached_vlm: bool = False
    vlm_mode: str = "explore"
    step_id: int = 0
    trigger_reason: str = ""


@dataclass
class MemoryUpdateSummary:
    cur_vp_id: Optional[str] = None
    room_label: str = "unknown"
    persistent_writes: int = 0
    transient_updates: int = 0
    debug_only_writes: int = 0
    object_merge_count_this_step: int = 0
    skipped_writes: int = 0
    written_node_ids: List[str] = field(default_factory=list)

    @property
    def memory_write_count(self) -> int:
        return self.persistent_writes + self.transient_updates


@dataclass
class StructureTarget:
    node_id: Optional[str] = None
    target_type: str = "none"
    position: Optional[np.ndarray] = None
    reason: str = "no_structure_target"

    def to_debug(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "target_type": self.target_type,
            "position": self.position.tolist() if self.position is not None else None,
            "reason": self.reason,
        }


@dataclass
class NavTarget:
    target_position: Optional[np.ndarray] = None
    target_node_id: Optional[str] = None
    target_type: str = "none"
    reason: str = "no_target"
    expected_phase_after_reach: NavPhase = NavPhase.GLOBAL_SEARCH

    def distance_from(self, position: Optional[np.ndarray]) -> Optional[float]:
        if position is None or self.target_position is None:
            return None
        return float(np.linalg.norm(self.target_position - position))

    def to_debug(self, position: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return {
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "target_node_id": self.target_node_id,
            "target_type": self.target_type,
            "reason": self.reason,
            "expected_phase_after_reach": self.expected_phase_after_reach.value,
            "target_distance": self.distance_from(position),
        }


@dataclass
class ServoAction:
    action: str
    reason: str
    next_phase: NavPhase
    target_position: Optional[np.ndarray] = None


@dataclass
class StopDecision:
    should_stop: bool
    layer1: bool
    layer2: bool
    layer3: bool
    reason: str

    def to_debug(self) -> Dict[str, Any]:
        return {
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "stop_reason": self.reason,
            "stop_allowed": self.should_stop,
        }


@dataclass
class PlanDecision:
    action: str = "navigate"
    plan_action: str = "navigate"
    target_position: Optional[np.ndarray] = None
    target_node_id: Optional[str] = None
    target_type: str = "none"
    reason: str = ""
    expected_phase_after_reach: NavPhase = NavPhase.GLOBAL_SEARCH
    candidate_ids: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    is_exploration: bool = True
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServoState:
    active_anchor_id: Optional[str] = None
    best_bbox_area: float = 0.0
    best_stop_pose: Optional[np.ndarray] = None
    visual_advance_steps: int = 0
    forward_action_count: int = 0
    approach_travel_distance: float = 0.0
    last_approach_position: Optional[np.ndarray] = None
    bbox_history: List[float] = field(default_factory=list)
    lost_count: int = 0
    confirm_buffer: List[bool] = field(default_factory=list)
    stop_buffer: List[bool] = field(default_factory=list)
    fresh_vlm_stop_step: Optional[int] = None
    last_fresh_bbox: float = 0.0
    aligned_at_peak: bool = False
    peak_bearing: str = "unknown"
    plateau_forward_count: int = 0
    best_anchor_distance: float = float("inf")
    last_anchor_distance: Optional[float] = None
    anchor_distance_history: List[float] = field(default_factory=list)
    instance_distance_history: List[float] = field(default_factory=list)
    best_instance_distance: float = float("inf")
    entry_step: int = 0

    def reset(self) -> None:
        self.active_anchor_id = None
        self.best_bbox_area = 0.0
        self.best_stop_pose = None
        self.visual_advance_steps = 0
        self.forward_action_count = 0
        self.approach_travel_distance = 0.0
        self.last_approach_position = None
        self.bbox_history = []
        self.lost_count = 0
        self.confirm_buffer = []
        self.stop_buffer = []
        self.fresh_vlm_stop_step = None
        self.last_fresh_bbox = 0.0
        self.aligned_at_peak = False
        self.peak_bearing = "unknown"
        self.plateau_forward_count = 0
        self.best_anchor_distance = float("inf")
        self.last_anchor_distance = None
        self.anchor_distance_history = []
        self.instance_distance_history = []
        self.best_instance_distance = float("inf")
        self.entry_step = 0


class GoalManager:
    def __init__(self) -> None:
        self.current_goal: Optional[GoalNode] = None
        self.previous_goal_label: Optional[str] = None
        self.reuse_debug: Dict[str, Any] = {}

    @property
    def target_object(self) -> Optional[str]:
        return getattr(self.current_goal, "target_object", None)

    @property
    def target_labels(self) -> Set[str]:
        return _split_compound_label(self.target_object)

    @property
    def room_prior(self) -> List[str]:
        return _as_list(getattr(self.current_goal, "room_prior", []))

    @property
    def landmark_prior(self) -> List[str]:
        return _as_list(getattr(self.current_goal, "landmarks", []))

    @property
    def goal_key(self) -> str:
        return (self.target_object or "").lower().strip()

    @property
    def is_repeated_goal(self) -> bool:
        return bool(self.previous_goal_label and self.previous_goal_label == self.target_object)

    def goal_context(self) -> str:
        parts = []
        if self.room_prior:
            parts.append("Target likely appears in: " + ", ".join(self.room_prior))
        if self.landmark_prior:
            parts.append("Useful visible context objects: " + ", ".join(self.landmark_prior))
        parts.append(
            "Only report objects that are actually visible in the image. "
            "If the target or context is not visible, say it is not visible; do not guess from priors."
        )
        return "; ".join(parts)

    def set_new_goal(self, goal: GoalNode, topo_map: DynamicTopoMap) -> None:
        old = self.target_object
        self.previous_goal_label = old
        self.current_goal = goal
        self.reuse_debug = self._scan_reuse(topo_map)

    def _scan_reuse(self, topo_map: DynamicTopoMap) -> Dict[str, Any]:
        labels = self.target_labels
        same_label = []
        unfailed = []
        failed_active = []
        failed_expired = []
        contexts = []
        step = topo_map.current_step
        for node in topo_map.get_nodes_by_type(NodeType.OBJECT):
            role = node.attributes.get("semantic_role")
            if role == "context_object":
                if node.label in self.landmark_prior:
                    contexts.append(node.node_id)
                continue
            if not _label_matches(node.label, labels):
                continue
            same_label.append(node.node_id)
            until = int(node.attributes.get("blacklisted_until", -1))
            if until >= step:
                failed_active.append(node.node_id)
            elif int(node.attributes.get("failed_approach_count", 0)) > 0:
                failed_expired.append(node.node_id)
            else:
                unfailed.append(node.node_id)
            node.attributes["repeated_goal_source"] = self.is_repeated_goal
        return {
            "is_repeated_goal": self.is_repeated_goal,
            "same_label_anchors": same_label,
            "unfailed_anchors": unfailed,
            "failed_active_anchors": failed_active,
            "failed_expired_anchors": failed_expired,
            "same_room_context": contexts,
            "memory_reuse_hits": len(unfailed) + len(failed_expired),
        }


class PerceptionManager:
    def __init__(self, config: ConfTopoConfig, new_config: NewGoatConfig) -> None:
        self.config = config
        self.new_config = new_config
        self.light = LightPerceiver(room_labels=config.perception.room_labels)
        self.builder = ClipGdinoReportBuilder()
        self.trigger = PerceptionTrigger(config.perception, config.memory)
        self.trigger_state = TriggerState()
        self.vlm_perceiver: Optional[VLMPerceiver] = None
        self.last_vlm_report: Optional[PerceptionReport] = None
        self.last_vlm_goal: str = ""
        self.last_vlm_position: Optional[np.ndarray] = None
        self.last_vlm_heading: float = 0.0
        self.last_vlm_mode: str = ""
        self.last_vlm_step: Optional[int] = None

    def reset_short_term(self) -> None:
        self.last_vlm_report = None
        self.last_vlm_goal = ""
        self.last_vlm_position = None
        self.last_vlm_heading = 0.0
        self.last_vlm_mode = ""
        self.last_vlm_step = None
        self.trigger_state = TriggerState()

    def set_vlm_perceiver(self, perceiver: Optional[VLMPerceiver]) -> None:
        self.vlm_perceiver = perceiver

    def set_goal(self, goal: Optional[GoalNode]) -> None:
        if goal is None:
            return
        self.light.set_goal_labels(
            [goal.target_object],
            goal.target_embedding[np.newaxis, :] if goal.target_embedding is not None else None,
        )
        if goal.landmarks:
            self.light.set_landmark_labels(goal.landmarks, goal.landmark_embeddings)

    def run_light(self, visual_embed: Optional[np.ndarray], step_id: int) -> PerceptionPacket:
        out = self.light.perceive(visual_embed) if visual_embed is not None else {}
        report = self.builder.build_light(out, visual_embed, step_id)
        return PerceptionPacket(report=report, source=report.source, step_id=step_id)

    def run(
        self,
        *,
        rgb: Any,
        visual_embed: Optional[np.ndarray],
        position: Optional[np.ndarray],
        heading: float,
        topo_map: DynamicTopoMap,
        goal_manager: GoalManager,
        nav_phase: NavPhase,
        step_id: int,
        has_near_goal_object: bool,
    ) -> PerceptionPacket:
        packet = self.run_light(visual_embed, step_id)
        vlm_mode = "confirm" if nav_phase in (
            NavPhase.LOCAL_VISUAL_APPROACH,
            NavPhase.STOP_VERIFY,
            NavPhase.PEAK_RETURN,
        ) else "explore"
        self.trigger_state.nav_phase = {
            NavPhase.LOCAL_VISUAL_APPROACH: "approach_confirm",
            NavPhase.STOP_VERIFY: "confirm",
            NavPhase.PEAK_RETURN: "approach",
        }.get(nav_phase, "explore")

        force_fresh = nav_phase == NavPhase.STOP_VERIFY
        if force_fresh and self.vlm_perceiver is not None and rgb is not None:
            should_run, reason = True, "stop_verify_fresh"
        else:
            should_run, reason = self._should_run_vlm(
                packet.report, rgb, position, topo_map, has_near_goal_object, step_id,
            )
            if (
                not should_run
                and nav_phase == NavPhase.GLOBAL_SEARCH
                and not self._goal_has_anchor(topo_map, goal_manager)
            ):
                interval = max(2, int(self.config.perception.heavy_interval) // 2)
                if (
                    self.trigger_state.goal_local_step % interval == 0
                    and (
                        self.trigger_state.last_heavy_step is None
                        or step_id - self.trigger_state.last_heavy_step >= interval
                    )
                ):
                    should_run, reason = True, "search_no_anchor"
        if should_run and self.vlm_perceiver is not None and rgb is not None:
            report = self.vlm_perceiver.perceive(
                rgb,
                goal_manager.target_object or "explore",
                visual_embed=visual_embed,
                step_id=step_id,
                context=goal_manager.goal_context(),
            )
            self.last_vlm_report = report
            self.last_vlm_goal = goal_manager.goal_key
            self.last_vlm_position = None if position is None else position.copy()
            self.last_vlm_heading = float(heading)
            self.last_vlm_mode = vlm_mode
            self.last_vlm_step = step_id
            self.trigger.record_run(self.trigger_state, step_id, reason)
            return PerceptionPacket(
                report=report,
                source="vlm",
                fresh_vlm=True,
                cached_vlm=False,
                vlm_mode=vlm_mode,
                step_id=step_id,
                trigger_reason=reason,
            )

        if nav_phase != NavPhase.STOP_VERIFY:
            cached = self._valid_cached_vlm(goal_manager.goal_key, position, heading, vlm_mode, step_id)
            if cached is not None:
                cached_report = PerceptionReport.from_dict(cached.to_dict())
                cached_report.source = "cached_vlm"
                return PerceptionPacket(
                    report=cached_report,
                    source="cached_vlm",
                    fresh_vlm=False,
                    cached_vlm=True,
                    vlm_mode=vlm_mode,
                    step_id=step_id,
                    trigger_reason="cache_valid",
                )
        packet.vlm_mode = vlm_mode
        packet.trigger_reason = reason
        return packet

    def _goal_has_anchor(self, topo_map: DynamicTopoMap, goal_manager: GoalManager) -> bool:
        step = topo_map.current_step
        for node in topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.attributes.get("semantic_role") != "object_anchor":
                continue
            if not _label_matches(node.label, goal_manager.target_labels):
                continue
            if int(node.attributes.get("blacklisted_until", -1)) >= step:
                continue
            if float(node.confidence) >= self.new_config.anchor_min_confidence:
                return True
        return False

    def _should_run_vlm(
        self,
        report: PerceptionReport,
        rgb: Any,
        position: Optional[np.ndarray],
        topo_map: DynamicTopoMap,
        has_near_goal_object: bool,
        step_id: int,
    ) -> Tuple[bool, str]:
        if self.vlm_perceiver is None:
            return False, "no_vlm_perceiver"
        self.trigger_state.goal_local_step += 1
        return self.trigger.should_run(
            self.trigger_state,
            step_id,
            rgb,
            report.best_goal_sim,
            position,
            topo_map,
            has_near_goal_object,
            heavy_perceiver_available=True,
        )

    def _valid_cached_vlm(
        self,
        goal_key: str,
        position: Optional[np.ndarray],
        heading: float,
        vlm_mode: str,
        step_id: int,
    ) -> Optional[PerceptionReport]:
        if self.last_vlm_report is None or self.last_vlm_step is None:
            return None
        if step_id - self.last_vlm_step > self.new_config.vlm_cache_max_age:
            return None
        if self.last_vlm_goal != goal_key or self.last_vlm_mode != vlm_mode:
            return None
        if position is not None and self.last_vlm_position is not None:
            if float(np.linalg.norm(position - self.last_vlm_position)) > self.new_config.vlm_cache_max_position_delta:
                return None
        if abs(_angle_delta(float(heading), self.last_vlm_heading)) > self.new_config.vlm_cache_max_heading_delta:
            return None
        return self.last_vlm_report


class MemoryWriter:
    def __init__(self, new_config: NewGoatConfig) -> None:
        self.cfg = new_config
        self.cur_vp_id: Optional[str] = None
        self.prev_vp_id: Optional[str] = None
        self.last_frontier_position: Optional[np.ndarray] = None
        self.write_throttle: Dict[Tuple[str, str, str], int] = {}

    def reset_episode(self) -> None:
        self.cur_vp_id = None
        self.prev_vp_id = None
        self.last_frontier_position = None
        self.write_throttle = {}

    def reset_goal(self) -> None:
        self.write_throttle = {}

    def update(
        self,
        topo_map: DynamicTopoMap,
        packet: PerceptionPacket,
        nav_phase: NavPhase,
        position: Optional[np.ndarray],
        heading: float,
        visual_embed: Optional[np.ndarray],
        goal_manager: GoalManager,
        active_anchor_id: Optional[str],
    ) -> MemoryUpdateSummary:
        summary = MemoryUpdateSummary(room_label=packet.report.room_label)
        if position is None:
            summary.debug_only_writes += 1
            return summary
        cur_vp = self._write_waypoint(topo_map, position, visual_embed)
        summary.cur_vp_id = cur_vp
        self._write_frontiers(topo_map, position, heading, cur_vp)
        self._write_room(topo_map, packet.report, position, cur_vp, summary)
        self._write_objects(
            topo_map, packet, nav_phase, position, heading, cur_vp,
            goal_manager, active_anchor_id, summary,
        )
        return summary

    def _write_waypoint(
        self,
        topo_map: DynamicTopoMap,
        position: np.ndarray,
        visual_embed: Optional[np.ndarray],
    ) -> str:
        nearest = topo_map.find_nearest_node(position, NodeType.WAYPOINT_VISITED)
        if nearest is not None and float(np.linalg.norm(nearest.position - position)) <= self.cfg.waypoint_merge_radius:
            nearest.position = position.astype(np.float32)
            nearest.step_id = topo_map.current_step
            nearest.visit_count += 1
            self.cur_vp_id = nearest.node_id
            return nearest.node_id
        node_id = topo_map.add_node(
            NodeType.WAYPOINT_VISITED,
            position=position,
            embedding=visual_embed,
            confidence=1.0,
            attributes={"semantic_role": "visited_waypoint"},
        )
        if self.cur_vp_id and self.cur_vp_id != node_id:
            topo_map.add_edge(self.cur_vp_id, node_id, EdgeType.NAVIGABLE)
        self.prev_vp_id = self.cur_vp_id
        self.cur_vp_id = node_id
        return node_id

    def _write_frontiers(
        self,
        topo_map: DynamicTopoMap,
        position: np.ndarray,
        heading: float,
        cur_vp: str,
    ) -> None:
        moved = self.last_frontier_position is None or (
            float(np.linalg.norm(position - self.last_frontier_position)) >= self.cfg.min_move_for_frontier
        )
        all_consumed = not any(
            not n.attributes.get("consumed")
            and int(n.attributes.get("blacklisted_until", -1)) < topo_map.current_step
            for n in topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER)
        )
        if not moved and not all_consumed:
            return
        self.last_frontier_position = position.copy()
        step_size = self.cfg.frontier_step_size
        if all_consumed and not moved:
            step_size = step_size * 1.5
        for delta in (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi):
            angle = heading + delta
            pos = position + np.array(
                [-math.sin(angle) * step_size, 0.0, -math.cos(angle) * step_size],
                dtype=np.float32,
            )
            if topo_map.has_nearby_visited(pos, radius=self.cfg.frontier_merge_radius):
                continue
            existing = topo_map.find_nearest_node(pos, NodeType.WAYPOINT_FRONTIER)
            if existing is not None and float(np.linalg.norm(existing.position - pos)) < self.cfg.frontier_merge_radius:
                continue
            fid = topo_map.add_node(
                NodeType.WAYPOINT_FRONTIER,
                position=pos,
                confidence=0.5,
                attributes={"semantic_role": "frontier"},
            )
            topo_map.add_edge(cur_vp, fid, EdgeType.NAVIGABLE)

    def _write_room(
        self,
        topo_map: DynamicTopoMap,
        report: PerceptionReport,
        position: np.ndarray,
        cur_vp: str,
        summary: MemoryUpdateSummary,
    ) -> None:
        if report.room_label == "unknown" or report.room_confidence < self.cfg.room_writeback_confidence:
            return
        cur_node = topo_map.get_node(cur_vp)
        if cur_node is not None:
            cur_node.attributes["room_label"] = report.room_label
            cur_node.attributes["room_confidence"] = float(report.room_confidence)
        nearby_rooms = topo_map.find_nodes_within_radius(position, radius=5.0, node_type=NodeType.ROOM)
        same_label = [n for n in nearby_rooms if n.label == report.room_label]
        if same_label:
            best = min(same_label, key=lambda n: float(np.linalg.norm(n.position - position)))
            best.attributes["last_seen_step"] = topo_map.current_step
            best.confidence = max(best.confidence, float(report.room_confidence))
            summary.transient_updates += 1
            return
        if nearby_rooms and float(np.linalg.norm(nearby_rooms[0].position - position)) < 2.0:
            summary.transient_updates += 1
            return
        rid = topo_map.add_node(
            NodeType.ROOM,
            position=position,
            label=report.room_label,
            confidence=float(report.room_confidence),
            attributes={"semantic_role": "room_summary", "summary_type": "room_region"},
        )
        topo_map.add_edge(cur_vp, rid, EdgeType.BELONGS_TO)
        summary.persistent_writes += 1
        summary.written_node_ids.append(rid)

    def _write_objects(
        self,
        topo_map: DynamicTopoMap,
        packet: PerceptionPacket,
        nav_phase: NavPhase,
        position: np.ndarray,
        heading: float,
        cur_vp: str,
        goal_manager: GoalManager,
        active_anchor_id: Optional[str],
        summary: MemoryUpdateSummary,
    ) -> None:
        if not packet.report.objects:
            return

        suppress_writes = (
            packet.cached_vlm
            or (packet.vlm_mode == "confirm"
                and nav_phase in (
                    NavPhase.LOCAL_VISUAL_APPROACH,
                    NavPhase.STOP_VERIFY,
                    NavPhase.PEAK_RETURN,
                ))
        )
        if suppress_writes:
            self._transient_anchor_update(topo_map, packet.report.objects, active_anchor_id, position, summary)
            return

        for obs in packet.report.objects:
            role = self._semantic_role(obs, goal_manager)
            key = (cur_vp, str(obs.label).lower().strip(), nav_phase.value)
            last = self.write_throttle.get(key)
            if last is not None and topo_map.current_step - last <= self.cfg.object_write_throttle_steps:
                summary.skipped_writes += 1
                continue
            self.write_throttle[key] = topo_map.current_step

            if role == "object_anchor":
                target_relevance = 1.0
            elif is_structural_label(obs.label):
                target_relevance = 0.25
            else:
                target_relevance = 0.0

            obj_id, merged = topo_map.upsert_object_observation(
                label=obs.label,
                bbox=obs.bbox,
                confidence=obs.confidence,
                position=position,
                embedding=obs.embedding,
                viewpoint_id=cur_vp,
                view_heading=heading,
                room_context=packet.report.room_label,
                target_relevance=target_relevance,
                source=obs.source,
                spatial_attrs={
                    "semantic_role": role,
                    "anchor_waypoint_id": cur_vp,
                    "anchor_waypoint_position": position.tolist(),
                    "position_source": "anchor_waypoint",
                    "bearing": obs.bearing,
                    "range_bin": obs.range_bin,
                    "visibility": obs.visibility,
                    "visible": obs.visible,
                    "bbox_area": _bbox_area(obs.bbox),
                },
            )
            node = topo_map.get_node(obj_id)
            if node is not None:
                node.attributes["semantic_role"] = role
                if role == "object_anchor":
                    node.attributes["goal_detection_step"] = topo_map.current_step
                    node.attributes["goal_detection_confidence"] = float(obs.confidence)
            summary.persistent_writes += 1
            summary.object_merge_count_this_step += int(bool(merged))
            summary.written_node_ids.append(obj_id)

    def _semantic_role(self, obs: ObjectObservation, goal_manager: GoalManager) -> str:
        if _label_matches(obs.label, goal_manager.target_labels):
            return "object_anchor"
        if _label_matches(obs.label, {x.lower() for x in goal_manager.landmark_prior}):
            return "context_object"
        if is_structural_label(obs.label):
            return "context_object"
        return "environment_object"

    def _transient_anchor_update(
        self,
        topo_map: DynamicTopoMap,
        observations: List[ObjectObservation],
        active_anchor_id: Optional[str],
        position: np.ndarray,
        summary: MemoryUpdateSummary,
    ) -> None:
        if not active_anchor_id:
            summary.debug_only_writes += len(observations)
            return
        node = topo_map.get_node(active_anchor_id)
        if node is None:
            summary.debug_only_writes += len(observations)
            return
        best = max(observations, key=lambda o: _bbox_area(o.bbox), default=None)
        if best is None:
            return
        attrs = node.attributes.setdefault("active_anchor_state", {})
        attrs.update({
            "last_seen_step": topo_map.current_step,
            "bbox_area": _bbox_area(best.bbox),
            "bearing": best.bearing,
            "range_bin": best.range_bin,
            "visibility": best.visibility,
            "visible": best.visible,
            "observer_position": position.tolist(),
        })
        summary.transient_updates += 1


class StructurePlanner:
    def select(self, topo_map: DynamicTopoMap, goal: GoalManager, position: Optional[np.ndarray]) -> StructureTarget:
        labels = {x.lower() for x in goal.room_prior + goal.landmark_prior}
        room_labels = {x.lower() for x in goal.room_prior}
        best: Optional[SemanticNode] = None
        best_score = -1.0
        for node in topo_map.get_nodes_by_type(NodeType.ROOM) + topo_map.get_nodes_by_type(NodeType.LANDMARK) + topo_map.get_nodes_by_type(NodeType.OBJECT):
            role = node.attributes.get("semantic_role")
            label = (node.label or "").lower()
            if node.node_type == NodeType.OBJECT and role not in ("context_object", "environment_object"):
                continue
            if labels and not _label_matches(label, labels):
                continue
            score = float(node.confidence)
            if node.node_type == NodeType.ROOM and _label_matches(label, room_labels):
                score += 2.0
            if role == "context_object":
                score += 0.8
            if position is not None:
                score -= 0.02 * float(np.linalg.norm(node.position - position))
            if score > best_score:
                best = node
                best_score = score
        if best is None:
            return StructureTarget(reason="no_matching_structure")
        return StructureTarget(best.node_id, best.attributes.get("semantic_role", best.node_type.value), best.position.copy(), "semantic_context_match")


class NavigationPlanner:
    def __init__(self, new_config: NewGoatConfig) -> None:
        self.cfg = new_config

    def select(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        structure: StructureTarget,
        nav_phase: NavPhase,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str] = None,
    ) -> Tuple[NavTarget, List[str], List[float]]:
        if nav_phase not in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY, NavPhase.STOP):
            anchor = self._best_object_anchor(topo_map, goal, position)
            if anchor is not None:
                route = self._resolve_object_route(anchor, position, cur_vp_id)
                if route is not None:
                    return route, [anchor.node_id], [1.0]
        frontier = self._best_frontier(topo_map, position, structure, goal)
        if frontier is not None:
            return (
                NavTarget(frontier.position.copy(), frontier.node_id, "frontier", "search_frontier", NavPhase.GLOBAL_SEARCH),
                [frontier.node_id],
                [0.5],
            )
        visited = self._farthest_visited(topo_map, position)
        if visited is not None:
            return (
                NavTarget(visited.position.copy(), visited.node_id, "visited", "visited_fallback", NavPhase.GLOBAL_SEARCH),
                [visited.node_id],
                [0.1],
            )
        return NavTarget(reason="no_navigation_candidates"), [], []

    def _best_object_anchor(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        position: Optional[np.ndarray],
    ) -> Optional[SemanticNode]:
        best = None
        best_score = -1.0
        step = topo_map.current_step
        for node in topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.attributes.get("semantic_role") != "object_anchor":
                continue
            if not _label_matches(node.label, goal.target_labels):
                continue
            if int(node.attributes.get("blacklisted_until", -1)) >= step:
                continue
            conf = float(node.confidence)
            if conf < self.cfg.anchor_min_confidence:
                continue
            detection_step = int(node.attributes.get("goal_detection_step", 0))
            age = step - detection_step
            if age > self.cfg.anchor_stale_age and int(node.attributes.get("failed_approach_count", 0)) > 0:
                continue
            multi_view = int(node.attributes.get("multi_view_count", node.visit_count))
            if multi_view <= 1 and conf < self.cfg.anchor_single_frame_min_confidence:
                continue
            score = conf + float(node.attributes.get("target_relevance", 0.0))
            if node.attributes.get("repeated_goal_source"):
                score += 0.3
            if goal.is_repeated_goal:
                reuse = goal.reuse_debug or {}
                if node.node_id in reuse.get("unfailed_anchors", []):
                    score += 0.6
                if int(node.attributes.get("failed_approach_count", 0)) > 0:
                    score -= 0.5
            if node.attributes.get("cross_goal_preserved"):
                score += 0.2
            if position is not None:
                score -= 0.02 * float(np.linalg.norm(self._anchor_position(node) - position))
            if score > best_score:
                best = node
                best_score = score
        return best

    def _anchor_waypoint_position(self, node: SemanticNode) -> Optional[np.ndarray]:
        pos = node.attributes.get("anchor_waypoint_position")
        if pos is None:
            return None
        return np.array(pos, dtype=np.float32)

    def _best_approach_position(self, node: SemanticNode) -> Optional[np.ndarray]:
        pos = node.attributes.get("best_approach_position")
        if pos is None:
            return None
        return np.array(pos, dtype=np.float32)

    def at_anchor_waypoint(
        self,
        anchor: SemanticNode,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> bool:
        if position is None:
            return False
        wp_pos = self._anchor_waypoint_position(anchor)
        wp_id = anchor.attributes.get("anchor_waypoint_id")
        if cur_vp_id and wp_id and cur_vp_id == wp_id:
            return True
        if wp_pos is not None:
            return float(np.linalg.norm(position - wp_pos)) <= self.cfg.anchor_waypoint_reach_distance
        return False

    def _resolve_object_route(
        self,
        anchor: SemanticNode,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> Optional[NavTarget]:
        """Stage-1/2 coarse routing: waypoint first, then optional approach pose."""
        if self.at_anchor_waypoint(anchor, position, cur_vp_id):
            approach = self._best_approach_position(anchor)
            if approach is not None and position is not None:
                if float(np.linalg.norm(position - approach)) > 0.35:
                    return NavTarget(
                        approach.copy(),
                        anchor.node_id,
                        "object_approach",
                        "goal_approach",
                        NavPhase.LOCAL_VISUAL_APPROACH,
                    )
            return NavTarget(
                reason="at_anchor_waypoint_servo",
                target_node_id=anchor.node_id,
                expected_phase_after_reach=NavPhase.LOCAL_VISUAL_APPROACH,
            )
        wp_pos = self._anchor_waypoint_position(anchor)
        route_pos = wp_pos if wp_pos is not None else self._anchor_position(anchor)
        return NavTarget(
            route_pos.copy(),
            anchor.node_id,
            "object_anchor",
            "goal_anchor",
            NavPhase.LOCAL_VISUAL_APPROACH,
        )

    def _anchor_position(self, node: SemanticNode) -> np.ndarray:
        pos = node.attributes.get("anchor_waypoint_position") or node.attributes.get("best_approach_position")
        if pos is None:
            return node.position.copy()
        return np.array(pos, dtype=np.float32)

    def _best_frontier(
        self,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray],
        structure: StructureTarget,
        goal: Optional[GoalManager] = None,
    ) -> Optional[SemanticNode]:
        if position is None:
            return None
        context_nodes = self._context_object_positions(topo_map, goal) if goal else []
        room_prior_positions = self._room_prior_positions(topo_map, goal) if goal else []
        candidates: List[Tuple[float, float, SemanticNode]] = []
        for node in topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER):
            if node.attributes.get("consumed") or int(node.attributes.get("blacklisted_until", -1)) >= topo_map.current_step:
                continue
            dist = float(np.linalg.norm(node.position - position))
            if dist < self.cfg.min_frontier_distance:
                continue
            sem_score = 0.0
            if structure.position is not None:
                struct_dist = float(np.linalg.norm(node.position - structure.position))
                sem_score += max(0.0, 1.0 - struct_dist / 10.0)
            for ctx_pos in context_nodes:
                ctx_dist = float(np.linalg.norm(node.position - ctx_pos))
                if ctx_dist < 8.0:
                    sem_score += 2.5 * max(0.0, 1.0 - ctx_dist / 8.0)
            for rp_pos in room_prior_positions:
                rp_dist = float(np.linalg.norm(node.position - rp_pos))
                if rp_dist < 12.0:
                    sem_score += 3.5 * max(0.0, 1.0 - rp_dist / 12.0)
            if goal is not None and not self._goal_has_any_anchor(topo_map, goal):
                if structure.position is not None:
                    struct_dist = float(np.linalg.norm(node.position - structure.position))
                    sem_score += 2.0 * max(0.0, 1.0 - struct_dist / 8.0)
            candidates.append((dist, sem_score, node))
        if not candidates:
            return None
        max_dist = max(d for d, _, _ in candidates)
        best_node = None
        best_score = -1.0
        for dist, sem, node in candidates:
            geo_score = dist / max(max_dist, 1.0)
            total = geo_score + sem
            if total > best_score:
                best_score = total
                best_node = node
        return best_node

    def _goal_has_any_anchor(self, topo_map: DynamicTopoMap, goal: GoalManager) -> bool:
        step = topo_map.current_step
        for node in topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.attributes.get("semantic_role") != "object_anchor":
                continue
            if not _label_matches(node.label, goal.target_labels):
                continue
            if int(node.attributes.get("blacklisted_until", -1)) >= step:
                continue
            return True
        return False

    def _context_object_positions(self, topo_map: DynamicTopoMap, goal: Optional[GoalManager]) -> List[np.ndarray]:
        if goal is None:
            return []
        ctx_labels = {x.lower() for x in goal.landmark_prior}
        result = []
        for node in topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.attributes.get("semantic_role") in ("context_object", "environment_object"):
                if _label_matches(node.label, ctx_labels):
                    result.append(node.position)
        return result

    def _room_prior_positions(self, topo_map: DynamicTopoMap, goal: Optional[GoalManager]) -> List[np.ndarray]:
        if goal is None:
            return []
        rp_labels = {x.lower() for x in goal.room_prior}
        result = []
        for node in topo_map.get_nodes_by_type(NodeType.ROOM):
            if _label_matches(node.label, rp_labels):
                result.append(node.position)
        return result

    def _farthest_visited(
        self,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray],
    ) -> Optional[SemanticNode]:
        """Pick the farthest visited waypoint as exploration fallback.

        Avoids the deadlock where the nearest visited is the current position.
        """
        if position is None:
            return None
        best = None
        best_dist = -1.0
        for node in topo_map.get_nodes_by_type(NodeType.WAYPOINT_VISITED):
            dist = float(np.linalg.norm(node.position - position))
            if dist < 0.5:
                continue
            if dist > best_dist:
                best = node
                best_dist = dist
        return best


class LocalVisualServo:
    def __init__(self, new_config: NewGoatConfig, state: ServoState) -> None:
        self.cfg = new_config
        self.state = state

    def reset(self) -> None:
        self.state.reset()

    def enter(self, anchor_id: Optional[str], step: int) -> None:
        if anchor_id != self.state.active_anchor_id:
            self.state.reset()
            self.state.active_anchor_id = anchor_id
            self.state.entry_step = step
        elif self.state.entry_step <= 0:
            self.state.entry_step = step

    def update_evidence(
        self,
        packet: PerceptionPacket,
        goal: GoalManager,
        position: Optional[np.ndarray],
        anchor_distance: Optional[float] = None,
        instance_distance: Optional[float] = None,
    ) -> Dict[str, Any]:
        visible_obs = self._goal_observations(packet.report.objects, goal)
        best = max(visible_obs, key=lambda o: _bbox_area(o.bbox), default=None)
        goal_visible = packet.report.goal_visible or best is not None
        raw_bbox = _bbox_area(best.bbox) if best is not None else 0.0
        bearing = best.bearing if best is not None else "unknown"
        range_bin = best.range_bin if best is not None else "unknown"

        if packet.fresh_vlm and raw_bbox > 0.0:
            self.state.last_fresh_bbox = raw_bbox
        if packet.cached_vlm:
            effective_bbox = raw_bbox if raw_bbox > 0.0 else self.state.last_fresh_bbox
        else:
            effective_bbox = raw_bbox

        confirm = bool(goal_visible and effective_bbox >= self.cfg.bbox_min_approach)
        aligned = bearing in _CENTER_BEARINGS
        stop_pose = bool(confirm and effective_bbox >= self.cfg.bbox_min_stop and aligned)
        if packet.fresh_vlm:
            self._append(self.state.confirm_buffer, confirm)
            self._append(self.state.stop_buffer, stop_pose)
            if stop_pose:
                self.state.fresh_vlm_stop_step = packet.step_id
        elif packet.cached_vlm and raw_bbox >= self.cfg.bbox_min_approach:
            self._append(self.state.confirm_buffer, confirm)
        if confirm:
            self.state.lost_count = 0
            if effective_bbox >= self.state.best_bbox_area:
                self.state.best_bbox_area = effective_bbox
                self.state.best_stop_pose = None if position is None else position.copy()
                self.state.peak_bearing = bearing
                self.state.aligned_at_peak = aligned
                if packet.fresh_vlm or effective_bbox >= self.cfg.bbox_min_stop:
                    self.state.visual_advance_steps += 1
        elif (
            packet.fresh_vlm
            and effective_bbox < self.cfg.bbox_min_approach
            and self.state.best_bbox_area < self.cfg.bbox_stop_min
        ):
            self.state.lost_count += 1
        self.state.bbox_history.append(effective_bbox)
        if position is not None:
            if self.state.last_approach_position is not None:
                step_dist = float(np.linalg.norm(position - self.state.last_approach_position))
                if step_dist > 0.05:
                    self.state.approach_travel_distance += step_dist
            self.state.last_approach_position = position.copy()
        if anchor_distance is not None:
            self.state.last_anchor_distance = anchor_distance
            if anchor_distance < self.state.best_anchor_distance:
                self.state.best_anchor_distance = anchor_distance
            self.state.anchor_distance_history.append(anchor_distance)
            if len(self.state.anchor_distance_history) > 5:
                self.state.anchor_distance_history = self.state.anchor_distance_history[-5:]
        if instance_distance is not None and np.isfinite(instance_distance):
            if instance_distance < self.state.best_instance_distance:
                self.state.best_instance_distance = instance_distance
            self.state.instance_distance_history.append(float(instance_distance))
            if len(self.state.instance_distance_history) > 5:
                self.state.instance_distance_history = self.state.instance_distance_history[-5:]
        retreating = self._is_retreating(anchor_distance, instance_distance)
        plateau = self._bbox_plateau()
        return {
            "goal_visible": goal_visible,
            "bbox_area": raw_bbox,
            "effective_bbox_area": effective_bbox,
            "best_bbox_area": self.state.best_bbox_area,
            "bearing": bearing,
            "range_bin": range_bin,
            "forward_count": self.state.forward_action_count,
            "approach_distance": self.state.approach_travel_distance,
            "bbox_plateau": plateau,
            "aligned_at_peak": self.state.aligned_at_peak,
            "anchor_distance": anchor_distance,
            "best_anchor_distance": (
                None if self.state.best_anchor_distance >= float("inf")
                else self.state.best_anchor_distance
            ),
            "retreating": retreating,
            "instance_distance": instance_distance,
            "best_instance_distance": (
                None if self.state.best_instance_distance >= float("inf")
                else self.state.best_instance_distance
            ),
        }

    def _is_retreating(
        self,
        anchor_distance: Optional[float],
        instance_distance: Optional[float] = None,
    ) -> bool:
        """True when agent is monotonically moving away from the target."""
        if instance_distance is not None and np.isfinite(instance_distance):
            hist = self.state.instance_distance_history
            if len(hist) >= 3:
                recent = hist[-3:]
                if all(recent[i + 1] - recent[i] > 0.05 for i in range(len(recent) - 1)):
                    return (recent[-1] - recent[0]) >= self.cfg.stop_retreat_margin
            return False
        if anchor_distance is None:
            return False
        if anchor_distance <= self.cfg.anchor_at_zone_distance:
            return False
        hist = self.state.anchor_distance_history
        if len(hist) < 3:
            return False
        recent = hist[-3:]
        if not all(recent[i + 1] - recent[i] > 0.04 for i in range(len(recent) - 1)):
            return False
        return (recent[-1] - recent[0]) >= self.cfg.stop_retreat_margin

    def _bbox_plateau(self) -> bool:
        """True when bbox growth has stalled over the recent window."""
        w = self.cfg.bbox_plateau_window
        hist = self.state.bbox_history
        if len(hist) < w + 1:
            return False
        recent = hist[-w:]
        older = hist[-(w + 1)]
        if older <= 0:
            return False
        growth = (max(recent) - older) / older
        return growth < self.cfg.bbox_plateau_growth

    def act(self, evidence: Dict[str, Any], step: int) -> ServoAction:
        if step - self.state.entry_step > self.cfg.local_approach_timeout_steps:
            return ServoAction("fail_anchor", "local_approach_timeout", NavPhase.RECOVERY)
        lost_limit = (
            self.cfg.servo_lost_count_max
            if self.state.best_bbox_area >= self.cfg.bbox_stop_min
            else max(3, self.cfg.servo_lost_count_max - 2)
        )
        if self.state.lost_count >= lost_limit:
            return ServoAction("fail_anchor", "target_lost", NavPhase.RECOVERY)
        bearing = evidence.get("bearing", "unknown")
        if bearing in ("left", "left_front", "front-left"):
            return ServoAction("turn_left", "servo_align_left", NavPhase.LOCAL_VISUAL_APPROACH)
        if bearing in ("right", "right_front", "front-right"):
            return ServoAction("turn_right", "servo_align_right", NavPhase.LOCAL_VISUAL_APPROACH)
        bbox_area = float(evidence.get("effective_bbox_area", evidence.get("bbox_area", 0.0)))
        peak_target = max(self.cfg.bbox_min_stop, self.state.best_bbox_area * self.cfg.bbox_peak_ratio)
        below_peak = bbox_area < peak_target
        not_enough_forward = self.state.forward_action_count < self.cfg.min_forward_before_stop
        not_enough_distance = self.state.approach_travel_distance < self.cfg.min_approach_distance
        at_stop_band = self.state.best_bbox_area >= self.cfg.bbox_stop_min
        if below_peak or not_enough_forward or not_enough_distance:
            if not at_stop_band:
                self.state.plateau_forward_count = 0
            self.state.forward_action_count += 1
            return ServoAction("move_forward", "servo_advance", NavPhase.LOCAL_VISUAL_APPROACH)
        plateau = bool(evidence.get("bbox_plateau", False))
        if not plateau:
            if not at_stop_band:
                self.state.plateau_forward_count = 0
            self.state.forward_action_count += 1
            if at_stop_band:
                self.state.plateau_forward_count += 1
            return ServoAction("move_forward", "servo_advance_to_plateau", NavPhase.LOCAL_VISUAL_APPROACH)
        if self.state.best_bbox_area < self.cfg.bbox_stop_min:
            self.state.plateau_forward_count += 1
            self.state.forward_action_count += 1
            return ServoAction("move_forward", "servo_plateau_below_stop_bbox", NavPhase.LOCAL_VISUAL_APPROACH)
        if self.state.plateau_forward_count < self.cfg.min_forward_after_plateau:
            self.state.plateau_forward_count += 1
            self.state.forward_action_count += 1
            return ServoAction("move_forward", "servo_plateau_creep", NavPhase.LOCAL_VISUAL_APPROACH)
        return ServoAction("hold", "ready_for_stop_verify", NavPhase.STOP_VERIFY)

    def _goal_observations(self, objects: List[ObjectObservation], goal: GoalManager) -> List[ObjectObservation]:
        return [o for o in objects if _label_matches(o.label, goal.target_labels)]

    def _append(self, buf: List[bool], value: bool) -> None:
        buf.append(bool(value))
        if len(buf) > self.cfg.stop_buffer_size:
            del buf[:-self.cfg.stop_buffer_size]


class StopVerifier:
    def __init__(self, new_config: NewGoatConfig, servo_state: ServoState) -> None:
        self.cfg = new_config
        self.state = servo_state

    def can_stop(
        self,
        packet: PerceptionPacket,
        goal: GoalManager,
        position: Optional[np.ndarray],
        servo_evidence: Optional[Dict[str, Any]] = None,
    ) -> StopDecision:
        layer1 = sum(self.state.confirm_buffer[-self.cfg.stop_buffer_size:]) >= self.cfg.servo_entry_evidence
        approach_enough = (
            self.state.forward_action_count >= self.cfg.min_forward_before_stop
            and self.state.approach_travel_distance >= self.cfg.min_approach_distance
        )
        range_bin = str((servo_evidence or {}).get("range_bin", "unknown")).lower()
        bbox_close_enough = self.state.best_bbox_area >= self.cfg.bbox_stop_min
        bbox_growth_ok = bbox_close_enough
        plateau_ok = bool((servo_evidence or {}).get("bbox_plateau", False))
        retreating = bool((servo_evidence or {}).get("retreating", False))
        creep_done = (
            not retreating
            and self.state.plateau_forward_count >= self.cfg.min_forward_after_plateau
        )
        layer2 = bbox_growth_ok and approach_enough and creep_done and (
            self.state.visual_advance_steps >= self.cfg.servo_entry_evidence or plateau_ok
        )
        stop_votes = sum(self.state.stop_buffer[-self.cfg.stop_buffer_size:])
        multi_angle_ok = (
            self.state.aligned_at_peak
            or stop_votes >= self.cfg.servo_entry_evidence
            or (range_bin == "near" and bbox_close_enough and plateau_ok)
        )
        near_peak = True
        if self.state.best_stop_pose is not None and position is not None:
            dist = float(np.linalg.norm(position - self.state.best_stop_pose))
            near_peak = (
                dist <= self.cfg.stop_pose_radius
                or self.state.best_bbox_area >= self.cfg.bbox_min_stop * 4.0
            )
        goal_obs = [
            o for o in packet.report.objects
            if _label_matches(o.label, goal.target_labels)
        ]
        fresh_raw_bbox = max((_bbox_area(o.bbox) for o in goal_obs), default=0.0)
        fresh_stop_ok = packet.fresh_vlm and fresh_raw_bbox >= self.cfg.bbox_min_stop
        instance_distance = (servo_evidence or {}).get("instance_distance")
        instance_close = (
            instance_distance is not None
            and np.isfinite(instance_distance)
            and float(instance_distance) <= self.cfg.success_stop_distance
        )
        layer3 = multi_angle_ok and near_peak and fresh_stop_ok and stop_votes >= self.cfg.servo_entry_evidence
        should = bool(layer1 and layer2 and layer3 and instance_close and not retreating)
        reason = "instance_confirmed_stop" if should else self._reason(
            layer1, layer2, layer3, multi_angle_ok, fresh_stop_ok, near_peak, retreating, instance_close,
        )
        return StopDecision(should, layer1, layer2, layer3, reason)

    def _reason(
        self,
        layer1: bool,
        layer2: bool,
        layer3: bool,
        multi_angle_ok: bool,
        fresh_stop_ok: bool,
        near_peak: bool,
        retreating: bool,
        instance_close: bool,
    ) -> str:
        if retreating:
            return "layer2_retreating_from_target"
        if not layer1:
            return "layer1_target_not_confirmed"
        if not layer2:
            return "layer2_approach_incomplete"
        if not multi_angle_ok:
            return "layer3_multi_angle_not_verified"
        if not fresh_stop_ok:
            return "layer3_fresh_vlm_not_confirmed"
        if not near_peak:
            return "layer3_stop_pose_not_ok"
        if not instance_close:
            return "layer4_instance_not_close_enough"
        if not layer3:
            return "layer3_stop_buffer_insufficient"
        return "not_ready"


class RecoveryManager:
    def __init__(self, new_config: NewGoatConfig) -> None:
        self.cfg = new_config
        self.recovery_reason: str = ""
        self.recent_positions: List[np.ndarray] = []

    def reset(self) -> None:
        self.recovery_reason = ""
        self.recent_positions = []

    def note_position(self, position: Optional[np.ndarray]) -> bool:
        if position is None:
            return False
        self.recent_positions.append(position.copy())
        if len(self.recent_positions) > self.cfg.no_progress_window:
            self.recent_positions = self.recent_positions[-self.cfg.no_progress_window:]
        if len(self.recent_positions) < self.cfg.no_progress_window:
            return False
        moved = float(np.linalg.norm(self.recent_positions[-1] - self.recent_positions[0]))
        if moved < self.cfg.no_progress_distance:
            self.recovery_reason = "no_progress"
            return True
        return False

    def fail_anchor(self, topo_map: DynamicTopoMap, anchor_id: Optional[str], reason: str) -> None:
        self.recovery_reason = reason
        if not anchor_id:
            return
        node = topo_map.get_node(anchor_id)
        if node is None:
            return
        node.attributes["failed_approach_count"] = int(node.attributes.get("failed_approach_count", 0)) + 1
        node.attributes["blacklisted_until"] = topo_map.current_step + self.cfg.anchor_blacklist_steps
        node.attributes["last_failure_reason"] = reason

    def on_navigation_event(self, topo_map: DynamicTopoMap, target_node_id: Optional[str], event: str) -> Dict[str, Any]:
        out = {"target_node_id": target_node_id, "event": event, "action": "ignored"}
        if target_node_id is None:
            return out
        node = topo_map.get_node(target_node_id)
        if node is None:
            return out
        if event == "collision_blocked":
            node.attributes["blacklisted_until"] = topo_map.current_step + self.cfg.anchor_blacklist_steps
            self.recovery_reason = "collision_blocked"
            out["action"] = "blacklisted_target"
        elif event == "target_reached":
            if node.node_type == NodeType.WAYPOINT_FRONTIER:
                node.attributes["consumed"] = True
                out["action"] = "consumed_frontier"
            elif node.attributes.get("semantic_role") == "object_anchor":
                out["action"] = "object_anchor_reached"
            else:
                out["action"] = "target_reached"
        return out


class DebugTracer:
    def build(
        self,
        *,
        nav_phase: NavPhase,
        phase_transition_reason: str,
        packet: PerceptionPacket,
        memory: MemoryUpdateSummary,
        structure: StructureTarget,
        nav_target: NavTarget,
        servo_state: ServoState,
        servo_evidence: Dict[str, Any],
        stop: StopDecision,
        recovery_reason: str,
        goal_debug: Dict[str, Any],
        position: Optional[np.ndarray],
        success_stop_distance: float = 1.0,
    ) -> Dict[str, Any]:
        return {
            "nav_phase": nav_phase.value,
            "phase_transition_reason": phase_transition_reason,
            "perception_source": packet.source,
            "fresh_vlm": packet.fresh_vlm,
            "cached_vlm": packet.cached_vlm,
            "vlm_mode": packet.vlm_mode,
            "room_label": packet.report.room_label,
            "selected_structure_target": structure.to_debug(),
            "selected_nav_target": nav_target.to_debug(position),
            "target_distance": nav_target.distance_from(position),
            "active_anchor_id": servo_state.active_anchor_id,
            "anchor_status": {
                "lost_count": servo_state.lost_count,
                "confirm_buffer": list(servo_state.confirm_buffer),
                "stop_buffer": list(servo_state.stop_buffer),
            },
            "bbox_area": servo_evidence.get("bbox_area", 0.0),
            "effective_bbox_area": servo_evidence.get("effective_bbox_area", servo_evidence.get("bbox_area", 0.0)),
            "best_bbox_area": servo_state.best_bbox_area,
            "visual_advance_steps": servo_state.visual_advance_steps,
            "forward_action_count": servo_state.forward_action_count,
            "plateau_forward_count": servo_state.plateau_forward_count,
            "anchor_distance": servo_evidence.get("anchor_distance"),
            "best_anchor_distance": servo_evidence.get("best_anchor_distance"),
            "instance_distance": servo_evidence.get("instance_distance"),
            "best_instance_distance": servo_evidence.get("best_instance_distance"),
            "success_stop_distance": success_stop_distance,
            "retreating": servo_evidence.get("retreating", False),
            "approach_travel_distance": round(servo_state.approach_travel_distance, 4),
            "bbox_plateau": servo_evidence.get("bbox_plateau", False),
            "aligned_at_peak": servo_state.aligned_at_peak,
            "peak_bearing": servo_state.peak_bearing,
            "bearing": servo_evidence.get("bearing", "unknown"),
            "range_bin": servo_evidence.get("range_bin", "unknown"),
            "layer1": stop.layer1,
            "layer2": stop.layer2,
            "layer3": stop.layer3,
            "fresh_vlm": packet.fresh_vlm,
            "stop_reason": stop.reason,
            "recovery_reason": recovery_reason,
            "memory_write_count": memory.memory_write_count,
            "object_merge_count_this_step": memory.object_merge_count_this_step,
            "persistent_writes": memory.persistent_writes,
            "transient_updates": memory.transient_updates,
            "debug_only_writes": memory.debug_only_writes,
            "goal_reuse": goal_debug,
        }


class GoatAgent(ConfTopoBaseAgent):
    def __init__(self, config: Optional[ConfTopoConfig] = None, new_config: Optional[NewGoatConfig] = None):
        super().__init__(config)
        self.new_config = new_config or NewGoatConfig()
        self.goal_manager = GoalManager()
        self.perception_manager = PerceptionManager(self.config, self.new_config)
        self.memory_writer = MemoryWriter(self.new_config)
        self.structure_planner = StructurePlanner()
        self.navigation_planner = NavigationPlanner(self.new_config)
        self.servo_state = ServoState()
        self.local_servo = LocalVisualServo(self.new_config, self.servo_state)
        self.stop_verifier = StopVerifier(self.new_config, self.servo_state)
        self.recovery_manager = RecoveryManager(self.new_config)
        self.debug_tracer = DebugTracer()
        self.nav_phase = NavPhase.GLOBAL_SEARCH
        self.phase_enter_step = 0
        self.phase_transition_reason = "init"
        self._cur_rgb: Any = None
        self._cur_rgb_embed: Optional[np.ndarray] = None
        self._position: Optional[np.ndarray] = None
        self._origin_position: Optional[np.ndarray] = None
        self._heading: float = 0.0
        self._last_packet = PerceptionPacket(PerceptionReport(), "none")
        self._last_memory = MemoryUpdateSummary()
        self._last_structure = StructureTarget()
        self._last_nav_target = NavTarget()
        self._last_servo_evidence: Dict[str, Any] = {}
        self._last_stop = StopDecision(False, False, False, False, "not_evaluated")
        self._last_debug: Dict[str, Any] = {}
        self._last_navigation_event: Dict[str, Any] = {}
        self._heavy_perception_calls: int = 0
        self._object_merge_count: int = 0
        self._goals_completed: int = 0
        self._consumed_frontier_ids: Set[str] = set()
        self._visited_fallback_count: int = 0
        self._no_candidates_count: int = 0
        self._instance_distance: Optional[float] = None
        self._goal_enter_step: int = -1
        if self.config.perception.backend == "vlm":
            self._init_vlm_perceiver()

    def _init_vlm_perceiver(self) -> None:
        from conftopo.perception.vlm_backend import Qwen3VLBackend
        pcfg = self.config.perception
        backend = Qwen3VLBackend(
            api_base=pcfg.vlm_api_base,
            model=pcfg.vlm_model,
            timeout=pcfg.vlm_timeout,
        )
        self.perception_manager.set_vlm_perceiver(VLMPerceiver(backend))

    @property
    def perceiver(self) -> LightPerceiver:
        return self.perception_manager.light

    def reset(self):
        super().reset()
        self.goal_manager = GoalManager()
        self.perception_manager.reset_short_term()
        self.memory_writer.reset_episode()
        self.local_servo.reset()
        self.recovery_manager.reset()
        self.nav_phase = NavPhase.GLOBAL_SEARCH
        self.phase_enter_step = 0
        self.phase_transition_reason = "reset"
        self._origin_position = None
        self._heavy_perception_calls = 0
        self._object_merge_count = 0
        self._goals_completed = 0
        self._consumed_frontier_ids = set()
        self._visited_fallback_count = 0
        self._no_candidates_count = 0
        self._instance_distance = None
        self._goal_enter_step = -1

    def _target_output_for_node(self, node: SemanticNode) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resolve executor target for a semantic node (waypoint-first object routing)."""
        target_pos = node.position.copy()
        extras: Dict[str, Any] = {}
        if (
            node.node_type == NodeType.OBJECT
            and node.attributes.get("semantic_role") == "object_anchor"
        ):
            if self.navigation_planner.at_anchor_waypoint(
                node, self._position, self.memory_writer.cur_vp_id,
            ):
                approach = self.navigation_planner._best_approach_position(node)
                if approach is not None:
                    target_pos = approach.copy()
                    extras["target_anchor_type"] = "object_approach"
                    extras["semantic_target_node_id"] = node.node_id
                else:
                    wp_pos = self.navigation_planner._anchor_waypoint_position(node)
                    if wp_pos is not None:
                        target_pos = wp_pos.copy()
                    extras["target_anchor_type"] = "object_anchor"
                    extras["semantic_target_node_id"] = node.node_id
            else:
                wp_pos = self.navigation_planner._anchor_waypoint_position(node)
                route_pos = wp_pos if wp_pos is not None else self.navigation_planner._anchor_position(node)
                target_pos = route_pos.copy()
                extras.update({
                    "target_anchor_type": "object_anchor",
                    "semantic_target_node_id": node.node_id,
                    "anchor_waypoint_id": node.attributes.get("anchor_waypoint_id"),
                    "anchor_waypoint_position": node.attributes.get("anchor_waypoint_position"),
                })
        return target_pos, extras

    def set_vlm_perceiver(self, perceiver: Optional[VLMPerceiver]) -> None:
        self.perception_manager.set_vlm_perceiver(perceiver)

    def set_new_goal(self, goal: GoalNode):
        old_map_nodes = self.topo_map.num_nodes
        self._mark_cross_goal_preserved()
        self.reset_keep_memory()
        if not getattr(goal, "room_prior", None):
            rp = _category_priors(goal.target_object, OBJECT_CATEGORY_ROOM_PRIORS)
            if rp:
                goal.room_prior = rp
        if not getattr(goal, "landmarks", None):
            lp = _category_priors(goal.target_object, OBJECT_CATEGORY_LANDMARK_PRIORS)
            if lp:
                goal.landmarks = lp
        if self.instruction_graph is None:
            self.instruction_graph = InstructionGraph(goal_type="object_goal", goal_nodes=[goal])
        else:
            self.instruction_graph.set_current_goal(goal)
        self.goal_manager.set_new_goal(goal, self.topo_map)
        self.perception_manager.set_goal(goal)
        self.perception_manager.reset_short_term()
        self.memory_writer.reset_goal()
        self.local_servo.reset()
        self.recovery_manager.reset()
        self.nav_phase = NavPhase.GLOBAL_SEARCH
        self.phase_enter_step = self.topo_map.current_step
        self._goal_enter_step = self.topo_map.current_step
        self.phase_transition_reason = f"new_goal_keep_memory:{old_map_nodes}"
        self._goals_completed += 1

    def _mark_cross_goal_preserved(self) -> None:
        """Protect existing object_anchor, room, and context nodes from pruning across goals."""
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            role = node.attributes.get("semantic_role")
            if role in ("object_anchor", "context_object"):
                node.attributes["cross_goal_preserved"] = True
            elif float(node.attributes.get("target_relevance", 0.0)) > 0:
                node.attributes["cross_goal_preserved"] = True
        for node in self.topo_map.get_nodes_by_type(NodeType.ROOM):
            node.attributes["cross_goal_preserved"] = True
        for node in self.topo_map.get_nodes_by_type(NodeType.LANDMARK):
            node.attributes["cross_goal_preserved"] = True

    def observe(self, obs: Dict[str, Any]) -> None:
        self._cur_rgb = obs.get("rgb")
        self._cur_rgb_embed = None
        if obs.get("rgb_embed") is not None:
            self._cur_rgb_embed = np.asarray(obs["rgb_embed"], dtype=np.float32)
        pos = obs.get("position")
        if pos is not None:
            world = np.asarray(pos, dtype=np.float32)
            if self._origin_position is None:
                self._origin_position = world.copy()
            self._position = world - self._origin_position
        self._heading = float(obs.get("heading", 0.0))
        inst = obs.get("instance_distance")
        if inst is not None and np.isfinite(inst):
            self._instance_distance = float(inst)
        else:
            self._instance_distance = None
        self._last_packet = self.perception_manager.run_light(self._cur_rgb_embed, self.topo_map.current_step)

    def update_memory(self) -> None:
        if self.recovery_manager.note_position(self._position):
            if self.nav_phase not in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY):
                self._transition(NavPhase.RECOVERY, "no_progress")
        packet = self.perception_manager.run(
            rgb=self._cur_rgb,
            visual_embed=self._cur_rgb_embed,
            position=self._position,
            heading=self._heading,
            topo_map=self.topo_map,
            goal_manager=self.goal_manager,
            nav_phase=self.nav_phase,
            step_id=self.topo_map.current_step,
            has_near_goal_object=self._has_near_goal_anchor(),
        )
        self._last_packet = packet
        if packet.fresh_vlm:
            self._heavy_perception_calls += 1
        self._last_memory = self.memory_writer.update(
            self.topo_map,
            packet,
            self.nav_phase,
            self._position,
            self._heading,
            self._cur_rgb_embed,
            self.goal_manager,
            self.servo_state.active_anchor_id,
        )
        self._object_merge_count += self._last_memory.object_merge_count_this_step
        self.topo_map.decay_all_confidences()
        self.topo_map.merge_nearby_nodes(NodeType.WAYPOINT_FRONTIER)
        self.topo_map.adaptive_granularity(self._position)
        self.topo_map.prune_low_confidence(self._position)

    def plan(self) -> PlanDecision:
        structure = self.structure_planner.select(self.topo_map, self.goal_manager, self._position)
        nav_target, candidate_ids, scores = self.navigation_planner.select(
            self.topo_map,
            self.goal_manager,
            structure,
            self.nav_phase,
            self._position,
            cur_vp_id=self.memory_writer.cur_vp_id,
        )
        self._last_structure = structure
        self._last_nav_target = nav_target
        if nav_target.reason == "visited_fallback":
            self._visited_fallback_count += 1
        elif nav_target.reason == "no_navigation_candidates":
            self._no_candidates_count += 1

        self._maybe_enter_servo_near_anchor(nav_target)

        if nav_target.reason == "at_anchor_waypoint_servo":
            anchor_id = nav_target.target_node_id
            if anchor_id and self.servo_state.active_anchor_id != anchor_id:
                self.local_servo.enter(anchor_id, self.topo_map.current_step)
            if self.nav_phase not in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY, NavPhase.STOP):
                self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "at_anchor_waypoint_servo")
            servo_decision = self._plan_servo_phase(nav_target, candidate_ids, scores)
            if servo_decision is not None:
                return servo_decision

        phase_age = self.topo_map.current_step - self.phase_enter_step
        if phase_age > self.new_config.phase_timeout_steps:
            if self.nav_phase == NavPhase.ROUTE_TO_STRUCTURE:
                if structure.node_id:
                    sn = self.topo_map.get_node(structure.node_id)
                    if sn is not None:
                        sn.attributes["blacklisted_until"] = self.topo_map.current_step + self.new_config.anchor_blacklist_steps
                self._transition(NavPhase.GLOBAL_SEARCH, "route_to_structure_timeout")
            elif self.nav_phase == NavPhase.ROUTE_TO_OBJECT_ANCHOR:
                self.recovery_manager.fail_anchor(self.topo_map, self.servo_state.active_anchor_id, "route_timeout")
                self._transition(NavPhase.RECOVERY, "route_to_anchor_timeout")
            elif self.nav_phase == NavPhase.STOP_VERIFY:
                self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "stop_verify_timeout")

        if self.nav_phase in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY):
            servo_decision = self._plan_servo_phase(nav_target, candidate_ids, scores)
            if servo_decision is not None:
                return servo_decision

        if self.nav_phase == NavPhase.RECOVERY:
            self._transition(NavPhase.GLOBAL_SEARCH, self.recovery_manager.recovery_reason or "recovery_complete")

        if nav_target.target_node_id and nav_target.target_type == "object_anchor":
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "selected_object_anchor")
        elif nav_target.target_node_id and nav_target.target_type == "object_approach":
            self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "selected_object_approach")
        elif structure.node_id:
            self._transition(NavPhase.ROUTE_TO_STRUCTURE, "selected_structure")
        else:
            self._transition(NavPhase.GLOBAL_SEARCH, "search_frontier")

        return self._decision(
            "navigate",
            "navigate",
            nav_target.target_position,
            nav_target.target_node_id,
            nav_target.target_type,
            nav_target.reason,
            candidate_ids,
            scores,
            nav_target.expected_phase_after_reach,
        )

    def _plan_servo_phase(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> Optional[PlanDecision]:
        servo_evidence = self.local_servo.update_evidence(
            self._last_packet,
            self.goal_manager,
            self._position,
            self._active_anchor_distance(),
            self._instance_distance,
        )
        self._last_servo_evidence = servo_evidence
        stop = self.stop_verifier.can_stop(
            self._last_packet, self.goal_manager, self._position, servo_evidence,
        )
        self._last_stop = stop

        if stop.should_stop and self.nav_phase in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY):
            self._transition(NavPhase.STOP, "stop_verified")
            return self._decision("stop", "stop", None, None, "stop", stop.reason, candidate_ids, scores)

        servo = self.local_servo.act(servo_evidence, self.topo_map.current_step)
        if servo.action == "fail_anchor":
            self.recovery_manager.fail_anchor(self.topo_map, self.servo_state.active_anchor_id, servo.reason)
            self._transition(NavPhase.GLOBAL_SEARCH, servo.reason)
            nav_target, candidate_ids, scores = self.navigation_planner.select(
                self.topo_map,
                self.goal_manager,
                self._last_structure,
                self.nav_phase,
                self._position,
                cur_vp_id=self.memory_writer.cur_vp_id,
            )
            self._last_nav_target = nav_target
            if nav_target.target_node_id and nav_target.target_type == "object_anchor":
                self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "recovery_reroute_anchor")
            elif self._last_structure.node_id:
                self._transition(NavPhase.ROUTE_TO_STRUCTURE, "recovery_reroute_structure")
            else:
                self._transition(NavPhase.GLOBAL_SEARCH, "recovery_reroute_search")
            return self._decision(
                "navigate", "navigate",
                nav_target.target_position, nav_target.target_node_id,
                nav_target.target_type, nav_target.reason,
                candidate_ids, scores, nav_target.expected_phase_after_reach,
            )
        if servo.next_phase != self.nav_phase:
            self._transition(servo.next_phase, servo.reason)
            if self.nav_phase == NavPhase.STOP_VERIFY and stop.should_stop:
                self._transition(NavPhase.STOP, "stop_verified")
                return self._decision("stop", "stop", None, None, "stop", stop.reason, candidate_ids, scores)
        if servo.action in ("turn_left", "turn_right", "move_forward"):
            return self._decision(servo.action, "approach_confirm", None, None, "servo", servo.reason, candidate_ids, scores)
        if servo.action == "hold":
            return self._decision("move_forward", "approach_confirm", None, None, "servo_hold_creep", servo.reason, candidate_ids, scores)
        return self._decision("move_forward", "approach_confirm", None, None, "servo_guard", "servo_phase_guard", candidate_ids, scores)

    def _maybe_enter_servo_near_anchor(self, nav_target: NavTarget) -> None:
        if self.nav_phase in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY, NavPhase.STOP):
            return
        if self._position is None:
            return
        if self.topo_map.current_step <= self._goal_enter_step:
            return

        anchor: Optional[SemanticNode] = None
        if nav_target.target_type == "object_anchor" and nav_target.target_node_id:
            anchor = self.topo_map.get_node(nav_target.target_node_id)
        if anchor is None:
            anchor = self.navigation_planner._best_object_anchor(
                self.topo_map, self.goal_manager, self._position,
            )
        if anchor is None:
            return

        at_waypoint = self.navigation_planner.at_anchor_waypoint(
            anchor, self._position, self.memory_writer.cur_vp_id,
        )
        anchor_pos = self.navigation_planner._anchor_position(anchor)
        dist = float(np.linalg.norm(anchor_pos - self._position))
        if not at_waypoint and dist > self.new_config.anchor_servo_enter_distance:
            return
        if self.goal_manager.is_repeated_goal and not at_waypoint:
            return
        if self.goal_manager.is_repeated_goal and not self._last_packet.fresh_vlm:
            return

        goal_visible = self._last_packet.report.goal_visible
        if not goal_visible:
            goal_visible = any(
                _label_matches(obs.label, self.goal_manager.target_labels)
                for obs in self._last_packet.report.objects
            )
        if not goal_visible and self.servo_state.best_bbox_area < self.new_config.bbox_min_approach:
            return
        best_obs = max(
            (o for o in self._last_packet.report.objects if _label_matches(o.label, self.goal_manager.target_labels)),
            key=lambda o: _bbox_area(o.bbox),
            default=None,
        )
        if best_obs is not None:
            raw_bbox = _bbox_area(best_obs.bbox)
            range_bin = str(best_obs.range_bin or "unknown").lower()
            if raw_bbox < self.new_config.bbox_min_approach and range_bin not in ("near", "close"):
                return
        elif self.goal_manager.is_repeated_goal:
            return

        if self.servo_state.active_anchor_id != anchor.node_id:
            self.local_servo.enter(anchor.node_id, self.topo_map.current_step)
        self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "near_anchor_goal_visible")

    def act(self, plan_output: PlanDecision) -> Dict[str, Any]:
        debug = self.debug_tracer.build(
            nav_phase=self.nav_phase,
            phase_transition_reason=self.phase_transition_reason,
            packet=self._last_packet,
            memory=self._last_memory,
            structure=self._last_structure,
            nav_target=self._last_nav_target,
            servo_state=self.servo_state,
            servo_evidence=self._last_servo_evidence,
            stop=self._last_stop,
            recovery_reason=self.recovery_manager.recovery_reason,
            goal_debug=self.goal_manager.reuse_debug,
            position=self._position,
            success_stop_distance=self.new_config.success_stop_distance,
        )
        self._last_debug = debug
        return {
            "action": plan_output.action,
            "plan_action": plan_output.plan_action,
            "target_position": (
                plan_output.target_position.astype(np.float32)
                if plan_output.target_position is not None else None
            ),
            "target_node_id": plan_output.target_node_id,
            "target_type": plan_output.target_type,
            "is_exploration": plan_output.is_exploration,
            "candidate_ids": plan_output.candidate_ids,
            "scores": plan_output.scores,
            "mode": plan_output.reason,
            "nav_phase": self.nav_phase.value,
            "stop_debug": debug,
            "sticky_debug": debug,
            "navigation_debug": debug,
            "goal_reuse_debug": self.goal_manager.reuse_debug,
        }

    def on_navigation_event(self, target_node_id: Optional[str], event: str) -> Dict[str, Any]:
        out = self.recovery_manager.on_navigation_event(self.topo_map, target_node_id, event)
        self._last_navigation_event = out
        if out.get("action") == "object_anchor_reached":
            self.local_servo.enter(target_node_id, self.topo_map.current_step)
            self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "object_anchor_reached")
        elif out.get("action") == "consumed_frontier":
            if target_node_id:
                self._consumed_frontier_ids.add(target_node_id)
            self._transition(NavPhase.GLOBAL_SEARCH, out.get("event", event))
        elif out.get("action") == "blacklisted_target":
            self._transition(NavPhase.RECOVERY, out.get("event", event))
        return out

    @property
    def memory_stats(self) -> Dict[str, Any]:
        object_nodes = self.topo_map.get_nodes_by_type(NodeType.OBJECT)
        landmark_nodes = self.topo_map.get_nodes_by_type(NodeType.LANDMARK)
        mean_conf = float(np.mean([n.confidence for n in object_nodes])) if object_nodes else 0.0
        return {
            "total_nodes": self.topo_map.num_nodes,
            "visited_waypoints": len(self.topo_map.get_visited()),
            "frontiers": len(self.topo_map.get_frontiers()),
            "objects": len(object_nodes),
            "rooms": len(self.topo_map.get_nodes_by_type(NodeType.ROOM)),
            "landmarks": len(landmark_nodes),
            "step": self._step_count,
            "goals_completed": self._goals_completed,
            "consumed_frontiers": len(self._consumed_frontier_ids),
            "heavy_perception_calls": self._heavy_perception_calls,
            "object_merge_count": self._object_merge_count,
            "mean_object_confidence": mean_conf,
            "nav_phase": self.nav_phase.value,
            "last_debug": self._last_debug,
            "last_heavy": {
                "ran": self._last_packet.fresh_vlm,
                "reason": self._last_packet.trigger_reason,
                "detections": len(self._last_packet.report.objects) if self._last_packet.fresh_vlm else 0,
            },
            "visited_fallback_count": self._visited_fallback_count,
            "no_candidates_count": self._no_candidates_count,
        }

    def _decision(
        self,
        action: str,
        plan_action: str,
        target_position: Optional[np.ndarray],
        target_node_id: Optional[str],
        target_type: str,
        reason: str,
        candidate_ids: List[str],
        scores: List[float],
        expected: NavPhase = NavPhase.GLOBAL_SEARCH,
    ) -> PlanDecision:
        return PlanDecision(
            action=action,
            plan_action=plan_action,
            target_position=target_position,
            target_node_id=target_node_id,
            target_type=target_type,
            reason=reason,
            expected_phase_after_reach=expected,
            candidate_ids=candidate_ids,
            scores=[float(s) for s in scores],
            is_exploration=target_type == "frontier",
        )

    def _transition(self, phase: NavPhase, reason: str) -> None:
        if phase != self.nav_phase:
            self.nav_phase = phase
            self.phase_enter_step = self.topo_map.current_step
            self.phase_transition_reason = reason
        else:
            self.phase_transition_reason = reason

    def _active_anchor_distance(self) -> Optional[float]:
        if self._position is None or not self.servo_state.active_anchor_id:
            return None
        node = self.topo_map.get_node(self.servo_state.active_anchor_id)
        if node is None:
            return None
        anchor_pos = self.navigation_planner._anchor_position(node)
        return float(np.linalg.norm(anchor_pos - self._position))

    def _has_near_goal_anchor(self) -> bool:
        if self._position is None:
            return False
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.attributes.get("semantic_role") != "object_anchor":
                continue
            if not _label_matches(node.label, self.goal_manager.target_labels):
                continue
            pos = node.attributes.get("anchor_waypoint_position")
            anchor_pos = np.array(pos, dtype=np.float32) if pos is not None else node.position
            if float(np.linalg.norm(anchor_pos - self._position)) <= self.new_config.anchor_servo_enter_distance:
                return True
        return False


ConfTopoGOATAgent = GoatAgent

ConfTopoGOATAgentNew = GoatAgent
