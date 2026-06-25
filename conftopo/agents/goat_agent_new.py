"""ConfTopo-GOAT Agent (modular scheduler).

Base implementation for perception, memory, proposals, and servo mechanics.
For the paper-facing navigation state machine, import ``ConfTopoGOATAgent``
from ``conftopo.agents.goat_agent_final`` (or ``conftopo.agents``).
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
from conftopo.core.instruction_graph import GoalNode, InstructionGraph, GoalProposal, normalize_goal_node
from conftopo.core.landmark_roles import is_structural_label
from conftopo.perception.clip_gdino_report_builder import ClipGdinoReportBuilder
from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.perception_trigger import PerceptionTrigger, TriggerState
from conftopo.core.confidence import (
    AttributeMatcher,
    ConfidenceFactors,
    RelationScorer,
    compute_semantic_confidence,
    compute_task_score,
)
from conftopo.core.hypothesis_pool import HypothesisPool, Hypothesis
from conftopo.perception.vlm_perceiver import VLMPerceiver


CLIP_PROPOSAL_SCORE_CAP = 0.65


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


_CENTER_BEARINGS = frozenset({"center", "front", "front-center"})
_STOP_RANGE_BINS = frozenset({"near", "very_near", "close"})


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


def _compute_effective_size(
    bbox: Sequence[float] | None,
    range_bin: str | None = None,
    min_bbox_area: float = 0.04,
) -> float:
    """Bbox normalised area with a range_bin fallback.

    The VLM often omits bounding boxes even when it correctly identifies
    the target.  When bbox area is very small or missing, the estimated
    visual size from range_bin serves as a reasonable proxy:

        close -> 0.25              (passes both approach and stop thresholds)
        very_near -> 0.18          (passes stop threshold exactly)
        near -> 0.12               (passes approach only)
    """
    area = _bbox_area(bbox)
    if area >= min_bbox_area:
        return area
    proxy = {
        "close": 0.25,
        "very_near": 0.18,
        "near": 0.10,
        "mid": 0.05,
        "far": 0.02,
    }.get(str(range_bin).lower().strip() if range_bin else "", 0.0)
    return max(area, proxy)


def _bbox_growth_metrics(
    state: "ServoState",
    cfg: NewGoatConfig,
) -> Dict[str, float]:
    """Relative bbox growth since servo entry — RGB-only proxy for getting closer."""
    hist = [v for v in state.bbox_history if v > 0.0]
    baseline = float(state.servo_entry_bbox_baseline)
    if baseline <= 0.0 and hist:
        baseline = float(hist[0])
    best = float(state.best_bbox_area)
    growth_ratio = 0.0
    if baseline > 0.0 and best > baseline:
        growth_ratio = (best - baseline) / baseline
    recent_growth = 0.0
    window = max(1, int(cfg.bbox_plateau_window))
    if len(hist) >= 2:
        anchor_idx = max(0, len(hist) - window - 1)
        older = float(hist[anchor_idx])
        if older > 0.0:
            recent_growth = (float(hist[-1]) - older) / older
    effective_growth = max(growth_ratio, recent_growth)
    return {
        "baseline": baseline,
        "growth_ratio": growth_ratio,
        "recent_growth": recent_growth,
        "effective_growth": effective_growth,
    }


def _bbox_progress_close_enough(
    state: "ServoState",
    cfg: NewGoatConfig,
    *,
    plateau: bool,
    range_bin: str,
) -> bool:
    """Prefer bbox area growth / plateau over a fixed absolute stop threshold."""
    metrics = _bbox_growth_metrics(state, cfg)
    range_near = str(range_bin or "unknown").lower() in ("very_near", "close")
    growth_ok = metrics["effective_growth"] >= cfg.bbox_min_growth_ratio
    absolute_ok = state.best_bbox_area >= cfg.bbox_min_stop
    return growth_ok or plateau or range_near or absolute_ok


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
    "heater": ["bedroom", "living room", "hallway", "bathroom", "office"],
    "radiator": ["bedroom", "living room", "hallway", "bathroom", "office"],
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

# Extra labels VLM may use instead of the exact GOAT goal name.
GOAL_LABEL_ALIASES: Dict[str, List[str]] = {
    "rack": ["rack", "shelf", "shelving", "bookcase"],
    "bed": ["bed", "mattress", "queen bed", "king bed", "double bed", "bunk bed"],
    "heater": ["heater", "radiator", "space heater", "heating unit", "baseboard heater"],
    "wardrobe": ["wardrobe", "closet", "armoire", "cabinet"],
    "sink": ["sink", "basin", "washbasin"],
    "toilet": ["toilet", "commode"],
}


def _expanded_goal_labels(goal_manager: "GoalManager") -> Set[str]:
    labels = set(_split_compound_label(goal_manager.target_object))
    key = (goal_manager.target_object or "").lower().strip()
    if key in GOAL_LABEL_ALIASES:
        labels.update(GOAL_LABEL_ALIASES[key])
    for token in list(labels):
        if token in GOAL_LABEL_ALIASES:
            labels.update(GOAL_LABEL_ALIASES[token])
    return labels


def _goal_label_matches(label: Optional[str], goal_manager: "GoalManager") -> bool:
    return _label_matches(label, _expanded_goal_labels(goal_manager))


def _goal_has_active_anchor(
    topo_map: DynamicTopoMap,
    goal: "GoalManager",
    cfg: "NewGoatConfig",
) -> bool:
    """True when a non-blacklisted goal object_anchor is present in memory."""
    step = topo_map.current_step
    for node in topo_map.object_memory.get_all():
        if node.attributes.get("semantic_role") != "object_anchor":
            continue
        if not _label_matches(node.label, goal.target_labels):
            continue
        if int(node.attributes.get("blacklisted_until", -1)) >= step:
            continue
        if float(node.confidence) < cfg.anchor_min_confidence:
            continue
        return True
    return False


def _category_priors(target_object: Optional[str], table: Dict[str, List[str]]) -> List[str]:
    if not target_object:
        return []
    text = str(target_object).lower()
    for key, vals in table.items():
        if key in text:
            return list(vals)
    return []


def _room_matches_prior(node: SemanticNode, room_labels: Set[str]) -> bool:
    if not room_labels:
        return False
    label = (node.label or "").lower()
    base = str(node.attributes.get("base_label", "")).lower()
    return _label_matches(label, room_labels) or _label_matches(base, room_labels)


class NavPhase(str, Enum):
    GLOBAL_SEARCH = "GLOBAL_SEARCH"
    ROUTE_TO_STRUCTURE = "ROUTE_TO_STRUCTURE"
    ROUTE_TO_OBJECT_ANCHOR = "ROUTE_TO_OBJECT_ANCHOR"
    TRACK_TARGET = "TRACK_TARGET"
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
    stop_buffer_size: int = 5
    track_buffer_size: int = 5
    track_visible_min: int = 2
    track_bbox_min: int = 1
    anchor_blacklist_steps: int = 40
    object_write_throttle_steps: int = 4
    no_progress_window: int = 8
    no_progress_distance: float = 0.12
    phase_timeout_steps: int = 40
    local_approach_timeout_steps: int = 48
    bbox_min_approach: float = 0.04
    bbox_min_stop: float = 0.18
    bbox_min_growth_ratio: float = 0.25
    bbox_peak_ratio: float = 0.80
    stop_pose_radius: float = 0.55
    room_writeback_confidence: float = 0.7
    min_forward_before_stop: int = 2
    min_approach_distance: float = 0.5
    bbox_plateau_window: int = 3
    bbox_plateau_growth: float = 0.02
    bbox_stop_min: float = 0.35
    min_forward_after_plateau: int = 2
    servo_lost_count_max: int = 5
    anchor_min_confidence: float = 0.3
    anchor_stale_age: int = 20
    anchor_single_frame_min_confidence: float = 0.35
    anchor_servo_enter_distance: float = 2.0
    anchor_waypoint_reach_distance: float = 0.55
    visual_retreat_ratio: float = 0.35
    anchor_at_zone_distance: float = 0.6
    triangulation_min_baseline: float = 0.4
    triangulation_min_angle: float = math.radians(8.0)
    triangulation_max_condition: float = 100.0
    triangulation_max_range: float = 12.0
    camera_hfov_degrees: float = 90.0
    materialize_clip_threshold: float = 0.22
    materialize_room_prior_clip_threshold: float = 0.18
    materialize_alias_min_confidence: float = 0.20
    materialize_synthesize_conf_floor: float = 0.30
    frontier_room_prior_boost: float = 2.5
    frontier_room_prior_boost_no_anchor: float = 4.0
    fresh_anchor_single_frame_steps: int = 24
    cross_goal_semantic_bonus: float = 0.30
    semantic_room_bonus: float = 3.0
    stop_min_fresh_bbox: float = 0.06
    stop_short_approach_max: float = 0.8
    stop_mode: str = "simple"
    stop_close_min: int = 2
    track_timeout_steps: int = 24


@dataclass
class PerceptionPacket:
    report: PerceptionReport
    source: str
    fresh_vlm: bool = False
    cached_vlm: bool = False
    vlm_mode: str = "explore"
    step_id: int = 0
    trigger_reason: str = ""
    vlm_cache_age: Optional[int] = None
    vlm_position_delta: Optional[float] = None
    vlm_heading_delta: Optional[float] = None


@dataclass
class MemoryUpdateSummary:
    cur_vp_id: Optional[str] = None
    room_label: str = "unknown"
    persistent_writes: int = 0
    transient_updates: int = 0
    debug_only_writes: int = 0
    object_merge_count_this_step: int = 0
    skipped_writes: int = 0
    goal_anchor_written: bool = False
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
    centered_buffer: List[bool] = field(default_factory=list)
    close_buffer: List[bool] = field(default_factory=list)
    track_buffer: List[Dict[str, Any]] = field(default_factory=list)
    fresh_vlm_stop_step: Optional[int] = None
    last_fresh_bbox: float = 0.0
    aligned_at_peak: bool = False
    peak_bearing: str = "unknown"
    plateau_forward_count: int = 0
    best_anchor_distance: float = float("inf")
    last_anchor_distance: Optional[float] = None
    anchor_distance_history: List[float] = field(default_factory=list)
    entry_step: int = 0
    last_alignment_action: Optional[str] = None
    alignment_flip_count: int = 0
    alignment_flip_streak: int = 0
    alignment_break_count: int = 0
    servo_entry_bbox_baseline: float = 0.0

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
        self.centered_buffer = []
        self.close_buffer = []
        self.track_buffer = []
        self.fresh_vlm_stop_step = None
        self.last_fresh_bbox = 0.0
        self.aligned_at_peak = False
        self.peak_bearing = "unknown"
        self.plateau_forward_count = 0
        self.best_anchor_distance = float("inf")
        self.last_anchor_distance = None
        self.anchor_distance_history = []
        self.entry_step = 0
        self.last_alignment_action = None
        self.alignment_flip_count = 0
        self.alignment_flip_streak = 0
        self.alignment_break_count = 0
        self.servo_entry_bbox_baseline = 0.0


class GoalManager:
    def __init__(self) -> None:
        self.current_goal: Optional[GoalNode] = None
        self.previous_goal_label: Optional[str] = None
        self.reuse_debug: Dict[str, Any] = {}
        self._active_hypothesis_positions: List[np.ndarray] = []
        self.runtime_room_prior: List[str] = []
        self.runtime_landmarks: List[str] = []

    @property
    def target_object(self) -> Optional[str]:
        return getattr(self.current_goal, "target_object", None)

    @property
    def target_labels(self) -> Set[str]:
        return _expanded_goal_labels(self)

    @property
    def canonical_target_label(self) -> str:
        return (self.target_object or "").lower().strip()

    @property
    def room_prior(self) -> List[str]:
        return self._dedupe(_as_list(getattr(self.current_goal, "room_prior", [])) + self.runtime_room_prior)

    @property
    def landmark_prior(self) -> List[str]:
        return self._dedupe(_as_list(getattr(self.current_goal, "landmarks", [])) + self.runtime_landmarks)

    @property
    def goal_key(self) -> str:
        return (self.target_object or "").lower().strip()

    @property
    def is_repeated_goal(self) -> bool:
        return bool(self.previous_goal_label and self.previous_goal_label == self.target_object)

    @property
    def active_hypothesis_positions(self) -> List[np.ndarray]:
        return list(self._active_hypothesis_positions)

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

    def set_new_goal(
        self,
        goal: GoalNode,
        topo_map: DynamicTopoMap,
        runtime_room_prior: Optional[List[str]] = None,
        runtime_landmarks: Optional[List[str]] = None,
    ) -> None:
        old = self.target_object
        self.previous_goal_label = old
        self.current_goal = goal
        self.runtime_room_prior = self._dedupe(runtime_room_prior or [])
        self.runtime_landmarks = self._dedupe(runtime_landmarks or [])
        self.reuse_debug = self._scan_reuse(topo_map)

    @staticmethod
    def _dedupe(labels: List[str]) -> List[str]:
        seen = set()
        out = []
        for label in labels:
            text = str(label).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

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
            # P11: strong_negative_evidence counts as failed candidate
            strong_neg = float(node.attributes.get("strong_negative_evidence", 0.0))
            until = int(node.attributes.get("blacklisted_until", -1))
            if until >= step:
                failed_active.append(node.node_id)
            elif strong_neg > 0.5:
                failed_expired.append(node.node_id)
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
        self.hypothesis_pool = HypothesisPool(
            default_ttl=20,
            decay_rate=0.92,
            verify_seen_count=2,
            verify_confidence=0.45,
            rejected_cooldown=12,
        )
        self._clip_hypothesis_threshold: float = 0.22
        self._vlm_confirmed_labels = None
        self.vlm_perceiver: Optional[VLMPerceiver] = None
        self.last_vlm_report: Optional[PerceptionReport] = None
        self.last_vlm_goal: str = ""
        self.last_vlm_position: Optional[np.ndarray] = None
        self.last_vlm_heading: float = 0.0
        self.last_vlm_mode: str = ""
        self.last_vlm_step: Optional[int] = None
        self.last_vlm_rgb: Optional[np.ndarray] = None

    def reset_short_term(self) -> None:
        self.hypothesis_pool.clear()
        self._vlm_confirmed_labels = None
        self.last_vlm_report = None
        self.last_vlm_goal = ""
        self.last_vlm_position = None
        self.last_vlm_heading = 0.0
        self.last_vlm_mode = ""
        self.last_vlm_step = None
        self.last_vlm_rgb = None
        self.trigger_state = TriggerState()

    def set_vlm_perceiver(self, perceiver: Optional[VLMPerceiver]) -> None:
        self.vlm_perceiver = perceiver

    def set_goal(self, goal: Optional[GoalNode], landmarks: Optional[List[str]] = None) -> None:
        if goal is None:
            return
        self.light.set_goal_labels(
            [goal.target_object],
            goal.target_embedding[np.newaxis, :] if goal.target_embedding is not None else None,
        )
        effective_landmarks = landmarks if landmarks is not None else goal.landmarks
        if effective_landmarks:
            embeddings = goal.landmark_embeddings if list(effective_landmarks) == list(goal.landmarks) else None
            self.light.set_landmark_labels(effective_landmarks, embeddings)

    def run_light(
        self,
        visual_embed: Optional[np.ndarray],
        step_id: int,
        goal_visual_embed: Optional[np.ndarray] = None,
    ) -> PerceptionPacket:
        if visual_embed is not None:
            out = self.light.perceive(visual_embed, goal_visual_embed=goal_visual_embed)
        else:
            out = {}
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
        recent_actions: Sequence[str] = (),
    ) -> PerceptionPacket:
        packet = self.run_light(visual_embed, step_id)

        # ── P0: Feed CLIP hints into HypothesisPool ──
        self._feed_clip_hypotheses(packet, goal_manager, position, step_id)
        vlm_mode = "confirm" if nav_phase in (
            NavPhase.TRACK_TARGET,
            NavPhase.LOCAL_VISUAL_APPROACH,
            NavPhase.STOP_VERIFY,
            NavPhase.PEAK_RETURN,
        ) else "explore"
        self.trigger_state.nav_phase = {
            NavPhase.TRACK_TARGET: "approach_confirm",
            NavPhase.LOCAL_VISUAL_APPROACH: "approach_confirm",
            NavPhase.STOP_VERIFY: "confirm",
            NavPhase.PEAK_RETURN: "approach",
        }.get(nav_phase, "explore")

        force_fresh = nav_phase == NavPhase.STOP_VERIFY
        force_reason = getattr(self, "_force_heavy_reason", None)
        if force_reason and self.vlm_perceiver is not None and rgb is not None:
            should_run, reason = True, str(force_reason)
        elif force_fresh and self.vlm_perceiver is not None and rgb is not None:
            should_run, reason = True, "stop_verify_fresh"
        else:
            should_run, reason = self._should_run_vlm(
                packet.report, rgb, position, topo_map, goal_manager, has_near_goal_object, step_id,
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
            previous_rgb = None
            if (
                vlm_mode == "confirm"
                and self.last_vlm_rgb is not None
                and self.last_vlm_goal == goal_manager.goal_key
                and self.last_vlm_mode == "confirm"
            ):
                previous_rgb = self.last_vlm_rgb
            report = self.vlm_perceiver.perceive(
                rgb,
                goal_manager.target_object or "explore",
                visual_embed=visual_embed,
                step_id=step_id,
                context=goal_manager.goal_context(),
                mode=vlm_mode,
                previous_rgb=previous_rgb,
                action_history=recent_actions,
            )
            clip_goal_sim = float(packet.report.best_goal_sim)
            self._materialize_goal_perception(report, goal_manager, clip_goal_sim)
            self._feed_vlm_weak_hypotheses(report, goal_manager, position, step_id)
            self.last_vlm_report = report
            self.last_vlm_goal = goal_manager.goal_key
            self.last_vlm_position = None if position is None else position.copy()
            self.last_vlm_heading = float(heading)
            self.last_vlm_mode = vlm_mode
            self.last_vlm_step = step_id
            self.last_vlm_rgb = np.asarray(rgb).copy()
            self.trigger.record_run(self.trigger_state, step_id, reason)

            # ── P0: Store VLM-confirmed labels for hypothesis promotion ──
            self._vlm_confirmed_labels = set(
                str(o.label).lower().strip()
                for o in report.objects
                if _label_matches(o.label, goal_manager.target_labels)
            )
            return PerceptionPacket(
                report=report,
                source="vlm",
                fresh_vlm=True,
                cached_vlm=False,
                vlm_mode=vlm_mode,
                step_id=step_id,
                trigger_reason=reason,
            )

        # Cache is unsafe in visual servo: only fresh VLM can update
        # bbox_history, best_bbox_area, and visual_advance_steps.
        if nav_phase not in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY, NavPhase.PEAK_RETURN):
            cached = self._valid_cached_vlm(goal_manager.goal_key, position, heading, vlm_mode, step_id)
            if cached is not None:
                cached_report = PerceptionReport.from_dict(cached.to_dict())
                cached_report.source = "cached_vlm"
                cache_age = int(step_id - self.last_vlm_step)
                pos_delta = None
                if position is not None and self.last_vlm_position is not None:
                    pos_delta = float(np.linalg.norm(position - self.last_vlm_position))
                heading_delta = abs(_angle_delta(float(heading), self.last_vlm_heading))
                return PerceptionPacket(
                    report=cached_report,
                    source="cached_vlm",
                    fresh_vlm=False,
                    cached_vlm=True,
                    vlm_mode=vlm_mode,
                    step_id=step_id,
                    trigger_reason="cache_valid",
                    vlm_cache_age=cache_age,
                    vlm_position_delta=pos_delta,
                    vlm_heading_delta=heading_delta,
                )
        packet.vlm_mode = vlm_mode
        packet.trigger_reason = reason
        self._vlm_confirmed_labels = None
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
        goal_manager: GoalManager,
        has_near_goal_object: bool,
        step_id: int,
    ) -> Tuple[bool, str]:
        if self.vlm_perceiver is None:
            return False, "no_vlm_perceiver"
        self.trigger_state.goal_local_step += 1

        goal_key = goal_manager.goal_key
        if goal_key and self._hypotheses_need_verify(goal_key):
            interval = max(2, int(self.config.perception.heavy_interval) // 2)
            last = self.trigger_state.last_heavy_step
            if last is None or step_id - last >= interval:
                return True, "hypothesis_verify"
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


    def _feed_clip_hypotheses(
        self,
        packet: PerceptionPacket,
        goal_manager: GoalManager,
        position: Optional[np.ndarray],
        step_id: int,
    ) -> None:
        goal_key = goal_manager.goal_key
        if not goal_key:
            return
        best_goal_sim = packet.report.best_goal_sim
        if best_goal_sim < self._clip_hypothesis_threshold:
            return

        hyp = Hypothesis(
            id="",
            goal_id=goal_key,
            kind="object",
            label=goal_key,
            source="clip",
            anchor_node_id=None,
            position=position.copy() if position is not None else None,
            score=float(best_goal_sim),
            confidence=float(best_goal_sim),
            first_seen_step=step_id,
            last_seen_step=step_id,
            seen_count=1,
            attributes={"goal_key": goal_key, "best_sim": float(best_goal_sim)},
        )
        self.hypothesis_pool.add_or_update(hyp)

    def _materialize_goal_perception(
        self,
        report: PerceptionReport,
        goal_manager: GoalManager,
        clip_goal_sim: float = 0.0,
    ) -> None:
        """Bridge VLM/CLIP signals into a writable goal object observation.

        Planner completion requires an ``object_anchor`` node.  VLM often reports
        unrelated ``objects[]`` entries while omitting the goal label, or sets
        top-level ``goal_visible`` without a matching detection row.
        """
        if not goal_manager.canonical_target_label:
            return

        matches = [
            o for o in report.objects
            if _goal_label_matches(o.label, goal_manager)
        ]
        min_conf = float(self.config.perception.object_threshold)
        clip_sim = float(clip_goal_sim)

        alias_min = float(self.new_config.materialize_alias_min_confidence)
        if matches:
            best = max(matches, key=lambda o: _bbox_area(o.bbox) + float(o.confidence))
            conf = float(best.confidence)
            if conf >= min_conf or conf >= alias_min:
                best.label = goal_manager.canonical_target_label
                best.confidence = max(conf, min_conf)
                report.goal_visible = True
                report.goal_match_confidence = max(
                    float(report.goal_match_confidence),
                    float(best.confidence),
                )
                self._promote_goal_visibility(report, best)
            return

        top_conf = float(report.goal_match_confidence)
        clip_threshold = float(self.new_config.materialize_clip_threshold)
        room_clip_threshold = float(self.new_config.materialize_room_prior_clip_threshold)
        room_label = str(report.room_label or "").lower().strip()
        room_prior = {x.lower() for x in goal_manager.room_prior}
        room_matches_prior = bool(room_label and room_label in room_prior)
        synth_floor = float(self.new_config.materialize_synthesize_conf_floor)
        should_synthesize = (
            (report.goal_visible and top_conf >= min_conf)
            or top_conf >= max(min_conf, synth_floor)
            or clip_sim >= clip_threshold
            or (room_matches_prior and clip_sim >= room_clip_threshold)
            or (report.goal_visible and clip_sim >= room_clip_threshold)
        )
        if not should_synthesize:
            return

        synth_conf = max(top_conf, clip_sim * 0.85, min_conf)
        if room_matches_prior:
            synth_conf = max(synth_conf, clip_sim * 0.90, room_clip_threshold)
        bearing = str(report.target_direction or "unknown").lower()
        if bearing == "unknown":
            bearing = "center"
        range_bin = self._infer_range_bin(report)
        synthetic = ObjectObservation(
            label=goal_manager.canonical_target_label,
            bbox=None,
            confidence=float(synth_conf),
            source="vlm",
            visible=True,
            visibility="partially_visible" if report.target_visibility == "partial" else "visible",
            bearing=bearing,
            range_bin=range_bin,
            spatial_relation=[],
            bbox_confidence="medium",
        )
        report.objects.append(synthetic)
        report.goal_visible = True
        report.goal_match_confidence = max(top_conf, synth_conf)
        self._promote_goal_visibility(report, synthetic)

    @staticmethod
    def _promote_goal_visibility(report: PerceptionReport, obs: ObjectObservation) -> None:
        if str(report.target_visibility).lower() in (
            "not_visible", "uncertain", "unknown", "",
        ):
            obj_vis = str(getattr(obs, "visibility", "unknown")).lower()
            if obj_vis in ("visible", "clear"):
                report.target_visibility = "clear"
            elif obj_vis in ("partially_visible", "partial"):
                report.target_visibility = "partial"
            elif bool(getattr(obs, "visible", False)):
                report.target_visibility = "partial"
        if str(report.target_direction).lower() == "unknown":
            bearing = str(getattr(obs, "bearing", "unknown")).lower()
            if bearing not in ("unknown", ""):
                report.target_direction = bearing

    @staticmethod
    def _infer_range_bin(report: PerceptionReport) -> str:
        scale = str(getattr(report, "apparent_scale", "unknown")).lower()
        mapping = {
            "tiny": "far",
            "small": "mid",
            "medium": "near",
            "large": "very_near",
            "very_large": "close",
        }
        if scale in mapping:
            return mapping[scale]
        vis = str(report.target_visibility).lower()
        if vis == "clear":
            return "near"
        if vis == "partial":
            return "mid"
        return "unknown"

    def _reconcile_goal_from_detections(
        self,
        report: PerceptionReport,
        goal_manager: GoalManager,
        clip_goal_sim: float = 0.0,
    ) -> None:
        self._materialize_goal_perception(report, goal_manager, clip_goal_sim)

    def _feed_vlm_weak_hypotheses(
        self,
        report: PerceptionReport,
        goal_manager: GoalManager,
        position: Optional[np.ndarray],
        step_id: int,
    ) -> None:
        goal_key = goal_manager.goal_key
        if not goal_key or not report.objects:
            return
        threshold = float(self.config.perception.object_detection_threshold)
        confirmed: List[ObjectObservation] = []
        weak: List[ObjectObservation] = []
        for obs in report.objects:
            if _goal_label_matches(obs.label, goal_manager):
                confirmed.append(obs)
            elif float(obs.confidence) >= threshold:
                confirmed.append(obs)
            else:
                weak.append(obs)
        report.objects = confirmed
        for obs in weak:
            label = str(obs.label or "").strip().lower()
            if not label:
                continue
            self.hypothesis_pool.add_or_update(Hypothesis(
                id="",
                goal_id=goal_key,
                kind="object",
                label=label,
                source="vlm_weak",
                anchor_node_id=None,
                position=position.copy() if position is not None else None,
                score=float(obs.confidence),
                confidence=float(obs.confidence),
                attributes=dict(getattr(obs, "attributes", {}) or {}),
                weak_bbox=list(obs.bbox) if obs.bbox is not None else None,
                weak_relations=list(getattr(obs, "spatial_relation", []) or []),
                first_seen_step=step_id,
                last_seen_step=step_id,
                seen_count=1,
                trigger_reason="vlm_weak_detection",
            ))

    def _hypotheses_need_verify(self, goal_key: str) -> bool:
        top = self.hypothesis_pool.get_top_for_vlm(goal_id=goal_key, k=1)
        return bool(top and top[0].status == "needs_verify")

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
        self._navmesh_snap: Optional[Any] = None

    def set_navmesh_snap(self, snap_fn: Optional[Any]) -> None:
        """Inject a callable(pos_3d) -> snapped_pos_3d_or_None for frontier validation."""
        self._navmesh_snap = snap_fn

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
        cur_vp = self._write_waypoint(topo_map, position, heading, visual_embed)
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
        heading: float,
        visual_embed: Optional[np.ndarray],
    ) -> str:
        nearest = topo_map.find_nearest_node(position, NodeType.WAYPOINT_VISITED)
        if nearest is not None and float(np.linalg.norm(nearest.position - position)) <= self.cfg.waypoint_merge_radius:
            nearest.step_id = topo_map.current_step
            nearest.visit_count += 1
            nearest.attributes["heading"] = float(heading)
            if self.cur_vp_id and self.cur_vp_id != nearest.node_id:
                topo_map.add_edge(
                    self.cur_vp_id,
                    nearest.node_id,
                    EdgeType.NAVIGABLE,
                    weight=float(np.linalg.norm(
                        topo_map.get_node(self.cur_vp_id).position - nearest.position
                    )) if topo_map.get_node(self.cur_vp_id) is not None else 1.0,
                )
            self.prev_vp_id = self.cur_vp_id
            self.cur_vp_id = nearest.node_id
            return nearest.node_id
        node_id = topo_map.add_node(
            NodeType.WAYPOINT_VISITED,
            position=position,
            embedding=visual_embed,
            confidence=1.0,
            attributes={
                "semantic_role": "visited_waypoint",
                "heading": float(heading),
            },
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
            if self._navmesh_snap is not None:
                snapped = self._navmesh_snap(pos)
                if snapped is None:
                    continue
                pos = snapped
            if topo_map.has_nearby_visited(pos, radius=self.cfg.frontier_merge_radius):
                continue
            existing = topo_map.find_nearest_node(pos, NodeType.WAYPOINT_FRONTIER)
            if existing is not None and float(np.linalg.norm(existing.position - pos)) < self.cfg.frontier_merge_radius:
                continue
            fid = topo_map.add_node(
                NodeType.WAYPOINT_FRONTIER,
                position=pos,
                confidence=0.5,
                attributes={
                    "semantic_role": "frontier",
                    "anchor_waypoint_id": cur_vp,
                    "direction_delta": delta,
                },
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
            self._transient_anchor_update(
                topo_map,
                packet.report.objects,
                active_anchor_id,
                position,
                heading,
                cur_vp,
                goal_manager,
                summary,
            )
            return

        for obs in packet.report.objects:
            role = self._semantic_role(obs, goal_manager)
            if role != "object_anchor":
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
                object_attrs=getattr(obs, "attributes", None),
            )
            node = topo_map.get_node(obj_id)
            if node is not None:
                node.attributes["semantic_role"] = role
                if role == "object_anchor":
                    node.attributes["goal_detection_step"] = topo_map.current_step
                    summary.goal_anchor_written = True

                # ── P2: Create explicit semantic relation edges ──
                # IN_ROOM: bind object to its room
                cur_room_label = packet.report.room_label
                if cur_room_label and cur_room_label != "unknown":
                    nearby_rooms = topo_map.find_nodes_within_radius(
                        position, 5.0, NodeType.ROOM,
                    )
                    for room in nearby_rooms:
                        if room.label == cur_room_label:
                            if not topo_map.graph.has_edge(obj_id, room.node_id):
                                topo_map.add_edge(obj_id, room.node_id, EdgeType.IN_ROOM,
                                    relation=f"{node.label}_in_{room.label}",
                                    confidence=float(node.confidence),
                                )
                            break

                # ANCHORED_TO: bind object to its anchor waypoint
                anchor_wp_id = node.attributes.get("anchor_waypoint_id")
                if anchor_wp_id and topo_map.has_node(anchor_wp_id):
                    if not topo_map.graph.has_edge(obj_id, anchor_wp_id):
                        topo_map.add_edge(obj_id, anchor_wp_id, EdgeType.ANCHORED_TO,
                            relation=f"{node.label}_anchored_to_{anchor_wp_id}",
                            confidence=float(node.confidence),
                        )

                # NEAR: bind to nearby objects / landmarks
                nearby_semantic = topo_map.find_nodes_within_radius(
                    position, topo_map.merge_radius * 2.0,
                )
                for nearby in nearby_semantic:
                    if nearby.node_id == obj_id:
                        continue
                    if nearby.node_type in (NodeType.OBJECT, NodeType.LANDMARK):
                        if not topo_map.graph.has_edge(obj_id, nearby.node_id):
                            topo_map.add_edge(obj_id, nearby.node_id, EdgeType.NEAR,
                                relation=f"{node.label}_near_{nearby.label}",
                                confidence=float(node.confidence) * float(nearby.confidence),
                            )

                if role == "object_anchor":
                    node.attributes["goal_detection_step"] = topo_map.current_step
                    node.attributes["goal_detection_confidence"] = float(obs.confidence)
                    self._record_target_geometry(
                        topo_map, node, obs, position, heading, cur_vp,
                    )
            summary.persistent_writes += 1
            summary.object_merge_count_this_step += int(bool(merged))
            summary.written_node_ids.append(obj_id)

    def _semantic_role(self, obs: ObjectObservation, goal_manager: GoalManager) -> str:
        if _goal_label_matches(obs.label, goal_manager):
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
        heading: float,
        cur_vp: str,
        goal_manager: GoalManager,
        summary: MemoryUpdateSummary,
    ) -> None:
        if not active_anchor_id:
            summary.debug_only_writes += len(observations)
            return
        node = topo_map.get_node(active_anchor_id)
        if node is None:
            summary.debug_only_writes += len(observations)
            return
        best = max(
            (
                obs for obs in observations
                if _label_matches(obs.label, goal_manager.target_labels)
            ),
            key=lambda o: _bbox_area(o.bbox),
            default=None,
        )
        if best is None:
            summary.debug_only_writes += len(observations)
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
        self._record_target_geometry(
            topo_map, node, best, position, heading, cur_vp,
        )
        summary.transient_updates += 1

    def _record_target_geometry(
        self,
        topo_map: DynamicTopoMap,
        node: SemanticNode,
        observation: ObjectObservation,
        position: np.ndarray,
        heading: float,
        viewpoint_id: Optional[str],
    ) -> None:
        """Accumulate coarse bearing rays and update a conservative object estimate."""
        ray_heading = self._observation_ray_heading(observation, heading)
        if ray_heading is None:
            return
        records = node.attributes.setdefault("bearing_observations", [])
        record = {
            "observer_position": np.asarray(position, dtype=np.float32).tolist(),
            "viewpoint_id": viewpoint_id,
            "view_heading": float(heading),
            "ray_heading": float(ray_heading),
            "bearing": str(observation.bearing or "unknown").lower(),
            "range_bin": str(observation.range_bin or "unknown").lower(),
            "bbox": (
                [float(v) for v in observation.bbox]
                if observation.bbox is not None else None
            ),
            "confidence": float(observation.confidence),
            "step_id": topo_map.current_step,
        }
        if records:
            previous = records[-1]
            prev_pos = np.asarray(previous.get("observer_position"), dtype=np.float32)
            same_position = float(np.linalg.norm(prev_pos - position)) < 0.1
            same_ray = abs(_angle_delta(
                float(previous.get("ray_heading", ray_heading)), ray_heading,
            )) < math.radians(3.0)
            if same_position and same_ray:
                records[-1] = record
            else:
                records.append(record)
        else:
            records.append(record)
        if len(records) > 12:
            del records[:-12]

        estimate = self._triangulate_object_position(records)
        if estimate is None:
            return
        node.attributes["estimated_object_position"] = estimate.tolist()
        node.attributes["triangulation_view_count"] = len(records)
        node.attributes["triangulation_step"] = topo_map.current_step

        approach = self._best_observed_approach(records, estimate)
        if approach is not None:
            node.attributes["best_approach_position"] = approach.tolist()
            node.attributes["best_approach_source"] = "bearing_triangulation"

    def _observation_ray_heading(
        self,
        observation: ObjectObservation,
        heading: float,
    ) -> Optional[float]:
        bbox = observation.bbox
        if bbox is not None and len(bbox) >= 4:
            x1, _, x2, _ = [float(v) for v in bbox[:4]]
            if max(abs(x1), abs(x2)) <= 1.5:
                center_x = min(1.0, max(0.0, (x1 + x2) * 0.5))
                offset = (center_x - 0.5) * math.radians(self.cfg.camera_hfov_degrees)
                return _angle_delta(float(heading), float(offset))
        bearing = str(observation.bearing or "unknown").lower()
        offsets = {
            "left": math.radians(25.0),
            "left_front": math.radians(15.0),
            "front-left": math.radians(15.0),
            "center": 0.0,
            "front": 0.0,
            "front-center": 0.0,
            "right_front": math.radians(-15.0),
            "front-right": math.radians(-15.0),
            "right": math.radians(-25.0),
        }
        if bearing not in offsets:
            return None
        return _angle_delta(float(heading), -float(offsets[bearing]))

    def _triangulate_object_position(
        self,
        records: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        usable = [
            record for record in records
            if record.get("observer_position") is not None
            and record.get("ray_heading") is not None
        ]
        if len(usable) < 2:
            return None
        points = np.asarray(
            [record["observer_position"] for record in usable],
            dtype=np.float64,
        )
        planar = points[:, [0, 2]]
        max_baseline = max(
            float(np.linalg.norm(planar[i] - planar[j]))
            for i in range(len(planar))
            for j in range(i)
        )
        if max_baseline < self.cfg.triangulation_min_baseline:
            return None
        headings = [float(record["ray_heading"]) for record in usable]
        max_angle = max(
            abs(_angle_delta(headings[i], headings[j]))
            for i in range(len(headings))
            for j in range(i)
        )
        max_angle = min(max_angle, abs(math.pi - max_angle))
        if max_angle < self.cfg.triangulation_min_angle:
            return None

        matrix = np.zeros((2, 2), dtype=np.float64)
        rhs = np.zeros(2, dtype=np.float64)
        rays = []
        for point, ray_heading in zip(planar, headings):
            direction = np.array(
                [-math.sin(ray_heading), -math.cos(ray_heading)],
                dtype=np.float64,
            )
            projection = np.eye(2, dtype=np.float64) - np.outer(direction, direction)
            matrix += projection
            rhs += projection @ point
            rays.append((point, direction))
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition) or condition > self.cfg.triangulation_max_condition:
            return None
        center = np.linalg.solve(matrix, rhs)
        forward_ranges = [
            float(np.dot(center - point, direction))
            for point, direction in rays
        ]
        if any(distance <= 0.0 or distance > self.cfg.triangulation_max_range for distance in forward_ranges):
            return None
        y = float(np.mean(points[:, 1]))
        return np.array([center[0], y, center[1]], dtype=np.float32)

    @staticmethod
    def _best_observed_approach(
        records: List[Dict[str, Any]],
        estimate: np.ndarray,
    ) -> Optional[np.ndarray]:
        candidates = []
        for record in records:
            raw = record.get("observer_position")
            if raw is None:
                continue
            position = np.asarray(raw, dtype=np.float32)
            distance = float(np.linalg.norm(position - estimate))
            candidates.append((distance, position))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1].copy()


class StructurePlanner:
    def __init__(self, new_config: Optional[NewGoatConfig] = None) -> None:
        self.cfg = new_config or NewGoatConfig()

    def select(self, topo_map: DynamicTopoMap, goal: GoalManager, position: Optional[np.ndarray]) -> StructureTarget:
        labels = {x.lower() for x in goal.room_prior + goal.landmark_prior}
        room_labels = {x.lower() for x in goal.room_prior}
        no_goal_anchor = not _goal_has_active_anchor(topo_map, goal, self.cfg)
        best: Optional[SemanticNode] = None
        best_score = -1.0
        for node in topo_map.get_nodes_by_type(NodeType.ROOM) + topo_map.get_nodes_by_type(NodeType.LANDMARK) + topo_map.get_nodes_by_type(NodeType.OBJECT) + topo_map.get_nodes_by_type(NodeType.GOAL_REGION) + topo_map.get_nodes_by_type(NodeType.OBJECT_SUMMARY):
            if int(node.attributes.get("blacklisted_until", -1)) >= topo_map.current_step:
                continue
            role = node.attributes.get("semantic_role")
            label = (node.label or "").lower()
            if node.node_type == NodeType.OBJECT and role not in ("context_object", "environment_object"):
                continue
            if labels and not _label_matches(label, labels):
                continue
            score = float(node.confidence)
            if node.node_type == NodeType.ROOM and _label_matches(label, room_labels):
                score += 2.0
                if no_goal_anchor:
                    score += 2.5
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
        proposals = self.collect_goal_proposals(
            topo_map, goal, structure, nav_phase, position, cur_vp_id,
        )
        scored = self.score_goal_proposals(proposals, goal, topo_map, position)
        best = self.select_best_proposal(scored)
        if best is None:
            return NavTarget(reason="no_navigation_candidates"), [], []
        nav = self.proposal_to_nav_target(best, topo_map, position, cur_vp_id)
        ranked = sorted(scored, key=lambda p: p.score, reverse=True)
        return nav, [p.candidate_node_id for p in ranked[:10]], [p.score for p in ranked[:10]]

    def collect_goal_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        structure: StructureTarget,
        nav_phase: NavPhase,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str] = None,
    ) -> List[GoalProposal]:
        proposals: List[GoalProposal] = []
        proposals.extend(self._collect_object_proposals(topo_map, goal, nav_phase, position, cur_vp_id))
        proposals.extend(self._collect_cross_goal_semantic_proposals(
            topo_map, goal, structure, position,
        ))
        if structure.node_id and not any(
            p.candidate_node_id == structure.node_id for p in proposals
        ):
            proposals.append(GoalProposal(
                goal_id=goal.goal_key,
                candidate_node_id=structure.node_id,
                candidate_type=structure.target_type,
                target_position=structure.position.copy() if structure.position is not None else None,
                score=0.15,
                source="room" if structure.target_type == "room" else "structure",
                can_stop=False,
                requires_verification=True,
                evidence_refs=[{"reason": structure.reason}],
            ))
        proposals.extend(self._collect_frontier_proposals(topo_map, goal, structure, position))
        if not proposals:
            visited = self._farthest_visited(topo_map, position)
            if visited is not None:
                proposals.append(GoalProposal(
                    goal_id=goal.goal_key,
                    candidate_node_id=visited.node_id,
                    candidate_type="visited",
                    target_position=visited.position.copy(),
                    score=0.05,
                    source="visited_fallback",
                    can_stop=False,
                    requires_verification=True,
                ))
        return proposals

    def _collect_cross_goal_semantic_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        structure: StructureTarget,
        position: Optional[np.ndarray],
    ) -> List[GoalProposal]:
        """Reuse preserved / room-prior-matching semantic nodes for the goal.

        Includes cross-goal preserved nodes AND same-task room nodes that
        match the goal's room_prior (they won't have cross_goal_preserved yet
        because that flag is only set at the next goal switch).
        """
        proposals: List[GoalProposal] = []
        step = topo_map.current_step
        room_labels = {x.lower() for x in goal.room_prior}
        landmark_labels = {x.lower() for x in goal.landmark_prior}
        seen_ids: Set[str] = set()
        too_close = 0.6

        def _is_too_close(node: SemanticNode) -> bool:
            if position is None or node.position is None:
                return False
            return float(np.linalg.norm(node.position - position)) < too_close

        def _append_semantic(node: SemanticNode, candidate_type: str, reason: str,
                             require_preserved: bool = True) -> None:
            if node.node_id in seen_ids:
                return
            if int(node.attributes.get("blacklisted_until", -1)) >= step:
                return
            if require_preserved and not node.attributes.get("cross_goal_preserved"):
                return
            if _is_too_close(node):
                return
            seen_ids.add(node.node_id)
            proposals.append(GoalProposal(
                goal_id=goal.goal_key,
                candidate_node_id=node.node_id,
                candidate_type=candidate_type,
                target_position=node.position.copy(),
                score=float(node.confidence),
                semantic_score=float(node.confidence),
                source="semantic_memory",
                status="preserved",
                can_stop=False,
                requires_verification=True,
                evidence_refs=[{
                    "reason": reason,
                    "cross_goal_preserved": bool(node.attributes.get("cross_goal_preserved")),
                    "semantic_role": node.attributes.get("semantic_role", node.node_type.value),
                }],
            ))

        for node in topo_map.get_nodes_by_type(NodeType.ROOM):
            if not _room_matches_prior(node, room_labels):
                continue
            ctype = "room_summary" if node.attributes.get("summary_type") == "room_region" else "room"
            _append_semantic(node, ctype, "room_prior_reuse", require_preserved=False)

        for node in topo_map.get_nodes_by_type(NodeType.GOAL_REGION):
            if room_labels and not _room_matches_prior(node, room_labels):
                region_label = str(node.attributes.get("region_label", node.label or "")).lower()
                if not _label_matches(region_label, room_labels):
                    continue
            _append_semantic(node, "goal_region", "cross_goal_region_reuse")

        for node in topo_map.get_nodes_by_type(NodeType.LANDMARK):
            if landmark_labels and not _label_matches(node.label, landmark_labels):
                continue
            if not landmark_labels and not node.attributes.get("cross_goal_preserved"):
                continue
            _append_semantic(node, "landmark", "cross_goal_landmark_reuse")

        for node in topo_map.object_memory.get_all():
            role = node.attributes.get("semantic_role")
            if role == "object_anchor" and _label_matches(node.label, goal.target_labels):
                _append_semantic(node, "object_anchor", "cross_goal_same_label_anchor")
                continue
            if role == "context_object":
                label_match = bool(landmark_labels and _label_matches(node.label, landmark_labels))
                ctx_room = str(node.attributes.get("room_context", "")).lower()
                room_match = bool(room_labels and _label_matches(ctx_room, room_labels))
                if label_match or room_match:
                    _append_semantic(node, role, "cross_goal_context_reuse")

        if structure.node_id and structure.node_id not in seen_ids:
            node = topo_map.get_node(structure.node_id)
            if node is not None and node.attributes.get("cross_goal_preserved"):
                _append_semantic(
                    node,
                    structure.target_type,
                    "cross_goal_structure_reuse",
                )

        return proposals

    def _collect_object_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        nav_phase: NavPhase,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> List[GoalProposal]:
        proposals: List[GoalProposal] = []
        step = topo_map.current_step
        for node in topo_map.object_memory.get_all():
            if node.attributes.get("semantic_role") != "object_anchor":
                continue
            if not _label_matches(node.label, goal.target_labels):
                continue
            if int(node.attributes.get("blacklisted_until", -1)) >= step:
                continue
            if float(node.confidence) < self.cfg.anchor_min_confidence:
                continue
            multi_view = int(node.attributes.get("multi_view_count", node.visit_count))
            detection_step = int(node.attributes.get("goal_detection_step", 0))
            fresh_anchor = (step - detection_step) <= self.cfg.fresh_anchor_single_frame_steps
            min_single_conf = (
                self.cfg.materialize_alias_min_confidence
                if fresh_anchor
                else self.cfg.anchor_single_frame_min_confidence
            )
            if multi_view <= 1 and float(node.confidence) < min_single_conf:
                continue
            route = self._resolve_object_route(node, position, cur_vp_id)
            if route is None:
                continue
            can_stop = route.target_type in ("object_anchor", "object_approach")
            proposals.append(GoalProposal(
                goal_id=goal.goal_key,
                candidate_node_id=node.node_id,
                candidate_type=route.target_type,
                anchor_node_id=node.attributes.get("anchor_waypoint_id"),
                target_position=route.target_position.copy() if route.target_position is not None else None,
                score=float(node.confidence),
                semantic_score=float(node.confidence),
                source="object_memory",
                status="active",
                can_stop=can_stop,
                requires_verification=not can_stop,
                evidence_refs=[{
                    "nav_reason": route.reason,
                    "expected_phase_after_reach": route.expected_phase_after_reach.value,
                    "memory_state": node.attributes.get("memory_state", "confirmed"),
                }],
            ))
        return proposals

    def _frontier_semantic_score(
        self,
        topo_map: DynamicTopoMap,
        node: SemanticNode,
        goal: Optional[GoalManager],
        structure: StructureTarget,
    ) -> float:
        sem_score = 0.0
        if structure.position is not None:
            struct_dist = float(np.linalg.norm(node.position - structure.position))
            sem_score += max(0.0, 1.0 - struct_dist / 10.0)
        if goal and goal.target_object:
            ts = compute_task_score(
                target_object=goal.target_object or "",
                attributes=getattr(goal.current_goal, "attributes", []),
                relations=getattr(goal.current_goal, "relations", []),
                observed_label=node.label or "",
                observed_room=node.attributes.get("room_context", ""),
                room_prior=goal.room_prior,
                landmarks=goal.landmark_prior,
                map_relation_labels=node.attributes.get("view_object_labels", []),
            )
            sem_score += ts * 2.0
        for ctx_pos in self._context_object_positions(topo_map, goal):
            ctx_dist = float(np.linalg.norm(node.position - ctx_pos))
            if ctx_dist < 8.0:
                sem_score += 2.5 * max(0.0, 1.0 - ctx_dist / 8.0)
        no_goal_anchor = goal is not None and not _goal_has_active_anchor(topo_map, goal, self.cfg)
        room_boost = (
            self.cfg.frontier_room_prior_boost_no_anchor
            if no_goal_anchor
            else self.cfg.frontier_room_prior_boost
        )
        rp_best = 0.0
        for rp_pos in self._room_prior_positions(topo_map, goal):
            rp_dist = float(np.linalg.norm(node.position - rp_pos))
            if rp_dist < 12.0:
                rp_best = max(rp_best, room_boost * max(0.0, 1.0 - rp_dist / 12.0))
        sem_score += rp_best
        hypothesis_positions = goal.active_hypothesis_positions if goal else []
        for hyp_pos in hypothesis_positions:
            hyp_dist = float(np.linalg.norm(node.position - hyp_pos))
            if hyp_dist < 8.0:
                sem_score += 2.0 * max(0.0, 1.0 - hyp_dist / 8.0)
        node.attributes["frontier_semantic_value"] = float(sem_score)
        return sem_score

    def _collect_frontier_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        structure: StructureTarget,
        position: Optional[np.ndarray],
    ) -> List[GoalProposal]:
        if position is None:
            return []
        proposals: List[GoalProposal] = []
        frontiers = []
        for node in topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER):
            if node.attributes.get("consumed") or int(node.attributes.get("blacklisted_until", -1)) >= topo_map.current_step:
                continue
            dist = float(np.linalg.norm(node.position - position))
            if dist < self.cfg.min_frontier_distance:
                continue
            anchor_wp = node.attributes.get("anchor_waypoint_id")
            risk = 0.0
            if anchor_wp and topo_map.graph.has_edge(anchor_wp, node.node_id):
                edge_data = topo_map.graph.edges[anchor_wp, node.node_id]
                trav = float(edge_data.get("traversability", 1.0))
                blocked_count = int(edge_data.get("blocked_count", 0))
                if trav < 0.2 or blocked_count > 3:
                    continue
                risk = max(0.0, (1.0 - trav)) + 0.2 * blocked_count
            sem = self._frontier_semantic_score(topo_map, node, goal, structure)
            frontiers.append((dist, sem, risk, node))
        if not frontiers:
            return []
        max_dist = max(d for d, _, _, _ in frontiers)
        for dist, sem, risk, node in frontiers:
            geo_score = dist / max(max_dist, 1.0)
            proposals.append(GoalProposal(
                goal_id=goal.goal_key,
                candidate_node_id=node.node_id,
                candidate_type="frontier",
                anchor_node_id=node.attributes.get("anchor_waypoint_id"),
                target_position=node.position.copy(),
                score=geo_score + sem - risk,
                semantic_score=sem,
                frontier_value=sem,
                reachability_score=max(0.0, 1.0 - min(dist / 20.0, 1.0)),
                distance_cost=min(dist / 20.0, 1.0),
                risk_penalty=risk,
                source="frontier",
                can_stop=False,
                requires_verification=True,
            ))
        return proposals

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
            # Section 9/10: attribute confidence + negative evidence in scoring
            attr_conf = float(node.attributes.get("attribute_confidence", 0.0))
            neg_ev = float(node.attributes.get("negative_evidence", 0.0))
            strong_neg = float(node.attributes.get("strong_negative_evidence", 0.0))
            score = conf + float(node.attributes.get("target_relevance", 0.0))
            # attribute_confidence: small reliability bonus unless goal has explicit attributes
            score += attr_conf * 0.05
            # negative evidence: separate weak (low score) from strong (explicit rejection)
            score -= neg_ev * 0.20
            score -= strong_neg * 0.50
            # P12: task_score for unified task-driven ranking
            ts = compute_task_score(
                target_object=goal.target_object or "",
                observed_label=node.label,
                observed_attributes=node.attributes.get("object_attributes", {}),
                observed_room=node.attributes.get("room_context", ""),
                observed_relations=node.attributes.get("spatial_relation", []),
                room_prior=goal.room_prior,
                landmarks=goal.landmark_prior,
            )
            score += ts * 0.3
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
        ap_id = node.attributes.get("approach_point_id")
        if ap_id is not None:
            ap_node = node
            # Resolve from the object's own attributes if called with a non-object node
            actual_node = None
            # Try via topo_map if accessible; fallback to attribute check
            pos = node.attributes.get("best_approach_position")
            if pos is not None:
                return np.array(pos, dtype=np.float32)
            return None
    def at_anchor_waypoint(
        self,
        anchor: SemanticNode,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> bool:
        if position is None:
            return False
        wp_pos = self._anchor_waypoint_position(anchor)
        if wp_pos is not None:
            return float(np.linalg.norm(position - wp_pos)) <= self.cfg.anchor_waypoint_reach_distance
        return float(np.linalg.norm(position - anchor.position)) <= self.cfg.anchor_waypoint_reach_distance

    def _resolve_object_route(
        self,
        anchor: SemanticNode,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> Optional[NavTarget]:
        # P3: Prefer ApproachPoint when available
        ap_id = anchor.attributes.get("approach_point_id")
        if ap_id is not None and position is not None:
            ap_node = None
            # Search ApproachPoints directly is not possible without topo_map ref.
            # We'll rely on the attribute: best_approach_position
            approach_pos = self._best_approach_position(anchor)
            if approach_pos is not None and position is not None:
                if float(np.linalg.norm(position - approach_pos)) > 0.35:
                    return NavTarget(
                        approach_pos.copy(),
                        anchor.node_id,
                        "object_approach",
                        "goal_approach_via_approach_point",
                        NavPhase.LOCAL_VISUAL_APPROACH,
                    )
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
        # P3: Prefer ApproachPoint node when available
        ap_id = node.attributes.get("approach_point_id")
        if ap_id is not None:
            tnode = node
            # Look up the actual node through the graph context
            pos = node.attributes.get("best_approach_position")
            if pos is not None:
                return np.array(pos, dtype=np.float32)
        wp_pos = node.attributes.get("anchor_waypoint_position") or node.attributes.get("best_approach_position")
        if wp_pos is None:
            return node.position.copy()
        return np.array(wp_pos, dtype=np.float32)
        return np.array(pos, dtype=np.float32)


    def _compute_reachability(self, candidates, position):
        if not candidates or position is None:
            return [0.0] * len(candidates), [1.0] * len(candidates)
        reach = []
        cost = []
        for node in candidates:
            d = float(np.linalg.norm(node.position - position))
            # Simple distance-based proxy for reachability
            reach.append(1.0)  # Default: reachable
            cost.append(min(d / 20.0, 1.0))  # Normalized path cost proxy
        return reach, cost
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
        hypothesis_positions: List[np.ndarray] = []
        if goal:
            hypothesis_positions = goal.active_hypothesis_positions
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
            # P12: task_score influences frontier value
            if goal and goal.target_object:
                ts = compute_task_score(
                    target_object=goal.target_object or "",
                    observed_label=node.label or "",
                    observed_room=node.attributes.get("room_context", ""),
                    room_prior=goal.room_prior,
                    landmarks=goal.landmark_prior,
                )
                sem_score += ts * 2.0
            for ctx_pos in context_nodes:
                ctx_dist = float(np.linalg.norm(node.position - ctx_pos))
                if ctx_dist < 8.0:
                    sem_score += 2.5 * max(0.0, 1.0 - ctx_dist / 8.0)
            no_goal_anchor = goal is not None and not _goal_has_active_anchor(topo_map, goal, self.cfg)
            room_boost = (
                self.cfg.frontier_room_prior_boost_no_anchor
                if no_goal_anchor
                else self.cfg.frontier_room_prior_boost
            )
            for rp_pos in room_prior_positions:
                rp_dist = float(np.linalg.norm(node.position - rp_pos))
                if rp_dist < 12.0:
                    sem_score += room_boost * max(0.0, 1.0 - rp_dist / 12.0)
            for hyp_pos in hypothesis_positions:
                hyp_dist = float(np.linalg.norm(node.position - hyp_pos))
                if hyp_dist < 8.0:
                    sem_score += 2.0 * max(0.0, 1.0 - hyp_dist / 8.0)
            if goal is not None and not self._goal_has_any_anchor(topo_map, goal):
                if structure.position is not None:
                    struct_dist = float(np.linalg.norm(node.position - structure.position))
                    sem_score += 2.0 * max(0.0, 1.0 - struct_dist / 8.0)
                # Persist frontier semantic value
                node.attributes["frontier_semantic_value"] = float(sem_score)
            # P10: Persist frontier semantic value
            node.attributes["frontier_semantic_value"] = float(sem_score)
            # P11: Check if navigation edge to this frontier is blocked
            anchor_wp = node.attributes.get("anchor_waypoint_id")
            if anchor_wp and topo_map.graph.has_edge(anchor_wp, node.node_id):
                edge_data = topo_map.graph.edges[anchor_wp, node.node_id]
                trav = float(edge_data.get("traversability", 1.0))
                blocked_count = int(edge_data.get("blocked_count", 0))
                if trav < 0.2 or blocked_count > 3:
                    continue  # Skip blocked frontier
                if blocked_count > 1:
                    sem_score -= blocked_count * 0.5  # Penalize frequently blocked
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

    # ============ P12: Proposal pipeline ============

    def generate_goal_proposals(
        self,
        candidates: List[SemanticNode],
        scores: List[float],
        goal_labels: Set[str],
    ) -> List[GoalProposal]:
        """Wrap planner candidates + scores into GoalProposal objects."""
        proposals = []
        for idx, node in enumerate(candidates):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            can_stop = node.node_type in (
                NodeType.OBJECT,
                NodeType.WAYPOINT_APPROACH,
            ) and _label_matches(node.label, goal_labels)
            proposals.append(GoalProposal(
                goal_id=str(goal_labels),
                candidate_node_id=node.node_id,
                candidate_type=node.node_type.value,
                target_position=node.position.copy(),
                score=score,
                source="navigation_planner",
                can_stop=can_stop,
                requires_verification=not can_stop,
            ))
        return proposals

    def score_goal_proposals(
        self,
        proposals: List[GoalProposal],
        goal: GoalManager,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray] = None,
    ) -> List[GoalProposal]:
        """Unify task-aware, confidence-aware proposal scoring."""
        for p in proposals:
            node = topo_map.get_node(p.candidate_node_id)
            if node is not None:
                relation_labels = self._map_relation_labels(topo_map, node.node_id)
                observed_attrs = node.attributes.get("object_attributes", {})
                observed_relations = list(node.attributes.get("spatial_relation", []))
                observed_relations.extend(relation_labels)
                p.semantic_score = max(float(p.semantic_score), float(node.confidence))
                p.attribute_score = AttributeMatcher.score(
                    getattr(goal.current_goal, "attributes", []), observed_attrs,
                )
                p.relation_score = RelationScorer.score(
                    goal_relations=getattr(goal.current_goal, "relations", []),
                    goal_landmarks=goal.landmark_prior,
                    observed_relations=observed_relations,
                    observed_room=node.attributes.get("room_context", ""),
                    map_relation_labels=relation_labels,
                    room_prior=goal.room_prior,
                )
                p.task_score = compute_task_score(
                    target_object=goal.target_object or "",
                    attributes=getattr(goal.current_goal, "attributes", []),
                    relations=getattr(goal.current_goal, "relations", []),
                    observed_label=node.label,
                    observed_attributes=observed_attrs,
                    observed_room=node.attributes.get("room_context", ""),
                    observed_relations=observed_relations,
                    map_relation_labels=relation_labels,
                    room_prior=goal.room_prior,
                    landmarks=goal.landmark_prior,
                    history_success_count=int(node.attributes.get("history_success_count", 0)),
                    history_fail_count=int(node.attributes.get("failed_approach_count", 0)),
                )
                p.negative_evidence = float(node.attributes.get("negative_evidence", 0.0))
                p.risk_penalty += p.negative_evidence
                p.risk_penalty += float(node.attributes.get("strong_negative_evidence", 0.0))
            elif p.source in ("clip", "vlm_weak", "detector_weak", "hypothesis"):
                p.semantic_score = max(float(p.semantic_score), float(p.score))
                p.task_score = max(float(p.task_score), float(p.score))

            if position is not None and p.target_position is not None:
                dist = float(np.linalg.norm(p.target_position - position))
                p.distance_cost = min(dist / 20.0, 1.0)
                p.reachability_score = max(float(p.reachability_score), 1.0 - p.distance_cost)

            reuse_bonus = 0.0
            if goal.is_repeated_goal:
                reuse = goal.reuse_debug or {}
                if p.candidate_node_id in reuse.get("unfailed_anchors", []):
                    reuse_bonus = 1.0
                elif p.candidate_node_id in reuse.get("failed_active_anchors", []):
                    reuse_bonus = -0.5
            p.history_bonus = reuse_bonus

            if p.source in ("clip", "vlm_weak", "detector_weak", "hypothesis"):
                p.can_stop = False
                p.requires_verification = True
                p.score = min(float(p.score), CLIP_PROPOSAL_SCORE_CAP)
                continue

            if p.source == "object_memory":
                p.score = (
                    0.35 * p.semantic_score
                    + 0.25 * p.task_score
                    + 0.15 * p.attribute_score
                    + 0.15 * p.relation_score
                    + 0.15 * p.reachability_score
                    + p.history_bonus
                    - 0.20 * p.distance_cost
                    - 0.40 * p.risk_penalty
                )
                p.can_stop = p.can_stop and not p.requires_verification
                if p.can_stop:
                    p.score = max(float(p.score), CLIP_PROPOSAL_SCORE_CAP + 0.05)
            elif p.source == "semantic_memory":
                p.can_stop = False
                p.requires_verification = True
                node = topo_map.get_node(p.candidate_node_id)
                if node is not None and node.attributes.get("cross_goal_preserved"):
                    p.history_bonus += float(self.cfg.cross_goal_semantic_bonus)
                ctype = p.candidate_type or ""
                is_room = ctype in ("room", "room_summary", "goal_region")
                room_bonus = float(self.cfg.semantic_room_bonus) if is_room else 0.0
                p.score = (
                    0.30 * p.semantic_score
                    + 0.35 * p.task_score
                    + 0.20 * p.relation_score
                    + 0.15 * p.reachability_score
                    + p.history_bonus
                    + room_bonus
                    - 0.10 * p.distance_cost
                    - 0.15 * p.risk_penalty
                )
            elif p.source in ("frontier", "room", "structure", "visited_fallback"):
                p.can_stop = False
                p.requires_verification = True
                p.score = (
                    0.25 * p.semantic_score
                    + 0.25 * p.task_score
                    + 0.20 * p.relation_score
                    + 0.20 * p.frontier_value
                    + 0.20 * p.reachability_score
                    + p.history_bonus
                    - 0.15 * p.distance_cost
                    - 0.30 * p.risk_penalty
                )
            else:
                p.score = float(p.score) + 0.3 * float(p.semantic_score) + reuse_bonus
        return sorted(proposals, key=lambda p: p.score, reverse=True)

    def select_best_proposal(
        self, proposals: List[GoalProposal]
    ) -> Optional[GoalProposal]:
        active = [p for p in proposals if p.status in ("active", "needs_verify", "confirmed", "preserved")]
        if not active:
            return None
        # object_memory completes the goal — hard priority.
        # semantic_memory and frontier compete on score (no hard gate).
        object_proposals = [p for p in active if p.source == "object_memory"]
        if object_proposals:
            return max(object_proposals, key=lambda p: p.score)
        return max(active, key=lambda p: p.score)

    def proposal_to_nav_target(
        self,
        proposal: GoalProposal,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> NavTarget:
        node = topo_map.get_node(proposal.candidate_node_id)
        if proposal.source == "object_memory" and node is not None:
            route = self._resolve_object_route(node, position, cur_vp_id)
            if route is not None:
                return route
        reason = f"proposal_{proposal.source}"
        expected = NavPhase.GLOBAL_SEARCH
        if proposal.candidate_type in ("object_anchor", "object_approach"):
            expected = NavPhase.LOCAL_VISUAL_APPROACH
        elif proposal.candidate_type in (
            "room", "landmark", "room_summary", "object_summary",
            "goal_region", "context_object",
        ):
            expected = NavPhase.ROUTE_TO_STRUCTURE
        elif proposal.source in ("hypothesis", "clip", "vlm_weak", "detector_weak"):
            reason = "hypothesis_verify_explore"
        elif proposal.source == "semantic_memory":
            reason = "cross_goal_semantic_reuse"
        elif proposal.candidate_type == "frontier":
            reason = "search_frontier"
        elif proposal.candidate_type == "visited":
            reason = "visited_fallback"
        return NavTarget(
            proposal.target_position.copy() if proposal.target_position is not None else None,
            proposal.candidate_node_id if node is not None else proposal.anchor_node_id,
            proposal.candidate_type,
            reason,
            expected,
        )

    def _map_relation_labels(self, topo_map: DynamicTopoMap, node_id: str) -> List[str]:
        if node_id not in topo_map.graph:
            return []
        labels: List[str] = []
        for neighbor_id in topo_map.graph.neighbors(node_id):
            neighbor = topo_map.get_node(neighbor_id)
            if neighbor is None:
                continue
            edge = topo_map.graph.edges[node_id, neighbor_id]
            edge_type = str(edge.get("edge_type", ""))
            relation = str(edge.get("relation", ""))
            if edge_type in (EdgeType.NEAR.value, EdgeType.IN_ROOM.value):
                labels.append(neighbor.label)
                labels.append(f"{edge_type}:{neighbor.label}")
            if relation:
                labels.append(relation)
        return [x for x in labels if x]


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
    ) -> Dict[str, Any]:
        visible_obs = self._goal_observations(packet.report.objects, goal)
        best = max(visible_obs, key=lambda o: _bbox_area(o.bbox), default=None)
        goal_visible = packet.report.goal_visible or best is not None
        # ── signal separation ──
        # raw_bbox_area  = real detection bbox area (0 if VLM gave no bbox)
        # raw_range_bin  = VLM distance semantics ("close".."far")
        # effective_bbox = display-only, never used for stop evidence
        raw_bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        raw_range_bin = str(best.range_bin).lower() if best is not None else "unknown"
        bearing = (
            packet.report.target_direction
            if packet.report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        )
        target_visibility = packet.report.target_visibility
        relative_progress = packet.report.relative_progress
        recommended_action = packet.report.recommended_action

        # last_fresh_bbox only from real bbox on fresh calls
        if packet.fresh_vlm and raw_bbox_area > 0.0:
            self.state.last_fresh_bbox = raw_bbox_area
            if self.state.servo_entry_bbox_baseline <= 0.0:
                self.state.servo_entry_bbox_baseline = raw_bbox_area
        if packet.cached_vlm:
            effective_bbox = raw_bbox_area if raw_bbox_area > 0.0 else self.state.last_fresh_bbox
        else:
            effective_bbox = raw_bbox_area

        confirm = bool(goal_visible)
        aligned = bearing in _CENTER_BEARINGS
        visibility_clear = target_visibility == "clear"
        visual_stop_candidate = bool(
            packet.report.stop_candidate
            and confirm
            and aligned
            and visibility_clear
        )
        stop_pose = bool(
            visual_stop_candidate
            and relative_progress != "farther"
        )
        if packet.fresh_vlm:
            self._append(self.state.confirm_buffer, confirm)
            self._append(self.state.stop_buffer, stop_pose)
            self._append(self.state.centered_buffer, aligned)
            self._append(
                self.state.close_buffer,
                raw_range_bin in ("near", "very_near", "close")
                or raw_bbox_area >= self.cfg.bbox_min_stop,
            )
            if stop_pose:
                self.state.fresh_vlm_stop_step = packet.step_id
        elif packet.cached_vlm and raw_bbox_area >= self.cfg.bbox_min_approach:
            self._append(self.state.confirm_buffer, confirm)
            self._append(self.state.centered_buffer, aligned)
            self._append(
                self.state.close_buffer,
                raw_range_bin in ("near", "very_near", "close")
                or raw_bbox_area >= self.cfg.bbox_min_stop,
            )
        if confirm:
            self.state.lost_count = 0
            # best_bbox_area --- fresh VLM + real bbox only ---
            if packet.fresh_vlm and raw_bbox_area >= self.state.best_bbox_area and raw_bbox_area > 0.0:
                self.state.best_bbox_area = raw_bbox_area
                self.state.best_stop_pose = None if position is None else position.copy()
                self.state.peak_bearing = bearing
                self.state.aligned_at_peak = aligned
            # visual_advance_steps --- fresh + stop_candidate only ---
            if packet.fresh_vlm and packet.report.stop_candidate:
                self.state.visual_advance_steps += 1
        elif not packet.fresh_vlm and not packet.cached_vlm:
            # Blind step — no VLM data (cache disabled + trigger not fired).
            # Treat as lost to prevent servo from blindly advancing.
            self.state.lost_count += 1
        elif packet.fresh_vlm and raw_bbox_area < self.cfg.bbox_min_approach:
            self.state.lost_count += 1
        # bbox_history --- fresh real bbox only ---
        if packet.fresh_vlm and raw_bbox_area > 0.0:
            self.state.bbox_history.append(raw_bbox_area)
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
        retreating = self._is_retreating()
        plateau = self._bbox_plateau()
        growth = _bbox_growth_metrics(self.state, self.cfg)
        bbox_has_meaningful_growth = growth["effective_growth"] >= self.cfg.bbox_min_growth_ratio
        return {
            "goal_visible": goal_visible,
            "bbox_area": effective_bbox,
            "raw_bbox_area": raw_bbox_area,
            "raw_bbox_confidence": (str(best.bbox_confidence).lower() if best is not None else "unknown"),
            "effective_bbox_area": effective_bbox,
            "best_bbox_area": self.state.best_bbox_area,
            "bearing": bearing,
            "range_bin": raw_range_bin,
            "target_visibility": target_visibility,
            "relative_progress": relative_progress,
            "stop_candidate": packet.report.stop_candidate,
            "visual_stop_candidate": visual_stop_candidate,
            "recommended_action": recommended_action,
            "goal_match_confidence": packet.report.goal_match_confidence,
            "forward_count": self.state.forward_action_count,
            "approach_distance": self.state.approach_travel_distance,
            "bbox_plateau": plateau,
            "bbox_growth_ratio": growth["growth_ratio"],
            "bbox_recent_growth": growth["recent_growth"],
            "bbox_effective_growth": growth["effective_growth"],
            "bbox_has_meaningful_growth": bbox_has_meaningful_growth,
            "aligned_at_peak": self.state.aligned_at_peak,
            "anchor_distance": anchor_distance,
            "best_anchor_distance": (
                None if self.state.best_anchor_distance >= float("inf")
                else self.state.best_anchor_distance
            ),
            "fresh_vlm": packet.fresh_vlm,
            "retreating": retreating,
        }

    def _is_retreating(self) -> bool:
        """Detect a sustained visual retreat without using simulator GT."""
        hist = [v for v in self.state.bbox_history if v > 0.0]
        if self.state.best_bbox_area < self.cfg.bbox_min_stop or len(hist) < 3:
            return False
        recent = hist[-3:]
        if not all(recent[i + 1] < recent[i] for i in range(len(recent) - 1)):
            return False
        return recent[-1] <= self.state.best_bbox_area * (1.0 - self.cfg.visual_retreat_ratio)

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
            if self.state.best_bbox_area >= self.cfg.bbox_min_stop
            else max(3, self.cfg.servo_lost_count_max - 2)
        )
        if self.state.lost_count >= lost_limit:
            return ServoAction("fail_anchor", "target_lost", NavPhase.RECOVERY)
        bearing = evidence.get("bearing", "unknown")
        goal_visible = bool(evidence.get("goal_visible", False))
        bbox_area = float(evidence.get("effective_bbox_area", evidence.get("bbox_area", 0.0)))
        range_bin = str(evidence.get("range_bin", "unknown")).lower()
        recommended_action = str(evidence.get("recommended_action", "")).lower()
        desired_turn = None
        if bearing in ("left", "left_front", "front-left") or recommended_action == "turn_left":
            desired_turn = "turn_left"
        elif bearing in ("right", "right_front", "front-right") or recommended_action == "turn_right":
            desired_turn = "turn_right"
        if desired_turn is not None:
            previous_turn = self.state.last_alignment_action
            if previous_turn is not None and previous_turn != desired_turn:
                self.state.alignment_flip_count += 1
                self.state.alignment_flip_streak += 1
                if (
                    self.state.alignment_flip_streak >= 2
                    and goal_visible
                    and (bbox_area >= self.cfg.bbox_min_approach
                         or range_bin in ("near", "very_near", "close"))
                    and range_bin != "far"
                ):
                    # Two consecutive side crossings bracket the image center.
                    # Advance once instead of entering a left/right limit cycle.
                    self.state.alignment_break_count += 1
                    self.state.alignment_flip_streak = 0
                    self.state.last_alignment_action = None
                    self.state.forward_action_count += 1
                    return ServoAction(
                        "move_forward",
                        "servo_alignment_flip_advance",
                        NavPhase.LOCAL_VISUAL_APPROACH,
                    )
            elif previous_turn == desired_turn:
                self.state.alignment_flip_streak = 0
            self.state.last_alignment_action = desired_turn
            reason = "servo_align_left" if desired_turn == "turn_left" else "servo_align_right"
            return ServoAction(desired_turn, reason, NavPhase.LOCAL_VISUAL_APPROACH)
        self.state.last_alignment_action = None
        self.state.alignment_flip_streak = 0
        # ── VLM recommended_action ──
        # VLM is the semantic direction selector.  When it gives a clear
        # recommendation while aligned, follow it over the state machine.
        if recommended_action == "move_forward":
            self.state.forward_action_count += 1
            return ServoAction("move_forward", "vlm_forward", NavPhase.LOCAL_VISUAL_APPROACH)
        if recommended_action in ("hold_and_verify", "stop_candidate"):
            if self.state.plateau_forward_count < self.cfg.min_forward_after_plateau:
                self.state.plateau_forward_count += 1
                self.state.forward_action_count += 1
                return ServoAction("move_forward", "vlm_creep", NavPhase.LOCAL_VISUAL_APPROACH)
            return ServoAction("hold", "vlm_hold", NavPhase.STOP_VERIFY)
        # at_stop_band: we are visually close enough to consider stopping.
        # Uses real bbox peak OR fresh range_bin ("very_near" / "close")
        # as independent signals.  Cached data does NOT qualify.
        peak_stop = (
            (
                bool(evidence.get("bbox_has_meaningful_growth", False))
                or bool(evidence.get("bbox_plateau", False))
            )
            and evidence.get("raw_bbox_confidence", "medium") != "low"
        ) or (
            self.state.best_bbox_area >= self.cfg.bbox_min_stop
            and evidence.get("raw_bbox_confidence", "medium") != "low"
        )
        range_stop = evidence.get("fresh_vlm", False) and range_bin in ("very_near", "close")
        at_stop_band = peak_stop or range_stop
        peak_target = max(
            self.cfg.bbox_min_approach,
            self.state.best_bbox_area * self.cfg.bbox_peak_ratio,
        )
        below_peak = bbox_area < peak_target
        # When range_stop is the driving signal (no real bbox peak),
        # below_peak is irrelevant — VLM says we are close enough.
        if range_stop and not peak_stop:
            below_peak = False
        not_enough_forward = self.state.forward_action_count < self.cfg.min_forward_before_stop
        not_enough_distance = self.state.approach_travel_distance < self.cfg.min_approach_distance
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
        # range_stop bypasses the bbox-plateau dead-end:
        # if VLM says "very_near"/"close" but we have no real bbox peak,
        # skip the "plateau below stop bbox" path and fall through to stop.
        if self.state.best_bbox_area < self.cfg.bbox_min_stop:
            if range_stop or bool(evidence.get("bbox_has_meaningful_growth", False)):
                pass  # fall through to plateau creep → hold → STOP_VERIFY
            else:
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
        visible_count = sum(self.state.confirm_buffer[-self.cfg.stop_buffer_size:])
        centered_count = sum(self.state.centered_buffer[-self.cfg.stop_buffer_size:])
        close_count = sum(self.state.close_buffer[-self.cfg.stop_buffer_size:])
        layer1 = visible_count >= self.cfg.servo_entry_evidence
        approach_enough = (
            self.state.forward_action_count >= self.cfg.min_forward_before_stop
            and self.state.approach_travel_distance >= self.cfg.min_approach_distance
        )
        relative_progress = str(
            (servo_evidence or {}).get("relative_progress", "uncertain")
        ).lower()
        anchor_distance = (servo_evidence or {}).get("anchor_distance")
        near_anchor = (
            anchor_distance is None
            or float(anchor_distance) <= self.cfg.anchor_servo_enter_distance
        )
        raw_bbox_close = self.state.best_bbox_area >= self.cfg.bbox_min_stop
        range_near_stop = bool(
            (servo_evidence or {}).get("range_bin", "unknown") in ("very_near", "close")
        )
        plateau_ok = bool((servo_evidence or {}).get("bbox_plateau", False))
        growth_ok = bool((servo_evidence or {}).get("bbox_has_meaningful_growth", False))
        bbox_close_enough = (
            growth_ok
            or plateau_ok
            or range_near_stop
            or raw_bbox_close
        )
        bbox_growth_ok = bbox_close_enough
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
            or (bbox_close_enough and plateau_ok)
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
        best_fresh = max(goal_obs, key=lambda o: _bbox_area(o.bbox), default=None)
        fresh_raw_bbox_area = _bbox_area(best_fresh.bbox) if best_fresh is not None else 0.0
        fresh_range_bin = str(best_fresh.range_bin).lower() if best_fresh is not None else "unknown"
        fresh_bbox_confidence = str(best_fresh.bbox_confidence).lower() if best_fresh is not None else "unknown"
        fresh_bearing = str(
            packet.report.target_direction
            if packet.report.target_direction != "unknown"
            else (best_fresh.bearing if best_fresh is not None else "unknown")
        ).lower()
        current_centered = fresh_bearing in _CENTER_BEARINGS
        fresh_visibility_clear = packet.report.target_visibility == "clear"
        progress_ok = relative_progress != "farther"
        # P10: task_score influences stop — attribute match matters
        best_task_obs = max(
            (o for o in packet.report.objects
             if _label_matches(o.label, goal.target_labels)),
            key=lambda o: _bbox_area(o.bbox) + float(o.confidence),
            default=None,
        )
        if best_task_obs is not None:
            obs_attrs = dict(best_task_obs.attributes or {})
        else:
            obs_attrs = {}
        task_score_stop = compute_task_score(
            target_object=goal.target_object or "",
            attributes=getattr(goal.current_goal, "attributes", []),
            relations=getattr(goal.current_goal, "relations", []),
            observed_label=best_task_obs.label if best_task_obs else "",
            observed_attributes=obs_attrs,
            observed_relations=list(getattr(best_task_obs, "spatial_relation", []) or []) if best_task_obs else [],
            room_prior=goal.room_prior,
            landmarks=goal.landmark_prior,
        ) if best_task_obs else 0.0

        short_approach = (
            self.state.approach_travel_distance < self.cfg.stop_short_approach_max
        )
        fresh_stop_ok = (
            packet.fresh_vlm
            and packet.report.goal_visible
            and task_score_stop > 0.3
            and fresh_raw_bbox_area >= self.cfg.stop_min_fresh_bbox
            and (
                packet.report.stop_candidate
                or fresh_range_bin in ("very_near", "close")
                or bool((servo_evidence or {}).get("bbox_has_meaningful_growth", False))
                or bool((servo_evidence or {}).get("bbox_plateau", False))
                or (
                    fresh_raw_bbox_area >= self.cfg.bbox_min_stop
                    and fresh_bbox_confidence in ("medium", "high", "unknown")
                )
            )
            and (not short_approach or packet.report.stop_candidate)
            and current_centered
            and fresh_visibility_clear
            and progress_ok
        )
        layer3 = (
            multi_angle_ok
            and near_peak
            and near_anchor
            and fresh_stop_ok
            and centered_count >= self.cfg.servo_entry_evidence
            and close_count >= 1
            and stop_votes >= self.cfg.servo_entry_evidence
        )
        should = bool(layer1 and layer2 and layer3 and not retreating)
        reason = "visual_confirmed_stop" if should else self._reason(
            layer1, layer2, layer3, multi_angle_ok, fresh_stop_ok,
            near_peak, near_anchor, retreating, progress_ok,
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
        near_anchor: bool,
        retreating: bool,
        progress_ok: bool,
    ) -> str:
        if retreating:
            return "layer2_retreating_from_target"
        if not layer1:
            return "layer1_target_not_confirmed"
        if not layer2:
            return "layer2_approach_incomplete"
        if not multi_angle_ok:
            return "layer3_multi_angle_not_verified"
        if not progress_ok:
            return "layer3_target_getting_farther"
        if not fresh_stop_ok:
            return "layer3_fresh_vlm_not_confirmed"
        if not near_peak:
            return "layer3_stop_pose_not_ok"
        if not near_anchor:
            return "layer3_not_near_anchor"
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
        if self.recent_positions:
            step_move = float(np.linalg.norm(position - self.recent_positions[-1]))
            if self.recovery_reason == "no_progress" and step_move >= self.cfg.no_progress_distance:
                self.recovery_reason = ""
                self.recent_positions = [position.copy()]
                return False
        self.recent_positions.append(position.copy())
        if len(self.recent_positions) > self.cfg.no_progress_window:
            self.recent_positions = self.recent_positions[-self.cfg.no_progress_window:]
        if len(self.recent_positions) < self.cfg.no_progress_window:
            return False
        moved = float(np.linalg.norm(self.recent_positions[-1] - self.recent_positions[0]))
        if moved >= self.cfg.no_progress_distance:
            if self.recovery_reason == "no_progress":
                self.recovery_reason = ""
            return False
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
        elif event in ("unreachable", "snap_failed", "not_navigable"):
            node.attributes["unreachable_count"] = int(node.attributes.get("unreachable_count", 0)) + 1
            node.attributes["last_unreachable_step"] = topo_map.current_step
            node.attributes["last_block_reason"] = event
            if node.node_type == NodeType.WAYPOINT_FRONTIER:
                node.attributes["consumed"] = True
                out["action"] = "consumed_frontier"
            else:
                node.attributes["blacklisted_until"] = topo_map.current_step + self.cfg.anchor_blacklist_steps
                out["action"] = "blacklisted_target"
            self.recovery_reason = event
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
                    "centered_buffer": list(servo_state.centered_buffer),
                "close_buffer": list(servo_state.close_buffer),
                "track_buffer": list(servo_state.track_buffer),
            },
            "bbox_area": servo_evidence.get("bbox_area", 0.0),
            "effective_bbox_area": servo_evidence.get("effective_bbox_area", servo_evidence.get("bbox_area", 0.0)),
            "best_bbox_area": servo_state.best_bbox_area,
            "visual_advance_steps": servo_state.visual_advance_steps,
            "forward_action_count": servo_state.forward_action_count,
            "alignment_flip_count": servo_state.alignment_flip_count,
            "alignment_flip_streak": servo_state.alignment_flip_streak,
            "alignment_break_count": servo_state.alignment_break_count,
            "plateau_forward_count": servo_state.plateau_forward_count,
            "anchor_distance": servo_evidence.get("anchor_distance"),
            "best_anchor_distance": servo_evidence.get("best_anchor_distance"),
            "retreating": servo_evidence.get("retreating", False),
            "approach_travel_distance": round(servo_state.approach_travel_distance, 4),
            "bbox_plateau": servo_evidence.get("bbox_plateau", False),
            "aligned_at_peak": servo_state.aligned_at_peak,
            "peak_bearing": servo_state.peak_bearing,
            "bearing": servo_evidence.get("bearing", "unknown"),
            "range_bin": servo_evidence.get("range_bin", "unknown"),
            "target_visibility": servo_evidence.get("target_visibility", "not_visible"),
            "relative_progress": servo_evidence.get("relative_progress", "uncertain"),
            "vlm_stop_candidate": servo_evidence.get("stop_candidate", False),
            "visual_stop_candidate": servo_evidence.get("visual_stop_candidate", False),
            "recommended_action": servo_evidence.get("recommended_action", "search"),
            "goal_match_confidence": servo_evidence.get("goal_match_confidence", 0.0),
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
                        "hypothesis_pool": {
                            "active_count": len(
                                getattr(packet, "_pool_hypotheses", [])
                            ),
                        },
            "goal_reuse": goal_debug,
        }


class GoatAgent(ConfTopoBaseAgent):
    def __init__(self, config: Optional[ConfTopoConfig] = None, new_config: Optional[NewGoatConfig] = None):
        super().__init__(config)
        self.new_config = new_config or NewGoatConfig()
        self.goal_manager = GoalManager()
        self.perception_manager = PerceptionManager(self.config, self.new_config)
        self.memory_writer = MemoryWriter(self.new_config)
        self.structure_planner = StructurePlanner(new_config)
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
        self._last_goal_proposals: List[GoalProposal] = []
        self._last_selected_goal_proposal: Optional[GoalProposal] = None
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
        self._target_object_detected_this_scan: bool = False
        self._goal_enter_step: int = -1
        self._recent_actions: List[str] = []
        self._environment_landmark_labels: Set[str] = set()
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

    def set_environment_landmark_labels(self, labels: List[str]) -> None:
        """Mark landmark labels that are scene-level anchors rather than goal hints."""
        self._environment_landmark_labels = {str(label) for label in labels}

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
        self._goal_enter_step = -1
        self._recent_actions = []

    def _target_output_for_node(self, node: SemanticNode) -> Tuple[np.ndarray, Dict[str, Any]]:
        target_pos = node.position.copy()
        extras: Dict[str, Any] = {}

        # P3: WAYPOINT_APPROACH nodes route directly to their position
        if node.node_type == NodeType.WAYPOINT_APPROACH:
            target_pos = node.position.copy()
            extras["target_type"] = "approach_point"
            extras["source_object_id"] = node.attributes.get("source_object_id")
            return target_pos, extras

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

    def set_pathfinder(self, sim) -> None:
        """Inject simulator pathfinder for frontier navmesh validation.

        Call after ``make_sim()`` so that frontiers are only created at
        navigable positions.
        """
        pf = getattr(sim, "pathfinder", None)
        if pf is None:
            return

        def _snap(pos: np.ndarray) -> Optional[np.ndarray]:
            try:
                snapped = np.asarray(pf.snap_point(pos), dtype=np.float32)
            except Exception:
                return None
            if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
                return None
            if float(np.linalg.norm(snapped[[0, 2]] - pos[[0, 2]])) > 0.75:
                return None
            try:
                if not pf.is_navigable(snapped):
                    return None
            except Exception:
                return None
            return snapped

        self.memory_writer.set_navmesh_snap(_snap)

    def set_new_goal(self, goal: GoalNode):
        goal = normalize_goal_node(goal)
        old_map_nodes = self.topo_map.num_nodes
        self._mark_cross_goal_preserved()
        self.reset_keep_memory()
        runtime_room_prior: List[str] = []
        runtime_landmarks: List[str] = []
        if not getattr(goal, "room_prior", None):
            rp = _category_priors(goal.target_object, OBJECT_CATEGORY_ROOM_PRIORS)
            if rp:
                runtime_room_prior = rp
        if not getattr(goal, "landmarks", None):
            lp = _category_priors(goal.target_object, OBJECT_CATEGORY_LANDMARK_PRIORS)
            if lp:
                runtime_landmarks = lp
        if self.instruction_graph is None:
            self.instruction_graph = InstructionGraph(goal_type="object_goal", goal_nodes=[goal])
        else:
            self.instruction_graph.set_current_goal(goal)
        self.goal_manager.set_new_goal(goal, self.topo_map, runtime_room_prior, runtime_landmarks)
        self.perception_manager.set_goal(goal, landmarks=self.goal_manager.landmark_prior)
        self.perception_manager.reset_short_term()
        self.memory_writer.reset_goal()
        self.local_servo.reset()
        self.recovery_manager.reset()
        self._recent_actions = []
        self.nav_phase = NavPhase.GLOBAL_SEARCH
        self.phase_enter_step = self.topo_map.current_step
        self._goal_enter_step = self.topo_map.current_step
        self.phase_transition_reason = f"new_goal_keep_memory:{old_map_nodes}"
        self._goals_completed += 1

    def _mark_cross_goal_preserved(self) -> None:
        """Protect nodes from pruning across goals.

        Cross-goal preservation means the memory trace survives a goal switch;
        it does not require the object to stay in detailed active form.
        DynamicTopoMap.adaptive_granularity decides whether preserved objects
        remain detailed or fold into landmark / room summaries.

        Preserves:
        - OBJECT (object_anchor, context_object, target_relevance > 0)
        - All ROOM nodes
        - All LANDMARK nodes
        - WAYPOINT_APPROACH (linked to preserved OBJECTs)
        - OBJECT_SUMMARY (linked to preserved ROOMs)
        - GOAL_REGION (linked to preserved ROOMs)
        """
        preserved_object_ids = set()
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            role = node.attributes.get("semantic_role")
            if role in ("object_anchor", "context_object"):
                node.attributes["cross_goal_preserved"] = True
                node.attributes["long_term_retained"] = True
                node.attributes.setdefault("allow_fold", role != "object_anchor")
                preserved_object_ids.add(node.node_id)
            elif float(node.attributes.get("target_relevance", 0.0)) > 0:
                node.attributes["cross_goal_preserved"] = True
                node.attributes["long_term_retained"] = True
                node.attributes.setdefault("allow_fold", True)
                preserved_object_ids.add(node.node_id)
            elif float(node.attributes.get("strong_negative_evidence", 0.0)) <= 0.3:
                # Objects without strong rejection also preserved
                node.attributes["cross_goal_preserved"] = True
                node.attributes["long_term_retained"] = True
                node.attributes.setdefault("allow_fold", True)
                preserved_object_ids.add(node.node_id)

        for node in self.topo_map.get_nodes_by_type(NodeType.ROOM):
            node.attributes["cross_goal_preserved"] = True
        preserved_room_ids = {n.node_id for n in self.topo_map.get_nodes_by_type(NodeType.ROOM)}

        for node in self.topo_map.get_nodes_by_type(NodeType.LANDMARK):
            node.attributes["cross_goal_preserved"] = True

        for node in self.topo_map.get_nodes_by_type(NodeType.GOAL_REGION):
            node.attributes["cross_goal_preserved"] = True

        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT_SUMMARY):
            node.attributes["cross_goal_preserved"] = True

        # P11: Preserve ApproachPoints linked to preserved OBJECTs
        for node in self.topo_map.get_nodes_by_type(NodeType.WAYPOINT_APPROACH):
            src_id = node.attributes.get("source_object_id")
            if src_id in preserved_object_ids:
                node.attributes["cross_goal_preserved"] = True

        # P11: Preserve ObjectSummary linked to preserved ROOMs
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT_SUMMARY):
            room_id = node.attributes.get("room_node_id")
            if room_id in preserved_room_ids:
                node.attributes["cross_goal_preserved"] = True

        # P11: Preserve GoalRegion linked to preserved ROOMs
        for node in self.topo_map.get_nodes_by_type(NodeType.GOAL_REGION):
            room_id = node.attributes.get("room_node_id")
            if room_id in preserved_room_ids:
                node.attributes["cross_goal_preserved"] = True

    def observe(self, obs: Dict[str, Any]) -> None:
        self._cur_rgb = obs.get("rgb")
        self._cur_rgb_embed = None
        self._cur_image_goal_embed = None
        if obs.get("rgb_embed") is not None:
            self._cur_rgb_embed = np.asarray(obs["rgb_embed"], dtype=np.float32)
        if obs.get("image_goal_embed") is not None:
            self._cur_image_goal_embed = np.asarray(obs["image_goal_embed"], dtype=np.float32)
        pos = obs.get("position")
        if pos is not None:
            world = np.asarray(pos, dtype=np.float32)
            if self._origin_position is None:
                self._origin_position = world.copy()
            self._position = world - self._origin_position
        self._heading = float(obs.get("heading", 0.0))
        self._last_packet = self.perception_manager.run_light(
            self._cur_rgb_embed,
            self.topo_map.current_step,
            goal_visual_embed=self._cur_image_goal_embed,
        )

    def update_memory(self) -> None:
        if self.recovery_manager.note_position(self._position):
            if self.nav_phase not in (NavPhase.TRACK_TARGET, NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY):
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
            recent_actions=self._recent_actions,
        )
        self._last_packet = packet

        # ── P0: Pass hypothesis positions to planner ──
        self.goal_manager._active_hypothesis_positions = [
            h.position for h in self.perception_manager.hypothesis_pool.get_active(
                goal_id=self.goal_manager.goal_key, include_needs_verify=True,
            )
            if h.position is not None
        ]
        if packet.fresh_vlm:
            self._heavy_perception_calls += 1
        self._target_object_detected_this_scan = False
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
        self._target_object_detected_this_scan = bool(
            self._last_memory.goal_anchor_written
            or any(
                _goal_label_matches(o.label, self.goal_manager)
                for o in packet.report.objects
            )
        )
        # Promote/reject hypotheses after confirmed memory writes are known.
        self._update_hypotheses_from_memory()
        self._object_merge_count += self._last_memory.object_merge_count_this_step
        self.topo_map.decay_all_confidences()
        # ── P0: Decay hypothesis pool ──
        self.perception_manager.hypothesis_pool.decay(self.topo_map.current_step)
        self.topo_map.merge_nearby_nodes(NodeType.WAYPOINT_FRONTIER)
        self.topo_map.adaptive_granularity(self._position)
        self.topo_map.prune_low_confidence(self._position)

    def _collect_hypothesis_goal_proposals(self) -> List[GoalProposal]:
        proposals: List[GoalProposal] = []
        for hyp in self.perception_manager.hypothesis_pool.get_active(
            goal_id=self.goal_manager.goal_key,
            include_needs_verify=True,
        ):
            if hyp.position is None or hyp.source not in ("clip", "vlm_weak", "detector_weak"):
                continue
            score = min(float(hyp.confidence), CLIP_PROPOSAL_SCORE_CAP)
            proposals.append(GoalProposal(
                goal_id=self.goal_manager.goal_key,
                candidate_node_id=hyp.id,
                candidate_type=hyp.kind,
                anchor_node_id=hyp.anchor_node_id,
                target_position=hyp.position.copy(),
                score=score,
                semantic_score=score,
                task_score=score,
                source=hyp.source,
                status=hyp.status,
                can_stop=False,
                requires_verification=True,
                evidence_refs=[{
                    "hypothesis_id": hyp.id,
                    "trigger_reason": hyp.trigger_reason,
                    "seen_count": hyp.seen_count,
                    "weak_bbox": hyp.weak_bbox,
                    "weak_relations": list(hyp.weak_relations),
                }],
            ))
        return proposals

    def _update_track_buffer(self) -> None:
        packet = self._last_packet
        if not packet.fresh_vlm:
            return
        if (
            self.servo_state.track_buffer
            and int(self.servo_state.track_buffer[-1].get("step_id", -1)) == int(packet.step_id)
        ):
            return
        best = max(
            (o for o in packet.report.objects
             if _label_matches(o.label, self.goal_manager.target_labels)),
            key=lambda o: _bbox_area(o.bbox) + float(o.confidence),
            default=None,
        )
        visible = bool(packet.report.goal_visible or best is not None)
        bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        bearing = (
            packet.report.target_direction
            if packet.report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        )
        range_bin = str(best.range_bin if best is not None else "unknown").lower()
        sample = {
            "step_id": int(packet.step_id),
            "visible": visible,
            "bbox_valid": bbox_area > 0.0,
            "confidence": float(best.confidence) if best is not None else float(packet.report.goal_match_confidence),
            "centered": str(bearing).lower() in _CENTER_BEARINGS,
            "bearing": str(bearing).lower(),
            "range_bin": range_bin,
        }
        self.servo_state.track_buffer.append(sample)
        if len(self.servo_state.track_buffer) > self.new_config.track_buffer_size:
            del self.servo_state.track_buffer[:-self.new_config.track_buffer_size]

    def _target_seen_recently(self) -> bool:
        window = self.servo_state.track_buffer[-self.new_config.track_buffer_size:]
        return sum(bool(x.get("visible")) for x in window) >= self.new_config.track_visible_min

    def _track_ready_for_approach(self) -> bool:
        window = self.servo_state.track_buffer[-self.new_config.track_buffer_size:]
        visible_count = sum(bool(x.get("visible")) for x in window)
        bbox_count = sum(bool(x.get("bbox_valid")) for x in window)
        return (
            visible_count >= self.new_config.track_visible_min
            and bbox_count >= self.new_config.track_bbox_min
        )

    def _track_scan_decision(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
        reason: str = "track_target_scan",
    ) -> PlanDecision:
        bearing = "unknown"
        for item in reversed(self.servo_state.track_buffer):
            if item.get("bearing") and item.get("bearing") != "unknown":
                bearing = str(item.get("bearing"))
                break
        if bearing in ("left", "left_front", "front-left"):
            action = "turn_left"
        elif bearing in ("right", "right_front", "front-right"):
            action = "turn_right"
        else:
            action = "turn_left" if self.topo_map.current_step % 2 == 0 else "turn_right"
        return self._decision(
            action,
            "track_target",
            None,
            nav_target.target_node_id or self.servo_state.active_anchor_id,
            "track_target",
            reason,
            candidate_ids,
            scores,
            NavPhase.TRACK_TARGET,
        )

    def _plan_track_phase(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
        self._update_track_buffer()
        if self._track_ready_for_approach():
            self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "track_target_confirmed")
            return self._plan_servo_phase(nav_target, candidate_ids, scores)
        return self._track_scan_decision(nav_target, candidate_ids, scores)

    def plan(self) -> PlanDecision:
        structure = self.structure_planner.select(self.topo_map, self.goal_manager, self._position)
        proposals = self.navigation_planner.collect_goal_proposals(
            self.topo_map, self.goal_manager, structure, self.nav_phase,
            self._position, self.memory_writer.cur_vp_id,
        )
        proposals.extend(self._collect_hypothesis_goal_proposals())
        scored_proposals = self.navigation_planner.score_goal_proposals(
            proposals, self.goal_manager, self.topo_map, self._position,
        )
        selected_proposal = self.navigation_planner.select_best_proposal(scored_proposals)
        if selected_proposal is None:
            nav_target = NavTarget(reason="no_navigation_candidates")
            candidate_ids: List[str] = []
            scores: List[float] = []
        else:
            nav_target = self.navigation_planner.proposal_to_nav_target(
                selected_proposal,
                self.topo_map,
                self._position,
                self.memory_writer.cur_vp_id,
            )
            ranked = scored_proposals[:10]
            candidate_ids = [p.candidate_node_id for p in ranked]
            scores = [float(p.score) for p in ranked]
        self._last_structure = structure
        self._last_nav_target = nav_target
        self._last_goal_proposals = scored_proposals
        self._last_selected_goal_proposal = selected_proposal
        if nav_target.reason == "visited_fallback":
            self._visited_fallback_count += 1
        elif nav_target.reason == "no_navigation_candidates":
            self._no_candidates_count += 1

        self._maybe_enter_servo_near_anchor(nav_target)

        if nav_target.reason == "at_anchor_waypoint_servo":
            anchor_id = nav_target.target_node_id
            if anchor_id and self.servo_state.active_anchor_id != anchor_id:
                self.local_servo.enter(anchor_id, self.topo_map.current_step)
            if self.nav_phase not in (NavPhase.TRACK_TARGET, NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY, NavPhase.STOP):
                self._transition(NavPhase.TRACK_TARGET, "at_anchor_waypoint_track")
            if self.nav_phase == NavPhase.TRACK_TARGET:
                return self._plan_track_phase(nav_target, candidate_ids, scores)
            servo_decision = self._plan_servo_phase(nav_target, candidate_ids, scores)
            if servo_decision is not None:
                return servo_decision

        phase_age = self.topo_map.current_step - self.phase_enter_step
        if phase_age > self.new_config.phase_timeout_steps:
            if self.nav_phase == NavPhase.ROUTE_TO_STRUCTURE:
                if (
                    nav_target.target_node_id
                    and nav_target.target_type in (
                        "room", "landmark", "room_summary", "object_summary",
                        "goal_region", "context_object",
                    )
                ):
                    sn = self.topo_map.get_node(nav_target.target_node_id)
                    if sn is not None:
                        sn.attributes["blacklisted_until"] = self.topo_map.current_step + self.new_config.anchor_blacklist_steps
                self._transition(NavPhase.GLOBAL_SEARCH, "route_to_structure_timeout")
            elif self.nav_phase == NavPhase.ROUTE_TO_OBJECT_ANCHOR:
                failed_anchor_id = (
                    nav_target.target_node_id
                    if nav_target.target_type in ("object_anchor", "object_approach")
                    else self.servo_state.active_anchor_id
                )
                self.recovery_manager.fail_anchor(
                    self.topo_map, failed_anchor_id, "route_timeout",
                )
                self._transition(NavPhase.RECOVERY, "route_to_anchor_timeout")
            elif self.nav_phase == NavPhase.STOP_VERIFY:
                self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "stop_verify_timeout")

        if self.nav_phase == NavPhase.TRACK_TARGET:
            return self._plan_track_phase(nav_target, candidate_ids, scores)

        if self.nav_phase in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY):
            servo_decision = self._plan_servo_phase(nav_target, candidate_ids, scores)
            if servo_decision is not None:
                return servo_decision

        if self.nav_phase == NavPhase.RECOVERY:
            self._transition(NavPhase.GLOBAL_SEARCH, self.recovery_manager.recovery_reason or "recovery_complete")

        if nav_target.target_node_id and nav_target.target_type == "object_anchor":
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "selected_object_anchor")
        elif nav_target.target_node_id and nav_target.target_type == "object_approach":
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "selected_object_approach")
        elif nav_target.target_node_id and nav_target.target_type in (
            "room", "landmark", "room_summary", "object_summary",
            "goal_region", "context_object",
        ):
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
        self._update_track_buffer()
        if self.nav_phase == NavPhase.LOCAL_VISUAL_APPROACH and not self._target_seen_recently():
            self._transition(NavPhase.TRACK_TARGET, "target_not_seen_recently")
            return self._track_scan_decision(
                nav_target, candidate_ids, scores, "track_target_reacquire",
            )

        servo_evidence = self.local_servo.update_evidence(
            self._last_packet,
            self.goal_manager,
            self._position,
            self._active_anchor_distance(),
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
            elif nav_target.target_node_id and nav_target.target_type in (
                "room", "landmark", "room_summary", "object_summary",
                "goal_region", "context_object",
            ):
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
            verify_action = (
                "turn_left"
                if self.topo_map.current_step % 2 == 0
                else "turn_right"
            )
            return self._decision(
                verify_action,
                "approach_confirm",
                None,
                None,
                "stop_verify_scan",
                "stop_verify_fresh_view",
                candidate_ids,
                scores,
            )
        return self._decision("move_forward", "approach_confirm", None, None, "servo_guard", "servo_phase_guard", candidate_ids, scores)

    def _maybe_enter_servo_near_anchor(self, nav_target: NavTarget) -> None:
        if self.nav_phase in (NavPhase.TRACK_TARGET, NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY, NavPhase.STOP):
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

        if self.servo_state.active_anchor_id != anchor.node_id:
            self.local_servo.enter(anchor.node_id, self.topo_map.current_step)
        self._transition(NavPhase.TRACK_TARGET, "near_anchor_track_target")

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
        )
        self._last_debug = debug
        self._recent_actions.append(plan_output.action)
        self._recent_actions = self._recent_actions[-5:]
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
            "target_object_detected_this_scan": self._target_object_detected_this_scan,
        }

    def on_navigation_event(self, target_node_id: Optional[str], event: str) -> Dict[str, Any]:
        out = self.recovery_manager.on_navigation_event(self.topo_map, target_node_id, event)
        self._last_navigation_event = out
        if out.get("action") == "object_anchor_reached":
            self.local_servo.enter(target_node_id, self.topo_map.current_step)
            self._transition(NavPhase.TRACK_TARGET, "object_anchor_reached_track")
        elif out.get("action") == "consumed_frontier":
            if target_node_id:
                self._consumed_frontier_ids.add(target_node_id)
            self._transition(NavPhase.GLOBAL_SEARCH, out.get("event", event))
        elif out.get("action") == "blacklisted_target":
            self._transition(NavPhase.RECOVERY, out.get("event", event))
        return out

    def _update_hypotheses_from_memory(self) -> None:
        packet = self._last_packet
        goal_key = self.goal_manager.goal_key
        if not goal_key or not packet.fresh_vlm:
            return
    
        promoted_hypothesis_ids = set()
        confirmed_labels = {
            str(obs.label).lower().strip()
            for obs in packet.report.objects
            if obs.label
        }
        for node_id in self._last_memory.written_node_ids:
            node = self.topo_map.get_node(node_id)
            if node is None or node.node_type != NodeType.OBJECT:
                continue
            promoted = self.perception_manager.hypothesis_pool.promote_by_goal_and_label(
                goal_id=goal_key,
                label=node.label,
                object_node_id=node_id,
            )
            if promoted is not None:
                promoted_hypothesis_ids.add(promoted.id)
    
        for hyp in self.perception_manager.hypothesis_pool.get_active(
            goal_id=goal_key, include_needs_verify=True,
        ):
            if hyp.id in promoted_hypothesis_ids or hyp.status != "needs_verify":
                continue
            if hyp.label not in confirmed_labels:
                self.perception_manager.hypothesis_pool.reject(hyp.id, "vlm_not_confirmed")

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
            "goal_proposals": [
                {
                    "node_id": p.candidate_node_id,
                    "type": p.candidate_type,
                    "score": p.score,
                    "source": p.source,
                    "semantic_score": p.semantic_score,
                    "task_score": p.task_score,
                    "attribute_score": p.attribute_score,
                    "relation_score": p.relation_score,
                    "reachability_score": p.reachability_score,
                    "can_stop": p.can_stop,
                    "requires_verification": p.requires_verification,
                }
                for p in self._last_goal_proposals
            ],
            "selected_goal_proposal": (
                {
                    "node_id": self._last_selected_goal_proposal.candidate_node_id,
                    "type": self._last_selected_goal_proposal.candidate_type,
                    "score": self._last_selected_goal_proposal.score,
                    "source": self._last_selected_goal_proposal.source,
                    "can_stop": self._last_selected_goal_proposal.can_stop,
                    "requires_verification": self._last_selected_goal_proposal.requires_verification,
                }
                if self._last_selected_goal_proposal is not None else None
            ),
            "mean_object_confidence": mean_conf,
            "nav_phase": self.nav_phase.value,
            "last_debug": self._last_debug,
                        "hypotheses": [
                            h for h in self.perception_manager.hypothesis_pool.to_debug_list(
                                goal_id=self.goal_manager.goal_key, limit=10,
                            )
                        ],
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
