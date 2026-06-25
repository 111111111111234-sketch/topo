"""Clean proposal-driven GOAT + ETP-lite navigation agent.

This variant keeps the generic base pieces from ``goat_agent_new`` but owns
its ETP-lite navigation implementation in this file, making the policy explicit:

    SEARCH -> ROUTE_TO_PROPOSAL -> TRACK_TARGET -> LOCAL_APPROACH
    -> VERIFY_STOP -> STOP

Global navigation is TopoMap proposal driven.  Local visual servo and stop
verification are only used after an object anchor has been routed to and
tracked.  Exploration adds three generic mechanisms missing from the first
ETP-lite smoke tests:

* dynamic ghost radius when goal evidence is stale,
* active scan after reaching a new waypoint,
* failed-region penalties after route/track/approach failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType, SemanticNode
from conftopo.core.instruction_graph import GoalProposal
from conftopo.agents.goat_agent_etpnav import ETPTaskTelemetry
from conftopo.agents.goat_agent_new import (
    _CENTER_BEARINGS,
    _STOP_RANGE_BINS,
    _bbox_area,
    GoatAgent,
    GoalManager,
    MemoryWriter,
    NavPhase,
    NavTarget,
    NavigationPlanner,
    NewGoatConfig,
    PlanDecision,
    StopDecision,
    StopVerifier,
    StructureTarget,
    _label_matches,
    _goal_has_active_anchor,
)


@dataclass
class ETPGoatConfig(NewGoatConfig):
    """Config for the self-contained clean RGB-only ETP-lite layer."""

    ghost_rays: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 1.5),
        (0.0, 2.5),
        (0.0, 3.5),
        (math.radians(35.0), 2.5),
        (math.radians(-35.0), 2.5),
        (math.radians(80.0), 2.0),
        (math.radians(-80.0), 2.0),
    ])
    ghost_merge_radius: float = 0.9
    ghost_min_distance: float = 0.8
    ghost_confidence: float = 0.42
    ghost_graph_distance_cap: float = 20.0
    ghost_novelty_weight: float = 0.35
    ghost_confidence_weight: float = 0.25
    graph_reachability_weight: float = 0.35
    graph_path_cost_weight: float = 0.22
    semantic_context_weight: float = 0.65
    semantic_direction_weight: float = 0.45
    repeat_candidate_penalty_weight: float = 0.12
    candidate_block_ttl: int = 30
    route_next_hop_enabled: bool = True
    min_forward_before_stop: int = 2
    min_approach_distance: float = 0.6
    bbox_min_growth_ratio: float = 0.20
    stop_require_bbox_growth: bool = False
    stop_min_fresh_bbox: float = 0.10
    stop_short_approach_max: float = 1.2
    stop_memory_enabled: bool = True
    stop_memory_min_write_score: float = 0.36
    stop_memory_proposal_threshold: float = 0.68
    stop_memory_current_radius: float = 0.65
    stop_memory_blacklist_steps: int = 35
    stop_memory_score_weight: float = 0.85
    stop_memory_min_bbox_score: float = 0.65
    stop_memory_require_anchor: bool = True
    stop_success_distance: float = 1.2
    far_visible_max_bbox: float = 0.12


@dataclass
class ResolvedAnchorRoute:
    """Unified memory proposal → anchor waypoint navigation target."""

    route_node_id: str
    route_position: np.ndarray
    target_type: str
    expected_phase: NavPhase
    reason: str
    object_node_id: Optional[str] = None
    linked_object_anchor_id: Optional[str] = None


def _best_goal_observation(packet, goal: GoalManager):
    goal_obs = [
        o for o in packet.report.objects
        if _label_matches(o.label, goal.target_labels)
    ]
    return max(goal_obs, key=lambda o: _bbox_area(o.bbox), default=None)


def _goal_evidence_visible(packet, goal: GoalManager) -> bool:
    best = _best_goal_observation(packet, goal)
    return bool(packet.report.goal_visible or best is not None)


def _goal_evidence_range_bin(packet, goal: GoalManager, evidence: Optional[Dict[str, Any]] = None) -> str:
    best = _best_goal_observation(packet, goal)
    if best is not None and best.range_bin:
        return str(best.range_bin).lower()
    if evidence and evidence.get("range_bin"):
        return str(evidence["range_bin"]).lower()
    return "unknown"


def classify_goal_evidence(
    packet,
    goal: GoalManager,
    servo_evidence: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify current-frame goal evidence: none / far_visible / near_visible."""
    if not _goal_evidence_visible(packet, goal):
        return "none"
    best = _best_goal_observation(packet, goal)
    bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
    range_bin = _goal_evidence_range_bin(packet, goal, servo_evidence)
    bearing = str(
        packet.report.target_direction
        if packet.report.target_direction != "unknown"
        else (best.bearing if best is not None else "unknown")
    ).lower()
    centered = bearing in _CENTER_BEARINGS
    range_close = range_bin in ("very_near", "close")
    if range_close and centered and bbox_area >= 0.06:
        return "near_visible"
    if range_close or (centered and bbox_area >= 0.10):
        return "near_visible"
    return "far_visible"


class ETPBBoxGrowthStopVerifier(StopVerifier):
    """Stop only from VERIFY_STOP with near-visible evidence."""

    def is_near_stop_evidence(
        self,
        packet,
        goal: GoalManager,
        servo_evidence: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if classify_goal_evidence(packet, goal, servo_evidence) != "near_visible":
            return False
        evidence = servo_evidence or {}
        best = _best_goal_observation(packet, goal)
        bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        bearing = str(
            packet.report.target_direction
            if packet.report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        ).lower()
        centered = bearing in _CENTER_BEARINGS
        range_bin = _goal_evidence_range_bin(packet, goal, evidence)
        range_close_ok = range_bin in ("very_near", "close")
        recent_visible = sum(self.state.confirm_buffer[-self.cfg.stop_buffer_size:])
        recent_centered = sum(self.state.centered_buffer[-self.cfg.stop_buffer_size:])
        multi_frame_confirm = bool(
            recent_visible >= self.cfg.servo_entry_evidence
            and recent_centered >= self.cfg.servo_entry_evidence
        )
        fresh_target = bool(
            packet.fresh_vlm
            and _goal_evidence_visible(packet, goal)
            and float(packet.report.goal_match_confidence) >= 0.50
        )
        anchor_dist = evidence.get("anchor_distance")
        anchor_near = (
            anchor_dist is not None
            and float(anchor_dist) <= float(getattr(self.cfg, "anchor_servo_enter_distance", 2.0))
        )
        return bool(
            fresh_target
            and centered
            and multi_frame_confirm
            and (range_close_ok or anchor_near or bbox_area >= float(self.cfg.stop_min_fresh_bbox))
        )

    def is_verify_candidate(
        self,
        packet,
        goal: GoalManager,
        servo_evidence: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.is_near_stop_evidence(packet, goal, servo_evidence)

    def can_stop(
        self,
        packet,
        goal: GoalManager,
        position: Optional[np.ndarray],
        servo_evidence: Optional[Dict[str, Any]] = None,
    ) -> StopDecision:
        evidence = servo_evidence or {}
        evidence_type = classify_goal_evidence(packet, goal, evidence)
        if evidence_type == "far_visible":
            return StopDecision(False, False, False, False, "far_visible_no_stop")
        if evidence_type == "none":
            return StopDecision(False, False, False, False, "no_fresh_target")

        best = _best_goal_observation(packet, goal)
        bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        bearing = str(
            packet.report.target_direction
            if packet.report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        ).lower()
        range_bin = _goal_evidence_range_bin(packet, goal, evidence)

        if not self.is_near_stop_evidence(packet, goal, evidence):
            if bbox_area < float(getattr(self.cfg, "far_visible_max_bbox", 0.12)):
                return StopDecision(False, False, False, False, "fresh_bbox_too_small")
            return StopDecision(False, False, False, False, "not_in_stop_band")

        fresh_target = bool(packet.fresh_vlm and _goal_evidence_visible(packet, goal))
        centered = bearing in _CENTER_BEARINGS
        clear = packet.report.target_visibility in ("clear", "visible", "mostly_visible")
        recent_visible = sum(self.state.confirm_buffer[-self.cfg.stop_buffer_size:])
        recent_centered = sum(self.state.centered_buffer[-self.cfg.stop_buffer_size:])
        multi_frame_confirm = bool(
            recent_visible >= self.cfg.servo_entry_evidence
            and recent_centered >= self.cfg.servo_entry_evidence
        )
        visual_ok = bool(
            fresh_target
            and centered
            and clear
            and bbox_area >= self.cfg.stop_min_fresh_bbox
            and multi_frame_confirm
        )

        approach_ok = bool(
            self.state.forward_action_count >= self.cfg.min_forward_before_stop
            and self.state.approach_travel_distance >= self.cfg.min_approach_distance
        )
        growth_ok = bool(evidence.get("bbox_has_meaningful_growth", False))
        retreating = bool(evidence.get("retreating", False))
        progress_ok = str(evidence.get("relative_progress", "uncertain")).lower() != "farther"
        range_close_ok = range_bin in ("very_near", "close")
        plateau_ok = bool(evidence.get("bbox_plateau", False))
        stop_candidate = bool(packet.report.stop_candidate)
        proximity_ok = bool(growth_ok or plateau_ok or range_close_ok or stop_candidate)
        strong_multiframe_confirm = bool(recent_visible >= 3 and recent_centered >= 3)
        confidence_ok = bool(growth_ok or strong_multiframe_confirm)
        approach_progress_ok = bool(
            approach_ok
            and proximity_ok
            and confidence_ok
            and not retreating
            and progress_ok
        )
        verify_ok = bool((plateau_ok or range_close_ok or stop_candidate) and multi_frame_confirm)

        should = bool(visual_ok and approach_progress_ok and verify_ok)
        if should:
            reason = "etp_bbox_growth_stop"
        elif not fresh_target:
            reason = "no_fresh_target"
        elif not centered:
            reason = "target_not_centered"
        elif not clear:
            reason = "target_not_clear"
        elif bbox_area < self.cfg.stop_min_fresh_bbox:
            reason = "fresh_bbox_too_small"
        elif not multi_frame_confirm:
            reason = "need_multiframe_confirm"
        elif not approach_ok:
            reason = "approach_not_enough"
        elif not proximity_ok:
            reason = "no_proximity_evidence"
        elif not confidence_ok:
            reason = "need_growth_or_strong_multiframe"
        elif retreating or not progress_ok:
            reason = "not_approaching"
        else:
            reason = "not_in_stop_band"
        return StopDecision(should, visual_ok, approach_progress_ok, verify_ok, reason)


@dataclass
class FailedRegion:
    center: np.ndarray
    radius: float
    expires_at: int
    goal_key: str
    reason: str

    def contains(self, position: Optional[np.ndarray], goal_key: str, step: int) -> bool:
        if position is None or step > self.expires_at:
            return False
        if self.goal_key and self.goal_key != goal_key:
            return False
        return float(np.linalg.norm(position - self.center)) <= self.radius


@dataclass
class CleanETPGoatConfig(ETPGoatConfig):
    """Navigation config for the clean hierarchical ETP-lite agent."""

    normal_ghost_rays: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 1.5),
        (0.0, 2.5),
        (0.0, 3.5),
        (math.radians(35.0), 2.5),
        (math.radians(-35.0), 2.5),
        (math.radians(80.0), 2.0),
        (math.radians(-80.0), 2.0),
    ])
    wide_ghost_rays: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 2.5),
        (0.0, 4.0),
        (0.0, 5.5),
        (math.radians(35.0), 4.0),
        (math.radians(-35.0), 4.0),
        (math.radians(80.0), 3.0),
        (math.radians(-80.0), 3.0),
    ])
    escape_ghost_rays: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 3.5),
        (0.0, 5.5),
        (0.0, 7.0),
        (math.radians(35.0), 5.5),
        (math.radians(-35.0), 5.5),
        (math.radians(80.0), 4.0),
        (math.radians(-80.0), 4.0),
        (math.radians(150.0), 3.5),
        (math.radians(-150.0), 3.5),
    ])
    no_evidence_wide_steps: int = 30
    no_evidence_escape_steps: int = 60
    failed_region_radius: float = 2.5
    failed_region_ttl: int = 60
    failed_region_penalty: float = 1.2
    active_scan_turns: int = 3
    goal_servo_cooldown_steps: int = 3


def _default_reuse_counts() -> Dict[str, int]:
    return {
        "memory_reuse_hit": 0,
        "object_anchor_reuse_hit": 0,
        "stop_memory_reuse_hit": 0,
        "semantic_reuse_without_anchor_route": 0,
        "failed_region_penalty_applications": 0,
    }


@dataclass
class CleanETPTaskTelemetry(ETPTaskTelemetry):
    """Per-goal diagnostics including long-term memory reuse vs local control."""

    reuse_counts: Dict[str, int] = field(default_factory=_default_reuse_counts)
    last_selected_node_id: Optional[str] = None
    last_selected_source: Optional[str] = None
    last_semantic_reuse_node_id: Optional[str] = None
    last_failed_region_penalty: float = 0.0
    phase_after_reuse: Optional[str] = None

    def record_proposal(self, proposal: Optional[GoalProposal]) -> None:
        if proposal is None:
            return
        ctype = str(proposal.candidate_type or "").lower().strip()
        if ctype in ("object_anchor", "object_approach", "stop_waypoint_then_anchor"):
            self.proposal_source_counts["object_anchor"] += 1
            return
        super().record_proposal(proposal)

    def reset(self) -> None:
        super().reset()
        self.reuse_counts = _default_reuse_counts()
        self.last_selected_node_id = None
        self.last_selected_source = None
        self.last_semantic_reuse_node_id = None
        self.last_failed_region_penalty = 0.0
        self.phase_after_reuse = None

    def record_reuse_selection(
        self,
        proposal: Optional[GoalProposal],
        etp_route: Optional[Dict[str, Any]],
        nav_phase: NavPhase,
    ) -> None:
        if proposal is None:
            return
        route = etp_route or {}
        self.last_selected_node_id = proposal.candidate_node_id
        self.last_selected_source = str(proposal.source or "")
        self.last_semantic_reuse_node_id = route.get("semantic_reuse_node_id")
        self.phase_after_reuse = _telemetry_phase_key(nav_phase)
        source = self.last_selected_source
        ctype = str(proposal.candidate_type or "")
        if source == "object_memory" and ctype in ("object_anchor", "object_approach", "stop_waypoint_then_anchor"):
            self.reuse_counts["object_anchor_reuse_hit"] += 1
            self.reuse_counts["memory_reuse_hit"] += 1
        elif source == "semantic_memory" and ctype in ("object_anchor", "object_approach"):
            self.reuse_counts["memory_reuse_hit"] += 1
        elif source == "stop_memory":
            if ctype in ("stop_waypoint_then_anchor", "stop_waypoint"):
                self.reuse_counts["stop_memory_reuse_hit"] += 1
                self.reuse_counts["memory_reuse_hit"] += 1
            if route.get("linked_object_anchor_id"):
                self.reuse_counts["object_anchor_reuse_hit"] += 1
        elif self.last_semantic_reuse_node_id and source == "ghost_candidate":
            self.reuse_counts["semantic_reuse_without_anchor_route"] += 1
        for evidence in proposal.evidence_refs:
            penalty = float(evidence.get("failed_region_penalty", 0.0) or 0.0)
            if penalty > 0:
                self.last_failed_region_penalty = penalty
                self.reuse_counts["failed_region_penalty_applications"] += 1
                break

    def snapshot(self) -> Dict[str, Any]:
        out = super().snapshot()
        out["reuse"] = {
            "counts": dict(self.reuse_counts),
            "last_selected_node_id": self.last_selected_node_id,
            "last_selected_source": self.last_selected_source,
            "last_semantic_reuse_node_id": self.last_semantic_reuse_node_id,
            "last_failed_region_penalty": self.last_failed_region_penalty,
            "phase_after_reuse": self.phase_after_reuse,
        }
        return out


def _telemetry_phase_key(phase: NavPhase) -> Optional[str]:
    if phase in (NavPhase.GLOBAL_SEARCH, NavPhase.RECOVERY, NavPhase.ROUTE_TO_STRUCTURE):
        return "SEARCH"
    if phase == NavPhase.ROUTE_TO_OBJECT_ANCHOR:
        return "ROUTE_TO_OBJECT_ANCHOR"
    if phase == NavPhase.TRACK_TARGET:
        return "TRACK"
    if phase in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.PEAK_RETURN):
        return "APPROACH"
    if phase == NavPhase.STOP_VERIFY:
        return "STOP_VERIFY"
    return None


class ETPNavMemoryWriter(MemoryWriter):
    """Writes RGB-only ghost candidates around visited waypoints."""

    def __init__(self, new_config: ETPGoatConfig) -> None:
        super().__init__(new_config)
        self.cfg: ETPGoatConfig = new_config

    def _write_frontiers(
        self,
        topo_map: DynamicTopoMap,
        position: np.ndarray,
        heading: float,
        cur_vp: str,
    ) -> None:
        moved = self.last_frontier_position is None or (
            float(np.linalg.norm(position - self.last_frontier_position))
            >= self.cfg.min_move_for_frontier
        )
        active_candidates = [
            n for n in topo_map.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE)
            if not n.attributes.get("consumed")
            and int(n.attributes.get("blacklisted_until", -1)) < topo_map.current_step
        ]
        if not moved and active_candidates:
            return

        self.last_frontier_position = position.copy()
        cur_node = topo_map.get_node(cur_vp)
        cur_conf = float(cur_node.confidence) if cur_node is not None else 1.0
        room_label = ""
        if cur_node is not None:
            room_label = str(
                cur_node.attributes.get("room_label")
                or cur_node.attributes.get("view_room_label")
                or ""
            )

        for heading_delta, distance in self.cfg.ghost_rays:
            angle = heading + float(heading_delta)
            candidate_pos = position + np.array(
                [-math.sin(angle) * distance, 0.0, -math.cos(angle) * distance],
                dtype=np.float32,
            )
            if self._navmesh_snap is not None:
                snapped = self._navmesh_snap(candidate_pos)
                if snapped is None:
                    continue
                candidate_pos = np.asarray(snapped, dtype=np.float32)

            if topo_map.has_nearby_visited(candidate_pos, radius=self.cfg.ghost_min_distance):
                continue
            existing = self._nearby_open_candidate(topo_map, candidate_pos)
            if existing is not None:
                self._merge_candidate(existing, candidate_pos, cur_vp, distance, heading_delta)
                continue

            cand_id = topo_map.add_node(
                NodeType.WAYPOINT_CANDIDATE,
                position=candidate_pos,
                confidence=min(0.95, self.cfg.ghost_confidence * (0.75 + 0.25 * cur_conf)),
                label=room_label,
                attributes={
                    "semantic_role": "ghost_candidate",
                    "source": "rgb_rule_ghost",
                    "state": "candidate",
                    "consumed": False,
                    "blocked": False,
                    "anchor_waypoint_id": cur_vp,
                    "anchor_waypoint_position": position.astype(np.float32).tolist(),
                    "heading_delta": float(heading_delta),
                    "distance": float(distance),
                    "traversability": 1.0,
                    "visit_attempts": 0,
                    "seen_count": 1,
                    "room_context": room_label,
                },
            )
            topo_map.add_edge(cur_vp, cand_id, EdgeType.NAVIGABLE, weight=float(distance))

    def _nearby_open_candidate(
        self, topo_map: DynamicTopoMap, position: np.ndarray
    ) -> Optional[SemanticNode]:
        nearest = topo_map.find_nearest_node(position, NodeType.WAYPOINT_CANDIDATE)
        if nearest is None:
            return None
        if nearest.attributes.get("consumed"):
            return None
        if int(nearest.attributes.get("blacklisted_until", -1)) >= topo_map.current_step:
            return None
        if float(np.linalg.norm(nearest.position - position)) > self.cfg.ghost_merge_radius:
            return None
        return nearest

    @staticmethod
    def _merge_candidate(
        node: SemanticNode,
        position: np.ndarray,
        anchor_waypoint_id: str,
        distance: float,
        heading_delta: float,
    ) -> None:
        seen = int(node.attributes.get("seen_count", 1))
        node.position = ((node.position * seen) + position) / float(seen + 1)
        node.confidence = min(0.95, float(node.confidence) + 0.05)
        node.attributes["seen_count"] = seen + 1
        fronts = list(node.attributes.get("anchor_waypoint_ids", []))
        if anchor_waypoint_id not in fronts:
            fronts.append(anchor_waypoint_id)
        node.attributes["anchor_waypoint_ids"] = fronts
        node.attributes.setdefault("anchor_waypoint_id", anchor_waypoint_id)
        node.attributes["distance"] = min(float(node.attributes.get("distance", distance)), float(distance))
        node.attributes["heading_delta"] = float(heading_delta)


class CleanETPMemoryWriter(ETPNavMemoryWriter):
    """ETP ghost writer with dynamic radius levels."""

    def __init__(self, new_config: CleanETPGoatConfig) -> None:
        super().__init__(new_config)
        self.cfg: CleanETPGoatConfig = new_config
        self.exploration_level: str = "normal"

    def _active_rays(self) -> List[Tuple[float, float]]:
        if self.exploration_level == "escape":
            return self.cfg.escape_ghost_rays
        if self.exploration_level == "wide":
            return self.cfg.wide_ghost_rays
        return self.cfg.normal_ghost_rays

    def _write_frontiers(
        self,
        topo_map: DynamicTopoMap,
        position: np.ndarray,
        heading: float,
        cur_vp: str,
    ) -> None:
        original = self.cfg.ghost_rays
        self.cfg.ghost_rays = self._active_rays()
        try:
            super()._write_frontiers(topo_map, position, heading, cur_vp)
        finally:
            self.cfg.ghost_rays = original


class ETPGraphNavigationPlanner(NavigationPlanner):
    """Graph-aware proposal planner for visited/ghost waypoint navigation."""

    def __init__(self, new_config: ETPGoatConfig) -> None:
        super().__init__(new_config)
        self.cfg: ETPGoatConfig = new_config
        self._last_route_debug: Dict[str, Any] = {}

    def collect_goal_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        structure: StructureTarget,
        nav_phase: NavPhase,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str] = None,
    ) -> List[GoalProposal]:
        proposals = super().collect_goal_proposals(
            topo_map, goal, structure, nav_phase, position, cur_vp_id,
        )
        seen = {p.candidate_node_id for p in proposals}
        for p in self._collect_candidate_proposals(topo_map, goal, structure, position, cur_vp_id):
            if p.candidate_node_id not in seen:
                proposals.append(p)
                seen.add(p.candidate_node_id)
        for p in self._collect_stop_memory_proposals(topo_map, goal, position, cur_vp_id):
            if p.candidate_node_id not in seen:
                proposals.append(p)
                seen.add(p.candidate_node_id)
        return proposals

    def _collect_candidate_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        structure: StructureTarget,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> List[GoalProposal]:
        if position is None:
            return []
        proposals: List[GoalProposal] = []
        no_goal_anchor = not _goal_has_active_anchor(topo_map, goal, self.cfg)
        for node in topo_map.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE):
            if node.attributes.get("consumed"):
                continue
            if int(node.attributes.get("blacklisted_until", -1)) >= topo_map.current_step:
                continue
            if node.attributes.get("blocked"):
                until = int(node.attributes.get("blocked_until_step", -1))
                if until < 0 or until >= topo_map.current_step:
                    continue
            route_cost, route_path, anchor_id = self._route_cost_to_node(
                topo_map, cur_vp_id, node.node_id, position,
            )
            if not np.isfinite(route_cost):
                continue

            graph_cost = min(route_cost / self.cfg.ghost_graph_distance_cap, 1.0)
            reach = max(0.0, 1.0 - graph_cost)
            novelty = self._candidate_novelty(topo_map, node)
            sem = self._frontier_semantic_score(topo_map, node, goal, structure)
            semantic_context_node_id, semantic_context_score = self._semantic_context_for_candidate(
                topo_map, node, goal, structure,
            )
            direction_score = self._semantic_direction_score(
                topo_map, node, position, semantic_context_node_id, structure,
            )
            if no_goal_anchor:
                sem += 0.4
            risk = self._candidate_risk(node)
            repeat_penalty = self.cfg.repeat_candidate_penalty_weight * max(
                0,
                int(node.attributes.get("visit_attempts", 0))
                + int(node.attributes.get("seen_count", 1)) - 1,
            )
            score = (
                sem
                + self.cfg.semantic_context_weight * semantic_context_score
                + self.cfg.semantic_direction_weight * direction_score
                + self.cfg.ghost_novelty_weight * novelty
                + self.cfg.ghost_confidence_weight * float(node.confidence)
                + self.cfg.graph_reachability_weight * reach
                - self.cfg.graph_path_cost_weight * graph_cost
                - risk
                - repeat_penalty
            )
            proposals.append(GoalProposal(
                goal_id=goal.goal_key,
                candidate_node_id=node.node_id,
                candidate_type="waypoint_candidate",
                anchor_node_id=anchor_id,
                target_position=node.position.copy(),
                score=score,
                semantic_score=sem,
                frontier_value=novelty,
                reachability_score=reach,
                distance_cost=graph_cost,
                risk_penalty=risk,
                source="ghost_candidate",
                can_stop=False,
                requires_verification=True,
                evidence_refs=[{
                    "route_path": route_path,
                    "route_cost": route_cost,
                    "anchor_waypoint_id": anchor_id,
                    "seen_count": int(node.attributes.get("seen_count", 1)),
                    "semantic_context_node_id": semantic_context_node_id,
                    "semantic_context_score": semantic_context_score,
                    "semantic_direction_score": direction_score,
                    "repeat_penalty": repeat_penalty,
                }],
            ))
        return proposals

    def _collect_stop_memory_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> List[GoalProposal]:
        if not self.cfg.stop_memory_enabled or position is None:
            return []
        proposals: List[GoalProposal] = []
        for node in topo_map.get_nodes_by_type(NodeType.WAYPOINT_VISITED):
            attrs = node.attributes
            if int(attrs.get("stop_blacklisted_until", -1)) >= topo_map.current_step:
                continue
            score = float(attrs.get("goal_stop_score", 0.0))
            if score < self.cfg.stop_memory_proposal_threshold:
                continue
            stop_anchor_id = attrs.get("goal_stop_object_id")
            if self.cfg.stop_memory_require_anchor and not stop_anchor_id:
                continue
            if self.cfg.stop_memory_require_anchor and self._valid_stop_anchor(
                topo_map, goal, str(stop_anchor_id),
            ) is None:
                continue
            evidence = attrs.get("goal_stop_evidence") or {}
            if float(evidence.get("bbox_score", 0.0)) < self.cfg.stop_memory_min_bbox_score:
                continue
            if attrs.get("goal_key") and attrs.get("goal_key") != goal.goal_key:
                continue
            if float(np.linalg.norm(node.position - position)) < self.cfg.stop_memory_current_radius:
                continue
            route_cost, route_path, anchor_id = self._route_cost_to_node(
                topo_map, cur_vp_id, node.node_id, position,
            )
            if not np.isfinite(route_cost):
                continue
            graph_cost = min(route_cost / self.cfg.ghost_graph_distance_cap, 1.0)
            reach = max(0.0, 1.0 - graph_cost)
            proposals.append(GoalProposal(
                goal_id=goal.goal_key,
                candidate_node_id=node.node_id,
                candidate_type="stop_waypoint",
                anchor_node_id=anchor_id,
                target_position=node.position.copy(),
                score=self.cfg.stop_memory_score_weight * score + 0.3 * reach - 0.15 * graph_cost,
                semantic_score=score,
                task_score=score,
                reachability_score=reach,
                distance_cost=graph_cost,
                source="stop_memory",
                can_stop=False,
                requires_verification=True,
                evidence_refs=[{
                    "goal_stop_score": score,
                    "route_path": route_path,
                    "linked_object_anchor_id": stop_anchor_id,
                    "goal_stop_step": attrs.get("goal_stop_step"),
                }],
            ))
        return proposals

    def score_goal_proposals(
        self,
        proposals: List[GoalProposal],
        goal: GoalManager,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray] = None,
        cur_vp_id: Optional[str] = None,
    ) -> List[GoalProposal]:
        scored = super().score_goal_proposals(proposals, goal, topo_map, position)
        for p in scored:
            node = topo_map.get_node(p.candidate_node_id)
            if node is None:
                continue
            route_cost, route_path, anchor_id = self._route_cost_to_node(
                topo_map, cur_vp_id, node.node_id, position,
            )
            if not np.isfinite(route_cost):
                p.reachability_score = 0.0
                p.distance_cost = 1.0
                p.risk_penalty += 0.6
                p.score -= 0.6
                continue
            graph_cost = min(route_cost / self.cfg.ghost_graph_distance_cap, 1.0)
            reach = max(0.0, 1.0 - graph_cost)
            p.reachability_score = max(float(p.reachability_score), reach)
            p.distance_cost = graph_cost
            p.anchor_node_id = p.anchor_node_id or anchor_id
            p.evidence_refs.append({
                "graph_route_path": route_path,
                "graph_route_cost": route_cost,
                "graph_anchor_id": anchor_id,
            })
            p.score += (
                self.cfg.graph_reachability_weight * reach
                - self.cfg.graph_path_cost_weight * graph_cost
            )
            if p.source == "ghost_candidate":
                p.score = max(float(p.score), 0.08)
            elif p.source == "stop_memory":
                p.score += self.cfg.stop_memory_score_weight * float(p.semantic_score)
        return sorted(scored, key=lambda p: p.score, reverse=True)

    def resolve_proposal_to_anchor(
        self,
        proposal: GoalProposal,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> Optional[ResolvedAnchorRoute]:
        """Unified memory proposal → anchor waypoint navigation target."""
        node = topo_map.get_node(proposal.candidate_node_id)
        if node is None:
            return None

        if proposal.source in ("object_memory", "semantic_memory"):
            if node.node_type != NodeType.OBJECT:
                return None
            if node.attributes.get("semantic_role") != "object_anchor":
                return None
            if not _label_matches(node.label, goal.target_labels):
                return None
            route = self._resolve_object_route(node, position, cur_vp_id)
            route_cost, route_path, anchor_wp_id = self._route_cost_to_node(
                topo_map, cur_vp_id, node.node_id, position,
            )
            if route is not None and route.target_position is not None:
                route_pos = route.target_position.copy()
            else:
                route_pos = self._object_anchor_route_position(
                    topo_map, node, cur_vp_id, route_path,
                )
            if route_pos is None:
                return None
            return ResolvedAnchorRoute(
                route_node_id=str(node.node_id),
                route_position=route_pos,
                target_type="object_anchor",
                expected_phase=NavPhase.TRACK_TARGET,
                reason="resolve_object_memory_anchor",
                object_node_id=node.node_id,
                linked_object_anchor_id=node.node_id,
            )

        if proposal.source == "stop_memory" and node.node_type == NodeType.WAYPOINT_VISITED:
            linked_id = node.attributes.get("goal_stop_object_id")
            for evidence in proposal.evidence_refs:
                candidate = evidence.get("linked_object_anchor_id")
                if candidate:
                    linked_id = candidate
                    break
            route_cost, route_path, _anchor_id = self._route_cost_to_node(
                topo_map, cur_vp_id, node.node_id, position,
            )
            route_pos = self._next_route_position(topo_map, cur_vp_id, route_path, node)
            if route_pos is None:
                route_pos = node.position.copy()
            linked_node = topo_map.get_node(str(linked_id)) if linked_id else None
            linked_ok = (
                linked_node is not None
                and self._valid_stop_anchor(topo_map, goal, str(linked_id)) is not None
                and _label_matches(linked_node.label, goal.target_labels)
            )
            if linked_ok:
                return ResolvedAnchorRoute(
                    route_node_id=node.node_id,
                    route_position=route_pos,
                    target_type="stop_waypoint_then_anchor",
                    expected_phase=NavPhase.TRACK_TARGET,
                    reason="resolve_stop_memory_with_anchor",
                    linked_object_anchor_id=str(linked_id),
                )
            return ResolvedAnchorRoute(
                route_node_id=node.node_id,
                route_position=route_pos,
                target_type="stop_waypoint_scan_only",
                expected_phase=NavPhase.GLOBAL_SEARCH,
                reason="resolve_stop_memory_scan_only",
            )

        return None

    def _apply_resolved_proposal_type(self, proposal: GoalProposal, resolved: ResolvedAnchorRoute) -> None:
        proposal.candidate_type = resolved.target_type
        if resolved.linked_object_anchor_id:
            proposal.anchor_node_id = resolved.linked_object_anchor_id
        proposal.can_stop = False
        proposal.requires_verification = True

    def select_best_proposal(
        self,
        proposals: List[GoalProposal],
        topo_map: Optional[DynamicTopoMap] = None,
        goal: Optional[GoalManager] = None,
        position: Optional[np.ndarray] = None,
        cur_vp_id: Optional[str] = None,
    ) -> Optional[GoalProposal]:
        active = [
            p for p in proposals
            if p.status in ("active", "needs_verify", "confirmed", "preserved")
        ]
        if not active:
            return None

        if topo_map is not None and goal is not None:
            anchor_routes: List[GoalProposal] = []
            for proposal in active:
                resolved = self.resolve_proposal_to_anchor(
                    proposal, topo_map, goal, position, cur_vp_id,
                )
                if resolved is None:
                    continue
                if resolved.target_type in ("object_anchor", "stop_waypoint_then_anchor"):
                    self._apply_resolved_proposal_type(proposal, resolved)
                    anchor_routes.append(proposal)
            if anchor_routes:
                return max(anchor_routes, key=lambda p: p.score)

        object_anchors = [
            p for p in active
            if p.source == "object_memory"
            and p.candidate_type in ("object_anchor", "object_approach")
        ]
        if object_anchors:
            return max(object_anchors, key=lambda p: p.score)
        semantic_anchors = [
            p for p in active
            if p.source == "semantic_memory"
            and p.candidate_type in ("object_anchor", "object_approach")
        ]
        if semantic_anchors:
            return max(semantic_anchors, key=lambda p: p.score)
        stop_memory = [
            p for p in active
            if p.source == "stop_memory"
            and any(
                evidence.get("linked_object_anchor_id")
                for evidence in p.evidence_refs
            )
        ]
        if stop_memory:
            best_stop = max(stop_memory, key=lambda p: p.score)
            if float(best_stop.score) >= float(self.cfg.stop_memory_proposal_threshold):
                return best_stop
        return super().select_best_proposal(active)

    def proposal_to_nav_target(
        self,
        proposal: GoalProposal,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
        goal: Optional[GoalManager] = None,
    ) -> NavTarget:
        resolved: Optional[ResolvedAnchorRoute] = None
        if goal is not None:
            resolved = self.resolve_proposal_to_anchor(
                proposal, topo_map, goal, position, cur_vp_id,
            )

        base = super().proposal_to_nav_target(proposal, topo_map, position, cur_vp_id)
        node = topo_map.get_node(proposal.candidate_node_id)
        if node is None:
            self._last_route_debug = {}
            return base

        route_cost, route_path, anchor_id = self._route_cost_to_node(
            topo_map, cur_vp_id, node.node_id, position,
        )
        target_position = base.target_position
        reason = base.reason
        expected = base.expected_phase_after_reach
        target_type = base.target_type

        if resolved is not None:
            target_position = resolved.route_position.copy()
            reason = resolved.reason
            expected = resolved.expected_phase
            if resolved.target_type == "object_anchor":
                target_type = "object_anchor"
            elif resolved.target_type == "stop_waypoint_then_anchor":
                target_type = "stop_waypoint"
            elif resolved.target_type == "stop_waypoint_scan_only":
                target_type = "stop_waypoint"
                expected = NavPhase.GLOBAL_SEARCH
        elif node.node_type == NodeType.WAYPOINT_CANDIDATE:
            target_type = "waypoint_candidate"
            reason = "clean_etp_ghost_candidate"
            expected = NavPhase.GLOBAL_SEARCH
            target_position = self._next_route_position(topo_map, cur_vp_id, route_path, node)
        elif proposal.source == "stop_memory":
            target_type = "stop_waypoint"
            reason = "clean_etp_stop_memory_route"
            expected = NavPhase.TRACK_TARGET
            target_position = self._next_route_position(topo_map, cur_vp_id, route_path, node)
        elif node.attributes.get("semantic_role") == "object_anchor":
            target_position = self._object_anchor_route_position(topo_map, node, cur_vp_id, route_path)
            reason = "clean_etp_object_anchor_via_waypoint"
            expected = NavPhase.TRACK_TARGET
        elif proposal.source in ("semantic_memory", "room", "structure") or target_type in (
            "room", "landmark", "room_summary", "goal_region", "context_object",
        ):
            target_position = self._next_route_position(topo_map, cur_vp_id, route_path, node)
            reason = "clean_etp_semantic_route"
            expected = NavPhase.ROUTE_TO_STRUCTURE

        semantic_reuse_node_id = None
        if proposal.source == "semantic_memory":
            semantic_reuse_node_id = proposal.candidate_node_id
        elif proposal.source == "stop_memory":
            for evidence in proposal.evidence_refs:
                linked_id = evidence.get("linked_object_anchor_id")
                if linked_id:
                    semantic_reuse_node_id = str(linked_id)
                    break
        elif proposal.source == "ghost_candidate":
            for evidence in proposal.evidence_refs:
                context_id = evidence.get("semantic_context_node_id")
                if context_id:
                    semantic_reuse_node_id = str(context_id)
                    break

        self._last_route_debug = {
            "selected_node_id": proposal.candidate_node_id,
            "selected_type": target_type,
            "selected_source": proposal.source,
            "selected_candidate_type": proposal.candidate_type,
            "resolved_target_type": resolved.target_type if resolved else None,
            "semantic_reuse_node_id": semantic_reuse_node_id,
            "anchor_waypoint_id": anchor_id,
            "route_path": route_path,
            "route_cost": route_cost if np.isfinite(route_cost) else None,
            "next_subgoal_position": target_position.tolist() if target_position is not None else None,
            "linked_object_anchor_id": (
                resolved.linked_object_anchor_id if resolved else None
            ),
        }
        route_node_id = proposal.candidate_node_id
        if resolved is not None and resolved.target_type == "object_anchor":
            route_node_id = str(resolved.object_node_id or proposal.candidate_node_id)
        return NavTarget(
            target_position=target_position.copy() if target_position is not None else None,
            target_node_id=route_node_id,
            target_type=target_type,
            reason=reason,
            expected_phase_after_reach=expected,
        )

    def _route_cost_to_node(
        self,
        topo_map: DynamicTopoMap,
        cur_vp_id: Optional[str],
        node_id: str,
        position: Optional[np.ndarray],
    ) -> Tuple[float, List[str], Optional[str]]:
        node = topo_map.get_node(node_id)
        if node is None:
            return float("inf"), [], None
        if cur_vp_id is None or topo_map.get_node(cur_vp_id) is None:
            if position is None:
                return float("inf"), [], None
            return float(np.linalg.norm(node.position - position)), [node_id], node_id

        if node.node_type == NodeType.WAYPOINT_CANDIDATE:
            anchor_id = node.attributes.get("anchor_waypoint_id") or self._nearest_visited_id(topo_map, node.position)
            if anchor_id is None:
                return float("inf"), [], None
            anchor = topo_map.get_node(anchor_id)
            if anchor is None:
                return float("inf"), [], None
            path = topo_map.shortest_path(cur_vp_id, anchor_id) or []
            if cur_vp_id != anchor_id and not path:
                return float("inf"), [], anchor_id
            prefix_cost = self._path_cost(topo_map, path)
            tail = float(np.linalg.norm(node.position - anchor.position))
            return prefix_cost + tail, path + [node_id], anchor_id

        anchor_id = node.attributes.get("anchor_waypoint_id")
        if not anchor_id and node.node_type != NodeType.WAYPOINT_VISITED:
            anchor_id = self._nearest_visited_id(topo_map, node.position)
        target_id = anchor_id if anchor_id and topo_map.get_node(anchor_id) is not None else node_id
        if target_id == cur_vp_id:
            return 0.0, [cur_vp_id], target_id
        path = topo_map.shortest_path(cur_vp_id, target_id) or []
        if not path:
            return float("inf"), [], target_id
        residual = 0.0
        target_node = topo_map.get_node(target_id)
        if target_node is not None and target_id != node_id:
            residual = float(np.linalg.norm(node.position - target_node.position))
        return self._path_cost(topo_map, path) + residual, path, target_id

    @staticmethod
    def _path_cost(topo_map: DynamicTopoMap, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0
        cost = 0.0
        for a, b in zip(path[:-1], path[1:]):
            if topo_map.graph.has_edge(a, b):
                cost += float(topo_map.graph.edges[a, b].get("weight", 1.0))
            else:
                na = topo_map.get_node(a)
                nb = topo_map.get_node(b)
                if na is not None and nb is not None:
                    cost += float(np.linalg.norm(na.position - nb.position))
        return cost

    @staticmethod
    def _nearest_visited_id(topo_map: DynamicTopoMap, position: np.ndarray) -> Optional[str]:
        nearest = topo_map.find_nearest_node(position, NodeType.WAYPOINT_VISITED)
        return nearest.node_id if nearest is not None else None

    @staticmethod
    def _candidate_novelty(topo_map: DynamicTopoMap, node: SemanticNode) -> float:
        nearest = topo_map.find_nearest_node(node.position, NodeType.WAYPOINT_VISITED)
        if nearest is None:
            return 1.0
        dist = float(np.linalg.norm(node.position - nearest.position))
        return min(dist / 4.0, 1.0)

    @staticmethod
    def _candidate_risk(node: SemanticNode) -> float:
        blocked = int(node.attributes.get("blocked_count", 0))
        unreachable = int(node.attributes.get("unreachable_count", 0))
        attempts = int(node.attributes.get("visit_attempts", 0))
        traversability = float(node.attributes.get("traversability", 1.0))
        return max(0.0, 1.0 - traversability) + 0.15 * blocked + 0.20 * unreachable + 0.08 * attempts

    def _semantic_context_for_candidate(
        self,
        topo_map: DynamicTopoMap,
        node: SemanticNode,
        goal: GoalManager,
        structure: StructureTarget,
    ) -> Tuple[Optional[str], float]:
        best_id: Optional[str] = None
        best_score = 0.0

        def update(candidate_id: Optional[str], score: float) -> None:
            nonlocal best_id, best_score
            if candidate_id and score > best_score:
                best_id = candidate_id
                best_score = float(score)

        if structure.node_id and structure.position is not None:
            dist = float(np.linalg.norm(node.position - structure.position))
            update(structure.node_id, max(0.0, 1.0 - dist / 10.0))

        room_labels = {x.lower() for x in goal.room_prior}
        for room in topo_map.get_nodes_by_type(NodeType.ROOM):
            if room_labels:
                room_text = " ".join([
                    str(room.label or ""),
                    str(room.attributes.get("room_label", "")),
                    str(room.attributes.get("region_label", "")),
                ]).lower()
                if not any(label and label in room_text for label in room_labels):
                    continue
            dist = float(np.linalg.norm(node.position - room.position))
            if dist < 12.0:
                update(room.node_id, max(0.0, 1.0 - dist / 12.0))

        landmark_labels = {x.lower() for x in goal.landmark_prior}
        for obj in topo_map.get_nodes_by_type(NodeType.OBJECT):
            role = obj.attributes.get("semantic_role")
            if role not in ("context_object", "environment_object", "object_anchor"):
                continue
            label_match = bool(landmark_labels and _label_matches(obj.label, landmark_labels))
            room_context = str(obj.attributes.get("room_context", "")).lower()
            room_match = bool(room_labels and _label_matches(room_context, room_labels))
            if not label_match and not room_match:
                continue
            dist = float(np.linalg.norm(node.position - obj.position))
            if dist < 8.0:
                update(obj.node_id, max(0.0, 1.0 - dist / 8.0))

        for landmark in topo_map.get_nodes_by_type(NodeType.LANDMARK):
            if landmark_labels and not _label_matches(landmark.label, landmark_labels):
                continue
            dist = float(np.linalg.norm(node.position - landmark.position))
            if dist < 8.0:
                update(landmark.node_id, max(0.0, 1.0 - dist / 8.0))

        return best_id, best_score

    @staticmethod
    def _semantic_direction_score(
        topo_map: DynamicTopoMap,
        node: SemanticNode,
        position: np.ndarray,
        semantic_context_node_id: Optional[str],
        structure: StructureTarget,
    ) -> float:
        target_pos = None
        if semantic_context_node_id:
            ctx = topo_map.get_node(semantic_context_node_id)
            if ctx is not None:
                target_pos = ctx.position
        if target_pos is None and structure.position is not None:
            target_pos = structure.position
        if target_pos is None:
            return 0.0
        before = float(np.linalg.norm(target_pos - position))
        after = float(np.linalg.norm(target_pos - node.position))
        if before <= 1e-3:
            return 0.0
        return max(-0.5, min(1.0, (before - after) / max(before, 1.0)))

    @staticmethod
    def _valid_stop_anchor(
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        anchor_id: str,
    ) -> Optional[SemanticNode]:
        anchor = topo_map.get_node(anchor_id)
        if anchor is None:
            return None
        if anchor.node_type != NodeType.OBJECT:
            return None
        if anchor.attributes.get("semantic_role") != "object_anchor":
            return None
        if int(anchor.attributes.get("blacklisted_until", -1)) >= topo_map.current_step:
            return None
        if not _label_matches(anchor.label, goal.target_labels):
            return None
        return anchor

    def _next_route_position(
        self,
        topo_map: DynamicTopoMap,
        cur_vp_id: Optional[str],
        route_path: List[str],
        final_node: SemanticNode,
    ) -> np.ndarray:
        if not self.cfg.route_next_hop_enabled or not route_path:
            return final_node.position.copy()
        if cur_vp_id and len(route_path) >= 2 and route_path[0] == cur_vp_id:
            next_id = route_path[1]
            next_node = topo_map.get_node(next_id)
            if next_node is not None:
                return next_node.position.copy()
        anchor_id = final_node.attributes.get("anchor_waypoint_id")
        if anchor_id and cur_vp_id != anchor_id:
            anchor = topo_map.get_node(anchor_id)
            if anchor is not None:
                return anchor.position.copy()
        return final_node.position.copy()

    def _object_anchor_route_position(
        self,
        topo_map: DynamicTopoMap,
        anchor_node: SemanticNode,
        cur_vp_id: Optional[str],
        route_path: List[str],
    ) -> np.ndarray:
        if route_path:
            return self._next_route_position(topo_map, cur_vp_id, route_path, anchor_node)
        wp_id = anchor_node.attributes.get("anchor_waypoint_id")
        wp_node = topo_map.get_node(wp_id) if wp_id else None
        if wp_node is not None:
            return wp_node.position.copy()
        wp_pos = anchor_node.attributes.get("anchor_waypoint_position")
        if wp_pos is not None:
            return np.asarray(wp_pos, dtype=np.float32)
        return anchor_node.position.copy()


class CleanETPPlanner(ETPGraphNavigationPlanner):
    """Planner with failed-region penalties on every proposal."""

    def __init__(self, new_config: CleanETPGoatConfig) -> None:
        super().__init__(new_config)
        self.cfg: CleanETPGoatConfig = new_config
        self.failed_regions: List[FailedRegion] = []

    def add_failed_region(
        self,
        center: Optional[np.ndarray],
        *,
        goal_key: str,
        step: int,
        reason: str,
    ) -> None:
        if center is None:
            return
        self.failed_regions.append(FailedRegion(
            center=np.asarray(center, dtype=np.float32).copy(),
            radius=float(self.cfg.failed_region_radius),
            expires_at=int(step + self.cfg.failed_region_ttl),
            goal_key=str(goal_key or ""),
            reason=str(reason),
        ))
        self.failed_regions = self.failed_regions[-32:]

    def _failed_region_penalty(
        self,
        position: Optional[np.ndarray],
        goal_key: str,
        step: int,
    ) -> Tuple[float, List[str]]:
        penalty = 0.0
        reasons: List[str] = []
        active: List[FailedRegion] = []
        for region in self.failed_regions:
            if step > region.expires_at:
                continue
            active.append(region)
            if region.contains(position, goal_key, step):
                penalty += self.cfg.failed_region_penalty
                reasons.append(region.reason)
        self.failed_regions = active
        return penalty, reasons

    def score_goal_proposals(
        self,
        proposals: List[GoalProposal],
        goal,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray] = None,
        cur_vp_id: Optional[str] = None,
    ) -> List[GoalProposal]:
        scored = super().score_goal_proposals(proposals, goal, topo_map, position, cur_vp_id)
        for proposal in scored:
            penalty, reasons = self._failed_region_penalty(
                proposal.target_position,
                goal.goal_key,
                topo_map.current_step,
            )
            if penalty <= 0:
                continue
            proposal.risk_penalty += penalty
            proposal.score -= penalty
            proposal.evidence_refs.append({
                "failed_region_penalty": penalty,
                "failed_region_reasons": reasons,
            })
        return sorted(scored, key=lambda p: p.score, reverse=True)


class ConfTopoGOATCleanETPNavAgent(GoatAgent):
    """Clean hierarchical ETP-lite GOAT agent."""

    def __init__(
        self,
        config: Optional[ConfTopoConfig] = None,
        new_config: Optional[CleanETPGoatConfig] = None,
    ) -> None:
        clean_config = new_config or CleanETPGoatConfig()
        super().__init__(config=config, new_config=clean_config)
        self.new_config: CleanETPGoatConfig = clean_config
        self.memory_writer = CleanETPMemoryWriter(clean_config)
        self.navigation_planner = CleanETPPlanner(clean_config)
        self.stop_verifier = ETPBBoxGrowthStopVerifier(clean_config, self.servo_state)
        self._etp_stop_memory_target_id: Optional[str] = None
        self._goal_evidence_step: int = -1
        self._scan_turns_remaining: int = 0
        self._last_promoted_waypoint_id: Optional[str] = None
        self._goal_servo_unlock_step: int = 0
        self.task_telemetry = CleanETPTaskTelemetry()
        self._etp_stop_evaluated_step: int = -1

    def set_new_goal(self, goal):
        super().set_new_goal(goal)
        self._reset_local_navigation_state("goal_changed_reset")

    def record_executed_action(self, low_action: str) -> None:
        self.task_telemetry.record_action(low_action)

    def _reset_local_navigation_state(self, reason: str) -> None:
        """Clear last-mile control state; long-term TopoMap is kept by GoatAgent."""
        self.local_servo.reset()
        self.servo_state.reset()
        self.recovery_manager.reset()
        self._last_nav_target = NavTarget()
        self._last_selected_goal_proposal = None
        self._last_goal_proposals = []
        self._last_servo_evidence = {}
        self._last_stop = StopDecision(False, False, False, False, "not_evaluated")
        self._etp_stop_memory_target_id = None
        self._scan_turns_remaining = 0
        self._last_promoted_waypoint_id = None
        self._goal_evidence_step = self.topo_map.current_step
        self._target_object_detected_this_scan = False
        self._goal_servo_unlock_step = (
            self.topo_map.current_step + int(self.new_config.goal_servo_cooldown_steps)
        )
        self.task_telemetry.reset()
        self._transition(NavPhase.GLOBAL_SEARCH, reason)

    def _anchor_matches_current_goal(self, anchor_id: Optional[str]) -> bool:
        if not anchor_id:
            return True
        node = self.topo_map.get_node(anchor_id)
        if node is None or node.node_type != NodeType.OBJECT:
            return True
        return _label_matches(node.label, self.goal_manager.target_labels)

    def _target_seen_for_current_goal(self) -> bool:
        packet = self._last_packet
        report = packet.report
        if report.goal_visible:
            return True
        if any(_label_matches(o.label, self.goal_manager.target_labels) for o in report.objects):
            return True
        if float(report.goal_match_confidence) >= 0.55 and (
            packet.fresh_vlm or packet.cached_vlm
        ):
            return True
        return sum(self.servo_state.confirm_buffer[-3:]) >= 1

    def _servo_allowed_for_current_goal(self) -> bool:
        if self.topo_map.current_step < self._goal_servo_unlock_step:
            return False
        if not self._anchor_matches_current_goal(self.servo_state.active_anchor_id):
            return False
        if self.nav_phase in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY):
            return self._target_seen_for_current_goal()
        if self.nav_phase == NavPhase.TRACK_TARGET:
            return True
        return self._target_seen_for_current_goal()

    def _maybe_enter_servo_near_anchor(self, nav_target: NavTarget) -> None:
        if not self._servo_allowed_for_current_goal():
            return
        if nav_target.target_type in ("object_anchor", "object_approach"):
            if not self._anchor_matches_current_goal(nav_target.target_node_id):
                return
        super()._maybe_enter_servo_near_anchor(nav_target)

    def _track_ready_for_approach(self) -> bool:
        if not self._servo_allowed_for_current_goal():
            return False
        if not self._anchor_matches_current_goal(self.servo_state.active_anchor_id):
            return False
        return super()._track_ready_for_approach()

    def _update_goal_evidence_buffer(self) -> None:
        """Record current-goal perception evidence every step, any nav phase."""
        packet = self._last_packet
        goal = self.goal_manager
        if packet is None:
            return
        if (
            self.servo_state.track_buffer
            and int(self.servo_state.track_buffer[-1].get("step_id", -1)) == int(packet.step_id)
        ):
            return

        best = _best_goal_observation(packet, goal)
        visible = _goal_evidence_visible(packet, goal)
        if not visible and float(packet.report.goal_match_confidence) < 0.45:
            return

        bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        bearing = str(
            packet.report.target_direction
            if packet.report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        ).lower()
        range_bin = _goal_evidence_range_bin(packet, goal, self._last_servo_evidence)
        centered = bearing in _CENTER_BEARINGS
        evidence_type = classify_goal_evidence(packet, goal, self._last_servo_evidence)
        if packet.fresh_vlm:
            freshness = "fresh"
        elif packet.cached_vlm:
            freshness = "cached"
        else:
            freshness = "none"

        sample = {
            "step_id": int(packet.step_id),
            "visible": visible,
            "label_match": visible,
            "bbox_valid": bbox_area > 0.0,
            "bbox_area": bbox_area,
            "centered": centered,
            "bearing": bearing,
            "range_bin": range_bin,
            "fresh_or_cached": freshness,
            "source": str(getattr(packet.report, "source", "unknown")),
            "evidence_type": evidence_type,
            "confidence": float(best.confidence) if best is not None else float(packet.report.goal_match_confidence),
            "nav_phase": self.nav_phase.value,
        }
        self.servo_state.track_buffer.append(sample)
        if len(self.servo_state.track_buffer) > self.new_config.track_buffer_size:
            del self.servo_state.track_buffer[:-self.new_config.track_buffer_size]

        label_hit = visible and (
            packet.fresh_vlm
            or packet.cached_vlm
            or float(packet.report.goal_match_confidence) >= 0.50
        )
        self.servo_state.confirm_buffer.append(label_hit)
        self.servo_state.centered_buffer.append(centered and label_hit)
        if len(self.servo_state.confirm_buffer) > self.new_config.stop_buffer_size:
            del self.servo_state.confirm_buffer[:-self.new_config.stop_buffer_size]
        if len(self.servo_state.centered_buffer) > self.new_config.stop_buffer_size:
            del self.servo_state.centered_buffer[:-self.new_config.stop_buffer_size]

        if visible:
            self._goal_evidence_step = self.topo_map.current_step
            self.task_telemetry.record_track_buffer_sample(
                sample,
                fresh=bool(packet.fresh_vlm),
                cached=bool(packet.cached_vlm),
            )

    def _record_task_telemetry(self, plan_output: PlanDecision) -> None:
        self.task_telemetry.record_phase(self.nav_phase)
        proposal = getattr(self, "_last_selected_goal_proposal", None)
        self.task_telemetry.record_proposal(proposal)
        route = (plan_output.debug or {}).get("etp_route", {})
        self.task_telemetry.record_reuse_selection(proposal, route, self.nav_phase)
        if (
            self.nav_phase in (NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY)
            and self._etp_stop_evaluated_step == self.topo_map.current_step
        ):
            stop = getattr(self, "_last_stop", None)
            if stop is not None and not stop.should_stop and plan_output.action != "stop":
                self.task_telemetry.record_stop_block(
                    stop.reason,
                    getattr(self, "_last_servo_evidence", None),
                )

    def _plan_servo_phase(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ):
        if not self._servo_allowed_for_current_goal():
            self._transition(NavPhase.GLOBAL_SEARCH, "servo_blocked_no_goal_evidence")
            return self._decision(
                "turn_left",
                "scan",
                None,
                None,
                "no_candidate_scan",
                "servo_blocked_no_goal_evidence",
                candidate_ids,
                scores,
                NavPhase.GLOBAL_SEARCH,
            )

        self._update_goal_evidence_buffer()
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
        self._etp_stop_evaluated_step = self.topo_map.current_step

        verifier = self.stop_verifier
        verify_candidate = (
            verifier.is_verify_candidate(
                self._last_packet, self.goal_manager, servo_evidence,
            )
            if isinstance(verifier, ETPBBoxGrowthStopVerifier)
            else stop.should_stop
        )

        if self.nav_phase == NavPhase.LOCAL_VISUAL_APPROACH:
            if verify_candidate:
                self._transition(NavPhase.STOP_VERIFY, "approach_stop_candidate")
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
                    "approach_stop_candidate",
                    candidate_ids,
                    scores,
                )

        if self.nav_phase == NavPhase.STOP_VERIFY:
            evidence_type = classify_goal_evidence(
                self._last_packet, self.goal_manager, servo_evidence,
            )
            if evidence_type == "far_visible":
                self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "verify_far_visible_reapproach")
            elif stop.should_stop:
                self._transition(NavPhase.STOP, "stop_verified")
                return self._decision(
                    "stop", "stop", None, None, "stop", stop.reason, candidate_ids, scores,
                )

        servo = self.local_servo.act(servo_evidence, self.topo_map.current_step)
        if servo.action == "fail_anchor":
            self.recovery_manager.fail_anchor(
                self.topo_map, self.servo_state.active_anchor_id, servo.reason,
            )
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

        if servo.action in ("turn_left", "turn_right", "move_forward"):
            return self._decision(
                servo.action, "approach_confirm", None, None, "servo", servo.reason,
                candidate_ids, scores,
            )
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
        return self._decision(
            "move_forward", "approach_confirm", None, None, "servo_guard",
            "servo_phase_guard", candidate_ids, scores,
        )

    def _exploration_level(self) -> str:
        if self.topo_map.current_step <= self._goal_enter_step:
            return "normal"
        last = self._goal_evidence_step
        stale = self.topo_map.current_step - last if last >= 0 else self.topo_map.current_step - self._goal_enter_step
        if stale >= self.new_config.no_evidence_escape_steps:
            return "escape"
        if stale >= self.new_config.no_evidence_wide_steps:
            return "wide"
        return "normal"

    def update_memory(self) -> None:
        if isinstance(self.memory_writer, CleanETPMemoryWriter):
            self.memory_writer.exploration_level = self._exploration_level()
        pm = self.perception_manager
        prev_force = getattr(pm, "_force_heavy_reason", None)
        if self._scan_turns_remaining > 0:
            pm._force_heavy_reason = "active_scan_after_waypoint"
        try:
            super().update_memory()
        finally:
            pm._force_heavy_reason = prev_force
        self._update_goal_evidence_buffer()
        self._update_etp_stop_memory()
        if self._target_object_detected_this_scan:
            self._goal_evidence_step = self.topo_map.current_step

    def _active_scan_decision(self, candidate_ids: List[str], scores: List[float]) -> Optional[PlanDecision]:
        if self._scan_turns_remaining <= 0:
            return None
        self._scan_turns_remaining -= 1
        return self._decision(
            "turn_left",
            "scan",
            None,
            self._last_promoted_waypoint_id,
            "waypoint_scan",
            "active_scan_after_waypoint",
            candidate_ids,
            scores,
            NavPhase.GLOBAL_SEARCH,
        )

    def plan(self) -> PlanDecision:
        if self._scan_turns_remaining > 0 and self._target_object_detected_this_scan:
            self._scan_turns_remaining = 0

        scan_decision = self._active_scan_decision([], [])
        if scan_decision is not None:
            return scan_decision

        structure = self.structure_planner.select(self.topo_map, self.goal_manager, self._position)
        proposals = self.navigation_planner.collect_goal_proposals(
            self.topo_map, self.goal_manager, structure, self.nav_phase,
            self._position, self.memory_writer.cur_vp_id,
        )
        proposals.extend(self._collect_hypothesis_goal_proposals())
        scored_proposals = self.navigation_planner.score_goal_proposals(
            proposals, self.goal_manager, self.topo_map, self._position,
            cur_vp_id=self.memory_writer.cur_vp_id,
        )
        selected_proposal = self.navigation_planner.select_best_proposal(
            scored_proposals,
            self.topo_map,
            self.goal_manager,
            self._position,
            self.memory_writer.cur_vp_id,
        )
        if selected_proposal is None:
            self._last_structure = structure
            self._last_goal_proposals = scored_proposals
            self._last_selected_goal_proposal = None
            self._last_nav_target = NavTarget(reason="no_navigation_candidates")
            self._no_candidates_count += 1
            self._transition(NavPhase.GLOBAL_SEARCH, "no_navigation_candidates_scan")
            return self._decision(
                "turn_left",
                "scan",
                None,
                None,
                "no_candidate_scan",
                "no_navigation_candidates",
                [],
                [],
                NavPhase.GLOBAL_SEARCH,
            )

        nav_target = self.navigation_planner.proposal_to_nav_target(
            selected_proposal,
            self.topo_map,
            self._position,
            self.memory_writer.cur_vp_id,
            goal=self.goal_manager,
        )
        ranked = scored_proposals[:10]
        candidate_ids = [p.candidate_node_id for p in ranked]
        scores = [float(p.score) for p in ranked]

        self._last_structure = structure
        self._last_nav_target = nav_target
        self._last_goal_proposals = scored_proposals
        self._last_selected_goal_proposal = selected_proposal
        self._etp_stop_memory_target_id = (
            nav_target.target_node_id
            if nav_target.target_type == "stop_waypoint"
            else None
        )
        if nav_target.reason == "visited_fallback":
            self._visited_fallback_count += 1

        self._maybe_enter_servo_near_anchor(nav_target)

        if nav_target.reason == "at_anchor_waypoint_servo":
            if not self._servo_allowed_for_current_goal():
                self._transition(NavPhase.GLOBAL_SEARCH, "servo_blocked_at_anchor")
            else:
                anchor_id = nav_target.target_node_id
                if anchor_id and self.servo_state.active_anchor_id != anchor_id:
                    self.local_servo.enter(anchor_id, self.topo_map.current_step)
                if self.nav_phase not in (
                    NavPhase.TRACK_TARGET,
                    NavPhase.LOCAL_VISUAL_APPROACH,
                    NavPhase.STOP_VERIFY,
                    NavPhase.STOP,
                ):
                    self._transition(NavPhase.TRACK_TARGET, "at_anchor_waypoint_track")
                if self.nav_phase == NavPhase.TRACK_TARGET:
                    return self._plan_track_phase(nav_target, candidate_ids, scores)
                servo_decision = self._plan_servo_phase(nav_target, candidate_ids, scores)
                if servo_decision is not None:
                    return servo_decision

        phase_age = self.topo_map.current_step - self.phase_enter_step
        if phase_age > self.new_config.phase_timeout_steps:
            if self.nav_phase == NavPhase.ROUTE_TO_STRUCTURE:
                self._transition(NavPhase.GLOBAL_SEARCH, "route_to_structure_timeout")
            elif self.nav_phase == NavPhase.ROUTE_TO_OBJECT_ANCHOR:
                failed_anchor_id = (
                    nav_target.target_node_id
                    if nav_target.target_type in ("object_anchor", "object_approach")
                    else self.servo_state.active_anchor_id
                )
                failed_node = self.topo_map.get_node(failed_anchor_id) if failed_anchor_id else None
                self.recovery_manager.fail_anchor(self.topo_map, failed_anchor_id, "route_timeout")
                self.navigation_planner.add_failed_region(
                    failed_node.position if failed_node is not None else self._position,
                    goal_key=self.goal_manager.goal_key,
                    step=self.topo_map.current_step,
                    reason="route_timeout",
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
            self._transition(
                NavPhase.GLOBAL_SEARCH,
                self.recovery_manager.recovery_reason or "recovery_complete",
            )

        if nav_target.target_node_id and nav_target.target_type in ("object_anchor", "object_approach"):
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "clean_etp_selected_object_anchor")
        elif nav_target.target_node_id and nav_target.target_type == "stop_waypoint":
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "clean_etp_selected_stop_memory")
        elif nav_target.target_node_id and nav_target.target_type in (
            "room", "landmark", "room_summary", "object_summary",
            "goal_region", "context_object",
        ):
            self._transition(NavPhase.ROUTE_TO_STRUCTURE, "clean_etp_selected_structure")
        else:
            self._transition(NavPhase.GLOBAL_SEARCH, "clean_etp_search_ghost")

        decision = self._decision(
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
        decision.debug["etp_route"] = dict(self.navigation_planner._last_route_debug)
        return decision

    def act(self, plan_output: PlanDecision) -> Dict[str, Any]:
        self._record_task_telemetry(plan_output)
        out = super().act(plan_output)
        out["task_telemetry"] = self.task_telemetry.snapshot()
        if plan_output.debug:
            out.setdefault("navigation_debug", {}).update(plan_output.debug)
            etp_route = plan_output.debug.get("etp_route", {})
            out["etp_route_debug"] = etp_route
            if etp_route.get("semantic_reuse_node_id"):
                out.setdefault("semantic_target_node_id", etp_route["semantic_reuse_node_id"])
        return out

    def _plan_track_phase(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
        self._update_goal_evidence_buffer()
        if self._track_ready_for_approach():
            self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "track_target_confirmed")
            servo_decision = self._plan_servo_phase(nav_target, candidate_ids, scores)
            if servo_decision is not None:
                return servo_decision

        phase_age = self.topo_map.current_step - self.phase_enter_step
        if phase_age > self.new_config.track_timeout_steps:
            failed_anchor_id = nav_target.target_node_id or self.servo_state.active_anchor_id
            failed_node = self.topo_map.get_node(failed_anchor_id) if failed_anchor_id else None
            self.recovery_manager.fail_anchor(self.topo_map, failed_anchor_id, "clean_track_timeout")
            if isinstance(self.navigation_planner, CleanETPPlanner):
                self.navigation_planner.add_failed_region(
                    failed_node.position if failed_node is not None else self._position,
                    goal_key=self.goal_manager.goal_key,
                    step=self.topo_map.current_step,
                    reason="track_timeout",
                )
            self.local_servo.reset()
            self._transition(NavPhase.GLOBAL_SEARCH, "clean_track_timeout")
            return self._decision(
                "turn_right",
                "recovery",
                None,
                None,
                "track_timeout",
                "clean_track_timeout",
                candidate_ids,
                scores,
                NavPhase.GLOBAL_SEARCH,
            )

        return self._track_scan_decision(nav_target, candidate_ids, scores)

    def on_navigation_event(self, target_node_id: Optional[str], event: str) -> Dict[str, Any]:
        node = self.topo_map.get_node(target_node_id) if target_node_id else None
        before_type = node.node_type if node is not None else None
        if node is not None and node.node_type == NodeType.WAYPOINT_CANDIDATE:
            node.attributes["visit_attempts"] = int(node.attributes.get("visit_attempts", 0)) + 1
            if event == "target_reached":
                self.topo_map.promote_frontier_to_visited(node.node_id)
                self._transition(NavPhase.GLOBAL_SEARCH, "clean_etp_candidate_promoted")
                out = {
                    "target_node_id": target_node_id,
                    "event": event,
                    "action": "candidate_promoted",
                    "reason": "clean_etp_candidate_reached",
                }
            elif event in ("unreachable", "snap_failed", "not_navigable", "collision_blocked", "no_progress_multigoal"):
                node.attributes["blocked_count"] = int(node.attributes.get("blocked_count", 0)) + 1
                node.attributes["blacklisted_until"] = (
                    self.topo_map.current_step + self.new_config.candidate_block_ttl
                )
                self.topo_map.consume_node(node.node_id, event)
                self._transition(NavPhase.GLOBAL_SEARCH, "clean_etp_candidate_consumed")
                out = {
                    "target_node_id": target_node_id,
                    "event": event,
                    "action": "consumed_candidate",
                    "reason": event,
                }
            else:
                out = {
                    "target_node_id": target_node_id,
                    "event": event,
                    "action": "candidate_event_ignored",
                    "reason": event,
                }
        elif (
            node is not None
            and node.node_type == NodeType.WAYPOINT_VISITED
            and node.attributes.get("goal_stop_score") is not None
            and target_node_id == self._etp_stop_memory_target_id
            and event == "target_reached"
        ):
            anchor_id = node.attributes.get("goal_stop_object_id")
            anchor_ok = bool(
                anchor_id
                and self.topo_map.get_node(anchor_id) is not None
                and self._anchor_matches_current_goal(str(anchor_id))
            )
            if anchor_ok:
                self.local_servo.enter(str(anchor_id), self.topo_map.current_step)
                self._transition(NavPhase.TRACK_TARGET, "clean_etp_stop_waypoint_reached")
                out = {
                    "target_node_id": target_node_id,
                    "event": event,
                    "action": "anchor_reached_awaiting_confirm",
                    "reason": "clean_etp_stop_memory_requires_fresh_confirm",
                    "goal_stop_score": float(node.attributes.get("goal_stop_score", 0.0)),
                    "linked_object_anchor_id": anchor_id,
                }
            else:
                self._scan_turns_remaining = int(self.new_config.active_scan_turns)
                self._transition(NavPhase.GLOBAL_SEARCH, "stop_memory_scan_no_linked_anchor")
                out = {
                    "target_node_id": target_node_id,
                    "event": event,
                    "action": "stop_viewpoint_scan_only",
                    "reason": "stop_memory_no_linked_anchor",
                    "goal_stop_score": float(node.attributes.get("goal_stop_score", 0.0)),
                    "linked_object_anchor_id": anchor_id,
                }
        elif (
            node is not None
            and node.node_type == NodeType.WAYPOINT_VISITED
            and node.attributes.get("goal_stop_score") is not None
            and target_node_id == self._etp_stop_memory_target_id
            and event in ("unreachable", "snap_failed", "not_navigable", "collision_blocked", "no_progress_multigoal")
        ):
            node.attributes["stop_blacklisted_until"] = (
                self.topo_map.current_step + self.new_config.stop_memory_blacklist_steps
            )
            node.attributes["goal_stop_score"] = max(0.0, float(node.attributes.get("goal_stop_score", 0.0)) - 0.25)
            self._transition(NavPhase.GLOBAL_SEARCH, "clean_etp_stop_waypoint_failed")
            out = {
                "target_node_id": target_node_id,
                "event": event,
                "action": "blacklisted_stop_waypoint",
                "reason": event,
            }
        else:
            out = super().on_navigation_event(target_node_id, event)

        if out.get("action") == "candidate_promoted" and before_type == NodeType.WAYPOINT_CANDIDATE:
            self._last_promoted_waypoint_id = target_node_id
            self._scan_turns_remaining = int(self.new_config.active_scan_turns)

        if event in ("unreachable", "snap_failed", "not_navigable", "collision_blocked", "no_progress_multigoal"):
            failed_pos = node.position if node is not None else self._position
            if isinstance(self.navigation_planner, CleanETPPlanner):
                self.navigation_planner.add_failed_region(
                    failed_pos,
                    goal_key=self.goal_manager.goal_key,
                    step=self.topo_map.current_step,
                    reason=str(event),
                )
        return out

    def _update_etp_stop_memory(self) -> None:
        if not self.new_config.stop_memory_enabled:
            return
        cur_vp = self.memory_writer.cur_vp_id
        if not cur_vp:
            return
        node = self.topo_map.get_node(cur_vp)
        if node is None or node.node_type != NodeType.WAYPOINT_VISITED:
            return
        score, evidence = self._compute_etp_stop_score()
        anchor = self._best_stop_anchor()
        if self.new_config.stop_memory_require_anchor and anchor is None:
            return
        if float(evidence.get("bbox_score", 0.0)) < self.new_config.stop_memory_min_bbox_score:
            return
        if score < self.new_config.stop_memory_min_write_score:
            old = float(node.attributes.get("goal_stop_score", 0.0))
            if old > 0.0:
                node.attributes["goal_stop_score"] = max(0.0, old * 0.96)
            return
        old = float(node.attributes.get("goal_stop_score", 0.0))
        node.attributes["goal_stop_score"] = max(old * 0.90, score)
        node.attributes["goal_stop_step"] = self.topo_map.current_step
        node.attributes["goal_key"] = self.goal_manager.goal_key
        node.attributes["goal_stop_evidence"] = evidence
        if anchor is not None:
            node.attributes["goal_stop_object_id"] = anchor.node_id
            node.attributes["goal_stop_object_label"] = anchor.label

    def _compute_etp_stop_score(self) -> Tuple[float, Dict[str, Any]]:
        packet = self._last_packet
        report = packet.report
        best = max(
            (o for o in report.objects if _label_matches(o.label, self.goal_manager.target_labels)),
            key=lambda o: _bbox_area(o.bbox) + float(o.confidence),
            default=None,
        )
        goal_visible = bool(report.goal_visible or best is not None)
        confidence = max(
            float(report.goal_match_confidence),
            float(best.confidence) if best is not None else 0.0,
            float(report.best_goal_sim),
        )
        bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        bbox_score = min(bbox_area / max(self.new_config.bbox_min_stop, 1e-6), 1.0)
        bearing = (
            report.target_direction
            if report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        )
        range_bin = str(best.range_bin if best is not None else "unknown").lower()
        centered = str(bearing).lower() in _CENTER_BEARINGS
        close = range_bin in _STOP_RANGE_BINS or str(report.apparent_scale).lower() in ("large", "close")
        stop_candidate = bool(report.stop_candidate)
        uncertainty = float(report.uncertainty)
        score = (
            0.18 * float(goal_visible)
            + 0.25 * confidence
            + 0.18 * bbox_score
            + 0.12 * float(centered)
            + 0.15 * float(close)
            + 0.22 * float(stop_candidate)
            - 0.18 * uncertainty
        )
        score = float(max(0.0, min(1.0, score)))
        evidence = {
            "goal_visible": goal_visible,
            "confidence": confidence,
            "bbox_area": bbox_area,
            "bbox_score": bbox_score,
            "bearing": str(bearing).lower(),
            "centered": centered,
            "range_bin": range_bin,
            "close": close,
            "stop_candidate": stop_candidate,
            "uncertainty": uncertainty,
            "fresh_vlm": bool(packet.fresh_vlm),
        }
        return score, evidence

    def _best_stop_anchor(self) -> Optional[SemanticNode]:
        if self.servo_state.active_anchor_id:
            node = self.topo_map.get_node(self.servo_state.active_anchor_id)
            if node is not None:
                return node
        return self.navigation_planner._best_object_anchor(
            self.topo_map,
            self.goal_manager,
            self._position,
        )

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
        decision = super()._decision(
            action,
            plan_action,
            target_position,
            target_node_id,
            target_type,
            reason,
            candidate_ids,
            scores,
            expected,
        )
        decision.is_exploration = target_type in ("frontier", "waypoint_candidate")
        return decision

    @property
    def memory_stats(self) -> Dict[str, Any]:
        stats = super().memory_stats
        stats["clean_etpnav"] = {
            "exploration_level": self._exploration_level(),
            "scan_turns_remaining": self._scan_turns_remaining,
            "failed_region_count": (
                len(self.navigation_planner.failed_regions)
                if isinstance(self.navigation_planner, CleanETPPlanner)
                else 0
            ),
            "goal_evidence_step": self._goal_evidence_step,
        }
        return stats


ConfTopoGOATAgentCleanETPNav = ConfTopoGOATCleanETPNavAgent


__all__ = [
    "CleanETPGoatConfig",
    "CleanETPTaskTelemetry",
    "CleanETPMemoryWriter",
    "CleanETPPlanner",
    "ConfTopoGOATCleanETPNavAgent",
    "ConfTopoGOATAgentCleanETPNav",
]
