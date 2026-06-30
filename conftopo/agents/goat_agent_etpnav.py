"""RGB-only GOAT agent with an ETPNav-style waypoint/ghost graph.

This variant borrows ETPNav's navigation *structure* without copying its
RGB-D waypoint predictor or VLN-BERT planner:

    local ghost candidates -> online topo graph -> graph-aware scoring
    -> route to anchor waypoint -> local advance / visual servo

The semantic memory, confidence updates, room/object summaries, and visual
servo come from ``goat_agent_new``.  Only the navigation-layer candidate
generation and routing policy are replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType, SemanticNode
from conftopo.core.instruction_graph import GoalProposal
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
    """Config for the RGB-only ETP-lite navigation layer."""

    # Rule-based RGB-only candidate rays.  Distances are meters in the
    # agent/world frame; heading deltas are radians relative to current heading.
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

    # ETPNav-style stop memory: record a STOP score on visited waypoints,
    # then route back to the best historical stop waypoint for fresh
    # visual confirmation.  This is GOAT-specific, so it does not use
    # R2R ground-truth distance.
    stop_memory_enabled: bool = True
    stop_memory_min_write_score: float = 0.36
    stop_memory_proposal_threshold: float = 0.68
    stop_memory_current_radius: float = 0.65
    stop_memory_blacklist_steps: int = 35
    stop_memory_score_weight: float = 0.85
    stop_memory_min_bbox_score: float = 0.65
    stop_memory_require_anchor: bool = True
    track_timeout_reroute_steps: int = 35
    track_near_anchor_vlm_distance: float = 2.5
    prefer_goal_anchor_bonus: float = 1.25
    no_anchor_object_ghost_boost: float = 1.0
    materialize_clip_threshold: float = 0.18
    materialize_room_prior_clip_threshold: float = 0.14
    materialize_alias_min_confidence: float = 0.16
    ghost_rays_expanded: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 2.5),
        (0.0, 4.0),
        (0.0, 5.5),
        (math.radians(45.0), 4.0),
        (math.radians(-45.0), 4.0),
        (math.radians(90.0), 3.0),
        (math.radians(-90.0), 3.0),
    ])
    explore_stuck_steps: int = 12
    explore_stuck_min_move: float = 0.12
    explore_region_blacklist_radius: float = 2.5
    explore_region_blacklist_steps: int = 40
    ghost_arrival_scan_turns: int = 3
    stop_cached_max_age_steps: int = 2
    stop_cached_max_position_delta: float = 0.35
    stop_cached_max_heading_delta_deg: float = 20.0


def _default_phase_counts() -> Dict[str, int]:
    return {
        "SEARCH": 0,
        "ROUTE_TO_OBJECT_ANCHOR": 0,
        "TRACK": 0,
        "APPROACH": 0,
        "STOP_VERIFY": 0,
    }


def _default_action_counts() -> Dict[str, int]:
    return {
        "forward": 0,
        "turn_left": 0,
        "turn_right": 0,
        "stop": 0,
    }


def _default_stop_block_reasons() -> Dict[str, int]:
    return {
        "no_fresh_target": 0,
        "not_centered": 0,
        "bbox_too_small": 0,
        "no_growth": 0,
        "not_close": 0,
        "no_approach_progress": 0,
        "retreating": 0,
        "no_multi_frame_confirm": 0,
    }


def _default_proposal_source_counts() -> Dict[str, int]:
    return {
        "frontier": 0,
        "ghost": 0,
        "semantic_anchor": 0,
        "object_memory": 0,
        "object_anchor": 0,
    }


def _default_track_buffer_stats() -> Dict[str, int]:
    return {
        "visible_count": 0,
        "centered_count": 0,
        "bbox_count": 0,
        "cached_count": 0,
        "fresh_count": 0,
    }


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


def _telemetry_action_key(low_action: str) -> Optional[str]:
    action = str(low_action).lower().strip()
    if action == "move_forward":
        return "forward"
    if action in ("turn_left", "turn_right", "stop"):
        return action
    return None


def _telemetry_proposal_source(proposal: GoalProposal) -> Optional[str]:
    source = str(proposal.source or "").lower().strip()
    ctype = str(proposal.candidate_type or "").lower().strip()
    if source == "frontier":
        return "frontier"
    if source == "ghost_candidate" or ctype == "waypoint_candidate":
        return "ghost"
    if source == "semantic_memory" and ctype == "object_anchor":
        return "semantic_anchor"
    if ctype == "object_anchor":
        return "object_anchor"
    if source == "object_memory":
        return "object_memory"
    return None


def _telemetry_stop_block_reason(
    reason: str,
    servo_evidence: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    evidence = servo_evidence or {}
    raw = str(reason or "").lower().strip()
    if raw == "no_fresh_target" or raw == "target_not_clear":
        return "no_fresh_target"
    if raw == "far_visible_no_stop":
        return "far_visible_no_stop"
    if raw == "target_not_centered":
        return "not_centered"
    if raw == "fresh_bbox_too_small":
        return "bbox_too_small"
    if raw in ("need_growth_or_strong_multiframe", "no_proximity_evidence"):
        return "no_growth"
    if raw == "not_in_stop_band":
        return "not_close"
    if raw == "approach_not_enough":
        return "no_approach_progress"
    if raw == "need_multiframe_confirm":
        return "no_multi_frame_confirm"
    if raw == "not_approaching":
        if bool(evidence.get("retreating")):
            return "retreating"
        return "no_approach_progress"
    return None


@dataclass
class ETPTaskTelemetry:
    """Per-goal counters for diagnosing ETPNav navigation failures."""

    phase_counts: Dict[str, int] = field(default_factory=_default_phase_counts)
    action_counts: Dict[str, int] = field(default_factory=_default_action_counts)
    stop_block_reasons: Dict[str, int] = field(default_factory=_default_stop_block_reasons)
    proposal_source_counts: Dict[str, int] = field(default_factory=_default_proposal_source_counts)
    track_buffer_stats: Dict[str, int] = field(default_factory=_default_track_buffer_stats)

    def reset(self) -> None:
        self.phase_counts = _default_phase_counts()
        self.action_counts = _default_action_counts()
        self.stop_block_reasons = _default_stop_block_reasons()
        self.proposal_source_counts = _default_proposal_source_counts()
        self.track_buffer_stats = _default_track_buffer_stats()

    def record_phase(self, phase: NavPhase) -> None:
        key = _telemetry_phase_key(phase)
        if key is not None:
            self.phase_counts[key] += 1

    def record_action(self, low_action: str) -> None:
        key = _telemetry_action_key(low_action)
        if key is not None:
            self.action_counts[key] += 1

    def record_proposal(self, proposal: Optional[GoalProposal]) -> None:
        if proposal is None:
            return
        key = _telemetry_proposal_source(proposal)
        if key is not None:
            self.proposal_source_counts[key] += 1

    def record_stop_block(
        self,
        reason: str,
        servo_evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = _telemetry_stop_block_reason(reason, servo_evidence)
        if key is not None:
            self.stop_block_reasons[key] += 1

    def record_track_buffer_sample(
        self,
        sample: Dict[str, Any],
        *,
        fresh: bool,
        cached: bool,
    ) -> None:
        if bool(sample.get("visible")):
            self.track_buffer_stats["visible_count"] += 1
        if bool(sample.get("centered")):
            self.track_buffer_stats["centered_count"] += 1
        if bool(sample.get("bbox_valid")):
            self.track_buffer_stats["bbox_count"] += 1
        if fresh:
            self.track_buffer_stats["fresh_count"] += 1
        if cached:
            self.track_buffer_stats["cached_count"] += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "phase_counts": dict(self.phase_counts),
            "action_counts": dict(self.action_counts),
            "stop_block_reasons": dict(self.stop_block_reasons),
            "proposal_source_counts": dict(self.proposal_source_counts),
            "track_buffer_stats": dict(self.track_buffer_stats),
        }


class ETPBBoxGrowthStopVerifier(StopVerifier):
    """ETPNav stop policy: require visual approach progress, then fresh confirm."""

    def can_stop(
        self,
        packet,
        goal: GoalManager,
        position: Optional[np.ndarray],
        servo_evidence: Optional[Dict[str, Any]] = None,
    ) -> StopDecision:
        evidence = servo_evidence or {}
        goal_obs = [
            o for o in packet.report.objects
            if _label_matches(o.label, goal.target_labels)
        ]
        best = max(goal_obs, key=lambda o: _bbox_area(o.bbox), default=None)
        bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        bearing = str(
            packet.report.target_direction
            if packet.report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        ).lower()
        range_bin = str(best.range_bin if best is not None else evidence.get("range_bin", "unknown")).lower()

        current_target_ok = bool(
            packet.fresh_vlm
            and (packet.report.goal_visible or best is not None)
            and float(packet.report.goal_match_confidence) >= 0.50
        )
        cache_age = int(packet.vlm_cache_age) if packet.vlm_cache_age is not None else 999
        position_delta = (
            float(packet.vlm_position_delta)
            if packet.vlm_position_delta is not None
            else 999.0
        )
        yaw_delta_deg = (
            math.degrees(abs(float(packet.vlm_heading_delta)))
            if packet.vlm_heading_delta is not None
            else 999.0
        )
        recent_visible = sum(self.state.confirm_buffer[-self.cfg.stop_buffer_size:])
        recent_centered = sum(self.state.centered_buffer[-self.cfg.stop_buffer_size:])
        recent_close = sum(self.state.close_buffer[-self.cfg.stop_buffer_size:])
        recent_target_ok = bool(
            packet.cached_vlm
            and (packet.report.goal_visible or best is not None)
            and float(packet.report.goal_match_confidence) >= 0.50
            and cache_age <= int(self.cfg.stop_cached_max_age_steps)
            and position_delta <= float(self.cfg.stop_cached_max_position_delta)
            and yaw_delta_deg <= float(self.cfg.stop_cached_max_heading_delta_deg)
            and recent_visible >= 3
            and recent_centered >= 3
        )
        final_visual_ok = bool(current_target_ok or recent_target_ok)
        centered = bearing in _CENTER_BEARINGS
        clear = packet.report.target_visibility == "clear"
        multi_frame_confirm = bool(
            recent_visible >= self.cfg.servo_entry_evidence
            and recent_centered >= self.cfg.servo_entry_evidence
        )
        strong_multiframe_confirm = bool(recent_visible >= 3 and recent_centered >= 3)
        visual_ok = bool(
            final_visual_ok
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
        no_growth_close_ok = bool(
            recent_visible >= 3
            and recent_centered >= 3
            and recent_close >= 1
            and self.state.forward_action_count >= self.cfg.min_forward_before_stop
            and self.state.approach_travel_distance >= self.cfg.min_approach_distance
            and not retreating
        )
        proximity_ok = bool(growth_ok or plateau_ok or range_close_ok or stop_candidate)
        confidence_ok = bool(
            growth_ok
            or strong_multiframe_confirm
            or (range_close_ok and multi_frame_confirm and float(packet.report.goal_match_confidence) >= 0.55)
        )
        approach_progress_ok = bool(
            approach_ok
            and not retreating
            and progress_ok
            and (growth_ok or no_growth_close_ok or (proximity_ok and confidence_ok))
        )
        stop_band = bool(
            plateau_ok
            or range_close_ok
            or stop_candidate
        )
        verify_ok = bool(stop_band and multi_frame_confirm)

        should = bool(visual_ok and approach_progress_ok and verify_ok)
        if should:
            reason = "etp_bbox_growth_stop"
        elif not final_visual_ok:
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


class ETPNavMemoryWriter(MemoryWriter):
    """Writes ETPNav-style ghost candidates instead of four coarse frontiers."""

    def __init__(self, new_config: ETPGoatConfig) -> None:
        super().__init__(new_config)
        self.cfg: ETPGoatConfig = new_config
        self.ghost_escalation_level: int = 0

    def reset_goal(self) -> None:
        super().reset_goal()
        self.ghost_escalation_level = 0

    def _active_ghost_rays(self) -> List[Tuple[float, float]]:
        if self.ghost_escalation_level >= 1:
            return list(self.cfg.ghost_rays_expanded)
        return list(self.cfg.ghost_rays)

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

        for heading_delta, distance in self._active_ghost_rays():
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


class ETPGraphNavigationPlanner(NavigationPlanner):
    """Graph-aware proposal planner for visited/ghost waypoint navigation."""

    def __init__(self, new_config: ETPGoatConfig) -> None:
        super().__init__(new_config)
        self.cfg: ETPGoatConfig = new_config
        self._last_route_debug: Dict[str, Any] = {}
        self.prefer_goal_anchor_until_step: int = 0
        self.prefer_goal_anchor_reroute: bool = False
        self.explore_blacklist_center: Optional[np.ndarray] = None
        self.explore_blacklist_until_step: int = -1

    def _candidate_in_explore_blacklist(
        self,
        topo_map: DynamicTopoMap,
        node: SemanticNode,
    ) -> bool:
        if topo_map.current_step > self.explore_blacklist_until_step:
            return False
        center = self.explore_blacklist_center
        if center is None:
            return False
        return float(np.linalg.norm(node.position - center)) <= float(
            self.cfg.explore_region_blacklist_radius
        )

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

    def _collect_frontier_proposals(
        self,
        topo_map: DynamicTopoMap,
        goal: GoalManager,
        structure: StructureTarget,
        position: Optional[np.ndarray],
    ) -> List[GoalProposal]:
        # Keep legacy frontiers as a fallback only; RGB-rule ghosts are the
        # primary exploration surface.
        return super()._collect_frontier_proposals(topo_map, goal, structure, position)

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
            if self._candidate_in_explore_blacklist(topo_map, node):
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
                # Current local servo/stop verifier should handle this pose.
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
        prefer_goal = topo_map.current_step <= self.prefer_goal_anchor_until_step
        no_goal_anchor = not _goal_has_active_anchor(topo_map, goal, self.cfg)
        for p in scored:
            node = topo_map.get_node(p.candidate_node_id)
            if node is None:
                continue
            if p.source == "object_memory":
                bonus = self.cfg.prefer_goal_anchor_bonus if prefer_goal else 0.0
                if no_goal_anchor:
                    bonus += 0.35
                p.score = float(p.score) + bonus
            elif (
                p.source == "semantic_memory"
                and node.attributes.get("semantic_role") == "object_anchor"
                and _label_matches(node.label, goal.target_labels)
            ):
                bonus = self.cfg.prefer_goal_anchor_bonus if prefer_goal else 0.20
                p.score = float(p.score) + bonus
                p.candidate_type = "object_anchor"
            elif p.source == "ghost_candidate" and no_goal_anchor:
                p.score = float(p.score) + self.cfg.no_anchor_object_ghost_boost * float(
                    (p.evidence_refs[-1] if p.evidence_refs else {}).get("semantic_context_score", 0.0)
                )
        return sorted(scored, key=lambda p: p.score, reverse=True)

    def select_best_proposal(
        self, proposals: List[GoalProposal],
    ) -> Optional[GoalProposal]:
        active = [p for p in proposals if p.status in ("active", "needs_verify", "confirmed", "preserved")]
        if not active:
            return None
        if self.prefer_goal_anchor_reroute:
            object_proposals = [p for p in active if p.source == "object_memory"]
            if object_proposals:
                return max(object_proposals, key=lambda p: p.score)
            semantic_anchors = [
                p for p in active
                if p.source == "semantic_memory" and p.candidate_type == "object_anchor"
            ]
            if semantic_anchors:
                return max(semantic_anchors, key=lambda p: p.score)
        return super().select_best_proposal(proposals)

    def proposal_to_nav_target(
        self,
        proposal: GoalProposal,
        topo_map: DynamicTopoMap,
        position: Optional[np.ndarray],
        cur_vp_id: Optional[str],
    ) -> NavTarget:
        base = super().proposal_to_nav_target(proposal, topo_map, position, cur_vp_id)
        node = topo_map.get_node(proposal.candidate_node_id)
        if node is None:
            self._last_route_debug = {}
            return base

        if proposal.source == "object_memory" and node.attributes.get("semantic_role") == "object_anchor":
            route_cost, route_path, anchor_id = self._route_cost_to_node(
                topo_map, cur_vp_id, node.node_id, position,
            )
            target_position = self._object_anchor_route_position(topo_map, node, cur_vp_id, route_path)
            self._last_route_debug = {
                "selected_node_id": proposal.candidate_node_id,
                "selected_type": "object_anchor",
                "selected_source": proposal.source,
                "selected_candidate_type": "object_anchor",
                "semantic_reuse_node_id": proposal.candidate_node_id,
                "anchor_waypoint_id": anchor_id,
                "route_path": route_path,
                "route_cost": route_cost if np.isfinite(route_cost) else None,
                "next_subgoal_position": target_position.tolist() if target_position is not None else None,
            }
            return NavTarget(
                target_position=target_position.copy() if target_position is not None else None,
                target_node_id=proposal.candidate_node_id,
                target_type="object_anchor",
                reason="etp_object_anchor_via_waypoint",
                expected_phase_after_reach=NavPhase.LOCAL_VISUAL_APPROACH,
            )

        route_cost, route_path, anchor_id = self._route_cost_to_node(
            topo_map, cur_vp_id, node.node_id, position,
        )
        target_position = base.target_position
        reason = base.reason
        expected = base.expected_phase_after_reach
        target_type = base.target_type

        if node.node_type == NodeType.WAYPOINT_CANDIDATE:
            target_type = "waypoint_candidate"
            reason = "etp_ghost_candidate"
            expected = NavPhase.GLOBAL_SEARCH
            target_position = self._next_route_position(topo_map, cur_vp_id, route_path, node)
        elif proposal.source == "stop_memory":
            target_type = "stop_waypoint"
            reason = "etp_stop_memory_route"
            expected = NavPhase.TRACK_TARGET
            target_position = self._next_route_position(topo_map, cur_vp_id, route_path, node)
        elif node.attributes.get("semantic_role") == "object_anchor":
            # Route to the observing waypoint first; visual servo handles the
            # final RGB-only confirmation.
            target_position = self._object_anchor_route_position(topo_map, node, cur_vp_id, route_path)
            reason = "etp_object_anchor_via_waypoint"
            expected = NavPhase.LOCAL_VISUAL_APPROACH
        elif proposal.source in ("semantic_memory", "room", "structure") or target_type in (
            "room", "landmark", "room_summary", "goal_region", "context_object",
        ):
            target_position = self._next_route_position(topo_map, cur_vp_id, route_path, node)
            reason = "etp_semantic_route"
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
            "semantic_reuse_node_id": semantic_reuse_node_id,
            "anchor_waypoint_id": anchor_id,
            "route_path": route_path,
            "route_cost": route_cost if np.isfinite(route_cost) else None,
            "next_subgoal_position": target_position.tolist() if target_position is not None else None,
        }
        return NavTarget(
            target_position=target_position.copy() if target_position is not None else None,
            target_node_id=proposal.candidate_node_id,
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
        """Return the semantic memory node most responsible for a ghost score."""
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
        goal_labels = {x.lower() for x in goal.target_labels}
        for obj in topo_map.get_nodes_by_type(NodeType.OBJECT):
            role = obj.attributes.get("semantic_role")
            if role not in ("context_object", "environment_object", "object_anchor"):
                continue
            if goal_labels and _label_matches(obj.label, goal_labels):
                dist = float(np.linalg.norm(node.position - obj.position))
                if dist < 12.0:
                    update(obj.node_id, max(0.0, 1.0 - dist / 12.0) + 0.35)
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
        # Route paths include cur_vp as the first node when available.  If the
        # final node is a candidate, the previous node is its anchor; once the
        # agent is already at that anchor, return the candidate position.
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


class ConfTopoGOATETPNavAgent(GoatAgent):
    """GOAT agent using an ETPNav-style RGB-only graph navigation layer."""

    def __init__(
        self,
        config: Optional[ConfTopoConfig] = None,
        new_config: Optional[ETPGoatConfig] = None,
    ):
        etp_config = new_config or ETPGoatConfig()
        super().__init__(config=config, new_config=etp_config)
        self.new_config: ETPGoatConfig = etp_config
        self.memory_writer = ETPNavMemoryWriter(self.new_config)
        self.navigation_planner = ETPGraphNavigationPlanner(self.new_config)
        self.stop_verifier = ETPBBoxGrowthStopVerifier(self.new_config, self.servo_state)
        self._etp_stop_memory_target_id: Optional[str] = None
        self._etp_prefer_goal_anchor_until: int = 0
        self.task_telemetry = ETPTaskTelemetry()
        self._etp_stop_evaluated_step: int = -1
        self._explore_stuck_steps: int = 0
        self._explore_last_position: Optional[np.ndarray] = None
        self._explore_anchor_best_distance: Optional[float] = None
        self._explore_pending_turn: bool = False
        self._ghost_scan_turns_left: int = 0
        self._ghost_scan_turn_right: bool = True

    def set_new_goal(self, goal):
        super().set_new_goal(goal)
        self.task_telemetry.reset()
        self._reset_explore_escape_state()

    def _reset_explore_escape_state(self) -> None:
        self._explore_stuck_steps = 0
        self._explore_last_position = None
        self._explore_anchor_best_distance = None
        self._explore_pending_turn = False
        self._ghost_scan_turns_left = 0
        self.memory_writer.ghost_escalation_level = 0
        self.navigation_planner.explore_blacklist_center = None
        self.navigation_planner.explore_blacklist_until_step = -1

    def _in_explore_phase(self) -> bool:
        return self.nav_phase in (
            NavPhase.GLOBAL_SEARCH,
            NavPhase.ROUTE_TO_STRUCTURE,
            NavPhase.RECOVERY,
        )

    def _had_explore_progress_this_step(self) -> bool:
        packet = self._last_packet
        memory = self._last_memory
        report = packet.report
        if report.goal_visible:
            return True
        if any(_label_matches(o.label, self.goal_manager.target_labels) for o in report.objects):
            return True
        if memory.goal_anchor_written or memory.object_merge_count_this_step > 0:
            return True
        for node_id in memory.written_node_ids:
            node = self.topo_map.get_node(node_id)
            if node is not None and _label_matches(node.label, self.goal_manager.target_labels):
                return True
        if self._position is not None and self._explore_last_position is not None:
            moved = float(np.linalg.norm(self._position - self._explore_last_position))
            if moved >= float(self.new_config.explore_stuck_min_move):
                return True
        anchor = self.navigation_planner._best_object_anchor(
            self.topo_map, self.goal_manager, self._position,
        )
        if anchor is not None and self._position is not None:
            dist = float(np.linalg.norm(anchor.position - self._position))
            if self._explore_anchor_best_distance is None or dist < self._explore_anchor_best_distance - 0.2:
                self._explore_anchor_best_distance = dist
                return True
        return False

    def _update_explore_stuck_state(self) -> None:
        if not self._in_explore_phase() or self._ghost_scan_turns_left > 0:
            self._explore_stuck_steps = 0
            if self._position is not None:
                self._explore_last_position = self._position.copy()
            return
        if self._had_explore_progress_this_step():
            self._explore_stuck_steps = 0
            if self.memory_writer.ghost_escalation_level > 0:
                self.memory_writer.ghost_escalation_level = 0
            if self._position is not None:
                self._explore_last_position = self._position.copy()
            return
        self._explore_stuck_steps += 1
        if self._explore_stuck_steps < int(self.new_config.explore_stuck_steps):
            if self._position is not None:
                self._explore_last_position = self._position.copy()
            return
        self._explore_stuck_steps = 0
        if self.memory_writer.ghost_escalation_level < 1:
            self.memory_writer.ghost_escalation_level = 1
            self._explore_pending_turn = True
        elif self._position is not None:
            center = self._position.copy()
            until = self.topo_map.current_step + int(self.new_config.explore_region_blacklist_steps)
            self.navigation_planner.explore_blacklist_center = center
            self.navigation_planner.explore_blacklist_until_step = until
            for node in self.topo_map.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE):
                if float(np.linalg.norm(node.position - center)) <= float(
                    self.new_config.explore_region_blacklist_radius
                ):
                    node.attributes["blacklisted_until"] = until
            self._explore_pending_turn = True
        if self._position is not None:
            self._explore_last_position = self._position.copy()

    def _begin_ghost_arrival_scan(self) -> None:
        self._ghost_scan_turns_left = int(self.new_config.ghost_arrival_scan_turns)
        self._ghost_scan_turn_right = True
        self._explore_stuck_steps = 0

    def _plan_ghost_arrival_scan(
        self,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
        turn = "turn_right" if self._ghost_scan_turn_right else "turn_left"
        self._ghost_scan_turn_right = not self._ghost_scan_turn_right
        self._ghost_scan_turns_left = max(0, self._ghost_scan_turns_left - 1)
        return self._decision(
            turn,
            "ghost_scan",
            None,
            None,
            "servo",
            "ghost_arrival_scan",
            candidate_ids,
            scores,
            NavPhase.GLOBAL_SEARCH,
        )

    def record_executed_action(self, low_action: str) -> None:
        self.task_telemetry.record_action(low_action)

    def _record_task_telemetry(self, plan_output: PlanDecision) -> None:
        self.task_telemetry.record_phase(self.nav_phase)
        self.task_telemetry.record_proposal(getattr(self, "_last_selected_goal_proposal", None))
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
        decision = super()._plan_servo_phase(nav_target, candidate_ids, scores)
        self._etp_stop_evaluated_step = self.topo_map.current_step
        return decision

    def _sync_prefer_goal_anchor_window(self) -> None:
        self.navigation_planner.prefer_goal_anchor_until_step = int(self._etp_prefer_goal_anchor_until)
        self.navigation_planner.prefer_goal_anchor_reroute = (
            self.topo_map.current_step <= self._etp_prefer_goal_anchor_until
        )

    def _begin_goal_anchor_reroute(self, reason: str) -> None:
        until = self.topo_map.current_step + int(self.new_config.track_timeout_reroute_steps)
        self._etp_prefer_goal_anchor_until = max(self._etp_prefer_goal_anchor_until, until)
        self._sync_prefer_goal_anchor_window()

    def _should_force_track_vlm(self) -> bool:
        if self.nav_phase not in (
            NavPhase.TRACK_TARGET,
            NavPhase.LOCAL_VISUAL_APPROACH,
            NavPhase.STOP_VERIFY,
        ):
            return False
        anchor_id = self.servo_state.active_anchor_id
        if anchor_id is None:
            return False
        anchor = self.topo_map.get_node(anchor_id)
        if anchor is None or self._position is None:
            return False
        dist = float(np.linalg.norm(anchor.position - self._position))
        return dist <= float(self.new_config.track_near_anchor_vlm_distance)

    def _update_track_buffer(self) -> None:
        packet = self._last_packet
        before_len = len(self.servo_state.track_buffer)
        if packet.fresh_vlm:
            super()._update_track_buffer()
            if len(self.servo_state.track_buffer) > before_len:
                self.task_telemetry.record_track_buffer_sample(
                    self.servo_state.track_buffer[-1],
                    fresh=True,
                    cached=False,
                )
            return
        if self.nav_phase not in (NavPhase.TRACK_TARGET, NavPhase.LOCAL_VISUAL_APPROACH, NavPhase.STOP_VERIFY):
            return
        if not packet.cached_vlm:
            return
        report = packet.report
        best = max(
            (o for o in report.objects if _label_matches(o.label, self.goal_manager.target_labels)),
            key=lambda o: _bbox_area(o.bbox) + float(o.confidence),
            default=None,
        )
        visible = bool(report.goal_visible or best is not None)
        if not visible and float(report.goal_match_confidence) < 0.45:
            return
        if (
            self.servo_state.track_buffer
            and int(self.servo_state.track_buffer[-1].get("step_id", -1)) == int(packet.step_id)
        ):
            return
        bbox_area = _bbox_area(best.bbox) if best is not None else 0.0
        bearing = (
            report.target_direction
            if report.target_direction != "unknown"
            else (best.bearing if best is not None else "unknown")
        )
        range_bin = str(best.range_bin if best is not None else "unknown").lower()
        sample = {
            "step_id": int(packet.step_id),
            "visible": visible,
            "bbox_valid": bbox_area > 0.0,
            "confidence": float(best.confidence) if best is not None else float(report.goal_match_confidence),
            "centered": str(bearing).lower() in _CENTER_BEARINGS,
            "bearing": str(bearing).lower(),
            "range_bin": range_bin,
        }
        self.servo_state.track_buffer.append(sample)
        if len(self.servo_state.track_buffer) > self.new_config.track_buffer_size:
            del self.servo_state.track_buffer[:-self.new_config.track_buffer_size]
        self.task_telemetry.record_track_buffer_sample(
            sample,
            fresh=False,
            cached=True,
        )

    def update_memory(self) -> None:
        pm = self.perception_manager
        prev_force = getattr(pm, "_force_heavy_reason", None)
        if self._ghost_scan_turns_left > 0:
            pm._force_heavy_reason = "ghost_arrival_scan"
        elif self._should_force_track_vlm():
            pm._force_heavy_reason = "etp_track_near_anchor"
        elif (
            self.nav_phase == NavPhase.GLOBAL_SEARCH
            and not _goal_has_active_anchor(self.topo_map, self.goal_manager, self.new_config)
            and self.topo_map.current_step <= self._etp_prefer_goal_anchor_until
        ):
            pm._force_heavy_reason = "etp_search_no_anchor"
        try:
            super().update_memory()
        finally:
            pm._force_heavy_reason = prev_force
        self._update_etp_stop_memory()
        self._update_explore_stuck_state()

    def _plan_track_phase(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
        self._update_track_buffer()
        if self._track_ready_for_approach():
            self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "track_target_confirmed")
            servo_decision = self._plan_servo_phase(nav_target, candidate_ids, scores)
            if servo_decision is not None:
                return servo_decision

        phase_age = self.topo_map.current_step - self.phase_enter_step
        if phase_age > self.new_config.track_timeout_steps:
            failed_anchor_id = (
                nav_target.target_node_id
                or self.servo_state.active_anchor_id
            )
            if failed_anchor_id:
                self.recovery_manager.fail_anchor(
                    self.topo_map,
                    failed_anchor_id,
                    "etp_track_timeout",
                )
            self.local_servo.reset()
            self._begin_goal_anchor_reroute("etp_track_timeout")
            self._transition(NavPhase.GLOBAL_SEARCH, "etp_track_timeout")
            return self._decision(
                "turn_right",
                "recovery",
                None,
                None,
                "track_timeout",
                "etp_track_timeout",
                candidate_ids,
                scores,
                NavPhase.GLOBAL_SEARCH,
            )

        return self._track_scan_decision(nav_target, candidate_ids, scores)

    def plan(self) -> PlanDecision:
        self._sync_prefer_goal_anchor_window()
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

        if self._ghost_scan_turns_left > 0:
            self._last_structure = structure
            self._last_nav_target = nav_target
            self._last_goal_proposals = scored_proposals
            self._last_selected_goal_proposal = selected_proposal
            return self._plan_ghost_arrival_scan(candidate_ids, scores)

        if self._explore_pending_turn and self._in_explore_phase():
            self._explore_pending_turn = False
            self._last_structure = structure
            self._last_nav_target = nav_target
            self._last_goal_proposals = scored_proposals
            self._last_selected_goal_proposal = selected_proposal
            return self._decision(
                "turn_right",
                "explore_escape",
                None,
                None,
                "recovery",
                "explore_stuck_turn",
                candidate_ids,
                scores,
                NavPhase.GLOBAL_SEARCH,
            )

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
        elif nav_target.reason == "no_navigation_candidates":
            self._no_candidates_count += 1

        self._maybe_enter_servo_near_anchor(nav_target)

        if nav_target.reason == "at_anchor_waypoint_servo":
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
                self.recovery_manager.fail_anchor(self.topo_map, failed_anchor_id, "route_timeout")
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
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "etp_selected_object_anchor")
        elif nav_target.target_node_id and nav_target.target_type == "stop_waypoint":
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "etp_selected_stop_memory")
        elif nav_target.target_node_id and nav_target.target_type in (
            "room", "landmark", "room_summary", "object_summary",
            "goal_region", "context_object",
        ):
            self._transition(NavPhase.ROUTE_TO_STRUCTURE, "etp_selected_structure")
        else:
            self._transition(NavPhase.GLOBAL_SEARCH, "etp_search_ghost")

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

    def on_navigation_event(self, target_node_id: Optional[str], event: str) -> Dict[str, Any]:
        node = self.topo_map.get_node(target_node_id) if target_node_id else None
        if node is not None and node.node_type == NodeType.WAYPOINT_CANDIDATE:
            node.attributes["visit_attempts"] = int(node.attributes.get("visit_attempts", 0)) + 1
            if event == "target_reached":
                self.topo_map.promote_frontier_to_visited(node.node_id)
                self._begin_ghost_arrival_scan()
                self._transition(NavPhase.GLOBAL_SEARCH, "etp_candidate_promoted")
                return {
                    "target_node_id": target_node_id,
                    "event": event,
                    "action": "candidate_promoted",
                    "reason": "etp_candidate_reached",
                }
            if event in ("unreachable", "snap_failed", "not_navigable", "collision_blocked", "no_progress_multigoal"):
                node.attributes["blocked_count"] = int(node.attributes.get("blocked_count", 0)) + 1
                node.attributes["blacklisted_until"] = (
                    self.topo_map.current_step + self.new_config.candidate_block_ttl
                )
                self.topo_map.consume_node(node.node_id, event)
                self._transition(NavPhase.GLOBAL_SEARCH, "etp_candidate_consumed")
                return {
                    "target_node_id": target_node_id,
                    "event": event,
                    "action": "consumed_candidate",
                    "reason": event,
                }
        if (
            node is not None
            and node.node_type == NodeType.WAYPOINT_VISITED
            and node.attributes.get("goal_stop_score") is not None
            and target_node_id == self._etp_stop_memory_target_id
            and event == "target_reached"
        ):
            anchor_id = node.attributes.get("goal_stop_object_id")
            if anchor_id and self.topo_map.get_node(anchor_id) is not None:
                self.local_servo.enter(anchor_id, self.topo_map.current_step)
            self._transition(NavPhase.TRACK_TARGET, "etp_stop_waypoint_reached")
            return {
                "target_node_id": target_node_id,
                "event": event,
                "action": "anchor_reached_awaiting_confirm",
                "reason": "etp_stop_memory_requires_fresh_confirm",
                "goal_stop_score": float(node.attributes.get("goal_stop_score", 0.0)),
                "linked_object_anchor_id": anchor_id,
            }
        if (
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
            self._transition(NavPhase.GLOBAL_SEARCH, "etp_stop_waypoint_failed")
            return {
                "target_node_id": target_node_id,
                "event": event,
                "action": "blacklisted_stop_waypoint",
                "reason": event,
            }
        return super().on_navigation_event(target_node_id, event)

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

    def _valid_stop_anchor(self, topo_map: DynamicTopoMap, anchor_id: str) -> Optional[SemanticNode]:
        anchor = topo_map.get_node(anchor_id)
        if anchor is None:
            return None
        if anchor.node_type != NodeType.OBJECT:
            return None
        if anchor.attributes.get("semantic_role") != "object_anchor":
            return None
        if int(anchor.attributes.get("blacklisted_until", -1)) >= topo_map.current_step:
            return None
        if not _label_matches(anchor.label, self.goal_manager.target_labels):
            return None
        return anchor

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


ConfTopoGOATAgentETPNav = ConfTopoGOATETPNavAgent


__all__ = [
    "ETPGoatConfig",
    "ETPTaskTelemetry",
    "ETPNavMemoryWriter",
    "ETPGraphNavigationPlanner",
    "ConfTopoGOATETPNavAgent",
    "ConfTopoGOATAgentETPNav",
]

