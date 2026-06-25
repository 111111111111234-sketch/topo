"""Final ConfTopo-GOAT navigation agent.

This module is the stable experiment / paper-facing entry point for the
simplified navigation state machine:

    SEARCH -> GO_TO_ANCHOR -> TRACK -> APPROACH -> VERIFY_STOP -> STOP

The implementation intentionally reuses ``goat_agent_new`` for perception,
memory, proposal ranking, and low-level servo mechanics.  Keeping the final
entry point in a separate module gives experiments a clean import target while
avoiding a copy-pasted agent fork.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from conftopo.agents.goat_agent_new import (
    GoatAgent as _GoatAgentNew,
    NavPhase,
    NavTarget,
    NewGoatConfig,
    PlanDecision,
    StopDecision,
    _bbox_area,
    _label_matches,
    compute_task_score,
)
from conftopo.core.dynamic_topo_map import NodeType, SemanticNode


class GoatAgentFinal(_GoatAgentNew):
    """ConfTopo-GOAT agent with the final simplified navigation logic.

    State mapping:
    - SEARCH: ``GLOBAL_SEARCH`` / ``ROUTE_TO_STRUCTURE``
    - GO_TO_ANCHOR: ``ROUTE_TO_OBJECT_ANCHOR``
    - TRACK: ``TRACK_TARGET``
    - APPROACH: ``LOCAL_VISUAL_APPROACH``
    - VERIFY_STOP: ``STOP_VERIFY``
    - STOP: ``STOP``
    """

    def plan(self) -> PlanDecision:
        """Simplified final navigation state machine.

        SEARCH -> GO_TO_ANCHOR -> TRACK -> APPROACH -> VERIFY_STOP -> STOP
        """
        nav_target, candidate_ids, scores = self._select_navigation_target()

        if self.nav_phase == NavPhase.STOP:
            return self._decision("stop", "stop", None, None, "stop", "already_stopped", candidate_ids, scores)

        if self.nav_phase == NavPhase.RECOVERY:
            self._transition(NavPhase.GLOBAL_SEARCH, self.recovery_manager.recovery_reason or "recovery_complete")
            return self._navigate_to(nav_target, candidate_ids, scores)

        if self.nav_phase == NavPhase.STOP_VERIFY:
            return self._plan_servo_phase(nav_target, candidate_ids, scores)

        if self.nav_phase == NavPhase.LOCAL_VISUAL_APPROACH:
            return self._plan_servo_phase(nav_target, candidate_ids, scores)

        if self.nav_phase == NavPhase.TRACK_TARGET:
            track_age = self.topo_map.current_step - self.phase_enter_step
            if track_age > self.new_config.track_timeout_steps:
                timeout_decision = self._handle_track_timeout(nav_target, candidate_ids, scores)
                if timeout_decision is not None:
                    return timeout_decision
            return self._plan_track_phase(nav_target, candidate_ids, scores)

        phase_age = self.topo_map.current_step - self.phase_enter_step
        if phase_age > self.new_config.phase_timeout_steps:
            timeout_decision = self._handle_phase_timeout(nav_target, candidate_ids, scores)
            if timeout_decision is not None:
                return timeout_decision

        anchor = self._anchor_for_target(nav_target)
        if anchor is None:
            self._transition(self._search_phase_for(nav_target), self._search_reason_for(nav_target))
            return self._navigate_to(nav_target, candidate_ids, scores)

        if not self._near_anchor(anchor):
            self._transition(NavPhase.ROUTE_TO_OBJECT_ANCHOR, "go_to_anchor")
            return self._navigate_to_anchor(anchor, candidate_ids, scores)

        if self.servo_state.active_anchor_id != anchor.node_id:
            self.local_servo.enter(anchor.node_id, self.topo_map.current_step)
        self._transition(NavPhase.TRACK_TARGET, "track_near_anchor")
        return self._plan_track_phase(nav_target, candidate_ids, scores)

    def _select_navigation_target(self) -> Tuple[NavTarget, List[str], List[float]]:
        structure = self.structure_planner.select(self.topo_map, self.goal_manager, self._position)
        proposals = self.navigation_planner.collect_goal_proposals(
            self.topo_map,
            self.goal_manager,
            structure,
            self.nav_phase,
            self._position,
            self.memory_writer.cur_vp_id,
        )
        proposals.extend(self._collect_hypothesis_goal_proposals())
        scored = self.navigation_planner.score_goal_proposals(
            proposals, self.goal_manager, self.topo_map, self._position,
        )
        selected = self.navigation_planner.select_best_proposal(scored)
        if selected is None:
            nav_target = NavTarget(reason="no_navigation_candidates")
            candidate_ids: List[str] = []
            scores: List[float] = []
        else:
            nav_target = self.navigation_planner.proposal_to_nav_target(
                selected,
                self.topo_map,
                self._position,
                self.memory_writer.cur_vp_id,
            )
            ranked = scored[:10]
            candidate_ids = [p.candidate_node_id for p in ranked]
            scores = [float(p.score) for p in ranked]

        self._last_structure = structure
        self._last_nav_target = nav_target
        self._last_goal_proposals = scored
        self._last_selected_goal_proposal = selected
        if nav_target.reason == "visited_fallback":
            self._visited_fallback_count += 1
        elif nav_target.reason == "no_navigation_candidates":
            self._no_candidates_count += 1
        return nav_target, candidate_ids, scores

    def _anchor_for_target(self, nav_target: NavTarget) -> Optional[SemanticNode]:
        if nav_target.target_type in ("object_anchor", "object_approach") and nav_target.target_node_id:
            node = self.topo_map.get_node(nav_target.target_node_id)
            if node is not None and node.node_type == NodeType.OBJECT:
                return node
        return self.navigation_planner._best_object_anchor(
            self.topo_map, self.goal_manager, self._position,
        )

    def _near_anchor(self, anchor: SemanticNode) -> bool:
        if self._position is None:
            return False
        if self.navigation_planner.at_anchor_waypoint(
            anchor, self._position, self.memory_writer.cur_vp_id,
        ):
            return True
        anchor_pos = self.navigation_planner._anchor_position(anchor)
        return float(np.linalg.norm(anchor_pos - self._position)) <= self.new_config.anchor_servo_enter_distance

    def _navigate_to_anchor(
        self,
        anchor: SemanticNode,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
        route = self.navigation_planner._resolve_object_route(
            anchor, self._position, self.memory_writer.cur_vp_id,
        )
        if route is not None:
            return self._decision(
                "navigate",
                "navigate",
                route.target_position,
                route.target_node_id,
                route.target_type,
                route.reason,
                candidate_ids,
                scores,
                route.expected_phase_after_reach,
            )
        return self._decision(
            "navigate",
            "navigate",
            anchor.position.copy(),
            anchor.node_id,
            "object_anchor",
            "go_to_anchor",
            candidate_ids,
            scores,
            NavPhase.TRACK_TARGET,
        )

    def _handle_phase_timeout(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> Optional[PlanDecision]:
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
                    sn.attributes["blacklisted_until"] = (
                        self.topo_map.current_step + self.new_config.anchor_blacklist_steps
                    )
            self._transition(NavPhase.GLOBAL_SEARCH, "route_to_structure_timeout")
            return self._navigate_to(nav_target, candidate_ids, scores)
        if self.nav_phase == NavPhase.ROUTE_TO_OBJECT_ANCHOR:
            failed_anchor_id = (
                nav_target.target_node_id
                if nav_target.target_type in ("object_anchor", "object_approach")
                else self.servo_state.active_anchor_id
            )
            self.recovery_manager.fail_anchor(self.topo_map, failed_anchor_id, "route_timeout")
            self._transition(NavPhase.RECOVERY, "route_to_anchor_timeout")
            return self._navigate_to(nav_target, candidate_ids, scores)
        if self.nav_phase == NavPhase.STOP_VERIFY:
            self._transition(NavPhase.LOCAL_VISUAL_APPROACH, "stop_verify_timeout")
        return None

    def _handle_track_timeout(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
        anchor_id = self.servo_state.active_anchor_id
        if anchor_id:
            self.recovery_manager.fail_anchor(self.topo_map, anchor_id, "track_timeout")
        self._transition(NavPhase.GLOBAL_SEARCH, "track_timeout")
        return self._navigate_to(nav_target, candidate_ids, scores)

    @staticmethod
    def _search_phase_for(nav_target: NavTarget) -> NavPhase:
        if nav_target.target_type in (
            "room", "landmark", "room_summary", "object_summary",
            "goal_region", "context_object",
        ):
            return NavPhase.ROUTE_TO_STRUCTURE
        return NavPhase.GLOBAL_SEARCH

    @staticmethod
    def _search_reason_for(nav_target: NavTarget) -> str:
        if nav_target.reason == "no_navigation_candidates":
            return "search_no_candidates"
        if nav_target.target_type == "frontier":
            return "search_frontier"
        if nav_target.target_type in (
            "room", "landmark", "room_summary", "object_summary",
            "goal_region", "context_object",
        ):
            return "search_structure"
        return "search_candidate"

    def _navigate_to(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
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

    def _plan_servo_phase(
        self,
        nav_target: NavTarget,
        candidate_ids: List[str],
        scores: List[float],
    ) -> PlanDecision:
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

        if self.new_config.stop_mode == "simple":
            stop = self._simple_stop_verify(servo_evidence)
        else:
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
            return self._navigate_to(nav_target, candidate_ids, scores)
        if servo.next_phase != self.nav_phase:
            self._transition(servo.next_phase, servo.reason)
            if self.nav_phase == NavPhase.STOP_VERIFY and stop.should_stop:
                self._transition(NavPhase.STOP, "stop_verified")
                return self._decision("stop", "stop", None, None, "stop", stop.reason, candidate_ids, scores)
        if servo.action in ("turn_left", "turn_right", "move_forward"):
            return self._decision(servo.action, "approach_confirm", None, None, "servo", servo.reason, candidate_ids, scores)
        if servo.action == "hold":
            verify_action = "turn_left" if self.topo_map.current_step % 2 == 0 else "turn_right"
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

    def _simple_stop_verify(self, servo_evidence: dict) -> StopDecision:
        window = self.new_config.stop_buffer_size
        min_count = self.new_config.servo_entry_evidence
        close_min = self.new_config.stop_close_min
        visible_count = sum(self.servo_state.confirm_buffer[-window:])
        centered_count = sum(self.servo_state.centered_buffer[-window:])
        close_count = sum(self.servo_state.close_buffer[-window:])
        progress_ok = str(servo_evidence.get("relative_progress", "uncertain")).lower() != "farther"
        forward_ok = self.servo_state.forward_action_count >= self.new_config.min_forward_before_stop
        fresh_ok = bool(self._last_packet.fresh_vlm)
        task_score = self._current_stop_task_score()

        seen = visible_count >= min_count
        centered = centered_count >= min_count
        close = close_count >= close_min
        task_ok = task_score > 0.3
        should = bool(fresh_ok and seen and centered and close and task_ok and forward_ok and progress_ok)
        reason = "simple_visual_confirmed_stop" if should else self._simple_stop_reason(
            fresh_ok, seen, centered, close, task_ok, forward_ok, progress_ok,
        )
        return StopDecision(should, seen, forward_ok, centered and close and fresh_ok and task_ok, reason)

    def _current_stop_task_score(self) -> float:
        best = max(
            (o for o in self._last_packet.report.objects
             if _label_matches(o.label, self.goal_manager.target_labels)),
            key=lambda o: _bbox_area(o.bbox) + float(o.confidence),
            default=None,
        )
        if best is None:
            return 0.0
        return compute_task_score(
            target_object=self.goal_manager.target_object or "",
            attributes=getattr(self.goal_manager.current_goal, "attributes", []),
            relations=getattr(self.goal_manager.current_goal, "relations", []),
            observed_label=best.label,
            observed_attributes=dict(best.attributes or {}),
            observed_relations=list(getattr(best, "spatial_relation", []) or []),
            room_prior=self.goal_manager.room_prior,
            landmarks=self.goal_manager.landmark_prior,
        )

    @staticmethod
    def _simple_stop_reason(
        fresh_ok: bool,
        seen: bool,
        centered: bool,
        close: bool,
        task_ok: bool,
        forward_ok: bool,
        progress_ok: bool,
    ) -> str:
        if not fresh_ok:
            return "simple_stop_needs_fresh_vlm"
        if not seen:
            return "simple_stop_target_not_stable"
        if not centered:
            return "simple_stop_target_not_centered"
        if not close:
            return "simple_stop_target_not_close"
        if not task_ok:
            return "simple_stop_task_score_low"
        if not forward_ok:
            return "simple_stop_approach_incomplete"
        if not progress_ok:
            return "simple_stop_target_getting_farther"
        return "simple_stop_not_ready"


GoatAgent = GoatAgentFinal
ConfTopoGOATAgent = GoatAgentFinal
ConfTopoGOATAgentFinal = GoatAgentFinal

__all__ = [
    "GoatAgent",
    "GoatAgentFinal",
    "ConfTopoGOATAgent",
    "ConfTopoGOATAgentFinal",
    "NavPhase",
    "NewGoatConfig",
]
