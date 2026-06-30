"""ConfTopo-GOAT Agent: 面向 GOAT-Bench 多目标连续导航的完整 Agent.

ConfTopo-GOAT 是主模型版本:
- 输入: 单视角 RGB (egocentric, ~79° HFOV, 无深度) + object/language/image goal
- 感知: RGB → CLIP 语义标注 (room type / object / landmark)
- 记忆: DynamicTopoMap 随 agent 移动逐步积累，跨目标不清空
- 规划: RuleScorer (Phase 2) → GraphRetrievalPlanner (Phase 4)
- 控制: 输出 target_position → 外部 PointNav DD-PPO 执行运动

Usage:
    agent = ConfTopoGOATAgent(config)

    # Episode loop (多目标)
    for goal in episode.goals:
        agent.set_new_goal(goal)  # 记忆不清空
        while not done:
            action = agent.step(obs)  # observe → update_memory → plan → act
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, EdgeType, SemanticNode
from conftopo.core.instruction_graph import InstructionGraph, GoalNode
from conftopo.core.rule_scorer import compute_semantic_bias
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import Qwen3VLBackend
from conftopo.perception.vlm_perceiver import VLMPerceiver


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

# Memory-first navigation thresholds
ANCHOR_PROMOTE_MIN_SEEN = 2
ANCHOR_PROMOTE_MIN_CONFIRMS = 2
ANCHOR_RELIABLE_MIN_BBOX_AREA = 0.06
FAILED_MEMORY_BLOCK_TTL = 80
LOCAL_SPIN_RESELECT_STEPS = 4
POSITION_IDLE_EPS = 0.08
VISITED_PENALTY_WITHOUT_MEMORY = 25.0


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


class SimpleNavPhase(str, Enum):
    SEARCH = "SEARCH"
    ROUTE_TO_ANCHOR = "ROUTE_TO_ANCHOR"
    SCAN_TRACK = "SCAN_TRACK"
    VISUAL_APPROACH = "VISUAL_APPROACH"
    VERIFY_STOP = "VERIFY_STOP"
    RECOVER = "RECOVER"
    STOP = "STOP"


@dataclass
class StopDecision:
    can_stop: bool
    reason: str
    goal_visible: bool = False
    need_scan: bool = False
    need_approach: bool = False
    need_recover: bool = False
    bbox_center: Optional[float] = None
    bbox_area: float = 0.0
    range_bin: str = "unknown"
    visibility: str = "unknown"
    centered: bool = False
    close: bool = False
    fresh_vlm: bool = False

    def to_debug(self) -> Dict[str, Any]:
        return {
            "stop_allowed": self.can_stop,
            "stop_reason": self.reason,
            "stop_goal_visible": self.goal_visible,
            "stop_need_scan": self.need_scan,
            "stop_need_approach": self.need_approach,
            "stop_need_recover": self.need_recover,
            "stop_bbox_center": self.bbox_center,
            "stop_bbox_area": self.bbox_area,
            "stop_range_bin": self.range_bin,
            "stop_visibility": self.visibility,
            "stop_centered": self.centered,
            "stop_close": self.close,
            "stop_fresh_vlm": self.fresh_vlm,
        }


class ConfTopoGOATAgent(ConfTopoBaseAgent):
    """Complete ConfTopo agent for GOAT-Bench.

    自主完成: 感知 → frontier 生成 → 记忆更新 → 目标选择
    只有底层运动控制 (PointNav) 是外部组件。
    """

    def __init__(self, config: Optional[ConfTopoConfig] = None):
        super().__init__(config)
        self.perceiver = LightPerceiver(
            room_labels=self.config.perception.room_labels,
        )
        self.vlm_perceiver = None
        if self.config.perception.backend == "vlm":
            self.vlm_perceiver = VLMPerceiver(Qwen3VLBackend(
                api_base=self.config.perception.vlm_api_base,
                model=self.config.perception.vlm_model,
                timeout=self.config.perception.vlm_timeout,
            ))

        # Agent state
        # Topo memory stores episode-start-relative positions.
        self._position: Optional[np.ndarray] = None
        self._origin_position: Optional[np.ndarray] = None
        self._heading: float = 0.0
        self._prev_position: Optional[np.ndarray] = None
        self._cur_vp_id: Optional[str] = None

        # Observation cache (set in observe())
        self._cur_rgb_embed: Optional[np.ndarray] = None
        self._cur_perception: Optional[Dict] = None
        self._cur_rgb: Optional[np.ndarray] = None
        self._cur_vlm_report: Optional[Dict[str, Any]] = None
        self._last_vlm_report: Optional[Dict[str, Any]] = None
        self._last_vlm_step: int = -1
        self._last_vlm_position: Optional[np.ndarray] = None
        self._last_vlm_trigger_reason: str = ""
        self._last_vlm_mode: str = ""
        self._vlm_confirm_buffer: List[bool] = []
        self._vlm_centered_buffer: List[bool] = []
        self._vlm_close_buffer: List[bool] = []

        # Frontier generation parameters
        self._frontier_step_size: float = 2.5  # estimated distance for new frontiers
        self._frontier_merge_radius: float = 1.5
        self._min_move_for_frontier: float = 0.5  # minimum displacement to generate frontiers

        # Multi-goal tracking
        self._goal_idx: int = 0
        self._goals_completed: int = 0

        # Planning stability state
        self._sticky_target_id: Optional[str] = None
        self._sticky_last_distance: Optional[float] = None
        self._sticky_last_heading: Optional[float] = None
        self._sticky_no_progress_steps: int = 0
        self._sticky_release_reason: str = ""
        self._consumed_frontier_ids: Set[str] = set()
        self._blocked_target_until: Dict[str, int] = {}
        self._last_skipped_candidates: List[Dict[str, Any]] = []
        self._last_navigation_event: Dict[str, Any] = {}
        self._last_plan_debug: Dict[str, Any] = {}
        self._goal_local_step: int = 0
        self._seen_goal_buffer: List[bool] = []
        self._near_anchor_buffer: List[bool] = []
        self._stop_confirm_buffer: List[bool] = []
        self._active_anchor_id: Optional[str] = None
        self._scan_after_anchor_steps: int = 0
        self._scan_turns_remaining: int = 0
        self._last_reached_anchor_id: Optional[str] = None
        self._last_goal_seen_step: int = 0
        self._runtime_room_prior: List[str] = []
        self._runtime_landmark_prior: List[str] = []
        self._nav_phase: SimpleNavPhase = SimpleNavPhase.SEARCH
        self._phase_transition_reason: str = "init"
        self._last_stop_decision: StopDecision = StopDecision(False, "not_evaluated")
        self._last_proposal_type: Optional[str] = None
        self._last_proposal_source: Optional[str] = None
        self._last_proposal_score: Optional[float] = None
        self._approach_steps: int = 0
        self._approach_forward_count: int = 0
        self._approach_travel_distance: float = 0.0
        self._approach_last_position: Optional[np.ndarray] = None
        self._approach_lost_count: int = 0
        self._approach_no_progress_count: int = 0
        self._approach_last_bbox_area: float = 0.0
        self._approach_max_steps: int = 24
        self._approach_lost_max: int = 6
        self._min_forward_before_stop: int = 2
        self._min_approach_distance: float = 0.5
        self._recover_turn_steps: int = 0
        self._recover_turn_origin: Optional[np.ndarray] = None
        self._spin_anchor_position: Optional[np.ndarray] = None
        self._spin_idle_steps: int = 0

    def _failed_memory_block_ttl(self) -> int:
        return max(FAILED_MEMORY_BLOCK_TTL, int(self.config.planning.blocked_target_ttl))

    def reset(self):
        """Full reset for new episode (clear memory)."""
        super().reset()
        self._position = None
        self._origin_position = None
        self._heading = 0.0
        self._prev_position = None
        self._cur_vp_id = None
        self._cur_rgb_embed = None
        self._cur_perception = None
        self._cur_rgb = None
        self._cur_vlm_report = None
        self._last_vlm_report = None
        self._last_vlm_step = -1
        self._last_vlm_position = None
        self._last_vlm_trigger_reason = ""
        self._last_vlm_mode = ""
        self._vlm_confirm_buffer = []
        self._vlm_centered_buffer = []
        self._vlm_close_buffer = []
        self._goal_idx = 0
        self._goals_completed = 0
        self._sticky_target_id = None
        self._sticky_last_distance = None
        self._sticky_last_heading = None
        self._sticky_no_progress_steps = 0
        self._sticky_release_reason = ""
        self._consumed_frontier_ids = set()
        self._blocked_target_until = {}
        self._last_skipped_candidates = []
        self._last_navigation_event = {}
        self._last_plan_debug = {}
        self._goal_local_step = 0
        self._seen_goal_buffer = []
        self._near_anchor_buffer = []
        self._stop_confirm_buffer = []
        self._active_anchor_id = None
        self._scan_after_anchor_steps = 0
        self._scan_turns_remaining = 0
        self._last_reached_anchor_id = None
        self._last_goal_seen_step = 0
        self._runtime_room_prior = []
        self._runtime_landmark_prior = []
        self._nav_phase = SimpleNavPhase.SEARCH
        self._phase_transition_reason = "reset"
        self._last_stop_decision = StopDecision(False, "not_evaluated")
        self._last_proposal_type = None
        self._last_proposal_source = None
        self._last_proposal_score = None
        self._reset_approach_progress()
        self._recover_turn_steps = 0
        self._recover_turn_origin = None
        self._spin_anchor_position = None
        self._spin_idle_steps = 0

    def _reset_approach_progress(self) -> None:
        self._approach_steps = 0
        self._approach_forward_count = 0
        self._approach_travel_distance = 0.0
        self._approach_last_position = None
        self._approach_lost_count = 0
        self._approach_no_progress_count = 0
        self._approach_last_bbox_area = 0.0

    def _should_resume_approach_progress(self, prev_phase: SimpleNavPhase, reason: str) -> bool:
        if prev_phase == SimpleNavPhase.VERIFY_STOP:
            return True
        if prev_phase == SimpleNavPhase.SCAN_TRACK and reason != "scan_complete_approach":
            return True
        return False

    def set_new_goal(self, goal: GoalNode):
        """Switch to a new goal within the same episode (memory preserved).

        This is the key for multi-goal: DynamicTopoMap is NOT cleared.
        """
        self.reset_keep_memory()
        self._clear_sticky("new_goal")
        self._goal_local_step = 0
        self._seen_goal_buffer = []
        self._near_anchor_buffer = []
        self._stop_confirm_buffer = []
        self._vlm_confirm_buffer = []
        self._vlm_centered_buffer = []
        self._vlm_close_buffer = []
        self._cur_vlm_report = None
        self._active_anchor_id = None
        self._scan_after_anchor_steps = 0
        self._scan_turns_remaining = 0
        self._last_reached_anchor_id = None
        self._last_goal_seen_step = self.topo_map.current_step
        self._last_vlm_trigger_reason = ""
        self._last_vlm_mode = ""
        self._nav_phase = SimpleNavPhase.SEARCH
        self._phase_transition_reason = "new_goal"
        self._last_stop_decision = StopDecision(False, "not_evaluated")
        self._last_proposal_type = None
        self._last_proposal_source = None
        self._last_proposal_score = None
        self._reset_approach_progress()
        self._recover_turn_steps = 0
        self._recover_turn_origin = None
        self._spin_anchor_position = None
        self._spin_idle_steps = 0
        goal_name = str(goal.target_object).lower().strip()
        self._runtime_room_prior = OBJECT_CATEGORY_ROOM_PRIORS.get(goal_name, [])
        self._runtime_landmark_prior = OBJECT_CATEGORY_LANDMARK_PRIORS.get(goal_name, [])

        if self.instruction_graph is None:
            self.instruction_graph = InstructionGraph(
                goal_type="object_goal",
                goal_nodes=[goal],
            )
        else:
            self.instruction_graph.set_current_goal(goal)

        # Update perceiver with new goal labels
        goal_labels = [goal.target_object]
        goal_labels.extend(goal.attributes)
        self.perceiver.set_goal_labels(
            labels=[goal.target_object],
            embeddings=goal.target_embedding[np.newaxis, :] if goal.target_embedding is not None else None,
        )
        if goal.landmarks:
            self.perceiver.set_landmark_labels(
                labels=goal.landmarks,
                embeddings=goal.landmark_embeddings,
            )

        self._goal_idx += 1
        self._sanitize_stale_goal_anchors()

    def _current_goal_tokens(self) -> set[str]:
        if self.instruction_graph is None:
            return set()
        goal = self.instruction_graph.get_current_goal()
        target = str(getattr(goal, "target_object", "") or "").lower().strip()
        tokens = {target} if target else set()
        for attr in getattr(goal, "attributes", []) or []:
            text = str(attr).lower().strip()
            if text:
                tokens.add(text)
        return tokens

    def _label_matches_current_goal(self, label: str, attrs: Optional[Dict[str, Any]] = None) -> bool:
        if self.instruction_graph is None:
            return True
        tokens = self._current_goal_tokens()
        if not tokens:
            return True

        candidates: list[str] = []
        for value in (label, (attrs or {}).get("canonical_label"), (attrs or {}).get("raw_label")):
            text = str(value or "").lower().strip()
            if text:
                candidates.append(text)

        for cand in candidates:
            for token in tokens:
                if cand == token or token in cand or cand in token:
                    return True
        return False

    def _sanitize_stale_goal_anchors(self) -> None:
        block_ttl = self._failed_memory_block_ttl()
        for node in self.topo_map._nodes.values():
            if node.node_type == NodeType.OBJECT:
                if node.attributes.get("semantic_role") not in ("object_anchor", "clip_hypothesis", "vlm_hypothesis"):
                    continue
                if self._label_matches_current_goal(node.label, node.attributes):
                    continue
                self._block_target(node.node_id, "previous_goal_anchor", ttl=block_ttl)
                continue

            if node.node_type != NodeType.WAYPOINT_VISITED:
                continue
            linked = node.attributes.get("goal_stop_object_id")
            if not linked:
                continue
            anchor = self.topo_map.get_node(str(linked))
            if anchor is None or not self._label_matches_current_goal(anchor.label, anchor.attributes):
                node.attributes.pop("goal_stop_object_id", None)
                node.attributes.pop("goal_stop_object_label", None)

    def _begin_anchor_scan_session(self, anchor_id: str, phase_reason: str) -> bool:
        if (
            self._last_reached_anchor_id == anchor_id
            and self._scan_turns_remaining > 0
            and self._nav_phase == SimpleNavPhase.SCAN_TRACK
        ):
            self._active_anchor_id = anchor_id
            return False

        self._active_anchor_id = anchor_id
        self._last_reached_anchor_id = anchor_id
        self._scan_turns_remaining = 4
        self._scan_after_anchor_steps = 0
        self._enter_phase(SimpleNavPhase.SCAN_TRACK, phase_reason)
        return True

    def observe(self, obs: Dict[str, Any]) -> None:
        """Process current observation.

        Expected obs keys:
            - 'rgb': raw RGB image [H, W, 3] (for visualization, not used directly)
            - 'rgb_embed': CLIP visual embedding [D] (pre-computed by encoder)
            - 'position': agent position [3] (x, y, z)
            - 'heading': agent heading in radians
            - 'depth' (optional): depth image
        """
        raw_position = np.array(obs['position'], dtype=np.float32)
        if self._origin_position is None:
            self._origin_position = raw_position.copy()
        self._prev_position = self._position
        self._position = raw_position - self._origin_position
        # 2.5D indoor nav: keep topo / targets on the episode ground plane.
        self._position[1] = 0.0
        self._heading = obs.get('heading', 0.0)
        self._cur_rgb = obs.get('rgb')
        if self._spin_anchor_position is None:
            self._spin_anchor_position = self._position.copy()
            self._spin_idle_steps = 0
        else:
            moved = float(np.linalg.norm(self._position - self._spin_anchor_position))
            if moved < POSITION_IDLE_EPS:
                self._spin_idle_steps += 1
            else:
                self._spin_anchor_position = self._position.copy()
                self._spin_idle_steps = 0

        # Visual embedding (CLIP)
        if 'rgb_embed' in obs:
            self._cur_rgb_embed = np.array(obs['rgb_embed'], dtype=np.float32)
        elif 'pano_rgb_embed' in obs:
            self._cur_rgb_embed = np.array(obs['pano_rgb_embed'], dtype=np.float32)
        else:
            self._cur_rgb_embed = None

        if self._nav_phase in (
            SimpleNavPhase.VISUAL_APPROACH,
            SimpleNavPhase.SCAN_TRACK,
            SimpleNavPhase.VERIFY_STOP,
        ):
            if self._active_anchor_id and self._last_reached_anchor_id:
                self._accumulate_approach_travel()

        # Run CLIP perception
        if self._cur_rgb_embed is not None:
            self._cur_perception = self.perceiver.perceive(self._cur_rgb_embed)
        else:
            self._cur_perception = {}
        self._goal_local_step += 1
        if float(self._cur_perception.get("best_goal_sim", 0.0)) > 0.45:
            self._last_goal_seen_step = self.topo_map.current_step
        self._maybe_run_vlm()

    def _current_goal_text(self) -> str:
        if self.instruction_graph is None:
            return ""
        goal = self.instruction_graph.get_current_goal()
        if goal is None:
            return ""
        target = str(getattr(goal, "target_object", "") or "")
        desc = str(getattr(goal, "description", "") or "")
        attrs = " ".join(str(a) for a in getattr(goal, "attributes", []) or [])
        return " ".join(part for part in (desc, attrs, target) if part).strip() or target

    def _vlm_context(self, mode: str) -> str:
        room = str(self._cur_perception.get("room_label", "unknown")) if self._cur_perception else "unknown"
        goal_sim = float(self._cur_perception.get("best_goal_sim", 0.0)) if self._cur_perception else 0.0
        anchor = f"active_anchor={self._active_anchor_id}" if self._active_anchor_id else "no_active_anchor"
        return f"mode={mode}; phase={self._nav_phase.value}; clip_room={room}; clip_goal_sim={goal_sim:.3f}; {anchor}"

    def _enter_phase(self, phase: SimpleNavPhase, reason: str) -> None:
        if phase != self._nav_phase:
            prev_phase = self._nav_phase
            self._nav_phase = phase
            if phase == SimpleNavPhase.VISUAL_APPROACH:
                resume = self._should_resume_approach_progress(prev_phase, reason)
                if resume:
                    self._approach_steps = 0
                    self._approach_lost_count = 0
                    self._approach_no_progress_count = 0
                    self._approach_last_bbox_area = 0.0
                else:
                    self._reset_approach_progress()
                self._approach_last_position = (
                    self._position.copy() if self._position is not None else None
                )
            if phase in (SimpleNavPhase.SEARCH, SimpleNavPhase.RECOVER):
                self._clear_sticky(reason)
        self._phase_transition_reason = reason

    def _anchor_distance(self) -> Optional[float]:
        if not self._active_anchor_id or self._position is None:
            return None
        anchor = self.topo_map.get_node(self._active_anchor_id)
        if anchor is None:
            return None
        anchor_pos = anchor.attributes.get("anchor_waypoint_position")
        if anchor_pos is None:
            anchor_pos = anchor.position
        return float(np.linalg.norm(np.asarray(anchor_pos, dtype=np.float32) - self._position))

    def _approach_requirements_met(self) -> bool:
        return (
            self._approach_forward_count >= self._min_forward_before_stop
            and self._approach_travel_distance >= self._min_approach_distance
        )

    def _scan_should_enter_approach(self, decision: StopDecision) -> bool:
        if decision.goal_visible or decision.need_approach:
            return True
        report = self._cur_vlm_report or {}
        best = self._best_vlm_goal_object(report)
        return best is not None and self._bbox_area(best.get("bbox")) > 0.0

    def _accumulate_approach_travel(self) -> None:
        if self._nav_phase not in (
            SimpleNavPhase.VISUAL_APPROACH,
            SimpleNavPhase.SCAN_TRACK,
            SimpleNavPhase.VERIFY_STOP,
        ):
            return
        if not self._active_anchor_id or not self._last_reached_anchor_id or self._position is None:
            return
        if self._approach_last_position is None:
            self._approach_last_position = self._position.copy()
            return
        delta = float(np.linalg.norm(self._position - self._approach_last_position))
        if delta > 1e-4:
            self._approach_travel_distance += delta
        self._approach_last_position = self._position.copy()

    def _record_plan_proposal(
        self,
        *,
        proposal_type: Optional[str],
        proposal_source: Optional[str],
        proposal_score: Optional[float],
    ) -> None:
        self._last_proposal_type = proposal_type
        self._last_proposal_source = proposal_source
        self._last_proposal_score = proposal_score

    def _step_telemetry(self, plan_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        plan_output = plan_output or {}
        debug = dict(plan_output.get("sticky_debug", self._last_plan_debug) or {})
        debug.update(self._vlm_debug())
        debug.update(self._last_stop_decision.to_debug())
        anchor_dist = self._anchor_distance()
        debug.update({
            "nav_phase": self._nav_phase.value,
            "phase_reason": self._phase_transition_reason,
            "proposal_type": debug.get("proposal_type", self._last_proposal_type),
            "proposal_source": debug.get("proposal_source", self._last_proposal_source),
            "proposal_score": debug.get("proposal_score", self._last_proposal_score),
            "active_anchor_id": self._active_anchor_id,
            "anchor_distance": None if anchor_dist is None else round(anchor_dist, 4),
            "approach_steps": self._approach_steps,
            "approach_forward_count": self._approach_forward_count,
            "approach_travel_distance": round(self._approach_travel_distance, 4),
        })
        return debug

    def _base_action_debug(self, plan_output: Dict[str, Any]) -> Dict[str, Any]:
        return self._step_telemetry(plan_output)

    def _should_run_vlm(self) -> Tuple[bool, str]:
        if self.vlm_perceiver is None or self._cur_rgb is None:
            return False, "no_vlm"

        goal_sim = float(self._cur_perception.get("best_goal_sim", 0.0)) if self._cur_perception else 0.0
        if goal_sim > 0.35:
            return True, "clip_goal_candidate"

        if self._active_anchor_id and self._last_reached_anchor_id:
            if self._scan_turns_remaining > 0 or self._scan_after_anchor_steps < 4:
                return True, "anchor_scan_verify"

        if self._active_anchor_id and self._scan_after_anchor_steps >= 2:
            return True, "stop_verify"

        if self._nav_phase == SimpleNavPhase.SCAN_TRACK:
            return True, "scan_track_fresh_vlm"
        if self._nav_phase == SimpleNavPhase.VISUAL_APPROACH:
            if self.topo_map.current_step - self._last_vlm_step >= 2:
                return True, "visual_approach_refresh"
        if self._nav_phase == SimpleNavPhase.VERIFY_STOP:
            return True, "verify_stop_fresh_vlm"

        if self.topo_map.current_step - self._last_vlm_step > 15:
            return True, "periodic_explore"

        if not self._has_goal_anchor() and self._goal_local_step % 10 == 0:
            return True, "search_no_anchor"

        return False, "skip"

    def _maybe_run_vlm(self) -> None:
        should_run, reason = self._should_run_vlm()
        self._last_vlm_trigger_reason = reason
        self._cur_vlm_report = None
        if not should_run:
            if self._last_vlm_report is not None:
                self._cur_vlm_report = dict(self._last_vlm_report, fresh=False, trigger_reason=reason)
            return

        mode = "confirm" if self._nav_phase in (
            SimpleNavPhase.SCAN_TRACK,
            SimpleNavPhase.VISUAL_APPROACH,
            SimpleNavPhase.VERIFY_STOP,
        ) else "explore"
        self._last_vlm_mode = mode
        try:
            report = self.vlm_perceiver.perceive(
                np.asarray(self._cur_rgb),
                self._current_goal_text(),
                visual_embed=self._cur_rgb_embed,
                step_id=self.topo_map.current_step,
                context=self._vlm_context(mode),
                mode=mode,
            )
        except Exception as exc:
            self._cur_vlm_report = {
                "fresh": False,
                "step": self.topo_map.current_step,
                "mode": mode,
                "trigger_reason": f"vlm_error:{type(exc).__name__}",
                "error": str(exc),
                "goal_visible": False,
                "stop_candidate": False,
                "objects": [],
            }
            return

        self._cur_vlm_report = self._vlm_report_to_dict(report, fresh=True, mode=mode, reason=reason)
        self._last_vlm_report = self._cur_vlm_report
        self._last_vlm_step = self.topo_map.current_step
        self._last_vlm_position = self._position.copy() if self._position is not None else None
        if bool(self._cur_vlm_report.get("goal_visible", False)):
            self._last_goal_seen_step = self.topo_map.current_step

    def _vlm_report_to_dict(
        self,
        report: PerceptionReport,
        *,
        fresh: bool,
        mode: str,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "fresh": bool(fresh),
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

    def update_memory(self) -> None:
        """Update DynamicTopoMap with new information.

        1. Add current position as visited waypoint
        2. Run room/object classification → add semantic nodes
        3. Generate frontier-like nodes from unexplored directions
        4. Apply confidence decay and pruning
        """
        if self._position is None:
            return

        self.topo_map.step()

        # 1. Add/update visited waypoint
        cur_vp = self._add_visited_waypoint()

        self._consume_reached_frontiers(cur_vp)

        # 2. Add semantic nodes (room, landmark detection)
        self._add_semantic_nodes(cur_vp)

        # 3. Generate frontier-like nodes
        self._generate_frontiers(cur_vp)

        # 4. Memory maintenance
        self.topo_map.decay_all_confidences()
        self.topo_map.merge_nearby_nodes(NodeType.WAYPOINT_FRONTIER)
        if self.topo_map.num_nodes > self.config.memory.max_nodes:
            self.topo_map.prune_low_confidence()

    def _add_visited_waypoint(self) -> str:
        """Add current position as visited waypoint, connect to previous."""
        # Check if near existing visited node
        nearest = self.topo_map.find_nearest_node(
            self._position, node_type=NodeType.WAYPOINT_VISITED
        )
        if nearest and np.linalg.norm(nearest.position - self._position) < 0.5:
            # Already have a node here, update it
            nearest.visit_count += 1
            nearest.confidence = min(1.0, nearest.confidence + 0.05)
            nearest.step_id = self.topo_map.current_step
            if self._cur_rgb_embed is not None:
                nearest.embedding = self._cur_rgb_embed
            cur_vp = nearest.node_id
        else:
            # Check if stepping onto a frontier
            frontier = self.topo_map.find_nearest_node(
                self._position, node_type=NodeType.WAYPOINT_FRONTIER
            )
            if frontier and np.linalg.norm(frontier.position - self._position) < 1.5:
                self.topo_map.promote_frontier_to_visited(frontier.node_id)
                frontier.position = self._position.copy()
                if self._cur_rgb_embed is not None:
                    frontier.embedding = self._cur_rgb_embed
                cur_vp = frontier.node_id
            else:
                cur_vp = self.topo_map.add_node(
                    NodeType.WAYPOINT_VISITED,
                    position=self._position.copy(),
                    embedding=self._cur_rgb_embed,
                    confidence=0.9,
                )

        # Connect to previous waypoint
        if self._cur_vp_id is not None and self._cur_vp_id != cur_vp:
            self.topo_map.add_edge(self._cur_vp_id, cur_vp, EdgeType.NAVIGABLE)

        self._cur_vp_id = cur_vp
        return cur_vp

    def _clear_sticky(self, reason: str = "") -> None:
        self._sticky_target_id = None
        self._sticky_last_distance = None
        self._sticky_last_heading = None
        self._sticky_no_progress_steps = 0
        self._sticky_release_reason = reason

    def _active_blocked_targets(self) -> Dict[str, int]:
        expired = [nid for nid, until in self._blocked_target_until.items() if until <= self.topo_map.current_step]
        for nid in expired:
            self._blocked_target_until.pop(nid, None)
            node = self.topo_map.get_node(nid)
            if node is not None:
                node.attributes.pop("blocked_reason", None)
                node.attributes.pop("blocked_until_step", None)
        return dict(self._blocked_target_until)

    def _is_blocked_target(self, node_id: str) -> bool:
        return node_id in self._active_blocked_targets()

    def _block_target(self, node_id: str, reason: str, ttl: Optional[int] = None) -> None:
        node = self.topo_map.get_node(node_id)
        if node is None:
            return
        ttl = self.config.planning.blocked_target_ttl if ttl is None else ttl
        until = self.topo_map.current_step + max(1, int(ttl))
        self._blocked_target_until[node_id] = until
        node.attributes["blocked_reason"] = reason
        node.attributes["blocked_until_step"] = until
        node.confidence = min(node.confidence, 0.1)

    def _consume_or_block_target(self, node_id: str, reason: str) -> Dict[str, Any]:
        node = self.topo_map.get_node(node_id)
        event = {"target_node_id": node_id, "reason": reason, "action": "missing"}
        if node is None:
            self._last_navigation_event = event
            return event
        node.attributes["consume_reason"] = reason
        node.attributes["consume_step"] = self.topo_map.current_step
        linked_anchor_id = node.attributes.get("goal_stop_object_id")
        if linked_anchor_id and reason in (
            "unreachable",
            "snap_failed",
            "not_navigable",
            "no_progress",
            "collision_blocked",
        ):
            linked_anchor_id = str(linked_anchor_id)
            self._block_target(linked_anchor_id, reason)
            if self._active_anchor_id == linked_anchor_id:
                self._active_anchor_id = None
            event["linked_object_anchor_id"] = linked_anchor_id
            event["linked_anchor_action"] = "blocked_target"
        if reason == "target_reached" and node.attributes.get("goal_stop_object_id"):
            anchor_id = str(node.attributes["goal_stop_object_id"])
            anchor = self.topo_map.get_node(anchor_id)
            if anchor is None or self._is_blocked_target(anchor_id):
                node.attributes.pop("goal_stop_object_id", None)
                node.attributes.pop("goal_stop_object_label", None)
            elif not self._label_matches_current_goal(anchor.label, anchor.attributes):
                node.attributes.pop("goal_stop_object_id", None)
                node.attributes.pop("goal_stop_object_label", None)
            elif not self._is_confirmed_anchor(anchor):
                node.attributes.pop("goal_stop_object_id", None)
                node.attributes.pop("goal_stop_object_label", None)
            else:
                self._begin_anchor_scan_session(anchor_id, "anchor_waypoint_reached")
                event.update({
                    "action": "anchor_waypoint_reached",
                    "linked_object_anchor_id": anchor_id,
                    "scan_turns_remaining": self._scan_turns_remaining,
                })
                if self._sticky_target_id == node_id:
                    self._clear_sticky(reason)
                self._last_navigation_event = event
                return event
        if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
            self._consumed_frontier_ids.add(node.node_id)
            node.attributes["consumed"] = True
            node.confidence = min(node.confidence, 0.05)
            event["action"] = "consumed_frontier"
        else:
            self._block_target(node.node_id, reason)
            event["action"] = "blocked_target"
        if self._sticky_target_id == node_id:
            self._clear_sticky(reason)
        self._last_navigation_event = event
        return event

    def on_navigation_event(self, target_node_id: Optional[str], event: str) -> Dict[str, Any]:
        if target_node_id is None:
            out = {"target_node_id": None, "reason": event, "action": "ignored"}
            self._last_navigation_event = out
            return out
        return self._consume_or_block_target(target_node_id, event)

    def _consume_reached_frontiers(self, cur_vp: str) -> None:
        radius = self.config.planning.frontier_consume_radius
        for node in self.topo_map.find_nodes_within_radius(
            self._position, radius=radius, node_type=NodeType.WAYPOINT_FRONTIER
        ):
            self._consume_or_block_target(node.node_id, "frontier_reached")

    def _is_consumed_frontier(self, node) -> bool:
        return (
            node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE)
            and (node.node_id in self._consumed_frontier_ids or node.attributes.get("consumed", False))
        )

    def _bbox_center_coords(self, bbox: Any) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None, None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None, None
        if x2 <= x1 or y2 <= y1:
            return None, None
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def _bbox_is_reliable(self, bbox: Any) -> bool:
        area = self._bbox_area(bbox)
        if area < ANCHOR_RELIABLE_MIN_BBOX_AREA:
            return False
        cx, cy = self._bbox_center_coords(bbox)
        if cx is None or cy is None:
            return False
        return 0.25 <= cx <= 0.75 and 0.2 <= cy <= 0.8

    def _is_memory_hypothesis(self, node: SemanticNode) -> bool:
        return node.attributes.get("semantic_role") in ("clip_hypothesis", "vlm_hypothesis")

    def _is_confirmed_anchor(self, node: SemanticNode) -> bool:
        if node.attributes.get("semantic_role") != "object_anchor":
            return False
        if bool(node.attributes.get("anchor_confirmed")):
            return True
        seen = int(node.attributes.get("seen_count", 0))
        confirms = int(node.attributes.get("promote_confirm_count", 0))
        return seen >= ANCHOR_PROMOTE_MIN_SEEN and confirms >= ANCHOR_PROMOTE_MIN_CONFIRMS

    def _memory_confidence_score(self, node: SemanticNode) -> float:
        score = float(node.confidence)
        attrs = node.attributes
        score += 0.15 * int(attrs.get("seen_count", 0))
        score += 0.2 * int(attrs.get("promote_confirm_count", 0))
        if self._bbox_is_reliable(attrs.get("bbox")):
            score += 0.25
        if attrs.get("semantic_role") == "object_anchor":
            score += 0.35
        failure_count = int(attrs.get("failure_count", 0))
        score -= 0.35 * failure_count
        stale = self.topo_map.current_step - int(attrs.get("last_seen_step", node.step_id))
        score -= max(0.0, stale - 40) * 0.01
        if self._is_blocked_target(node.node_id):
            score -= 5.0
        return score

    def _memory_route_waypoint(self, node: SemanticNode) -> Optional[SemanticNode]:
        """Return the waypoint to route to for a memory object (anchor or hypothesis)."""
        if node.node_type != NodeType.OBJECT:
            return None
        role = node.attributes.get("semantic_role")
        if role not in ("object_anchor", "clip_hypothesis", "vlm_hypothesis"):
            return None
        anchor_wp_id = node.attributes.get("anchor_waypoint_id")
        if not anchor_wp_id:
            return None
        return self.topo_map.get_node(str(anchor_wp_id))

    def _anchor_route_waypoint(self, node: SemanticNode) -> Optional[SemanticNode]:
        """Return the waypoint to route to for a confirmed anchor or hypothesis."""
        if not self._is_confirmed_anchor(node) and not self._is_memory_hypothesis(node):
            return None
        return self._memory_route_waypoint(node)

    def _has_goal_anchor(self) -> bool:
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            if not self._is_confirmed_anchor(node):
                continue
            if not self._label_matches_current_goal(node.label, node.attributes):
                continue
            if self._is_blocked_target(node.node_id):
                continue
            if self._memory_confidence_score(node) >= self.config.perception.object_threshold:
                return True
        return False

    def _has_high_confidence_memory(self, memory_exploit_candidates: List[Tuple[SemanticNode, str, float]]) -> bool:
        return self._has_goal_anchor() or bool(memory_exploit_candidates)

    def _hypothesis_observation_waypoint(self, hypothesis_node: SemanticNode) -> Optional[SemanticNode]:
        wp_id = hypothesis_node.attributes.get("anchor_waypoint_id")
        if not wp_id:
            return None
        return self.topo_map.get_node(str(wp_id))

    def _hypothesis_has_observable_viewpoint(self, hypothesis_node: SemanticNode) -> bool:
        obs_wp = self._hypothesis_observation_waypoint(hypothesis_node)
        if obs_wp is None or self._is_blocked_target(obs_wp.node_id):
            return False
        wp_reason = self._candidate_skip_reason(obs_wp)
        if wp_reason in ("current", "too_close"):
            return False
        if self._position is None:
            return True
        dist = float(np.linalg.norm(obs_wp.position - self._position))
        return dist >= self.config.planning.target_too_close_radius

    def _maybe_release_idle_spin(self) -> None:
        if self._spin_idle_steps < LOCAL_SPIN_RESELECT_STEPS:
            return
        if self._sticky_target_id:
            self._block_target(self._sticky_target_id, "idle_spin", ttl=20)
        self._clear_sticky("idle_spin_reselect")
        self._spin_idle_steps = 0
        if self._position is not None:
            self._spin_anchor_position = self._position.copy()
        self._phase_transition_reason = "idle_spin_reselect"

    def _is_pure_revisit_candidate(self, node: SemanticNode, anchor_id: Optional[str]) -> bool:
        if anchor_id:
            return False
        return node.node_type == NodeType.WAYPOINT_VISITED

    def _filter_exploration_candidates(
        self,
        candidates: List[SemanticNode],
        candidate_ids: List[str],
        candidate_anchor_ids: List[Optional[str]],
        *,
        has_high_confidence: bool,
    ) -> Tuple[List[SemanticNode], List[str], List[Optional[str]]]:
        if has_high_confidence or not candidates:
            return candidates, candidate_ids, candidate_anchor_ids

        frontiers = [
            i for i, node in enumerate(candidates)
            if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE)
        ]
        if not frontiers:
            return candidates, candidate_ids, candidate_anchor_ids

        keep_indices: List[int] = []
        for i, node in enumerate(candidates):
            anchor_id = candidate_anchor_ids[i]
            if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
                keep_indices.append(i)
                continue
            if anchor_id:
                anchor = self.topo_map.get_node(anchor_id)
                if anchor is not None and self._is_memory_hypothesis(anchor):
                    if self._hypothesis_has_observable_viewpoint(anchor):
                        keep_indices.append(i)
                continue
            self._last_skipped_candidates.append({
                "node_id": node.node_id,
                "type": node.node_type.value,
                "reason": "visited_blocked_without_high_confidence",
            })

        return (
            [candidates[i] for i in keep_indices],
            [candidate_ids[i] for i in keep_indices],
            [candidate_anchor_ids[i] for i in keep_indices],
        )

    def _proposal_type_for_node(self, node: SemanticNode, anchor_id: Optional[str]) -> str:
        if anchor_id:
            anchor = self.topo_map.get_node(anchor_id)
            if anchor is not None and self._is_confirmed_anchor(anchor):
                return "memory_exploit"
            return "hypothesis_verify"
        if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
            return "frontier_explore"
        return str(node.attributes.get("semantic_role", node.node_type.value))

    def _frontier_explore_score(self, node: SemanticNode, base_score: float) -> float:
        if self._position is None:
            return base_score
        dist = float(np.linalg.norm(node.position - self._position))
        base_score += min(5.0, dist * 0.65)
        base_score += self._room_prior_boost(node, has_goal_anchor=False)
        if self._cur_vp_id and node.node_id == self._cur_vp_id:
            base_score -= 4.0
        nearby_visited = self.topo_map.find_nodes_within_radius(
            node.position, radius=1.5, node_type=NodeType.WAYPOINT_VISITED
        )
        base_score -= 1.2 * len(nearby_visited)
        if self._is_blocked_target(node.node_id):
            base_score -= 5.0
        return base_score

    def _room_prior_boost(self, node: SemanticNode, has_goal_anchor: bool) -> float:
        if not self._runtime_room_prior:
            return 0.0

        room_label = str(node.attributes.get("room_label", "")).lower().strip()
        if not room_label:
            nearby_rooms = self.topo_map.find_nodes_within_radius(
                node.position, radius=5.0, node_type=NodeType.ROOM
            )
            if nearby_rooms:
                room_label = str(nearby_rooms[0].label).lower().strip()

        if room_label and room_label in self._runtime_room_prior:
            return 3.0 if not has_goal_anchor else 2.0
        return 0.0

    def _landmark_prior_boost(self, node: SemanticNode, has_goal_anchor: bool) -> float:
        if has_goal_anchor or not self._runtime_landmark_prior:
            return 0.0
        nearby = self.topo_map.find_nodes_within_radius(
            node.position, radius=5.0, node_type=NodeType.LANDMARK
        )
        for landmark in nearby:
            if str(landmark.label).lower().strip() in self._runtime_landmark_prior:
                return 1.5
        return 0.0

    def _scan_target_position(self) -> Optional[np.ndarray]:
        if self._position is None:
            return None
        angle = self._heading + np.pi / 2.0
        return np.array([
            self._position[0] - np.sin(angle),
            self._position[1],
            self._position[2] - np.cos(angle),
        ], dtype=np.float32)

    def _stop_visible_confirmed_snapshot(self) -> bool:
        """Read-only VLM stop evidence check, including the current frame."""
        report = self._cur_vlm_report
        if not report:
            return False
        best = self._best_vlm_goal_object(report)
        if best is None:
            return False
        bbox = best.get("bbox")
        bbox_ok = self._bbox_area(bbox) >= 0.06
        range_bin = str(best.get("range_bin", "unknown")).lower()
        close_ok = range_bin in ("near", "very_near", "close")
        visibility = str(best.get("visibility", "unknown")).lower()
        visible_ok = visibility in ("visible", "clear", "mostly_visible")
        centered_ok = self._bbox_centered(bbox)
        current_ok = bool(report.get("goal_visible", False)) and bbox_ok and close_ok and visible_ok and centered_ok
        seen = (self._vlm_confirm_buffer + [current_ok])[-5:]
        return sum(seen) >= 2 and bool(report.get("stop_candidate", False))

    def _best_vlm_goal_object(self, report: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        report = self._cur_vlm_report if report is None else report
        if not report:
            return None
        objects = [dict(o) for o in report.get("objects", []) if str(o.get("label", "")).strip()]
        if not objects:
            return None
        exact = [o for o in objects if self._label_matches_current_goal(str(o.get("label", "")), o)]
        candidates = exact if exact else objects
        if not exact and not bool(report.get("goal_visible", False)):
            return None
        return max(candidates, key=lambda o: float(o.get("confidence", 0.0) or 0.0))

    def _bbox_area(self, bbox: Any) -> float:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return 0.0
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return 0.0
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return max(0.0, min(1.0, (x2 - x1) * (y2 - y1)))

    def _bbox_centered(self, bbox: Any) -> bool:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            x1, _y1, x2, _y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return False
        cx = 0.5 * (x1 + x2)
        return 0.35 <= cx <= 0.65

    def _bbox_center_x(self, bbox: Any) -> Optional[float]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, _y1, x2, _y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        return 0.5 * (x1 + x2)

    def _vlm_debug(self) -> Dict[str, Any]:
        report = self._cur_vlm_report or {}
        best = self._best_vlm_goal_object(report) if report else None
        return {
            "vlm_fresh": bool(report.get("fresh", False)),
            "vlm_trigger_reason": self._last_vlm_trigger_reason,
            "vlm_mode": report.get("mode", self._last_vlm_mode),
            "vlm_goal_visible": bool(report.get("goal_visible", False)),
            "vlm_stop_candidate": bool(report.get("stop_candidate", False)),
            "vlm_best_bbox": None if best is None else best.get("bbox"),
            "vlm_range_bin": None if best is None else best.get("range_bin"),
            "vlm_visibility": None if best is None else best.get("visibility"),
        }

    def _sticky_plan_if_valid(self, candidates, candidate_ids, scores) -> Optional[Dict[str, Any]]:
        cfg = self.config.planning
        if self._spin_idle_steps >= LOCAL_SPIN_RESELECT_STEPS:
            return None
        if not cfg.sticky_target_enabled or self._sticky_target_id is None:
            return None
        if self._sticky_target_id not in candidate_ids:
            self._clear_sticky("target_not_candidate")
            return None
        node = self.topo_map.get_node(self._sticky_target_id)
        if node is None or self._is_consumed_frontier(node) or self._is_blocked_target(node.node_id):
            self._clear_sticky("target_unavailable")
            return None
        dist = float(np.linalg.norm(node.position - self._position))
        if dist <= cfg.sticky_reach_radius:
            self._consume_or_block_target(node.node_id, "target_reached")
            self._clear_sticky("target_reached")
            return None
        heading_changed = False
        if self._sticky_last_heading is not None:
            heading_changed = abs(_angle_delta(self._heading, self._sticky_last_heading)) > 0.05
        if self._sticky_last_distance is not None:
            progress = self._sticky_last_distance - dist
            if heading_changed:
                self._sticky_no_progress_steps = 0
            elif progress < cfg.sticky_min_progress:
                self._sticky_no_progress_steps += 1
            else:
                self._sticky_no_progress_steps = 0
        self._sticky_last_distance = dist
        self._sticky_last_heading = self._heading
        if self._sticky_no_progress_steps >= cfg.sticky_release_after_no_progress:
            self._consume_or_block_target(node.node_id, "no_progress")
            self._clear_sticky("no_progress")
            return None
        idx = candidate_ids.index(self._sticky_target_id)
        proposal_type = self._proposal_type_for_node(node, node.attributes.get("goal_stop_object_id"))
        self._last_plan_debug = {
            "sticky_target_id": self._sticky_target_id,
            "sticky_used": True,
            "sticky_distance": dist,
            "sticky_no_progress_steps": self._sticky_no_progress_steps,
            "sticky_release_reason": "",
            "consumed_frontiers": sorted(self._consumed_frontier_ids),
            "blocked_targets": self._active_blocked_targets(),
            "skipped_candidates": self._last_skipped_candidates,
            "navigation_event": self._last_navigation_event,
            "active_anchor_id": self._active_anchor_id,
            "scan_turns_remaining": self._scan_turns_remaining,
            "proposal_type": proposal_type,
            "proposal_source": "sticky",
            "proposal_score": scores[idx],
        }
        self._record_plan_proposal(
            proposal_type=self._last_plan_debug["proposal_type"],
            proposal_source="sticky",
            proposal_score=scores[idx],
        )
        sticky_debug = self._step_telemetry({"sticky_debug": self._last_plan_debug})
        self._last_plan_debug = sticky_debug
        return {
            "target_node_id": node.node_id,
            "target_position": node.position.copy(),
            "is_exploration": node.node_type == NodeType.WAYPOINT_FRONTIER,
            "target_type": proposal_type,
            "linked_object_anchor_id": node.attributes.get("goal_stop_object_id"),
            "proposal_type": self._last_proposal_type,
            "proposal_source": self._last_proposal_source,
            "proposal_score": self._last_proposal_score,
            "scores": scores,
            "candidate_ids": candidate_ids,
            "sticky_debug": sticky_debug,
        }

    def _add_semantic_nodes(self, cur_vp: str) -> None:
        """Add room/landmark/object nodes from perception results."""
        if not self._cur_perception:
            return

        pcfg = self.config.perception

        # Room node
        room_label = self._cur_perception.get("room_label", "unknown")
        room_conf = float(self._cur_perception.get("room_confidence", 0.0))
        if room_label != "unknown" and room_conf >= pcfg.room_threshold:
            cur_node = self.topo_map.get_node(cur_vp)
            if cur_node is not None:
                cur_node.attributes["room_label"] = room_label
                cur_node.attributes["room_confidence"] = room_conf
            existing_rooms = self.topo_map.find_nodes_within_radius(
                self._position, radius=5.0, node_type=NodeType.ROOM
            )
            matched_room = next((r for r in existing_rooms if r.label == room_label), None)
            if matched_room:
                matched_room.confidence = max(matched_room.confidence, room_conf)
                matched_room.step_id = self.topo_map.current_step
                if self._cur_rgb_embed is not None:
                    matched_room.embedding = self._cur_rgb_embed
            else:
                room_id = self.topo_map.add_node(
                    NodeType.ROOM,
                    position=self._position.copy(),
                    embedding=self._cur_rgb_embed,
                    confidence=room_conf,
                    label=room_label,
                )
                self.topo_map.add_edge(cur_vp, room_id, EdgeType.BELONGS_TO)

        # CLIP only writes weak hypotheses. VLM is required for formal object anchors.
        for label, sim in self._cur_perception.get("goal_scores", []):
            sim = float(sim)
            if sim < pcfg.object_threshold:
                continue
            existing = self.topo_map.find_nodes_within_radius(
                self._position, radius=2.0, node_type=NodeType.OBJECT
            )
            matched = next((
                o for o in existing
                if o.label == label and o.attributes.get("semantic_role") == "clip_hypothesis"
            ), None)
            if matched:
                matched.confidence = max(matched.confidence, sim)
                matched.step_id = self.topo_map.current_step
                matched.attributes["semantic_role"] = "clip_hypothesis"
                matched.attributes["needs_vlm_verify"] = True
                matched.attributes["source"] = "clip"
                matched.attributes["anchor_waypoint_id"] = cur_vp
                matched.attributes["anchor_waypoint_position"] = self._position.copy().tolist()
                matched.attributes["position_source"] = "observed_from"
                matched.attributes["last_seen_step"] = self.topo_map.current_step
                matched.attributes["seen_count"] = int(matched.attributes.get("seen_count", 1)) + 1
                if self._cur_rgb_embed is not None:
                    matched.embedding = self._cur_rgb_embed
            else:
                obj_id = self.topo_map.add_node(
                    NodeType.OBJECT,
                    position=self._position.copy(),
                    embedding=self._cur_rgb_embed,
                    confidence=sim,
                    label=label,
                    attributes={
                        "semantic_role": "clip_hypothesis",
                        "needs_vlm_verify": True,
                        "source": "clip",
                        "anchor_waypoint_id": cur_vp,
                        "anchor_waypoint_position": self._position.copy().tolist(),
                        "position_source": "observed_from",
                        "first_seen_step": self.topo_map.current_step,
                        "last_seen_step": self.topo_map.current_step,
                        "seen_count": 1,
                    },
                )
                self.topo_map.add_edge(cur_vp, obj_id, EdgeType.OBSERVED_AT)
            self._last_goal_seen_step = self.topo_map.current_step

        self._write_vlm_hypothesis(cur_vp)
        self._maybe_promote_vlm_hypotheses(cur_vp)

        # Landmark node
        for label, sim in self._cur_perception.get("landmark_scores", []):
            sim = float(sim)
            if sim < pcfg.landmark_threshold:
                continue
            existing = self.topo_map.find_nodes_within_radius(
                self._position, radius=3.0, node_type=NodeType.LANDMARK
            )
            matched = next((lm for lm in existing if lm.label == label), None)
            if matched:
                matched.confidence = max(matched.confidence, sim)
                matched.step_id = self.topo_map.current_step
                if self._cur_rgb_embed is not None:
                    matched.embedding = self._cur_rgb_embed
            else:
                lm_id = self.topo_map.add_node(
                    NodeType.LANDMARK,
                    position=self._position.copy(),
                    embedding=self._cur_rgb_embed,
                    confidence=sim,
                    label=label,
                )
                self.topo_map.add_edge(cur_vp, lm_id, EdgeType.VISIBLE_FROM)

    def _should_promote_hypothesis_to_anchor(self, node: SemanticNode) -> bool:
        if not self._is_memory_hypothesis(node):
            return False
        if int(node.attributes.get("seen_count", 0)) < ANCHOR_PROMOTE_MIN_SEEN:
            return False
        if float(node.confidence) < self.config.perception.object_threshold:
            return False
        if not self._bbox_is_reliable(node.attributes.get("bbox")):
            return False
        confirms = int(node.attributes.get("promote_confirm_count", 0))
        if confirms >= ANCHOR_PROMOTE_MIN_CONFIRMS:
            return True
        recent_confirms = sum(1 for ok in self._vlm_confirm_buffer[-4:] if ok)
        report = self._cur_vlm_report or {}
        return recent_confirms >= ANCHOR_PROMOTE_MIN_CONFIRMS and bool(report.get("stop_candidate"))

    def _promote_hypothesis_to_anchor(self, node: SemanticNode) -> None:
        node.attributes["semantic_role"] = "object_anchor"
        node.attributes["anchor_confirmed"] = True
        node.attributes["promoted_step"] = self.topo_map.current_step
        node.confidence = max(float(node.confidence), self.config.perception.object_threshold)

    def _maybe_promote_vlm_hypotheses(self, cur_vp: str) -> None:
        for node in self.topo_map.find_nodes_within_radius(
            self._position, radius=3.0, node_type=NodeType.OBJECT
        ):
            if not self._label_matches_current_goal(node.label, node.attributes):
                continue
            if not self._is_memory_hypothesis(node):
                continue
            if self._should_promote_hypothesis_to_anchor(node):
                self._promote_hypothesis_to_anchor(node)
                self.topo_map.add_edge(cur_vp, node.node_id, EdgeType.OBSERVED_AT)

    def _write_vlm_hypothesis(self, cur_vp: str) -> None:
        """Write weak VLM evidence into hypothesis pool; do not hijack navigation."""
        report = self._cur_vlm_report
        if not report or not report.get("fresh"):
            return
        best = self._best_vlm_goal_object(report)
        if best is None:
            return

        label = str(best.get("label", "")).strip()
        if not label:
            return
        conf = float(best.get("confidence", report.get("goal_match_confidence", 0.0)) or 0.0)
        bbox = best.get("bbox")
        range_bin = str(best.get("range_bin", "unknown"))
        visibility = str(best.get("visibility", "unknown"))
        reliable = self._bbox_is_reliable(bbox)
        visible_now = bool(report.get("goal_visible", False))
        if visible_now and reliable:
            self._vlm_confirm_buffer = (self._vlm_confirm_buffer + [True])[-5:]
        elif visible_now:
            self._vlm_confirm_buffer = (self._vlm_confirm_buffer + [False])[-5:]
        attributes = {
            "semantic_role": "vlm_hypothesis",
            "source": "vlm",
            "bbox": bbox,
            "bbox_area": self._bbox_area(bbox),
            "range_bin": range_bin,
            "visibility": visibility,
            "bearing": str(best.get("bearing", report.get("target_direction", "unknown"))),
            "vlm_confidence": conf,
            "object_attributes": dict(best.get("attributes") or {}),
            "anchor_waypoint_id": cur_vp,
            "anchor_waypoint_position": self._position.copy().tolist(),
            "position_source": "observed_from",
            "last_seen_step": self.topo_map.current_step,
            "needs_vlm_verify": True,
            "bbox_reliable": reliable,
        }

        existing = self.topo_map.find_nodes_within_radius(
            self._position, radius=2.5, node_type=NodeType.OBJECT
        )
        matched = next((o for o in existing if str(o.label).lower().strip() == label.lower()), None)
        if matched:
            matched.confidence = max(float(matched.confidence), conf)
            matched.step_id = self.topo_map.current_step
            matched.attributes.update(attributes)
            matched.attributes["seen_count"] = int(matched.attributes.get("seen_count", 0)) + 1
            if visible_now and reliable:
                matched.attributes["promote_confirm_count"] = int(
                    matched.attributes.get("promote_confirm_count", 0)
                ) + 1
            if self._cur_rgb_embed is not None:
                matched.embedding = self._cur_rgb_embed
            obj_id = matched.node_id
        else:
            attributes["first_seen_step"] = self.topo_map.current_step
            attributes["seen_count"] = 1
            attributes["promote_confirm_count"] = 1 if visible_now and reliable else 0
            obj_id = self.topo_map.add_node(
                NodeType.OBJECT,
                position=self._position.copy(),
                embedding=self._cur_rgb_embed,
                confidence=conf,
                label=label,
                attributes=attributes,
            )

        self.topo_map.add_edge(cur_vp, obj_id, EdgeType.OBSERVED_AT)
        if self._is_confirmed_anchor(self.topo_map.get_node(obj_id)):
            self.topo_map.add_edge(obj_id, cur_vp, EdgeType.ANCHORED_TO)
            self._last_goal_seen_step = self.topo_map.current_step

    def _write_vlm_object_anchor(self, cur_vp: str) -> None:
        """Backward-compatible alias for legacy callers."""
        self._write_vlm_hypothesis(cur_vp)
        self._maybe_promote_vlm_hypotheses(cur_vp)

    def _generate_frontiers(self, cur_vp: str) -> None:
        """Generate frontier-like nodes from unexplored directions.

        Strategies (纯 RGB, 无深度, 单视角):
        1. 位移式: agent 移动时，在未探索方向生成 frontier
        2. 视觉式: 高 goal similarity 方向优先生成 frontier (if pano available)
        """
        if self._prev_position is None:
            # First step: generate frontiers in cardinal directions
            self._generate_initial_frontiers(cur_vp)
            return

        displacement = np.linalg.norm(self._position - self._prev_position)
        if displacement < self._min_move_for_frontier:
            return  # Didn't move enough

        step_size = self._current_frontier_step_size()

        # Generate frontiers perpendicular to movement direction
        move_dir = self._position - self._prev_position
        move_dir_2d = np.array([move_dir[0], move_dir[2]])
        move_dist = np.linalg.norm(move_dir_2d)
        if move_dist < 1e-6:
            return

        move_dir_2d /= move_dist

        left = np.array([-move_dir_2d[1], move_dir_2d[0]])
        right = np.array([move_dir_2d[1], -move_dir_2d[0]])
        forward_left = move_dir_2d + left
        forward_right = move_dir_2d + right
        forward_left /= max(np.linalg.norm(forward_left), 1e-6)
        forward_right /= max(np.linalg.norm(forward_right), 1e-6)

        # Forward-biased exploration, widening when goal evidence gets stale.
        directions = [
            move_dir_2d,
            left,
            right,
            forward_left,
            forward_right,
        ]
        no_goal_steps = self.topo_map.current_step - self._last_goal_seen_step
        if no_goal_steps > 60:
            back_left = -move_dir_2d + left
            back_right = -move_dir_2d + right
            back_left /= max(np.linalg.norm(back_left), 1e-6)
            back_right /= max(np.linalg.norm(back_right), 1e-6)
            directions.extend([back_left, back_right])

        for d in directions:
            est_pos = np.array([
                self._position[0] + d[0] * step_size,
                self._position[1],  # keep y (height)
                self._position[2] + d[1] * step_size,
            ], dtype=np.float32)

            # Don't add if near existing visited/frontier node
            if self.topo_map.has_nearby_visited(est_pos, radius=self._frontier_merge_radius):
                continue
            existing_frontier = self.topo_map.find_nearest_node(
                est_pos, node_type=NodeType.WAYPOINT_FRONTIER
            )
            if existing_frontier and np.linalg.norm(existing_frontier.position - est_pos) < self._frontier_merge_radius:
                continue

            fid = self.topo_map.add_node(
                NodeType.WAYPOINT_FRONTIER,
                position=est_pos,
                confidence=0.3,
                attributes={
                    "semantic_role": "frontier",
                    "anchor_waypoint_id": cur_vp,
                    "frontier_step_size": step_size,
                },
            )
            self.topo_map.add_edge(cur_vp, fid, EdgeType.NAVIGABLE)

    def _current_frontier_step_size(self) -> float:
        no_goal_steps = self.topo_map.current_step - self._last_goal_seen_step
        if no_goal_steps > 60:
            return 5.5
        if no_goal_steps > 30:
            return 4.0
        return self._frontier_step_size

    def _generate_initial_frontiers(self, cur_vp: str) -> None:
        """Generate frontiers in 4 cardinal directions at episode start."""
        step_size = self._current_frontier_step_size()
        heading_rad = self._heading
        for angle_offset in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            angle = heading_rad + angle_offset
            est_pos = np.array([
                self._position[0] - step_size * np.sin(angle),
                self._position[1],
                self._position[2] - step_size * np.cos(angle),
            ], dtype=np.float32)

            fid = self.topo_map.add_node(
                NodeType.WAYPOINT_FRONTIER,
                position=est_pos,
                confidence=0.2,
                attributes={
                    "semantic_role": "frontier",
                    "anchor_waypoint_id": cur_vp,
                    "frontier_step_size": step_size,
                },
            )
            self.topo_map.add_edge(cur_vp, fid, EdgeType.NAVIGABLE)

    def _candidate_skip_reason(self, node: SemanticNode) -> Optional[str]:
        if node.node_id == self._cur_vp_id:
            return "current"
        if self._is_consumed_frontier(node):
            return "consumed"
        if self._is_blocked_target(node.node_id):
            return "blocked"
        if self._position is not None:
            dist = float(np.linalg.norm(node.position - self._position))
            if dist <= self.config.planning.target_too_close_radius:
                if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
                    self._consume_or_block_target(node.node_id, "too_close")
                return "too_close"
        return None

    def _debug_plan_state(self) -> Dict[str, Any]:
        debug = {
            "sticky_target_id": self._sticky_target_id,
            "sticky_used": False,
            "sticky_distance": self._sticky_last_distance,
            "sticky_no_progress_steps": self._sticky_no_progress_steps,
            "sticky_release_reason": self._sticky_release_reason,
            "consumed_frontiers": sorted(self._consumed_frontier_ids),
            "blocked_targets": self._active_blocked_targets(),
            "skipped_candidates": self._last_skipped_candidates,
            "navigation_event": self._last_navigation_event,
            "active_anchor_id": self._active_anchor_id,
            "scan_turns_remaining": self._scan_turns_remaining,
            "proposal_type": self._last_proposal_type,
            "proposal_source": self._last_proposal_source,
            "proposal_score": self._last_proposal_score,
        }
        return self._step_telemetry({"sticky_debug": debug})

    def plan(self) -> Dict[str, Any]:
        """Determine next target based on goal + memory.

        Returns dict with:
            - target_node_id: best node to navigate to
            - target_position: 3D position of target
            - is_exploration: whether this is frontier exploration
            - scores: all node scores
        """
        if self.instruction_graph is None or self._cur_vp_id is None:
            empty_debug = self._step_telemetry()
            return {
                "target_node_id": None,
                "target_position": None,
                "is_exploration": True,
                "sticky_debug": empty_debug,
            }

        self._last_navigation_event = {}
        self._maybe_release_idle_spin()
        primary_candidates = []
        primary_ids = []
        primary_anchor_ids = []
        fallback_candidates = []
        fallback_ids = []
        self._last_skipped_candidates = []
        has_goal_anchor = self._has_goal_anchor()
        recover_mode = self._nav_phase == SimpleNavPhase.RECOVER
        memory_exploit_candidates: List[Tuple[SemanticNode, str, float]] = []
        hypothesis_candidates: List[Tuple[SemanticNode, str, float]] = []

        for node in self.topo_map._nodes.values():
            if node.node_type == NodeType.OBJECT:
                if recover_mode:
                    continue
                if not self._label_matches_current_goal(node.label, node.attributes):
                    self._last_skipped_candidates.append({
                        "node_id": node.node_id,
                        "type": node.node_type.value,
                        "reason": "object_label_not_current_goal",
                    })
                    continue
                if self._is_blocked_target(node.node_id):
                    self._last_skipped_candidates.append({
                        "node_id": node.node_id,
                        "type": node.node_type.value,
                        "reason": "blocked",
                    })
                    continue

                is_confirmed = self._is_confirmed_anchor(node)
                is_hypothesis = self._is_memory_hypothesis(node)
                if not is_confirmed and not is_hypothesis:
                    continue

                memory_wp = self._memory_route_waypoint(node)
                if memory_wp is None:
                    self._last_skipped_candidates.append({
                        "node_id": node.node_id,
                        "type": node.node_type.value,
                        "reason": "object_without_anchor",
                    })
                    continue

                wp_reason = self._candidate_skip_reason(memory_wp)
                mem_score = self._memory_confidence_score(node)

                if is_hypothesis and wp_reason in ("current", "too_close"):
                    self._last_skipped_candidates.append({
                        "node_id": node.node_id,
                        "type": node.node_type.value,
                        "reason": "hypothesis_weak_at_current_location",
                    })
                    continue

                if is_confirmed and wp_reason in ("current", "too_close"):
                    memory_wp.attributes["goal_stop_object_id"] = node.node_id
                    memory_wp.attributes["goal_stop_object_label"] = node.label
                    if (
                        self._scan_turns_remaining <= 0
                        and self._scan_after_anchor_steps >= 4
                        and not self._stop_visible_confirmed_snapshot()
                    ):
                        self._block_target(node.node_id, "anchor_scan_no_confirm", ttl=self._failed_memory_block_ttl())
                        node.attributes["failure_count"] = int(node.attributes.get("failure_count", 0)) + 1
                        if self._active_anchor_id == node.node_id:
                            self._active_anchor_id = None
                        self._enter_phase(SimpleNavPhase.RECOVER, "anchor_scan_no_confirm")
                        self._last_skipped_candidates.append({
                            "node_id": node.node_id,
                            "type": node.node_type.value,
                            "reason": "anchor_scan_no_confirm",
                        })
                        continue
                    scan_started = self._begin_anchor_scan_session(node.node_id, "verify_local_confirmed_anchor")
                    proposal_score = 120.0 + mem_score
                    self._record_plan_proposal(
                        proposal_type="memory_exploit_current",
                        proposal_source="memory",
                        proposal_score=proposal_score,
                    )
                    self._last_plan_debug = self._debug_plan_state()
                    self._last_plan_debug.update({
                        "anchor_waypoint_reason": wp_reason,
                        "scan_session_started": scan_started,
                        "memory_confidence": mem_score,
                    })
                    return {
                        "target_node_id": memory_wp.node_id,
                        "target_position": memory_wp.position.copy(),
                        "is_exploration": False,
                        "target_type": "memory_exploit_current",
                        "linked_object_anchor_id": node.node_id,
                        "proposal_type": "memory_exploit_current",
                        "proposal_source": "memory",
                        "proposal_score": proposal_score,
                        "scores": [],
                        "candidate_ids": [memory_wp.node_id],
                        "sticky_debug": self._last_plan_debug,
                    }

                if wp_reason is not None:
                    self._last_skipped_candidates.append({
                        "node_id": memory_wp.node_id,
                        "type": memory_wp.node_type.value,
                        "reason": f"anchor_waypoint_{wp_reason}",
                    })
                    continue

                if is_confirmed and mem_score >= self.config.perception.object_threshold:
                    memory_exploit_candidates.append((memory_wp, node.node_id, mem_score))
                elif is_hypothesis:
                    obs_wp = self._hypothesis_observation_waypoint(node)
                    if obs_wp is None or not self._hypothesis_has_observable_viewpoint(node):
                        self._last_skipped_candidates.append({
                            "node_id": node.node_id,
                            "type": node.node_type.value,
                            "reason": "hypothesis_no_observable_viewpoint",
                        })
                        continue
                    hypothesis_candidates.append((obs_wp, node.node_id, mem_score))
                continue

            reason = self._candidate_skip_reason(node)
            if reason is not None:
                self._last_skipped_candidates.append({"node_id": node.node_id, "type": node.node_type.value, "reason": reason})
                continue
            if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
                primary_candidates.append(node)
                primary_ids.append(node.node_id)
                primary_anchor_ids.append(None)
            elif node.node_type == NodeType.WAYPOINT_VISITED and not recover_mode and has_goal_anchor:
                fallback_candidates.append(node)
                fallback_ids.append(node.node_id)

        for memory_wp, anchor_id, mem_score in memory_exploit_candidates:
            primary_candidates.append(memory_wp)
            primary_ids.append(memory_wp.node_id)
            primary_anchor_ids.append(anchor_id)
        for memory_wp, anchor_id, mem_score in hypothesis_candidates:
            primary_candidates.append(memory_wp)
            primary_ids.append(memory_wp.node_id)
            primary_anchor_ids.append(anchor_id)

        has_high_confidence = self._has_high_confidence_memory(memory_exploit_candidates)
        if not has_high_confidence:
            fallback_candidates = []
            fallback_ids = []

        if recover_mode and primary_candidates:
            fallback_candidates = []
            fallback_ids = []

        candidates = primary_candidates if primary_candidates else fallback_candidates
        candidate_ids = primary_ids if primary_candidates else fallback_ids
        candidate_anchor_ids = primary_anchor_ids if primary_candidates else [None] * len(fallback_candidates)
        candidates, candidate_ids, candidate_anchor_ids = self._filter_exploration_candidates(
            candidates,
            candidate_ids,
            candidate_anchor_ids,
            has_high_confidence=has_high_confidence,
        )

        if not candidates:
            self._last_plan_debug = self._debug_plan_state()
            return {"target_node_id": None, "target_position": None, "is_exploration": True, "candidate_ids": [], "scores": [], "sticky_debug": self._last_plan_debug}

        # Score candidates
        scores = compute_semantic_bias(
            goal_graph=self.instruction_graph,
            topo_map=self.topo_map,
            candidate_node_ids=candidate_ids,
            agent_position=self._position,
            normalize=False,  # use raw scores for selection
        )
        scores = list(scores)
        for i, node in enumerate(candidates):
            scores[i] += self._room_prior_boost(node, has_goal_anchor)
            scores[i] += self._landmark_prior_boost(node, has_goal_anchor)
            if candidate_anchor_ids[i]:
                anchor = self.topo_map.get_node(candidate_anchor_ids[i])
                if anchor is not None:
                    mem_score = self._memory_confidence_score(anchor)
                    if self._is_confirmed_anchor(anchor):
                        scores[i] += 50.0 + 8.0 * mem_score
                    else:
                        scores[i] += 18.0 + 4.0 * mem_score
            elif node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
                scores[i] = self._frontier_explore_score(node, scores[i])
                if recover_mode:
                    scores[i] += 6.0
            elif self._is_pure_revisit_candidate(node, candidate_anchor_ids[i]) and not has_high_confidence:
                scores[i] -= VISITED_PENALTY_WITHOUT_MEMORY
            elif recover_mode:
                scores[i] -= 3.0

        sticky_plan = self._sticky_plan_if_valid(candidates, candidate_ids, scores)
        if sticky_plan is not None:
            return sticky_plan

        best_idx = int(np.argmax(scores))
        best_node = candidates[best_idx]
        best_anchor_id = candidate_anchor_ids[best_idx]
        proposal_type = self._proposal_type_for_node(best_node, best_anchor_id)
        proposal_source = (
            "memory" if proposal_type == "memory_exploit"
            else "hypothesis" if proposal_type == "hypothesis_verify"
            else "frontier" if proposal_type == "frontier_explore"
            else "visited"
        )
        proposal_score = float(scores[best_idx])
        if best_anchor_id:
            best_anchor = self.topo_map.get_node(best_anchor_id)
            best_node.attributes["goal_stop_object_id"] = best_anchor_id
            best_node.attributes["goal_stop_object_label"] = best_anchor.label if best_anchor is not None else ""
            if best_anchor is not None and self._is_confirmed_anchor(best_anchor):
                self._active_anchor_id = best_anchor_id
            if self._nav_phase not in (
                SimpleNavPhase.SCAN_TRACK,
                SimpleNavPhase.VISUAL_APPROACH,
                SimpleNavPhase.VERIFY_STOP,
            ):
                phase_reason = (
                    "route_to_confirmed_anchor"
                    if best_anchor is not None and self._is_confirmed_anchor(best_anchor)
                    else "route_to_hypothesis"
                )
                self._enter_phase(SimpleNavPhase.ROUTE_TO_ANCHOR, phase_reason)
        elif recover_mode:
            self._enter_phase(SimpleNavPhase.SEARCH, "recovery_escape_frontier")
        if self.config.planning.sticky_target_enabled and proposal_type in (
            "frontier_explore",
            "memory_exploit",
            "memory_exploit_current",
            "hypothesis_verify",
        ):
            self._sticky_target_id = best_node.node_id
        else:
            self._sticky_target_id = None
        self._sticky_last_distance = float(np.linalg.norm(best_node.position - self._position))
        self._sticky_last_heading = self._heading
        self._sticky_no_progress_steps = 0
        self._record_plan_proposal(
            proposal_type=proposal_type,
            proposal_source=proposal_source,
            proposal_score=proposal_score,
        )
        self._last_plan_debug = self._debug_plan_state()
        self._sticky_release_reason = ""

        return {
            "target_node_id": best_node.node_id,
            "target_position": best_node.position.copy(),
            "is_exploration": best_node.node_type == NodeType.WAYPOINT_FRONTIER,
            "target_type": proposal_type,
            "linked_object_anchor_id": best_anchor_id,
            "proposal_type": proposal_type,
            "proposal_source": proposal_source,
            "proposal_score": proposal_score,
            "scores": scores,
            "candidate_ids": candidate_ids,
            "sticky_debug": self._last_plan_debug,
        }

    def _turn_scan_action(self, plan_output: Dict[str, Any], mode: str, reason: str) -> Dict[str, Any]:
        return {
            "action": "turn_left",
            "plan_action": "scan",
            "mode": mode,
            "reason": reason,
            "target_position": self._scan_target_position(),
            "target_node_id": plan_output.get("target_node_id"),
            "linked_object_anchor_id": self._active_anchor_id,
            "scan_turns_remaining": self._scan_turns_remaining,
            "candidate_ids": plan_output.get("candidate_ids", []),
            "scores": plan_output.get("scores", []),
            "is_exploration": True,
            "sticky_debug": self._base_action_debug(plan_output),
        }

    def _recover_from_active_anchor(self, reason: str) -> None:
        if self._active_anchor_id:
            anchor = self.topo_map.get_node(self._active_anchor_id)
            if anchor is not None:
                anchor.attributes["failure_count"] = int(anchor.attributes.get("failure_count", 0)) + 1
            self._block_target(self._active_anchor_id, reason, ttl=self._failed_memory_block_ttl())
        self._active_anchor_id = None
        self._last_reached_anchor_id = None
        self._scan_turns_remaining = 0
        self._scan_after_anchor_steps = 0
        self._reset_approach_progress()
        self._recover_turn_steps = 0
        self._recover_turn_origin = self._position.copy() if self._position is not None else None
        self._enter_phase(SimpleNavPhase.RECOVER, reason)

    def _visual_approach_action(self, decision: StopDecision, plan_output: Dict[str, Any]) -> Dict[str, Any]:
        self._approach_steps += 1
        if not decision.goal_visible:
            self._approach_lost_count += 1
        else:
            self._approach_lost_count = 0
        if decision.bbox_area <= self._approach_last_bbox_area + 0.005:
            self._approach_no_progress_count += 1
        else:
            self._approach_no_progress_count = 0
        self._approach_last_bbox_area = max(self._approach_last_bbox_area, decision.bbox_area)

        if (
            self._approach_steps > self._approach_max_steps
            or self._approach_lost_count > self._approach_lost_max
            or self._approach_no_progress_count > 8
        ):
            self._recover_from_active_anchor("visual_approach_timeout")
            return self._turn_scan_action(plan_output, "recover_scan", "visual_approach_timeout")

        if decision.need_scan:
            self._enter_phase(SimpleNavPhase.SCAN_TRACK, decision.reason)
            return self._turn_scan_action(plan_output, "approach_lost_scan", decision.reason)

        if decision.close and decision.centered and self._approach_requirements_met():
            self._enter_phase(SimpleNavPhase.VERIFY_STOP, "close_centered_verify")
            return self._turn_scan_action(plan_output, "stop_verify_scan", "close_centered_verify")

        if decision.bbox_center is not None:
            if decision.bbox_center < 0.42:
                action, reason = "turn_left", "approach_align_left"
            elif decision.bbox_center > 0.58:
                action, reason = "turn_right", "approach_align_right"
            else:
                action, reason = "move_forward", "approach_forward"
        else:
            bearing = str((self._best_vlm_goal_object() or {}).get("bearing", "")).lower()
            if "left" in bearing:
                action, reason = "turn_left", "approach_bearing_left"
            elif "right" in bearing:
                action, reason = "turn_right", "approach_bearing_right"
            else:
                action, reason = "move_forward", "approach_forward_unknown_center"

        if action == "move_forward":
            self._approach_forward_count += 1

        return {
            "action": action,
            "plan_action": "approach_confirm",
            "mode": "visual_approach",
            "reason": reason,
            "target_position": None,
            "target_node_id": plan_output.get("target_node_id"),
            "linked_object_anchor_id": self._active_anchor_id,
            "candidate_ids": plan_output.get("candidate_ids", []),
            "scores": plan_output.get("scores", []),
            "is_exploration": False,
            "sticky_debug": self._base_action_debug(plan_output),
        }

    def act(self, plan_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert plan output to action for PointNav controller."""
        target_pos = plan_output.get("target_position")
        self._last_stop_decision = self._evaluate_stop()
        decision = self._last_stop_decision

        if decision.can_stop:
            if self._nav_phase == SimpleNavPhase.VERIFY_STOP and self._approach_requirements_met():
                self._enter_phase(SimpleNavPhase.STOP, "verified_stop")
                return {
                    "action": "stop",
                    "mode": "verified_stop",
                    "target_node_id": plan_output.get("target_node_id"),
                    "linked_object_anchor_id": self._active_anchor_id,
                    "candidate_ids": plan_output.get("candidate_ids", []),
                    "scores": plan_output.get("scores", []),
                    "is_exploration": False,
                    "sticky_debug": self._base_action_debug(plan_output),
                }
            if self._nav_phase == SimpleNavPhase.VISUAL_APPROACH and self._approach_requirements_met():
                self._enter_phase(SimpleNavPhase.VERIFY_STOP, "approach_complete_verify")
                return self._turn_scan_action(plan_output, "stop_verify_scan", "approach_complete_verify")
            if self._nav_phase != SimpleNavPhase.VISUAL_APPROACH:
                self._enter_phase(SimpleNavPhase.VISUAL_APPROACH, "stop_requires_approach_first")
            return self._visual_approach_action(decision, plan_output)

        if self._nav_phase in (SimpleNavPhase.SCAN_TRACK, SimpleNavPhase.VERIFY_STOP) and decision.need_approach:
            self._enter_phase(SimpleNavPhase.VISUAL_APPROACH, decision.reason)

        if self._nav_phase == SimpleNavPhase.VISUAL_APPROACH:
            return self._visual_approach_action(decision, plan_output)

        if decision.need_recover and self._nav_phase in (
            SimpleNavPhase.SCAN_TRACK,
            SimpleNavPhase.VISUAL_APPROACH,
            SimpleNavPhase.VERIFY_STOP,
        ):
            self._recover_from_active_anchor(decision.reason)
            return self._turn_scan_action(plan_output, "recover_scan", decision.reason)

        if self._scan_turns_remaining > 0:
            self._scan_turns_remaining -= 1
            self._scan_after_anchor_steps += 1
            if self._scan_turns_remaining <= 0 and self._nav_phase == SimpleNavPhase.SCAN_TRACK:
                if self._scan_should_enter_approach(decision):
                    self._enter_phase(SimpleNavPhase.VISUAL_APPROACH, "scan_complete_approach")
                else:
                    self._recover_from_active_anchor("anchor_scan_no_confirm")
                    return self._turn_scan_action(plan_output, "recover_scan", "anchor_scan_no_confirm")
            return self._turn_scan_action(plan_output, "scan_after_anchor", "scan_turn")

        if decision.need_scan and self._nav_phase in (SimpleNavPhase.SCAN_TRACK, SimpleNavPhase.VERIFY_STOP):
            return self._turn_scan_action(plan_output, "stop_need_scan", decision.reason)

        if self._nav_phase == SimpleNavPhase.RECOVER:
            if target_pos is not None and plan_output.get("target_type") in (
                "frontier_explore",
                "frontier",
                "memory_exploit",
                "hypothesis_verify",
                "memory_exploit_current",
            ):
                self._recover_turn_steps = 0
                self._recover_turn_origin = None
                return {
                    "action": "navigate",
                    "target_position": target_pos,
                    "target_node_id": plan_output.get("target_node_id"),
                    "target_type": plan_output.get("target_type"),
                    "linked_object_anchor_id": plan_output.get("linked_object_anchor_id"),
                    "candidate_ids": plan_output.get("candidate_ids", []),
                    "scores": plan_output.get("scores", []),
                    "is_exploration": plan_output.get("is_exploration", False),
                    "sticky_debug": self._base_action_debug(plan_output),
                }
            self._recover_turn_steps += 1
            return self._turn_scan_action(plan_output, "recover_scan", self._phase_transition_reason)

        if target_pos is None:
            return self._turn_scan_action(plan_output, "no_candidate_scan", "no_navigation_proposal")

        return {
            "action": "navigate",
            "target_position": target_pos,
            "target_node_id": plan_output.get("target_node_id"),
            "target_type": plan_output.get("target_type"),
            "linked_object_anchor_id": plan_output.get("linked_object_anchor_id"),
            "candidate_ids": plan_output.get("candidate_ids", []),
            "scores": plan_output.get("scores", []),
            "is_exploration": plan_output.get("is_exploration", False),
            "sticky_debug": self._base_action_debug(plan_output),
        }

    def on_goal_reached(self) -> bool:
        """Called when current goal is reached.

        Returns True if there are more goals.
        """
        self._goals_completed += 1
        if self.instruction_graph:
            return self.instruction_graph.advance()
        return False

    def _evaluate_stop(self) -> StopDecision:
        report = self._cur_vlm_report
        if not report or not report.get("fresh"):
            return StopDecision(False, "no_fresh_vlm", need_scan=True, fresh_vlm=False)
        if self._goal_local_step < 8:
            return StopDecision(False, "warmup", fresh_vlm=True)
        if not self._active_anchor_id:
            return StopDecision(False, "no_anchor", need_recover=True, fresh_vlm=True)
        if self._last_reached_anchor_id and self._scan_after_anchor_steps < 2:
            return StopDecision(False, "scan_warmup", need_scan=True, fresh_vlm=True)

        anchor = self.topo_map.get_node(self._active_anchor_id)
        if anchor is None:
            return StopDecision(False, "anchor_missing", need_recover=True, fresh_vlm=True)

        anchor_pos = anchor.attributes.get("anchor_waypoint_position")
        if anchor_pos is None:
            anchor_pos = anchor.position
        anchor_dist = float(np.linalg.norm(np.asarray(anchor_pos, dtype=np.float32) - self._position))
        near_anchor = anchor_dist <= 1.2
        self._near_anchor_buffer.append(near_anchor)
        self._near_anchor_buffer = self._near_anchor_buffer[-5:]
        if not near_anchor:
            return StopDecision(False, "not_near_anchor", need_recover=True, fresh_vlm=True)

        best = self._best_vlm_goal_object(report)
        if best is None:
            self._vlm_confirm_buffer.append(False)
            self._vlm_confirm_buffer = self._vlm_confirm_buffer[-5:]
            return StopDecision(False, "not_visible", need_scan=True, fresh_vlm=True)

        bbox = best.get("bbox")
        bbox_area = self._bbox_area(bbox)
        bbox_center = self._bbox_center_x(bbox)
        bbox_ok = bbox_area >= 0.06
        range_bin = str(best.get("range_bin", "unknown")).lower()
        close_ok = range_bin in ("near", "very_near", "close")
        visibility = str(best.get("visibility", "unknown")).lower()
        visible_ok = visibility in ("visible", "clear", "mostly_visible")
        centered_ok = self._bbox_centered(bbox)
        goal_visible = bool(report.get("goal_visible", False))
        stop_candidate = bool(report.get("stop_candidate", False))

        current_ok = goal_visible and bbox_ok and close_ok and visible_ok and centered_ok
        self._vlm_confirm_buffer.append(current_ok)
        self._vlm_confirm_buffer = self._vlm_confirm_buffer[-5:]
        self._vlm_centered_buffer.append(centered_ok)
        self._vlm_centered_buffer = self._vlm_centered_buffer[-5:]
        self._vlm_close_buffer.append(close_ok)
        self._vlm_close_buffer = self._vlm_close_buffer[-5:]

        visual_confirm = sum(self._vlm_confirm_buffer) >= 2 and stop_candidate
        approach_ok = self._approach_requirements_met()
        in_verify_phase = self._nav_phase == SimpleNavPhase.VERIFY_STOP
        can_stop = visual_confirm and approach_ok and in_verify_phase

        if can_stop:
            return StopDecision(
                True, "verified_stop", goal_visible=True, bbox_center=bbox_center,
                bbox_area=bbox_area, range_bin=range_bin, visibility=visibility,
                centered=centered_ok, close=close_ok, fresh_vlm=True,
            )
        if visual_confirm and not approach_ok:
            return StopDecision(
                False, "approach_not_enough", goal_visible=goal_visible, need_approach=True,
                bbox_center=bbox_center, bbox_area=bbox_area, range_bin=range_bin,
                visibility=visibility, centered=centered_ok, close=close_ok, fresh_vlm=True,
            )
        if visual_confirm and not in_verify_phase:
            return StopDecision(
                False, "need_visual_approach", goal_visible=goal_visible, need_approach=True,
                bbox_center=bbox_center, bbox_area=bbox_area, range_bin=range_bin,
                visibility=visibility, centered=centered_ok, close=close_ok, fresh_vlm=True,
            )

        if not goal_visible or not visible_ok:
            reason, need_scan, need_approach = "not_visible", True, False
        elif not bbox_ok:
            reason, need_scan, need_approach = "bbox_too_small", False, True
        elif not centered_ok:
            reason, need_scan, need_approach = "not_centered", False, True
        elif not close_ok:
            reason, need_scan, need_approach = "too_far", False, True
        elif not stop_candidate:
            reason, need_scan, need_approach = "no_stop_candidate", False, True
        else:
            reason, need_scan, need_approach = "need_more_confirm", False, True

        return StopDecision(
            False, reason, goal_visible=goal_visible, need_scan=need_scan,
            need_approach=need_approach, bbox_center=bbox_center, bbox_area=bbox_area,
            range_bin=range_bin, visibility=visibility, centered=centered_ok,
            close=close_ok, fresh_vlm=True,
        )

    def should_stop(self) -> bool:
        """Backward-compatible stop wrapper."""
        self._last_stop_decision = self._evaluate_stop()
        return self._last_stop_decision.can_stop

    # ==================== Statistics ====================

    @property
    def memory_stats(self) -> Dict[str, int]:
        """Return memory statistics for analysis."""
        return {
            "total_nodes": self.topo_map.num_nodes,
            "visited_waypoints": len(self.topo_map.get_visited()),
            "frontiers": len(self.topo_map.get_frontiers()),
            "candidate_waypoints": len(self.topo_map.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE)),
            "objects": len(self.topo_map.get_nodes_by_type(NodeType.OBJECT)),
            "rooms": len(self.topo_map.get_nodes_by_type(NodeType.ROOM)),
            "landmarks": len(self.topo_map.get_nodes_by_type(NodeType.LANDMARK)),
            "step": self._step_count,
            "goals_completed": self._goals_completed,
            "consumed_frontiers": len(self._consumed_frontier_ids),
            "blocked_targets": len(self._active_blocked_targets()),
        }



///////////////////////////////////////////////////////
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
        node = self._find_matching_node(label, pose.position)
        if node is None:
            node_id = self._create_object_node(label, confidence, obj, pose, waypoint_id)
        else:
            node_id = node.node_id
            self._update_object_node(node, confidence, obj, pose, waypoint_id)

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
    ) -> str:
        attrs = self._node_attrs(confidence, obj, pose, waypoint_id, seen_count=1)
        node_id = self.topo_map.add_node(
            NodeType.OBJECT,
            position=pose.position.copy(),
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
    ) -> None:
        seen_count = int(node.attributes.get("seen_count", 1)) + 1
        repeat_bonus = min(0.15, 0.02 * seen_count)
        node.confidence = min(1.0, max(float(node.confidence), confidence) + repeat_bonus)
        node.step_id = self.topo_map.current_step
        node.attributes.update(
            self._node_attrs(confidence, obj, pose, waypoint_id, seen_count=seen_count)
        )

    def _node_attrs(
        self,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
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
        node.attributes["seen_count"] = entry.seen_count
        node.attributes["last_seen_step"] = entry.last_seen_step
        node.attributes["failed_count"] = entry.failed_count
        node.attributes["unreachable_count"] = entry.unreachable_count
        node.attributes["memory_state"] = "confirmed"
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
            target_position = np.asarray(entry.anchor_waypoint_position, dtype=np.float32)
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
        bbox = _target_bbox(vlm_report, goal)
        if bbox is None:
            return {"action": "turn_left", "reason": "verify_scan_no_target"}
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
        state.verify_attempts += 1
        if not vlm_report or not bool(vlm_report.get("fresh", False)):
            return False
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
        return bbox_area >= 0.12 and range_bin in {"near", "very_near", "close"}

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
            and min(state.verify_area_window[-2:]) >= 0.06
            and state.verify_area_window[-1] >= 0.9 * state.verify_area_window[-2]
        )
        strong_visual_stop = (
            bbox_area >= 0.03
            and range_bin in {"near", "very_near", "close"}
            and bbox_confidence != "low"
            and visibility not in {"occluded", "not_visible"}
            and centered
            and (area_growing or area_holding_close)
        )
        vlm_stop_candidate = (
            bool(vlm_report.get("stop_candidate", False))
            and bbox_area >= 0.03
            and range_bin in {"near", "very_near", "close"}
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
            bbox_area < 0.03
            or center is None
            or not 0.15 <= center <= 0.90
            or range_bin not in {"near", "very_near", "close"}
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
        self.memory.mark_navigation_target_failed(
            target_id,
            target_type=target_type,
            unreachable=True,
            reason="stuck_no_progress",
        )
        self.state.stuck_steps = 0

        action.clear()
        action.update(
            {
                "action": "turn_left",
                "reason": "stuck_recovery",
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
