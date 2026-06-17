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

from typing import Any, Dict, List, Optional, Set, Tuple
import math
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, EdgeType, SemanticNode
from conftopo.core.instruction_graph import InstructionGraph, GoalNode
from conftopo.core.landmark_roles import is_structural_label
from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
from conftopo.core.rule_scorer import compute_semantic_bias
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.heavy_perceiver import GroundingDINOBackend, HeavyPerceiver, ObjectObservation
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.clip_gdino_report_builder import ClipGdinoReportBuilder
from conftopo.perception.perception_trigger import PerceptionTrigger, TriggerState
from conftopo.perception.vlm_perceiver import VLMPerceiver
from conftopo.navigation.pathfinder_executor import GlobalGraphPlanner


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def _split_compound_label(label: str) -> Set[str]:
    """Split compound target like 'oven and stove' into {'oven', 'stove', 'oven and stove'}."""
    parts = {label.lower().strip()}
    if " and " in label:
        parts.update(p.strip().lower() for p in label.split(" and ") if p.strip())
    if " or " in label:
        parts.update(p.strip().lower() for p in label.split(" or ") if p.strip())
    return parts


def _label_matches_goal(obs_label: str, target_labels: Set[str]) -> bool:
    """Fuzzy label match: exact, substring-contains, or single-word overlap.

    VLMs may return abbreviated or paraphrased labels (e.g. "toy" for goal
    "plush toy", "couch" for "sofa").  This helper accepts a match when:
      1. Exact match (obs_label in target_labels), OR
      2. obs_label is a substring of any target label, OR
      3. Any target label is a substring of obs_label, OR
      4. Any individual word in obs_label appears in a target label's words
         (bidirectional single-word overlap, ignoring very short words).
    """
    obs = obs_label.lower().strip()
    if obs in target_labels:
        return True
    for tl in target_labels:
        if obs in tl or tl in obs:
            return True
    obs_words = {w for w in obs.split() if len(w) > 2}
    for tl in target_labels:
        tl_words = {w for w in tl.split() if len(w) > 2}
        if obs_words & tl_words:
            return True
    return False


DEFAULT_HEAVY_OBJECT_VOCABULARY = [
    "rack",
    "chair",
    "table",
    "door",
    "sofa",
    "bed",
    "sink",
    "toilet",
    "cabinet",
    "fridge",
    "tv",
    "plant",
]

# Vocabulary used when the planner's structure target is a portal or a
# structural landmark — keep ONLY visually grounded object-like words that
# GroundingDINO can reliably detect.  Spatial-concept words (hallway,
# corridor, stair, entrance, opening...) are NOT objects; they produce
# near-zero-confidence noise detections and must NOT appear here.
DEFAULT_STRUCTURAL_HEAVY_VOCABULARY = [
    "door",
    "doorway",
    "window",
    "arch",
    "archway",
    "counter",
    "gate",
]

# v5: object-category search priors. When a goal carries no dataset-provided
# room_prior / landmarks, fall back to these so exploration is biased toward
# rooms / landmarks where the target usually lives (instead of random
# frontier wandering). Keys are matched against goal.target_object substrings.
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
    "plush toy": ["bed", "sofa", "couch", "shelf", "cabinet", "table"],
    "stuffed": ["bed", "sofa", "couch", "shelf", "cabinet", "table"],
    "toy": ["bed", "sofa", "couch", "shelf", "cabinet", "table"],
    "wardrobe": ["bed", "cabinet"],
    "pillow": ["bed", "sofa"],
    "tv": ["sofa", "cabinet"],
    "towel": ["sink", "toilet"],
    "plate": ["table", "counter"],
}


def _category_priors(target_object: Optional[str], table: Dict[str, List[str]]) -> List[str]:
    """Return prior labels for the first category key found in ``target_object``."""
    if not target_object:
        return []
    text = str(target_object).lower()
    for key, vals in table.items():
        if key in text:
            return list(vals)
    return []


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

        # Agent state
        # Topo memory stores episode-start-relative positions.
        self._position: Optional[np.ndarray] = None
        self._origin_position: Optional[np.ndarray] = None
        self._heading: float = 0.0
        self._prev_position: Optional[np.ndarray] = None
        self._cur_vp_id: Optional[str] = None

        # Observation cache (set in observe())
        self._cur_rgb: Optional[Any] = None
        self._cur_rgb_embed: Optional[np.ndarray] = None
        self._cur_report: PerceptionReport = PerceptionReport()
        self._report_builder = ClipGdinoReportBuilder()
        self._perception_trigger = PerceptionTrigger(self.config.perception, self.config.memory)
        self._trigger_state = TriggerState()
        # Temporally-smoothed room classifier (replaces raw single-frame label).
        self._room_clf = RoomClassifier(RoomClassifierConfig())
        self.heavy_perceiver: Optional[HeavyPerceiver] = (
            HeavyPerceiver(
                GroundingDINOBackend(
                    config_path=self.config.perception.groundingdino_config,
                    checkpoint_path=self.config.perception.groundingdino_checkpoint,
                    device=self.config.perception.groundingdino_device,
                    box_threshold=self.config.perception.object_detection_threshold,
                    text_threshold=self.config.perception.groundingdino_text_threshold,
                ),
                min_confidence=self.config.perception.object_detection_threshold,
            )
            if self.config.perception.heavy_enabled
            else None
        )
        self.vlm_perceiver: Optional[VLMPerceiver] = None
        if self.config.perception.backend == "vlm":
            self._init_vlm_perceiver()

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
        self._environment_landmark_labels: Set[str] = set()
        self._goal_local_step: int = 0
        self._goal_travel_distance: float = 0.0
        self._last_heavy_step: Optional[int] = None
        self._last_heavy_summary_step: Optional[int] = None
        self._heavy_perception_calls: int = 0
        self._object_merge_count: int = 0
        self._last_heavy_debug: Dict[str, Any] = {}
        self._vlm_perception_attempts: int = 0
        self._last_stop_debug: Dict[str, Any] = {}
        self._recent_positions: List[np.ndarray] = []
        self._last_stuck_recovery_step: Optional[int] = None
        self._stuck_recovery_count: int = 0
        self._heavy_confirm_eval_step: Optional[int] = None
        self._nav_phase: str = "explore"
        self._confirm_buffer: List[bool] = []
        self._stop_buffer: List[bool] = []
        self._approach_remaining: int = 0
        self._approach_bearing: str = "center"
        self._approach_lost_count: int = 0
        self._approach_steps_used: int = 0
        self._approach_best_bbox_area: float = 0.0
        self._approach_peak_travel: float = 0.0
        self._approach_best_anchor_dist: float = float("inf")
        self._approach_entry_anchor_dist: float = float("inf")
        self._approach_align_count: int = 0
        self._approach_advance_steps: int = 0
        self._goal_latched: bool = False
        self._previous_goal_label: Optional[str] = None
        self._previous_goal_object_nodes: List[str] = []
        self._servo_prev_bbox_area: float = 0.0
        self._servo_bbox_shrink_count: int = 0
        self._servo_visual_advance_count: int = 0
        self._servo_returning_to_peak: bool = False
        self._approach_peak_position: Optional[np.ndarray] = None
        self._confirm_window_active: bool = False
        self._confirm_window_steps: int = 0
        self._confirm_window_lost_count: int = 0
        self._anchor_progress_window: List[float] = []
        self._explored_rooms_no_target: Dict[str, int] = {}
        self._room_step_counter: Dict[str, int] = {}
        self._cur_vlm_mode: str = "explore"
        self._reset_reground_state()
        self._global_planner = GlobalGraphPlanner()
        # Optional callable injected by the runtime to ask the navmesh
        # whether a candidate position is reachable. Signature:
        #   probe_fn(target_position_world: np.ndarray, target_node_id: str)
        #     -> {"reachable": bool, "geodesic_distance": float}
        # When ``None`` the planner falls back to a topo-graph reachability
        # estimate (NAVIGABLE shortest path from the current waypoint).
        self._reachability_probe = None

    def _init_vlm_perceiver(self) -> None:
        """Lazily initialise the VLM perceiver from config."""
        from conftopo.perception.vlm_backend import Qwen3VLBackend
        pcfg = self.config.perception
        backend = Qwen3VLBackend(
            api_base=pcfg.vlm_api_base,
            model=pcfg.vlm_model,
            timeout=pcfg.vlm_timeout,
        )
        self.vlm_perceiver = VLMPerceiver(backend)

    def reset(self):
        """Full reset for new episode (clear memory)."""
        self._position = None
        self._origin_position = None
        self._heading = 0.0
        self._prev_position = None
        self._cur_vp_id = None
        self._cur_rgb = None
        self._cur_rgb_embed = None
        self._cur_report = PerceptionReport()
        self._room_clf.reset()
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
        self._last_structure_target_id: Optional[str] = None
        self._goal_local_step = 0
        self._goal_travel_distance = 0.0
        self._last_heavy_step = None
        self._last_heavy_summary_step = None
        self._trigger_state = TriggerState()
        self._heavy_perception_calls = 0
        self._object_merge_count = 0
        self._last_heavy_debug = {}
        self._vlm_perception_attempts = 0
        self._nav_phase = "explore"
        self._confirm_buffer = []
        self._stop_buffer = []
        self._approach_remaining = 0
        self._approach_bearing = "center"
        self._approach_lost_count = 0
        self._approach_steps_used = 0
        self._approach_best_bbox_area = 0.0
        self._approach_peak_travel = 0.0
        self._approach_best_anchor_dist = float("inf")
        self._approach_entry_anchor_dist = float("inf")
        self._approach_peak_position = None
        self._approach_align_count = 0
        self._approach_advance_steps = 0
        self._goal_latched = False
        self._previous_goal_label = None
        self._previous_goal_object_nodes = []
        self._servo_prev_bbox_area = 0.0
        self._servo_bbox_shrink_count = 0
        self._servo_visual_advance_count = 0
        self._servo_returning_to_peak = False
        self._confirm_window_active = False
        self._confirm_window_steps = 0
        self._confirm_window_lost_count = 0
        self._anchor_progress_window = []
        self._explored_rooms_no_target = {}
        self._room_step_counter = {}
        self._cur_vlm_mode = "explore"
        self._reset_reground_state()

    def set_environment_landmark_labels(self, labels: List[str]) -> None:
        """Mark landmark labels that are scene-level anchors rather than goal hints."""
        self._environment_landmark_labels = {str(label) for label in labels}

    def set_heavy_perceiver(self, perceiver: Optional[HeavyPerceiver]) -> None:
        """Inject a heavy perceiver backend, mainly for tests and offline runs."""
        self.heavy_perceiver = perceiver
        self.config.perception.heavy_enabled = perceiver is not None

    def set_new_goal(self, goal: GoalNode):
        """Switch to a new goal within the same episode (memory preserved).

        This is the key for multi-goal: DynamicTopoMap is NOT cleared.
        """
        old_label = None
        if self.instruction_graph is not None:
            old_goal = self.instruction_graph.get_current_goal()
            old_label = getattr(old_goal, "target_object", None) if old_goal else None
        old_goal_nodes: List[str] = []
        if old_label:
            old_targets = _split_compound_label(old_label)
            for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
                if node.label and _label_matches_goal(node.label, old_targets):
                    old_goal_nodes.append(node.node_id)

        self.reset_keep_memory()
        self._clear_sticky("new_goal")

        self._previous_goal_label = old_label
        self._previous_goal_object_nodes = old_goal_nodes

        # v5: backfill category search priors when the dataset goal has none,
        # so exploration is biased toward likely rooms / landmarks.
        if not getattr(goal, "room_prior", None):
            room_prior = _category_priors(goal.target_object, OBJECT_CATEGORY_ROOM_PRIORS)
            if room_prior:
                goal.room_prior = room_prior
        if not getattr(goal, "landmarks", None):
            landmark_prior = _category_priors(goal.target_object, OBJECT_CATEGORY_LANDMARK_PRIORS)
            if landmark_prior:
                goal.landmarks = landmark_prior

        if self.instruction_graph is None:
            self.instruction_graph = InstructionGraph(
                goal_type="object_goal",
                goal_nodes=[goal],
            )
        else:
            self.instruction_graph.set_current_goal(goal)

        self.perceiver.set_goal_labels(
            labels=[goal.target_object],
            embeddings=goal.target_embedding[np.newaxis, :] if goal.target_embedding is not None else None,
        )
        if goal.landmarks:
            self.perceiver.set_landmark_labels(
                labels=goal.landmarks,
                embeddings=goal.landmark_embeddings,
            )

        # Reset target_relevance for non-current-goal objects, but preserve
        # all existing nodes across goal switches for multi-goal memory.
        for node in self.topo_map._nodes.values():
            node.attributes["cross_goal_preserved"] = True
            if node.node_type in (NodeType.OBJECT, NodeType.LANDMARK):
                node.attributes["target_relevance"] = 0.0

        if (self._previous_goal_label
                and self._previous_goal_label == goal.target_object
                and self._previous_goal_object_nodes):
            for nid in self._previous_goal_object_nodes:
                prev_node = self.topo_map.get_node(nid)
                if prev_node is not None:
                    prev_node.attributes["target_relevance"] = 0.8
                    prev_node.attributes["repeated_goal_source"] = True
        self._goal_idx += 1
        self._goal_local_step = 0
        self._goal_travel_distance = 0.0
        self._last_heavy_step = None
        self._last_stop_debug = {}
        self._recent_positions = []
        self._last_stuck_recovery_step = None
        self._stuck_recovery_count = 0
        self._heavy_confirm_eval_step = None
        self._nav_phase = "explore"
        self._confirm_buffer = []
        self._stop_buffer = []
        self._approach_remaining = 0
        self._approach_bearing = "center"
        self._approach_lost_count = 0
        self._approach_steps_used = 0
        self._approach_best_bbox_area = 0.0
        self._approach_peak_travel = 0.0
        self._approach_best_anchor_dist = float("inf")
        self._approach_entry_anchor_dist = float("inf")
        self._approach_peak_position = None
        self._approach_align_count = 0
        self._approach_advance_steps = 0
        self._goal_latched = False
        self._servo_prev_bbox_area = 0.0
        self._servo_bbox_shrink_count = 0
        self._servo_visual_advance_count = 0
        self._servo_returning_to_peak = False
        self._confirm_window_active = False
        self._confirm_window_steps = 0
        self._confirm_window_lost_count = 0
        self._anchor_progress_window = []
        self._explored_rooms_no_target = {}
        self._room_step_counter = {}
        self._cur_vlm_mode = "explore"
        self._reset_reground_state()

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
        self._cur_rgb = obs.get("rgb")
        if self._origin_position is None:
            self._origin_position = raw_position.copy()
        self._prev_position = self._position
        self._position = raw_position - self._origin_position
        if self._prev_position is not None and self._position is not None:
            self._goal_travel_distance += float(np.linalg.norm(self._position - self._prev_position))
        self._heading = obs.get('heading', 0.0)
        self._recent_positions.append(self._position.copy())
        if len(self._recent_positions) > 30:
            self._recent_positions = self._recent_positions[-30:]

        # Visual embedding (CLIP)
        if 'rgb_embed' in obs:
            self._cur_rgb_embed = np.array(obs['rgb_embed'], dtype=np.float32)
        elif 'pano_rgb_embed' in obs:
            self._cur_rgb_embed = np.array(obs['pano_rgb_embed'], dtype=np.float32)
        else:
            self._cur_rgb_embed = None

        # Run CLIP perception → unified PerceptionReport
        light_out = self.perceiver.perceive(self._cur_rgb_embed) if self._cur_rgb_embed is not None else {}
        self._cur_report = self._report_builder.build_light(
            light_out, self._cur_rgb_embed, self.topo_map.current_step,
        )

    def update_memory(self) -> None:
        """Update DynamicTopoMap with new information.

        1. Add current position as visited waypoint
        2. Run room/object classification → add semantic nodes
        3. Generate frontier-like nodes from unexplored directions
        4. Apply confidence decay and pruning
        """
        if self._position is None:
            return

        self._goal_local_step += 1

        # 1. Add/update visited waypoint
        cur_vp = self._add_visited_waypoint()

        self._consume_reached_frontiers(cur_vp)

        # 2. Add semantic nodes (room, landmark detection)
        room_label = self._add_semantic_nodes(cur_vp)

        # 2b. Add object-level heavy detections when triggered;
        # returns final room label (VLM room preferred over CLIP room).
        room_label = self._add_heavy_object_nodes(cur_vp, room_label) or room_label
        self._record_view_object_labels(cur_vp)

        # 3. Generate frontier-like nodes
        self._generate_frontiers(cur_vp)

        # 4. Memory maintenance
        self.topo_map.decay_all_confidences()
        self.topo_map.merge_nearby_nodes(NodeType.WAYPOINT_FRONTIER)
        self.topo_map.adaptive_granularity(self._position)
        self.topo_map.prune_low_confidence(self._position)
        if self.topo_map.num_nodes > self.config.memory.max_nodes:
            self.topo_map.prune_low_confidence(self._position)

        # 5. Explicit waypoint->room binding. Done after adaptive_granularity
        # so room summaries created/refreshed this step are visible.
        self.topo_map.assign_waypoint_to_room(cur_vp, view_room_label=room_label)

        # 6. Negative memory: track rooms explored without finding the goal.
        self._update_negative_room_memory()

    def _update_negative_room_memory(self) -> None:
        """Mark a room as searched if agent has spent 20+ steps in it
        without detecting the goal object."""
        cur_room = self._current_room_id()
        if cur_room is None or cur_room in self._explored_rooms_no_target:
            return
        self._room_step_counter[cur_room] = self._room_step_counter.get(cur_room, 0) + 1
        if self._room_step_counter[cur_room] < 20:
            return
        has_goal_in_room = False
        if self.instruction_graph is not None:
            goal = self.instruction_graph.get_current_goal()
            target_label = getattr(goal, "target_object", None) if goal is not None else None
            if target_label:
                target_labels = _split_compound_label(target_label)
                for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
                    if node.label and _label_matches_goal(node.label, target_labels):
                        obj_room = node.attributes.get("anchor_room_id")
                        if obj_room == cur_room:
                            has_goal_in_room = True
                            break
        if not has_goal_in_room:
            self._explored_rooms_no_target[cur_room] = self._goal_local_step

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
        if node.node_type == NodeType.OBJECT and self._is_goal_object_node(node):
            self._goal_latched = False

    def _consume_or_block_target(self, node_id: str, reason: str) -> Dict[str, Any]:
        node = self.topo_map.get_node(node_id)
        event = {"target_node_id": node_id, "reason": reason, "action": "missing"}
        if node is None:
            self._last_navigation_event = event
            return event
        node.attributes["consume_reason"] = reason
        node.attributes["consume_step"] = self.topo_map.current_step
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

    def _is_anchor_object(self, node: SemanticNode) -> bool:
        return (
            node.node_type == NodeType.OBJECT
            and node.attributes.get("position_source") == "anchor_waypoint"
        )

    def _anchor_dist(self, node: SemanticNode) -> float:
        """Distance from agent to the anchor waypoint of an object node."""
        anchor_id = (
            node.attributes.get("anchor_waypoint_id")
            or node.attributes.get("observed_from")
        )
        anchor = self.topo_map.get_node(anchor_id) if anchor_id else None
        pos = anchor.position if anchor is not None else node.position
        if self._position is None:
            return float("inf")
        return float(np.linalg.norm(pos - self._position))

    def on_navigation_event(self, target_node_id: Optional[str], event: str) -> Dict[str, Any]:
        if target_node_id is None:
            out = {"target_node_id": None, "reason": event, "action": "ignored"}
            self._last_navigation_event = out
            return out
        if event == "collision_blocked":
            return self._handle_collision_blocked(target_node_id)
        if event == "target_reached":
            return self._handle_target_reached(target_node_id)
        return self._consume_or_block_target(target_node_id, event)

    def _handle_target_reached(self, target_node_id: str) -> Dict[str, Any]:
        """Handle executor reach events with node-type-specific semantics.

        For OBJECT nodes all positions are semantic anchors (waypoints), not
        metric object centers.  Reaching the anchor triggers
        ``approach_confirm`` if this is a goal object.
        """
        node = self.topo_map.get_node(target_node_id)
        if node is None:
            out = {"target_node_id": target_node_id, "reason": "target_reached", "action": "missing"}
            self._last_navigation_event = out
            return out

        if node.node_type == NodeType.OBJECT:
            anchor_d = self._anchor_dist(node)
            if self._is_goal_object_node(node):
                if (self._nav_phase != "approach_confirm"
                        and anchor_d <= self._APPROACH_CONFIRM_ANCHOR_RADIUS
                        and self._vlm_sees_goal_now()):
                    self._enter_approach_confirm()
                out = {
                    "target_node_id": target_node_id,
                    "reason": "anchor_reached_enter_approach_confirm",
                    "action": "anchor_reached_awaiting_confirm",
                    "anchor_dist": round(anchor_d, 3),
                    "vlm_confirmed": self._nav_phase == "approach_confirm",
                }
                self._last_navigation_event = out
                return out
            return self._consume_or_block_target(target_node_id, "target_reached")

        return self._consume_or_block_target(target_node_id, "target_reached")

    def _handle_collision_blocked(self, target_node_id: str) -> Dict[str, Any]:
        """Handle collision without wrongly consuming/blocking the target.

        - FRONTIER: block (but do not permanently consume)
        - OBJECT: mark current approach viewpoint as failed, keep the object
        - VISITED/other: reduce edge traversability, clear sticky
        """
        node = self.topo_map.get_node(target_node_id)
        result = {"target_node_id": target_node_id, "reason": "collision_blocked", "action": "ignored"}
        if node is None:
            self._last_navigation_event = result
            return result

        if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
            self._block_target(target_node_id, "collision_blocked")
            result["action"] = "blocked_frontier"
        elif node.node_type == NodeType.OBJECT:
            failed = node.attributes.setdefault("failed_approach_viewpoints", [])
            if self._cur_vp_id and self._cur_vp_id not in failed:
                failed.append(self._cur_vp_id)
            result["action"] = "marked_failed_approach"
        else:
            result["action"] = "collision_noted"

        if self._cur_vp_id:
            self.topo_map.reduce_edge_traversability(self._cur_vp_id, target_node_id)

        if self._sticky_target_id == target_node_id:
            self._clear_sticky("collision_blocked")
        self._last_navigation_event = result
        return result

    def _consume_reached_frontiers(self, cur_vp: str) -> None:
        radius = self.config.planning.frontier_consume_radius
        for node in self.topo_map.find_nodes_within_radius(
            self._position, radius=radius, node_type=NodeType.WAYPOINT_FRONTIER
        ):
            self._consume_or_block_target(node.node_id, "frontier_reached")

    def _reset_reground_state(self) -> None:
        self._reground_state = "idle"
        self._reground_scan_rotated = 0
        self._reground_anchor_node_id = None
        self._reground_target_node_id = None
        self._reground_anchor_position = None
        self._target_object_detected_this_scan = False

    def _try_start_regrounding(self, plan_output: Dict[str, Any]) -> bool:
        if self._position is None or self._reground_state != "idle":
            return False
        if not plan_output.get("requires_regrounding"):
            return False
        target_pos = plan_output.get("target_position")
        if target_pos is None:
            return False
        anchor_pos = np.array(target_pos, dtype=np.float32)
        if float(np.linalg.norm(anchor_pos - self._position)) >= 0.8:
            return False
        self._reground_state = "scanning"
        self._reground_scan_rotated = 0
        self._reground_anchor_node_id = plan_output.get("anchor_waypoint_id")
        self._reground_anchor_position = anchor_pos.copy()
        self._target_object_detected_this_scan = False
        return True

    def _reground_scan_action(self) -> Dict[str, Any]:
        self._reground_scan_rotated += 1
        if self._reground_scan_rotated >= 3:
            if not self._target_object_detected_this_scan:
                if self._reground_anchor_node_id:
                    self._block_target(self._reground_anchor_node_id, "reground_failed", ttl=50)
                cur_room = self._current_room_id()
                if cur_room:
                    self._explored_rooms_no_target[cur_room] = self._goal_local_step
                self._nav_phase = "explore"
            self._reset_reground_state()
        return {
            "action": "turn_right",
            "target_position": None,
            "mode": "local_reground_scan",
            "reground_state": self._reground_state,
            "target_node_id": self._reground_target_node_id,
        }

    def _is_consumed_frontier(self, node) -> bool:
        return (
            node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE)
            and (node.node_id in self._consumed_frontier_ids or node.attributes.get("consumed", False))
        )

    def _resolve_object_waypoint_anchor(
        self, node: SemanticNode
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Pick the nearest visited waypoint to a semantic OBJECT node.

        Returns ``(target_position, extras)`` or ``(None, {})`` when no
        waypoint exists yet (the caller should fall back to legacy
        approach-pose targeting).
        """
        approach = node.attributes.get("best_approach_position")
        if approach is not None:
            probe_pos = np.asarray(approach, dtype=np.float32)
        else:
            probe_pos = node.position
        nearest_wp = self.topo_map.find_nearest_node(
            probe_pos, NodeType.WAYPOINT_VISITED
        )
        if nearest_wp is None:
            return None, {}
        if (
            self._cur_vp_id is not None
            and nearest_wp.node_id == self._cur_vp_id
            and approach is not None
        ):
            # We are already at the nearest waypoint; using it as target
            # would cause an immediate reach trigger. Fall back to the
            # object approach pose so the executor still moves closer.
            return np.asarray(approach, dtype=np.float32), {
                "target_anchor_type": "object_approach",
                "semantic_target_node_id": node.node_id,
            }
        extras = {
            "target_anchor_type": "object_waypoint_anchor",
            "semantic_target_node_id": node.node_id,
            "anchor_waypoint_id": nearest_wp.node_id,
            "anchor_waypoint_position": nearest_wp.position.tolist(),
            "requires_regrounding": True,
        }
        return nearest_wp.position.copy(), extras

    def _target_output_for_node(self, node: SemanticNode) -> Tuple[np.ndarray, Dict[str, Any]]:
        target_pos = node.position.copy()
        extras: Dict[str, Any] = {}
        if node.attributes.get("is_semantic_anchor"):
            anchor = self.topo_map.get_object_anchor(node.node_id)
            anchor_wp_pos = anchor.get("waypoint_position") if anchor else None
            if anchor_wp_pos is not None:
                target_pos = np.array(anchor_wp_pos, dtype=np.float32)
                extras.update({
                    "requires_regrounding": True,
                    "semantic_target_node_id": node.node_id,
                    "target_anchor_type": "folded_object_anchor",
                    "anchor_waypoint_id": anchor.get("waypoint_id"),
                    "anchor_waypoint_position": anchor.get("waypoint_position"),
                    "anchor_room_id": anchor.get("room_id"),
                })
            elif getattr(self.config.planning, "object_route_to_waypoint_anchor", False):
                # Folded anchor without a stored waypoint position — fall
                # back to the nearest visited waypoint so we still route
                # to the navigation backbone.
                wp_pos, wp_extras = self._resolve_object_waypoint_anchor(node)
                if wp_pos is not None:
                    target_pos = wp_pos
                    extras.update(wp_extras)
                    extras["target_anchor_type"] = "folded_object_anchor"
        elif node.node_type == NodeType.OBJECT:
            anchor_id = node.attributes.get("anchor_waypoint_id") or node.attributes.get("observed_from")
            anchor = self.topo_map.get_node(anchor_id) if anchor_id else None
            if anchor is not None:
                range_bin = node.attributes.get("range_bin", "unknown")
                return anchor.position.copy(), {
                    "target_anchor_type": "object_anchor",
                    "semantic_target_node_id": node.node_id,
                    "anchor_waypoint_id": anchor.node_id,
                    "anchor_waypoint_position": anchor.position.tolist(),
                    "requires_regrounding": range_bin in ("near", "mid", "unknown"),
                    "vlm_range_bin": range_bin,
                    "vlm_bearing": node.attributes.get("bearing", "unknown"),
                    "vlm_visibility": node.attributes.get("visibility", "unknown"),
                }
            approach = node.attributes.get("best_approach_position")
            if approach is not None:
                target_pos = np.array(approach, dtype=np.float32)
                extras["target_anchor_type"] = "object_approach"
                extras["semantic_target_node_id"] = node.node_id
        return target_pos, extras

    def _sticky_plan_if_valid(self, candidates, candidate_ids, scores) -> Optional[Dict[str, Any]]:
        cfg = self.config.planning
        if not cfg.sticky_target_enabled or self._sticky_target_id is None:
            return None
        node = self.topo_map.get_node(self._sticky_target_id)
        if node is None or self._is_consumed_frontier(node) or self._is_blocked_target(node.node_id):
            self._clear_sticky("target_unavailable")
            return None
        # Allow sticky to persist even if the node was filtered out of candidates
        # this step (e.g. due to structure_target change). Only release if the
        # node itself has been deleted from the topo_map.
        target_pos, target_extras = self._target_output_for_node(node)
        dist = float(np.linalg.norm(target_pos - self._position))
        if dist <= cfg.sticky_reach_radius:
            if target_extras.get("requires_regrounding"):
                plan = {
                    "target_node_id": node.node_id,
                    "target_position": target_pos,
                    "is_exploration": node.node_type == NodeType.WAYPOINT_FRONTIER,
                    "scores": scores,
                    "candidate_ids": candidate_ids,
                    "sticky_debug": self._last_plan_debug,
                }
                plan.update(target_extras)
                return plan
            if node.node_type == NodeType.OBJECT and self._is_goal_object_node(node):
                # Reaching the approach waypoint is not goal completion for objects.
                pass
            else:
                self._consume_or_block_target(node.node_id, "target_reached")
                self._clear_sticky("target_reached")
                return None

        # Release: distance increased significantly (detour)
        detour_ratio = getattr(cfg, "sticky_detour_release_ratio", 1.5)
        if self._sticky_last_distance is not None and dist > self._sticky_last_distance * detour_ratio:
            self._consume_or_block_target(node.node_id, "sticky_detour")
            self._clear_sticky("detour")
            return None

        # Release: a closer confirmed goal object exists
        if self._is_goal_object_node(node):
            for obj in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
                if obj.node_id != node.node_id and self._is_goal_object_node(obj):
                    obj_anchor_d = self._anchor_dist(obj)
                    if obj_anchor_d < dist * 0.5:
                        self._clear_sticky("closer_goal_found")
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
        }
        plan = {
            "target_node_id": node.node_id,
            "target_position": target_pos,
            "is_exploration": node.node_type == NodeType.WAYPOINT_FRONTIER,
            "scores": scores,
            "candidate_ids": candidate_ids,
            "sticky_debug": self._last_plan_debug,
        }
        plan.update(target_extras)
        return plan

    def _record_view_context(self, cur_vp: str, room_label: str, room_conf: float) -> None:
        """Store CLIP room/view tags on the waypoint instead of spawning clip_room nodes."""
        node = self.topo_map.get_node(cur_vp)
        if node is None:
            return
        node.attributes["view_room_label"] = room_label
        node.attributes["view_room_confidence"] = float(room_conf)
        node.step_id = self.topo_map.current_step

    def _record_view_context_smoothed(
        self,
        cur_vp: str,
        room_label: str,
        room_conf: float,
        room_scores: Dict[str, float],
        is_transition: bool,
    ) -> None:
        """Store smoothed room classification on the waypoint.

        Writes both top-1 ``view_room_label`` (for backward-compat) and the
        full ``room_scores`` distribution so consumers can read the posterior.
        """
        node = self.topo_map.get_node(cur_vp)
        if node is None:
            return
        node.attributes["view_room_label"] = room_label
        node.attributes["view_room_confidence"] = float(room_conf)
        node.attributes["room_scores"] = {k: round(v, 4) for k, v in room_scores.items()}
        if is_transition:
            node.attributes["room_transition"] = True
        node.step_id = self.topo_map.current_step

    def _record_scene_vocabulary(self, cur_vp: str, label: str, confidence: float) -> None:
        """Store environment vocabulary hits on the waypoint (not map landmarks)."""
        node = self.topo_map.get_node(cur_vp)
        if node is None:
            return
        tags = node.attributes.setdefault("scene_vocabulary", [])
        tags.append({"label": label, "confidence": float(confidence), "step": self.topo_map.current_step})
        if len(tags) > 12:
            del tags[:-12]

    def _record_view_object_labels(self, cur_vp: str) -> None:
        """Store object labels seen from this waypoint for later search scoring."""
        node = self.topo_map.get_node(cur_vp)
        if node is None or self._cur_report.source == "none":
            return
        labels: set = set()
        threshold = float(self.config.perception.object_threshold)
        for obs in self._cur_report.objects:
            if obs.label and float(obs.confidence) >= threshold * 0.8:
                labels.add(str(obs.label).strip().lower())
        for label, sim in self._cur_report.goal_scores:
            if float(sim) >= threshold:
                labels.add(str(label).strip().lower())
        for label, sim in self._cur_report.landmark_scores:
            if float(sim) >= float(self.config.perception.landmark_threshold):
                labels.add(str(label).strip().lower())
        if not labels:
            return
        seen = node.attributes.setdefault("view_object_labels", [])
        for lab in sorted(labels):
            if lab and lab not in seen:
                seen.append(lab)
        if len(seen) > 20:
            del seen[:-20]

    def _link_semantic_to_room_summary(self, node_id: str, room_label: Optional[str]) -> None:
        if not room_label or room_label in ("unknown", ""):
            return
        if self._position is None:
            return
        summary = self.topo_map.find_nearby_room_summary(self._position, label=room_label)
        if summary is not None:
            self.topo_map.add_edge(node_id, summary.node_id, EdgeType.BELONGS_TO)

    def _add_semantic_nodes(self, cur_vp: str) -> Optional[str]:
        """Add navigation landmarks / light objects from perception.

        Environment CLIP tags and instantaneous room labels are stored on the
        current waypoint as context, not as spatial map nodes. Distant room
        summaries are created later by DynamicTopoMap.adaptive_granularity().
        """
        if self._cur_report.source == "none":
            return None

        pcfg = self.config.perception
        raw_label = self._cur_report.room_label
        raw_conf = self._cur_report.room_confidence

        recent_obj_labels = [obs.label for obs in self._cur_report.objects]

        clf_result = self._room_clf.update(
            raw_label=raw_label,
            raw_confidence=raw_conf,
            position=self._position,
            object_labels=recent_obj_labels,
        )
        room_label = clf_result.confirmed_label or "unknown"
        room_conf = clf_result.top_confidence

        if room_label != "unknown" and room_conf >= pcfg.room_threshold:
            self._record_view_context_smoothed(
                cur_vp, room_label, room_conf,
                clf_result.scores, clf_result.is_transition,
            )
        elif clf_result.scores:
            node = self.topo_map.get_node(cur_vp)
            if node is not None:
                node.attributes["room_scores"] = {
                    k: round(v, 4) for k, v in clf_result.scores.items()
                }

        # CLIP goal object nodes — skip when VLM is the primary object source
        # to avoid duplicate / false-positive anchor nodes.
        if pcfg.backend != "vlm":
            for label, sim in self._cur_report.goal_scores:
                sim = float(sim)
                if sim < pcfg.object_threshold:
                    continue
                existing = self.topo_map.find_nodes_within_radius(
                    self._position, radius=2.0, node_type=NodeType.OBJECT
                )
                matched = next(
                    (
                        o for o in existing
                        if o.label == label and self._object_room_compatible(o, room_label)
                    ),
                    None,
                )
                if matched:
                    matched.confidence = max(matched.confidence, sim)
                    matched.step_id = self.topo_map.current_step
                    self._update_object_room_context(matched, room_label)
                    if self._cur_rgb_embed is not None:
                        matched.embedding = self._cur_rgb_embed
                else:
                    clip_attrs = self._object_room_attributes(room_label, "light_clip")
                    clip_attrs.update({
                        "anchor_waypoint_id": cur_vp,
                        "observed_from": cur_vp,
                        "position_source": "anchor_waypoint",
                    })
                    vp_node = self.topo_map.get_node(cur_vp)
                    clip_pos = vp_node.position.copy() if vp_node is not None else self._position.copy()
                    obj_id = self.topo_map.add_node(
                        NodeType.OBJECT,
                        position=clip_pos,
                        embedding=self._cur_rgb_embed,
                        confidence=sim,
                        label=label,
                        attributes=clip_attrs,
                    )
                    self.topo_map.add_edge(cur_vp, obj_id, EdgeType.OBSERVED_AT)
                    self._link_semantic_to_room_summary(obj_id, room_label)

        # Navigation landmark: only goal/instruction hints, not environment vocabulary.
        for label, sim in self._cur_report.landmark_scores:
            sim = float(sim)
            if sim < pcfg.landmark_threshold:
                continue
            if label in self._environment_landmark_labels:
                self._record_scene_vocabulary(cur_vp, label, sim)
                continue
            existing = self.topo_map.find_nodes_within_radius(
                self._position, radius=3.0, node_type=NodeType.LANDMARK
            )
            matched = next((lm for lm in existing if lm.label == label), None)
            if matched:
                matched.confidence = max(matched.confidence, sim)
                matched.step_id = self.topo_map.current_step
                self._update_landmark_history(matched, cur_vp, sim)
                if self._cur_rgb_embed is not None:
                    matched.embedding = self._cur_rgb_embed
            else:
                lm_id = self.topo_map.add_node(
                    NodeType.LANDMARK,
                    position=self._position.copy(),
                    embedding=self._cur_rgb_embed,
                    confidence=sim,
                    label=label,
                    attributes=self._new_landmark_attributes(label, cur_vp, sim),
                )
                self.topo_map.add_edge(cur_vp, lm_id, EdgeType.VISIBLE_FROM)
                self._link_semantic_to_room_summary(lm_id, room_label)
        return room_label if room_label != "unknown" else None

    def _new_landmark_attributes(self, label: str, viewpoint_id: str, confidence: float) -> Dict[str, Any]:
        source = "environment" if label in self._environment_landmark_labels else "goal_hint"
        obs = {
            "viewpoint_id": viewpoint_id,
            "confidence": float(confidence),
            "step_id": self.topo_map.current_step,
            "source": source,
        }
        return {
            "landmark_source": source,
            "observations": [obs],
            "viewpoints": [viewpoint_id],
            "first_seen_step": self.topo_map.current_step,
            "last_seen_step": self.topo_map.current_step,
            "multi_view_count": 1,
            "granularity": "landmark",
        }

    def _update_landmark_history(self, node: SemanticNode, viewpoint_id: str, confidence: float) -> None:
        source = node.attributes.get("landmark_source", "goal_hint")
        obs = {
            "viewpoint_id": viewpoint_id,
            "confidence": float(confidence),
            "step_id": self.topo_map.current_step,
            "source": source,
        }
        node.attributes.setdefault("observations", []).append(obs)
        if viewpoint_id not in node.attributes.setdefault("viewpoints", []):
            node.attributes["viewpoints"].append(viewpoint_id)
        node.attributes.setdefault("first_seen_step", self.topo_map.current_step)
        node.attributes["last_seen_step"] = self.topo_map.current_step
        node.attributes["multi_view_count"] = max(
            1,
            len(node.attributes.get("viewpoints", [])),
            len(node.attributes.get("observations", [])),
        )

    @staticmethod
    def _normalized_room_label(room_label: Optional[str]) -> Optional[str]:
        if room_label is None:
            return None
        room = str(room_label).strip()
        if not room or room == "unknown":
            return None
        return room

    def _object_room_attributes(self, room_label: Optional[str], source: str) -> Dict[str, Any]:
        room = self._normalized_room_label(room_label)
        return {
            "room_context": room,
            "room_contexts": [room] if room is not None else [],
            "source": source,
        }

    def _object_room_compatible(self, node: SemanticNode, room_label: Optional[str]) -> bool:
        room = self._normalized_room_label(room_label)
        if room is None:
            return True
        known = node.attributes.get("room_contexts")
        if known is None:
            previous = node.attributes.get("room_context")
            known = [previous] if previous is not None else []
        known_rooms = {
            str(value).strip()
            for value in known
            if value is not None and self._normalized_room_label(str(value))
        }
        return not known_rooms or room in known_rooms

    def _update_object_room_context(self, node: SemanticNode, room_label: Optional[str]) -> None:
        room = self._normalized_room_label(room_label)
        if room is None:
            return
        node.attributes["room_context"] = room
        rooms = node.attributes.setdefault("room_contexts", [])
        if room not in rooms:
            rooms.append(room)

    def _heavy_labels(
        self,
        reason: str = "",
        summary=None,
        structure_target: Optional[SemanticNode] = None,
    ) -> List[str]:
        labels: List[str] = []
        goal = self.instruction_graph.get_current_goal() if self.instruction_graph else None
        if goal is not None and getattr(goal, "target_object", None):
            target = str(goal.target_object)
            # Split compound labels so GroundingDINO gets single-object prompts
            for part in _split_compound_label(target):
                labels.append(part)
            labels.extend(str(a) for a in getattr(goal, "attributes", []) if a)
            labels.extend(str(lm) for lm in getattr(goal, "landmarks", []) if lm)
        for label, sim in self._cur_report.goal_scores:
            if float(sim) >= self.config.perception.object_threshold:
                labels.append(str(label))

        align = bool(getattr(self.config.perception, "heavy_align_with_structure_target", True))
        used_structure_vocab = False
        if align and structure_target is not None:
            attrs = structure_target.attributes
            if structure_target.node_type == NodeType.ROOM:
                contains = attrs.get("contains_labels", [])
                if contains:
                    labels.extend(str(lbl) for lbl in contains if lbl)
                    used_structure_vocab = True
            elif structure_target.node_type == NodeType.LANDMARK and (
                attrs.get("structure_role") == "portal"
                or attrs.get("synthetic_portal")
                or is_structural_label(structure_target.label)
            ):
                labels.extend(DEFAULT_STRUCTURAL_HEAVY_VOCABULARY)
                # Pull in target room labels if portal recorded them.
                for room_label in attrs.get("structure_pair_labels", []) or []:
                    if room_label:
                        labels.append(str(room_label))
                used_structure_vocab = True

        if reason == "coarse_summary_context" and summary is not None:
            contains = summary.attributes.get("contains_labels", [])
            labels.extend(str(lbl) for lbl in contains if lbl)
        elif not used_structure_vocab:
            labels.extend(DEFAULT_HEAVY_OBJECT_VOCABULARY)
        seen = set()
        result = [x for x in labels if not (x in seen or seen.add(x))]
        max_labels = int(self.config.perception.heavy_summary_max_labels)
        return result[:max_labels]

    def _nearest_goal_anchor_dist(self) -> float:
        """Distance from agent to the closest goal-matching object anchor."""
        if self._position is None or self.instruction_graph is None:
            return float("inf")
        goal = self.instruction_graph.get_current_goal()
        target_label = getattr(goal, "target_object", None) if goal is not None else None
        if target_label is None:
            return float("inf")
        target_labels = _split_compound_label(target_label)
        best = float("inf")
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.label and _label_matches_goal(node.label, target_labels):
                d = self._anchor_dist(node)
                if d < best:
                    best = d
        return best

    def _nearest_goal_object_node(self) -> Optional[SemanticNode]:
        """Return the closest goal-matching object node, if any."""
        if self._position is None or self.instruction_graph is None:
            return None
        goal = self.instruction_graph.get_current_goal()
        target_label = getattr(goal, "target_object", None) if goal is not None else None
        if target_label is None:
            return None
        target_labels = _split_compound_label(target_label)
        best_node: Optional[SemanticNode] = None
        best_dist = float("inf")
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.label and _label_matches_goal(node.label, target_labels):
                d = self._anchor_dist(node)
                if d < best_dist:
                    best_dist = d
                    best_node = node
        return best_node

    def _try_refresh_goal_anchor_from_vlm(self, vlm: Dict[str, Any]) -> bool:
        """Promote goal object anchor when current VLM view is visually closer."""
        if not vlm.get("available") or not vlm.get("goal_visible"):
            return False
        if not (vlm.get("stop_grounding_ok") or vlm.get("grounding_ok")):
            return False
        node = self._nearest_goal_object_node()
        if node is None or self._cur_vp_id is None:
            return False
        vp = self.topo_map.get_node(self._cur_vp_id)
        if vp is None:
            return False
        return self.topo_map.promote_object_anchor_if_better(
            node,
            vp.position,
            self._cur_vp_id,
            vlm.get("bbox"),
            vlm.get("range_bin"),
            float(vlm.get("confidence", 0)),
        )

    def _check_goal_no_progress(self) -> None:
        """Goal-level no-progress: if the agent has not gotten meaningfully
        closer to its nearest goal anchor over a sliding window, abandon the
        current anchor / structure target so the planner re-selects (REPLAN).

        This complements ``_is_stuck`` (which only watches raw displacement):
        here we watch whether we are actually approaching the goal.
        """
        anchor_dist = self._nearest_goal_anchor_dist()
        if anchor_dist == float("inf"):
            # No goal anchor in memory -> nothing to measure against; reset.
            self._anchor_progress_window = []
            return
        self._anchor_progress_window.append(anchor_dist)
        if len(self._anchor_progress_window) > self._GOAL_NO_PROGRESS_STEPS:
            self._anchor_progress_window = self._anchor_progress_window[-self._GOAL_NO_PROGRESS_STEPS:]
        if len(self._anchor_progress_window) < self._GOAL_NO_PROGRESS_STEPS:
            return
        window = self._anchor_progress_window
        # Improvement = how much closer we got from the window start's best.
        improvement = max(window) - min(window)
        recent_best = min(window[-1], anchor_dist)
        earliest_best = min(window[: max(1, self._GOAL_NO_PROGRESS_STEPS // 4)])
        if (recent_best >= earliest_best - self._GOAL_NO_PROGRESS_EPS
                and improvement < self._GOAL_NO_PROGRESS_EPS):
            blocked_id = self._sticky_target_id or self._last_structure_target_id
            if blocked_id is not None:
                self._block_target(blocked_id, "goal_no_progress", ttl=40)
                if self._sticky_target_id == blocked_id:
                    self._clear_sticky("goal_no_progress")
            self._anchor_progress_window = []

    def _check_search_no_progress(self) -> None:
        """When no goal anchor exists, detect if we're stuck in one area.

        Every 30 steps without detecting the goal object, mark the current
        room as searched and block the current structure target to force a
        room switch.
        """
        if self._nearest_goal_anchor_dist() != float("inf"):
            return
        if self._goal_local_step <= 0 or self._goal_local_step % 30 != 0:
            return
        cur_room = self._current_room_id()
        if cur_room:
            self._explored_rooms_no_target[cur_room] = self._goal_local_step
            if self._last_structure_target_id:
                self._block_target(self._last_structure_target_id, "search_no_progress", ttl=30)
                self._clear_sticky("search_no_progress")

    def _should_run_heavy_perception(self) -> Tuple[bool, str]:
        self._trigger_state.last_heavy_step = self._last_heavy_step
        self._trigger_state.last_heavy_summary_step = self._last_heavy_summary_step
        self._trigger_state.goal_local_step = self._goal_local_step
        self._trigger_state.reground_state = self._reground_state
        self._trigger_state.nav_phase = self._nav_phase
        self._trigger_state.nearest_anchor_dist = self._nearest_goal_anchor_dist()
        self._trigger_state.confirm_window_active = self._confirm_window_active
        return self._perception_trigger.should_run(
            self._trigger_state,
            self.topo_map.current_step,
            self._cur_rgb,
            self._cur_report.best_goal_sim,
            self._position,
            self.topo_map,
            self._has_near_goal_object_node(radius=2.0),
            self.heavy_perceiver is not None or self.vlm_perceiver is not None,
        )

    def _object_position_for_observation(self, obs: ObjectObservation, cur_vp: str) -> np.ndarray:
        """Return map position for an object observation.

        All objects use the observing waypoint as their position (semantic
        anchor).  bbox is retained for grounding / stop confirmation only,
        never for depth estimation.
        """
        vp = self.topo_map.get_node(cur_vp)
        if vp is not None:
            return vp.position.copy()
        if self._position is not None:
            return self._position.copy()
        return np.zeros(3, dtype=np.float32)

    def _spatial_attrs_for_observation(self, obs: ObjectObservation, cur_vp: str) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "anchor_waypoint_id": cur_vp,
            "observed_from": cur_vp,
            "position_source": "anchor_waypoint",
        }
        if obs.source == "vlm":
            room_context = obs.room_context
            if room_context is None and self._cur_report.room_label != "unknown":
                room_context = self._cur_report.room_label
            attrs.update({
                "bearing": obs.bearing or "unknown",
                "range_bin": obs.range_bin or "unknown",
                "visibility": obs.visibility or "unknown",
                "visible": obs.visible if obs.visible is not None else obs.visibility != "not_visible",
                "spatial_relation": list(obs.spatial_relation or []),
                "vlm_room_context": room_context,
            })
        return attrs

    def _add_heavy_object_nodes(self, cur_vp: str, room_label: Optional[str]) -> Optional[str]:
        """Run heavy / VLM perception and upsert objects.  Returns final room label."""
        should_run, reason = self._should_run_heavy_perception()
        self._last_heavy_debug = {"ran": False, "reason": reason, "detections": 0}
        if not should_run:
            return room_label

        labels: List[str] = []
        struct_target_id = getattr(self, "_last_structure_target_id", None)
        backend_name = "vlm" if self.vlm_perceiver is not None else "clip_groundingdino"
        vlm_mode = "confirm" if self._nav_phase in ("confirm", "approach", "approach_confirm") else "explore"

        # --- VLM path: query the VLM and get a full PerceptionReport directly ---
        if self.vlm_perceiver is not None:
            self._vlm_perception_attempts += 1
            try:
                goal = self.instruction_graph.get_current_goal() if self.instruction_graph else None
                goal_text = getattr(goal, "target_object", "explore") if goal else "explore"
                labels = [str(goal_text)]
                vlm_report = self.vlm_perceiver.perceive(
                    self._cur_rgb,
                    goal_text,
                    visual_embed=self._cur_rgb_embed,
                    step_id=self.topo_map.current_step,
                    mode=vlm_mode,
                )
                observations = vlm_report.objects
                self._cur_report = vlm_report
            except Exception as exc:
                self._last_heavy_debug = {
                    "ran": False,
                    "reason": f"vlm_error:{type(exc).__name__}:{exc}",
                    "detections": 0,
                    "backend": "vlm",
                    "attempts": self._vlm_perception_attempts,
                }
                return room_label
        else:
            # --- Classic CLIP + GroundingDINO path ---
            summary = None
            if reason == "coarse_summary_context" and self._position is not None:
                summary = self.topo_map.find_nearby_room_summary(
                    self._position, radius=self.config.memory.summary_radius)
            structure_target = None
            if struct_target_id:
                structure_target = self.topo_map.get_node(struct_target_id)
            labels = self._heavy_labels(reason=reason, summary=summary, structure_target=structure_target)
            if not labels:
                self._last_heavy_debug = {"ran": False, "reason": "no_labels", "detections": 0}
                return room_label
            try:
                observations = self.heavy_perceiver.perceive(
                    self._cur_rgb,
                    labels,
                    visual_embedding=self._cur_rgb_embed,
                    view_heading=self._heading,
                    step_id=self.topo_map.current_step,
                )
            except RuntimeError as exc:
                self._last_heavy_debug = {"ran": False, "reason": f"backend_unavailable:{exc}", "detections": 0}
                return room_label
            light_out = self.perceiver.perceive(self._cur_rgb_embed) if self._cur_rgb_embed is not None else {}
            self._cur_report = self._report_builder.build_full(
                light_out, observations, self._cur_rgb_embed, self.topo_map.current_step,
            )

        self._perception_trigger.record_run(self._trigger_state, self.topo_map.current_step, reason)
        self._last_heavy_step = self._trigger_state.last_heavy_step
        self._last_heavy_summary_step = self._trigger_state.last_heavy_summary_step
        self._heavy_perception_calls += 1
        self._cur_vlm_mode = vlm_mode
        goal = self.instruction_graph.get_current_goal() if self.instruction_graph else None
        room_label = self._cur_report.room_label if self._cur_report.source != "none" else None
        room_priors = set(getattr(goal, "room_prior", []) or [])
        merges = 0
        for obs in observations:
            goal_target = getattr(goal, "target_object", None) if goal is not None else None
            goal_labels = _split_compound_label(goal_target) if goal_target else set()
            target_relevance = 1.0 if goal_labels and _label_matches_goal(obs.label, goal_labels) else 0.0
            # Structural detections are valuable as room anchors even when
            # they're not the goal: give them a small relevance so the
            # Phase 1 promotion gate can lift them into landmarks.
            if target_relevance <= 0.0 and is_structural_label(obs.label):
                target_relevance = max(target_relevance, 0.25)
            room_prior_score = 1.0 if room_label in room_priors else 0.0
            object_position = self._object_position_for_observation(obs, cur_vp)
            spatial_attrs = self._spatial_attrs_for_observation(obs, cur_vp)
            obj_id, merged = self.topo_map.upsert_object_observation(
                label=obs.label,
                bbox=obs.bbox,
                confidence=obs.confidence,
                position=object_position,
                embedding=obs.embedding,
                viewpoint_id=cur_vp,
                view_heading=obs.view_heading,
                room_context=room_label,
                target_relevance=target_relevance,
                room_prior_score=room_prior_score,
                source=obs.source,
                spatial_attrs=spatial_attrs,
            )
            self._link_semantic_to_room_summary(obj_id, room_label)
            if obs.source != "vlm":
                summary_id = self.topo_map.mark_recovered_from_summary(
                    obj_id,
                    object_position,
                    label=obs.label,
                )
                if summary_id is not None:
                    self.topo_map.add_edge(obj_id, summary_id, EdgeType.BELONGS_TO)
            obj_node = self.topo_map.get_node(obj_id)
            if obj_node is not None and target_relevance > 0.0:
                obj_node.attributes["goal_detection_step"] = self.topo_map.current_step
                obj_node.attributes["goal_detection_confidence"] = float(obs.confidence)
            if (
                obj_node is not None
                and self._reground_state in ("scanning", "searching")
                and self._is_goal_object_node(obj_node)
            ):
                self._target_object_detected_this_scan = True
                self._reground_target_node_id = obj_id
            if merged:
                merges += 1
                self._object_merge_count += 1
        self._last_heavy_debug = {
            "ran": True,
            "reason": reason,
            "detections": len(observations),
            "labels": labels,
            "merged": merges,
            "structure_target_id": struct_target_id,
            "backend": backend_name,
            "attempts": self._vlm_perception_attempts if backend_name == "vlm" else None,
            "vlm_mode": vlm_mode if backend_name == "vlm" else None,
            "nav_phase": self._nav_phase,
        }
        # Prefer VLM room over CLIP room when available.
        heavy_room = self._cur_report.room_label if self._cur_report.source != "none" else None
        if heavy_room and heavy_room != "unknown":
            return heavy_room
        return room_label

    def _generate_frontiers(self, cur_vp: str) -> None:
        """Generate frontier-like nodes from unexplored directions.

        Strategies (纯 RGB, 无深度, 单视角):
        1. 位移式: agent 移动时，在未探索方向生成 frontier
        2. 视觉式: 高 goal similarity 方向优先生成 frontier (if pano available)
        """
        if self._prev_position is None or self._goal_local_step <= 3:
            # First step: generate frontiers in cardinal directions
            self._generate_initial_frontiers(cur_vp)
            return

        displacement = np.linalg.norm(self._position - self._prev_position)
        if displacement < self._min_move_for_frontier:
            return  # Didn't move enough

        # Generate frontiers perpendicular to movement direction
        move_dir = self._position - self._prev_position
        move_dir_2d = np.array([move_dir[0], move_dir[2]])
        move_dist = np.linalg.norm(move_dir_2d)
        if move_dist < 1e-6:
            return

        move_dir_2d /= move_dist

        # Forward direction
        directions = [
            move_dir_2d,                                    # forward
            np.array([-move_dir_2d[1], move_dir_2d[0]]),  # left
            np.array([move_dir_2d[1], -move_dir_2d[0]]),  # right
        ]

        for d in directions:
            est_pos = np.array([
                self._position[0] + d[0] * self._frontier_step_size,
                self._position[1],  # keep y (height)
                self._position[2] + d[1] * self._frontier_step_size,
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
            )
            self.topo_map.add_edge(cur_vp, fid, EdgeType.NAVIGABLE)

    def _generate_initial_frontiers(self, cur_vp: str) -> None:
        """Generate frontiers in 4 cardinal directions at episode start."""
        heading_rad = self._heading
        for angle_offset in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            angle = heading_rad + angle_offset
            est_pos = np.array([
                self._position[0] - self._frontier_step_size * np.sin(angle),
                self._position[1],
                self._position[2] - self._frontier_step_size * np.cos(angle),
            ], dtype=np.float32)

            fid = self.topo_map.add_node(
                NodeType.WAYPOINT_FRONTIER,
                position=est_pos,
                confidence=0.2,
            )
            self.topo_map.add_edge(cur_vp, fid, EdgeType.NAVIGABLE)

    def _candidate_skip_reason(self, node: SemanticNode) -> Optional[str]:
        if node.node_id == self._cur_vp_id:
            return "current"
        if node.attributes.get("folded_detail") or node.attributes.get("folded"):
            if node.attributes.get("is_semantic_anchor") and self._is_goal_object_node(node):
                return None
            if float(node.attributes.get("target_relevance", 0.0)) > 0.0:
                return None
            return "folded_detail"
        if self._is_consumed_frontier(node):
            return "consumed"
        if node.node_type == NodeType.OBJECT and not self._object_search_candidate_allowed(node):
            return "object_not_current_goal_confirmed"
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
        return {
            "sticky_target_id": self._sticky_target_id,
            "sticky_used": False,
            "sticky_distance": self._sticky_last_distance,
            "sticky_no_progress_steps": self._sticky_no_progress_steps,
            "sticky_release_reason": self._sticky_release_reason,
            "consumed_frontiers": sorted(self._consumed_frontier_ids),
            "blocked_targets": self._active_blocked_targets(),
            "skipped_candidates": self._last_skipped_candidates,
            "navigation_event": self._last_navigation_event,
            "structure_target_id": getattr(self, "_last_structure_target_id", None),
        }

    # ------------------------------------------------------------------
    # Phase 3: two-stage room-centric planner
    # ------------------------------------------------------------------
    def _structure_node_has_semantic_match(self, node: SemanticNode, goal) -> bool:
        """Return True iff ``node`` has an explicit goal-relevant signal.

        Matches on (case-insensitive):
        * ROOM: room.label in goal.room_prior, OR any goal target part appears
          in room.attributes.contains_labels.
        * LANDMARK / portal: node.label appears in goal.landmarks, OR the
          goal target itself is a structural label (e.g. target == "door").
        """
        if goal is None:
            return False
        target_label = (getattr(goal, "target_object", "") or "").strip()
        target_parts = _split_compound_label(target_label) if target_label else set()
        room_prior = {
            str(r).strip().lower()
            for r in (getattr(goal, "room_prior", []) or [])
            if str(r).strip()
        }
        landmarks = {
            str(l).strip().lower()
            for l in (getattr(goal, "landmarks", []) or [])
            if str(l).strip()
        }
        node_label = (node.label or "").strip().lower()

        if node.node_type == NodeType.ROOM:
            if node_label and node_label in room_prior:
                return True
            contains_labels = {
                str(x).strip().lower()
                for x in node.attributes.get("contains_labels", [])
                if str(x).strip()
            }
            if contains_labels & target_parts:
                return True
            if contains_labels & landmarks:
                return True
            # Structural keywords occasionally present in summary text.
            summary_blob = node.attributes.get("semantic_summary", {}) or {}
            summary_labels = {
                str(x).strip().lower()
                for x in summary_blob.get("contains_labels", {}) or {}
            }
            if summary_labels & target_parts:
                return True
            return False

        if node.node_type == NodeType.LANDMARK:
            if node_label and node_label in landmarks:
                return True
            if node_label and node_label in target_parts:
                return True
            return False

        # Synthetic portals or other structural nodes.
        passage_type = (node.attributes.get("passage_type") or "").strip().lower()
        if passage_type and (passage_type in landmarks or passage_type in target_parts):
            return True
        return False

    def _select_structure_target(self) -> Optional[str]:
        """Stage 1: pick the most goal-relevant structure-layer node.

        Returns the node id of a ROOM (room_region) / structural LANDMARK /
        synthetic portal that scores highest under
        :func:`compute_semantic_bias`, provided the score clears
        ``planning.structure_target_score_threshold``.

        When ``planning.structure_target_require_semantic_match`` is set
        (default), candidates without an explicit semantic match against
        the current goal are filtered out before scoring. This prevents
        the planner from anchoring to an arbitrary nearby room when the
        memory has no goal-relevant structural information yet — in that
        case Stage 1 returns ``None`` and the planner runs single-stage.
        """
        if self.instruction_graph is None:
            return None
        struct_nodes = self.topo_map.structure_layer_nodes()
        if not struct_nodes:
            return None

        require_match = bool(getattr(
            self.config.planning,
            "structure_target_require_semantic_match",
            False,
        ))
        current_goal = self.instruction_graph.get_current_goal()
        if require_match:
            if current_goal is None:
                return None
            struct_nodes = [
                n for n in struct_nodes
                if self._structure_node_has_semantic_match(n, current_goal)
            ]
            if not struct_nodes:
                return None

        ids = [node.node_id for node in struct_nodes]
        scores = compute_semantic_bias(
            goal_graph=self.instruction_graph,
            topo_map=self.topo_map,
            candidate_node_ids=ids,
            agent_position=self._position,
            normalize=False,
            current_node_id=self._cur_vp_id,
        )
        if scores.size == 0:
            return None
        # Distance-gradient bonus: gives a slight nudge toward the nearest
        # structure node so exploration stays anchored even before room
        # summaries accumulate contains_labels. Only applied when we have
        # already filtered to semantically-matching candidates, otherwise
        # the bonus would drag the planner toward arbitrary nearby rooms.
        if self._position is not None and require_match:
            max_dist = max(
                float(np.linalg.norm(n.position - self._position))
                for n in struct_nodes
            ) or 1.0
            for i, node in enumerate(struct_nodes):
                d = float(np.linalg.norm(node.position - self._position))
                scores[i] = float(scores[i]) + 0.1 * (1.0 - d / max_dist)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < self.config.planning.structure_target_score_threshold:
            return None
        return ids[best_idx]

    def _is_stuck(self, window: int = 10, threshold: float = 0.05, cooldown: int = 25) -> bool:
        """Return True if the agent has barely moved over *window* steps.

        Rotation-only steps (recovery scans) do not count as progress, so a
        cooldown prevents re-entering recovery every step while turning.
        """
        if self._last_stuck_recovery_step is not None:
            if self.topo_map.current_step - self._last_stuck_recovery_step < cooldown:
                return False
        if len(self._recent_positions) < window:
            return False
        recent = self._recent_positions[-window:]
        total_displacement = float(np.linalg.norm(recent[-1] - recent[0]))
        return total_displacement < threshold

    def _is_goal_object_node(self, node: SemanticNode) -> bool:
        if self.instruction_graph is None:
            return False
        goal = self.instruction_graph.get_current_goal()
        target_label = getattr(goal, "target_object", None) if goal is not None else None
        if target_label is None:
            return False
        target_labels = _split_compound_label(target_label)
        return node.node_type == NodeType.OBJECT and bool(node.label and _label_matches_goal(node.label, target_labels))

    def _object_direct_nav_allowed(self, node: SemanticNode) -> bool:
        if node.node_type != NodeType.OBJECT:
            return True
        if not self._is_goal_object_node(node):
            return False
        if float(node.attributes.get("target_relevance", 0.0)) <= 0.0:
            return False
        if node.attributes.get("repeated_goal_source"):
            return True
        detected_step = node.attributes.get("goal_detection_step")
        if detected_step is None:
            return False
        age = self.topo_map.current_step - int(detected_step)
        if age > 12:
            return False
        if float(node.attributes.get("goal_detection_confidence", 0.0)) < 0.30:
            return False
        return True

    def _has_near_goal_object_node(self, radius: float = 2.0) -> bool:
        """Check if any goal-matching OBJECT node is within *radius* of the agent."""
        if self._position is None or self.instruction_graph is None:
            return False
        goal = self.instruction_graph.get_current_goal()
        target_label = getattr(goal, "target_object", None) if goal is not None else None
        if target_label is None:
            return False
        target_labels = _split_compound_label(target_label)
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.label and _label_matches_goal(node.label, target_labels):
                if self._anchor_dist(node) <= radius:
                    return True
        return False

    def _candidate_anchored_skip_reason(
        self,
        node: SemanticNode,
        structure_target_id: Optional[str],
    ) -> Optional[str]:
        """Stage 2 filter: drop far semantic objects when anchored."""
        if structure_target_id is None:
            return None
        if not self.config.planning.far_object_skip_when_anchored:
            return None
        if node.node_type != NodeType.OBJECT:
            return None
        if self._is_goal_object_node(node):
            return None
        if float(node.attributes.get("target_relevance", 0.0)) > 0.0:
            return None
        anchor = self.topo_map.get_node(structure_target_id)
        if anchor is None:
            return None
        dist = float(np.linalg.norm(node.position - anchor.position))
        if dist > self.config.planning.structure_anchor_radius:
            return "far_semantic_object_outside_anchor"
        return None

    def set_reachability_probe(self, probe_fn) -> None:
        """Inject a navmesh-aware reachability callable used by ``plan()``.

        ``probe_fn(target_position_world, target_node_id)`` should return a
        mapping with at least ``{"reachable": bool, "geodesic_distance":
        float}``. Pass ``None`` to disable the probe and fall back to the
        topo-graph estimate.
        """
        self._reachability_probe = probe_fn

    def _compute_reachability_components(
        self, candidates: List[SemanticNode]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (reachable_score, normalized_path_cost) per candidate.

        ``reachable_score`` is 1.0 when the candidate is reachable from
        the current waypoint (via the injected navmesh probe or — when
        absent — via the topo NAVIGABLE shortest path), else 0.0.

        ``normalized_path_cost`` is the candidate's path length divided
        by the largest finite path length in the batch (so it lies in
        ``[0, 1]``). Unreachable candidates take the cap value 1.0.
        """
        n = len(candidates)
        reach = np.zeros(n, dtype=np.float32)
        cost = np.ones(n, dtype=np.float32)
        if n == 0 or self._cur_vp_id is None or self._position is None:
            return reach, cost

        probe_fn = getattr(self, "_reachability_probe", None)
        cur_node = self.topo_map.get_node(self._cur_vp_id)
        cur_pos = cur_node.position if cur_node is not None else self._position

        raw: List[Tuple[bool, float]] = []
        for node in candidates:
            target_pos, _ = self._target_output_for_node(node)
            target_pos = np.asarray(target_pos, dtype=np.float32)

            if probe_fn is not None:
                try:
                    result = probe_fn(target_pos, node.node_id) or {}
                    reachable = bool(result.get("reachable", False))
                    geo = float(result.get("geodesic_distance", float("inf")))
                    if not reachable or not np.isfinite(geo):
                        raw.append((False, float("inf")))
                    else:
                        raw.append((True, geo))
                    continue
                except Exception:
                    # Probe failed — fall through to graph-based estimate.
                    pass

            # Graph-based fallback: anchor the candidate to its nearest
            # visited waypoint, then sum NAVIGABLE distances along the
            # shortest path from cur_vp.
            if node.node_type == NodeType.WAYPOINT_VISITED:
                anchor_id = node.node_id
                anchor_pos = node.position
            else:
                anchor_node = self.topo_map.find_nearest_node(
                    target_pos, NodeType.WAYPOINT_VISITED
                )
                if anchor_node is None:
                    raw.append((False, float("inf")))
                    continue
                anchor_id = anchor_node.node_id
                anchor_pos = anchor_node.position

            if anchor_id == self._cur_vp_id:
                # No graph hop required; use Euclidean residual.
                residual = float(np.linalg.norm(target_pos - cur_pos))
                raw.append((True, residual))
                continue

            path = self.topo_map.shortest_path(
                self._cur_vp_id, anchor_id, EdgeType.NAVIGABLE
            )
            if not path:
                raw.append((False, float("inf")))
                continue
            total = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edge_data = self.topo_map.graph.get_edge_data(u, v) or {}
                dist_m = edge_data.get("distance_m")
                if dist_m is None or not np.isfinite(float(dist_m)):
                    nu = self.topo_map.get_node(u)
                    nv = self.topo_map.get_node(v)
                    if nu is None or nv is None:
                        total = float("inf")
                        break
                    total += float(np.linalg.norm(nu.position - nv.position))
                else:
                    total += float(dist_m)
            if not np.isfinite(total):
                raw.append((False, float("inf")))
                continue
            # Add residual hop from the anchor waypoint to the target.
            total += float(np.linalg.norm(target_pos - anchor_pos))
            raw.append((True, total))

        finite_costs = [c for r, c in raw if r and np.isfinite(c)]
        norm_max = max(finite_costs) if finite_costs else 1.0
        if norm_max <= 1e-6:
            norm_max = 1.0
        for i, (r, c) in enumerate(raw):
            if r and np.isfinite(c):
                reach[i] = 1.0
                cost[i] = float(min(c / norm_max, 1.0))
            else:
                reach[i] = 0.0
                cost[i] = 1.0
        return reach, cost

    def _apply_reachability_components(
        self, candidates: List[SemanticNode], scores: np.ndarray
    ) -> np.ndarray:
        cfg = self.config.planning
        w_reach = float(getattr(cfg, "reachability_score_weight", 0.0))
        w_cost = float(getattr(cfg, "path_cost_score_weight", 0.0))
        if scores.size == 0 or (w_reach == 0.0 and w_cost == 0.0):
            return scores
        reach, cost = self._compute_reachability_components(candidates)
        adjusted = scores.astype(np.float32).copy()
        adjusted = adjusted + w_reach * reach - w_cost * cost
        return adjusted

    def _apply_structure_anchor_bonus(
        self,
        candidates: List[SemanticNode],
        scores: np.ndarray,
        structure_target_id: Optional[str],
    ) -> np.ndarray:
        """Boost candidates anchored to (or inside) the structure target."""
        if structure_target_id is None or scores.size == 0:
            return scores
        anchor = self.topo_map.get_node(structure_target_id)
        if anchor is None:
            return scores
        bonus = float(self.config.planning.structure_anchor_bonus)
        radius = float(self.config.planning.structure_anchor_radius)
        if bonus <= 0 or radius <= 0:
            return scores
        boosted = scores.astype(np.float32).copy()
        for idx, node in enumerate(candidates):
            if node.node_id == structure_target_id:
                boosted[idx] += bonus
                continue
            if node.node_type not in (
                NodeType.WAYPOINT_FRONTIER,
                NodeType.WAYPOINT_CANDIDATE,
                NodeType.WAYPOINT_VISITED,
                NodeType.OBJECT,
            ):
                continue
            if node.node_type == NodeType.OBJECT:
                obj_room_id = node.attributes.get("anchor_room_id") or node.attributes.get("fused_into_summary_id")
                if obj_room_id == structure_target_id:
                    boosted[idx] += bonus
                continue
            room_id = node.attributes.get("room_id")
            if room_id == structure_target_id:
                boosted[idx] += bonus
                continue
            dist = float(np.linalg.norm(node.position - anchor.position))
            if dist <= radius:
                # Linear falloff from full bonus at 0 to 0 at radius.
                boosted[idx] += bonus * max(0.0, 1.0 - dist / radius)
        return boosted

    def _in_goal_search_mode(self) -> bool:
        """True when we have no goal object anchor and must search by context."""
        return self._nearest_goal_anchor_dist() == float("inf")

    def _goal_context_room_priors(self) -> set:
        if self.instruction_graph is None:
            return set()
        goal = self.instruction_graph.get_current_goal()
        if goal is None:
            return set()
        return {
            str(r).strip().lower()
            for r in (getattr(goal, "room_prior", []) or [])
            if str(r).strip()
        }

    def _goal_context_landmark_priors(self) -> set:
        if self.instruction_graph is None:
            return set()
        goal = self.instruction_graph.get_current_goal()
        if goal is None:
            return set()
        return {
            str(l).strip().lower()
            for l in (getattr(goal, "landmarks", []) or [])
            if str(l).strip()
        }

    def _label_in_priors(self, label: Optional[str], priors: set) -> bool:
        if not label or not priors:
            return False
        text = str(label).strip().lower()
        if text in priors:
            return True
        return any(p in text or text in p for p in priors)

    def _is_context_landmark_node(self, node: SemanticNode) -> bool:
        if node.node_type != NodeType.LANDMARK:
            return False
        return self._label_in_priors(node.label, self._goal_context_landmark_priors())

    def _is_context_object_node(self, node: SemanticNode) -> bool:
        if node.node_type != NodeType.OBJECT:
            return False
        if self._is_goal_object_node(node):
            return False
        return self._label_in_priors(node.label, self._goal_context_landmark_priors())

    def _object_search_candidate_allowed(self, node: SemanticNode) -> bool:
        if self._object_direct_nav_allowed(node):
            return True
        if not self._in_goal_search_mode():
            return False
        return self._is_context_object_node(node)

    def _waypoint_view_context_labels(self, node: SemanticNode) -> set:
        labels: set = set()
        view_room = node.attributes.get("view_room_label")
        if view_room and str(view_room).strip().lower() not in ("", "unknown"):
            labels.add(str(view_room).strip().lower())
        for lab in node.attributes.get("view_object_labels", []) or []:
            if lab:
                labels.add(str(lab).strip().lower())
        for entry in node.attributes.get("scene_vocabulary", []) or []:
            if isinstance(entry, dict) and entry.get("label"):
                labels.add(str(entry["label"]).strip().lower())
        room_scores = node.attributes.get("room_scores") or {}
        if isinstance(room_scores, dict):
            for room_name, score in room_scores.items():
                if float(score) >= 0.25:
                    labels.add(str(room_name).strip().lower())
        return labels

    def _nearest_context_landmark_dist(self, position: np.ndarray, landmark_priors: set) -> float:
        best = float("inf")
        for node in self.topo_map.get_nodes_by_type(NodeType.LANDMARK):
            if not self._label_in_priors(node.label, landmark_priors):
                continue
            d = float(np.linalg.norm(node.position - position))
            if d < best:
                best = d
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            if not self._label_in_priors(node.label, landmark_priors):
                continue
            d = float(np.linalg.norm(node.position - position))
            if d < best:
                best = d
        return best

    def _apply_goal_driven_search_boost(
        self,
        candidates: List[SemanticNode],
        scores: np.ndarray,
    ) -> np.ndarray:
        """Context-driven search when no goal object anchor exists.

        Scores candidates by room prior, landmark/object context, view memory,
        and unexplored frontiers — not only direct goal object detections.
        """
        if scores.size == 0 or not self._in_goal_search_mode():
            return scores
        room_priors = self._goal_context_room_priors()
        landmark_priors = self._goal_context_landmark_priors()
        if not room_priors and not landmark_priors:
            return scores

        boosted = scores.astype(np.float32).copy()
        for i, node in enumerate(candidates):
            if node.node_type == NodeType.ROOM:
                node_label = (node.label or "").strip().lower()
                if node_label and node_label in room_priors:
                    boosted[i] += 1.5
                contains = {
                    str(c).strip().lower()
                    for c in node.attributes.get("contains_labels", [])
                    if str(c).strip()
                }
                if contains & landmark_priors:
                    boosted[i] += 0.9
                if contains & room_priors:
                    boosted[i] += 0.4

            if node.node_type == NodeType.LANDMARK and self._label_in_priors(node.label, landmark_priors):
                boosted[i] += 1.4

            if node.node_type == NodeType.OBJECT and self._is_context_object_node(node):
                boosted[i] += 1.0

            context_labels = self._waypoint_view_context_labels(node)
            if context_labels & room_priors:
                boosted[i] += 0.8
            if context_labels & landmark_priors:
                boosted[i] += 0.6

            room_id = node.attributes.get("room_id")
            if room_id:
                room_node = self.topo_map.get_node(room_id)
                if room_node and room_node.label and room_node.label.lower() in room_priors:
                    boosted[i] += 0.7
                if room_id in self._explored_rooms_no_target:
                    boosted[i] -= 0.8
            elif node.node_type == NodeType.ROOM and node.node_id in self._explored_rooms_no_target:
                boosted[i] -= 0.8

            view_room = node.attributes.get("view_room_label", "")
            if view_room and str(view_room).lower() in room_priors:
                boosted[i] += 0.5

            if node.node_type == NodeType.WAYPOINT_FRONTIER:
                boosted[i] += 0.35
                near_ctx = self._nearest_context_landmark_dist(node.position, landmark_priors)
                if near_ctx < 4.0:
                    boosted[i] += 0.8 * max(0.0, 1.0 - near_ctx / 4.0)
                near_prior_room = False
                if view_room and str(view_room).lower() in room_priors:
                    near_prior_room = True
                if room_id:
                    room_node = self.topo_map.get_node(room_id)
                    if room_node and room_node.label and room_node.label.lower() in room_priors:
                        near_prior_room = True
                if near_prior_room:
                    boosted[i] += 0.5

        return boosted

    def _apply_goal_latch_boost(
        self,
        candidates: List[SemanticNode],
        scores: np.ndarray,
    ) -> np.ndarray:
        """When goal is latched, heavily boost goal-object nodes to keep
        the agent focused on the target instead of wandering off."""
        if not self._goal_latched or scores.size == 0:
            return scores
        boosted = scores.astype(np.float32).copy()
        for i, node in enumerate(candidates):
            if node.node_type == NodeType.OBJECT and self._is_goal_object_node(node):
                boosted[i] += 5.0
            elif node.node_type == NodeType.WAYPOINT_FRONTIER:
                anchor_dist = self._nearest_goal_anchor_dist()
                if anchor_dist != float("inf") and self._position is not None:
                    node_dist = float(np.linalg.norm(node.position - self._position))
                    if node_dist > anchor_dist + 2.0:
                        boosted[i] -= 2.0
        return boosted

    def _current_room_id(self) -> Optional[str]:
        """Return the room node id the agent is currently inside, if any."""
        if self._position is None:
            return None
        summary = self.topo_map.find_nearby_room_summary(
            self._position, radius=3.0,
        )
        return summary.node_id if summary is not None else None

    def _plan_regrounding(self) -> Optional[Dict[str, Any]]:
        """Simplified MVP regrounding: scan 3x then back to idle."""
        if self._reground_state == "scanning":
            return {
                "target_node_id": None,
                "target_position": None,
                "is_exploration": False,
                "mode": "local_reground_scan",
                "reason": "reached_folded_anchor",
                "candidate_ids": [],
                "scores": [],
                "sticky_debug": self._last_plan_debug,
            }
        return None

    def _resolve_navigable_target(self, node: SemanticNode) -> Optional[str]:
        if node.node_type in (NodeType.WAYPOINT_VISITED, NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE):
            return node.node_id
        if node.node_type in (NodeType.OBJECT, NodeType.LANDMARK):
            anchor_wp = node.attributes.get("anchor_waypoint_id")
            if anchor_wp:
                return anchor_wp
        nearest_wp = self.topo_map.find_nearest_node(node.position, NodeType.WAYPOINT_VISITED)
        return nearest_wp.node_id if nearest_wp else None

    def plan(self) -> Dict[str, Any]:
        """4-stage MVP plan dispatch.

        EXPLORE → NAV_TO_ANCHOR → APPROACH_CONFIRM → STOP
        """
        if self.instruction_graph is None or self._cur_vp_id is None:
            return {"plan_action": "no_target", "target_node_id": None, "target_position": None, "is_exploration": True}

        self._last_navigation_event = {}

        if self._nav_phase == "approach_confirm":
            return self._plan_approach_confirm()

        self._check_goal_no_progress()
        self._check_search_no_progress()

        self._update_nav_phase()

        if self._nav_phase == "approach_confirm":
            return self._plan_approach_confirm()

        if self._is_stuck(window=10, threshold=0.08, cooldown=25):
            self._last_stuck_recovery_step = self.topo_map.current_step
            self._stuck_recovery_count += 1
            self._recent_positions = self._recent_positions[-1:]
            if self._sticky_target_id:
                self._block_target(self._sticky_target_id, "stuck_no_progress", ttl=30)
                self._clear_sticky("stuck_recovery")
            recovery_turn = "turn_left" if self._stuck_recovery_count % 2 == 0 else "turn_right"
            return {
                "plan_action": "recover",
                "action": recovery_turn,
                "target_node_id": None,
                "target_position": None,
                "is_exploration": True,
                "mode": "stuck_recovery",
            }

        return self._plan_navigate()

    def _plan_approach_confirm(self) -> Dict[str, Any]:
        """Local visual servo: align, advance, hold, or stop based on VLM.

        Sub-states:
          SERVO_ALIGN   -- target off-center, turn toward bearing
          SERVO_ADVANCE -- target centered, bbox growing or stable, move forward
          SERVO_HOLD    -- bbox shrinking, hold position, evaluate stop
          SERVO_LOST    -- target not visible for 3+ steps, replan
        """
        vlm = self._current_goal_vlm_confirmation()
        goal_visible = bool(vlm.get("goal_visible")) if vlm.get("available") else False
        anchor_dist = self._nearest_goal_anchor_dist()
        if anchor_dist != float("inf"):
            self._approach_best_anchor_dist = min(self._approach_best_anchor_dist, anchor_dist)

        if vlm.get("available"):
            self._try_refresh_goal_anchor_from_vlm(vlm)
            bbox_area = float(vlm.get("bbox_area", 0))
            range_near = vlm.get("range_bin") == "near"
            if range_near and (vlm.get("stop_grounding_ok") or vlm.get("grounding_ok")):
                if bbox_area > self._approach_best_bbox_area:
                    self._approach_best_bbox_area = bbox_area
                    self._approach_peak_travel = self._goal_travel_distance
                    if self._position is not None:
                        self._approach_peak_position = self._position.copy()

            confirm_success = bool(
                vlm.get("goal_visible")
                and range_near
                and (vlm.get("stop_grounding_ok") or vlm.get("grounding_ok"))
                and float(vlm.get("confidence", 0)) >= 0.5
            )

            if self._heavy_confirm_eval_step != self.topo_map.current_step:
                self._confirm_buffer.append(confirm_success)
                if len(self._confirm_buffer) > 5:
                    self._confirm_buffer = self._confirm_buffer[-5:]
                self._heavy_confirm_eval_step = self.topo_map.current_step

            if bbox_area > 0 and self._servo_prev_bbox_area > 0:
                if bbox_area >= self._servo_prev_bbox_area * 0.98:
                    self._servo_visual_advance_count += 1
                    self._servo_bbox_shrink_count = 0
                elif bbox_area < self._servo_prev_bbox_area * 0.95:
                    self._servo_bbox_shrink_count += 1
                else:
                    self._servo_bbox_shrink_count = 0
            if bbox_area > 0:
                self._servo_prev_bbox_area = bbox_area

            if goal_visible:
                self._approach_lost_count = 0
                self._approach_bearing = vlm.get("bearing", self._approach_bearing)
            else:
                self._approach_lost_count += 1
        else:
            self._approach_lost_count += 1

        self._update_confirm_window(vlm)

        if self._stop_verifier_passed(vlm):
            self._last_stop_debug = {
                "stop_allowed": True,
                "reason": "mvp_stop_verifier",
                "nav_phase": self._nav_phase,
                "vlm_confirm": vlm,
                "confirm_buffer": list(self._confirm_buffer[-3:]),
                "approach_remaining": self._approach_remaining,
                "approach_steps_used": self._approach_steps_used,
                "advance_steps": self._approach_advance_steps,
                "approach_best_bbox_area": round(self._approach_best_bbox_area, 4),
                "anchor_dist": round(anchor_dist, 3) if anchor_dist != float("inf") else None,
                "approach_best_anchor_dist": round(self._approach_best_anchor_dist, 3)
                if self._approach_best_anchor_dist != float("inf") else None,
                "entry_anchor_dist": round(self._approach_entry_anchor_dist, 3)
                if self._approach_entry_anchor_dist != float("inf") else None,
                "goal_local_step": self._goal_local_step,
                "goal_travel_distance": round(self._goal_travel_distance, 3),
            }
            return {"plan_action": "stop", "target_node_id": None, "target_position": None, "is_exploration": False}

        # --- Passed peak: bbox shrinking for 2+ frames → return to peak position ---
        if (self._servo_bbox_shrink_count >= 2
                and self._approach_peak_position is not None
                and self._servo_visual_advance_count >= 2
                and sum(self._confirm_buffer[-3:]) >= 2
                and self._approach_steps_used >= self._APPROACH_MIN_STEPS_BEFORE_STOP):
            if not self._servo_returning_to_peak:
                self._servo_returning_to_peak = True
            peak_pos = self._approach_peak_position
            if self._position is not None:
                dist_to_peak = float(np.linalg.norm(self._position - peak_pos))
            else:
                dist_to_peak = 999.0
            if dist_to_peak < 0.3:
                self._last_stop_debug = {
                    "stop_allowed": True,
                    "reason": "peak_return_stop",
                    "nav_phase": self._nav_phase,
                    "dist_to_peak": round(dist_to_peak, 3),
                    "confirm_buffer": list(self._confirm_buffer[-3:]),
                    "approach_best_bbox_area": round(self._approach_best_bbox_area, 4),
                    "approach_steps_used": self._approach_steps_used,
                    "anchor_dist": round(anchor_dist, 3) if anchor_dist != float("inf") else None,
                }
                return {"plan_action": "stop", "target_node_id": None, "target_position": None, "is_exploration": False}
            self._approach_remaining -= 1
            self._approach_steps_used += 1
            self._last_stop_debug = {
                "stop_allowed": False,
                "reason": "peak_return_nav",
                "nav_phase": self._nav_phase,
                "dist_to_peak": round(dist_to_peak, 3),
                "approach_remaining": self._approach_remaining,
                "bbox_shrink_count": self._servo_bbox_shrink_count,
            }
            return {
                "plan_action": "approach_confirm",
                "action": "navigate_to",
                "target_node_id": None,
                "target_position": peak_pos.tolist(),
                "is_exploration": False,
                "mode": "peak_return",
                "approach_remaining": self._approach_remaining,
            }

        if self._approach_lost_count >= 3:
            self._nav_phase = "explore"
            self._last_stop_debug = {
                "stop_allowed": False,
                "reason": "servo_lost_target",
                "nav_phase": "explore",
                "approach_lost_count": self._approach_lost_count,
                "goal_local_step": self._goal_local_step,
            }
            return self._plan_navigate()

        if self._approach_remaining <= 0:
            if self._approach_budget_stop_eligible(vlm):
                self._last_stop_debug = {
                    "stop_allowed": True,
                    "reason": "approach_budget_stop_with_confirm",
                    "nav_phase": self._nav_phase,
                    "vlm_confirm": vlm,
                    "confirm_buffer": list(self._confirm_buffer[-3:]),
                    "stop_buffer": list(self._stop_buffer[-3:]),
                    "approach_best_bbox_area": round(self._approach_best_bbox_area, 4),
                    "approach_peak_travel": round(self._approach_peak_travel, 3),
                    "goal_travel_distance": round(self._goal_travel_distance, 3),
                }
                return {"plan_action": "stop", "target_node_id": None, "target_position": None, "is_exploration": False}
            self._nav_phase = "explore"
            self._last_stop_debug = {
                "stop_allowed": False,
                "reason": "approach_budget_exhausted",
                "nav_phase": "explore",
                "confirm_buffer": list(self._confirm_buffer[-3:]),
                "goal_local_step": self._goal_local_step,
            }
            return self._plan_navigate()

        self._approach_remaining -= 1
        self._approach_steps_used += 1

        # --- Align: at most 2 consecutive turns, then force advance ---
        bearing = self._approach_bearing
        align_needed = goal_visible and bearing not in ("center", "front", "front-center", "unknown")
        if align_needed and self._approach_align_count < self._SERVO_ALIGN_BURST:
            self._approach_align_count += 1
            if bearing in ("left", "left_front"):
                action = "turn_left"
            else:
                action = "turn_right"
            self._last_stop_debug = {
                "stop_allowed": False,
                "reason": "servo_align",
                "nav_phase": self._nav_phase,
                "bearing": bearing,
                "approach_remaining": self._approach_remaining,
                "align_count": self._approach_align_count,
            }
            return {
                "plan_action": "approach_confirm",
                "action": action,
                "target_node_id": None,
                "target_position": None,
                "is_exploration": False,
                "mode": "servo_align",
                "approach_remaining": self._approach_remaining,
            }
        self._approach_align_count = 0

        # --- Default action: advance (move_forward) ---
        self._approach_advance_steps += 1
        self._last_stop_debug = {
            "stop_allowed": False,
            "reason": "servo_advance",
            "nav_phase": self._nav_phase,
            "approach_remaining": self._approach_remaining,
            "goal_visible": goal_visible,
            "anchor_dist": round(anchor_dist, 3) if anchor_dist != float("inf") else None,
            "confirm_buffer": list(self._confirm_buffer[-3:]),
            "stop_buffer": list(self._stop_buffer[-3:]),
            "advance_steps": self._approach_advance_steps,
        }
        return {
            "plan_action": "approach_confirm",
            "action": "move_forward",
            "target_node_id": None,
            "target_position": None,
            "is_exploration": False,
            "mode": "servo_advance",
            "approach_remaining": self._approach_remaining,
        }

    def _approach_budget_stop_eligible(self, vlm: Dict[str, Any]) -> bool:
        """Allow STOP when approach budget ends at the best visual pose seen."""
        if self._approach_steps_used < self._APPROACH_MIN_STEPS_BEFORE_STOP:
            return False
        if self._approach_advance_steps < self._MIN_ADVANCE_BEFORE_STOP:
            return False
        if sum(self._confirm_buffer[-3:]) < 2:
            return False
        if self._servo_visual_advance_count < 2:
            return False
        if self._approach_best_bbox_area <= 0:
            return False
        if vlm.get("available"):
            bbox_area = float(vlm.get("bbox_area", 0))
            if bbox_area < self._approach_best_bbox_area * self._STOP_BBOX_PEAK_RATIO_BUDGET:
                return False
        return True

    def _is_vlm_recent(self, max_age: int = 2) -> bool:
        """True if the last VLM run was within *max_age* steps of now."""
        if self._last_heavy_step is None:
            return False
        return self.topo_map.current_step - self._last_heavy_step <= max_age

    def _has_approach_progress(self) -> bool:
        """True when the servo phase has shown real physical approach progress."""
        if self._approach_advance_steps < self._MIN_ADVANCE_BEFORE_STOP:
            return False
        if self._approach_entry_anchor_dist != float("inf") and self._approach_best_anchor_dist != float("inf"):
            if self._approach_best_anchor_dist >= self._approach_entry_anchor_dist - 0.1:
                if self._servo_visual_advance_count < 3:
                    return False
        return True

    def _stop_verifier_ready(self, vlm: Dict[str, Any]) -> bool:
        """Visual confirm readiness with progress guard."""
        if not vlm.get("available"):
            return False
        if not (vlm.get("fresh") or self._is_vlm_recent(2)):
            return False
        if not self._confirm_window_active:
            return False
        if self._approach_steps_used < self._APPROACH_MIN_STEPS_BEFORE_STOP:
            return False
        if self._approach_best_bbox_area < self._BBOX_MIN_AREA_FOR_APPROACH:
            return False
        if not self._has_approach_progress():
            return False
        return sum(self._confirm_buffer[-3:]) >= 2

    def _stop_verifier_passed(self, vlm: Dict[str, Any]) -> bool:
        """Pure visual STOP: bbox grew, near peak, then plateaued."""
        if not self._stop_verifier_ready(vlm):
            return False
        if self._servo_visual_advance_count < 2:
            return False
        bbox_area = float(vlm.get("bbox_area", 0))
        if self._approach_best_bbox_area > 0:
            if bbox_area < self._approach_best_bbox_area * self._STOP_BBOX_PEAK_RATIO_NEAR:
                return False
        return self._servo_bbox_shrink_count >= 1

    def _plan_navigate(self) -> Dict[str, Any]:
        """Standard candidate-based navigation (explore / nav_to_anchor)."""
        reground_plan = self._plan_regrounding()
        if reground_plan is not None:
            return reground_plan

        structure_target_id: Optional[str] = None
        if self.config.planning.two_stage_enabled:
            structure_target_id = self._select_structure_target()
        self._last_structure_target_id = structure_target_id

        primary_candidates = []
        primary_ids = []
        fallback_candidates = []
        fallback_ids = []
        self._last_skipped_candidates = []
        in_search_mode = self._in_goal_search_mode()

        for node in self.topo_map._nodes.values():
            reason = self._candidate_skip_reason(node)
            if reason is None:
                reason = self._candidate_anchored_skip_reason(node, structure_target_id)
            if reason is not None:
                self._last_skipped_candidates.append({"node_id": node.node_id, "type": node.node_type.value, "reason": reason})
                continue
            is_room_summary = (node.node_type == NodeType.ROOM
                               and node.attributes.get("summary_type") == "room_region")
            is_search_landmark = in_search_mode and self._is_context_landmark_node(node)
            is_search_context_object = in_search_mode and self._is_context_object_node(node)
            if (node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE, NodeType.OBJECT)
                    or is_room_summary or is_search_landmark or is_search_context_object):
                primary_candidates.append(node)
                primary_ids.append(node.node_id)
            elif node.node_type == NodeType.WAYPOINT_VISITED:
                fallback_candidates.append(node)
                fallback_ids.append(node.node_id)

        candidates = primary_candidates if primary_candidates else fallback_candidates
        candidate_ids = primary_ids if primary_candidates else fallback_ids

        if not candidates:
            self._last_plan_debug = self._debug_plan_state()
            self._last_stop_debug = {
                "stop_allowed": False, "reason": "no_candidates",
                "nav_phase": self._nav_phase, "goal_local_step": self._goal_local_step,
            }
            return {
                "plan_action": "no_target",
                "target_node_id": None,
                "target_position": None,
                "is_exploration": True,
                "candidate_ids": [],
                "scores": [],
                "sticky_debug": self._last_plan_debug,
                "structure_target_id": structure_target_id,
            }

        scores = compute_semantic_bias(
            goal_graph=self.instruction_graph,
            topo_map=self.topo_map,
            candidate_node_ids=candidate_ids,
            agent_position=self._position,
            normalize=False,
            current_node_id=self._cur_vp_id,
            explored_rooms_no_target=self._explored_rooms_no_target,
        )
        scores = self._apply_structure_anchor_bonus(candidates, scores, structure_target_id)
        scores = self._apply_reachability_components(candidates, scores)
        scores = self._apply_goal_driven_search_boost(candidates, scores)
        scores = self._apply_goal_latch_boost(candidates, scores)

        sticky_plan = self._sticky_plan_if_valid(candidates, candidate_ids, scores)
        if sticky_plan is not None:
            sticky_plan["structure_target_id"] = structure_target_id
            sticky_plan.setdefault("plan_action", "navigate")
            self._last_stop_debug = {
                "stop_allowed": False, "reason": "navigating",
                "nav_phase": self._nav_phase, "goal_local_step": self._goal_local_step,
            }
            return sticky_plan

        best_idx = int(np.argmax(scores))
        best_node = candidates[best_idx]
        self._sticky_target_id = best_node.node_id if self.config.planning.sticky_target_enabled else None
        self._sticky_last_distance = float(np.linalg.norm(best_node.position - self._position))
        self._sticky_last_heading = self._heading
        self._sticky_no_progress_steps = 0
        self._last_plan_debug = {
            "sticky_target_id": self._sticky_target_id,
            "sticky_used": False,
            "sticky_distance": self._sticky_last_distance,
            "sticky_no_progress_steps": 0,
            "sticky_release_reason": self._sticky_release_reason,
            "consumed_frontiers": sorted(self._consumed_frontier_ids),
            "blocked_targets": self._active_blocked_targets(),
            "skipped_candidates": self._last_skipped_candidates,
            "navigation_event": self._last_navigation_event,
            "structure_target_id": structure_target_id,
        }
        self._sticky_release_reason = ""
        target_pos, target_extras = self._target_output_for_node(best_node)

        self._last_stop_debug = {
            "stop_allowed": False, "reason": "navigating",
            "nav_phase": self._nav_phase, "goal_local_step": self._goal_local_step,
        }
        plan_output = {
            "plan_action": "navigate",
            "target_node_id": best_node.node_id,
            "target_position": target_pos,
            "is_exploration": best_node.node_type == NodeType.WAYPOINT_FRONTIER,
            "scores": scores,
            "candidate_ids": candidate_ids,
            "sticky_debug": self._last_plan_debug,
            "structure_target_id": structure_target_id,
        }
        plan_output.update(target_extras)
        return plan_output

    def act(self, plan_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert plan output to action for PointNav controller.

        Returns:
            dict with 'target_position' for PointNav, or 'stop' if goal reached.
        """
        debug_fields = {
            "mode": plan_output.get("mode"),
            "plan_action": plan_output.get("plan_action"),
            "requires_regrounding": plan_output.get("requires_regrounding", False),
            "semantic_target_node_id": plan_output.get("semantic_target_node_id"),
            "target_anchor_type": plan_output.get("target_anchor_type"),
            "anchor_waypoint_id": plan_output.get("anchor_waypoint_id"),
            "anchor_room_id": plan_output.get("anchor_room_id"),
            "vlm_range_bin": plan_output.get("vlm_range_bin"),
            "vlm_bearing": plan_output.get("vlm_bearing"),
            "vlm_visibility": plan_output.get("vlm_visibility"),
            "global_path": plan_output.get("global_path"),
            "reground_state": self._reground_state,
            "reground_target_node_id": self._reground_target_node_id,
            "reground_anchor_node_id": self._reground_anchor_node_id,
            "target_object_detected_this_scan": self._target_object_detected_this_scan,
            "stop_debug": self._last_stop_debug,
        }
        common = {
            "target_node_id": plan_output.get("target_node_id"),
            "candidate_ids": plan_output.get("candidate_ids", []),
            "scores": plan_output.get("scores", []),
            "is_exploration": plan_output.get("is_exploration", True),
            "sticky_debug": plan_output.get("sticky_debug", self._last_plan_debug),
        }

        if plan_output.get("mode") == "local_reground_scan":
            action = self._reground_scan_action()
            action.update({k: v for k, v in debug_fields.items() if v is not None})
            action.update(common)
            action["reground_state"] = self._reground_state
            return action

        if self._try_start_regrounding(plan_output):
            action = self._reground_scan_action()
            action.update({k: v for k, v in debug_fields.items() if v is not None})
            action.update(common)
            action["mode"] = "local_reground_scan"
            action["reground_state"] = self._reground_state
            action["reground_anchor_node_id"] = self._reground_anchor_node_id
            return action

        direct = plan_output.get("action")
        if direct in ("turn_left", "turn_right", "move_forward"):
            action = {"action": direct, **common}
            action.update({k: v for k, v in debug_fields.items() if v is not None})
            return action

        plan_action = plan_output.get("plan_action")
        if plan_action == "stop":
            action = {"action": "stop", **common}
            action.update({k: v for k, v in debug_fields.items() if v is not None})
            return action

        target_pos = plan_output.get("target_position")
        if target_pos is not None:
            action = {
                "action": "navigate",
                "target_position": target_pos,
                "is_exploration": plan_output.get("is_exploration", False),
                **common,
            }
            action.update({k: v for k, v in debug_fields.items() if v is not None})
            return action

        action = {
            "action": "turn_right",
            "mode": plan_output.get("mode", "no_target_fallback"),
            **common,
        }
        action.update({k: v for k, v in debug_fields.items() if v is not None})
        return action

    def on_goal_reached(self) -> bool:
        """Called when current goal is reached.

        Returns True if there are more goals.
        """
        self._goals_completed += 1
        if self.instruction_graph:
            return self.instruction_graph.advance()
        return False

    _APPROACH_CONFIRM_ANCHOR_RADIUS = 1.5
    _SERVO_MAX_STEPS = 32
    _APPROACH_MAX_STEPS = _SERVO_MAX_STEPS
    _APPROACH_MIN_STEPS_BEFORE_STOP = 8
    _MIN_ADVANCE_BEFORE_STOP = 3
    _STOP_SQUEEZE_REMAINING = 0
    _STOP_BBOX_PEAK_RATIO_NEAR = 0.80
    _STOP_BBOX_PEAK_RATIO_BUDGET = 0.70
    _SERVO_ALIGN_BURST = 2
    _BBOX_MIN_AREA_FOR_APPROACH = 0.008
    _BBOX_MIN_AREA_FOR_STOP = 0.018
    _BBOX_MAX_AREA_FOR_GROUNDING = 0.42
    _APPROACH_CONFIRM_ENTRY_MIN_TRAVEL = 1.0
    _CONFIRM_WINDOW_MAX_STEPS = 10
    _GOAL_NO_PROGRESS_STEPS = 40
    _GOAL_NO_PROGRESS_EPS = 0.5

    @staticmethod
    def _bbox_area(bbox: Optional[List[float]]) -> float:
        if not bbox or len(bbox) < 4:
            return 0.0
        w = max(0.0, float(bbox[2]) - float(bbox[0]))
        h = max(0.0, float(bbox[3]) - float(bbox[1]))
        return w * h

    def _bbox_grounding_flags(self, goal_visible: bool, bbox_area: float) -> Tuple[bool, bool]:
        """Return (approach_grounding_ok, stop_grounding_ok) from bbox area band."""
        if not goal_visible:
            return False, False
        if bbox_area <= 0.0 or bbox_area > self._BBOX_MAX_AREA_FOR_GROUNDING:
            return False, False
        approach_ok = bbox_area >= self._BBOX_MIN_AREA_FOR_APPROACH
        stop_ok = bbox_area >= self._BBOX_MIN_AREA_FOR_STOP
        return approach_ok, stop_ok

    def _vlm_sees_goal_now(self) -> bool:
        """Check if the latest VLM report shows a bbox-grounded goal."""
        vlm = self._current_goal_vlm_confirmation()
        return bool(
            vlm.get("available")
            and vlm.get("goal_visible")
            and vlm.get("grounding_ok")
        )

    def _update_nav_phase(self) -> None:
        """Evaluate and transition the 4-stage navigation state machine.

        States: explore → nav_to_anchor → approach_confirm → (stop via verifier)

        Entering ``approach_confirm`` requires VLM visual evidence to avoid
        false-positive anchor matches locking the agent far from the target.
        """
        if self.instruction_graph is None or self._position is None:
            return

        if self._nav_phase == "approach_confirm":
            return

        anchor_dist = self._nearest_goal_anchor_dist()
        has_memory = anchor_dist != float("inf")
        prev = self._nav_phase

        if prev == "explore":
            if has_memory and anchor_dist <= self._APPROACH_CONFIRM_ANCHOR_RADIUS:
                if (self._goal_travel_distance >= self._APPROACH_CONFIRM_ENTRY_MIN_TRAVEL
                        and self._vlm_sees_goal_now()):
                    self._enter_approach_confirm()
            elif has_memory:
                self._nav_phase = "nav_to_anchor"
                self._goal_latched = True

        elif prev == "nav_to_anchor":
            if anchor_dist <= self._APPROACH_CONFIRM_ANCHOR_RADIUS and self._vlm_sees_goal_now():
                self._enter_approach_confirm()
            elif not has_memory:
                self._nav_phase = "explore"

    def _enter_approach_confirm(self) -> None:
        """Transition into the local visual servo phase."""
        self._nav_phase = "approach_confirm"
        self._confirm_buffer = []
        self._stop_buffer = []
        self._approach_remaining = self._SERVO_MAX_STEPS
        self._approach_lost_count = 0
        self._approach_steps_used = 0
        self._approach_best_bbox_area = 0.0
        self._approach_peak_travel = 0.0
        entry_anchor = self._nearest_goal_anchor_dist()
        self._approach_entry_anchor_dist = entry_anchor if entry_anchor != float("inf") else float("inf")
        self._approach_best_anchor_dist = entry_anchor if entry_anchor != float("inf") else float("inf")
        self._approach_peak_position: Optional[np.ndarray] = None
        self._approach_align_count = 0
        self._approach_advance_steps = 0
        self._servo_prev_bbox_area = 0.0
        self._servo_bbox_shrink_count = 0
        self._servo_visual_advance_count = 0
        self._servo_returning_to_peak = False
        self._confirm_window_active = False
        self._confirm_window_steps = 0
        self._confirm_window_lost_count = 0
        self._last_heavy_step = None
        self._goal_latched = True
        vlm = self._current_goal_vlm_confirmation()
        self._approach_bearing = vlm.get("bearing", "center") if vlm.get("available") else "center"

    def _update_confirm_window(self, vlm: Dict[str, Any]) -> None:
        """Open / advance / close the CONFIRM_WINDOW.

        The window opens when VLM sees the goal nearby, or when
        ``confirm_buffer`` already has evidence. It stays open while
        the goal is visible (up to max steps) and closes after 3
        consecutive steps of target loss.  Re-opening is allowed.
        """
        vlm_close_enough = bool(
            vlm.get("available")
            and vlm.get("goal_visible")
            and (vlm.get("stop_grounding_ok") or vlm.get("grounding_ok"))
            and vlm.get("range_bin") == "near"
        )
        goal_visible_now = bool(vlm.get("available") and vlm.get("goal_visible"))
        has_confirm_evidence = sum(self._confirm_buffer[-3:]) >= 1

        if not self._confirm_window_active:
            if self._approach_steps_used >= self._APPROACH_MIN_STEPS_BEFORE_STOP:
                if vlm_close_enough or has_confirm_evidence:
                    self._confirm_window_active = True
                    self._confirm_window_steps = 0
                    self._confirm_window_lost_count = 0
        else:
            self._confirm_window_steps += 1
            if not goal_visible_now and not has_confirm_evidence:
                self._confirm_window_lost_count += 1
            else:
                self._confirm_window_lost_count = 0
            if (self._confirm_window_lost_count >= 3
                    or self._confirm_window_steps > self._CONFIRM_WINDOW_MAX_STEPS):
                self._confirm_window_active = False
                self._confirm_window_steps = 0
                self._confirm_window_lost_count = 0

    def _current_goal_vlm_confirmation(self) -> Dict[str, Any]:
        """Return current-frame VLM grounding for the active goal.

        Returns a flat dict with the fields that ``_plan_approach_confirm``
        and ``_stop_verifier_passed`` consume: ``available``, ``goal_visible``,
        ``grounding_ok``, ``stop_grounding_ok``, ``range_ok``, ``visibility_ok``,
        ``bearing``, ``confidence``, ``bbox_area``, ``vlm_mode``.
        """
        fresh = self._last_heavy_step == self.topo_map.current_step
        out: Dict[str, Any] = {
            "available": False,
            "goal_visible": False,
            "visibility_ok": False,
            "grounding_ok": False,
            "stop_grounding_ok": False,
            "range_ok": False,
            "bearing": "unknown",
            "range_bin": "unknown",
            "confidence": 0.0,
            "bbox_area": 0.0,
            "fresh": fresh,
            "vlm_mode": self._cur_vlm_mode,
        }
        if self._cur_report.source != "vlm" or not self._cur_report.objects or self.instruction_graph is None:
            return out
        goal = self.instruction_graph.get_current_goal()
        target_label = getattr(goal, "target_object", None) if goal is not None else None
        if target_label is None:
            return out
        target_labels = _split_compound_label(target_label)
        matches = [
            obs for obs in self._cur_report.objects
            if obs.label and _label_matches_goal(obs.label, target_labels) and float(obs.confidence) >= 0.35
        ]
        if not matches:
            return out

        best = max(matches, key=lambda obs: float(obs.confidence))
        visibility = best.visibility or "unknown"
        range_bin = best.range_bin or "unknown"
        bearing = best.bearing or "unknown"
        confidence = float(best.confidence)
        bbox_area = self._bbox_area(best.bbox)
        visible = best.visible if best.visible is not None else visibility != "not_visible"
        visibility_ok = bool(visible and visibility in ("visible", "partially_visible", "unknown"))
        goal_visible = bool(self._cur_report.goal_visible)
        grounding_ok, stop_grounding_ok = self._bbox_grounding_flags(goal_visible, bbox_area)
        range_ok = range_bin == "near" and stop_grounding_ok

        out.update({
            "available": True,
            "goal_visible": goal_visible,
            "visibility_ok": visibility_ok,
            "grounding_ok": grounding_ok,
            "stop_grounding_ok": stop_grounding_ok,
            "range_ok": range_ok,
            "bearing": bearing,
            "range_bin": range_bin,
            "confidence": confidence,
            "bbox_area": bbox_area,
            "bbox": best.bbox,
            "fresh": fresh,
            "vlm_mode": self._cur_vlm_mode,
        })
        return out

    def should_stop(self) -> bool:
        """Simplified stop gate.

        For VLM-backend object targets the real stop decision lives in
        ``_stop_verifier_passed`` (called only from ``_plan_approach_confirm``).
        ``should_stop()`` is kept for non-VLM / non-object paths.
        """
        goal_sim = self._cur_report.best_goal_sim

        warmup_ok = self._goal_local_step >= 5 and self._goal_travel_distance >= 0.25
        if not warmup_ok:
            self._last_stop_debug = {
                "stop_allowed": False,
                "reason": "blocked_by_warmup",
                "goal_sim": round(goal_sim, 4),
                "goal_local_step": self._goal_local_step,
                "goal_travel_distance": round(self._goal_travel_distance, 3),
                "nav_phase": self._nav_phase,
            }
            return False

        if self.config.perception.backend != "vlm" and goal_sim > 0.5:
            self._last_stop_debug = {
                "stop_allowed": True,
                "reason": "local_goal_similarity",
                "goal_sim": round(goal_sim, 4),
                "nav_phase": self._nav_phase,
            }
            return True

        self._last_stop_debug = {
            "stop_allowed": False,
            "reason": "vlm_stop_via_approach_confirm_only",
            "goal_sim": round(goal_sim, 4),
            "nav_phase": self._nav_phase,
            "goal_local_step": self._goal_local_step,
            "goal_travel_distance": round(self._goal_travel_distance, 3),
        }
        return False

    # ==================== Statistics ====================

    @property
    def memory_stats(self) -> Dict[str, Any]:
        """Return memory statistics for analysis."""
        object_nodes = self.topo_map.get_nodes_by_type(NodeType.OBJECT)
        landmark_nodes = self.topo_map.get_nodes_by_type(NodeType.LANDMARK)
        mean_object_conf = float(np.mean([n.confidence for n in object_nodes])) if object_nodes else 0.0
        granularity_counts = {"object": 0, "landmark": 0, "room_level": 0}
        folded_count = 0
        semantic_anchor_count = 0
        for n in object_nodes:
            g = n.attributes.get("granularity", "object")
            granularity_counts[g] = granularity_counts.get(g, 0) + 1
            if n.attributes.get("folded"):
                folded_count += 1
            if n.attributes.get("is_semantic_anchor"):
                semantic_anchor_count += 1
        for n in landmark_nodes:
            g = n.attributes.get("granularity", "landmark")
            granularity_counts[g] = granularity_counts.get(g, 0) + 1
            if n.attributes.get("folded"):
                folded_count += 1
            if n.attributes.get("is_semantic_anchor"):
                semantic_anchor_count += 1
        room_summaries = sum(
            1 for node in self.topo_map.get_nodes_by_type(NodeType.ROOM)
            if node.attributes.get("summary_type") == "room_region"
        )
        semantic_summary_count = sum(
            1 for node in self.topo_map.get_nodes_by_type(NodeType.ROOM)
            if bool(node.attributes.get("semantic_summary"))
        )
        visible_semantic = (
            sum(1 for n in object_nodes if not n.attributes.get("folded"))
            + sum(1 for n in landmark_nodes if not n.attributes.get("folded"))
            + room_summaries
        )
        return {
            "total_nodes": self.topo_map.num_nodes,
            "visited_waypoints": len(self.topo_map.get_visited()),
            "frontiers": len(self.topo_map.get_frontiers()),
            "candidate_waypoints": len(self.topo_map.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE)),
            "objects": len(object_nodes),
            "rooms": len(self.topo_map.get_nodes_by_type(NodeType.ROOM)),
            "landmarks": len(landmark_nodes),
            "step": self._step_count,
            "goals_completed": self._goals_completed,
            "consumed_frontiers": len(self._consumed_frontier_ids),
            "blocked_targets": len(self._active_blocked_targets()),
            "heavy_perception_calls": self._heavy_perception_calls,
            "vlm_perception_attempts": self._vlm_perception_attempts,
            "object_merge_count": self._object_merge_count,
            "mean_object_confidence": mean_object_conf,
            "room_summaries": room_summaries,
            "granularity_counts": granularity_counts,
            "folded_nodes": semantic_anchor_count,
            "semantic_anchors": semantic_anchor_count,
            "semantic_room_summaries": semantic_summary_count,
            "visible_semantic_nodes": visible_semantic,
            "granularity_debug": getattr(self.topo_map, "_last_granularity_debug", {}),
            "last_heavy": self._last_heavy_debug,
            "nav_phase": self._nav_phase,
        }
