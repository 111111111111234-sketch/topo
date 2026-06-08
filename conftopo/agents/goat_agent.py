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
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, EdgeType, SemanticNode
from conftopo.core.instruction_graph import InstructionGraph, GoalNode
from conftopo.core.rule_scorer import compute_semantic_bias
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.heavy_perceiver import GroundingDINOBackend, HeavyPerceiver, ObjectObservation


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


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
        self._cur_perception: Optional[Dict] = None
        self._cur_heavy_observations: List[ObjectObservation] = []
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
        self._last_heavy_step: Optional[int] = None
        self._heavy_perception_calls: int = 0
        self._object_merge_count: int = 0
        self._last_heavy_debug: Dict[str, Any] = {}

    def reset(self):
        """Full reset for new episode (clear memory)."""
        super().reset()
        self._position = None
        self._origin_position = None
        self._heading = 0.0
        self._prev_position = None
        self._cur_vp_id = None
        self._cur_rgb = None
        self._cur_rgb_embed = None
        self._cur_perception = None
        self._cur_heavy_observations = []
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
        self._last_heavy_step = None
        self._heavy_perception_calls = 0
        self._object_merge_count = 0
        self._last_heavy_debug = {}

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
        self.reset_keep_memory()
        self._clear_sticky("new_goal")

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
        self._goal_local_step = 0
        self._last_heavy_step = None

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
        self._heading = obs.get('heading', 0.0)

        # Visual embedding (CLIP)
        if 'rgb_embed' in obs:
            self._cur_rgb_embed = np.array(obs['rgb_embed'], dtype=np.float32)
        elif 'pano_rgb_embed' in obs:
            self._cur_rgb_embed = np.array(obs['pano_rgb_embed'], dtype=np.float32)
        else:
            self._cur_rgb_embed = None

        # Run CLIP perception
        if self._cur_rgb_embed is not None:
            self._cur_perception = self.perceiver.perceive(self._cur_rgb_embed)
        else:
            self._cur_perception = {}
        self._cur_heavy_observations = []

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
        self._goal_local_step += 1

        # 1. Add/update visited waypoint
        cur_vp = self._add_visited_waypoint()

        self._consume_reached_frontiers(cur_vp)

        # 2. Add semantic nodes (room, landmark detection)
        room_id = self._add_semantic_nodes(cur_vp)

        # 2b. Add object-level heavy detections when triggered
        self._add_heavy_object_nodes(cur_vp, room_id)

        # 3. Generate frontier-like nodes
        self._generate_frontiers(cur_vp)

        # 4. Memory maintenance
        self.topo_map.decay_all_confidences()
        self.topo_map.merge_nearby_nodes(NodeType.WAYPOINT_FRONTIER)
        self.topo_map.adaptive_granularity(self._position)
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

    def _sticky_plan_if_valid(self, candidates, candidate_ids, scores) -> Optional[Dict[str, Any]]:
        cfg = self.config.planning
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
        return {
            "target_node_id": node.node_id,
            "target_position": node.position.copy(),
            "is_exploration": node.node_type == NodeType.WAYPOINT_FRONTIER,
            "scores": scores,
            "candidate_ids": candidate_ids,
            "sticky_debug": self._last_plan_debug,
        }

    def _add_semantic_nodes(self, cur_vp: str) -> Optional[str]:
        """Add room/landmark/object nodes from perception results."""
        if not self._cur_perception:
            return None

        pcfg = self.config.perception
        room_id = None

        # Room node
        room_label = self._cur_perception.get("room_label", "unknown")
        room_conf = float(self._cur_perception.get("room_confidence", 0.0))
        if room_label != "unknown" and room_conf >= pcfg.room_threshold:
            existing_rooms = self.topo_map.find_nodes_within_radius(
                self._position, radius=5.0, node_type=NodeType.ROOM
            )
            matched_room = next((r for r in existing_rooms if r.label == room_label), None)
            if matched_room:
                matched_room.confidence = max(matched_room.confidence, room_conf)
                matched_room.step_id = self.topo_map.current_step
                if self._cur_rgb_embed is not None:
                    matched_room.embedding = self._cur_rgb_embed
                room_id = matched_room.node_id
            else:
                room_id = self.topo_map.add_node(
                    NodeType.ROOM,
                    position=self._position.copy(),
                    embedding=self._cur_rgb_embed,
                    confidence=room_conf,
                    label=room_label,
                )
                self.topo_map.add_edge(cur_vp, room_id, EdgeType.BELONGS_TO)

        # Goal object node
        for label, sim in self._cur_perception.get("goal_scores", []):
            sim = float(sim)
            if sim < pcfg.object_threshold:
                continue
            existing = self.topo_map.find_nodes_within_radius(
                self._position, radius=2.0, node_type=NodeType.OBJECT
            )
            matched = next((o for o in existing if o.label == label), None)
            if matched:
                matched.confidence = max(matched.confidence, sim)
                matched.step_id = self.topo_map.current_step
                if self._cur_rgb_embed is not None:
                    matched.embedding = self._cur_rgb_embed
            else:
                obj_id = self.topo_map.add_node(
                    NodeType.OBJECT,
                    position=self._position.copy(),
                    embedding=self._cur_rgb_embed,
                    confidence=sim,
                    label=label,
                )
                self.topo_map.add_edge(cur_vp, obj_id, EdgeType.OBSERVED_AT)

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
                    attributes={
                        "landmark_source": "environment"
                        if label in self._environment_landmark_labels
                        else "goal_hint"
                    },
                )
                self.topo_map.add_edge(cur_vp, lm_id, EdgeType.VISIBLE_FROM)
        return room_id

    def _heavy_labels(self) -> List[str]:
        labels: List[str] = []
        goal = self.instruction_graph.get_current_goal() if self.instruction_graph else None
        if goal is not None and getattr(goal, "target_object", None):
            labels.append(str(goal.target_object))
            labels.extend(str(a) for a in getattr(goal, "attributes", []) if a)
            labels.extend(str(lm) for lm in getattr(goal, "landmarks", []) if lm)
        for label, sim in (self._cur_perception or {}).get("goal_scores", []):
            if float(sim) >= self.config.perception.object_threshold:
                labels.append(str(label))
        seen = set()
        return [x for x in labels if not (x in seen or seen.add(x))]

    def _should_run_heavy_perception(self) -> Tuple[bool, str]:
        pcfg = self.config.perception
        if not pcfg.heavy_enabled or self.heavy_perceiver is None:
            return False, "disabled"
        if self._cur_rgb is None:
            return False, "missing_rgb"
        interval = max(1, int(pcfg.heavy_interval))
        if self._last_heavy_step is not None and self.topo_map.current_step - self._last_heavy_step < interval:
            return False, "cooldown"
        if self._last_heavy_step is None and self._goal_local_step <= max(1, int(pcfg.heavy_goal_warmup_steps)):
            return True, "goal_warmup"
        if self.topo_map.current_step % interval == 0:
            return True, "interval"
        if float((self._cur_perception or {}).get("best_goal_sim", 0.0)) >= pcfg.heavy_goal_sim_threshold:
            return True, "high_goal_similarity"
        if pcfg.heavy_on_frontier and self._position is not None:
            nearby_frontiers = self.topo_map.find_nodes_within_radius(
                self._position,
                self.config.memory.near_radius,
                NodeType.WAYPOINT_FRONTIER,
            )
            if nearby_frontiers:
                return True, "frontier_context"
        object_nodes = self.topo_map.get_nodes_by_type(NodeType.OBJECT)
        if not object_nodes or max(n.confidence for n in object_nodes) < pcfg.heavy_low_object_confidence:
            return True, "low_object_confidence"
        return False, "not_triggered"

    def _object_position_from_bbox(self, bbox: List[float], heading: float) -> np.ndarray:
        if self._position is None:
            return np.zeros(3, dtype=np.float32)
        x1, _, x2, _ = [float(v) for v in bbox]
        cx = (x1 + x2) * 0.5
        # Assume normalized bbox if values are <= 1, otherwise use a 640px image width fallback.
        image_width = 1.0 if max(abs(x1), abs(x2)) <= 1.0 else 640.0
        centered = (cx / image_width) - 0.5
        bearing = heading - centered * 1.2
        dist = min(self.config.memory.near_radius, 2.0)
        return np.array([
            self._position[0] - dist * np.sin(bearing),
            self._position[1],
            self._position[2] - dist * np.cos(bearing),
        ], dtype=np.float32)

    def _add_heavy_object_nodes(self, cur_vp: str, room_id: Optional[str]) -> None:
        should_run, reason = self._should_run_heavy_perception()
        self._last_heavy_debug = {"ran": False, "reason": reason, "detections": 0}
        if not should_run:
            return
        labels = self._heavy_labels()
        if not labels:
            self._last_heavy_debug = {"ran": False, "reason": "no_labels", "detections": 0}
            return
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
            return

        self._last_heavy_step = self.topo_map.current_step
        self._heavy_perception_calls += 1
        self._cur_heavy_observations = observations
        goal = self.instruction_graph.get_current_goal() if self.instruction_graph else None
        room_label = self._cur_perception.get("room_label") if self._cur_perception else None
        room_priors = set(getattr(goal, "room_prior", []) or [])
        merges = 0
        for obs in observations:
            target_relevance = 1.0 if goal is not None and obs.label == getattr(goal, "target_object", None) else 0.0
            room_prior_score = 1.0 if room_label in room_priors else 0.0
            obj_id, merged = self.topo_map.upsert_object_observation(
                label=obs.label,
                bbox=obs.bbox,
                confidence=obs.confidence,
                position=self._object_position_from_bbox(obs.bbox, obs.view_heading),
                embedding=obs.embedding,
                viewpoint_id=cur_vp,
                view_heading=obs.view_heading,
                room_context=room_label,
                target_relevance=target_relevance,
                room_prior_score=room_prior_score,
                source=obs.source,
            )
            if room_id is not None:
                self.topo_map.add_edge(obj_id, room_id, EdgeType.BELONGS_TO)
            if merged:
                merges += 1
                self._object_merge_count += 1
        self._last_heavy_debug = {
            "ran": True,
            "reason": reason,
            "detections": len(observations),
            "labels": labels,
            "merged": merges,
        }

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
        }

    def plan(self) -> Dict[str, Any]:
        """Determine next target based on goal + memory.

        Returns dict with:
            - target_node_id: best node to navigate to
            - target_position: 3D position of target
            - is_exploration: whether this is frontier exploration
            - scores: all node scores
        """
        if self.instruction_graph is None or self._cur_vp_id is None:
            return {"target_node_id": None, "target_position": None, "is_exploration": True}

        self._last_navigation_event = {}
        primary_candidates = []
        primary_ids = []
        fallback_candidates = []
        fallback_ids = []
        self._last_skipped_candidates = []

        for node in self.topo_map._nodes.values():
            reason = self._candidate_skip_reason(node)
            if reason is not None:
                self._last_skipped_candidates.append({"node_id": node.node_id, "type": node.node_type.value, "reason": reason})
                continue
            if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE, NodeType.OBJECT):
                primary_candidates.append(node)
                primary_ids.append(node.node_id)
            elif node.node_type == NodeType.WAYPOINT_VISITED:
                fallback_candidates.append(node)
                fallback_ids.append(node.node_id)

        candidates = primary_candidates if primary_candidates else fallback_candidates
        candidate_ids = primary_ids if primary_candidates else fallback_ids

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

        sticky_plan = self._sticky_plan_if_valid(candidates, candidate_ids, scores)
        if sticky_plan is not None:
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
        }
        self._sticky_release_reason = ""

        return {
            "target_node_id": best_node.node_id,
            "target_position": best_node.position.copy(),
            "is_exploration": best_node.node_type == NodeType.WAYPOINT_FRONTIER,
            "scores": scores,
            "candidate_ids": candidate_ids,
            "sticky_debug": self._last_plan_debug,
        }

    def act(self, plan_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert plan output to action for PointNav controller.

        Returns:
            dict with 'target_position' for PointNav, or 'stop' if goal reached.
        """
        target_pos = plan_output.get("target_position")
        if target_pos is None:
            # No target: random exploration or stop
            return {
                "action": "stop",
                "target_node_id": plan_output.get("target_node_id"),
                "candidate_ids": plan_output.get("candidate_ids", []),
                "scores": plan_output.get("scores", []),
                "is_exploration": plan_output.get("is_exploration", True),
                "sticky_debug": plan_output.get("sticky_debug", self._last_plan_debug),
            }

        return {
            "action": "navigate",
            "target_position": target_pos,
            "target_node_id": plan_output.get("target_node_id"),
            "candidate_ids": plan_output.get("candidate_ids", []),
            "scores": plan_output.get("scores", []),
            "is_exploration": plan_output.get("is_exploration", False),
            "sticky_debug": plan_output.get("sticky_debug", self._last_plan_debug),
        }

    def on_goal_reached(self) -> bool:
        """Called when current goal is reached.

        Returns True if there are more goals.
        """
        self._goals_completed += 1
        if self.instruction_graph:
            return self.instruction_graph.advance()
        return False

    def should_stop(self) -> bool:
        """Check if agent should call STOP for current goal.

        Uses goal similarity + proximity heuristic.
        """
        if self._cur_perception is None:
            return False
        goal_sim = self._cur_perception.get("best_goal_sim", 0.0)
        # High goal similarity at current location → likely found the target
        return goal_sim > 0.5

    # ==================== Statistics ====================

    @property
    def memory_stats(self) -> Dict[str, Any]:
        """Return memory statistics for analysis."""
        object_nodes = self.topo_map.get_nodes_by_type(NodeType.OBJECT)
        mean_object_conf = float(np.mean([n.confidence for n in object_nodes])) if object_nodes else 0.0
        return {
            "total_nodes": self.topo_map.num_nodes,
            "visited_waypoints": len(self.topo_map.get_visited()),
            "frontiers": len(self.topo_map.get_frontiers()),
            "candidate_waypoints": len(self.topo_map.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE)),
            "objects": len(object_nodes),
            "rooms": len(self.topo_map.get_nodes_by_type(NodeType.ROOM)),
            "landmarks": len(self.topo_map.get_nodes_by_type(NodeType.LANDMARK)),
            "step": self._step_count,
            "goals_completed": self._goals_completed,
            "consumed_frontiers": len(self._consumed_frontier_ids),
            "blocked_targets": len(self._active_blocked_targets()),
            "heavy_perception_calls": self._heavy_perception_calls,
            "object_merge_count": self._object_merge_count,
            "mean_object_confidence": mean_object_conf,
            "last_heavy": self._last_heavy_debug,
        }
