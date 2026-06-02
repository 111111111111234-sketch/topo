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

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, EdgeType, SemanticNode
from conftopo.core.instruction_graph import InstructionGraph, GoalNode
from conftopo.core.rule_scorer import compute_semantic_bias
from conftopo.perception.light_perceiver import LightPerceiver


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
        self._position: Optional[np.ndarray] = None
        self._heading: float = 0.0
        self._prev_position: Optional[np.ndarray] = None
        self._cur_vp_id: Optional[str] = None

        # Observation cache (set in observe())
        self._cur_rgb_embed: Optional[np.ndarray] = None
        self._cur_perception: Optional[Dict] = None

        # Frontier generation parameters
        self._frontier_step_size: float = 2.5  # estimated distance for new frontiers
        self._frontier_merge_radius: float = 1.5
        self._min_move_for_frontier: float = 0.5  # minimum displacement to generate frontiers

        # Multi-goal tracking
        self._goal_idx: int = 0
        self._goals_completed: int = 0

    def reset(self):
        """Full reset for new episode (clear memory)."""
        super().reset()
        self._position = None
        self._heading = 0.0
        self._prev_position = None
        self._cur_vp_id = None
        self._cur_rgb_embed = None
        self._cur_perception = None
        self._goal_idx = 0
        self._goals_completed = 0

    def set_new_goal(self, goal: GoalNode):
        """Switch to a new goal within the same episode (memory preserved).

        This is the key for multi-goal: DynamicTopoMap is NOT cleared.
        """
        self.reset_keep_memory()

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

    def observe(self, obs: Dict[str, Any]) -> None:
        """Process current observation.

        Expected obs keys:
            - 'rgb': raw RGB image [H, W, 3] (for visualization, not used directly)
            - 'rgb_embed': CLIP visual embedding [D] (pre-computed by encoder)
            - 'position': agent position [3] (x, y, z)
            - 'heading': agent heading in radians
            - 'depth' (optional): depth image
        """
        self._prev_position = self._position
        self._position = np.array(obs['position'], dtype=np.float32)
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

    def _add_semantic_nodes(self, cur_vp: str) -> None:
        """Add room/landmark/object nodes from perception results."""
        if not self._cur_perception:
            return

        pcfg = self.config.perception

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
                )
                self.topo_map.add_edge(cur_vp, lm_id, EdgeType.VISIBLE_FROM)

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

        # Get all candidate nodes (frontiers + unvisited waypoints + high-sim objects)
        candidates = []
        candidate_ids = []

        for node in self.topo_map._nodes.values():
            if node.node_id == self._cur_vp_id:
                continue
            if node.node_type in (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_VISITED):
                candidates.append(node)
                candidate_ids.append(node.node_id)
            elif node.node_type == NodeType.OBJECT:
                # Include high-similarity objects as direct targets
                candidates.append(node)
                candidate_ids.append(node.node_id)

        if not candidates:
            return {"target_node_id": None, "target_position": None, "is_exploration": True}

        # Score candidates
        scores = compute_semantic_bias(
            goal_graph=self.instruction_graph,
            topo_map=self.topo_map,
            candidate_node_ids=candidate_ids,
            agent_position=self._position,
            normalize=False,  # use raw scores for selection
        )

        best_idx = int(np.argmax(scores))
        best_node = candidates[best_idx]

        return {
            "target_node_id": best_node.node_id,
            "target_position": best_node.position.copy(),
            "is_exploration": best_node.node_type == NodeType.WAYPOINT_FRONTIER,
            "scores": scores,
            "candidate_ids": candidate_ids,
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
            }

        return {
            "action": "navigate",
            "target_position": target_pos,
            "target_node_id": plan_output.get("target_node_id"),
            "candidate_ids": plan_output.get("candidate_ids", []),
            "scores": plan_output.get("scores", []),
            "is_exploration": plan_output.get("is_exploration", False),
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
    def memory_stats(self) -> Dict[str, int]:
        """Return memory statistics for analysis."""
        return {
            "total_nodes": self.topo_map.num_nodes,
            "visited_waypoints": len(self.topo_map.get_visited()),
            "frontiers": len(self.topo_map.get_frontiers()),
            "objects": len(self.topo_map.get_nodes_by_type(NodeType.OBJECT)),
            "rooms": len(self.topo_map.get_nodes_by_type(NodeType.ROOM)),
            "landmarks": len(self.topo_map.get_nodes_by_type(NodeType.LANDMARK)),
            "step": self._step_count,
            "goals_completed": self._goals_completed,
        }
