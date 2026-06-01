"""ConfTopo-ETPNav Agent: bridges ConfTopo Core with ETPNav's GraphMap.

This agent observes ETPNav's graph updates and mirrors them into a
DynamicTopoMap for semantic scoring. It does NOT replace GraphMap
(ETPNav still uses its own GraphMap for navigation); it maintains a
parallel semantic memory for the bias computation.

Usage in ss_trainer_ETP.py:
    from conftopo.agents.etpnav_agent import ConfTopoETPNavAgent

    # At episode start:
    conftopo_agent = ConfTopoETPNavAgent(config)

    # After each graph update:
    conftopo_agent.on_graph_update(cur_vp, cur_pos, cur_embeds, cand_vps, cand_pos)

    # When computing navigation logits:
    semantic_bias = conftopo_agent.get_semantic_bias(gmap_vp_ids, agent_pos)
    nav_logits = nav_logits + alpha * semantic_bias
"""

from typing import List, Optional, Dict
import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, EdgeType
from conftopo.core.instruction_graph import InstructionGraph
from conftopo.core.rule_scorer import compute_semantic_bias


class ConfTopoETPNavAgent:
    """Lightweight agent that mirrors ETPNav's GraphMap into DynamicTopoMap."""

    def __init__(self, config: Optional[ConfTopoConfig] = None):
        if config is None:
            config = ConfTopoConfig()
        self.config = config
        self.topo_map = DynamicTopoMap(config.memory)
        self.instruction_graph: Optional[InstructionGraph] = None

        # Map ETPNav vp ids to DynamicTopoMap node ids
        self._vp_to_node: Dict[str, str] = {}
        self._alpha = config.planning.alpha

    def set_instruction_graph(self, ig: InstructionGraph):
        """Set the instruction/goal graph for this episode."""
        self.instruction_graph = ig

    def reset(self):
        """Reset for new episode."""
        self.topo_map.reset()
        self._vp_to_node.clear()

    def on_graph_update(
        self,
        cur_vp: str,
        cur_pos: np.ndarray,
        cur_embeds: Optional[np.ndarray],
        cand_vps: List[str],
        cand_pos: List[np.ndarray],
        prev_vp: Optional[str] = None,
    ):
        """Called after ETPNav's GraphMap.update_graph().

        Mirrors the update into DynamicTopoMap.
        """
        self.topo_map.step()

        # Add/update visited node
        if cur_vp not in self._vp_to_node:
            node_id = self.topo_map.add_node(
                NodeType.WAYPOINT_VISITED,
                position=cur_pos,
                embedding=cur_embeds,
                confidence=0.9,
                label=cur_vp,
            )
            self._vp_to_node[cur_vp] = node_id
        else:
            node_id = self._vp_to_node[cur_vp]
            node = self.topo_map.get_node(node_id)
            if node is not None:
                node.visit_count += 1
                node.confidence = min(1.0, node.confidence + 0.1)

        # Connect to previous
        if prev_vp is not None and prev_vp in self._vp_to_node:
            prev_node_id = self._vp_to_node[prev_vp]
            self.topo_map.add_edge(prev_node_id, node_id, EdgeType.NAVIGABLE)

        # Add/update frontier (ghost) nodes
        for cvp, cpos in zip(cand_vps, cand_pos):
            if cvp in self._vp_to_node:
                # Already known, maybe promote if now visited
                continue
            # Check if this is near an existing frontier
            existing = self.topo_map.find_nearest_node(
                cpos, node_type=NodeType.WAYPOINT_FRONTIER
            )
            if existing and np.linalg.norm(existing.position - cpos) < 1.5:
                # Update existing frontier's position estimate
                existing.position = (existing.position + cpos) / 2
                existing.confidence = min(1.0, existing.confidence + 0.05)
                self._vp_to_node[cvp] = existing.node_id
            else:
                fid = self.topo_map.add_node(
                    NodeType.WAYPOINT_FRONTIER,
                    position=cpos,
                    confidence=0.4,
                    label=cvp,
                )
                self._vp_to_node[cvp] = fid
                self.topo_map.add_edge(node_id, fid, EdgeType.NAVIGABLE)

    def on_visit_ghost(self, ghost_vp: str, actual_pos: np.ndarray):
        """Called when agent visits a ghost node (promotes to visited)."""
        if ghost_vp in self._vp_to_node:
            node_id = self._vp_to_node[ghost_vp]
            self.topo_map.promote_frontier_to_visited(node_id)
            node = self.topo_map.get_node(node_id)
            if node is not None:
                node.position = actual_pos

    def get_semantic_bias(
        self,
        gmap_vp_ids: List[str],
        agent_pos: np.ndarray,
    ) -> np.ndarray:
        """Compute semantic bias for ETPNav's candidate viewpoints.

        Returns scores aligned with gmap_vp_ids.
        When alpha=0, returns zeros (no effect on original ETPNav).
        """
        if self._alpha == 0.0 or self.instruction_graph is None:
            return np.zeros(len(gmap_vp_ids), dtype=np.float32)

        # Map ETPNav vp ids to DynamicTopoMap node ids
        node_ids = []
        for vp in gmap_vp_ids:
            if vp in self._vp_to_node:
                node_ids.append(self._vp_to_node[vp])
            else:
                node_ids.append(None)

        # For nodes we can't map, we'll get zero scores
        valid_node_ids = [nid for nid in node_ids if nid is not None]
        if not valid_node_ids:
            return np.zeros(len(gmap_vp_ids), dtype=np.float32)

        # Compute bias only for valid nodes
        valid_scores = compute_semantic_bias(
            goal_graph=self.instruction_graph,
            topo_map=self.topo_map,
            candidate_node_ids=valid_node_ids,
            agent_position=agent_pos,
            normalize=self.config.planning.normalize_scores,
        )

        # Map back to full gmap_vp_ids ordering
        scores = np.zeros(len(gmap_vp_ids), dtype=np.float32)
        valid_idx = 0
        for i, nid in enumerate(node_ids):
            if nid is not None:
                scores[i] = valid_scores[valid_idx]
                valid_idx += 1

        return scores * self._alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
