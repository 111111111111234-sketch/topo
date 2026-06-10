"""ETPNav GraphMap compatibility adapter for ConfTopo.

The adapter keeps ETPNav's original GraphMap behavior as the source of truth
and mirrors its visited/ghost graph into DynamicTopoMap. This gives Phase 1 a
drop-in compatibility layer without changing ETPNav navigation semantics when
semantic bias is disabled.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType
from conftopo.core.instruction_graph import InstructionGraph
from conftopo.core.rule_scorer import compute_semantic_bias


class ConfTopoETPNavAdapter:
    """Wrap an ETPNav GraphMap and mirror it into DynamicTopoMap.

    Unknown attributes and methods are forwarded to the wrapped GraphMap, so
    existing ETPNav code can keep using fields such as ``node_pos``,
    ``ghost_pos``, ``ghost_aug_pos``, ``shortest_path`` and ``get_pos_fts``.
    """

    def __init__(
        self,
        has_real_pos: bool = False,
        loc_noise: float = 0.5,
        merge_ghost: bool = True,
        ghost_aug: float = 0.0,
        config: Optional[ConfTopoConfig] = None,
        graph_map: Optional[Any] = None,
        graph_map_cls: Optional[type] = None,
    ):
        self.config = config or ConfTopoConfig()
        self.topo_map = DynamicTopoMap(self.config.memory)
        self.instruction_graph: Optional[InstructionGraph] = None
        self._vp_to_node: dict[Any, str] = {}

        if graph_map is not None:
            self._gmap = graph_map
        else:
            if graph_map_cls is None:
                graph_map_cls = self._load_etpnav_graph_map_cls()
            self._gmap = graph_map_cls(has_real_pos, loc_noise, merge_ghost, ghost_aug)

    @staticmethod
    def _load_etpnav_graph_map_cls() -> type:
        try:
            from vlnce_baselines.models.graph_utils import GraphMap
        except Exception as exc:  # pragma: no cover - depends on ETPNav env
            raise ImportError(
                "Cannot import ETPNav GraphMap. Pass graph_map or graph_map_cls "
                "when using the adapter outside the ETPNav environment."
            ) from exc
        return GraphMap

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gmap, name)

    @property
    def graph_map(self) -> Any:
        """The wrapped original ETPNav GraphMap."""
        return self._gmap

    @property
    def alpha(self) -> float:
        return float(self.config.planning.alpha)

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.config.planning.alpha = float(value)

    def set_instruction_graph(self, instruction_graph: InstructionGraph) -> None:
        self.instruction_graph = instruction_graph

    def identify_node(self, cur_pos, cur_ori, cand_ang, cand_dis):
        return self._gmap.identify_node(cur_pos, cur_ori, cand_ang, cand_dis)

    def delete_ghost(self, vp) -> None:
        self._gmap.delete_ghost(vp)
        self._sync_from_graph_map()

    def update_graph(
        self,
        prev_vp,
        step_id,
        cur_vp,
        cur_pos,
        cur_embeds,
        cand_vp,
        cand_pos,
        cand_embeds,
        cand_real_pos,
    ) -> None:
        """Run the original GraphMap update, then mirror the resulting graph."""
        self._gmap.update_graph(
            prev_vp,
            step_id,
            cur_vp,
            cur_pos,
            cur_embeds,
            cand_vp,
            cand_pos,
            cand_embeds,
            cand_real_pos,
        )
        self._sync_from_graph_map()

    def _sync_from_graph_map(self) -> None:
        """Rebuild DynamicTopoMap from the wrapped GraphMap state."""
        self.topo_map.reset()
        self._vp_to_node.clear()

        max_step = 0
        for step in getattr(self._gmap, "node_stepId", {}).values():
            max_step = max(max_step, int(step))
        self.topo_map._current_step = max_step

        for vp, pos in getattr(self._gmap, "node_pos", {}).items():
            node_id = self._node_id(vp)
            self.topo_map.add_node(
                NodeType.WAYPOINT_VISITED,
                position=np.asarray(pos, dtype=np.float32),
                embedding=self._embedding_or_none(getattr(self._gmap, "node_embeds", {}).get(vp)),
                confidence=0.9,
                label=str(vp),
                node_id=node_id,
                attributes={"etpnav_vp": vp, "source": "etpnav_node"},
            )
            node = self.topo_map.get_node(node_id)
            if node is not None:
                node.step_id = int(getattr(self._gmap, "node_stepId", {}).get(vp, 0))
                node.visit_count = 1
            self._vp_to_node[vp] = node_id

        ghost_positions = getattr(self._gmap, "ghost_mean_pos", None)
        if ghost_positions is None:
            ghost_positions = getattr(self._gmap, "ghost_pos", {})
            ghost_positions = {
                vp: np.mean(np.asarray(positions, dtype=np.float32), axis=0)
                for vp, positions in ghost_positions.items()
            }
        for vp, pos in ghost_positions.items():
            node_id = self._node_id(vp)
            self.topo_map.add_node(
                NodeType.WAYPOINT_FRONTIER,
                position=np.asarray(pos, dtype=np.float32),
                embedding=self._ghost_embedding(vp),
                confidence=0.4,
                label=str(vp),
                node_id=node_id,
                attributes={"etpnav_vp": vp, "source": "etpnav_ghost"},
            )
            self._vp_to_node[vp] = node_id

        graph_nx = getattr(self._gmap, "graph_nx", None)
        if graph_nx is not None:
            for u, v in graph_nx.edges():
                if u in self._vp_to_node and v in self._vp_to_node:
                    weight = float(graph_nx.edges[u, v].get("weight", 1.0))
                    self.topo_map.add_edge(self._vp_to_node[u], self._vp_to_node[v], EdgeType.NAVIGABLE, weight=weight)

        for ghost_vp, front_vps in getattr(self._gmap, "ghost_fronts", {}).items():
            if ghost_vp not in self._vp_to_node:
                continue
            for front_vp in front_vps:
                if front_vp in self._vp_to_node:
                    self.topo_map.add_edge(self._vp_to_node[front_vp], self._vp_to_node[ghost_vp], EdgeType.NAVIGABLE)

    def _node_id(self, vp: Any) -> str:
        return f"etp_{vp}"

    @staticmethod
    def _embedding_or_none(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _ghost_embedding(self, vp: Any) -> Optional[np.ndarray]:
        value = getattr(self._gmap, "ghost_embeds", {}).get(vp)
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            denom = max(float(value[1]), 1.0)
            emb = self._embedding_or_none(value[0])
            return None if emb is None else emb / denom
        return self._embedding_or_none(value)

    def get_semantic_bias(self, gmap_vp_ids: list[Any], agent_pos: np.ndarray) -> np.ndarray:
        """Return semantic bias aligned to ETPNav's ``gmap_vp_ids`` list."""
        if self.alpha == 0.0 or self.instruction_graph is None:
            return np.zeros(len(gmap_vp_ids), dtype=np.float32)

        node_ids = [self._vp_to_node.get(vp) for vp in gmap_vp_ids]
        valid_ids = [node_id for node_id in node_ids if node_id is not None]
        if not valid_ids:
            return np.zeros(len(gmap_vp_ids), dtype=np.float32)

        valid_scores = compute_semantic_bias(
            goal_graph=self.instruction_graph,
            topo_map=self.topo_map,
            candidate_node_ids=valid_ids,
            agent_position=np.asarray(agent_pos, dtype=np.float32),
            normalize=self.config.planning.normalize_scores,
        )

        scores = np.zeros(len(gmap_vp_ids), dtype=np.float32)
        valid_idx = 0
        for idx, node_id in enumerate(node_ids):
            if node_id is not None:
                scores[idx] = valid_scores[valid_idx]
                valid_idx += 1
        return scores * self.alpha


# Backward-compatible public name used by tests and package exports.
ConfTopoETPNavAgent = ConfTopoETPNavAdapter
