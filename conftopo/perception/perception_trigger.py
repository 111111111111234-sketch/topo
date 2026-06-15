"""Extracted heavy-perception trigger logic.

Mirrors the conditions from ``ConfTopoGOATAgent._should_run_heavy_perception``
so the agent can delegate the decision and the same policy can be reused by the
VLM perception path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from conftopo.config import MemoryConfig, PerceptionConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType


@dataclass
class TriggerState:
    """Mutable state carried across steps by the agent."""

    last_heavy_step: Optional[int] = None
    last_heavy_summary_step: Optional[int] = None
    goal_local_step: int = 0
    reground_state: str = "idle"


class PerceptionTrigger:
    """Stateless evaluator that decides whether heavy / VLM perception should fire."""

    def __init__(self, config: PerceptionConfig, memory_config: MemoryConfig):
        self._pcfg = config
        self._mcfg = memory_config

    def should_run(
        self,
        state: TriggerState,
        step: int,
        cur_rgb: Any,
        best_goal_sim: float,
        position: Optional[np.ndarray],
        topo_map: DynamicTopoMap,
        has_near_goal_object: bool,
        heavy_perceiver_available: bool,
    ) -> Tuple[bool, str]:
        pcfg = self._pcfg
        if not pcfg.heavy_enabled or not heavy_perceiver_available:
            return False, "disabled"
        if cur_rgb is None:
            return False, "missing_rgb"

        if state.reground_state in ("scanning", "searching"):
            return True, "local_regrounding"

        if has_near_goal_object:
            min_confirm_cd = 3
            if state.last_heavy_step is None or step - state.last_heavy_step >= min_confirm_cd:
                return True, "stop_confirmation_near_goal"

        interval = max(1, int(pcfg.heavy_interval))

        if state.last_heavy_step is not None and step - state.last_heavy_step < interval:
            return False, "cooldown"

        if state.last_heavy_step is None and state.goal_local_step <= max(1, int(pcfg.heavy_goal_warmup_steps)):
            return True, "goal_warmup"

        if position is not None:
            summary = topo_map.find_nearby_room_summary(
                position, radius=self._mcfg.summary_radius,
            )
            if summary is not None:
                summary_cooldown = max(1, int(pcfg.heavy_summary_cooldown))
                if (state.last_heavy_summary_step is None
                        or step - state.last_heavy_summary_step >= summary_cooldown):
                    return True, "coarse_summary_context"

        if step % interval == 0:
            return True, "interval"

        if best_goal_sim >= pcfg.heavy_goal_sim_threshold:
            return True, "high_goal_similarity"

        if pcfg.heavy_on_frontier and position is not None:
            nearby_frontiers = topo_map.find_nodes_within_radius(
                position, self._mcfg.near_radius, NodeType.WAYPOINT_FRONTIER,
            )
            if nearby_frontiers:
                return True, "frontier_context"

        object_nodes = topo_map.get_nodes_by_type(NodeType.OBJECT)
        if not object_nodes or max(n.confidence for n in object_nodes) < pcfg.heavy_low_object_confidence:
            return True, "low_object_confidence"

        return False, "not_triggered"

    def record_run(self, state: TriggerState, step: int, reason: str) -> None:
        """Update ``TriggerState`` after a successful heavy-perception run."""
        state.last_heavy_step = step
        if reason == "coarse_summary_context":
            state.last_heavy_summary_step = step
