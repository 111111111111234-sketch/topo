"""Tests for tiered SEARCH-stage VLM triggering in goat_agent_1."""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent_1 import AgentState, ClipHint, GOATState, PerceptionModule
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import GoalNode


def _module() -> PerceptionModule:
    config = ConfTopoConfig()
    config.perception.backend = "vlm"
    return PerceptionModule(config)


def _goal() -> GoalNode:
    return GoalNode(target_object="rack", target_embedding=np.zeros(512, dtype=np.float32))


def test_initial_search_scan_triggers_immediately():
    module = _module()
    state = AgentState(mode=GOATState.SEARCH)
    hint = ClipHint(best_goal_sim=0.0)

    assert module.should_trigger_vlm(
        clip_hint=hint, goal=_goal(), state=state, step_id=0,
    )
    assert module.last_vlm_reason == "initial_search_scan"


def test_early_search_scan_every_five_steps():
    module = _module()
    state = AgentState(mode=GOATState.SEARCH)
    hint = ClipHint(best_goal_sim=0.0)
    module.last_vlm_step = 0

    assert not module.should_trigger_vlm(
        clip_hint=hint, goal=_goal(), state=state, step_id=3,
    )
    assert module.last_vlm_reason == "skip"

    assert module.should_trigger_vlm(
        clip_hint=hint, goal=_goal(), state=state, step_id=5,
    )
    assert module.last_vlm_reason == "early_search_scan"


def test_periodic_search_every_eight_steps_after_early_window():
    module = _module()
    state = AgentState(mode=GOATState.SEARCH)
    hint = ClipHint(best_goal_sim=0.0)
    module.last_vlm_step = 30

    assert not module.should_trigger_vlm(
        clip_hint=hint, goal=_goal(), state=state, step_id=36,
    )
    assert module.last_vlm_reason == "skip"

    assert module.should_trigger_vlm(
        clip_hint=hint, goal=_goal(), state=state, step_id=38,
    )
    assert module.last_vlm_reason == "periodic_search"


def test_track_state_refreshes_every_two_steps():
    module = _module()
    state = AgentState(mode=GOATState.TRACK)
    hint = ClipHint(best_goal_sim=0.0)
    module.last_vlm_step = 10

    assert module.should_trigger_vlm(
        clip_hint=hint, goal=_goal(), state=state, step_id=12,
    )
    assert module.last_vlm_reason == "state_refresh"
