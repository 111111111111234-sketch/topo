"""Multi-factor confidence system for semantic topological memory."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to a node's confidence."""
    detection_score: float = 0.5    # VLM/CLIP detection confidence
    multi_view_count: int = 1       # how many views confirmed this node
    task_relevance: float = 0.0     # relevance to current goal
    time_decay: float = 1.0         # temporal decay factor
    room_prior_score: float = 0.0   # whether the object fits the current/goal room
    redundancy_penalty: float = 0.0 # repeated weak observations
    conflict_penalty: float = 0.0   # nearby conflicting labels/geometry
    staleness_steps: int = 0        # steps since last observation
    # For waypoint nodes
    navigability: float = 0.5       # confirmed navigable?
    execution_success: float = 0.0  # successfully visited?
    collision_history: float = 0.0  # collision/failure count
    backtrack_frequency: float = 0.0  # how often agent backtracks from here


# Default weights for semantic nodes (object / landmark / room)
SEMANTIC_WEIGHTS = {
    "detection": 0.3,
    "multi_view": 0.25,
    "task_relevance": 0.2,
    "room_prior": 0.15,
    "time_decay": 0.1,
}

# Default weights for topological nodes (waypoint / frontier)
TOPO_WEIGHTS = {
    "navigability": 0.3,
    "execution_success": 0.25,
    "collision_penalty": 0.2,
    "backtrack_penalty": 0.1,
    "time_decay": 0.15,
}


def compute_semantic_confidence(factors: ConfidenceFactors) -> float:
    """Compute confidence for semantic nodes (object/landmark/room).

    C_semantic = (w1*detection + w2*multi_view + w3*relevance) / sum(w) * time_decay
    """
    multi_view_bonus = min(1.0, factors.multi_view_count / 3.0)

    staleness_decay = factors.time_decay * (0.995 ** max(0, factors.staleness_steps))
    penalty = 0.05 * min(1.0, factors.redundancy_penalty) + 0.1 * min(1.0, factors.conflict_penalty)

    base = (
        SEMANTIC_WEIGHTS["detection"] * factors.detection_score
        + SEMANTIC_WEIGHTS["multi_view"] * multi_view_bonus
        + SEMANTIC_WEIGHTS["task_relevance"] * factors.task_relevance
        + SEMANTIC_WEIGHTS["room_prior"] * factors.room_prior_score
    )
    max_base = (
        SEMANTIC_WEIGHTS["detection"]
        + SEMANTIC_WEIGHTS["multi_view"]
        + SEMANTIC_WEIGHTS["task_relevance"]
        + SEMANTIC_WEIGHTS["room_prior"]
    )
    score = max(factors.detection_score, base / max_base) * staleness_decay - penalty

    return float(np.clip(score, 0.0, 1.0))


def compute_topo_confidence(factors: ConfidenceFactors) -> float:
    """Compute confidence for topological nodes (waypoint/frontier).

    C_topo = (w1 * navigability + w2 * execution_success
              - w3 * collision - w4 * backtrack) * time_decay
    Weights sum to ~1 for positive terms, penalties subtract.
    """
    base = (
        TOPO_WEIGHTS["navigability"] * factors.navigability
        + TOPO_WEIGHTS["execution_success"] * factors.execution_success
    )
    penalty = (
        TOPO_WEIGHTS["collision_penalty"] * min(1.0, factors.collision_history / 3.0)
        + TOPO_WEIGHTS["backtrack_penalty"] * min(1.0, factors.backtrack_frequency / 3.0)
    )
    # Normalize: max possible base is nav_w + exec_w = 0.55, scale to [0,1]
    max_base = TOPO_WEIGHTS["navigability"] + TOPO_WEIGHTS["execution_success"]
    score = (base - penalty) / max_base * factors.time_decay

    return float(np.clip(score, 0.0, 1.0))


def update_on_observation(
    current_confidence: float,
    detection_score: float,
    is_consistent: bool = True,
    learning_rate: float = 0.3,
) -> float:
    """Update confidence when node is re-observed.

    If observation is consistent with existing belief → confidence increases.
    If inconsistent → confidence decreases.
    """
    if is_consistent:
        target = max(current_confidence, detection_score)
        new_conf = current_confidence + learning_rate * (target - current_confidence)
    else:
        new_conf = current_confidence * (1.0 - learning_rate * 0.5)

    return float(np.clip(new_conf, 0.0, 1.0))


def temporal_decay(confidence: float, steps_elapsed: int, decay_rate: float = 0.95) -> float:
    """Apply temporal decay to confidence."""
    return float(confidence * (decay_rate ** steps_elapsed))
