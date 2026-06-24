"""Multi-factor confidence and task scoring for semantic topological memory."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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
    negative_evidence: float = 0.0     # weak negative: low-confidence / uncertain observations
    strong_negative_evidence: float = 0.0  # strong negative: explicit VLM rejection / failed verification
    multi_frame_consistency: int = 0   # consecutive consistent frames (fresh observations only)
    attribute_confidence: float = 0.0  # avg confidence of VLM-extracted attributes (color, shape, ...)


# Default weights for semantic nodes (object / landmark / room)
SEMANTIC_WEIGHTS = {
    "detection": 0.25,
    "multi_view": 0.20,
    "task_relevance": 0.15,
    "room_prior": 0.10,
    "attribute": 0.10,
    "negative_evidence": -0.05,
    "strong_negative_evidence": -0.15,
    "time_decay": 0.10,
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

    C_semantic = (w1*detection + w2*multi_view + w3*relevance
                  + w4*room_prior + w5*attribute
                  - w6*negative_evidence) / sum(positive_w) * time_decay
    """
    multi_view_bonus = min(1.0, factors.multi_view_count / 3.0)
    frame_bonus = min(1.0, factors.multi_frame_consistency / 5.0) * 0.1
    attribute_bonus = factors.attribute_confidence * 0.1

    staleness_decay = factors.time_decay * (0.995 ** max(0, factors.staleness_steps))
    weak_neg = min(1.0, factors.redundancy_penalty) * 0.05
    strong_neg = min(1.0, factors.negative_evidence) * 0.10 + min(1.0, factors.strong_negative_evidence) * 0.20
    conflict_pen = min(1.0, factors.conflict_penalty) * 0.10
    penalty = weak_neg + strong_neg + conflict_pen

    base = (
        SEMANTIC_WEIGHTS["detection"] * factors.detection_score
        + SEMANTIC_WEIGHTS["multi_view"] * multi_view_bonus
        + SEMANTIC_WEIGHTS["task_relevance"] * factors.task_relevance
        + SEMANTIC_WEIGHTS["room_prior"] * factors.room_prior_score
    )
    max_positive = sum(v for k, v in SEMANTIC_WEIGHTS.items() if v > 0)
    # staleness_decay wraps everything, not just attribute_bonus
    raw = (base / max_positive) + frame_bonus + attribute_bonus - penalty
    score = max(factors.detection_score, raw) * staleness_decay
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


def _clean_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _attribute_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    return _clean_text(value)


class AttributeMatcher:
    """Compare GoalGraph attributes with VLM object attributes.

    Attribute confidence answers "how reliable was the VLM attribute output".
    Attribute match answers "does the observed attribute satisfy the task".
    """

    @staticmethod
    def goal_attributes_to_dict(attributes: Any) -> Dict[str, str]:
        if attributes is None:
            return {}
        if isinstance(attributes, dict):
            return {
                _clean_text(k): _attribute_value(v)
                for k, v in attributes.items()
                if _clean_text(k) and _attribute_value(v)
            }
        result: Dict[str, str] = {}
        for item in _as_list(attributes):
            text = _clean_text(item)
            if not text:
                continue
            # LLM GoalGraph often puts free attributes in a list. Treat them
            # as values that can match any observed attribute field.
            result[f"attr_{len(result)}"] = text
        return result

    @classmethod
    def score(cls, goal_attributes: Any, observed_attributes: Any) -> float:
        goal_attrs = cls.goal_attributes_to_dict(goal_attributes)
        if not goal_attrs:
            return 0.0
        if not isinstance(observed_attributes, dict) or not observed_attributes:
            return 0.0

        observed_values = {
            _clean_text(k): _attribute_value(v)
            for k, v in observed_attributes.items()
            if _clean_text(k) and _attribute_value(v)
        }
        if not observed_values:
            return 0.0

        matches = 0.0
        for goal_key, goal_value in goal_attrs.items():
            if goal_key in observed_values:
                obs_value = observed_values[goal_key]
                if goal_value == obs_value:
                    matches += 1.0
                elif goal_value in obs_value or obs_value in goal_value:
                    matches += 0.5
                continue
            if any(goal_value == obs or goal_value in obs or obs in goal_value for obs in observed_values.values()):
                matches += 1.0
        return float(np.clip(matches / max(len(goal_attrs), 1), 0.0, 1.0))


class RelationScorer:
    """Score GoalGraph relations against known map/observation relation evidence."""

    @staticmethod
    def _reference(relation: Any) -> str:
        if isinstance(relation, dict):
            return _clean_text(relation.get("reference", ""))
        return _clean_text(getattr(relation, "reference", ""))

    @staticmethod
    def _relation_type(relation: Any) -> str:
        if isinstance(relation, dict):
            return _clean_text(relation.get("relation_type", ""))
        return _clean_text(getattr(relation, "relation_type", ""))

    @classmethod
    def score(
        cls,
        *,
        goal_relations: Any = None,
        goal_landmarks: Any = None,
        observed_relations: Any = None,
        observed_room: str = "",
        map_relation_labels: Optional[List[str]] = None,
        room_prior: Any = None,
    ) -> float:
        references = []
        relation_types = []
        for relation in _as_list(goal_relations):
            ref = cls._reference(relation)
            if ref:
                references.append(ref)
                relation_types.append(cls._relation_type(relation))
        references.extend(_clean_text(x) for x in _as_list(goal_landmarks) if _clean_text(x))
        references = list(dict.fromkeys(references))

        evidence = [_clean_text(x) for x in _as_list(observed_relations)]
        evidence.extend(_clean_text(x) for x in _as_list(map_relation_labels))
        evidence = [x for x in evidence if x]

        score = 0.0
        slots = 0
        for ref in references:
            slots += 1
            if any(ref in ev or ev in ref for ev in evidence):
                score += 1.0

        observed_room_clean = _clean_text(observed_room)
        for room in _as_list(room_prior):
            room_text = _clean_text(room)
            if not room_text:
                continue
            slots += 1
            if observed_room_clean and (room_text in observed_room_clean or observed_room_clean in room_text):
                score += 1.0

        # A first-pass near(reference) implementation: if the caller provided
        # map_relation_labels containing the reference, count it as satisfied.
        if any(rt == "near" for rt in relation_types) and references and evidence:
            near_hits = sum(1 for ref in references if any(ref in ev or ev in ref for ev in evidence))
            if near_hits:
                score += min(1.0, near_hits / max(len(references), 1))
                slots += 1

        if slots <= 0:
            return 0.0
        return float(np.clip(score / slots, 0.0, 1.0))


def update_memory_state(
    *,
    confidence: float,
    current_state: str = "candidate",
    confirmed: bool = False,
    preserved: bool = False,
    rejected: bool = False,
    expired: bool = False,
    strong_negative_evidence: float = 0.0,
    multi_view_count: int = 0,
    target_relevance: float = 0.0,
) -> str:
    """State transition for confidence-aware memory lifecycle."""
    if expired or confidence <= 0.02:
        return "expired"
    if rejected or strong_negative_evidence >= 0.8:
        return "rejected"
    if preserved:
        return "preserved"
    if confirmed or confidence >= 0.55 or multi_view_count >= 2:
        return "confirmed"
    if confidence >= 0.25 or target_relevance > 0.0:
        return "candidate"
    if current_state == "confirmed" and confidence >= 0.18:
        return "candidate"
    return "expired"



def compute_task_score(
    *,
    target_object: str = "",
    attributes: Any = None,
    room_prior: list = None,
    landmarks: list = None,
    relations: Any = None,
    observed_label: str = "",
    observed_attributes: Any = None,
    observed_room: str = "",
    observed_relations: list = None,
    map_relation_labels: Optional[List[str]] = None,
    history_success_count: int = 0,
    history_fail_count: int = 0,
) -> float:
    """Unified task-relevance score for a node given the current goal.

    1. label_match (exact / partial / none = 1.0 / 0.5 / 0.0)
    2. attribute_match (how many VLM attributes match goal attributes)
    3. room_prior_match (observed room in goal room_prior list)
    4. landmark_relation_match (observed relations mention goal landmarks)
    5. history_bonus (successful past approaches for same goal)

    Returns a float in [0, 1] where 1.0 = maximally task-relevant.
    """
    if attributes is None:
        attributes = {}
    if room_prior is None:
        room_prior = []
    if landmarks is None:
        landmarks = []
    if observed_attributes is None:
        observed_attributes = {}
    if observed_relations is None:
        observed_relations = []

    score = 0.0
    total_weight = 0.0

    # 1. Label match
    target = str(target_object).lower().strip()
    observed = str(observed_label).lower().strip()
    w_label = 0.40
    total_weight += w_label
    if target and target == observed:
        score += w_label * 1.0
    elif target and (target in observed or observed in target):
        score += w_label * 0.5

    # 2. Attribute match
    w_attr = 0.15
    total_weight += w_attr
    attr_score = AttributeMatcher.score(attributes, observed_attributes)
    score += w_attr * attr_score

    # 3. Room prior match
    w_room = 0.20
    total_weight += w_room
    if room_prior and observed_room:
        observed_room_lower = str(observed_room).lower().strip()
        for rp in room_prior:
            if rp.lower() in observed_room_lower or observed_room_lower in rp.lower():
                score += w_room * 1.0
                break

    # 4. Landmark / relation match
    w_relation = 0.15
    total_weight += w_relation
    relation_score = RelationScorer.score(
        goal_relations=relations,
        goal_landmarks=landmarks,
        observed_relations=observed_relations,
        observed_room=observed_room,
        map_relation_labels=map_relation_labels,
        room_prior=room_prior,
    )
    score += w_relation * relation_score

    # 5. History success bonus
    w_history = 0.10
    total_weight += w_history
    total_attempts = history_success_count + history_fail_count
    if total_attempts > 0:
        success_rate = history_success_count / total_attempts
        score += w_history * success_rate

    return float(np.clip(score / max(total_weight, 1e-8), 0.0, 1.0))


def temporal_decay(confidence: float, steps_elapsed: int, decay_rate: float = 0.95) -> float:
    """Apply temporal decay to confidence."""
    return float(confidence * (decay_rate ** steps_elapsed))
