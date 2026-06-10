"""Landmark role taxonomy for the room-centric topo map.

Two roles:
- ``structural``: door / corridor end / entrance / stair / opening / window /
  fixed counter / passage / arch. These are spatial anchors that link rooms
  together. They are eligible for the long-term spatial skeleton.
- ``semantic``: sofa / bed / sink / table / plant ... room-interior anchors
  useful for local localization or goal retrieval, but **not** part of the
  long-term skeleton by default.

Promotion from ``object`` to ``landmark`` is intentionally restrictive:
high confidence and multi-view stability are required, plus either a
structural label or strong task relevance (room anchoring / goal retrieval).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Label prior. Lower-case, matched as substring on the node label so that
# "front door" / "kitchen door" / "stair top" all classify as structural.
STRUCTURAL_LANDMARK_LABEL_TOKENS: Tuple[str, ...] = (
    "door",
    "doorway",
    "entrance",
    "exit",
    "corridor",
    "hallway",
    "passage",
    "opening",
    "arch",
    "archway",
    "gate",
    "gateway",
    "stair",
    "staircase",
    "elevator",
    "window",
    "counter",
)

# Labels we explicitly do NOT want to treat as structural even if the token
# happens to match (e.g. "table" contains no structural token, but listed
# here for documentation completeness / future extension).
SEMANTIC_LANDMARK_HINT_TOKENS: Tuple[str, ...] = (
    "sofa",
    "couch",
    "bed",
    "sink",
    "toilet",
    "table",
    "chair",
    "plant",
    "tv",
    "fridge",
    "cabinet",
    "lamp",
    "rack",
    "shelf",
    "vase",
    "mirror",
    "towel",
)


def _normalize_label(label: Any) -> str:
    if label is None:
        return ""
    return str(label).strip().lower()


def is_structural_label(label: Any) -> bool:
    """Return True when the label looks like a structural anchor."""
    text = _normalize_label(label)
    if not text:
        return False
    for token in STRUCTURAL_LANDMARK_LABEL_TOKENS:
        if token in text:
            return True
    return False


def classify_landmark_role(
    label: Any,
    attrs: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify a landmark as ``structural`` or ``semantic``.

    Explicit ``landmark_role`` in attrs wins. Otherwise the label is matched
    against :data:`STRUCTURAL_LANDMARK_LABEL_TOKENS`. Anything else is
    treated as ``semantic``.
    """
    attrs = attrs or {}
    explicit = attrs.get("landmark_role")
    if explicit in ("structural", "semantic"):
        return str(explicit)
    return "structural" if is_structural_label(label) else "semantic"


# --- Object -> landmark promotion gating ----------------------------------

# Defaults are intentionally conservative. Callers (DynamicTopoMap) may
# override via config in later phases.
DEFAULT_STRUCTURAL_MIN_CONFIDENCE = 0.45
DEFAULT_STRUCTURAL_MIN_MULTIVIEW = 1
DEFAULT_SEMANTIC_MIN_CONFIDENCE = 0.6
DEFAULT_SEMANTIC_MIN_MULTIVIEW = 2


def can_promote_object_to_landmark(
    *,
    label: Any,
    confidence: float,
    multi_view_count: int,
    target_relevance: float = 0.0,
    source: str = "",
    has_bbox: bool = False,
    structural_min_confidence: float = DEFAULT_STRUCTURAL_MIN_CONFIDENCE,
    structural_min_multiview: int = DEFAULT_STRUCTURAL_MIN_MULTIVIEW,
    semantic_min_confidence: float = DEFAULT_SEMANTIC_MIN_CONFIDENCE,
    semantic_min_multiview: int = DEFAULT_SEMANTIC_MIN_MULTIVIEW,
) -> Tuple[bool, str, str]:
    """Decide whether an object node may be promoted to a landmark.

    Returns ``(allowed, role, reason)``. ``role`` is one of
    ``"structural"`` or ``"semantic"``; ``reason`` is a short tag suitable
    for logging / debug.

    Rules:

    1. Structural labels (door, corridor, stair, ...) are the cheap path:
       moderate confidence is enough because they are room anchors.
    2. Semantic labels must clear a higher bar: high confidence **and**
       multi-view stable **and** either task-relevant or backed by a
       heavy detection / bbox history.

    The function does NOT mutate attributes; the caller is responsible for
    writing ``landmark_role`` back to the node.
    """
    role = "structural" if is_structural_label(label) else "semantic"
    conf = float(confidence or 0.0)
    mv = int(multi_view_count or 0)
    rel = float(target_relevance or 0.0)
    heavy = str(source or "").lower() in ("groundingdino", "heavy")

    if role == "structural":
        if conf < structural_min_confidence:
            return False, role, "structural_low_confidence"
        if mv < structural_min_multiview:
            return False, role, "structural_no_view"
        return True, role, "structural_ok"

    # semantic
    if conf < semantic_min_confidence:
        return False, role, "semantic_low_confidence"
    if mv < semantic_min_multiview and rel <= 0.0 and not heavy:
        return False, role, "semantic_unstable"
    if rel <= 0.0 and not heavy and not has_bbox:
        return False, role, "semantic_no_evidence"
    return True, role, "semantic_ok"
