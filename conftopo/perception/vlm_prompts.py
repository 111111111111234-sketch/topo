"""VLM prompt templates and response parsing for navigation perception."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from conftopo.perception.heavy_perceiver import normalize_bbox, normalize_bbox_flexible


_JSON_SCHEMA_BLOCK = """\
{
  "room": {"label": "<room type>", "confidence": <0.0-1.0>},
  "objects": [
    {
      "label": "<object name>",
      "bbox": [x1, y1, x2, y2],
      "bbox_confidence": "<high|medium|low>",
      "visible": <true|false>,
      "visibility": "<visible|partially_visible|not_visible|unknown>",
      "bearing": "<left|center|right|left_front|right_front|front|unknown>",
      "range": "<close|very_near|near|mid|far|unknown>",
      "relation": ["<spatial relation phrases>"],
      "room_context": "<room or unknown>",
      "attributes": {
        "color": {"value": "<color>", "confidence": <0.0-1.0>},
        "shape": {"value": "<shape>", "confidence": <0.0-1.0>},
        "material": {"value": "<material>", "confidence": <0.0-1.0>},
        "size": {"value": "<tiny|small|medium|large>", "confidence": <0.0-1.0>},
        "state": {"value": "<open|closed|on|off|...>", "confidence": <0.0-1.0>},
        "description": {"value": "<short description>", "confidence": <0.0-1.0>}
      },
      "confidence": <0.0-1.0>
    }
  ],
  "goal_visible": <true|false>,
  "goal_match_confidence": <0.0-1.0>,
  "target_direction": "<left|left_front|center|right_front|right|unknown>",
  "target_visibility": "<clear|partial|heavily_occluded|uncertain|not_visible>",
  "apparent_scale": "<tiny|small|medium|large|very_large|unknown>",
  "relative_progress": "<closer|unchanged|farther|uncertain>",
  "stop_candidate": <true|false>,
  "recommended_action": "<turn_left|turn_right|move_forward|hold_and_verify|search|stop_candidate>",
  "goal_reason": "<why you think the goal is visible or not>",
  "portals": ["<door/opening descriptions>"],
  "scene_summary": "<one sentence describing the scene>",
  "uncertainty": <0.0-1.0>
}"""

_COMMON_RULES = """\
Rules:
- Every object you report MUST include a normalised bounding box [x1, y1, x2, y2] with values in [0, 1] (top-left origin).  If the bounds are uncertain, still provide your best approximate bbox, but set bbox_confidence=\"low\".  If the object is not visible, do not invent a bbox.
- For each object report its visual attributes (color, shape, material, size, state). Each attribute has a "value" string and a "confidence" in [0,1]. If uncertain about an attribute, omit it.
- Do not estimate metric distance. Use only close, very_near, near, mid, far, or unknown.
- Use relation for coarse context such as "beside cabinet", "near wall", "on table", "near doorway".
- room.label must be one of: kitchen, living room, bedroom, bathroom, \
hallway, dining room, office, garage, laundry room, closet, staircase, \
balcony, unknown.
- Keep scene_summary under 30 words.
- range is advisory visual context only. It is not a final stop decision.
- stop_candidate is visual evidence for a controller, not a STOP command.
- If no previous reference image is provided, relative_progress must be uncertain.
- Do NOT include any text outside the JSON object."""

EXPLORE_SYSTEM_PROMPT = f"""\
You are a navigation perception agent inside a house in EXPLORATION mode. \
Your job is to discover potential targets, landmarks, rooms, and spatial cues. \
Analyse the egocentric RGB image and answer strictly in JSON.

Your response MUST be a valid JSON object with exactly these keys:
{_JSON_SCHEMA_BLOCK}

{_COMMON_RULES}
 - Report any object you can identify, with a confidence score that reflects \
 your certainty. Use the full 0.0–1.0 range: 0.0 = uncertain, 1.0 = certain.
- Report rooms, landmarks, portals, and spatial relations generously — they \
help build a memory map.
- goal_visible should be true if the target is possibly present, even \
partially or at a distance.
- If the navigation goal object is visible, you MUST include it in objects[] \
with label exactly matching the goal name (e.g. goal "bed" -> label "bed").
- Set goal_visible=true whenever the goal object appears in objects[]."""

CONFIRM_SYSTEM_PROMPT = f"""\
You are a navigation perception agent inside a house in CONFIRMATION mode. \
The robot is close to where it expects to find the target. Your job is to \
verify whether the navigation goal is visible, close, and directly in front. \
Analyse the egocentric RGB image and answer in JSON.

Your response MUST be a valid JSON object with exactly these keys:
{_JSON_SCHEMA_BLOCK}

{_COMMON_RULES}
- Set goal_visible=true if the target object appears to be present in the scene. \
Set goal_visible=false only when the object is definitely absent.
- bbox is used by the downstream controller as one piece of stop-verification evidence.  Do not enlarge or fabricate bbox to justify stopping.  Set stop_candidate=true ONLY when the target is clearly identifiable, \
centered, and sufficiently close.  When uncertain, set stop_candidate to false.
- Set target_visibility to "clear" if the target is at least partially \
unobstructed.
- Set target_direction to "center" or "front" if the target is roughly \
centered, even if slightly off-center.
- Pay special attention to range and bearing: the robot needs to know if the \
target is near and in front.
- Compare the current image with the labelled previous reference image when \
one is provided. Judge whether the target appears closer, unchanged, or farther.
- If the target is not ready as a stop candidate, recommend one local action: \
turn_left, turn_right, move_forward, hold_and_verify, or search."""

SYSTEM_PROMPT = EXPLORE_SYSTEM_PROMPT


def build_user_prompt(
    goal_text: str,
    context: Optional[str] = None,
    mode: str = "explore",
    has_previous_image: bool = False,
    action_history: Optional[List[str]] = None,
) -> str:
    """Construct the user-turn text sent alongside the image."""
    parts = [f"Current navigation goal: {goal_text}"]
    if context:
        parts.append(f"Context: {context}")
    if action_history:
        parts.append("Recent robot actions: " + " -> ".join(action_history[-5:]))
    if mode == "confirm":
        comparison = (
            "A PREVIOUS reference image is provided before the CURRENT image. "
            "Use it only for relative_progress."
            if has_previous_image
            else
            "No previous reference image is available. "
            "Set relative_progress to uncertain."
        )
        parts.append(
            "The robot believes the target may be nearby. "
            "Carefully verify whether the goal object is visible, where it is, "
            "whether it is a visual stop candidate, and which local action is "
            f"appropriate. {comparison} Return the JSON perception report."
        )
    else:
        parts.append(
            "Analyse the image and return the JSON perception report. "
            "If the goal object is visible, include it in objects[] with the exact goal label."
        )
    return "\n".join(parts)


def get_system_prompt(mode: str = "explore") -> str:
    """Return the appropriate system prompt for the given perception mode."""
    if mode == "confirm":
        return CONFIRM_SYSTEM_PROMPT
    return EXPLORE_SYSTEM_PROMPT


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def parse_vlm_json(raw: str) -> Dict[str, Any]:
    """Best-effort parse of VLM output into the canonical schema dict.

    Tries, in order:
    1. Direct ``json.loads`` on the raw string.
    2. Extract JSON from a fenced code block.
    3. Extract the first ``{...}`` substring.

    Falls back to a safe empty dict on failure.
    """
    text = raw.strip()

    for candidate in _candidates(text):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return _normalise(data)
        except (json.JSONDecodeError, ValueError):
            continue

    return _fallback()


def _candidates(text: str):
    yield text
    m = _JSON_BLOCK_RE.search(text)
    if m:
        yield m.group(1)
    m = _BARE_JSON_RE.search(text)
    if m:
        yield m.group(1)


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist with correct types."""
    room = data.get("room") or {}
    if not isinstance(room, dict):
        room = {}
    objects_raw = data.get("objects") or []
    if not isinstance(objects_raw, list):
        objects_raw = []
    objects: List[Dict[str, Any]] = []
    for o in objects_raw:
        if isinstance(o, dict) and "label" in o:
            relation = o.get("relation", o.get("spatial_relation", []))
            if isinstance(relation, str):
                relation = [relation]
            if not isinstance(relation, list):
                relation = []
            raw_attrs = o.get("attributes") or {}
            attributes = {}
            if isinstance(raw_attrs, dict):
                for _ak in ("color", "shape", "material", "size", "state", "description"):
                    _av = raw_attrs.get(_ak)
                    if isinstance(_av, dict) and "value" in _av:
                        attributes[_ak] = {
                            "value": str(_av.get("value", "")),
                            "confidence": _float01(_av.get("confidence", 0.5), 0.0),
                        }
                    elif isinstance(_av, str) and _av.strip():
                        attributes[_ak] = {"value": _av.strip(), "confidence": 0.5}
            objects.append({
                "label": str(o.get("label", "")),
                "bbox": normalize_bbox_flexible(o.get("bbox")),
                "visible": bool(o.get("visible", True)),
                "visibility": _enum(
                    o.get("visibility", "visible" if o.get("visible", True) else "not_visible"),
                    {"visible", "partially_visible", "not_visible", "unknown"},
                    "unknown",
                ),
                "bearing": _enum(
                    o.get("bearing", "unknown"),
                    {"left", "center", "right", "left_front", "right_front", "front", "unknown"},
                    "unknown",
                ),
                "range": _enum(
                    o.get("range", o.get("range_bin", "unknown")),
                    {"close", "very_near", "near", "mid", "far", "unknown"},
                    "unknown",
                ),
                "relation": [str(x) for x in relation],
                "room_context": str(o.get("room_context", "")) or None,
                "bbox_confidence": _enum(
                    o.get("bbox_confidence", "medium"),
                    {"high", "medium", "low"},
                    "medium",
                ),
                "attributes": attributes,
                "confidence": _float01(o.get("confidence", 0.0), 0.0),
            })
    return {
        "room": {
            "label": str(room.get("label", "unknown")),
            "confidence": _float01(room.get("confidence", 0.0), 0.0),
        },
        "objects": objects,
        "goal_visible": bool(data.get("goal_visible", False)),
        "goal_match_confidence": _float01(
            data.get("goal_match_confidence", 0.0), 0.0,
        ),
        "target_direction": _enum(
            data.get("target_direction", "unknown"),
            {"left", "left_front", "center", "right_front", "right", "unknown"},
            "unknown",
        ),
        "target_visibility": _enum(
            data.get("target_visibility", "not_visible"),
            {"clear", "partial", "heavily_occluded", "uncertain", "not_visible"},
            "uncertain",
        ),
        "apparent_scale": _enum(
            data.get("apparent_scale", "unknown"),
            {"tiny", "small", "medium", "large", "very_large", "unknown"},
            "unknown",
        ),
        "relative_progress": _enum(
            data.get("relative_progress", "uncertain"),
            {"closer", "unchanged", "farther", "uncertain"},
            "uncertain",
        ),
        "stop_candidate": bool(data.get("stop_candidate", False)),
        "recommended_action": _enum(
            data.get("recommended_action", "search"),
            {
                "turn_left", "turn_right", "move_forward",
                "hold_and_verify", "search", "stop_candidate",
            },
            "search",
        ),
        "goal_reason": str(data.get("goal_reason", "")),
        "portals": [str(p) for p in (data.get("portals") or [])],
        "scene_summary": str(data.get("scene_summary", "")),
        "uncertainty": float(data.get("uncertainty", 0.5)),
    }


def _fallback() -> Dict[str, Any]:
    return {
        "room": {"label": "unknown", "confidence": 0.0},
        "objects": [],
        "goal_visible": False,
        "goal_match_confidence": 0.0,
        "target_direction": "unknown",
        "target_visibility": "not_visible",
        "apparent_scale": "unknown",
        "relative_progress": "uncertain",
        "stop_candidate": False,
        "recommended_action": "search",
        "goal_reason": "parse_error",
        "portals": [],
        "scene_summary": "",
        "uncertainty": 1.0,
    }


def _enum(value: Any, allowed: set, default: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    return text if text in allowed else default


def _float01(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
