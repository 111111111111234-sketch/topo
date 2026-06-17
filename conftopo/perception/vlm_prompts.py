"""VLM prompt templates and response parsing for navigation perception."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from conftopo.perception.heavy_perceiver import normalize_bbox


_JSON_SCHEMA_BLOCK = """\
{
  "room": {"label": "<room type>", "confidence": <0.0-1.0>},
  "objects": [
    {
      "label": "<object name>",
      "bbox": [x1, y1, x2, y2],
      "visible": <true|false>,
      "visibility": "<visible|partially_visible|not_visible|unknown>",
      "bearing": "<left|center|right|left_front|right_front|front|unknown>",
      "range": "<near|mid|far|unknown>",
      "relation": ["<spatial relation phrases>"],
      "room_context": "<room or unknown>",
      "confidence": <0.0-1.0>
    }
  ],
  "goal_visible": <true|false>,
  "goal_reason": "<why you think the goal is visible or not>",
  "portals": ["<door/opening descriptions>"],
  "scene_summary": "<one sentence describing the scene>",
  "uncertainty": <0.0-1.0>
}"""

_COMMON_RULES = """\
Rules:
- bbox coordinates are normalised to [0,1] (top-left origin).
- Do not estimate metric distance. Use only the discrete range values near, mid, far, or unknown.
- Use relation for coarse context such as "beside cabinet", "near wall", "on table", "near doorway".
- room.label must be one of: kitchen, living room, bedroom, bathroom, \
hallway, dining room, office, garage, laundry room, closet, staircase, \
balcony, unknown.
- Keep scene_summary under 30 words.
- Do NOT include any text outside the JSON object."""

EXPLORE_SYSTEM_PROMPT = f"""\
You are a navigation perception agent inside a house in EXPLORATION mode. \
Your job is to discover potential targets, landmarks, rooms, and spatial cues. \
Analyse the egocentric RGB image and answer strictly in JSON.

Your response MUST be a valid JSON object with exactly these keys:
{_JSON_SCHEMA_BLOCK}

{_COMMON_RULES}
- Include any object that might be relevant to the navigation goal, even if \
you are not fully certain (confidence >= 0.3).
- Report rooms, landmarks, portals, and spatial relations generously — they \
help build a memory map.
- goal_visible should be true if the target is possibly present, even \
partially or at a distance."""

CONFIRM_SYSTEM_PROMPT = f"""\
You are a navigation perception agent inside a house in CONFIRMATION mode. \
The robot is close to where it expects to find the target. Your job is to \
strictly verify whether the navigation goal is actually visible, close, and \
directly in front. Analyse the egocentric RGB image and answer strictly in JSON.

Your response MUST be a valid JSON object with exactly these keys:
{_JSON_SCHEMA_BLOCK}

{_COMMON_RULES}
- Only report objects you are confident about (confidence >= 0.5).
- goal_visible must be true ONLY if the target object is clearly identifiable \
in the image. If uncertain, set goal_visible to false.
- Pay special attention to range and bearing: the robot needs to know if the \
target is near and in front.
- If you are not sure the object matches the goal, set confidence lower and \
goal_visible to false."""

SYSTEM_PROMPT = EXPLORE_SYSTEM_PROMPT


def build_user_prompt(
    goal_text: str,
    context: Optional[str] = None,
    mode: str = "explore",
) -> str:
    """Construct the user-turn text sent alongside the image."""
    parts = [f"Current navigation goal: {goal_text}"]
    if context:
        parts.append(f"Context: {context}")
    if mode == "confirm":
        parts.append(
            "The robot believes the target may be nearby. "
            "Carefully verify whether the goal object is visible, its range, "
            "and its bearing. Return the JSON perception report."
        )
    else:
        parts.append(
            "Analyse the image and return the JSON perception report."
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
            objects.append({
                "label": str(o.get("label", "")),
                "bbox": normalize_bbox(o.get("bbox")),
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
                    {"near", "mid", "far", "unknown"},
                    "unknown",
                ),
                "relation": [str(x) for x in relation],
                "room_context": str(o.get("room_context", "")) or None,
                "confidence": float(o.get("confidence", 0.0)),
            })
    return {
        "room": {
            "label": str(room.get("label", "unknown")),
            "confidence": float(room.get("confidence", 0.0)),
        },
        "objects": objects,
        "goal_visible": bool(data.get("goal_visible", False)),
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
        "goal_reason": "parse_error",
        "portals": [],
        "scene_summary": "",
        "uncertainty": 1.0,
    }


def _enum(value: Any, allowed: set, default: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    return text if text in allowed else default
