"""VLM prompt templates and response parsing for navigation perception."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """\
You are a navigation perception agent inside a house. You analyse a single \
egocentric RGB image and answer strictly in JSON.

Your response MUST be a valid JSON object with exactly these keys:
{
  "room": {"label": "<room type>", "confidence": <0.0-1.0>},
  "objects": [
    {"label": "<object name>", "bbox": [x1, y1, x2, y2], "confidence": <0.0-1.0>}
  ],
  "goal_visible": <true|false>,
  "goal_reason": "<why you think the goal is visible or not>",
  "portals": ["<door/opening descriptions>"],
  "scene_summary": "<one sentence describing the scene>",
  "uncertainty": <0.0-1.0>
}

Rules:
- bbox coordinates are normalised to [0,1] (top-left origin).
- Only include objects you are confident about (confidence >= 0.3).
- room.label must be one of: kitchen, living room, bedroom, bathroom, \
hallway, dining room, office, garage, laundry room, closet, staircase, \
balcony, unknown.
- Keep scene_summary under 30 words.
- Do NOT include any text outside the JSON object.\
"""


def build_user_prompt(goal_text: str, context: Optional[str] = None) -> str:
    """Construct the user-turn text sent alongside the image."""
    parts = [f"Current navigation goal: {goal_text}"]
    if context:
        parts.append(f"Context: {context}")
    parts.append(
        "Analyse the image and return the JSON perception report."
    )
    return "\n".join(parts)


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
            objects.append({
                "label": str(o.get("label", "")),
                "bbox": [float(v) for v in o.get("bbox", [0, 0, 0, 0])],
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
