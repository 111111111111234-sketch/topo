"""VLM-backed perceiver: wraps any VLMBackendBase into a PerceptionReport producer."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from conftopo.perception.heavy_perceiver import ObjectObservation, normalize_bbox
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import VLMBackendBase


def _calibrate_vlm_confidence(conf: float) -> float:
    """VLM outputs systematically conservative confidence scores (typically
    0.3-0.6 for correct detections).  Calibrate to better occupy the 0-1
    range so downstream confidence computation has a wider dynamic range.
    """
    if conf <= 0.0:
        return 0.0
    return min(1.0, conf * 1.4 + 0.05)


class VLMPerceiver:
    """High-level perceiver that calls a VLM backend and returns a unified report.

    Usage::

        perceiver = VLMPerceiver(backend=Qwen3VLBackend(...))
        report = perceiver.perceive(rgb, "Find the kitchen sink", step_id=42)
    """

    def __init__(self, backend: VLMBackendBase):
        self._backend = backend

    def perceive(
        self,
        rgb: np.ndarray,
        goal_text: str,
        visual_embed: Optional[np.ndarray] = None,
        step_id: int = 0,
        context: Optional[str] = None,
        mode: str = "explore",
        previous_rgb: Optional[np.ndarray] = None,
        action_history: Optional[Sequence[str]] = None,
    ) -> PerceptionReport:
        """Query the VLM and convert the response to a PerceptionReport.

        *mode*: ``"explore"`` for discovery or ``"confirm"`` for strict
        stop verification.  Selects the system prompt sent to the VLM.
        """
        if not getattr(self._backend, 'supports_multi_image', True):
            previous_rgb = None
        raw = self._backend.query(
            rgb,
            goal_text,
            context=context,
            mode=mode,
            previous_rgb=previous_rgb,
            action_history=action_history,
        )
        if mode == "confirm" and previous_rgb is None:
            raw = dict(raw)
            raw["relative_progress"] = "uncertain"
        return self._build_report(raw, visual_embed, step_id)

    @staticmethod
    def _build_report(
        data: dict,
        visual_embed: Optional[np.ndarray],
        step_id: int,
    ) -> PerceptionReport:
        room = data.get("room") or {}
        room_label = str(room.get("label", "unknown"))
        room_conf = float(room.get("confidence", 0.0))

        objects = []
        for o in data.get("objects", []):
            relation = o.get("relation", o.get("spatial_relation", []))
            if isinstance(relation, str):
                relation = [relation]
            if relation is None:
                relation = []
            cal_conf = _calibrate_vlm_confidence(float(o.get("confidence", 0.0)))
            objects.append(ObjectObservation(
                label=str(o.get("label", "")),
                bbox=normalize_bbox(o.get("bbox")),
                confidence=cal_conf,
                source="vlm",
                step_id=step_id,
                visible=bool(o.get("visible", True)),
                visibility=str(o.get("visibility", "unknown")),
                bearing=str(o.get("bearing", "unknown")),
                range_bin=str(o.get("range", o.get("range_bin", "unknown"))),
                spatial_relation=[str(x) for x in relation],
                bbox_confidence=str(o.get("bbox_confidence", "medium")),
                room_context=o.get("room_context"),
                attributes=dict(o.get("attributes") or {}),
            ))

        return PerceptionReport(
            room_label=room_label,
            room_confidence=room_conf,
            room_scores=[(room_label, room_conf)] if room_label != "unknown" else [],
            objects=objects,
            scene_summary=str(data.get("scene_summary", "")),
            goal_visible=bool(data.get("goal_visible", False)),
            goal_match_confidence=float(data.get("goal_match_confidence", 0.0)),
            target_direction=str(data.get("target_direction", "unknown")),
            target_visibility=str(data.get("target_visibility", "not_visible")),
            apparent_scale=str(data.get("apparent_scale", "unknown")),
            relative_progress=str(data.get("relative_progress", "uncertain")),
            stop_candidate=bool(data.get("stop_candidate", False)),
            recommended_action=str(data.get("recommended_action", "search")),
            goal_reason=str(data.get("goal_reason", "")),
            portals=[str(p) for p in (data.get("portals") or [])],
            uncertainty=float(data.get("uncertainty", 0.5)),
            visual_embed=visual_embed,
            source="vlm",
            is_full=True,
            step_id=step_id,
            raw=data,
        )
