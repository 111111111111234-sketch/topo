"""VLM-backed perceiver: wraps any VLMBackendBase into a PerceptionReport producer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from conftopo.perception.heavy_perceiver import ObjectObservation, normalize_bbox
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import VLMBackendBase


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
    ) -> PerceptionReport:
        """Query the VLM and convert the response to a PerceptionReport.

        *mode*: ``"explore"`` for discovery or ``"confirm"`` for strict
        stop verification.  Selects the system prompt sent to the VLM.
        """
        raw = self._backend.query(rgb, goal_text, context=context, mode=mode)
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
            objects.append(ObjectObservation(
                label=str(o.get("label", "")),
                bbox=normalize_bbox(o.get("bbox")),
                confidence=float(o.get("confidence", 0.0)),
                source="vlm",
                step_id=step_id,
                visible=bool(o.get("visible", True)),
                visibility=str(o.get("visibility", "unknown")),
                bearing=str(o.get("bearing", "unknown")),
                range_bin=str(o.get("range", o.get("range_bin", "unknown"))),
                spatial_relation=[str(x) for x in relation],
                room_context=o.get("room_context"),
            ))

        return PerceptionReport(
            room_label=room_label,
            room_confidence=room_conf,
            room_scores=[(room_label, room_conf)] if room_label != "unknown" else [],
            objects=objects,
            scene_summary=str(data.get("scene_summary", "")),
            goal_visible=bool(data.get("goal_visible", False)),
            goal_reason=str(data.get("goal_reason", "")),
            portals=[str(p) for p in (data.get("portals") or [])],
            uncertainty=float(data.get("uncertainty", 0.5)),
            visual_embed=visual_embed,
            source="vlm",
            is_full=True,
            step_id=step_id,
            raw=data,
        )
