"""Build PerceptionReport from CLIP LightPerceiver + GroundingDINO HeavyPerceiver outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.perception_report import PerceptionReport


class ClipGdinoReportBuilder:
    """Pure data-conversion layer: perceiver outputs -> PerceptionReport.

    No business logic lives here.  The builder is stateless and can be
    shared across episodes.
    """

    def build_light(
        self,
        perceiver_output: Dict[str, Any],
        visual_embed: Optional[np.ndarray],
        step_id: int = 0,
    ) -> PerceptionReport:
        """Every-step fast path: CLIP scores only, ``is_full=False``."""
        if not perceiver_output:
            return PerceptionReport(
                visual_embed=visual_embed,
                source="none",
                step_id=step_id,
            )
        return PerceptionReport(
            room_label=str(perceiver_output.get("room_label", "unknown")),
            room_confidence=float(perceiver_output.get("room_confidence", 0.0)),
            room_scores=list(perceiver_output.get("room_scores", [])),
            goal_scores=list(perceiver_output.get("goal_scores", [])),
            landmark_scores=list(perceiver_output.get("landmark_scores", [])),
            best_goal_sim=float(perceiver_output.get("best_goal_sim", 0.0)),
            best_landmark_sim=float(perceiver_output.get("best_landmark_sim", 0.0)),
            visual_embed=visual_embed,
            source="clip",
            is_full=False,
            step_id=step_id,
        )

    def build_full(
        self,
        perceiver_output: Dict[str, Any],
        heavy_observations: List[ObjectObservation],
        visual_embed: Optional[np.ndarray],
        step_id: int = 0,
    ) -> PerceptionReport:
        """Triggered path: CLIP + GroundingDINO detections, ``is_full=True``."""
        report = self.build_light(perceiver_output, visual_embed, step_id)
        report.objects = list(heavy_observations)
        report.source = "clip_groundingdino"
        report.is_full = True
        return report
