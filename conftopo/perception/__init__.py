"""ConfTopo perception modules."""

from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.clip_runtime import (
    ClipRuntimeEncoder,
    GoatModalityClipEncoder,
    agent_current_goal_type,
    encode_agent_rgb_embed,
    encode_agent_image_goal_embed,
)
from conftopo.perception.heavy_perceiver import (
    FakeGroundingDINOBackend,
    GroundingDINOBackend,
    HeavyPerceiver,
    ObjectObservation,
)
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.clip_gdino_report_builder import ClipGdinoReportBuilder
from conftopo.perception.perception_trigger import PerceptionTrigger, TriggerState
from conftopo.perception.vlm_backend import FakeVLMBackend, VLMBackendBase
from conftopo.perception.vlm_perceiver import VLMPerceiver

__all__ = [
    "LightPerceiver",
    "ClipRuntimeEncoder",
    "GoatModalityClipEncoder",
    "agent_current_goal_type",
    "encode_agent_rgb_embed",
    "encode_agent_image_goal_embed",
    "HeavyPerceiver",
    "ObjectObservation",
    "GroundingDINOBackend",
    "FakeGroundingDINOBackend",
    "PerceptionReport",
    "ClipGdinoReportBuilder",
    "PerceptionTrigger",
    "TriggerState",
    "VLMBackendBase",
    "FakeVLMBackend",
    "VLMPerceiver",
]
