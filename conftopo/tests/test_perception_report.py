"""Unit tests for PerceptionReport and ClipGdinoReportBuilder."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.perception.perception_report import PerceptionReport, EMPTY
from conftopo.perception.clip_gdino_report_builder import ClipGdinoReportBuilder
from conftopo.perception.heavy_perceiver import ObjectObservation


# ---- PerceptionReport schema tests ----

def test_default_values():
    r = PerceptionReport()
    assert r.room_label == "unknown"
    assert r.room_confidence == 0.0
    assert r.room_scores == []
    assert r.goal_scores == []
    assert r.landmark_scores == []
    assert r.best_goal_sim == 0.0
    assert r.best_landmark_sim == 0.0
    assert r.objects == []
    assert r.scene_summary == ""
    assert r.goal_visible is False
    assert r.goal_reason == ""
    assert r.portals == []
    assert r.uncertainty == 0.0
    assert r.visual_embed is None
    assert r.source == "none"
    assert r.is_full is False
    assert r.step_id == 0
    assert r.raw == {}


def test_empty_sentinel():
    assert EMPTY.source == "none"
    assert EMPTY.room_label == "unknown"
    assert EMPTY.objects == []


def test_to_dict_from_dict_roundtrip():
    embed = np.random.randn(8).astype(np.float32)
    obj = ObjectObservation(label="chair", bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.85)
    r = PerceptionReport(
        room_label="kitchen",
        room_confidence=0.92,
        room_scores=[("kitchen", 0.92), ("bathroom", 0.3)],
        goal_scores=[("sink", 0.7)],
        landmark_scores=[("counter", 0.6)],
        best_goal_sim=0.7,
        best_landmark_sim=0.6,
        objects=[obj],
        scene_summary="A kitchen with a sink.",
        goal_visible=True,
        goal_reason="sink detected in bbox",
        portals=["door_left"],
        uncertainty=0.15,
        visual_embed=embed,
        source="clip_groundingdino",
        is_full=True,
        step_id=42,
    )
    d = r.to_dict()
    r2 = PerceptionReport.from_dict(d)
    assert r2.room_label == "kitchen"
    assert r2.room_confidence == 0.92
    assert len(r2.room_scores) == 2
    assert r2.goal_scores == [("sink", 0.7)]
    assert r2.best_goal_sim == 0.7
    assert len(r2.objects) == 1
    assert r2.objects[0].label == "chair"
    assert r2.scene_summary == "A kitchen with a sink."
    assert r2.goal_visible is True
    assert r2.source == "clip_groundingdino"
    assert r2.is_full is True
    assert r2.step_id == 42
    np.testing.assert_allclose(r2.visual_embed, embed, atol=1e-6)


def test_from_dict_missing_keys():
    r = PerceptionReport.from_dict({})
    assert r.room_label == "unknown"
    assert r.source == "none"
    assert r.visual_embed is None


# ---- ClipGdinoReportBuilder tests ----

def test_builder_light_empty():
    builder = ClipGdinoReportBuilder()
    r = builder.build_light({}, None, step_id=0)
    assert r.source == "none"
    assert r.is_full is False


def test_builder_light_with_data():
    builder = ClipGdinoReportBuilder()
    light = {
        "room_label": "bedroom",
        "room_confidence": 0.88,
        "room_scores": [("bedroom", 0.88), ("living room", 0.2)],
        "goal_scores": [("bed", 0.75)],
        "landmark_scores": [("window", 0.5)],
        "best_goal_sim": 0.75,
        "best_landmark_sim": 0.5,
    }
    embed = np.ones(4, dtype=np.float32)
    r = builder.build_light(light, embed, step_id=5)
    assert r.room_label == "bedroom"
    assert r.room_confidence == 0.88
    assert r.best_goal_sim == 0.75
    assert r.source == "clip"
    assert r.is_full is False
    assert r.objects == []
    assert r.visual_embed is embed


def test_builder_full():
    builder = ClipGdinoReportBuilder()
    light = {
        "room_label": "kitchen",
        "room_confidence": 0.9,
        "room_scores": [],
        "goal_scores": [],
        "landmark_scores": [],
        "best_goal_sim": 0.4,
        "best_landmark_sim": 0.3,
    }
    objs = [
        ObjectObservation(label="cup", bbox=[0.1, 0.1, 0.2, 0.2], confidence=0.9),
        ObjectObservation(label="plate", bbox=[0.3, 0.3, 0.5, 0.5], confidence=0.8),
    ]
    r = builder.build_full(light, objs, None, step_id=10)
    assert r.source == "clip_groundingdino"
    assert r.is_full is True
    assert len(r.objects) == 2
    assert r.objects[0].label == "cup"
    assert r.room_label == "kitchen"


if __name__ == "__main__":
    test_default_values()
    test_empty_sentinel()
    test_to_dict_from_dict_roundtrip()
    test_from_dict_missing_keys()
    test_builder_light_empty()
    test_builder_light_with_data()
    test_builder_full()
    print("All PerceptionReport + Builder tests passed!")
