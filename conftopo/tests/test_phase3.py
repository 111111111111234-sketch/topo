"""Phase 3 tests: heavy object grounding and reliable memory."""

import os
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.agents.goat_agent import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.confidence import ConfidenceFactors, compute_semantic_confidence
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.heavy_perceiver import FakeGroundingDINOBackend, HeavyPerceiver
from conftopo.perception.heavy_perceiver import GroundingDINOBackend
from conftopo.perception.perception_report import PerceptionReport


def test_fake_groundingdino_outputs_object_observations():
    perceiver = HeavyPerceiver(
        FakeGroundingDINOBackend([{
            "label": "sink",
            "bbox": [10, 20, 40, 80],
            "confidence": 0.78,
        }]),
        min_confidence=0.2,
    )
    out = perceiver.perceive(
        np.zeros((64, 64, 3), dtype=np.uint8),
        ["sink"],
        visual_embedding=np.ones(4, dtype=np.float32),
        view_heading=0.5,
        step_id=7,
    )
    assert len(out) == 1
    assert out[0].label == "sink"
    assert out[0].bbox == [10.0, 20.0, 40.0, 80.0]
    assert abs(out[0].confidence - 0.78) < 1e-6
    assert out[0].step_id == 7


def test_groundingdino_backend_prepares_numpy_rgb(monkeypatch):
    class FakeTensor:
        pass

    class FakeImage:
        @staticmethod
        def fromarray(arr):
            assert arr.dtype == np.uint8
            assert arr.shape[-1] == 3
            return FakeImage()

        def convert(self, mode):
            assert mode == "RGB"
            return self

    class FakeTransform:
        def __call__(self, image, target):
            return FakeTensor(), target

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = FakeImage
    fake_image_module = types.ModuleType("PIL.Image")
    fake_image_module.fromarray = FakeImage.fromarray
    fake_transforms = types.ModuleType("groundingdino.datasets.transforms")
    fake_transforms.Compose = lambda steps: FakeTransform()
    fake_transforms.RandomResize = lambda *args, **kwargs: object()
    fake_transforms.ToTensor = lambda: object()
    fake_transforms.Normalize = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)
    monkeypatch.setitem(sys.modules, "groundingdino.datasets.transforms", fake_transforms)

    backend = GroundingDINOBackend()
    tensor = backend._prepare_image(np.zeros((8, 8, 4), dtype=np.float32))
    assert isinstance(tensor, FakeTensor)


def test_upsert_object_observation_merges_multiview():
    topo = DynamicTopoMap()
    topo.step()
    vp1 = topo.add_node(NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    emb = np.array([1, 0, 0], dtype=np.float32)
    obj1, merged1 = topo.upsert_object_observation(
        label="table",
        bbox=[10, 10, 40, 40],
        confidence=0.65,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        embedding=emb,
        viewpoint_id=vp1,
        view_heading=0.0,
    )
    topo.step()
    vp2 = topo.add_node(NodeType.WAYPOINT_VISITED, position=np.array([0.5, 0, 0], dtype=np.float32))
    obj2, merged2 = topo.upsert_object_observation(
        label="table",
        bbox=[12, 12, 42, 42],
        confidence=0.72,
        position=np.array([1.1, 0.0, 0.0], dtype=np.float32),
        embedding=emb,
        viewpoint_id=vp2,
        view_heading=0.1,
    )
    node = topo.get_node(obj1)
    assert obj1 == obj2
    assert merged1 is False
    assert merged2 is True
    assert len(topo.get_nodes_by_type(NodeType.OBJECT)) == 1
    assert node.attributes["multi_view_count"] >= 2
    assert node.confidence > 0.65
    assert vp2 in topo.get_neighbors(obj1, EdgeType.OBSERVED_AT)


def test_upsert_object_observation_keeps_conflicting_labels_separate():
    topo = DynamicTopoMap()
    pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    obj1, _ = topo.upsert_object_observation(
        label="chair",
        bbox=[0, 0, 20, 20],
        confidence=0.7,
        position=pos,
        view_heading=0.0,
    )
    obj2, merged = topo.upsert_object_observation(
        label="door",
        bbox=[0, 0, 20, 20],
        confidence=0.7,
        position=pos,
        view_heading=0.0,
    )
    assert obj1 != obj2
    assert merged is False
    assert len(topo.get_nodes_by_type(NodeType.OBJECT)) == 2


def test_upsert_object_observation_keeps_different_rooms_separate():
    topo = DynamicTopoMap()
    pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb = np.array([1, 0, 0], dtype=np.float32)
    obj1, _ = topo.upsert_object_observation(
        label="table",
        bbox=[10, 10, 40, 40],
        confidence=0.8,
        position=pos,
        embedding=emb,
        view_heading=0.0,
        room_context="kitchen",
    )
    obj2, merged = topo.upsert_object_observation(
        label="table",
        bbox=[10, 10, 40, 40],
        confidence=0.8,
        position=pos + np.array([0.1, 0.0, 0.0], dtype=np.float32),
        embedding=emb,
        view_heading=0.0,
        room_context="bedroom",
    )
    assert obj1 != obj2
    assert merged is False
    assert len(topo.get_nodes_by_type(NodeType.OBJECT)) == 2


def test_observed_at_edge_requires_room_or_nearby_viewpoint():
    topo = DynamicTopoMap()
    vp = topo.add_node(NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    obj, _ = topo.upsert_object_observation(
        label="plant",
        bbox=[0, 0, 20, 20],
        confidence=0.7,
        position=np.array([20.0, 0.0, 0.0], dtype=np.float32),
        viewpoint_id=vp,
        view_heading=0.0,
    )
    assert vp not in topo.get_neighbors(obj, EdgeType.OBSERVED_AT)


def test_confidence_multiview_and_staleness_conflict():
    strong = compute_semantic_confidence(ConfidenceFactors(
        detection_score=0.8,
        multi_view_count=3,
        task_relevance=1.0,
        room_prior_score=1.0,
    ))
    stale_conflict = compute_semantic_confidence(ConfidenceFactors(
        detection_score=0.8,
        multi_view_count=1,
        task_relevance=0.0,
        room_prior_score=0.0,
        staleness_steps=20,
        conflict_penalty=0.5,
    ))
    assert strong > stale_conflict
    assert strong > 0.8


def test_object_observation_serialization_roundtrip():
    topo = DynamicTopoMap()
    obj, _ = topo.upsert_object_observation(
        label="sink",
        bbox=[1, 2, 3, 4],
        confidence=0.8,
        position=np.array([1, 0, 2], dtype=np.float32),
        room_context="kitchen",
    )
    data = topo.to_dict()
    restored = DynamicTopoMap.from_dict(data)
    node = restored.get_node(obj)
    assert node.label == "sink"
    assert node.attributes["bbox_observations"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert node.attributes["room_context"] == "kitchen"
    assert node.confidence == data["nodes"][obj]["confidence"]


def test_confidence_decay_is_per_step_not_cumulative_age():
    topo = DynamicTopoMap()
    obj, _ = topo.upsert_object_observation(
        label="sink",
        bbox=[1, 2, 3, 4],
        confidence=0.8,
        position=np.array([1, 0, 2], dtype=np.float32),
    )
    start_confidence = topo.get_node(obj).confidence
    for _ in range(10):
        topo.step()
        topo.decay_all_confidences()

    expected = start_confidence * (topo.confidence_decay ** 10)
    assert abs(topo.get_node(obj).confidence - expected) < 1e-6
    assert topo.get_node(obj).confidence > 0.45


def test_goat_agent_advances_topomap_once_per_step():
    agent = ConfTopoGOATAgent(ConfTopoConfig())
    agent.set_new_goal(GoalNode(target_object="sink"))
    agent.step({
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "position": [0, 0, 0],
        "heading": 0.0,
    })
    assert agent.step_count == 1
    assert agent.topo_map.current_step == 1


def test_far_object_history_is_compressed():
    topo = DynamicTopoMap()
    obj = None
    for idx in range(5):
        obj, _ = topo.upsert_object_observation(
            label="rack",
            bbox=[idx, idx, idx + 10, idx + 10],
            confidence=0.3 + 0.05 * idx,
            position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
            view_heading=0.0,
        )
        topo.step()
    node = topo.get_node(obj)
    node.confidence = 0.4
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    attrs = node.attributes
    assert attrs["granularity"] == "room_level"
    assert attrs["history_compressed"] is True
    assert attrs["history_original_observation_count"] == 5
    assert len(attrs["bbox_observations"]) <= 3
    assert attrs["history_best_confidence"] >= 0.5


def test_far_landmark_history_is_compressed():
    topo = DynamicTopoMap()
    lm = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="door",
        attributes={
            "observations": [
                {"viewpoint_id": f"vp_{idx}", "confidence": 0.4 + 0.05 * idx, "step_id": idx}
                for idx in range(6)
            ],
            "viewpoints": [f"vp_{idx}" for idx in range(6)],
            "granularity": "landmark",
        },
    )
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    attrs = topo.get_node(lm).attributes
    assert attrs["history_compressed"] is True
    assert attrs["history_original_observation_count"] == 6
    assert len(attrs["observations"]) <= 3
    assert len(attrs["viewpoints"]) <= 2


def test_far_object_is_added_to_room_summary():
    topo = DynamicTopoMap()
    obj = None
    for idx in range(5):
        obj, _ = topo.upsert_object_observation(
            label="rack",
            bbox=[idx, idx, idx + 10, idx + 10],
            confidence=0.3 + 0.02 * idx,
            position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
            view_heading=0.0,
            room_context="storage",
        )
        topo.step()
    node = topo.get_node(obj)
    node.confidence = 0.35
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    summaries = [
        node for node in topo.get_nodes_by_type(NodeType.ROOM)
        if node.attributes.get("summary_type") == "room_region"
    ]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.label == "storage"
    assert "rack" in summary.attributes["contains_labels"]
    assert obj in summary.attributes["contains_node_ids"]
    assert summary.attributes["summary_observations"][-1]["reason"] == "far_low_confidence"
    assert summary.node_id in topo.get_neighbors(obj, EdgeType.BELONGS_TO)


def test_far_landmark_is_added_to_room_summary():
    topo = DynamicTopoMap()
    lm = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6,
        label="door",
        attributes={
            "observations": [
                {"viewpoint_id": f"vp_{idx}", "confidence": 0.5, "step_id": idx}
                for idx in range(4)
            ],
            "viewpoints": [f"vp_{idx}" for idx in range(4)],
            "granularity": "landmark",
        },
    )
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    summaries = [
        node for node in topo.get_nodes_by_type(NodeType.ROOM)
        if node.attributes.get("summary_type") == "room_region"
    ]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.label == "region"
    assert "door" in summary.attributes["contains_labels"]
    assert lm in summary.attributes["contains_node_ids"]
    assert summary.attributes["summary_observations"][-1]["reason"] == "far_landmark"


def test_summary_observations_are_bounded():
    config = ConfTopoConfig()
    config.memory.summary_max_observations = 3
    topo = DynamicTopoMap(config.memory)
    for idx in range(6):
        obj, _ = topo.upsert_object_observation(
            label=f"obj_{idx}",
            bbox=[0, 0, 10, 10],
            confidence=0.3,
            position=np.array([12.0 + 0.1 * idx, 0.0, 0.0], dtype=np.float32),
            view_heading=0.0,
            room_context="room",
        )
        topo.get_node(obj).confidence = 0.25
        topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    summary = next(
        node for node in topo.get_nodes_by_type(NodeType.ROOM)
        if node.attributes.get("summary_type") == "room_region"
    )
    assert len(summary.attributes["summary_observations"]) == 3
    assert len(summary.attributes["contains_labels"]) == 3
    assert summary.attributes["contains_labels"] == ["obj_3", "obj_4", "obj_5"]


def test_goat_agent_heavy_triggers_near_coarse_summary():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.heavy_interval = 100
    agent = ConfTopoGOATAgent(config)
    agent.set_heavy_perceiver(HeavyPerceiver(FakeGroundingDINOBackend([]), min_confidence=0.2))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._goal_local_step = 2
    agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.5,
        label="region",
        attributes={
            "summary_type": "room_region",
            "contains_labels": ["sink"],
            "contains_node_ids": ["obj_1"],
            "summary_observations": [],
            "source_granularities": ["room_level"],
        },
    )
    should_run, reason = agent._should_run_heavy_perception()
    assert should_run is True
    assert reason == "coarse_summary_context"


def test_heavy_detection_marks_recovered_from_summary():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.heavy_interval = 100
    config.perception.object_detection_threshold = 0.2
    agent = ConfTopoGOATAgent(config)
    agent.set_heavy_perceiver(HeavyPerceiver(
        FakeGroundingDINOBackend([{
            "label": "sink",
            "bbox": [100, 100, 200, 220],
            "confidence": 0.82,
        }]),
        min_confidence=0.2,
    ))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._heading = 0.0
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_rgb_embed = np.ones(4, dtype=np.float32)
    agent._cur_report = PerceptionReport()
    agent._goal_local_step = 2
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([-0.6, 0.0, -1.9], dtype=np.float32),
        confidence=0.5,
        label="region",
        attributes={
            "summary_type": "room_region",
            "contains_labels": ["sink"],
            "contains_node_ids": ["old_sink"],
            "summary_observations": [],
            "source_granularities": ["room_level"],
        },
    )
    agent._add_heavy_object_nodes(cur_vp, None)
    objects = agent.topo_map.get_nodes_by_type(NodeType.OBJECT)
    assert len(objects) == 1
    attrs = objects[0].attributes
    assert attrs["recovered_from_summary"] is True
    assert attrs["summary_node_id"]
    assert attrs["recovered_step"] == agent.topo_map.current_step


def test_goat_agent_heavy_trigger_and_cooldown():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.heavy_interval = 4
    config.perception.heavy_goal_warmup_steps = 1
    config.perception.object_detection_threshold = 0.2
    agent = ConfTopoGOATAgent(config)
    backend = FakeGroundingDINOBackend([{
        "label": "sink",
        "bbox": [100, 100, 200, 220],
        "confidence": 0.82,
    }])
    agent.set_heavy_perceiver(HeavyPerceiver(backend, min_confidence=0.2))
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy(), room_prior=["kitchen"]))
    agent.perceiver.room_labels = ["kitchen"]
    agent.perceiver.room_text_embeds = embed[np.newaxis, :].copy()
    obs = {
        "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        "position": [0, 0, 0],
        "heading": 0.0,
        "rgb_embed": embed.copy(),
    }
    agent.step(obs)
    agent.step({**obs, "position": [0.2, 0, 0]})
    stats = agent.memory_stats
    assert stats["objects"] >= 1
    assert stats["heavy_perception_calls"] == 1
    assert backend.calls == 1
    assert stats["last_heavy"]["ran"] is False


# ==================== Phase3 folding / pruning / heavy tests ====================


def test_far_object_anchor_retention():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))

    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_vp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))

    # Add a far object (beyond near_radius=3, beyond room_level_min_distance=8)
    obj_id, _ = agent.topo_map.upsert_object_observation(
        label="chair", bbox=[0, 0, 10, 10], confidence=0.3,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32), view_heading=0.0)
    agent.topo_map.step()
    agent.topo_map.adaptive_granularity(agent._position)

    node = agent.topo_map.get_node(obj_id)
    assert node.attributes.get("folded") is True
    assert node.attributes.get("folded_detail") is True
    assert node.attributes.get("is_semantic_anchor") is True
    assert node.attributes.get("is_active_detail") is False
    assert node.attributes.get("anchor_waypoint_id") is not None

    reason = agent._candidate_skip_reason(node)
    assert reason == "folded_detail"


def test_anchor_waypoint_is_visited():
    topo = DynamicTopoMap()
    visited_id = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    topo.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([11.5, 0.0, 0.0], dtype=np.float32),
    )
    obj_id, _ = topo.upsert_object_observation(
        label="chair",
        bbox=[0, 0, 10, 10],
        confidence=0.3,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        view_heading=0.0,
    )
    topo.step()
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))

    obj = topo.get_node(obj_id)
    assert obj.attributes["anchor_waypoint_id"] == visited_id
    anchor_wp = topo.get_node(obj.attributes["anchor_waypoint_id"])
    assert anchor_wp.node_type == NodeType.WAYPOINT_VISITED


def test_goal_relevant_folded_anchor_not_skipped():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_vp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    obj_id = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.8,
        label="sink",
        attributes={
            "folded": True,
            "folded_detail": True,
            "is_semantic_anchor": True,
            "target_relevance": 1.0,
        },
    )

    reason = agent._candidate_skip_reason(agent.topo_map.get_node(obj_id))
    assert reason is None


def test_plan_targets_anchor_waypoint_for_folded_goal_object():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    config.planning.two_stage_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))
    agent._position = np.zeros(3, dtype=np.float32)
    anchor_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    anchor_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=anchor_pos,
    )
    agent._cur_vp_id = anchor_id
    obj_id = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        label="sink",
        embedding=embed.copy(),
        attributes={
            "folded": True,
            "folded_detail": True,
            "is_semantic_anchor": True,
            "is_active_detail": False,
            "target_relevance": 1.0,
            "anchor_position": np.array([12.0, 0.0, 0.0], dtype=np.float32),
            "anchor_waypoint_id": anchor_id,
            "anchor_waypoint_position": anchor_pos.copy(),
            "anchor_room_id": None,
        },
    )

    plan = agent.plan()
    assert plan["target_node_id"] == obj_id
    assert plan["requires_regrounding"] is True
    assert plan["target_anchor_type"] == "folded_object_anchor"
    assert plan["anchor_waypoint_id"] == anchor_id
    assert np.allclose(plan["target_position"], anchor_pos)


def test_room_semantic_summary_records_folded_object():
    topo = DynamicTopoMap()
    topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    obj_id, _ = topo.upsert_object_observation(
        label="chairs",
        bbox=[0, 0, 10, 10],
        confidence=0.35,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        view_heading=0.0,
        room_context="storage",
    )
    topo.step()
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))

    obj = topo.get_node(obj_id)
    room = topo.get_node(obj.attributes["anchor_room_id"])
    assert "chairs" in room.attributes["contains_labels"]
    assert obj_id in room.attributes["contains_node_ids"]
    semantic_summary = room.attributes["semantic_summary"]
    assert semantic_summary["contains_labels"]["chair"] >= 1
    assert obj_id in semantic_summary["folded_object_ids"]
    assert "chair" in semantic_summary["representative_objects"]
    assert room.attributes["summary_text"]


def test_regrounding_starts_only_after_anchor_reached():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    agent._position = np.zeros(3, dtype=np.float32)
    plan_output = {
        "target_node_id": "obj_sink",
        "target_position": np.array([2.0, 0.0, 0.0], dtype=np.float32),
        "requires_regrounding": True,
        "anchor_waypoint_id": "wp_anchor",
    }

    action = agent.act(plan_output)
    assert action["action"] == "navigate"
    assert agent._reground_state == "idle"

    agent._position = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    plan_output["target_position"] = np.zeros(3, dtype=np.float32)
    action = agent.act(plan_output)
    assert action["action"] == "turn_right"
    assert action["mode"] == "local_reground_scan"
    assert agent._reground_state == "scanning"


def test_regrounding_forces_heavy_perception():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.heavy_interval = 100
    agent = ConfTopoGOATAgent(config)
    agent.set_heavy_perceiver(HeavyPerceiver(FakeGroundingDINOBackend([]), min_confidence=0.2))
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._last_heavy_step = agent.topo_map.current_step
    agent._reground_state = "scanning"

    should_run, reason = agent._should_run_heavy_perception()
    assert should_run is True
    assert reason == "local_regrounding"


def test_reground_scan_transitions_to_searching_after_rotations():
    agent = ConfTopoGOATAgent(ConfTopoConfig())
    agent._reground_state = "scanning"

    action = None
    for _ in range(agent._reground_scan_rotations):
        action = agent.act({"mode": "local_reground_scan"})

    assert action["action"] == "turn_right"
    assert agent._reground_state == "searching"
    assert agent._reground_steps == 0
    assert agent._reground_scan_rotated == 0


def test_reground_heavy_detection_records_target_object():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.object_detection_threshold = 0.2
    agent = ConfTopoGOATAgent(config)
    agent.set_heavy_perceiver(HeavyPerceiver(
        FakeGroundingDINOBackend([{
            "label": "sink",
            "bbox": [100, 100, 200, 220],
            "confidence": 0.82,
        }]),
        min_confidence=0.2,
    ))
    agent.set_new_goal(GoalNode(target_object="sink"))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._heading = 0.0
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_rgb_embed = np.ones(4, dtype=np.float32)
    agent._cur_report = PerceptionReport()
    agent._reground_state = "scanning"
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )

    agent._add_heavy_object_nodes(cur_vp, None)

    assert agent._target_object_detected_this_scan is True
    target = agent.topo_map.get_node(agent._reground_target_node_id)
    assert target is not None
    assert target.node_type == NodeType.OBJECT
    assert target.label == "sink"


def test_reground_search_exits_after_max_steps():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    agent.set_new_goal(GoalNode(target_object="sink"))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_vp_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.zeros(3, dtype=np.float32),
    )
    agent._reground_state = "searching"
    agent._reground_anchor_node_id = "wp_anchor"
    agent._reground_anchor_position = np.zeros(3, dtype=np.float32)
    agent._reground_max_steps = 1

    agent.plan()

    assert agent._reground_state == "failed"
    assert agent._reground_failed_anchor_node_id == "wp_anchor"

    action = agent.act({
        "target_node_id": "obj_sink",
        "target_position": np.zeros(3, dtype=np.float32),
        "requires_regrounding": True,
        "anchor_waypoint_id": "wp_anchor",
    })
    assert action["action"] in ("navigate", "stop")
    assert agent._reground_state == "failed"


def test_folded_object_unfolds_when_agent_near():
    topo = DynamicTopoMap()
    obj_id, _ = topo.upsert_object_observation(
        label="sink", bbox=[0, 0, 10, 10], confidence=0.3,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32), view_heading=0.0)
    topo.step()
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    assert topo.get_node(obj_id).attributes.get("folded") is True

    # Move agent close
    topo.adaptive_granularity(np.array([11.0, 0.0, 0.0], dtype=np.float32))
    assert topo.get_node(obj_id).attributes.get("folded") is False


def test_room_summary_is_plan_candidate():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))

    agent._position = np.zeros(3, dtype=np.float32)
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    agent._cur_vp_id = cur_vp

    summary_id = agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]})
    agent.topo_map.add_edge(cur_vp, summary_id, EdgeType.NAVIGABLE)

    plan = agent.plan()
    assert plan.get("target_node_id") == summary_id


def test_two_stage_planner_picks_goal_relevant_room_as_structure_target():
    """Stage 1 should pick the kitchen (contains sink) as structure target."""
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))

    agent._position = np.zeros(3, dtype=np.float32)
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    agent._cur_vp_id = cur_vp

    kitchen_id = agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]})
    agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="bedroom",
        attributes={"summary_type": "room_region", "contains_labels": ["bed"]})

    plan = agent.plan()
    assert plan.get("structure_target_id") == kitchen_id


def test_two_stage_planner_prefers_anchored_waypoint_over_far_object():
    """Frontier inside chosen room should outrank a stray far object."""
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))

    agent._position = np.zeros(3, dtype=np.float32)
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    agent._cur_vp_id = cur_vp

    # Kitchen with the goal object label as structure target.
    kitchen_id = agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]})
    # Frontier waypoint inside the kitchen anchor radius.
    frontier_id = agent.topo_map.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([4.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.4,
        attributes={"room_id": kitchen_id},
    )
    # An unrelated stray semantic object far away from the kitchen.
    far_chair = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.array([0.0, 0.0, 20.0], dtype=np.float32),
        confidence=0.6, label="chair",
        attributes={"granularity": "object"},
    )
    agent.topo_map.add_edge(cur_vp, frontier_id, EdgeType.NAVIGABLE)

    plan = agent.plan()
    assert plan.get("structure_target_id") == kitchen_id
    assert plan.get("target_node_id") in (frontier_id, kitchen_id)
    assert plan.get("target_node_id") != far_chair
    skipped_ids = {item["node_id"] for item in agent._last_skipped_candidates}
    assert far_chair in skipped_ids


def test_two_stage_planner_disabled_falls_back_to_legacy():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    config.planning.two_stage_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))

    agent._position = np.zeros(3, dtype=np.float32)
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    agent._cur_vp_id = cur_vp

    summary_id = agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]})
    agent.topo_map.add_edge(cur_vp, summary_id, EdgeType.NAVIGABLE)

    plan = agent.plan()
    assert plan.get("structure_target_id") is None
    assert plan.get("target_node_id") == summary_id


def test_two_stage_planner_keeps_goal_object_anchored():
    """Direct hit of the goal object should not be filtered by anchored skip."""
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))

    agent._position = np.zeros(3, dtype=np.float32)
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    agent._cur_vp_id = cur_vp

    kitchen_id = agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]})
    # Far goal-object hit: should NOT be filtered even though outside anchor.
    sink_id = agent.topo_map.add_node(
        NodeType.OBJECT,
        position=np.array([0.0, 0.0, 20.0], dtype=np.float32),
        confidence=0.9, label="sink",
        embedding=embed.copy(),
        attributes={"granularity": "object"},
    )

    agent.plan()
    skipped_ids = {item["node_id"] for item in agent._last_skipped_candidates}
    assert sink_id not in skipped_ids
    assert agent._last_structure_target_id == kitchen_id


def test_heavy_summary_labels_exclude_vocabulary():
    from conftopo.agents.goat_agent import DEFAULT_HEAVY_OBJECT_VOCABULARY
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy(),
                                landmarks=["faucet"]))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_report = PerceptionReport()

    class FakeSummary:
        attributes = {"contains_labels": ["towel", "mirror"]}

    labels = agent._heavy_labels(reason="coarse_summary_context", summary=FakeSummary())
    assert "sink" in labels
    assert "towel" in labels
    assert "mirror" in labels
    for vocab_word in DEFAULT_HEAVY_OBJECT_VOCABULARY:
        if vocab_word not in ("sink",):
            assert vocab_word not in labels, f"{vocab_word} should not be in summary-context labels"


def test_heavy_labels_align_with_room_structure_target():
    """Room structure target replaces broad vocab with its contains_labels."""
    from conftopo.agents.goat_agent import DEFAULT_HEAVY_OBJECT_VOCABULARY
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_report = PerceptionReport()

    room = agent.topo_map._nodes.setdefault(
        "kitchen_room",
        agent.topo_map.get_node(
            agent.topo_map.add_node(
                NodeType.ROOM,
                position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
                confidence=0.7,
                label="kitchen",
                attributes={
                    "summary_type": "room_region",
                    "contains_labels": ["sink", "faucet"],
                },
            )
        ),
    )

    labels = agent._heavy_labels(reason="interval", structure_target=room)
    assert "sink" in labels
    assert "faucet" in labels
    # Broad vocab is suppressed in favor of the room's own labels.
    suppressed = [v for v in DEFAULT_HEAVY_OBJECT_VOCABULARY if v not in ("sink",)]
    for vocab_word in suppressed:
        assert vocab_word not in labels


def test_heavy_labels_align_with_portal_structure_target():
    """Portal/structural-landmark target injects structural vocabulary."""
    from conftopo.agents.goat_agent import (
        DEFAULT_HEAVY_OBJECT_VOCABULARY,
        DEFAULT_STRUCTURAL_HEAVY_VOCABULARY,
    )
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_report = PerceptionReport()

    portal_id = agent.topo_map.add_node(
        NodeType.LANDMARK,
        position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6,
        label="door",
        attributes={
            "structure_role": "portal",
            "synthetic_portal": True,
            "structure_pair_labels": ["kitchen", "hallway"],
        },
    )
    portal = agent.topo_map.get_node(portal_id)
    labels = agent._heavy_labels(reason="interval", structure_target=portal)
    assert "sink" in labels  # goal label always kept
    assert any(w in labels for w in DEFAULT_STRUCTURAL_HEAVY_VOCABULARY)
    assert "hallway" in labels  # paired room label injected
    # broad vocabulary suppressed (except items that overlap with the
    # structural vocab like "door").
    suppressed = [
        v for v in DEFAULT_HEAVY_OBJECT_VOCABULARY
        if v not in ("sink", "door")
    ]
    for vocab_word in suppressed:
        assert vocab_word not in labels


def test_heavy_align_with_structure_target_disabled_uses_default_vocab():
    from conftopo.agents.goat_agent import DEFAULT_HEAVY_OBJECT_VOCABULARY
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.heavy_align_with_structure_target = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_report = PerceptionReport()

    room_id = agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7, label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]},
    )
    room = agent.topo_map.get_node(room_id)
    labels = agent._heavy_labels(reason="interval", structure_target=room)
    # With alignment disabled, default vocab still injected.
    for vocab_word in DEFAULT_HEAVY_OBJECT_VOCABULARY:
        assert vocab_word in labels


def test_heavy_detection_boosts_structural_label_relevance():
    """Structural detections (door) get a relevance bump even when not the goal."""
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.heavy_interval = 1
    config.perception.heavy_goal_warmup_steps = 99
    agent = ConfTopoGOATAgent(config)
    agent.set_heavy_perceiver(
        HeavyPerceiver(FakeGroundingDINOBackend([
            {"label": "door", "bbox": [0.4, 0.4, 0.6, 0.6], "confidence": 0.7},
        ]), min_confidence=0.2)
    )
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_report = PerceptionReport()
    agent._goal_local_step = 1
    cur_vp = agent.topo_map.add_node(
        NodeType.WAYPOINT_VISITED, position=np.zeros(3, dtype=np.float32))
    agent._cur_vp_id = cur_vp

    agent._add_heavy_object_nodes(cur_vp, room_label="kitchen")
    door_nodes = [
        n for n in agent.topo_map.get_nodes_by_type(NodeType.OBJECT)
        if n.label == "door"
    ]
    assert door_nodes, "expected a door OBJECT node from heavy perception"
    relevance = float(door_nodes[0].attributes.get("target_relevance", 0.0))
    assert relevance > 0.0, "structural label should receive relevance boost"


# -----------------------------------------------------------------------
# RoomClassifier tests
# -----------------------------------------------------------------------

def test_room_classifier_temporal_smoothing():
    """Single-frame observation must NOT confirm a room; needs K consistent frames."""
    from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
    cfg = RoomClassifierConfig(vote_window=5, confirm_vote_fraction=0.60,
                               min_raw_confidence=0.30, transition_min_displacement=0.0)
    clf = RoomClassifier(cfg)
    pos = np.zeros(3, dtype=np.float32)
    # One bedroom frame: should not confirm yet
    r = clf.update("bedroom", 0.45, position=pos)
    assert r.confirmed_label is None, "single frame must not confirm"
    # Three more bedroom frames: now 4/5 window → should confirm
    for _ in range(3):
        r = clf.update("bedroom", 0.45, position=pos)
    assert r.confirmed_label == "bedroom"
def test_room_classifier_temporal_smoothing():
    """Single-frame observation must NOT confirm a room; needs K consistent frames.

    With vote_window=4, half=2 frames are required before the first
    confirmation can fire (see room_classifier.py window_full_enough logic).
    """
    from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
    cfg = RoomClassifierConfig(vote_window=4, confirm_vote_fraction=0.60,
                               min_raw_confidence=0.30, transition_min_displacement=0.0)
    clf = RoomClassifier(cfg)
    pos = np.zeros(3, dtype=np.float32)
    # One bedroom frame: window has 1 item, half=2 so NOT yet confirmed
    r = clf.update("bedroom", 0.45, position=pos)
    assert r.confirmed_label is None, "single frame must not confirm (window not half full)"
    # Second frame: window has 2 items (half=2), fraction=1.0 ≥ 0.6 → confirmed
    r = clf.update("bedroom", 0.45, position=pos)
    assert r.confirmed_label == "bedroom", "two consistent frames should confirm"


def test_room_classifier_score_distribution():
    """scores dict must contain all observed labels and sum ~1."""
    from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
    cfg = RoomClassifierConfig(vote_window=5, confirm_vote_fraction=0.60,
                               min_raw_confidence=0.25, transition_min_displacement=0.0)
    clf = RoomClassifier(cfg)
    pos = np.zeros(3, dtype=np.float32)
    for lbl, conf in [("bedroom", 0.40), ("hallway", 0.35), ("bedroom", 0.42), ("bedroom", 0.38)]:
        clf.update(lbl, conf, position=pos)
    r = clf.update("bedroom", 0.41, position=pos)
    assert "bedroom" in r.scores
    assert "hallway" in r.scores
    assert r.scores["bedroom"] > r.scores["hallway"]
    total = sum(r.scores.values())
    assert abs(total - 1.0) < 0.05, f"scores should sum ~1, got {total}"


def test_room_classifier_transition_requires_displacement():
    """No transition should fire if agent hasn't moved enough."""
    from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
    cfg = RoomClassifierConfig(vote_window=3, confirm_vote_fraction=0.60,
                               min_raw_confidence=0.25,
                               transition_min_displacement=2.0)
    clf = RoomClassifier(cfg)
    pos = np.zeros(3, dtype=np.float32)
    # Confirm bedroom first with no movement
    for _ in range(3):
        clf.update("bedroom", 0.45, position=pos)
    # Now switch to living_room frames, same position
    transitions = []
    for _ in range(3):
        r = clf.update("living room", 0.45, position=pos)
        transitions.append(r.is_transition)
    assert not any(transitions), "transition should not fire without displacement"
    # Same switch but with displacement > threshold
    far_pos = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    clf2 = RoomClassifier(cfg)
    for _ in range(3):
        clf2.update("bedroom", 0.45, position=pos)
    found = False
    for _ in range(3):
        r = clf2.update("living room", 0.45, position=far_pos)
        if r.is_transition:
            found = True
            break
    assert found, "transition should fire with sufficient displacement"
def test_room_classifier_transition_requires_displacement():
    """No transition should fire if agent hasn't moved enough."""
    from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
    cfg = RoomClassifierConfig(vote_window=4, confirm_vote_fraction=0.60,
                               min_raw_confidence=0.25,
                               transition_min_displacement=2.0,
                               transition_holdoff_steps=0)
    clf = RoomClassifier(cfg)
    pos = np.zeros(3, dtype=np.float32)
    # Confirm bedroom first (need ≥2 frames since half=2)
    for _ in range(4):
        clf.update("bedroom", 0.45, position=pos)
    assert clf._confirmed == "bedroom"
    # Switch to living_room at same position — must NOT transition
    transitions = []
    for _ in range(4):
        r = clf.update("living room", 0.45, position=pos)
        transitions.append(r.is_transition)
    assert not any(transitions), "transition should not fire without displacement"
    # Same switch but with displacement > threshold
    far_pos = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    clf2 = RoomClassifier(cfg)
    for _ in range(4):
        clf2.update("bedroom", 0.45, position=pos)
    assert clf2._confirmed == "bedroom"
    found = False
    for _ in range(4):
        r = clf2.update("living room", 0.45, position=far_pos)
        if r.is_transition:
            found = True
            break
    assert found, "transition should fire with sufficient displacement"


def test_room_classifier_object_prior_boosts_room():
    """Detected 'bed' should boost bedroom over competing rooms."""
    from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
    cfg = RoomClassifierConfig(vote_window=3, confirm_vote_fraction=0.60,
                               min_raw_confidence=0.25, transition_min_displacement=0.0,
                               object_prior_weight=1.0)
    clf = RoomClassifier(cfg)
    pos = np.zeros(3, dtype=np.float32)
    # Feed ambiguous frames with a bed detection
    for _ in range(3):
        clf.update("bedroom", 0.30, position=pos, object_labels=["bed", "wardrobe"])
    r = clf.update("hallway", 0.30, position=pos, object_labels=["bed"])
    assert r.scores.get("bedroom", 0) > r.scores.get("hallway", 0), \
        "object prior should keep bedroom above hallway"


def test_room_classifier_noise_frame_ignored():
    """Frames below min_raw_confidence must not affect the vote window."""
    from conftopo.core.room_classifier import RoomClassifier, RoomClassifierConfig
    cfg = RoomClassifierConfig(vote_window=5, confirm_vote_fraction=0.60,
                               min_raw_confidence=0.32, transition_min_displacement=0.0)
    clf = RoomClassifier(cfg)
    pos = np.zeros(3, dtype=np.float32)
    # Four bedroom frames above threshold
    for _ in range(4):
        clf.update("bedroom", 0.45, position=pos)
    # Two very low-confidence hallway frames (should be ignored)
    for _ in range(2):
        clf.update("hallway", 0.20, position=pos)
    r = clf.update("bedroom", 0.45, position=pos)
    assert r.confirmed_label == "bedroom", "low-conf frames must not displace bedroom"


def test_agent_waypoint_stores_room_scores():
    """After update_memory, cur_vp should have room_scores attribute."""
    config = ConfTopoConfig()
    config.perception.heavy_enabled = False
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    agent.set_new_goal(GoalNode(target_object="sink", target_embedding=embed.copy()))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._heading = 0.0
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._cur_report = PerceptionReport(
        room_label="bedroom",
        room_confidence=0.45,
        source="clip",
    )
    # Run enough steps to get 5 consistent bedroom frames
    for _ in range(5):
        agent.update_memory()
    wp = agent.topo_map.get_node(agent._cur_vp_id)
    assert wp is not None
    scores = wp.attributes.get("room_scores")
    assert isinstance(scores, dict) and len(scores) > 0, \
        "waypoint should store room_scores distribution"


def test_heavy_summary_respects_cooldown():
    config = ConfTopoConfig()
    config.perception.heavy_enabled = True
    config.perception.heavy_interval = 1
    config.perception.heavy_summary_cooldown = 5
    agent = ConfTopoGOATAgent(config)
    agent.set_heavy_perceiver(HeavyPerceiver(FakeGroundingDINOBackend([]), min_confidence=0.2))
    agent._position = np.zeros(3, dtype=np.float32)
    agent._cur_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    agent._goal_local_step = 2

    agent.topo_map.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.5, label="region",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]})

    should_run, reason = agent._should_run_heavy_perception()
    assert should_run and reason == "coarse_summary_context"

    # Simulate that summary was triggered at step 0
    agent._last_heavy_step = 0
    agent._last_heavy_summary_step = 0
    agent.topo_map._current_step = 3  # within cooldown of 5

    should_run2, reason2 = agent._should_run_heavy_perception()
    # Should not trigger summary (cooldown) but may trigger interval
    assert reason2 != "coarse_summary_context" or not should_run2


def test_distance_aware_prune_removes_far_low_conf():
    topo = DynamicTopoMap()
    # Far object with low confidence
    obj_id, _ = topo.upsert_object_observation(
        label="vase", bbox=[0, 0, 5, 5], confidence=0.1,
        position=np.array([15.0, 0.0, 0.0], dtype=np.float32), view_heading=0.0)
    topo.get_node(obj_id).confidence = 0.15  # below far_prune_threshold=0.18
    topo.prune_low_confidence(np.zeros(3, dtype=np.float32))
    assert topo.get_node(obj_id) is None


def test_room_level_triggers_at_mid_far_distance():
    topo = DynamicTopoMap()
    # Object at 9m (between room_level_min_distance=8 and far_radius=10)
    obj_id, _ = topo.upsert_object_observation(
        label="lamp", bbox=[0, 0, 10, 10], confidence=0.4,
        position=np.array([9.0, 0.0, 0.0], dtype=np.float32), view_heading=0.0)
    topo.step()
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    node = topo.get_node(obj_id)
    assert node.attributes["granularity"] == "room_level"


def test_environment_landmark_not_created_as_map_node():
    config = ConfTopoConfig()
    config.perception.landmark_threshold = 0.1
    config.perception.room_threshold = 0.1
    agent = ConfTopoGOATAgent(config)
    embed = np.zeros(512, dtype=np.float32)
    embed[0] = 1.0
    env_labels = ["hallway", "kitchen", "door"]
    agent.set_environment_landmark_labels(env_labels)
    agent.perceiver.room_labels = ["kitchen"]
    agent.perceiver.room_text_embeds = embed[np.newaxis, :].copy()
    agent.perceiver.set_landmark_labels(
        labels=env_labels,
        embeddings=np.tile(embed[np.newaxis, :], (len(env_labels), 1)),
    )
    agent.step({"position": [0, 0, 0], "heading": 0.0, "rgb_embed": embed.copy()})
    assert agent.memory_stats["landmarks"] == 0
    wp = agent.topo_map.get_visited()[0]
    assert "scene_vocabulary" in wp.attributes
    assert wp.attributes.get("view_room_label") == "kitchen"


def test_object_promotes_to_landmark_at_mid_distance():
    """Structural label (door) under heavy detection promotes cheaply."""
    topo = DynamicTopoMap()
    obj_id, _ = topo.upsert_object_observation(
        label="door",
        bbox=[0, 0, 10, 10],
        confidence=0.55,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        view_heading=0.0,
        source="groundingdino",
    )
    topo.step()
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    node = topo.get_node(obj_id)
    assert node.node_type == NodeType.LANDMARK
    assert node.attributes.get("promoted_from_object") is True
    assert node.attributes.get("landmark_role") == "structural"


def test_semantic_object_not_auto_promoted():
    """Semantic label with low confidence / single view stays as object."""
    topo = DynamicTopoMap()
    obj_id, _ = topo.upsert_object_observation(
        label="rack",
        bbox=[0, 0, 10, 10],
        confidence=0.55,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        view_heading=0.0,
        source="groundingdino",
    )
    topo.step()
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    node = topo.get_node(obj_id)
    assert node.node_type == NodeType.OBJECT
    check = node.attributes.get("promotion_check") or {}
    assert check.get("allowed") is False
    assert check.get("role") == "semantic"


def test_semantic_object_promoted_when_task_relevant():
    """Semantic label with strong task relevance + high confidence promotes."""
    topo = DynamicTopoMap()
    obj_id, _ = topo.upsert_object_observation(
        label="tv",
        bbox=[0, 0, 10, 10],
        confidence=0.75,
        position=np.array([7.0, 0.0, 0.0], dtype=np.float32),
        view_heading=0.0,
        source="groundingdino",
        target_relevance=0.5,
    )
    topo.step()
    topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    node = topo.get_node(obj_id)
    assert node.node_type == NodeType.LANDMARK
    assert node.attributes.get("landmark_role") == "semantic"


def test_persistent_structure_only_for_structural_landmark():
    """Promoted semantic landmark is no longer unconditionally persistent."""
    topo = DynamicTopoMap()
    structural_id = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6,
        label="door",
        attributes={
            "landmark_source": "promoted_object",
            "promoted_from_object": True,
            "landmark_role": "structural",
        },
    )
    semantic_id = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6,
        label="vase",
        attributes={
            "landmark_source": "promoted_object",
            "promoted_from_object": True,
            "landmark_role": "semantic",
        },
    )
    assert topo._is_persistent_structure_node(topo.get_node(structural_id)) is True
    assert topo._is_persistent_structure_node(topo.get_node(semantic_id)) is False


def test_assign_waypoint_to_room_binds_explicit_room_id():
    """Visited waypoint should get an explicit room_id + BELONGS_TO edge."""
    topo = DynamicTopoMap()
    room_id = topo.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="kitchen",
        attributes={
            "summary_type": "room_region",
            "contains_labels": ["sink"],
        },
    )
    wp_id = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        attributes={"view_room_label": "kitchen"},
    )
    bound = topo.assign_waypoint_to_room(wp_id, view_room_label="kitchen")
    assert bound == room_id
    wp = topo.get_node(wp_id)
    assert wp.attributes.get("room_id") == room_id
    assert wp.attributes.get("room_label") == "kitchen"
    assert topo.graph.has_edge(wp_id, room_id)
    assert topo.graph.edges[wp_id, room_id]["edge_type"] == EdgeType.BELONGS_TO.value


def test_assign_waypoint_to_room_prefers_matching_label():
    """When two rooms exist, label match wins over pure distance."""
    topo = DynamicTopoMap()
    far_kitchen = topo.add_node(
        NodeType.ROOM,
        position=np.array([4.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]},
    )
    near_bedroom = topo.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="bedroom",
        attributes={"summary_type": "room_region", "contains_labels": ["bed"]},
    )
    wp_id = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        attributes={"view_room_label": "kitchen"},
    )
    bound = topo.assign_waypoint_to_room(wp_id)
    assert bound == far_kitchen
    assert topo.get_node(wp_id).attributes.get("room_id") == far_kitchen
    assert not topo.graph.has_edge(wp_id, near_bedroom)


def test_navigation_view_only_returns_waypoints():
    topo = DynamicTopoMap()
    wp1 = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
    )
    wp2 = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
    )
    frontier = topo.add_node(
        NodeType.WAYPOINT_FRONTIER,
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.5,
    )
    topo.add_node(
        NodeType.LANDMARK,
        position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6,
        label="door",
        attributes={"landmark_role": "structural"},
    )
    topo.add_edge(wp1, wp2, EdgeType.NAVIGABLE)
    topo.add_edge(wp2, frontier, EdgeType.NAVIGABLE)

    view = topo.get_navigation_view()
    types = {node["type"] for node in view["nodes"]}
    assert types <= {
        NodeType.WAYPOINT_VISITED.value,
        NodeType.WAYPOINT_FRONTIER.value,
        NodeType.WAYPOINT_CANDIDATE.value,
    }
    assert len(view["nodes"]) == 3
    assert all(edge["type"] == EdgeType.NAVIGABLE.value for edge in view["edges"])


def test_structure_view_excludes_semantic_landmarks():
    topo = DynamicTopoMap()
    room_id = topo.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["door"]},
    )
    structural_id = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([1.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.6,
        label="door",
        attributes={"landmark_role": "structural"},
    )
    semantic_id = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.6,
        label="vase",
        attributes={"landmark_role": "semantic"},
    )
    topo.add_edge(structural_id, room_id, EdgeType.ADJACENT_TO)
    topo.add_edge(semantic_id, room_id, EdgeType.BELONGS_TO)

    view = topo.get_structure_view()
    ids = {node["id"] for node in view["nodes"]}
    assert room_id in ids
    assert structural_id in ids
    assert semantic_id not in ids
    edge_keys = {
        (tuple(sorted((e["source"], e["target"]))), e["type"])
        for e in view["edges"]
    }
    assert (tuple(sorted((structural_id, room_id))), EdgeType.ADJACENT_TO.value) in edge_keys
    # Edge from semantic landmark is filtered out because semantic_id is not
    # part of the structure layer.
    for e in view["edges"]:
        assert semantic_id not in (e["source"], e["target"])


def test_structure_view_includes_waypoint_room_bindings():
    topo = DynamicTopoMap()
    room_id = topo.add_node(
        NodeType.ROOM,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": []},
    )
    wp_id = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        attributes={"view_room_label": "kitchen"},
    )
    topo.assign_waypoint_to_room(wp_id, view_room_label="kitchen")
    view = topo.get_structure_view()
    bindings = view["waypoint_room_bindings"]
    assert {"waypoint_id": wp_id, "room_id": room_id} in bindings


def test_skeleton_skips_semantic_landmark_as_portal():
    """Portal candidate selection should ignore semantic landmarks."""
    topo = DynamicTopoMap()
    wp_a = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        attributes={"view_room_label": "kitchen"},
    )
    wp_b = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([6.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        attributes={"view_room_label": "bedroom"},
    )
    topo.add_edge(wp_a, wp_b, EdgeType.NAVIGABLE)
    # Semantic landmark sitting right at the midpoint - tempting for a
    # naive portal picker but should be rejected.
    semantic_landmark = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="vase",
        attributes={
            "landmark_role": "semantic",
            "landmark_source": "promoted_object",
        },
    )
    # Run the skeleton maintenance via adaptive_granularity. Need room
    # summaries first - create them directly to make the test hermetic.
    topo.add_node(
        NodeType.ROOM,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": []},
    )
    topo.add_node(
        NodeType.ROOM,
        position=np.array([6.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="bedroom",
        attributes={"summary_type": "room_region", "contains_labels": []},
    )
    topo._maintain_spatial_structure_graph()
    sem_node = topo.get_node(semantic_landmark)
    assert sem_node is not None
    assert sem_node.attributes.get("structure_role") != "portal"
    assert sem_node.attributes.get("structure_anchor") is not True


def test_room_summary_position_stable_on_repeat_update():
    topo = DynamicTopoMap()
    obj_id, _ = topo.upsert_object_observation(
        label="door",
        bbox=[0, 0, 10, 10],
        confidence=0.6,
        position=np.array([5.0, 0.0, 2.0], dtype=np.float32),
        view_heading=0.0,
        room_context="hallway",
    )
    node = topo.get_node(obj_id)
    for _ in range(5):
        topo.adaptive_granularity(np.zeros(3, dtype=np.float32))
    summaries = [
        n for n in topo.get_nodes_by_type(NodeType.ROOM)
        if n.attributes.get("summary_type") == "room_region"
    ]
    assert len(summaries) == 1
    pos = summaries[0].position
    assert float(np.linalg.norm(pos - np.array([5.0, 0.0, 2.0], dtype=np.float32))) < 0.01


def test_persistent_structure_not_pruned():
    topo = DynamicTopoMap()
    summary_id = topo.add_node(
        NodeType.ROOM,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.05,
        label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]},
    )
    lm_id = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([11.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.05,
        label="door",
        attributes={"landmark_source": "promoted_object", "promoted_from_object": True},
    )
    topo.prune_low_confidence(np.zeros(3, dtype=np.float32))
    assert topo.get_node(summary_id) is not None
    assert topo.get_node(lm_id) is not None


def test_two_layer_filter_keeps_structure_and_near_objects():
    from conftopo.viz.memory_trace_viz import filter_topo_nodes

    agent = np.zeros(3, dtype=np.float32)
    nodes = [
        {"id": "wp", "type": "waypoint_visited", "position": [0, 0, 0], "confidence": 1.0, "attributes": {}},
        {"id": "far_room", "type": "room", "position": [20, 0, 0], "confidence": 0.6,
         "label": "bedroom", "attributes": {"summary_type": "room_region"}},
        {"id": "far_lm", "type": "landmark", "position": [18, 0, 0], "confidence": 0.5,
         "label": "door", "attributes": {"landmark_source": "promoted_object", "promoted_from_object": True}},
        {"id": "near_obj", "type": "object", "position": [1, 0, 0], "confidence": 0.8,
         "label": "chair", "attributes": {"granularity": "object"}},
        {"id": "far_obj", "type": "object", "position": [15, 0, 0], "confidence": 0.8,
         "label": "table", "attributes": {"granularity": "object"}},
    ]
    out = filter_topo_nodes(nodes, "two_layer", agent_pos=agent, near_radius=3.0, far_radius=10.0)
    ids = {n["id"] for n in out}
    assert "wp" in ids
    assert "far_room" in ids
    assert "far_lm" in ids
    assert "near_obj" in ids
    assert "far_obj" not in ids


def test_maintain_spatial_structure_links_rooms():
    topo = DynamicTopoMap()
    room_a = topo.add_node(
        NodeType.ROOM,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]},
    )
    room_b = topo.add_node(
        NodeType.ROOM,
        position=np.array([8.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="hallway",
        attributes={"summary_type": "room_region", "contains_labels": ["door"]},
    )
    portal_id = topo.add_node(
        NodeType.LANDMARK,
        position=np.array([4.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.8,
        label="door",
        attributes={"landmark_source": "promoted_object", "promoted_from_object": True},
    )
    wp1 = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
        label="wp1",
        attributes={},
    )
    wp2 = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([7.5, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
        label="wp2",
        attributes={},
    )
    topo._nodes[wp1].step_id = 1
    topo._nodes[wp2].step_id = 2
    topo.add_edge(wp1, wp2, EdgeType.NAVIGABLE, weight=7.0)
    topo._maintain_spatial_structure_graph()

    adjacent = [
        (u, v, data)
        for u, v, data in topo.graph.edges(data=True)
        if data.get("edge_type") == EdgeType.ADJACENT_TO.value
    ]
    synth_id = topo._pair_portal_node_id(room_a, room_b)
    portal_edges = [
        (u, v) for u, v, _ in adjacent
        if synth_id in (u, v)
    ]
    assert len(portal_edges) >= 2
    room_room = [
        (u, v) for u, v, _ in adjacent
        if {u, v} == {room_a, room_b}
    ]
    assert len(room_room) == 0
    portal = topo.get_node(synth_id)
    assert portal is not None
    assert portal.attributes.get("structure_role") == "portal"
    assert portal.attributes.get("synthetic_portal")


def test_spatial_structure_filter_keeps_only_anchors():
    from conftopo.viz.memory_trace_viz import filter_topo_nodes

    nodes = [
        {"id": "wp", "type": "waypoint_visited", "position": [0, 0, 0], "confidence": 1.0, "attributes": {}},
        {"id": "room", "type": "room", "position": [10, 0, 0], "confidence": 0.7,
         "label": "bedroom", "attributes": {"summary_type": "room_region"}},
        {"id": "anchor", "type": "landmark", "position": [9, 0, 0], "confidence": 0.6,
         "label": "door", "attributes": {"landmark_source": "goal_hint", "structure_role": "portal", "structure_anchor": True}},
        {"id": "extra", "type": "landmark", "position": [8, 0, 0], "confidence": 0.5,
         "label": "tv", "attributes": {"landmark_source": "promoted_object"}},
        {"id": "env", "type": "landmark", "position": [8, 0, 1], "confidence": 0.6,
         "label": "hallway", "attributes": {"landmark_source": "environment"}},
    ]
    out = filter_topo_nodes(nodes, "spatial_structure")
    ids = {n["id"] for n in out}
    assert "wp" not in ids
    assert "room" in ids
    assert "anchor" in ids
    assert "extra" not in ids
    assert "env" not in ids


def test_viz_navigation_view_mode_keeps_only_waypoints():
    from conftopo.viz.memory_trace_viz import filter_topo_nodes

    nodes = [
        {"id": "wp", "type": "waypoint_visited", "position": [0, 0, 0], "attributes": {}},
        {"id": "fr", "type": "waypoint_frontier", "position": [1, 0, 0], "attributes": {}},
        {"id": "ca", "type": "waypoint_candidate", "position": [2, 0, 0], "attributes": {}},
        {"id": "room", "type": "room", "position": [5, 0, 0],
         "label": "kitchen", "attributes": {"summary_type": "room_region"}},
        {"id": "door", "type": "landmark", "position": [4, 0, 0],
         "label": "door", "attributes": {"landmark_role": "structural"}},
        {"id": "obj", "type": "object", "position": [3, 0, 0],
         "label": "chair", "attributes": {}},
    ]
    out = filter_topo_nodes(nodes, "navigation_view")
    ids = {n["id"] for n in out}
    assert ids == {"wp", "fr", "ca"}


def test_viz_structure_view_drops_semantic_landmarks_and_waypoints():
    from conftopo.viz.memory_trace_viz import filter_topo_nodes

    nodes = [
        {"id": "wp", "type": "waypoint_visited", "position": [0, 0, 0], "attributes": {}},
        {"id": "room", "type": "room", "position": [5, 0, 0],
         "label": "kitchen", "attributes": {"summary_type": "room_region"}},
        {"id": "portal", "type": "landmark", "position": [3, 0, 0],
         "label": "door", "attributes": {"structure_role": "portal",
                                          "synthetic_portal": True}},
        {"id": "struct_door", "type": "landmark", "position": [4, 0, 0],
         "label": "door", "attributes": {"landmark_source": "promoted_object",
                                          "landmark_role": "structural"}},
        {"id": "sem_vase", "type": "landmark", "position": [4.5, 0, 0],
         "label": "vase", "attributes": {"landmark_source": "promoted_object",
                                          "landmark_role": "semantic"}},
        {"id": "obj", "type": "object", "position": [3, 0, 0],
         "label": "chair", "attributes": {}},
    ]
    out = filter_topo_nodes(nodes, "structure_view")
    ids = {n["id"] for n in out}
    assert ids == {"room", "portal", "struct_door"}


def test_trace_view_separates_navigation_and_structure_layers():
    """End-to-end smoke check that the two viz views are disjoint and
    consistent with the core get_navigation_view / get_structure_view.

    This is the trace-level human-readable sanity check from the plan:
    given the topo map, navigation_view stays in the waypoint layer and
    structure_view stays in the room+structural layer with no overlap.
    """
    from conftopo.viz.memory_trace_viz import filter_topo_nodes

    topo = DynamicTopoMap()
    cur_vp = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9, attributes={"view_room_label": "kitchen"},
    )
    far_vp = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([6.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9, attributes={"view_room_label": "bedroom"},
    )
    topo.add_edge(cur_vp, far_vp, EdgeType.NAVIGABLE)
    kitchen = topo.add_node(
        NodeType.ROOM, position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7, label="kitchen",
        attributes={"summary_type": "room_region", "contains_labels": ["sink"]},
    )
    bedroom = topo.add_node(
        NodeType.ROOM, position=np.array([6.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7, label="bedroom",
        attributes={"summary_type": "room_region", "contains_labels": ["bed"]},
    )
    topo.add_node(
        NodeType.LANDMARK, position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="door",
        attributes={"landmark_role": "structural",
                    "landmark_source": "promoted_object"},
    )
    topo.add_node(
        NodeType.LANDMARK, position=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        confidence=0.6, label="vase",
        attributes={"landmark_role": "semantic"},
    )
    topo._maintain_spatial_structure_graph()

    # Snapshot via the core API and re-filter via the viz API.
    core_nav = {n["id"] for n in topo.get_navigation_view()["nodes"]}
    core_struct = {n["id"] for n in topo.get_structure_view()["nodes"]}
    snapshot_nodes, _ = topo._topo_dict_snapshot()

    viz_nav = {n["id"] for n in filter_topo_nodes(snapshot_nodes, "navigation_view")}
    viz_struct = {n["id"] for n in filter_topo_nodes(snapshot_nodes, "structure_view")}

    # Core and viz layer views agree exactly.
    assert viz_nav == core_nav
    assert viz_struct == core_struct
    # Navigation and structure layers are disjoint at the node level.
    assert viz_nav.isdisjoint(viz_struct)
    # Both waypoints survive in the navigation view.
    assert cur_vp in viz_nav and far_vp in viz_nav
    # At least one room summary survives in the structure view.
    snapshot_by_id = {n["id"]: n for n in snapshot_nodes}
    room_ids_in_struct = {
        nid for nid in viz_struct
        if snapshot_by_id[nid]["type"] == "room"
    }
    assert len(room_ids_in_struct) >= 1


def test_traversable_skeleton_skips_proximity_only_rooms():
    topo = DynamicTopoMap()
    topo.add_node(
        NodeType.ROOM,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="kitchen",
        attributes={"summary_type": "room_region"},
    )
    topo.add_node(
        NodeType.ROOM,
        position=np.array([8.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="hallway",
        attributes={"summary_type": "room_region"},
    )
    topo.add_node(
        NodeType.ROOM,
        position=np.array([0.0, 0.0, 20.0], dtype=np.float32),
        confidence=0.7,
        label="garage",
        attributes={"summary_type": "room_region"},
    )
    topo._maintain_spatial_structure_graph()
    adjacent = [
        (u, v)
        for u, v, data in topo.graph.edges(data=True)
        if data.get("edge_type") == EdgeType.ADJACENT_TO.value
    ]
    assert len(adjacent) == 0


def test_compress_collinear_corridor_waypoints():
    topo = DynamicTopoMap()
    waypoint_ids = []
    for idx in range(9):
        waypoint_ids.append(topo.add_node(
            NodeType.WAYPOINT_VISITED,
            position=np.array([float(idx), 0.0, 0.0], dtype=np.float32),
            confidence=1.0,
            label=f"wp{idx}",
        ))
    for idx in range(8):
        topo.add_edge(waypoint_ids[idx], waypoint_ids[idx + 1], EdgeType.NAVIGABLE, weight=1.0)

    removed = topo.compress_distant_waypoints(np.array([8.0, 0.0, 0.0], dtype=np.float32))
    assert removed >= 3
    assert len(topo.get_visited()) < 9
    assert topo.shortest_path(waypoint_ids[0], waypoint_ids[-1]) is not None


def test_entrance_waypoints_survive_compression():
    topo = DynamicTopoMap()
    topo.add_node(
        NodeType.ROOM,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="hallway",
        attributes={"summary_type": "room_region"},
    )
    topo.add_node(
        NodeType.ROOM,
        position=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.7,
        label="bedroom",
        attributes={"summary_type": "room_region"},
    )
    left = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
        label="wp_left",
    )
    entrance = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
        label="wp_entrance",
    )
    right = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([9.0, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
        label="wp_right",
    )
    mid = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([6.0, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
        label="wp_mid",
    )
    for a, b in ((left, entrance), (entrance, mid), (mid, right)):
        topo.add_edge(a, b, EdgeType.NAVIGABLE, weight=1.0)

    topo.compress_distant_waypoints(np.array([9.0, 0.0, 0.0], dtype=np.float32))
    assert topo.get_node(entrance) is not None
    assert topo.get_node(entrance).attributes.get("waypoint_role") == "entrance"


def test_near_waypoints_not_compressed():
    topo = DynamicTopoMap()
    near_a = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
    )
    near_b = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
    )
    far = topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([12.0, 0.0, 0.0], dtype=np.float32),
        confidence=1.0,
    )
    topo.add_edge(near_a, near_b, EdgeType.NAVIGABLE)
    topo.add_edge(near_b, far, EdgeType.NAVIGABLE)

    removed = topo.compress_distant_waypoints(np.array([0.5, 0.0, 0.0], dtype=np.float32))
    assert removed == 0
    assert topo.get_node(near_a) is not None
    assert topo.get_node(near_b) is not None


def test_dedupe_room_summary_labels():
    topo = DynamicTopoMap()
    room_id = topo.add_node(
        NodeType.ROOM,
        position=np.zeros(3, dtype=np.float32),
        confidence=0.7,
        label="hallway",
        attributes={
            "summary_type": "room_region",
            "contains_labels": ["door", "door", "tv", "TV", "chair"],
        },
    )
    topo._dedupe_room_summaries()
    room = topo.get_node(room_id)
    assert room.attributes["contains_labels"] == ["door", "tv", "chair"]
    assert room.attributes["label_counts"]["door"] == 2


def test_build_spatial_skeleton_creates_portal_chain():
    from conftopo.viz.memory_trace_viz import build_spatial_skeleton

    nodes = [
        {"id": "r1", "type": "room", "position": [0, 0, 0], "confidence": 0.7,
         "label": "kitchen", "attributes": {"summary_type": "room_region"}},
        {"id": "r2", "type": "room", "position": [8, 0, 0], "confidence": 0.7,
         "label": "hallway", "attributes": {"summary_type": "room_region"}},
        {"id": "portal", "type": "landmark", "position": [4, 0, 0], "confidence": 0.8,
         "label": "door", "attributes": {"landmark_source": "promoted_object"}},
        {"id": "noise", "type": "landmark", "position": [1, 0, 1], "confidence": 0.4,
         "label": "tv", "attributes": {"landmark_source": "promoted_object"}},
        {"id": "w1", "type": "waypoint_visited", "position": [0.5, 0, 0], "step_id": 1, "attributes": {}},
        {"id": "w2", "type": "waypoint_visited", "position": [7.5, 0, 0], "step_id": 2, "attributes": {}},
    ]
    edges = [{"source": "w1", "target": "w2", "type": "navigable", "weight": 7.0}]
    sk_nodes, sk_edges = build_spatial_skeleton(nodes, edges)
    sk_ids = {n["id"] for n in sk_nodes}
    assert "r1" in sk_ids and "r2" in sk_ids
    portal_nodes = [
        n for n in sk_nodes
        if (n.get("attributes") or {}).get("synthetic_portal")
    ]
    assert len(portal_nodes) == 1
    assert portal_nodes[0]["id"].startswith("portal::")
    assert "noise" not in sk_ids
    adjacent = [e for e in sk_edges if e.get("type") == "adjacent_to"]
    assert len(adjacent) == 2
    assert not any(
        e["source"] in ("r1", "r2") and e["target"] in ("r1", "r2")
        for e in adjacent
    )


def test_distance_layers_filter_tiers():
    from conftopo.viz.memory_trace_viz import filter_topo_nodes, node_distance_tier

    agent = np.zeros(3, dtype=np.float32)
    nodes = [
        {"id": "wp", "type": "waypoint_visited", "position": [0, 0, 0], "confidence": 1.0, "attributes": {}},
        {"id": "near_obj", "type": "object", "position": [1, 0, 0], "confidence": 0.8,
         "label": "chair", "attributes": {"granularity": "object"}},
        {"id": "mid_lm", "type": "landmark", "position": [5, 0, 0], "confidence": 0.7,
         "label": "door", "attributes": {"landmark_source": "goal_hint", "granularity": "landmark"}},
        {"id": "env_lm", "type": "landmark", "position": [5, 0, 1], "confidence": 0.7,
         "label": "hallway", "attributes": {"landmark_source": "environment"}},
        {"id": "summary", "type": "room", "position": [12, 0, 0], "confidence": 0.6,
         "label": "bedroom", "attributes": {"summary_type": "room_region", "contains_labels": ["chair"]}},
        {"id": "folded", "type": "object", "position": [6, 0, 0], "confidence": 0.4,
         "label": "table", "attributes": {"granularity": "landmark", "folded": True}},
    ]
    assert node_distance_tier(nodes[1], agent) == "near"
    assert node_distance_tier(nodes[2], agent) == "mid"
    assert node_distance_tier(nodes[3], agent) is None
    assert node_distance_tier(nodes[4], agent) == "room"
    filtered = filter_topo_nodes(nodes, "distance_layers", agent_pos=agent, near_radius=3.0, far_radius=10.0)
    ids = {n["id"] for n in filtered}
    assert "near_obj" in ids
    assert "mid_lm" in ids
    assert "summary" in ids
    assert "env_lm" not in ids
    assert "folded" not in ids


def test_reanchor_room_summaries_from_waypoint_labels():
    topo = DynamicTopoMap()
    hallway = topo.add_node(
        NodeType.ROOM,
        position=np.array([20.0, 0.0, 20.0], dtype=np.float32),
        confidence=0.7,
        label="hallway",
        attributes={"summary_type": "room_region", "contains_labels": ["tv"]},
    )
    bedroom = topo.add_node(
        NodeType.ROOM,
        position=np.array([-20.0, 0.0, -20.0], dtype=np.float32),
        confidence=0.7,
        label="bedroom",
        attributes={"summary_type": "room_region", "contains_labels": ["bed"]},
    )
    for idx, pos in enumerate(
        (
            np.array([1.0, 0.0, 2.0], dtype=np.float32),
            np.array([2.0, 0.0, 3.0], dtype=np.float32),
        )
    ):
        topo.add_node(
            NodeType.WAYPOINT_VISITED,
            position=pos,
            confidence=1.0,
            attributes={"view_room_label": "hallway"},
        )
    topo.add_node(
        NodeType.WAYPOINT_VISITED,
        position=np.array([-1.0, 0.0, -2.0], dtype=np.float32),
        confidence=1.0,
        attributes={"view_room_label": "bedroom"},
    )
    synced = topo._sync_structure_rooms_from_waypoints()
    by_base = {
        str(room.attributes.get("base_label") or room.label.split("@")[0]).strip().lower(): room
        for room in synced
    }
    hall = by_base["hallway"]
    bed = by_base["bedroom"]
    assert float(np.linalg.norm(hall.position - np.array([1.5, 0.0, 2.5], dtype=np.float32))) < 0.2
    assert float(np.linalg.norm(bed.position - np.array([-1.0, 0.0, -2.0], dtype=np.float32))) < 0.2


def test_same_label_spatial_instances_create_multiple_rooms():
    from conftopo.viz.memory_trace_viz import build_spatial_skeleton

    nodes = [
        {"id": "w1", "type": "waypoint_visited", "position": [0.0, 0.0, 0.0], "step_id": 1,
         "attributes": {"view_room_label": "hallway"}},
        {"id": "w2", "type": "waypoint_visited", "position": [1.0, 0.0, 0.0], "step_id": 2,
         "attributes": {"view_room_label": "hallway"}},
        {"id": "w3", "type": "waypoint_visited", "position": [8.0, 0.0, -5.0], "step_id": 3,
         "attributes": {"view_room_label": "hallway"}},
        {"id": "w4", "type": "waypoint_visited", "position": [8.5, 0.0, -5.2], "step_id": 4,
         "attributes": {"view_room_label": "hallway"}},
    ]
    edges = [
        {"source": "w1", "target": "w2", "type": "navigable", "weight": 1.0},
        {"source": "w2", "target": "w3", "type": "navigable", "weight": 7.0},
        {"source": "w3", "target": "w4", "type": "navigable", "weight": 1.0},
    ]
    sk_nodes, sk_edges = build_spatial_skeleton(nodes, edges)
    hallway_rooms = [
        n for n in sk_nodes
        if n.get("type") == "room" and str(n.get("label", "")).startswith("hallway@")
    ]
    assert len(hallway_rooms) == 2
    assert any("east" in n["label"] or "west" in n["label"] for n in hallway_rooms)
    assert len(sk_edges) >= 2


def test_skeleton_uses_waypoint_clusters_without_semantic_annotations():
    from conftopo.core.gt_room_regions import rooms_from_gt_regions
    from conftopo.viz.memory_trace_viz import build_spatial_skeleton, load_origin_world

    trace_path = Path(__file__).resolve().parents[2] / "data/logs/goat_topo/phase3_loop_trace_v6.json"
    if not trace_path.is_file():
        return
    import json
    trace = json.loads(trace_path.read_text())
    origin, _ = load_origin_world(trace)
    st = trace["steps"][-1]
    nodes = st["topo"]["nodes"]
    assert rooms_from_gt_regions(nodes, origin, scene_file=trace.get("scene_file")) == []
    sk_nodes, sk_edges = build_spatial_skeleton(
        nodes, st["topo"]["edges"],
        origin_world=origin, scene_file=trace.get("scene_file"),
    )
    room_ids = {n["id"] for n in sk_nodes if n.get("type") == "room"}
    assert not any(rid.startswith("gt_room::") for rid in room_ids)
    assert len(room_ids) >= 3
    assert len(sk_edges) >= 4


def test_region_instances_split_distant_same_label():
    from conftopo.core.region_rooms import region_instances_from_waypoints

    waypoints = [
        {"id": "w1", "position": [1.0, 0.0, 1.0], "attributes": {"view_room_label": "hallway"}},
        {"id": "w2", "position": [1.5, 0.0, 1.2], "attributes": {"view_room_label": "hallway"}},
        {"id": "w3", "position": [9.0, 0.0, -5.0], "attributes": {"view_room_label": "hallway"}},
    ]
    instances = region_instances_from_waypoints(waypoints)
    assert len(instances) == 2
    assert all(inst["base_label"] == "hallway" for inst in instances)


def test_traversable_transitions_use_view_room_label():
    from conftopo.viz.memory_trace_viz import _traversable_room_transitions_dict

    nodes = [
        {"id": "hall", "type": "room", "position": [50, 0, 50], "confidence": 0.7,
         "label": "hallway", "attributes": {"summary_type": "room_region"}},
        {"id": "bed", "type": "room", "position": [-50, 0, -50], "confidence": 0.7,
         "label": "bedroom", "attributes": {"summary_type": "room_region"}},
        {"id": "w1", "type": "waypoint_visited", "position": [0, 0, 0], "step_id": 1,
         "attributes": {"view_room_label": "hallway"}},
        {"id": "w2", "type": "waypoint_visited", "position": [1, 0, 0], "step_id": 2,
         "attributes": {"view_room_label": "bedroom"}},
    ]
    edges = [{"source": "w1", "target": "w2", "type": "navigable", "weight": 1.0}]
    transitions = _traversable_room_transitions_dict(nodes, edges, summary_radius=5.0)
    assert len(transitions) == 1
    assert transitions[0][0]["attributes"]["base_label"] == "hallway"
    assert transitions[0][1]["attributes"]["base_label"] == "bedroom"
