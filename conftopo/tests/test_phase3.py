"""Phase 3 tests: heavy object grounding and reliable memory."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conftopo.agents.goat_agent import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.confidence import ConfidenceFactors, compute_semantic_confidence
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.heavy_perceiver import FakeGroundingDINOBackend, HeavyPerceiver


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
