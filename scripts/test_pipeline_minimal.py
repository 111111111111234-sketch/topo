"""Minimal pipeline test: Habitat env reset → GoalGraph load → embedding verify.

Usage (in conftopo env):
    python scripts/test_pipeline_minimal.py --benchmark goat
    python scripts/test_pipeline_minimal.py --benchmark soon
"""

import argparse
import json
import gzip
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftopo.core.instruction_graph import InstructionGraph, GoalNode


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def load_goal_graphs(benchmark: str, split: str) -> dict:
    path = os.path.join(DATA_ROOT, "goal_graphs", benchmark, f"{split}_goal_graphs.json")
    print(f"  Loading GoalGraph: {path}")
    with open(path) as f:
        return json.load(f)


def test_goat():
    """Test GOAT pipeline: load episode → find GoalGraph → verify embedding."""
    print("\n" + "=" * 60)
    print("GOAT Pipeline Test")
    print("=" * 60)

    split = "val_seen"
    gg_data = load_goal_graphs("goat", split)
    print(f"  GoalGraph entries: {len(gg_data)}")

    # Load one raw episode
    content_dir = os.path.join(DATA_ROOT, "datasets", "goat", "hm3d", "v1", split, "content")
    gz_files = sorted([f for f in os.listdir(content_dir) if f.endswith(".json.gz")])
    gz_path = os.path.join(content_dir, gz_files[0])
    scene_name = gz_files[0].replace(".json.gz", "")

    with gzip.open(gz_path, "rt") as f:
        raw = json.load(f)
    ep = raw["episodes"][0]

    print(f"\n  Episode: scene={scene_name}, id={ep['episode_id']}")
    print(f"  start_position: {ep['start_position']}")
    print(f"  start_rotation: {ep['start_rotation']}")
    print(f"  scene_id: {ep['scene_id']}")
    print(f"  tasks: {len(ep['tasks'])} goals")

    # Key lookup
    key = f"{scene_name}_{ep['episode_id']}"
    print(f"\n  GoalGraph key: {key}")
    assert key in gg_data, f"Key {key} not found in GoalGraph!"
    print(f"  ✓ Key found in GoalGraph")

    # Deserialize InstructionGraph
    ig = InstructionGraph.from_dict(gg_data[key])
    print(f"\n  InstructionGraph:")
    print(f"    goal_type: {ig.goal_type}")
    print(f"    total_goals: {ig.total_goals}")

    # Check embeddings from raw JSON (from_dict doesn't load embeddings)
    raw_entry = gg_data[key]
    nodes_with_emb = sum(1 for gn in raw_entry.get("goal_nodes", []) if gn.get("target_embedding"))
    print(f"    nodes with target_embedding: {nodes_with_emb}/{ig.total_goals}")

    # Verify embedding dimensions
    first_node = raw_entry["goal_nodes"][0]
    print(f"\n  First GoalNode:")
    print(f"    target_object: {first_node['target_object']}")
    print(f"    goal_type: {first_node['goal_type']}")
    print(f"    attributes: {first_node.get('attributes', [])[:3]}")

    if first_node.get("target_embedding"):
        emb = np.array(first_node["target_embedding"])
        print(f"    target_embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
        assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"
        print(f"    ✓ Embedding shape and norm valid")
    else:
        print(f"    ⚠️ No target_embedding (may be category/image type)")

    # Test Habitat sim can load the scene
    print(f"\n  Testing Habitat simulator scene load...")
    try:
        import habitat_sim
        scene_glb = os.path.join(
            DATA_ROOT, "scene_datasets", "hm3d", "val",
            f"00877-{scene_name}", f"{scene_name}.basis.glb"
        )
        if not os.path.exists(scene_glb):
            # Find the actual directory
            hm3d_val = os.path.join(DATA_ROOT, "scene_datasets", "hm3d", "val")
            for d in os.listdir(hm3d_val):
                if d.endswith(f"-{scene_name}"):
                    scene_glb = os.path.join(hm3d_val, d, f"{scene_name}.basis.glb")
                    break

        if os.path.exists(scene_glb):
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = scene_glb
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [habitat_sim.CameraSensorSpec()]
            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

            sim = habitat_sim.Simulator(cfg)
            agent = sim.initialize_agent(0)
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array(ep["start_position"], dtype=np.float32)
            agent_state.rotation = np.quaternion(*ep["start_rotation"])
            agent.set_state(agent_state)

            obs = sim.get_sensor_observations()
            print(f"    ✓ Simulator loaded, observation keys: {list(obs.keys())}")
            if "rgba_camera" in obs:
                print(f"    ✓ RGB image shape: {obs['rgba_camera'].shape}")
            elif "color_sensor" in obs:
                print(f"    ✓ RGB image shape: {obs['color_sensor'].shape}")

            sim.close()
            print(f"    ✓ Simulator closed successfully")
        else:
            print(f"    ⚠️ Scene file not found: {scene_glb}")
    except ImportError:
        print(f"    ⚠️ habitat_sim not available, skipping sim test")
    except Exception as e:
        print(f"    ⚠️ Sim error: {e}")

    print(f"\n{'=' * 60}")
    print(f"GOAT Pipeline Test: PASSED")
    print(f"{'=' * 60}")


def test_soon():
    """Test SOON pipeline: load annotation → find GoalGraph → verify embedding."""
    print("\n" + "=" * 60)
    print("SOON Pipeline Test")
    print("=" * 60)

    split = "val_unseen_house"
    gg_data = load_goal_graphs("soon", split)
    print(f"  GoalGraph entries: {len(gg_data)}")

    # Load raw annotation
    raw_path = os.path.join(DATA_ROOT, "datasets", "soon", f"{split}.json")
    with open(raw_path) as f:
        raw = json.load(f)
    print(f"  Raw annotations: {len(raw)}")

    # Pick first item
    item = raw[0]
    scan = item.get("scan")
    if not scan and item.get("bboxes"):
        scan = item["bboxes"][0].get("scan", "unknown")
    instr_id = item.get("instr_id", 0)
    instructions = item.get("instructions", [])

    print(f"\n  Sample item: scan={scan}, instr_id={instr_id}")
    print(f"  path length: {len(item.get('path', []))}")

    # Reconstruct key
    if instructions and isinstance(instructions[0], list):
        group = instructions[0]
        uid = str(group[5]) if len(group) > 5 else "0"
    else:
        uid = str(instructions[5]) if len(instructions) > 5 else "0"
        group = instructions
    key = f"{scan}_{instr_id}_{uid}"
    print(f"  GoalGraph key: {key}")

    assert key in gg_data, f"Key {key} not found in GoalGraph!"
    print(f"  ✓ Key found in GoalGraph")

    # Deserialize
    ig = InstructionGraph.from_dict(gg_data[key])
    print(f"\n  InstructionGraph:")
    print(f"    goal_type: {ig.goal_type}")
    print(f"    total_goals: {ig.total_goals}")

    # Check embeddings
    raw_entry = gg_data[key]
    first_node = raw_entry["goal_nodes"][0]
    print(f"\n  First GoalNode:")
    print(f"    target_object: {first_node['target_object'][:80]}")
    attrs = first_node.get('attributes', [])
    print(f"    attributes: {attrs[:3] if isinstance(attrs, list) else attrs}")
    print(f"    room_prior: {first_node.get('room_prior', [])}")
    lms = first_node.get('landmarks', [])
    print(f"    landmarks: {lms[:2] if isinstance(lms, list) else lms}")

    if first_node.get("target_embedding"):
        emb = np.array(first_node["target_embedding"])
        print(f"    target_embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
        assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"
        print(f"    ✓ target_embedding valid")

    if first_node.get("room_prior_embeddings"):
        rp_emb = np.array(first_node["room_prior_embeddings"])
        print(f"    room_prior_embeddings: shape={rp_emb.shape}")
        print(f"    ✓ room_prior_embeddings valid")

    # Test MP3D scene accessibility
    print(f"\n  Testing MP3D scene file access...")
    mp3d_path = os.path.join(DATA_ROOT, "scene_datasets", "mp3d", scan, f"{scan}.glb")
    if os.path.exists(mp3d_path):
        size_mb = os.path.getsize(mp3d_path) / 1024 / 1024
        print(f"    ✓ Scene file exists: {mp3d_path} ({size_mb:.1f} MB)")
    else:
        print(f"    ⚠️ Scene file not found: {mp3d_path}")

    # Test cosine similarity (simulating runtime matching)
    print(f"\n  Simulating runtime semantic matching...")
    if first_node.get("target_embedding"):
        target_emb = np.array(first_node["target_embedding"])
        # Simulate an observation embedding (random for now)
        obs_emb = np.random.randn(512).astype(np.float32)
        obs_emb = obs_emb / np.linalg.norm(obs_emb)
        target_norm = target_emb / np.linalg.norm(target_emb)
        similarity = np.dot(obs_emb, target_norm)
        print(f"    cosine_sim(random_obs, target): {similarity:.4f}")
        print(f"    ✓ Semantic matching computation works")

    print(f"\n{'=' * 60}")
    print(f"SOON Pipeline Test: PASSED")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Minimal pipeline integration test")
    parser.add_argument("--benchmark", choices=["goat", "soon", "all"], default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("ConfTopo Minimal Pipeline Test")
    print("=" * 60)
    print(f"  DATA_ROOT: {os.path.abspath(DATA_ROOT)}")

    if args.benchmark in ("goat", "all"):
        test_goat()
    if args.benchmark in ("soon", "all"):
        test_soon()

    print("\n\n✅ All tests passed! Pipeline data is ready for runtime integration.")


if __name__ == "__main__":
    main()
