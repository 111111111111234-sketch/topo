"""Run a minimal real GOAT-Bench episode through ConfTopo-GOAT.

This is a Phase 2 smoke test, not a metric-quality GOAT evaluation:
- loads a real GOAT episode json.gz
- loads a real HM3D scene in habitat-sim
- feeds real RGB/pose observations through ConfTopoGOATAgent
- executes a thin point-goal-style controller for a few steps
- writes a JSON trace proving the observe -> memory -> plan -> act loop ran
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conftopo.agents.goat_agent import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import GoalNode, InstructionGraph


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def normalize_quat(q: list[float]) -> np.ndarray:
    arr = np.array(q, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def quat_to_heading(q: list[float]) -> float:
    w, x, y, z = normalize_quat(q)
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
    return float(yaw)


def rgb_to_embedding(rgb: np.ndarray, dim: int = 512) -> np.ndarray:
    """Deterministic lightweight visual embedding placeholder."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    sample = rgb[:: max(1, rgb.shape[0] // 16), :: max(1, rgb.shape[1] // 16), :3]
    stats = np.concatenate([
        sample.mean(axis=(0, 1)),
        sample.std(axis=(0, 1)),
        np.percentile(sample.reshape(-1, 3), [10, 50, 90], axis=0).reshape(-1),
    ]).astype(np.float32)
    reps = int(np.ceil(dim / stats.size))
    emb = np.tile(stats, reps)[:dim]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def find_scene_file(scene_id: str, scene_root: Path) -> Path:
    scene_name = Path(scene_id).name.replace(".basis.glb", "")
    matches = sorted(scene_root.glob(f"**/{scene_name}.basis.glb"))
    if not matches:
        raise FileNotFoundError(f"Scene {scene_name} not found under {scene_root}")
    return matches[0]


def pick_episode(dataset_dir: Path, split: str, scene: str | None, episode_index: int) -> tuple[Path, dict[str, Any]]:
    content_dir = dataset_dir / split / "content"
    files = sorted(content_dir.glob("*.json.gz"))
    if scene:
        files = [p for p in files if p.stem.replace(".json", "") == scene]
    if not files:
        raise FileNotFoundError(f"No GOAT content files found in {content_dir}")
    for path in files:
        data = load_json_gz(path)
        episodes = data.get("episodes", [])
        if episode_index < len(episodes):
            return path, episodes[episode_index]
    raise IndexError(f"episode_index={episode_index} not available")


def load_goal_graph(goal_graph_dir: Path, split: str, scene_file: Path, episode_id: Any) -> InstructionGraph:
    path = goal_graph_dir / f"{split}_goal_graphs.json"
    with open(path) as f:
        all_goals = json.load(f)
    key = scene_file.name.replace(".json.gz", "") + "_" + str(episode_id)
    if key not in all_goals:
        raise KeyError(f"GoalGraph key not found: {key}")
    return InstructionGraph.from_dict(all_goals[key])


def make_sim(scene_file: Path):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_file)
    sim_cfg.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color_sensor"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [256, 256]
    sensor.position = [0.0, 1.25, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def angular_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def controller_step(sim, target: np.ndarray, origin: np.ndarray | None = None) -> str:
    state = sim.get_agent(0).get_state()
    pos = np.array(state.position, dtype=np.float32)
    if origin is not None:
        pos = pos - np.array(origin, dtype=np.float32)
    delta = target - pos
    if np.linalg.norm(delta[[0, 2]]) < 0.35:
        return "target_reached"
    target_heading = math.atan2(-float(delta[0]), -float(delta[2]))
    q = state.rotation
    heading = quat_to_heading([q.real, q.imag[0], q.imag[1], q.imag[2]])
    diff = angular_diff(target_heading, heading)
    if diff > 0.25:
        return "turn_left"
    if diff < -0.25:
        return "turn_right"
    return "move_forward"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene", default="4ok3usBNeis")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    parser.add_argument("--scene-root", default="data/scene_datasets/hm3d_val/val")
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--output", default="data/logs/goat_minimal/episode_trace.json")
    args = parser.parse_args()

    dataset_dir = ROOT / args.dataset_dir
    scene_path, episode = pick_episode(dataset_dir, args.split, args.scene, args.episode_index)
    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)
    ig = load_goal_graph(ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"])

    config = ConfTopoConfig()
    agent = ConfTopoGOATAgent(config)
    agent.set_goal(ig)
    first_goal = ig.get_current_goal()
    if isinstance(first_goal, GoalNode):
        agent.set_new_goal(first_goal)

    sim = make_sim(scene_file)
    sim_agent = sim.initialize_agent(0)

    import habitat_sim

    state = habitat_sim.AgentState()
    state.position = np.array(episode["start_position"], dtype=np.float32)
    q = normalize_quat(episode["start_rotation"])
    state.rotation = np.quaternion(q[0], q[1], q[2], q[3])
    sim_agent.set_state(state)
    origin = np.array(state.position, dtype=np.float32)

    trace: dict[str, Any] = {
        "split": args.split,
        "scene_file": str(scene_file),
        "episode_file": str(scene_path),
        "episode_id": episode["episode_id"],
        "num_tasks": len(episode.get("tasks", [])),
        "goal_type": ig.goal_type,
        "coordinate_frame": "episode_start_relative",
        "steps": [],
    }

    try:
        for step in range(args.max_steps):
            obs = sim.get_sensor_observations()
            state = sim_agent.get_state()
            rgb = obs.get("color_sensor")
            conf_obs = {
                "rgb": rgb,
                "rgb_embed": rgb_to_embedding(rgb),
                "position": np.array(state.position, dtype=np.float32),
                "heading": quat_to_heading([state.rotation.real, *list(state.rotation.imag)]),
            }
            out = agent.step(conf_obs)
            target = out.get("target_position")
            navigation_event = {}
            low_action = "stop" if target is None else controller_step(sim, np.asarray(target), origin=origin)
            if low_action == "target_reached":
                navigation_event = agent.on_navigation_event(out.get("target_node_id"), "target_reached")
            rel_position = np.asarray(state.position, dtype=np.float32) - origin
            trace["steps"].append({
                "step": step,
                "agent_action": out.get("action"),
                "low_action": low_action,
                "target_node_id": out.get("target_node_id"),
                "navigation_event": navigation_event,
                "target_position": None if target is None else np.asarray(target).round(4).tolist(),
                "position": rel_position.round(4).tolist(),
                "memory": agent.memory_stats,
            })
            if low_action == "stop":
                break
            if low_action != "target_reached":
                sim.step(low_action)
    finally:
        sim.close()

    trace["final_memory"] = agent.memory_stats
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(trace, f, indent=2)

    print(json.dumps({
        "ok": True,
        "output": str(output),
        "steps": len(trace["steps"]),
        "episode_id": trace["episode_id"],
        "scene_file": trace["scene_file"],
        "final_memory": trace["final_memory"],
    }, indent=2))


if __name__ == "__main__":
    main()
