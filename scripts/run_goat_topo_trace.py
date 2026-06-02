from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_goat_minimal import (
    ROOT,
    controller_step,
    find_scene_file,
    load_goal_graph,
    normalize_quat,
    pick_episode,
    quat_to_heading,
    rgb_to_embedding,
    make_sim,
)
from conftopo.agents.goat_agent import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception import ClipRuntimeEncoder


def _tolist(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_tolist(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tolist(v) for v in value)
    if isinstance(value, dict):
        return {k: _tolist(v) for k, v in value.items() if k != "per_view_goal_sims"}
    return value


def top_scores(scores: list, k: int = 5) -> list[dict[str, Any]]:
    out = []
    for label, score in scores[:k]:
        out.append({"label": label, "score": float(score)})
    return out


def snapshot_perception(agent: ConfTopoGOATAgent) -> dict[str, Any]:
    p = agent._cur_perception or {}
    return {
        "room_label": p.get("room_label", "unknown"),
        "room_confidence": float(p.get("room_confidence", 0.0)),
        "best_goal_sim": float(p.get("best_goal_sim", 0.0)),
        "best_landmark_sim": float(p.get("best_landmark_sim", 0.0)),
        "room_scores": top_scores(p.get("room_scores", [])),
        "goal_scores": top_scores(p.get("goal_scores", [])),
        "landmark_scores": top_scores(p.get("landmark_scores", [])),
        "best_view_idx": int(p.get("best_view_idx", 0)),
    }


def snapshot_topo(agent: ConfTopoGOATAgent) -> dict[str, Any]:
    topo = agent.topo_map
    nodes = []
    for node in topo._nodes.values():
        nodes.append({
            "id": node.node_id,
            "type": node.node_type.value,
            "position": np.asarray(node.position).round(4).tolist(),
            "confidence": float(node.confidence),
            "label": node.label,
            "step_id": int(node.step_id),
            "visit_count": int(node.visit_count),
        })
    edges = []
    for u, v, data in topo.graph.edges(data=True):
        edge_type = data.get("edge_type", "")
        if hasattr(edge_type, "value"):
            edge_type = edge_type.value
        edges.append({
            "source": u,
            "target": v,
            "type": edge_type,
            "weight": float(data.get("weight", 1.0)),
        })
    return {"nodes": nodes, "edges": edges}


def current_goal_summary(ig) -> dict[str, Any]:
    goal = ig.get_current_goal()
    if isinstance(goal, GoalNode):
        return {
            "target_object": goal.target_object,
            "goal_type": goal.goal_type,
            "attributes": goal.attributes,
            "room_prior": goal.room_prior,
            "landmarks": goal.landmarks,
            "has_target_embedding": goal.target_embedding is not None,
            "target_embedding_dim": None if goal.target_embedding is None else int(goal.target_embedding.shape[-1]),
        }
    return {"goal": str(goal)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene", default="4ok3usBNeis")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    parser.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--output", default="data/logs/goat_topo/topo_trace_semantic.json")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--use-placeholder-embed", action="store_true")
    args = parser.parse_args()

    dataset_dir = ROOT / args.dataset_dir
    scene_path, episode = pick_episode(dataset_dir, args.split, args.scene, args.episode_index)
    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)
    ig = load_goal_graph(ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"])

    config = ConfTopoConfig()
    config.perception.clip_model = args.clip_model
    config.perception.clip_device = args.clip_device
    agent = ConfTopoGOATAgent(config)
    agent.set_goal(ig)
    first_goal = ig.get_current_goal()
    if isinstance(first_goal, GoalNode):
        agent.set_new_goal(first_goal)

    encoder = None
    if not args.use_placeholder_embed:
        encoder = ClipRuntimeEncoder(config.perception.clip_model, config.perception.clip_device)
        agent.perceiver.room_text_embeds = encoder.encode_text(agent.perceiver.room_labels)

    sim = make_sim(scene_file)
    sim_agent = sim.initialize_agent(0)

    import habitat_sim
    state = habitat_sim.AgentState()
    state.position = np.array(episode["start_position"], dtype=np.float32)
    q = normalize_quat(episode["start_rotation"])
    if abs(q[0]) < 1e-6 and abs(q[2]) < 1e-6 and (abs(q[1]) > 1e-6 or abs(q[3]) > 1e-6):
        state.rotation = np.quaternion(q[3], q[0], q[1], q[2])
    else:
        state.rotation = np.quaternion(q[0], q[1], q[2], q[3])
    sim_agent.set_state(state)

    trace: dict[str, Any] = {
        "split": args.split,
        "scene_file": str(scene_file),
        "episode_file": str(scene_path),
        "episode_id": episode["episode_id"],
        "num_tasks": len(episode.get("tasks", [])),
        "tasks": episode.get("tasks", []),
        "goal_type": ig.goal_type,
        "current_goal": current_goal_summary(ig),
        "embedding_source": "placeholder" if args.use_placeholder_embed else f"clip:{args.clip_model}",
        "thresholds": {
            "object": config.perception.object_threshold,
            "room": config.perception.room_threshold,
            "landmark": config.perception.landmark_threshold,
        },
        "steps": [],
    }

    try:
        for step in range(args.max_steps):
            obs = sim.get_sensor_observations()
            state = sim_agent.get_state()
            rgb = obs.get("color_sensor")
            rgb_embed = rgb_to_embedding(rgb) if args.use_placeholder_embed else encoder.encode_image(rgb)
            conf_obs = {
                "rgb": rgb,
                "rgb_embed": rgb_embed,
                "position": np.array(state.position, dtype=np.float32),
                "heading": quat_to_heading([state.rotation.real, *list(state.rotation.imag)]),
            }
            out = agent.step(conf_obs)
            target = out.get("target_position")
            low_action = "stop" if target is None else controller_step(sim, np.asarray(target))
            candidate_ids = out.get("candidate_ids", []) or []
            scores = np.asarray(out.get("scores", []), dtype=np.float32).tolist()
            trace["steps"].append({
                "step": step,
                "agent_action": out.get("action"),
                "low_action": low_action,
                "target_node_id": out.get("target_node_id"),
                "target_position": None if target is None else np.asarray(target).round(4).tolist(),
                "position": np.asarray(state.position).round(4).tolist(),
                "candidate_ids": candidate_ids,
                "candidate_scores": [
                    {"node_id": node_id, "score": float(score)}
                    for node_id, score in zip(candidate_ids, scores)
                ],
                "perception": snapshot_perception(agent),
                "memory": agent.memory_stats,
                "topo": snapshot_topo(agent),
            })
            if low_action == "stop":
                break
            sim.step(low_action)
    finally:
        sim.close()

    trace["final_memory"] = agent.memory_stats
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_tolist(trace), indent=2))
    print(json.dumps({
        "ok": True,
        "output": str(output),
        "steps": len(trace["steps"]),
        "final_memory": trace["final_memory"],
    }, indent=2))


if __name__ == "__main__":
    main()
