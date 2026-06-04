from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
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
from conftopo.core.instruction_graph import GoalNode, InstructionGraph
from conftopo.perception import ClipRuntimeEncoder
from conftopo.navigation import CollisionLikeTracker, PathfinderExecutor


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
    return [{"label": label, "score": float(score)} for label, score in scores[:k]]


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
            "attributes": dict(node.attributes),
        })
    edges = []
    for u, v, data in topo.graph.edges(data=True):
        edge_type = data.get("edge_type", "")
        if hasattr(edge_type, "value"):
            edge_type = edge_type.value
        edges.append({"source": u, "target": v, "type": edge_type, "weight": float(data.get("weight", 1.0))})
    return {"nodes": nodes, "edges": edges}


def landmark_names(raw) -> list[str]:
    names = []
    if isinstance(raw, dict):
        raw = list(raw.keys())
    for item in raw or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("object") or item.get("label")
            if name:
                names.append(str(name))
        else:
            names.append(str(item))
    return names


def current_goal_summary(ig) -> dict[str, Any]:
    goal = ig.get_current_goal()
    if isinstance(goal, GoalNode):
        return {
            "target_object": goal.target_object,
            "goal_type": goal.goal_type,
            "attributes": goal.attributes,
            "room_prior": goal.room_prior,
            "landmarks": landmark_names(goal.landmarks),
            "has_target_embedding": goal.target_embedding is not None,
            "target_embedding_dim": None if goal.target_embedding is None else int(goal.target_embedding.shape[-1]),
        }
    return {"goal": str(goal)}


GOAL_MODALITY_ALIASES = {
    "object": {"object", "category"},
    "instruction": {"instruction", "description"},
    "description": {"instruction", "description"},
    "image": {"image"},
}


def goal_node_modality(goal: GoalNode) -> str:
    goal_type = (goal.goal_type or "").lower()
    if goal_type == "category":
        return "object"
    if goal_type == "description":
        return "instruction"
    return goal_type or "unknown"


def available_goal_modalities(ig: InstructionGraph) -> list[str]:
    seen = []
    for goal in ig.goal_nodes:
        modality = goal_node_modality(goal)
        if modality not in seen:
            seen.append(modality)
    return seen


def select_goal_modality(ig: InstructionGraph, requested: str) -> tuple[InstructionGraph, dict[str, Any]]:
    requested = (requested or "auto").lower()
    if requested == "auto":
        current = ig.get_current_goal()
        return ig, {
            "requested_goal_modality": "auto",
            "selected_goal_modality": goal_node_modality(current) if isinstance(current, GoalNode) else "unknown",
            "selected_goal_index": ig.current_idx,
            "available_goal_modalities": available_goal_modalities(ig),
        }

    aliases = GOAL_MODALITY_ALIASES[requested]
    for idx, goal in enumerate(ig.goal_nodes):
        if (goal.goal_type or "").lower() in aliases:
            selected = InstructionGraph(goal_type=ig.goal_type, goal_nodes=[goal])
            selected.set_current_goal_by_index(0)
            return selected, {
                "requested_goal_modality": requested,
                "selected_goal_modality": goal_node_modality(goal),
                "selected_goal_index": idx,
                "available_goal_modalities": available_goal_modalities(ig),
            }

    available = ", ".join(available_goal_modalities(ig)) or "none"
    raise ValueError(f"Goal modality '{requested}' not found for this episode. Available: {available}")


def planar_distance(a, b) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm((aa - bb)[[0, 2]]))


def semantic_decisions(perception: dict, thresholds: dict) -> dict[str, list[dict[str, Any]]]:
    mapping = {
        'objects': ('goal_scores', 'object'),
        'rooms': ('room_scores', 'room'),
        'landmarks': ('landmark_scores', 'landmark'),
    }
    out = {}
    for name, (score_key, threshold_key) in mapping.items():
        threshold = float(thresholds.get(threshold_key, 0.0))
        rows = []
        for item in perception.get(score_key, [])[:5]:
            score = float(item.get('score', 0.0))
            rows.append({
                'label': item.get('label', ''),
                'score': score,
                'threshold': threshold,
                'decision': 'accepted' if score >= threshold else 'below_threshold',
            })
        out[name] = rows
    return out


def build_selection_debug(target_node_id, candidate_scores, perception, thresholds, sticky_debug):
    ranked = sorted(candidate_scores, key=lambda x: float(x.get('score', 0.0)), reverse=True)
    selected_rank = None
    for idx, item in enumerate(ranked, start=1):
        if item.get('node_id') == target_node_id:
            selected_rank = idx
            break
    return {
        'selected_target': target_node_id,
        'selected_rank': selected_rank,
        'top_candidate_scores': ranked[:5],
        'all_candidate_scores': ranked,
        'semantic_decisions': semantic_decisions(perception, thresholds),
        'sticky_debug': sticky_debug or {},
        'skipped_candidates': (sticky_debug or {}).get('skipped_candidates', []),
        'blocked_targets': (sticky_debug or {}).get('blocked_targets', {}),
        'navigation_event': (sticky_debug or {}).get('navigation_event', {}),
        'navigation_debug': (sticky_debug or {}).get('navigation_debug', {}),
        'reachable_candidates': (sticky_debug or {}).get('reachable_candidates', []),
        'unreachable_candidates': (sticky_debug or {}).get('unreachable_candidates', []),
        'thresholds': thresholds,
    }


def max_score(trace: dict, key: str) -> float:
    vals = []
    for st in trace.get("steps", []):
        scores = st.get("perception", {}).get(key, [])
        if scores:
            vals.append(float(scores[0]["score"]))
    return max(vals) if vals else 0.0



def count_target_switches(steps: list[dict[str, Any]]) -> int:
    prev = None
    switches = 0
    for st in steps:
        cur = st.get("target_node_id")
        if prev is not None and cur != prev:
            switches += 1
        prev = cur
    return switches


def path_length(steps: list[dict[str, Any]]) -> float:
    total = 0.0
    prev = None
    for st in steps:
        pos = st.get("position")
        if pos is not None and prev is not None:
            total += float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - np.asarray(prev, dtype=np.float32)))
        prev = pos
    return total


def memory_reuse_count(steps: list[dict[str, Any]]) -> int:
    count = 0
    for st in steps:
        target = st.get("target_node_id") or ""
        if target.startswith(("obj_", "roo_", "lan_")):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene", default="4ok3usBNeis")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until the agent stops")
    parser.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    parser.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--goal-modality", choices=["auto", "object", "instruction", "description", "image"], default="auto")
    parser.add_argument("--output", default="data/logs/goat_topo/topo_trace_semantic.json")
    parser.add_argument("--frame-dir", default=None)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--object-threshold", type=float, default=None)
    parser.add_argument("--room-threshold", type=float, default=None)
    parser.add_argument("--landmark-threshold", type=float, default=None)
    parser.add_argument("--use-placeholder-embed", action="store_true")
    args = parser.parse_args()

    dataset_dir = ROOT / args.dataset_dir
    scene_path, episode = pick_episode(dataset_dir, args.split, args.scene, args.episode_index)
    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)
    ig = load_goal_graph(ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"])
    ig, goal_modality_debug = select_goal_modality(ig, args.goal_modality)

    config = ConfTopoConfig()
    config.perception.clip_model = args.clip_model
    config.perception.clip_device = args.clip_device
    if args.object_threshold is not None:
        config.perception.object_threshold = args.object_threshold
    if args.room_threshold is not None:
        config.perception.room_threshold = args.room_threshold
    if args.landmark_threshold is not None:
        config.perception.landmark_threshold = args.landmark_threshold

    agent = ConfTopoGOATAgent(config)
    agent.set_goal(ig)
    first_goal = ig.get_current_goal()
    if isinstance(first_goal, GoalNode):
        agent.set_new_goal(first_goal)

    encoder = None
    if not args.use_placeholder_embed:
        encoder = ClipRuntimeEncoder(config.perception.clip_model, config.perception.clip_device)
        agent.perceiver.room_text_embeds = encoder.encode_text(agent.perceiver.room_labels)
        if isinstance(first_goal, GoalNode):
            lm_names = landmark_names(first_goal.landmarks)
            if lm_names:
                lm_embeds = first_goal.landmark_embeddings
                if lm_embeds is None:
                    lm_embeds = encoder.encode_text(lm_names)
                agent.perceiver.set_landmark_labels(lm_names, lm_embeds)

    frame_dir = ROOT / args.frame_dir if args.frame_dir else None
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)

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
    origin = np.array(state.position, dtype=np.float32)
    nav_executor = PathfinderExecutor()
    collision_tracker = CollisionLikeTracker()
    previous_low_action = None

    trace: dict[str, Any] = {
        "split": args.split,
        "scene": args.scene,
        "episode_index": args.episode_index,
        "scene_file": str(scene_file),
        "episode_file": str(scene_path),
        "episode_id": episode["episode_id"],
        "origin_world": origin.round(4).tolist(),
        "start_rotation": episode["start_rotation"],
        "num_tasks": len(episode.get("tasks", [])),
        "tasks": episode.get("tasks", []),
        "goal_type": ig.goal_type,
        **goal_modality_debug,
        "current_goal": current_goal_summary(ig),
        "embedding_source": "placeholder" if args.use_placeholder_embed else f"clip:{args.clip_model}",
        "thresholds": {
            "object": config.perception.object_threshold,
            "room": config.perception.room_threshold,
            "landmark": config.perception.landmark_threshold,
        },
        "coordinate_frame": "episode_start_relative",
        "steps": [],
    }

    step = 0
    try:
        while args.max_steps <= 0 or step < args.max_steps:
            obs = sim.get_sensor_observations()
            state = sim_agent.get_state()
            rgb = obs.get("color_sensor")
            if rgb is not None and rgb.shape[-1] == 4:
                rgb_save = rgb[..., :3]
            else:
                rgb_save = rgb
            frame_rel = None
            if frame_dir is not None and rgb_save is not None:
                frame_path = frame_dir / f"rgb_{step:04d}.png"
                imageio.imwrite(frame_path, rgb_save)
                frame_rel = str(frame_path.relative_to(ROOT))
            rgb_embed = rgb_to_embedding(rgb) if args.use_placeholder_embed else encoder.encode_image(rgb)
            conf_obs = {
                "rgb": rgb,
                "rgb_embed": rgb_embed,
                "position": np.array(state.position, dtype=np.float32),
                "heading": quat_to_heading([state.rotation.real, *list(state.rotation.imag)]),
            }
            out = agent.step(conf_obs)
            rel_position = np.asarray(state.position, dtype=np.float32) - origin
            collision_debug = collision_tracker.update(previous_low_action, rel_position)
            navigation_event = {}
            navigation_debug = {}

            target = out.get("target_position")
            target_node_id = out.get("target_node_id")
            candidate_ids = out.get("candidate_ids", []) or []
            scores = np.asarray(out.get("scores", []), dtype=np.float32).tolist()
            perception = snapshot_perception(agent)

            reachable_debug = {"selected": None, "reachable_candidates": [], "unreachable_candidates": []}
            if candidate_ids:
                reachable_debug = nav_executor.select_reachable_candidate(sim, origin, agent.topo_map, candidate_ids, scores, top_k=5)
                selected = reachable_debug.get("selected")
                if selected is not None and selected.get("node_id") != target_node_id:
                    node = agent.topo_map.get_node(selected["node_id"])
                    if node is not None:
                        target_node_id = node.node_id
                        target = node.position.copy()

            if agent.should_stop():
                low_action = "stop"
                navigation_debug = {"reachable": True, "reason": "goal_similarity_stop", "best_goal_sim": float(perception.get("best_goal_sim", 0.0))}
            elif collision_debug.get("collision_like") and target_node_id is not None:
                navigation_event = agent.on_navigation_event(target_node_id, "collision_like")
                low_action = "target_reached"
                navigation_debug = {"reachable": False, "reason": "collision_like", "release_reason": "collision_like"}
            elif target is None:
                low_action = "stop"
                navigation_debug = {"reachable": False, "reason": "no_target"}
            else:
                low_action, navigation_debug = nav_executor.step(sim, np.asarray(target), origin, target_node_id=target_node_id)
                if low_action == "unreachable":
                    navigation_event = agent.on_navigation_event(target_node_id, navigation_debug.get("reason", "unreachable"))
                    fallback = reachable_debug.get("selected")
                    if fallback is not None and fallback.get("node_id") != target_node_id:
                        node = agent.topo_map.get_node(fallback["node_id"])
                        if node is not None:
                            target_node_id = node.node_id
                            target = node.position.copy()
                            low_action, navigation_debug = nav_executor.step(sim, np.asarray(target), origin, target_node_id=target_node_id)
                    if low_action == "unreachable":
                        low_action = "turn_left"
                if low_action == "target_reached":
                    target_dist = planar_distance(target, rel_position)
                    navigation_debug["final_target_distance"] = target_dist
                    if navigation_debug.get("reason") == "target_reached" or target_dist <= nav_executor.reach_radius:
                        navigation_event = agent.on_navigation_event(target_node_id, "target_reached")
                    else:
                        low_action = "move_forward"
                        navigation_debug["low_action"] = low_action
                        navigation_debug["reason"] = "intermediate_waypoint_reached"

            navigation_debug["collision_like"] = bool(collision_debug.get("collision_like"))
            navigation_debug["stuck_steps"] = int(collision_debug.get("stuck_steps", 0))
            navigation_debug["movement"] = collision_debug.get("movement")
            navigation_debug["reachable_candidates"] = reachable_debug.get("reachable_candidates", [])
            navigation_debug["unreachable_candidates"] = reachable_debug.get("unreachable_candidates", [])

            candidate_scores = [{"node_id": node_id, "score": float(score)} for node_id, score in zip(candidate_ids, scores)]
            sticky_debug = dict(out.get("sticky_debug", {}) or {})
            sticky_debug["navigation_debug"] = navigation_debug
            sticky_debug["reachable_candidates"] = reachable_debug.get("reachable_candidates", [])
            sticky_debug["unreachable_candidates"] = reachable_debug.get("unreachable_candidates", [])
            if navigation_event:
                sticky_debug["navigation_event"] = navigation_event
            if reachable_debug.get("unreachable_candidates"):
                existing_skipped = list(sticky_debug.get("skipped_candidates", []))
                for row in reachable_debug.get("unreachable_candidates", []):
                    existing_skipped.append({"node_id": row.get("node_id"), "type": row.get("type", ""), "reason": row.get("reason", "unreachable"), "score": row.get("score")})
                sticky_debug["skipped_candidates"] = existing_skipped
            thresholds = {
                "object": config.perception.object_threshold,
                "room": config.perception.room_threshold,
                "landmark": config.perception.landmark_threshold,
            }
            selection_debug = build_selection_debug(target_node_id, candidate_scores, perception, thresholds, sticky_debug)
            trace["steps"].append({
                "step": step,
                "rgb_frame": frame_rel,
                "agent_action": out.get("action"),
                "low_action": low_action,
                "target_node_id": target_node_id,
                "target_position": None if target is None else np.asarray(target).round(4).tolist(),
                "position": rel_position.round(4).tolist(),
                "heading": conf_obs["heading"],
                "candidate_ids": candidate_ids,
                "candidate_scores": candidate_scores,
                "sticky_debug": sticky_debug,
                "navigation_event": navigation_event,
                "navigation_debug": navigation_debug,
                "collision_like": bool(collision_debug.get("collision_like")),
                "stuck_steps": int(collision_debug.get("stuck_steps", 0)),
                "selection_debug": selection_debug,
                "perception": perception,
                "memory": agent.memory_stats,
                "topo": snapshot_topo(agent),
            })
            if low_action == "stop":
                break
            if low_action not in ("target_reached", "unreachable"):
                sim.step(low_action)
            previous_low_action = low_action
            step += 1
    finally:
        sim.close()

    trace["final_memory"] = agent.memory_stats
    trace["final_summary"] = {
        "object_nodes": agent.memory_stats["objects"],
        "room_nodes": agent.memory_stats["rooms"],
        "landmark_nodes": agent.memory_stats["landmarks"],
        "max_object_score": max_score(trace, "goal_scores"),
        "max_room_score": max_score(trace, "room_scores"),
        "max_landmark_score": max_score(trace, "landmark_scores"),
        "thresholds": trace["thresholds"],
        "episode_length": len(trace["steps"]),
        "collision_like_count": sum(1 for st in trace["steps"] if st.get("collision_like")),
        "frontier_visited_count": agent.memory_stats.get("consumed_frontiers", 0),
        "semantic_node_count": agent.memory_stats.get("objects", 0) + agent.memory_stats.get("rooms", 0) + agent.memory_stats.get("landmarks", 0),
        "memory_reuse_count": memory_reuse_count(trace["steps"]),
        "target_switch_count": count_target_switches(trace["steps"]),
        "path_length_relative": path_length(trace["steps"]),
        "sr_proxy": bool(agent.memory_stats["objects"] > 0 or agent.memory_stats["landmarks"] > 0),
        "spl_proxy": 0.0,
        "failure_reason": "" if (agent.memory_stats["objects"] > 0 and (not trace["current_goal"].get("landmarks") or agent.memory_stats["landmarks"] > 0)) else "semantic_node_missing_check_top_scores_and_thresholds",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_tolist(trace), indent=2))
    print(json.dumps({
        "ok": True,
        "output": str(output),
        "steps": len(trace["steps"]),
        "requested_goal_modality": trace["requested_goal_modality"],
        "selected_goal_modality": trace["selected_goal_modality"],
        "current_goal": trace["current_goal"],
        "final_memory": trace["final_memory"],
        "final_summary": trace["final_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
