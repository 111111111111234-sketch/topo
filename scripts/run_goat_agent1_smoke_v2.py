"""Smoke runner for refactored Agent1 (conftopo/agents/goat_agent_1.py).

Per-step trace includes topo / perception / selection_debug (same as v1 smoke).
With ``--viz``, runs ``scripts/visualize_goat_topo_trace.py`` for full topo/RGB/dual/audit outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conftopo.agents.goat_agent_1 import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception import (
    GoatModalityClipEncoder,
    encode_agent_image_goal_embed,
    encode_agent_rgb_embed,
)
from run_goat_minimal import (
    angular_diff,
    apply_episode_rotation,
    find_scene_file,
    load_json_gz,
    make_sim,
    pick_episode,
    quat_to_heading,
    rgb_to_embedding,
)
from run_goat_topo_trace import (
    build_selection_debug,
    snapshot_perception,
    snapshot_topo,
)

_INF = float("inf")

# Flatten refactored agent debug into v1-compatible step fields for visualization.
_DEBUG_FLAT_FROM_DEBUG = {
    "vlm_fresh": "fresh_vlm",
    "vlm_trigger_reason": "vlm_reason",
    "active_anchor_id": "active_object_node_id",
    "track_hits": "track_hits",
    "target_lost_steps": "target_lost_steps",
    "verify_hits": "verify_hits",
    "verify_attempts": "verify_attempts",
    "state_steps": "state_steps",
    "proposal_count": "proposal_count",
}


def _memory_stats(agent: ConfTopoGOATAgent) -> dict[str, Any]:
    topo = agent.topo_map
    nodes = list(topo._nodes.values())

    return {
        "total_nodes": len(nodes),
        "visited_waypoints": len(topo.get_nodes_by_type(NodeType.WAYPOINT_VISITED)),
        "frontiers": len(topo.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER)),
        "candidate_waypoints": len(topo.get_nodes_by_type(NodeType.WAYPOINT_CANDIDATE)),
        "objects": len(topo.get_nodes_by_type(NodeType.OBJECT)),
        "rooms": len(topo.get_nodes_by_type(NodeType.ROOM)),
        "landmarks": len(topo.get_nodes_by_type(NodeType.LANDMARK)),
        "hypotheses": len(agent.memory.hypothesis_pool),
        "object_memory_pool": len(agent.memory.object_memory_pool),
    }


def _snapshot_rgb(obs: dict[str, Any]) -> np.ndarray | None:
    rgb = obs.get("color_sensor")
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.array(arr, copy=True)


def controller_step(sim, target: np.ndarray, origin: np.ndarray | None = None) -> str:
    state = sim.get_agent(0).get_state()
    pos = np.array(state.position, dtype=np.float32)
    if origin is not None:
        pos -= np.array(origin, dtype=np.float32)
    delta = target - pos
    if np.linalg.norm(delta[[0, 2]]) < 0.35:
        return "target_reached"
    target_heading = math.atan2(-float(delta[0]), -float(delta[2]))
    q = state.rotation
    diff = angular_diff(
        target_heading,
        quat_to_heading([q.real, q.imag[0], q.imag[1], q.imag[2]]),
    )
    if diff > 0.25:
        return "turn_left"
    if diff < -0.25:
        return "turn_right"
    return "move_forward"


def _scene_basename(episode: dict[str, Any]) -> str:
    scene_id = str(episode.get("scene_id", ""))
    return scene_id.split("/")[-1].replace(".basis.glb", "").replace(".glb", "")


def _resolve_goal_entries(
    dataset_goals: dict[str, Any],
    episode: dict[str, Any],
    task_index: int,
) -> list[dict[str, Any]]:
    tasks = episode.get("tasks") or []
    if task_index >= len(tasks) or len(tasks[task_index]) < 2:
        return []
    goal_category, goal_type, *rest = tasks[task_index]
    goal_inst_id = rest[0] if rest else None

    matching = [
        entries
        for entries in dataset_goals.values()
        if isinstance(entries, list)
        and entries
        and isinstance(entries[0], dict)
        and entries[0].get("object_category") == goal_category
    ]
    if not matching:
        return []

    goal_entries = list(matching[0])
    scene_name = _scene_basename(episode)
    for child_cat in matching[0][0].get("children_object_categories") or []:
        child = dataset_goals.get(f"{scene_name}_{child_cat}")
        if isinstance(child, list):
            goal_entries.extend(child)

    if goal_type == "object":
        return [g for g in goal_entries if isinstance(g, dict)]
    if goal_inst_id is None:
        return []
    return [
        g
        for g in goal_entries
        if isinstance(g, dict) and str(g.get("object_id")) == str(goal_inst_id)
    ]


def _instance_positions(goal_entries: list[dict[str, Any]]) -> list[np.ndarray]:
    seen: set[tuple] = set()
    positions: list[np.ndarray] = []
    for g in goal_entries:
        pos_raw = g.get("position")
        if pos_raw is None:
            continue
        pos = np.asarray(pos_raw, dtype=np.float32)
        key = tuple(np.round(pos, 4).tolist())
        if key not in seen:
            seen.add(key)
            positions.append(pos)
    return positions


def _planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm((aa - bb)[[0, 2]]))


def _relative_position(world_pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    rel = np.asarray(world_pos, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    rel[1] = 0.0
    return rel


def _min_distance(agent_pos: np.ndarray, positions: list[np.ndarray]) -> float:
    if not positions:
        return _INF
    agent = np.asarray(agent_pos, dtype=np.float32)
    return min(_planar_distance(agent, p) for p in positions)


def _goal_summary(goal: GoalNode) -> dict[str, Any]:
    return {
        "target_object": goal.target_object,
        "goal_type": goal.goal_type,
        "attributes": goal.attributes,
        "room_prior": goal.room_prior,
        "landmarks": goal.landmarks or [],
        "has_target_embedding": goal.target_embedding is not None,
        "target_embedding_dim": None
        if goal.target_embedding is None
        else int(goal.target_embedding.shape[-1]),
    }


def _object_anchors(agent: ConfTopoGOATAgent) -> list[dict[str, Any]]:
    result = []
    for node in agent.topo_map._nodes.values():
        if node.node_type != NodeType.OBJECT:
            continue
        if node.attributes.get("semantic_role") != "object_anchor":
            continue
        attrs = node.attributes
        result.append(
            {
                "node_id": node.node_id,
                "label": node.label,
                "source": attrs.get("source"),
                "confidence": float(node.confidence),
                "anchor_waypoint_id": attrs.get("anchor_waypoint_id"),
                "seen_count": int(attrs.get("seen_count", 0) or 0),
                "last_seen_step": attrs.get("last_seen_step"),
            }
        )
    return result


def _candidate_scores_from_out(out: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    debug = out.get("debug") or {}
    top = list((debug.get("proposal_debug") or {}).get("top_proposals", []))
    if not top:
        selected = debug.get("selected_proposal")
        if isinstance(selected, dict):
            top = [selected]
    candidate_scores = [
        {"node_id": str(item.get("node_id", "")), "score": float(item.get("score", 0.0))}
        for item in top
    ]
    return [item["node_id"] for item in candidate_scores], candidate_scores


def _step_record(
    *,
    global_step: int,
    goal_index: int,
    goal_name: str,
    local_step: int,
    out: dict[str, Any],
    low_action: str,
    position: np.ndarray,
    world_position: np.ndarray,
    distance_to_target: float | None,
    goal_min_distance: float | None,
    memory: dict[str, Any],
    rgb_frame: str | None = None,
    topo: dict[str, Any] | None = None,
    sticky_debug: dict[str, Any] | None = None,
    perception: dict[str, Any] | None = None,
    selection_debug: dict[str, Any] | None = None,
    candidate_ids: list[str] | None = None,
    candidate_scores: list[dict[str, Any]] | None = None,
    heading: float | None = None,
) -> dict[str, Any]:
    debug = dict(out.get("debug") or {})
    state = str(debug.get("state", ""))
    selected = debug.get("selected_proposal") or {}

    target_pos = out.get("target_position")
    if target_pos is None and selected.get("target_position") is not None:
        target_pos = selected.get("target_position")

    rec: dict[str, Any] = {
        "step": global_step,
        "global_step": global_step,
        "goal_index": goal_index,
        "goal": goal_name,
        "local_step": local_step,
        "agent_action": out.get("action"),
        "low_action": low_action,
        "mode": out.get("mode")
        or ("verified_stop" if state == "STOP" or out.get("action") == "stop" else state.lower()),
        "nav_phase": state,
        "phase_reason": debug.get("transition_reason"),
        "state_steps": debug.get("state_steps"),
        "target_node_id": out.get("target_node_id") or debug.get("target_node_id"),
        "target_type": out.get("target_type") or debug.get("target_type"),
        "target_position": (
            None
            if target_pos is None
            else np.asarray(target_pos, dtype=np.float32).round(4).tolist()
        ),
        "position": position.round(4).tolist(),
        "world_position": world_position.round(4).tolist(),
        "distance_to_target": None
        if distance_to_target is None
        else round(float(distance_to_target), 4),
        "goal_min_distance": None
        if goal_min_distance is None
        else round(float(goal_min_distance), 4),
        "memory": memory,
        "sticky_debug": debug,
        "proposal_count": debug.get("proposal_count"),
        "proposal_type": selected.get("type"),
        "proposal_source": selected.get("source"),
        "proposal_score": selected.get("score"),
        "verify_hits": debug.get("verify_hits"),
        "verify_attempts": debug.get("verify_attempts"),
        "track_hits": debug.get("track_hits"),
        "target_lost_steps": debug.get("target_lost_steps"),
        "active_anchor_id": debug.get("active_object_node_id"),
        "vlm_fresh": debug.get("fresh_vlm"),
        "vlm_trigger_reason": debug.get("vlm_reason"),
        "vlm_goal_visible": debug.get("vlm_goal_visible"),
        "verified_stop": bool(state == "STOP" or out.get("action") == "stop"),
    }

    if rgb_frame is not None:
        rec["rgb_frame"] = rgb_frame
    if topo is not None:
        rec["topo"] = topo
    if sticky_debug is not None:
        rec["sticky_debug"] = sticky_debug
    if perception is not None:
        rec["perception"] = perception
    if selection_debug is not None:
        rec["selection_debug"] = selection_debug
    if candidate_ids is not None:
        rec["candidate_ids"] = candidate_ids
    if candidate_scores is not None:
        rec["candidate_scores"] = candidate_scores
    if heading is not None:
        rec["heading"] = round(float(heading), 4)
        rec["world_heading"] = round(float(heading), 4)

    if sticky_debug is None:
        rec["sticky_debug"] = debug
    for out_key, in_key in _DEBUG_FLAT_FROM_DEBUG.items():
        if in_key in debug and debug.get(in_key) is not None:
            rec[out_key] = debug.get(in_key)

    return rec


def _summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    steps = trace.get("steps", [])
    summaries = trace.get("task_summaries", [])

    phases = {s.get("nav_phase") for s in steps if s.get("nav_phase")}
    modes = {s.get("mode") for s in steps if s.get("mode")}

    return {
        "total_steps": len(steps),
        "vlm_fresh_steps": sum(1 for s in steps if s.get("vlm_fresh")),
        "vlm_goal_visible_steps": sum(1 for s in steps if s.get("vlm_goal_visible")),
        "vlm_trigger_reasons": sorted(
            {s.get("vlm_trigger_reason") for s in steps if s.get("vlm_trigger_reason")}
        ),
        "nav_phases": sorted(phases),
        "modes": sorted(modes),
        "saw_approach": any(s in phases for s in ("APPROACH", "VISUAL_APPROACH")),
        "saw_verify": "VERIFY_STOP" in phases,
        "saw_stop": "STOP" in phases or any(s.get("verified_stop") for s in steps),
        "goals_success": sum(1 for t in summaries if t.get("goal_success")),
    }


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke for refactored goat_agent_1.py")
    p.add_argument("--split", default="val_seen")
    p.add_argument("--scene", default="4ok3usBNeis")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--max-goals", type=int, default=3)
    p.add_argument("--steps-per-goal", type=int, default=120)
    p.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    p.add_argument("--scene-root", default="data/scene_datasets/hm3d_val/val")
    p.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    p.add_argument(
        "--output",
        default="data/logs/goat_topo/agent1_smoke_4ok3usBNeis_ep0_v2/trace.json",
    )
    p.add_argument(
        "--perception-backend", default="vlm", choices=["vlm", "clip_groundingdino"]
    )
    p.add_argument("--vlm-api-base", default="http://localhost:8000/v1")
    p.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--vlm-timeout", type=float, default=30.0)
    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--clip-image-model", default="RN50")
    p.add_argument("--clip-device", default="cuda")
    p.add_argument("--use-placeholder-embed", action="store_true")
    p.add_argument("--success-distance", type=float, default=1.0)
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--frame-dir", default=None)
    p.add_argument("--viz", action="store_true",
                   help="Run visualize_goat_topo_trace.py after trace is saved.")
    p.add_argument("--viz-stride", type=int, default=2)
    p.add_argument("--viz-fps", "--fps", type=int, default=6, dest="viz_fps")
    p.add_argument("--viz-out-dir", default=None,
                   help="Visualization output directory (default: <output_parent>/viz).")
    return p.parse_args()


def main() -> None:
    args = _build_args()

    scene_path, episode = pick_episode(
        ROOT / args.dataset_dir, args.split, args.scene, args.episode_index
    )
    dataset = load_json_gz(scene_path)
    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)

    from run_goat_topo_trace import load_goal_graph

    ig = load_goal_graph(
        ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"]
    )
    goals = [g for g in ig.goal_nodes if isinstance(g, GoalNode)][: args.max_goals]
    if not goals:
        raise RuntimeError("No goals found in goal graph")

    config = ConfTopoConfig()
    config.perception.backend = args.perception_backend
    config.perception.vlm_api_base = args.vlm_api_base
    config.perception.vlm_model = args.vlm_model
    config.perception.vlm_timeout = args.vlm_timeout
    config.perception.clip_device = args.clip_device
    config.perception.heavy_enabled = args.perception_backend == "vlm"

    agent = ConfTopoGOATAgent(config)

    encoder = None
    if not args.use_placeholder_embed:
        encoder = GoatModalityClipEncoder(
            args.clip_model, args.clip_image_model, args.clip_device
        )

    sim = make_sim(scene_file)
    if hasattr(agent, "set_pathfinder"):
        agent.set_pathfinder(sim)
    sim_agent = sim.initialize_agent(0)
    init_state = sim_agent.get_state()
    init_state.position = np.array(episode["start_position"], dtype=np.float32)
    apply_episode_rotation(init_state, episode)
    sim_agent.set_state(init_state)
    origin = np.array(init_state.position, dtype=np.float32)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_dir: Path | None = None
    if not args.no_save_frames:
        frame_dir = Path(args.frame_dir) if args.frame_dir else output.parent / "frames"
        if not frame_dir.is_absolute():
            frame_dir = ROOT / frame_dir
        frame_dir.mkdir(parents=True, exist_ok=True)

    trace: dict[str, Any] = {
        "agent": "goat_agent_1.ConfTopoGOATAgent",
        "smoke_type": "agent1_v2",
        "split": args.split,
        "scene": args.scene,
        "episode_index": args.episode_index,
        "scene_file": str(scene_file),
        "episode_file": str(scene_path),
        "episode_id": episode["episode_id"],
        "origin_world": origin.round(4).tolist(),
        "coordinate_frame": "episode_start_relative",
        "pose_sources": {
            "position": "episode_start_relative_xz_grounded",
            "world_position": "habitat_gt_world_position",
            "heading": "habitat_gt_heading",
        },
        "perception_backend": args.perception_backend,
        "max_goals": args.max_goals,
        "steps_per_goal": args.steps_per_goal,
        "goals": [g.target_object for g in goals],
        "current_goal": _goal_summary(goals[0]),
        "steps": [],
        "task_summaries": [],
        "goal_instances": [],
    }

    global_step = 0
    try:
        for goal_index, goal in enumerate(goals):
            agent.set_new_goal(goal)
            trace["current_goal"] = _goal_summary(goal)

            goal_entries = _resolve_goal_entries(dataset.get("goals", {}), episode, goal_index)
            instance_positions = _instance_positions(goal_entries)
            goal_object_ids = [
                str(g.get("object_id")) for g in goal_entries if g.get("object_id")
            ]
            goal_min_dist = _INF
            goal_stop_dist = _INF
            goal_stopped = False
            local_step = 0

            for local_step in range(args.steps_per_goal):
                obs = sim.get_sensor_observations()
                state = sim_agent.get_state()
                rgb = _snapshot_rgb(obs)
                if rgb is not None and int(rgb.max()) == 0:
                    obs = sim.get_sensor_observations()
                    rgb = _snapshot_rgb(obs)

                rgb_embed = encode_agent_rgb_embed(
                    encoder,
                    rgb,
                    agent,
                    use_placeholder=args.use_placeholder_embed,
                    placeholder_fn=rgb_to_embedding,
                )
                image_goal_embed = (
                    None
                    if args.use_placeholder_embed or rgb is None
                    else encode_agent_image_goal_embed(encoder, rgb, agent)
                )
                conf_obs = {
                    "rgb": rgb,
                    "rgb_embed": rgb_embed,
                    "image_goal_embed": image_goal_embed,
                    "position": np.array(state.position, dtype=np.float32),
                    "heading": quat_to_heading(
                        [state.rotation.real, *state.rotation.imag]
                    ),
                }

                out = agent.step(conf_obs)
                action = out.get("action")
                target = out.get("target_position")
                debug = out.get("debug") or {}
                phase = str(debug.get("state", ""))

                if action in ("move_forward", "turn_left", "turn_right"):
                    low_action = action
                elif action == "stop" or phase == "STOP":
                    low_action = "stop"
                    goal_stopped = True
                elif action == "navigate" and target is not None:
                    low_action = controller_step(sim, np.asarray(target), origin=origin)
                else:
                    low_action = "turn_left"

                if low_action == "target_reached" and hasattr(agent, "on_navigation_event"):
                    agent.on_navigation_event(out.get("target_node_id"), "target_reached")

                world_pos = np.asarray(state.position, dtype=np.float32)
                rel_pos = _relative_position(world_pos, origin)

                dist = _min_distance(world_pos, instance_positions) if instance_positions else None
                if dist is not None and dist < goal_min_dist:
                    goal_min_dist = dist
                if low_action == "stop" and dist is not None:
                    goal_stop_dist = dist

                frame_rel = None
                if frame_dir is not None and rgb is not None:
                    frame_path = frame_dir / f"rgb_{global_step:04d}.png"
                    imageio.imwrite(frame_path, rgb)
                    frame_rel = str(frame_path.relative_to(ROOT))

                perception = snapshot_perception(agent)
                candidate_ids, candidate_scores = _candidate_scores_from_out(out)
                sticky_debug = dict(debug)
                thresholds = {
                    "object": config.perception.object_threshold,
                    "room": config.perception.room_threshold,
                    "landmark": config.perception.landmark_threshold,
                }
                selection_debug = build_selection_debug(
                    out.get("target_node_id") or debug.get("target_node_id"),
                    candidate_scores,
                    perception,
                    thresholds,
                    sticky_debug,
                )

                trace["steps"].append(
                    _step_record(
                        global_step=global_step,
                        goal_index=goal_index,
                        goal_name=str(goal.target_object),
                        local_step=local_step,
                        out=out,
                        low_action=low_action,
                        position=rel_pos,
                        world_position=world_pos,
                        distance_to_target=dist,
                        goal_min_distance=None if goal_min_dist == _INF else goal_min_dist,
                        memory=_memory_stats(agent),
                        rgb_frame=frame_rel,
                        topo=snapshot_topo(agent),
                        sticky_debug=sticky_debug,
                        perception=perception,
                        selection_debug=selection_debug,
                        candidate_ids=candidate_ids,
                        candidate_scores=candidate_scores,
                        heading=float(conf_obs["heading"]),
                    )
                )
                global_step += 1

                if goal_stopped or low_action == "stop":
                    break
                if low_action != "target_reached":
                    sim.step(low_action)

            trace["task_summaries"].append(
                {
                    "goal_index": goal_index,
                    "target_object": goal.target_object,
                    "steps": local_step + 1,
                    "goal_stopped": goal_stopped,
                    "goal_stop_distance": None
                    if goal_stop_dist == _INF
                    else round(float(goal_stop_dist), 4),
                    "goal_min_distance": None
                    if goal_min_dist == _INF
                    else round(float(goal_min_dist), 4),
                    "goal_success": bool(
                        goal_stopped
                        and goal_stop_dist != _INF
                        and goal_stop_dist <= args.success_distance
                    ),
                    "goal_instance_count": len(instance_positions),
                    "goal_object_ids": goal_object_ids,
                    "goal_instance_positions_world": [
                        np.asarray(p, dtype=np.float32).round(4).tolist() for p in instance_positions
                    ],
                    "goal_instance_positions": [
                        _relative_position(p, origin).round(4).tolist() for p in instance_positions
                    ],
                    "object_anchors": _object_anchors(agent),
                }
            )
            trace["goal_instances"].append(
                {
                    "goal_index": goal_index,
                    "target_object": goal.target_object,
                    "positions_world": [
                        np.asarray(p, dtype=np.float32).round(4).tolist() for p in instance_positions
                    ],
                    "positions_relative": [
                        _relative_position(p, origin).round(4).tolist() for p in instance_positions
                    ],
                }
            )
    finally:
        sim.close()

    trace["final_memory"] = _memory_stats(agent)
    trace["final_object_anchors"] = _object_anchors(agent)
    trace["final_summary"] = {"episode_length": len(trace["steps"])}
    trace["smoke_summary"] = _summarize_trace(trace)

    output.write_text(json.dumps(trace, indent=2))

    viz_dir: Path | None = None
    if args.viz:
        viz_dir = Path(args.viz_out_dir) if args.viz_out_dir else output.parent / "viz"
        if not viz_dir.is_absolute():
            viz_dir = ROOT / viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/visualize_goat_topo_trace.py",
                "--trace",
                str(output.relative_to(ROOT)),
                "--out-dir",
                str(viz_dir.relative_to(ROOT)),
                "--stride",
                str(args.viz_stride),
                "--fps",
                str(args.viz_fps),
            ],
            cwd=ROOT,
            check=True,
        )

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "frame_dir": None if frame_dir is None else str(frame_dir),
                "viz_dir": None if viz_dir is None else str(viz_dir),
                "smoke_summary": trace["smoke_summary"],
                "goals": trace["goals"],
                "task_summaries": [
                    {
                        "target": t["target_object"],
                        "steps": t["steps"],
                        "goal_stopped": t["goal_stopped"],
                        "goal_stop_distance": t.get("goal_stop_distance"),
                        "goal_min_distance": t.get("goal_min_distance"),
                        "goal_success": t.get("goal_success"),
                        "object_anchors": len(t["object_anchors"]),
                    }
                    for t in trace["task_summaries"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
