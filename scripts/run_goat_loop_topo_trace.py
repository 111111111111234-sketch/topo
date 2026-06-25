from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from run_goat_minimal import (
    ROOT,
    find_scene_file,
    load_goal_graph,
    normalize_quat,
    pick_episode,
    quat_to_heading,
    rgb_to_embedding,
    make_sim,
)
from run_goat_topo_trace import (
    _tolist,
    configure_landmark_perception,
    count_target_switches,
    current_goal_summary,
    max_score,
    memory_reuse_count,
    path_length,
    parse_env_landmarks,
    select_goal_modality,
    snapshot_perception,
    snapshot_topo,
)
from conftopo.agents import ConfTopoGOATAgent
from conftopo.agents.goat_agent_etpnav import ConfTopoGOATAgentETPNav, ETPGoatConfig
from conftopo.agents.goat_agent_etpnav_clean import ConfTopoGOATAgentCleanETPNav, CleanETPGoatConfig
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import GoalNode
from conftopo.navigation import CollisionLikeTracker, PathfinderExecutor
from conftopo.perception import GoatModalityClipEncoder, encode_agent_rgb_embed


def world_to_relative(position_world: np.ndarray, origin_world: np.ndarray) -> np.ndarray:
    return np.asarray(position_world, dtype=np.float32) - np.asarray(origin_world, dtype=np.float32)


def shortest_path_points(sim, start_world: np.ndarray, end_world: np.ndarray) -> tuple[float, list[np.ndarray]]:
    from habitat_sim.nav import ShortestPath

    path = ShortestPath()
    path.requested_start = np.asarray(start_world, dtype=np.float32)
    path.requested_end = np.asarray(end_world, dtype=np.float32)
    found = bool(sim.pathfinder.find_path(path))
    points = [np.asarray(p, dtype=np.float32) for p in getattr(path, "points", [])]
    distance = float(getattr(path, "geodesic_distance", float("inf")))
    if not found or not np.isfinite(distance) or not points:
        return float("inf"), []
    return distance, points


def densify_points(points: list[np.ndarray], spacing: float = 0.75) -> list[np.ndarray]:
    if not points:
        return []
    dense = [np.asarray(points[0], dtype=np.float32)]
    for point in points[1:]:
        start = dense[-1]
        end = np.asarray(point, dtype=np.float32)
        dist = float(np.linalg.norm((end - start)[[0, 2]]))
        if dist <= spacing:
            dense.append(end)
            continue
        n = int(np.ceil(dist / spacing))
        for i in range(1, n + 1):
            dense.append((start + (end - start) * (i / n)).astype(np.float32))
    return dense


def path_stays_on_floor(points: list[np.ndarray], origin_world: np.ndarray, floor_y_tolerance: float) -> bool:
    if not points:
        return False
    ys = np.asarray([p[1] for p in points], dtype=np.float32)
    return bool(np.max(np.abs(ys - float(origin_world[1]))) <= floor_y_tolerance)


def sample_reachable_points(
    sim,
    origin_world: np.ndarray,
    count: int,
    min_start_dist: float,
    max_tries: int,
    floor_y_tolerance: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: list[np.ndarray] = []
    for _ in range(max_tries):
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not np.all(np.isfinite(point)):
            continue
        if abs(float(point[1] - origin_world[1])) > floor_y_tolerance:
            continue
        if float(np.linalg.norm((point - origin_world)[[0, 2]])) < min_start_dist:
            continue
        if any(float(np.linalg.norm((point - prev)[[0, 2]])) < 2.0 for prev in seen):
            continue
        geo, path = shortest_path_points(sim, origin_world, point)
        if not path or not np.isfinite(geo):
            continue
        if not path_stays_on_floor(path, origin_world, floor_y_tolerance):
            continue
        samples.append({"point": point, "start_geo": geo})
        seen.append(point)
        if len(samples) >= count:
            break
    return samples


def choose_loop_anchors(samples: list[dict[str, Any]], origin_world: np.ndarray, anchor_count: int) -> list[np.ndarray]:
    if len(samples) <= anchor_count:
        return [row["point"] for row in samples]
    anchors: list[np.ndarray] = []
    first = max(samples, key=lambda row: float(row["start_geo"]))["point"]
    anchors.append(first)
    while len(anchors) < anchor_count:
        best = None
        best_score = -float("inf")
        for row in samples:
            point = row["point"]
            if any(np.allclose(point, anchor) for anchor in anchors):
                continue
            min_anchor_dist = min(float(np.linalg.norm((point - anchor)[[0, 2]])) for anchor in anchors)
            start_dist = float(np.linalg.norm((point - origin_world)[[0, 2]]))
            score = min_anchor_dist + 0.25 * start_dist
            if score > best_score:
                best = point
                best_score = score
        if best is None:
            break
        anchors.append(best)
    return anchors


def ordered_loop_anchors(anchors: list[np.ndarray], origin_world: np.ndarray) -> list[np.ndarray]:
    if len(anchors) < 3:
        return anchors
    center = np.mean(np.asarray(anchors, dtype=np.float32)[:, [0, 2]], axis=0)
    return sorted(
        anchors,
        key=lambda p: float(np.arctan2(float(p[2] - center[1]), float(p[0] - center[0]))),
    )


def build_route_from_anchors(
    sim,
    origin_world: np.ndarray,
    anchors: list[np.ndarray],
    floor_y_tolerance: float,
) -> tuple[list[np.ndarray], float]:
    route_world: list[np.ndarray] = []
    total_geo = 0.0
    legs = [origin_world] + ordered_loop_anchors(anchors, origin_world) + [origin_world]
    for leg_idx, (start, end) in enumerate(zip(legs[:-1], legs[1:])):
        geo, points = shortest_path_points(sim, start, end)
        if not points:
            raise RuntimeError(f"Failed to connect loop leg {leg_idx}")
        if not path_stays_on_floor(points, origin_world, floor_y_tolerance):
            raise RuntimeError(f"Loop leg {leg_idx} leaves start floor")
        total_geo += geo
        if route_world:
            route_world.extend(points[1:])
        else:
            route_world.extend(points)
    return route_world, total_geo


def build_loop_route(
    sim,
    origin_world: np.ndarray,
    anchor_count: int = 3,
    sample_count: int = 80,
    min_start_dist: float = 4.0,
    min_extent: float = 4.0,
    max_geodesic_length: float = 18.0,
    floor_y_tolerance: float = 0.35,
    max_tries: int = 400,
) -> dict[str, Any]:
    samples = sample_reachable_points(sim, origin_world, sample_count, min_start_dist, max_tries, floor_y_tolerance)
    if len(samples) < 2:
        raise RuntimeError("Could not sample enough reachable same-floor points for loop trajectory")

    best = None
    best_any = None
    last_error = ""
    for n_anchors in range(max(2, anchor_count), 1, -1):
        for min_dist in [min_start_dist, max(2.5, min_start_dist * 0.75), max(1.8, min_start_dist * 0.5)]:
            pool = [row for row in samples if float(row["start_geo"]) >= min_dist]
            if len(pool) < n_anchors:
                continue
            anchors = choose_loop_anchors(pool, origin_world, n_anchors)
            try:
                raw_route, total_geo = build_route_from_anchors(sim, origin_world, anchors, floor_y_tolerance)
            except RuntimeError as exc:
                last_error = str(exc)
                continue
            route_world = densify_points(raw_route, spacing=0.75)
            route_rel_candidate = [world_to_relative(p, origin_world).round(4).tolist() for p in route_world]
            arr_candidate = np.asarray(route_rel_candidate, dtype=np.float32)
            extent_candidate = float(max(np.ptp(arr_candidate[:, 0]), np.ptp(arr_candidate[:, 2]))) if len(arr_candidate) else 0.0
            y_range_candidate = float(np.ptp(arr_candidate[:, 1])) if len(arr_candidate) else 0.0
            if extent_candidate < min_extent:
                last_error = f"Loop trajectory extent too small: {extent_candidate:.2f}m"
                continue
            candidate = (anchors, route_world, total_geo, extent_candidate, y_range_candidate)
            if best_any is None or total_geo < best_any[2]:
                best_any = candidate
            if total_geo <= max_geodesic_length:
                best = candidate
                break
            last_error = f"loop longer than requested: {total_geo:.2f}m > {max_geodesic_length:.2f}m"
        if best is not None:
            break
    if best is None:
        if best_any is not None:
            best = best_any
        else:
            raise RuntimeError(last_error or f"Could not build same-floor loop trajectory within {max_geodesic_length:.2f}m")

    anchors, route_world, total_geo, extent, y_range = best
    over_limit = total_geo > max_geodesic_length
    route_rel = [world_to_relative(p, origin_world).round(4).tolist() for p in route_world]
    anchor_rel = [world_to_relative(p, origin_world).round(4).tolist() for p in anchors]
    return {
        "kind": "auto_loop",
        "closed_loop": True,
        "same_floor": True,
        "waypoints_relative": anchor_rel,
        "path_points_relative": route_rel,
        "geodesic_length": float(total_geo),
        "extent": extent,
        "y_range": y_range,
        "floor_y_tolerance": float(floor_y_tolerance),
        "max_geodesic_length": float(max_geodesic_length),
        "over_max_geodesic_length": bool(over_limit),
        "sample_count": len(samples),
    }


def trajectory_target(route: list[list[float]], idx: int) -> np.ndarray | None:
    if idx >= len(route):
        return None
    return np.asarray(route[idx], dtype=np.float32)


def advance_route_index(current_relative: np.ndarray, route: list[list[float]], idx: int, reach_radius: float) -> int:
    while idx < len(route):
        target = np.asarray(route[idx], dtype=np.float32)
        if float(np.linalg.norm((target - current_relative)[[0, 2]])) > reach_radius:
            break
        idx += 1
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene", default="4ok3usBNeis")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until the loop trajectory completes")
    parser.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    parser.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--goal-modality", choices=["auto", "object", "instruction", "description", "image"], default="auto")
    parser.add_argument("--output", default="data/logs/goat_topo/loop_trace/topo_trace_semantic.json")
    parser.add_argument("--frame-dir", default=None)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-image-model", default="RN50", help="CLIP model for image-goal rgb_embed (GOAT official: RN50)")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument(
        "--perception-backend",
        choices=["clip_groundingdino", "vlm"],
        default="clip_groundingdino",
    )
    parser.add_argument("--vlm-api-base", default="http://localhost:8000/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--vlm-timeout", type=float, default=60.0)
    parser.add_argument("--object-threshold", type=float, default=None)
    parser.add_argument("--heavy-enabled", action="store_true")
    parser.add_argument("--heavy-interval", type=int, default=None)
    parser.add_argument("--object-detection-threshold", type=float, default=None)
    parser.add_argument("--groundingdino-config", default=None)
    parser.add_argument("--groundingdino-checkpoint", default=None)
    parser.add_argument("--groundingdino-device", default=None)
    parser.add_argument("--room-threshold", type=float, default=None)
    parser.add_argument("--landmark-threshold", type=float, default=None)
    parser.add_argument(
        "--env-landmarks",
        default="default",
        help="Comma-separated scene landmark labels, 'default' for built-ins, or 'none' to disable.",
    )
    parser.add_argument("--use-placeholder-embed", action="store_true")
    parser.add_argument("--loop-anchors", type=int, default=3)
    parser.add_argument("--loop-samples", type=int, default=80)
    parser.add_argument("--loop-min-start-dist", type=float, default=4.0)
    parser.add_argument("--loop-min-extent", type=float, default=4.0)
    parser.add_argument("--loop-max-geodesic-length", type=float, default=40.0)
    parser.add_argument("--loop-floor-y-tolerance", type=float, default=0.35)
    parser.add_argument("--loop-reach-radius", type=float, default=0.45)
    parser.add_argument("--loop-return-radius", type=float, default=0.75)
    parser.add_argument("--instruction", default="Walk one loop around the room on the same floor and return to the start.")
    parser.add_argument(
        "--agent",
        choices=["final", "etpnav", "clean"],
        default="final",
        help="Agent variant: goat_agent_final, goat_agent_etpnav, or goat_agent_etpnav_clean.",
    )
    args = parser.parse_args()
    env_landmark_labels = parse_env_landmarks(args.env_landmarks)

    dataset_dir = ROOT / args.dataset_dir
    scene_path, episode = pick_episode(dataset_dir, args.split, args.scene, args.episode_index)
    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)
    ig = load_goal_graph(ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"])
    ig, goal_modality_debug = select_goal_modality(ig, args.goal_modality)

    config = ConfTopoConfig()
    config.perception.clip_model = args.clip_model
    config.perception.clip_image_model = args.clip_image_model
    config.perception.clip_device = args.clip_device
    config.perception.backend = args.perception_backend
    config.perception.vlm_api_base = args.vlm_api_base
    config.perception.vlm_model = args.vlm_model
    config.perception.vlm_timeout = args.vlm_timeout
    if args.object_threshold is not None:
        config.perception.object_threshold = args.object_threshold
    config.perception.heavy_enabled = bool(args.heavy_enabled or args.perception_backend == "vlm")
    if args.heavy_interval is not None:
        config.perception.heavy_interval = args.heavy_interval
    if args.object_detection_threshold is not None:
        config.perception.object_detection_threshold = args.object_detection_threshold
    if args.groundingdino_config is not None:
        config.perception.groundingdino_config = args.groundingdino_config
    if args.groundingdino_checkpoint is not None:
        config.perception.groundingdino_checkpoint = args.groundingdino_checkpoint
    if args.groundingdino_device is not None:
        config.perception.groundingdino_device = args.groundingdino_device
    if args.room_threshold is not None:
        config.perception.room_threshold = args.room_threshold
    if args.landmark_threshold is not None:
        config.perception.landmark_threshold = args.landmark_threshold

    if args.agent == "clean":
        etp_config = CleanETPGoatConfig()
        agent = ConfTopoGOATAgentCleanETPNav(config, etp_config)
        agent_variant = "ConfTopoGOATCleanETPNavAgent"
    elif args.agent == "etpnav":
        etp_config = ETPGoatConfig()
        agent = ConfTopoGOATAgentETPNav(config, etp_config)
        agent_variant = "ConfTopoGOATAgentETPNav"
    else:
        agent = ConfTopoGOATAgent(config)
        agent_variant = "ConfTopoGOATAgentFinal"
    agent.set_goal(ig)
    first_goal = ig.get_current_goal()
    if isinstance(first_goal, GoalNode):
        agent.set_new_goal(first_goal)

    encoder = None
    if not args.use_placeholder_embed:
        encoder = GoatModalityClipEncoder(
            config.perception.clip_model,
            config.perception.clip_image_model,
            config.perception.clip_device,
        )
        agent.perceiver.room_text_embeds = encoder.encode_text(agent.perceiver.room_labels)
        active_landmark_labels = configure_landmark_perception(agent, first_goal, encoder, env_landmark_labels)
    else:
        active_landmark_labels = []

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
    trajectory = build_loop_route(
        sim,
        origin,
        anchor_count=args.loop_anchors,
        sample_count=args.loop_samples,
        min_start_dist=args.loop_min_start_dist,
        min_extent=args.loop_min_extent,
        max_geodesic_length=args.loop_max_geodesic_length,
        floor_y_tolerance=args.loop_floor_y_tolerance,
    )
    route = trajectory["path_points_relative"]
    route_idx = 1 if len(route) > 1 else 0
    nav_executor = PathfinderExecutor(reach_radius=args.loop_reach_radius)
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
        "instruction": args.instruction,
        "agent_variant": agent_variant,
        "current_goal": current_goal_summary(ig),
        "environment_landmarks": env_landmark_labels,
        "active_landmark_labels": active_landmark_labels,
        "embedding_source": (
            f"vlm:{args.vlm_model}" if args.perception_backend == "vlm"
            else ("placeholder" if args.use_placeholder_embed else f"clip:{args.clip_model}")
        ),
        "perception_backend": args.perception_backend,
        "thresholds": {
            "object": config.perception.object_threshold,
            "room": config.perception.room_threshold,
            "landmark": config.perception.landmark_threshold,
        },
        "coordinate_frame": "episode_start_relative",
        "pose_sources": {
            "position": "episode_start_relative_from_gt_world",
            "heading": "habitat_gt_heading",
            "world_position": "habitat_gt_world_position",
            "world_heading": "habitat_gt_heading",
        },
        "trajectory": trajectory,
        "control_mode": "trajectory_low_level",
        "steps": [],
    }

    completed = False
    final_distance_to_start = None
    step = 0
    try:
        while args.max_steps <= 0 or step < args.max_steps:
            obs = sim.get_sensor_observations()
            state = sim_agent.get_state()
            rgb = obs.get("color_sensor")
            rgb_save = rgb[..., :3] if rgb is not None and rgb.shape[-1] == 4 else rgb
            frame_rel = None
            if frame_dir is not None and rgb_save is not None:
                frame_path = frame_dir / f"rgb_{step:04d}.png"
                imageio.imwrite(frame_path, rgb_save)
                frame_rel = str(frame_path.relative_to(ROOT))

            rgb_embed = encode_agent_rgb_embed(
                encoder, rgb, agent,
                use_placeholder=args.use_placeholder_embed,
                placeholder_fn=rgb_to_embedding,
            )
            conf_obs = {
                "rgb": rgb,
                "rgb_embed": rgb_embed,
                "position": np.array(state.position, dtype=np.float32),
                "heading": quat_to_heading([state.rotation.real, *list(state.rotation.imag)]),
            }
            agent.topo_map.step()
            agent.observe(conf_obs)
            agent.update_memory()

            world_position = np.asarray(state.position, dtype=np.float32)
            world_heading = float(conf_obs["heading"])
            rel_position = world_to_relative(world_position, origin)
            route_idx = advance_route_index(rel_position, route, route_idx, args.loop_reach_radius)
            target = trajectory_target(route, route_idx)
            collision_debug = collision_tracker.update(previous_low_action, rel_position)
            if target is None:
                low_action = "stop"
                target_distance = 0.0
                completed = True
            else:
                target_distance = float(np.linalg.norm((target - rel_position)[[0, 2]]))
                low_action = nav_executor.low_level_action_to(sim, target, origin)
                if collision_debug.get("collision_like"):
                    low_action = "turn_left"

            final_distance_to_start = float(np.linalg.norm(rel_position[[0, 2]]))
            perception = snapshot_perception(agent)
            trace["steps"].append({
                "step": step,
                "rgb_frame": frame_rel,
                "agent_action": "trajectory_follow",
                "low_action": low_action,
                "target_node_id": None if target is None else f"traj_{route_idx:03d}",
                "target_position": None if target is None else target.round(4).tolist(),
                "position": rel_position.round(4).tolist(),
                "heading": world_heading,
                "world_position": world_position.round(4).tolist(),
                "world_heading": world_heading,
                "rgb_embedding": np.asarray(rgb_embed, dtype=np.float32).round(6).tolist(),
                "candidate_ids": [],
                "candidate_scores": [],
                "trajectory_progress": {
                    "route_index": int(route_idx),
                    "route_points": len(route),
                    "target_distance": float(target_distance),
                    "completed": bool(completed),
                    "distance_to_start": final_distance_to_start,
                },
                "navigation_debug": {
                    "controller_mode": "loop_trajectory_follow",
                    "collision_like": bool(collision_debug.get("collision_like")),
                    "stuck_steps": int(collision_debug.get("stuck_steps", 0)),
                    "movement": collision_debug.get("movement"),
                },
                "collision_like": bool(collision_debug.get("collision_like")),
                "stuck_steps": int(collision_debug.get("stuck_steps", 0)),
                "selection_debug": {},
                "perception": perception,
                "memory": agent.memory_stats,
                "topo": snapshot_topo(agent),
            })
            if low_action == "stop":
                break
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
        "heavy_perception_calls": agent.memory_stats.get("heavy_perception_calls", 0),
        "object_merge_count": agent.memory_stats.get("object_merge_count", 0),
        "mean_object_confidence": agent.memory_stats.get("mean_object_confidence", 0.0),
        "memory_reuse_count": memory_reuse_count(trace["steps"]),
        "target_switch_count": count_target_switches(trace["steps"]),
        "path_length_relative": path_length(trace["steps"]),
        "trajectory_completed": bool(completed),
        "trajectory_final_distance_to_start": final_distance_to_start,
        "trajectory_returned_to_start": bool(final_distance_to_start is not None and final_distance_to_start <= args.loop_return_radius),
        "trajectory_return_radius": float(args.loop_return_radius),
        "trajectory_geodesic_length": trajectory.get("geodesic_length"),
        "trajectory_extent": trajectory.get("extent"),
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
        "trajectory": trajectory,
        "final_memory": trace["final_memory"],
        "final_summary": trace["final_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
