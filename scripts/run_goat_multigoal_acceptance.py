from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from run_goat_minimal import ROOT, find_scene_file, make_sim, normalize_quat
from run_goat_topo_trace import load_goal_graph, pick_episode, quat_to_heading, rgb_to_embedding, snapshot_topo
from conftopo.agents.goat_agent import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import GoalNode
from conftopo.navigation import CollisionLikeTracker, PathfinderExecutor
from conftopo.perception import ClipRuntimeEncoder


def set_start_state(sim_agent, episode):
    import habitat_sim
    state = habitat_sim.AgentState()
    state.position = np.array(episode["start_position"], dtype=np.float32)
    q = normalize_quat(episode["start_rotation"])
    if abs(q[0]) < 1e-6 and abs(q[2]) < 1e-6 and (abs(q[1]) > 1e-6 or abs(q[3]) > 1e-6):
        state.rotation = np.quaternion(q[3], q[0], q[1], q[2])
    else:
        state.rotation = np.quaternion(q[0], q[1], q[2], q[3])
    sim_agent.set_state(state)
    return state


def is_semantic_node_id(node_id: str | None) -> bool:
    return bool(node_id) and node_id.startswith(("obj_", "roo_", "lan_"))


def known_node_ids(agent: ConfTopoGOATAgent) -> set[str]:
    return set(agent.topo_map._nodes.keys())


def semantic_node_ids(agent: ConfTopoGOATAgent) -> set[str]:
    return {nid for nid in known_node_ids(agent) if is_semantic_node_id(nid)}


def run_step(
    sim,
    sim_agent,
    agent,
    encoder,
    origin,
    executor,
    tracker,
    previous_action,
    use_placeholder,
    record_topo: bool,
    frame_dir: Path | None = None,
    step_index: int = 0,
):
    obs = sim.get_sensor_observations()
    state = sim_agent.get_state()
    rgb = obs.get("color_sensor")
    rgb_save = rgb[..., :3] if rgb is not None and rgb.shape[-1] == 4 else rgb
    frame_rel = None
    if frame_dir is not None and rgb_save is not None:
        frame_path = frame_dir / f"rgb_{step_index:04d}.png"
        imageio.imwrite(frame_path, rgb_save)
        frame_rel = str(frame_path.relative_to(ROOT))
    rgb_embed = rgb_to_embedding(rgb) if use_placeholder else encoder.encode_image(rgb)
    world_position = np.asarray(state.position, dtype=np.float32)
    world_heading = quat_to_heading([state.rotation.real, *list(state.rotation.imag)])
    out = agent.step({
        "rgb": rgb,
        "rgb_embed": rgb_embed,
        "position": world_position,
        "heading": world_heading,
    })
    rel_position = world_position - origin
    collision = tracker.update(previous_action, rel_position)
    target = out.get("target_position")
    target_id = out.get("target_node_id")
    if collision.get("collision_like") and target_id:
        agent.on_navigation_event(target_id, "collision_like")
        action = "target_reached"
        nav = {"reachable": False, "reason": "collision_like"}
    elif target is None:
        action = "stop"
        nav = {"reachable": False, "reason": "no_target"}
    else:
        candidate_ids = out.get("candidate_ids") or []
        scores = np.asarray(out.get("scores", []), dtype=np.float32).tolist()
        selected = executor.select_reachable_candidate(
            sim,
            origin,
            agent.topo_map,
            candidate_ids,
            scores,
            top_k=5,
        ).get("selected")
        if selected:
            node = agent.topo_map.get_node(selected["node_id"])
            if node is not None:
                target_id = node.node_id
                target = node.position
        action, nav = executor.step(sim, np.asarray(target), origin, target_node_id=target_id)
        if action == "target_reached":
            agent.on_navigation_event(target_id, "target_reached")
        elif action == "unreachable":
            agent.on_navigation_event(target_id, nav.get("reason", "unreachable"))
            action = "turn_left"
    if action not in ("stop", "target_reached", "unreachable"):
        sim.step(action)
    rec = {
        "step": int(step_index),
        "rgb_frame": frame_rel,
        "agent_action": "navigate",
        "low_action": action,
        "target_node_id": target_id,
        "target_position": np.asarray(target).round(4).tolist() if target is not None else None,
        "position": rel_position.round(4).tolist(),
        "heading": float(world_heading),
        "world_position": world_position.round(4).tolist(),
        "world_heading": float(world_heading),
        "rgb_embedding": np.asarray(rgb_embed, dtype=np.float32).round(6).tolist(),
        "navigation_debug": nav,
        "collision_like": bool(collision.get("collision_like")),
        "memory": agent.memory_stats,
    }
    if record_topo:
        rec["topo"] = snapshot_topo(agent)
    return rec, action


def long_stuck_targets(steps: list[dict[str, Any]], min_steps: int = 20) -> list[dict[str, Any]]:
    segments: list[tuple[str | None, int, int, list[list[float]]]] = []
    current = None
    start_idx = 0
    positions: list[list[float]] = []
    for idx, st in enumerate(steps):
        target = st.get("target_node_id")
        pos = st.get("position") or [0.0, 0.0, 0.0]
        if current is None:
            current = target
            start_idx = idx
            positions = [pos]
            continue
        if target != current:
            segments.append((current, start_idx, idx - 1, positions))
            current = target
            start_idx = idx
            positions = [pos]
        else:
            positions.append(pos)
    if current is not None:
        segments.append((current, start_idx, len(steps) - 1, positions))

    stuck = []
    for target, start, end, seg_positions in segments:
        span = end - start + 1
        if span < min_steps:
            continue
        arr = np.asarray(seg_positions, dtype=np.float32)
        displacement = float(np.linalg.norm(arr[-1, [0, 2]] - arr[0, [0, 2]])) if len(arr) else 0.0
        if displacement < 0.15:
            stuck.append({
                "target_node_id": target,
                "start_step": start,
                "end_step": end,
                "steps": span,
                "displacement": displacement,
            })
    return stuck


def planar_target_distance(position: list[float] | None, target: list[float] | None) -> float | None:
    if position is None or target is None:
        return None
    pos = np.asarray(position, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    return float(np.linalg.norm((pos - tgt)[[0, 2]]))


def build_multigoal_acceptance_report(
    trace: dict[str, Any],
    tasks: list[dict[str, Any]],
    goals: list[GoalNode],
) -> dict[str, Any]:
    steps = trace.get("steps", [])
    collision_count = sum(1 for st in steps if st.get("collision_like"))
    stuck = long_stuck_targets(steps, min_steps=20)

    memory_preservation_passed = all(t.get("memory_preserved") for t in tasks)

    final_mem = trace.get("final_memory", {})
    semantic_nodes_built = {
        "objects": int(final_mem.get("objects", 0)),
        "rooms": int(final_mem.get("rooms", 0)),
        "landmarks": int(final_mem.get("landmarks", 0)),
    }
    semantic_build_passed = any(
        int(t.get("objects_after", 0)) >= 1 or int(t.get("landmarks_after", 0)) >= 1
        for t in tasks
    )

    later_tasks = [t for t in tasks if int(t.get("task_index", 0)) >= 1]
    memory_reuse_passed = any(int(t.get("memory_reuse_hits", 0)) > 0 for t in later_tasks)
    semantic_reuse_passed = any(int(t.get("semantic_reuse_hits", 0)) > 0 for t in later_tasks)

    repeated_goal_checks: list[dict[str, Any]] = []
    repeated_goal_passed = True
    for later_idx, earlier_idx in ((3, 0), (2, 1)):
        if len(goals) <= later_idx or len(tasks) <= later_idx:
            continue
        if goals[earlier_idx].target_object != goals[later_idx].target_object:
            continue
        task_row = tasks[later_idx]
        hit = bool(task_row.get("repeated_goal_reuse_hits", 0) > 0 or task_row.get("repeated_goal_semantic_reuse"))
        repeated_goal_checks.append({
            "later_task_index": later_idx,
            "earlier_task_index": earlier_idx,
            "target_object": goals[later_idx].target_object,
            "passed": hit,
            "repeated_goal_reuse_hits": int(task_row.get("repeated_goal_reuse_hits", 0)),
        })
        if not hit:
            repeated_goal_passed = False

    if repeated_goal_checks:
        memory_reuse_passed = memory_reuse_passed or repeated_goal_passed

    navigation_stable = collision_count == 0 and len(stuck) == 0
    overall_passed = (
        memory_preservation_passed
        and semantic_build_passed
        and memory_reuse_passed
        and navigation_stable
    )

    return {
        "memory_preservation_passed": memory_preservation_passed,
        "semantic_nodes_built": semantic_nodes_built,
        "semantic_build_passed": semantic_build_passed,
        "memory_reuse_passed": memory_reuse_passed,
        "semantic_reuse_passed": semantic_reuse_passed,
        "repeated_goal_checks": repeated_goal_checks,
        "navigation_stable": navigation_stable,
        "collision_like_count": collision_count,
        "long_stuck_targets": stuck,
        "overall_passed": overall_passed,
        "thresholds": trace.get("thresholds", {}),
        "task_summaries": tasks,
        "final_summary": trace.get("final_summary", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--scene", default="GLAQ4DNUx5U")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--steps-per-goal", type=int, default=80)
    ap.add_argument("--max-goals", type=int, default=4)
    ap.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    ap.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    ap.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    ap.add_argument("--output", default="data/logs/goat_topo/multigoal_acceptance/topo_trace_multigoal.json")
    ap.add_argument("--report", default="data/logs/goat_topo/multigoal_acceptance/multigoal_acceptance_report.json")
    ap.add_argument("--frame-dir", default=None)
    ap.add_argument("--clip-model", default="ViT-B/32")
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument("--object-threshold", type=float, default=None)
    ap.add_argument("--room-threshold", type=float, default=None)
    ap.add_argument("--landmark-threshold", type=float, default=None)
    ap.add_argument(
        "--phase2-summary",
        default="data/logs/goat_topo/phase2_pathfinder_acceptance/summary.json",
        help="Load object/landmark thresholds from Phase2 acceptance summary",
    )
    ap.add_argument("--use-placeholder-embed", action="store_true")
    ap.add_argument("--record-topo", action="store_true", help="Store topo snapshot each step (for viz)")
    ap.add_argument("--viz", action="store_true", help="Run visualize_goat_topo_trace after trace is saved")
    ap.add_argument("--viz-stride", type=int, default=4)
    ap.add_argument("--viz-fps", type=int, default=6)
    ap.add_argument("--block-stuck-target-after", type=int, default=24)
    ap.add_argument("--block-stuck-target-min-progress", type=float, default=0.25)
    args = ap.parse_args()

    scene_path, episode = pick_episode(ROOT / args.dataset_dir, args.split, args.scene, args.episode_index)
    ig = load_goal_graph(ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"])
    goals = [g for g in ig.goal_nodes if isinstance(g, GoalNode)][: args.max_goals]
    if len(goals) < 2:
        raise RuntimeError("Need at least two GOAT goals for memory-reuse acceptance")

    thresholds = {}
    summary_path = ROOT / args.phase2_summary
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        thresholds = summary.get("scan_summary", {}).get("thresholds", {})
    if args.object_threshold is not None:
        thresholds["object"] = args.object_threshold
    if args.landmark_threshold is not None:
        thresholds["landmark"] = args.landmark_threshold
    if args.room_threshold is not None:
        thresholds["room"] = args.room_threshold

    config = ConfTopoConfig()
    config.perception.clip_model = args.clip_model
    config.perception.clip_device = args.clip_device
    if "object" in thresholds:
        config.perception.object_threshold = float(thresholds["object"])
    if "room" in thresholds:
        config.perception.room_threshold = float(thresholds["room"])
    if "landmark" in thresholds:
        config.perception.landmark_threshold = float(thresholds["landmark"])
    agent = ConfTopoGOATAgent(config)
    agent.set_goal(ig)
    encoder = None if args.use_placeholder_embed else ClipRuntimeEncoder(args.clip_model, args.clip_device)
    if encoder is not None:
        agent.perceiver.room_text_embeds = encoder.encode_text(agent.perceiver.room_labels)

    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)
    frame_dir = ROOT / args.frame_dir if args.frame_dir else None
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
    sim = make_sim(scene_file)
    sim_agent = sim.initialize_agent(0)
    state = set_start_state(sim_agent, episode)
    origin = np.array(state.position, dtype=np.float32)
    executor = PathfinderExecutor()
    tracker = CollisionLikeTracker()
    previous = None
    steps: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    task0_semantic_nodes: set[str] = set()
    try:
        for idx, goal in enumerate(goals):
            before = agent.memory_stats.copy()
            known_before = known_node_ids(agent)
            semantic_before = semantic_node_ids(agent)
            agent.set_new_goal(goal)
            start = len(steps)
            monitor_target_id = None
            monitor_start_step = len(steps)
            monitor_best_distance = None
            memory_reuse_hits = 0
            semantic_reuse_hits = 0
            repeated_goal_reuse_hits = 0
            for _ in range(args.steps_per_goal):
                rec, previous = run_step(
                    sim,
                    sim_agent,
                    agent,
                    encoder,
                    origin,
                    executor,
                    tracker,
                    previous,
                    args.use_placeholder_embed,
                    args.record_topo or args.viz,
                    frame_dir=frame_dir,
                    step_index=len(steps),
                )
                rec["task_index"] = idx
                rec["goal"] = {
                    "target_object": goal.target_object,
                    "room_prior": goal.room_prior,
                    "landmarks": goal.landmarks,
                }
                target_id = rec.get("target_node_id")
                if target_id and target_id in known_before:
                    memory_reuse_hits += 1
                    rec["memory_reuse"] = True
                if is_semantic_node_id(target_id) and target_id in known_before:
                    semantic_reuse_hits += 1
                    rec["semantic_reuse"] = True
                if idx == 3 and task0_semantic_nodes and target_id in task0_semantic_nodes:
                    repeated_goal_reuse_hits += 1
                    rec["repeated_goal_reuse"] = True
                target_distance = planar_target_distance(rec.get("position"), rec.get("target_position"))
                rec["target_distance"] = target_distance
                if target_id is None or rec["low_action"] in ("stop", "target_reached"):
                    monitor_target_id = None
                    monitor_best_distance = None
                    monitor_start_step = len(steps)
                elif target_id != monitor_target_id:
                    monitor_target_id = target_id
                    monitor_start_step = len(steps)
                    monitor_best_distance = target_distance
                elif target_distance is not None:
                    best = target_distance if monitor_best_distance is None else monitor_best_distance
                    if target_distance < best - args.block_stuck_target_min_progress:
                        monitor_start_step = len(steps)
                        monitor_best_distance = target_distance
                    elif len(steps) - monitor_start_step + 1 >= args.block_stuck_target_after:
                        event = agent.on_navigation_event(target_id, "no_progress_multigoal")
                        rec["blocked_stuck_target"] = {
                            "target_node_id": target_id,
                            "steps": int(len(steps) - monitor_start_step + 1),
                            "best_distance": float(best),
                            "current_distance": float(target_distance),
                            "min_progress": float(args.block_stuck_target_min_progress),
                            "event": event,
                        }
                        monitor_target_id = None
                        monitor_best_distance = None
                        monitor_start_step = len(steps)
                steps.append(rec)
                if rec["low_action"] == "stop":
                    break
            after = agent.memory_stats.copy()
            if idx == 0:
                task0_semantic_nodes = semantic_node_ids(agent)
            tasks.append({
                "task_index": idx,
                "target_object": goal.target_object,
                "node_count_before": before["total_nodes"],
                "node_count_after": after["total_nodes"],
                "objects_after": after["objects"],
                "rooms_after": after["rooms"],
                "landmarks_after": after["landmarks"],
                "memory_preserved": after["total_nodes"] >= before["total_nodes"],
                "known_nodes_before": len(known_before),
                "memory_reuse_hits": memory_reuse_hits,
                "semantic_reuse_hits": semantic_reuse_hits,
                "repeated_goal_reuse_hits": repeated_goal_reuse_hits,
                "repeated_goal_semantic_reuse": repeated_goal_reuse_hits > 0,
                "steps": len(steps) - start,
            })
    finally:
        sim.close()

    trace = {
        "coordinate_frame": "episode_start_relative",
        "scene": args.scene,
        "episode_index": args.episode_index,
        "scene_file": str(scene_file),
        "episode_file": str(scene_path),
        "episode_id": episode["episode_id"],
        "origin_world": origin.round(4).tolist(),
        "start_rotation": episode["start_rotation"],
        "pose_sources": {
            "position": "episode_start_relative_from_gt_world",
            "heading": "habitat_gt_heading",
            "world_position": "habitat_gt_world_position",
            "world_heading": "habitat_gt_heading",
        },
        "rgb_frame_dir": None if frame_dir is None else str(frame_dir.relative_to(ROOT)),
        "thresholds": {
            "object": config.perception.object_threshold,
            "room": config.perception.room_threshold,
            "landmark": config.perception.landmark_threshold,
        },
        "task_summaries": tasks,
        "steps": steps,
        "final_memory": agent.memory_stats,
        "final_summary": {
            "episode_length": len(steps),
            "target_switch_count": len(goals) - 1,
            "memory_reuse_count": sum(t["memory_reuse_hits"] for t in tasks),
            "semantic_reuse_count": sum(t["semantic_reuse_hits"] for t in tasks),
            "semantic_node_count": agent.memory_stats["objects"] + agent.memory_stats["rooms"] + agent.memory_stats["landmarks"],
            "collision_like_count": sum(1 for st in steps if st.get("collision_like")),
        },
    }
    report = build_multigoal_acceptance_report(trace, tasks, goals)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, ensure_ascii=False))

    report_path = ROOT / args.report
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if args.viz:
        viz_dir = out.parent / "viz"
        subprocess.run(
            [
                sys.executable,
                "scripts/visualize_goat_topo_trace.py",
                "--trace",
                str(out.relative_to(ROOT)),
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

    print(json.dumps({
        "ok": True,
        "output": str(out),
        "report": str(report_path),
        "overall_passed": report["overall_passed"],
        "task_summaries": tasks,
        "acceptance": {
            "memory_preservation_passed": report["memory_preservation_passed"],
            "semantic_build_passed": report["semantic_build_passed"],
            "memory_reuse_passed": report["memory_reuse_passed"],
            "navigation_stable": report["navigation_stable"],
        },
        "final_summary": trace["final_summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
