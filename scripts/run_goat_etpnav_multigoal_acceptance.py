"""Multigoal acceptance harness for ConfTopo-GOAT-ETPNav.

This is the ETPNav-style variant of ``run_goat_multigoal_acceptance.py``.
It uses ``ConfTopoGOATAgentETPNav``: RGB-only ghost candidates, graph-aware
route costs, candidate promote/consume, and waypoint stop memory.

Success criteria follow the GOAT-Bench benchmark definition (README):
  - Each subtask has a budget of 500 agent actions.
  - Success requires calling STOP while within 1m Euclidean distance of the
    current goal object instance (judged at the STOP position, not merely passing nearby).
  - ``goal_min_distance`` is recorded for debugging only.

Note: the bundled ``goat-bench`` Habitat training config uses
``success_distance: 0.25`` with geodesic distance to view points; that is the
RL training/eval implementation detail, not the paper benchmark definition above.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from run_goat_minimal import ROOT, find_scene_file, load_json_gz, make_sim, normalize_quat
from run_goat_topo_trace import load_goal_graph, pick_episode, quat_to_heading, rgb_to_embedding, snapshot_topo
from conftopo.agents.goat_agent_etpnav import ConfTopoGOATAgentETPNav, ETPGoatConfig
from conftopo.agents.goat_agent_etpnav_clean import ConfTopoGOATAgentCleanETPNav, CleanETPGoatConfig
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.navigation import CollisionLikeTracker, PathfinderExecutor
from conftopo.perception import GoatModalityClipEncoder, encode_agent_rgb_embed, encode_agent_image_goal_embed


DEFAULT_STEPS_PER_GOAL = 500  # GOAT-Bench README: 500 actions per subtask
DEFAULT_SUCCESS_DISTANCE = 1.0  # GOAT-Bench README: 1m Euclidean at STOP


def _scene_basename(episode: dict[str, Any]) -> str:
    scene_id = str(episode.get("scene_id", ""))
    return scene_id.split("/")[-1].replace(".basis.glb", "").replace(".glb", "")


def _resolve_goat_subtask_goal_entries(
    dataset_goals: dict[str, Any],
    episode: dict[str, Any],
    task_index: int,
) -> list[dict[str, Any]]:
    """Resolve GOAT goal dict entries for one subtask (mirrors GoatDatasetV1.from_json)."""
    episode_tasks = episode.get("tasks") or []
    if task_index >= len(episode_tasks):
        return []
    task = episode_tasks[task_index]
    if len(task) < 2:
        return []
    goal_category = task[0]
    goal_type = task[1]
    goal_inst_id = task[2] if len(task) > 2 else None

    same_cat_goal_lists = [
        entries
        for entries in dataset_goals.values()
        if isinstance(entries, list)
        and entries
        and isinstance(entries[0], dict)
        and entries[0].get("object_category") == goal_category
    ]
    if not same_cat_goal_lists:
        return []

    goal_entries = list(same_cat_goal_lists[0])
    children_categories = same_cat_goal_lists[0][0].get("children_object_categories") or []
    scene_name = _scene_basename(episode)
    for child_category in children_categories:
        child_key = f"{scene_name}_{child_category}"
        child_entries = dataset_goals.get(child_key)
        if isinstance(child_entries, list):
            goal_entries.extend(child_entries)

    if goal_type == "object":
        return [g for g in goal_entries if isinstance(g, dict)]
    if goal_inst_id is None:
        return []
    return [
        g for g in goal_entries
        if isinstance(g, dict) and str(g.get("object_id")) == str(goal_inst_id)
    ]


def _instance_positions_from_goal_entries(goal_entries: list[dict[str, Any]]) -> list[np.ndarray]:
    """GT object-center positions for the current subtask's target instance(s)."""
    positions: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for goal in goal_entries:
        position = goal.get("position")
        if position is None:
            continue
        pos = np.asarray(position, dtype=np.float32)
        key = tuple(np.round(pos, 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        positions.append(pos)
    return positions


def _euclidean_distance_to_instances(
    agent_world: np.ndarray,
    instance_positions: list[np.ndarray],
) -> float:
    """Planar-agnostic 3D distance to the nearest target instance center."""
    if not instance_positions:
        return float("inf")
    agent = np.asarray(agent_world, dtype=np.float32)
    return min(float(np.linalg.norm(agent - pos)) for pos in instance_positions)


def _view_points_from_goal_entries(goal_entries: list[dict[str, Any]]) -> list[np.ndarray]:
    """View-point positions used only for SPL geodesic optimal-path estimation."""
    viewpoints: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for goal in goal_entries:
        for view_point in goal.get("view_points") or []:
            if not isinstance(view_point, dict):
                continue
            agent_state = view_point.get("agent_state") or {}
            position = agent_state.get("position")
            if position is None:
                continue
            pos = np.asarray(position, dtype=np.float32)
            key = tuple(np.round(pos, 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            viewpoints.append(pos)
    return viewpoints


def _goat_geodesic_distance_to_viewpoints(
    sim,
    agent_world: np.ndarray,
    viewpoints: list[np.ndarray],
) -> float:
    """Official GOAT distance: geodesic from agent to nearest view point."""
    if not viewpoints:
        return float("inf")
    try:
        from habitat_sim.nav import MultiGoalShortestPath

        path = MultiGoalShortestPath()
        path.requested_start = np.asarray(agent_world, dtype=np.float32)
        path.requested_ends = np.asarray(viewpoints, dtype=np.float32)
        if not sim.pathfinder.find_path(path):
            return float("inf")
        geo = float(path.geodesic_distance)
        return geo if np.isfinite(geo) else float("inf")
    except Exception:
        return float("inf")


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
    return bool(node_id) and (
        node_id.startswith(("obj_", "roo_", "lan_"))
        or node_id.startswith("region::")
    )


def known_node_ids(agent: ConfTopoGOATAgentETPNav) -> set[str]:
    return set(agent.topo_map._nodes.keys())


def semantic_node_ids(agent: ConfTopoGOATAgentETPNav) -> set[str]:
    return {nid for nid in known_node_ids(agent) if is_semantic_node_id(nid)}


def durable_memory_node_ids(agent: ConfTopoGOATAgentETPNav) -> set[str]:
    """Semantic nodes that are expected to survive a goal switch."""
    durable: set[str] = set()
    for node_id, node in agent.topo_map._nodes.items():
        if node.node_type in (NodeType.ROOM, NodeType.LANDMARK):
            durable.add(node_id)
            continue
        if (
            node.node_type == NodeType.OBJECT
            and node.attributes.get("semantic_role") in ("object_anchor", "context_object")
        ):
            durable.add(node_id)
    return durable


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
    rgb_embed = encode_agent_rgb_embed(
        encoder, rgb, agent,
        use_placeholder=use_placeholder,
        placeholder_fn=rgb_to_embedding,
    )
    image_goal_embed = None
    if not use_placeholder and rgb is not None:
        image_goal_embed = encode_agent_image_goal_embed(encoder, rgb, agent)
    world_position = np.asarray(state.position, dtype=np.float32)
    world_heading = quat_to_heading([state.rotation.real, *list(state.rotation.imag)])
    out = agent.step({
        "rgb": rgb,
        "rgb_embed": rgb_embed,
        "image_goal_embed": image_goal_embed,
        "position": world_position,
        "heading": world_heading,
    })
    rel_position = world_position - origin
    collision = tracker.update(previous_action, rel_position)
    target = out.get("target_position")
    target_id = out.get("target_node_id")
    direct_action = out.get("action")
    plan_action = out.get("plan_action")
    direct_low_actions = {"turn_left", "turn_right", "move_forward"}
    if direct_action in direct_low_actions:
        action = direct_action
        nav = {
            "reachable": True,
            "reason": out.get("mode", "agent_direct_action"),
            "controller_mode": "agent_direct_action",
        }
    elif collision.get("collision_like") and target_id:
        agent.on_navigation_event(target_id, "collision_blocked")
        action = "turn_left"
        nav = {"reachable": False, "reason": "collision_blocked"}
    elif direct_action == "stop" or plan_action == "stop":
        action = "stop"
        nav = {"reachable": False, "reason": out.get("stop_debug", {}).get("reason", "should_stop")}
    elif target is None:
        # No semantic target should still explore.  Pure turning produced long
        # no-target stalls, so alternate short forward probes with turns.
        action = "move_forward" if (step_index % 3 == 0) else "turn_right"
        nav = {"reachable": True, "reason": plan_action or "no_target", "controller_mode": "no_target_explore"}
    else:
        candidate_ids = out.get("candidate_ids") or []
        scores = np.asarray(out.get("scores", []), dtype=np.float32).tolist()
        sel_result = executor.select_reachable_candidate(
            sim,
            origin,
            agent.topo_map,
            candidate_ids,
            scores,
            top_k=5,
        )
        reachable_cands = sel_result.get("reachable_candidates") or []
        if reachable_cands:
            for r in reachable_cands:
                r["nav_score"] = r["score"] - 0.02 * r.get("geodesic_distance", 0)
            reachable_cands.sort(key=lambda x: x["nav_score"], reverse=True)
            selected = reachable_cands[0]
        else:
            selected = None
        if selected:
            node = agent.topo_map.get_node(selected["node_id"])
            if node is not None:
                target_id = node.node_id
                selected_target = selected.get("target_position")
                if selected_target is not None:
                    target = np.asarray(selected_target, dtype=np.float32)
                else:
                    target_pos, _ = agent._target_output_for_node(node)
                    target = target_pos
        if selected is None and candidate_ids:
            for cid in candidate_ids[:5]:
                agent.on_navigation_event(cid, "unreachable")
            action = "move_forward" if (step_index % 3 == 0) else "turn_right"
            nav = {"reachable": False, "reason": "all_candidates_unreachable",
                   "controller_mode": "no_reachable_candidate"}
        else:
            action, nav = executor.step(sim, np.asarray(target), origin, target_node_id=target_id)
            if action == "target_reached":
                nav_event = agent.on_navigation_event(target_id, "target_reached")
                evt_action = nav_event.get("action")
                if evt_action == "should_stop":
                    action = "stop"
                    nav = {"reachable": True, "reason": nav_event.get("reason", "object_should_stop")}
                elif evt_action == "anchor_reached_awaiting_confirm":
                    action = "turn_left"
                    nav = {"reachable": False, "reason": evt_action}
                elif evt_action in (
                    "consumed_frontier",
                    "consumed_candidate",
                    "candidate_promoted",
                    "blocked_target",
                    "blacklisted_stop_waypoint",
                ):
                    action = "target_reached"
                    nav = {"reachable": True, "reason": nav_event.get("reason", "target_reached")}
                else:
                    nav = {"reachable": True, "reason": nav_event.get("reason", "target_reached")}
            elif action == "unreachable":
                agent.on_navigation_event(target_id, nav.get("reason", "unreachable"))
                action = "turn_left"
    if action not in ("stop", "target_reached", "unreachable"):
        sim.step(action)
    if hasattr(agent, "record_executed_action"):
        agent.record_executed_action(action)
    rec = {
        "step": int(step_index),
        "rgb_frame": frame_rel,
        "agent_action": out.get("action", "navigate"),
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
        "agent_output_action": out.get("action"),
        "plan_action": out.get("plan_action"),
        "plan_mode": out.get("mode"),
        "stop_debug": out.get("stop_debug"),
        "requires_regrounding": bool(out.get("requires_regrounding", False)),
        "target_anchor_type": out.get("target_anchor_type"),
        "semantic_target_node_id": out.get("semantic_target_node_id"),
        "etp_route_debug": out.get("etp_route_debug", {}),
        "anchor_waypoint_id": out.get("anchor_waypoint_id"),
        "anchor_room_id": out.get("anchor_room_id"),
        "reground_state": out.get("reground_state"),
        "reground_target_node_id": out.get("reground_target_node_id"),
        "reground_anchor_node_id": out.get("reground_anchor_node_id"),
        "target_object_detected_this_scan": bool(out.get("target_object_detected_this_scan", False)),
        "last_heavy": agent.memory_stats.get("last_heavy", {}),
        "task_telemetry": out.get("task_telemetry"),
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
    checked_later = set()
    for later_idx in range(len(goals) - 1, 0, -1):
        if later_idx >= len(tasks):
            continue
        for earlier_idx in range(later_idx):
            if earlier_idx >= len(goals):
                continue
            if goals[earlier_idx].target_object != goals[later_idx].target_object:
                continue
            if later_idx in checked_later:
                continue
            checked_later.add(later_idx)
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
    ap.add_argument(
        "--agent-variant",
        choices=["etpnav", "clean"],
        default="clean",
        help="ETPNav agent implementation: legacy patched etpnav or self-contained clean ETP-lite.",
    )
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--scene", default="GLAQ4DNUx5U")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--steps-per-goal", type=int, default=DEFAULT_STEPS_PER_GOAL)
    ap.add_argument("--max-goals", type=int, default=4)
    ap.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    ap.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    ap.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    ap.add_argument("--output", default="data/logs/goat_topo/etpnav_multigoal_acceptance/topo_trace_multigoal_etpnav.json")
    ap.add_argument("--report", default="data/logs/goat_topo/etpnav_multigoal_acceptance/multigoal_acceptance_report_etpnav.json")
    ap.add_argument("--frame-dir", default=None)
    ap.add_argument("--clip-model", default="ViT-B/32")
    ap.add_argument("--clip-image-model", default="RN50", help="CLIP model for image-goal rgb_embed (GOAT official: RN50)")
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument(
        "--perception-backend",
        choices=["clip_groundingdino", "vlm"],
        default="clip_groundingdino",
        help="Perception backend used by ConfTopoGOATAgentETPNav.",
    )
    ap.add_argument("--vlm-api-base", default="http://localhost:8000/v1")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--vlm-timeout", type=float, default=5.0)
    ap.add_argument("--object-threshold", type=float, default=None)
    ap.add_argument("--heavy-enabled", action="store_true")
    ap.add_argument("--heavy-interval", type=int, default=None)
    ap.add_argument(
        "--heavy-reground-cooldown",
        type=int,
        default=None,
        help="Minimum simulator steps between VLM/heavy calls during local regrounding.",
    )
    ap.add_argument(
        "--heavy-near-goal-cooldown",
        type=int,
        default=None,
        help="Minimum simulator steps between VLM/heavy calls when a goal object is nearby.",
    )
    ap.add_argument("--object-detection-threshold", type=float, default=None)
    ap.add_argument("--groundingdino-config", default=None)
    ap.add_argument("--groundingdino-checkpoint", default=None)
    ap.add_argument("--groundingdino-device", default=None)
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
    ap.add_argument("--ghost-merge-radius", type=float, default=None)
    ap.add_argument("--ghost-min-distance", type=float, default=None)
    ap.add_argument("--ghost-confidence", type=float, default=None)
    ap.add_argument("--ghost-graph-distance-cap", type=float, default=None)
    ap.add_argument("--disable-route-next-hop", action="store_true")
    ap.add_argument("--disable-stop-memory", action="store_true")
    ap.add_argument("--stop-memory-proposal-threshold", type=float, default=None)
    ap.add_argument("--stop-memory-min-write-score", type=float, default=None)
    ap.add_argument("--stop-memory-score-weight", type=float, default=None)
    ap.add_argument(
        "--success-distance",
        type=float,
        default=DEFAULT_SUCCESS_DISTANCE,
        help="GOAT-Bench success threshold: STOP position within this Euclidean distance (m) of the goal instance (default 1.0).",
    )
    args = ap.parse_args()

    scene_path, episode = pick_episode(ROOT / args.dataset_dir, args.split, args.scene, args.episode_index)
    dataset = load_json_gz(scene_path)
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
    config.perception.clip_image_model = args.clip_image_model
    config.perception.clip_device = args.clip_device
    config.perception.backend = args.perception_backend
    config.perception.vlm_api_base = args.vlm_api_base
    config.perception.vlm_model = args.vlm_model
    config.perception.vlm_timeout = args.vlm_timeout
    config.perception.heavy_enabled = bool(args.heavy_enabled or args.perception_backend == "vlm")
    if args.heavy_interval is not None:
        config.perception.heavy_interval = args.heavy_interval
    if args.heavy_reground_cooldown is not None:
        config.perception.heavy_reground_cooldown = args.heavy_reground_cooldown
    if args.heavy_near_goal_cooldown is not None:
        config.perception.heavy_near_goal_cooldown = args.heavy_near_goal_cooldown
    if args.object_detection_threshold is not None:
        config.perception.object_detection_threshold = args.object_detection_threshold
    if args.groundingdino_config is not None:
        config.perception.groundingdino_config = args.groundingdino_config
    if args.groundingdino_checkpoint is not None:
        config.perception.groundingdino_checkpoint = args.groundingdino_checkpoint
    if args.groundingdino_device is not None:
        config.perception.groundingdino_device = args.groundingdino_device
    if "object" in thresholds:
        config.perception.object_threshold = float(thresholds["object"])
    if "room" in thresholds:
        config.perception.room_threshold = float(thresholds["room"])
    if "landmark" in thresholds:
        config.perception.landmark_threshold = float(thresholds["landmark"])
    etp_config = CleanETPGoatConfig() if args.agent_variant == "clean" else ETPGoatConfig()
    if args.ghost_merge_radius is not None:
        etp_config.ghost_merge_radius = args.ghost_merge_radius
    if args.ghost_min_distance is not None:
        etp_config.ghost_min_distance = args.ghost_min_distance
    if args.ghost_confidence is not None:
        etp_config.ghost_confidence = args.ghost_confidence
    if args.ghost_graph_distance_cap is not None:
        etp_config.ghost_graph_distance_cap = args.ghost_graph_distance_cap
    if args.disable_route_next_hop:
        etp_config.route_next_hop_enabled = False
    if args.disable_stop_memory:
        etp_config.stop_memory_enabled = False
    if args.stop_memory_proposal_threshold is not None:
        etp_config.stop_memory_proposal_threshold = args.stop_memory_proposal_threshold
    if args.stop_memory_min_write_score is not None:
        etp_config.stop_memory_min_write_score = args.stop_memory_min_write_score
    if args.stop_memory_score_weight is not None:
        etp_config.stop_memory_score_weight = args.stop_memory_score_weight

    if args.agent_variant == "clean":
        agent = ConfTopoGOATAgentCleanETPNav(config, etp_config)
        agent_variant_name = "ConfTopoGOATCleanETPNavAgent"
    else:
        agent = ConfTopoGOATAgentETPNav(config, etp_config)
        agent_variant_name = "ConfTopoGOATAgentETPNav"
    agent.set_goal(ig)
    encoder = None if args.use_placeholder_embed else GoatModalityClipEncoder(
        args.clip_model, args.clip_image_model, args.clip_device,
    )
    if encoder is not None:
        agent.perceiver.room_text_embeds = encoder.encode_text(agent.perceiver.room_labels)

    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)
    frame_dir = ROOT / args.frame_dir if args.frame_dir else None
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
    sim = make_sim(scene_file)
    sim_agent = sim.initialize_agent(0)
    if hasattr(agent, "set_pathfinder"):
        agent.set_pathfinder(sim)
    # Success metric: Euclidean distance from STOP position to target instance center(s).
    state = set_start_state(sim_agent, episode)
    origin = np.array(state.position, dtype=np.float32)
    executor = PathfinderExecutor()
    tracker = CollisionLikeTracker()
    previous = None
    steps: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    task0_semantic_nodes: set[str] = set()
    per_task_semantic_nodes: dict[int, set[str]] = {}
    per_task_goal_label: dict[int, str] = {}
    try:
        for idx, goal in enumerate(goals):
            before = agent.memory_stats.copy()
            known_before = known_node_ids(agent)
            semantic_before = semantic_node_ids(agent)
            durable_before = durable_memory_node_ids(agent)
            agent.set_new_goal(goal)
            start = len(steps)
            monitor_target_id = None
            monitor_start_step = len(steps)
            monitor_best_distance = None
            goal_min_distance = float("inf")
            goal_min_step = -1
            goal_stopped = False
            goal_stop_distance = float("inf")
            distance_to_target = float("inf")
            goal_path_length = 0.0
            goal_prev_world = None
            memory_reuse_hits = 0
            semantic_reuse_hits = 0
            repeated_goal_reuse_hits = 0
            goal_entries = _resolve_goat_subtask_goal_entries(dataset.get("goals", {}), episode, idx)
            goal_instance_positions = _instance_positions_from_goal_entries(goal_entries)
            goal_view_points = _view_points_from_goal_entries(goal_entries)
            goal_object_ids = [str(g.get("object_id")) for g in goal_entries if g.get("object_id")]
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
                last_debug = (rec.get("memory") or {}).get("last_debug") or {}
                etp_route_debug = rec.get("etp_route_debug") or {}
                reuse_ids = {
                    node_id
                    for node_id in (
                        target_id,
                        rec.get("semantic_target_node_id"),
                        etp_route_debug.get("semantic_reuse_node_id"),
                        last_debug.get("active_anchor_id"),
                    )
                    if node_id
                }
                reused_known = reuse_ids & known_before
                reused_semantic = {
                    node_id for node_id in reused_known if is_semantic_node_id(node_id)
                }
                if reused_known:
                    memory_reuse_hits += len(reused_known)
                    rec["memory_reuse"] = True
                    rec["memory_reuse_node_ids"] = sorted(reused_known)
                if reused_semantic:
                    semantic_reuse_hits += len(reused_semantic)
                    rec["semantic_reuse"] = True
                    rec["semantic_reuse_node_ids"] = sorted(reused_semantic)
                if idx > 0 and reused_semantic:
                    for prev_idx, prev_label in per_task_goal_label.items():
                        if prev_idx < idx and prev_label == goal.target_object:
                            prev_nodes = per_task_semantic_nodes.get(prev_idx, set())
                            repeated_hits = reused_semantic & prev_nodes
                            if repeated_hits:
                                repeated_goal_reuse_hits += len(repeated_hits)
                                rec["repeated_goal_reuse"] = True
                                rec["repeated_goal_reuse_node_ids"] = sorted(repeated_hits)
                                break
                target_distance = planar_target_distance(rec.get("position"), rec.get("target_position"))
                rec["target_distance"] = target_distance
                world_pos_raw = rec.get("world_position")
                if world_pos_raw is not None and goal_instance_positions:
                    agent_world = np.asarray(world_pos_raw, dtype=np.float32)
                    distance_to_target = _euclidean_distance_to_instances(
                        agent_world, goal_instance_positions,
                    )
                    if distance_to_target < goal_min_distance:
                        goal_min_distance = distance_to_target
                        goal_min_step = len(steps)
                rec["distance_to_target"] = (
                    round(float(distance_to_target), 4) if np.isfinite(distance_to_target) else None
                )
                rec["goal_min_distance"] = goal_min_distance
                # Path length tracking (world coordinates)
                world_pos = rec.get("world_position")
                if world_pos is not None:
                    wp = np.asarray(world_pos, dtype=np.float32)
                    step_dist = float(np.linalg.norm(wp - goal_prev_world)) if goal_prev_world is not None else 0.0
                    goal_path_length += step_dist
                    goal_prev_world = wp
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
                    goal_stopped = True
                    goal_stop_distance = distance_to_target
                    break
            after = agent.memory_stats.copy()
            durable_after = durable_memory_node_ids(agent)
            preserved_durable = durable_before & durable_after
            missing_durable = durable_before - durable_after
            # Compute SPL: S_i * l_i / max(p_i, l_i)
            goal_spl = None
            goal_optimal_path = None
            goal_success_bool = (
                goal_stopped
                and goal_stop_distance != float("inf")
                and goal_stop_distance <= args.success_distance
            )
            if goal_success_bool and goal_path_length > 0:
                if goal_view_points and len(steps) > start:
                    first_step = steps[start]
                    start_world = np.asarray(
                        first_step.get("world_position", first_step.get("position", [0, 0, 0])),
                        dtype=np.float32,
                    )
                    if start_world.shape == (3,):
                        best_geodesic = _goat_geodesic_distance_to_viewpoints(
                            sim, start_world, goal_view_points,
                        )
                        if np.isfinite(best_geodesic) and best_geodesic > 0:
                            goal_optimal_path = best_geodesic
            if goal_optimal_path is not None and goal_optimal_path > 0 and goal_path_length > 0:
                goal_spl = round(goal_optimal_path / max(goal_path_length, goal_optimal_path), 4)
            per_task_semantic_nodes[idx] = semantic_node_ids(agent)
            per_task_goal_label[idx] = goal.target_object
            if idx == 0:
                task0_semantic_nodes = per_task_semantic_nodes[0]
            tasks.append({
                "task_index": idx,
                "target_object": goal.target_object,
                "node_count_before": before["total_nodes"],
                "node_count_after": after["total_nodes"],
                "objects_after": after["objects"],
                "rooms_after": after["rooms"],
                "landmarks_after": after["landmarks"],
                "memory_preserved": not missing_durable,
                "durable_nodes_before": len(durable_before),
                "durable_nodes_preserved": len(preserved_durable),
                "missing_durable_node_ids": sorted(missing_durable),
                "known_nodes_before": len(known_before),
                "memory_reuse_hits": memory_reuse_hits,
                "semantic_reuse_hits": semantic_reuse_hits,
                "repeated_goal_reuse_hits": repeated_goal_reuse_hits,
                "repeated_goal_semantic_reuse": repeated_goal_reuse_hits > 0,
                "steps": len(steps) - start,
                "goal_stopped": goal_stopped,
                "goal_stop_distance": round(float(goal_stop_distance), 4) if goal_stop_distance != float("inf") else None,
                "goal_min_distance": float(goal_min_distance) if goal_min_distance != float("inf") else None,
                "goal_view_points": len(goal_view_points),
                "goal_instance_count": len(goal_instance_positions),
                "goal_object_ids": goal_object_ids,
                "goal_success": goal_success_bool,
                "goal_budget_steps": args.steps_per_goal,
                "goal_path_length": round(goal_path_length, 4),
                "goal_optimal_path": round(goal_optimal_path, 4) if goal_optimal_path is not None else None,
                "goal_spl": goal_spl,
                "task_telemetry": (
                    agent.task_telemetry.snapshot()
                    if hasattr(agent, "task_telemetry")
                    else None
                ),
            })
    finally:
        sim.close()

    trace = {
        "agent_variant": agent_variant_name,
        "navigation_style": "rgb_only_etpnav_ghost_graph",
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
            "success_distance": args.success_distance,
            "steps_per_goal": args.steps_per_goal,
            "distance_metric": "euclidean_to_instance_position_at_stop",
            "success_requires_stop": True,
            "success_spec": "goat_bench_readme",
        },
        "perception": {
            "backend": config.perception.backend,
            "heavy_enabled": config.perception.heavy_enabled,
            "heavy_interval": config.perception.heavy_interval,
            "heavy_reground_cooldown": config.perception.heavy_reground_cooldown,
            "heavy_near_goal_cooldown": config.perception.heavy_near_goal_cooldown,
            "vlm_api_base": config.perception.vlm_api_base,
            "vlm_model": config.perception.vlm_model,
            "vlm_timeout": config.perception.vlm_timeout,
        },
        "etpnav": {
            "ghost_rays": [[float(a), float(d)] for a, d in etp_config.ghost_rays],
            "ghost_merge_radius": etp_config.ghost_merge_radius,
            "ghost_min_distance": etp_config.ghost_min_distance,
            "ghost_confidence": etp_config.ghost_confidence,
            "ghost_graph_distance_cap": etp_config.ghost_graph_distance_cap,
            "route_next_hop_enabled": etp_config.route_next_hop_enabled,
            "semantic_context_weight": etp_config.semantic_context_weight,
            "semantic_direction_weight": etp_config.semantic_direction_weight,
            "repeat_candidate_penalty_weight": etp_config.repeat_candidate_penalty_weight,
            "stop_memory_enabled": etp_config.stop_memory_enabled,
            "stop_memory_min_write_score": etp_config.stop_memory_min_write_score,
            "stop_memory_proposal_threshold": etp_config.stop_memory_proposal_threshold,
            "stop_memory_score_weight": etp_config.stop_memory_score_weight,
            "stop_memory_min_bbox_score": etp_config.stop_memory_min_bbox_score,
            "stop_memory_require_anchor": etp_config.stop_memory_require_anchor,
            "min_forward_before_stop": etp_config.min_forward_before_stop,
            "min_approach_distance": etp_config.min_approach_distance,
            "bbox_min_growth_ratio": etp_config.bbox_min_growth_ratio,
            "stop_require_bbox_growth": etp_config.stop_require_bbox_growth,
            "stop_min_fresh_bbox": etp_config.stop_min_fresh_bbox,
            "stop_short_approach_max": etp_config.stop_short_approach_max,
        },
        "task_summaries": tasks,
        "steps": steps,
        "final_memory": agent.memory_stats,
        "final_summary": {
            "episode_length": len(steps),
            "target_switch_count": len(goals) - 1,
            "goals_total": len(goals),
            "goals_success": sum(1 for t in tasks if t.get("goal_success")),
            "goals_min_distances": [t.get("goal_min_distance") for t in tasks],
            "goals_path_lengths": [t.get("goal_path_length") for t in tasks],
            "goals_optimal_paths": [t.get("goal_optimal_path") for t in tasks],
            "goals_spl": [t.get("goal_spl") for t in tasks],
            "avg_spl": round(sum(t.get("goal_spl", 0) or 0 for t in tasks) / max(len(tasks), 1), 4),
            "memory_reuse_count": sum(t["memory_reuse_hits"] for t in tasks),
            "semantic_reuse_count": sum(t["semantic_reuse_hits"] for t in tasks),
            "semantic_node_count": agent.memory_stats["objects"] + agent.memory_stats["rooms"] + agent.memory_stats["landmarks"],
            "heavy_perception_calls": agent.memory_stats.get("heavy_perception_calls", 0),
            "last_heavy": agent.memory_stats.get("last_heavy", {}),
            "object_merge_count": agent.memory_stats.get("object_merge_count", 0),
            "mean_object_confidence": agent.memory_stats.get("mean_object_confidence", 0.0),
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
        "agent_variant": trace["agent_variant"],
        "navigation_style": trace["navigation_style"],
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
