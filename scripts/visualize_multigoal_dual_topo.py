#!/usr/bin/env python3
"""Batch visualizer for multigoal GOAT topo traces.

Layout:
    first-person RGB | navigation-level topo map
    structure-level topo map | debug state panel

The script accepts report JSON paths and derives the sibling *_multigoal.json
trace path, or accepts trace paths directly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

NAV_TYPES = {"waypoint_visited", "waypoint_frontier", "waypoint_candidate", "object", "landmark"}
STRUCT_TYPES = {"room", "landmark"}

NAV_STYLE = {
    "waypoint_visited": ("#8a8a8a", "o", 14),
    "waypoint_frontier": ("#2563eb", "^", 34),
    "waypoint_candidate": ("#7dd3fc", "o", 14),
    "object": ("#f97316", "o", 28),
    "landmark": ("#22c55e", "s", 20),
}


def resolve_repo_path(path_like: str | Path | None) -> Path | None:
    """Resolve trace paths that may come from another workspace mount."""
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and (ROOT / path).exists():
        return ROOT / path
    parts = path.parts
    if "data" in parts:
        candidate = ROOT.joinpath(*parts[parts.index("data") :])
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def fig_to_rgb(fig):
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()


def derive_trace_path(path: str) -> Path:
    p = Path(path)
    if p.name.endswith("_report.json"):
        p = p.with_name(p.name.replace("_report.json", "_multigoal.json"))
    resolved = resolve_repo_path(p)
    if resolved is None:
        raise FileNotFoundError(f"Trace not found for {path}")
    return resolved


def xyz_to_xz(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("position")
    if value is None or len(value) < 3:
        return None
    return float(value[0]), float(value[2])


def node_pos(node):
    return xyz_to_xz(node.get("position"))


def load_origin_world(trace: dict):
    if trace.get("origin_world") is not None:
        return np.asarray(trace["origin_world"], dtype=np.float32), "trace.origin_world"
    episode_file = resolve_repo_path(trace.get("episode_file"))
    episode_id = trace.get("episode_id")
    if episode_file is None or episode_id is None:
        return None, "missing origin_world and episode_file"
    try:
        with gzip.open(episode_file, "rt") as f:
            data = json.load(f)
        for episode in data.get("episodes", []):
            if str(episode.get("episode_id")) == str(episode_id):
                return np.asarray(episode["start_position"], dtype=np.float32), f"{episode_file}:episode_{episode_id}"
    except Exception as exc:
        return None, f"failed to read origin from episode_file: {exc}"
    return None, f"episode_id={episode_id} not found in {episode_file}"


def infer_position_frame(trace: dict, origin_world=None):
    frame = trace.get("coordinate_frame")
    if frame:
        return frame, "trace.coordinate_frame"
    steps = trace.get("steps") or []
    if not steps:
        return "episode_start_relative", "default: no steps"
    first = np.asarray(steps[0].get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    if origin_world is not None:
        origin = np.asarray(origin_world, dtype=np.float32)
        if np.linalg.norm(first[[0, 2]] - origin[[0, 2]]) < 0.5:
            return "world", "inferred: first step near origin_world"
    if np.linalg.norm(first[[0, 2]]) < 0.5:
        return "episode_start_relative", "inferred: first step near [0, 0]"
    return "world", "inferred: legacy trace without coordinate_frame"


def position_to_world(pos, origin_world, frame):
    arr = np.asarray(pos, dtype=np.float32)
    if origin_world is not None and frame == "episode_start_relative":
        return arr + origin_world
    return arr


def infer_navmesh_path(trace: dict, override: str | None = None):
    if override:
        path = resolve_repo_path(override)
        return path, "override" if path else f"override not found: {override}"
    scene_file = resolve_repo_path(trace.get("scene_file"))
    if scene_file is not None:
        candidates = []
        name = scene_file.name
        if name.endswith(".basis.glb"):
            candidates.append(scene_file.with_name(name.replace(".basis.glb", ".basis.navmesh")))
        if scene_file.suffix:
            candidates.append(scene_file.with_suffix(".navmesh"))
        candidates.append(scene_file.with_name(scene_file.stem + ".navmesh"))
        for candidate in candidates:
            if candidate.exists():
                return candidate, "scene_file"
    scene = trace.get("scene") or (Path(str(trace.get("scene_file", ""))).name.split(".")[0] if trace.get("scene_file") else "")
    scene = str(scene).replace(".basis", "")
    if scene:
        matches = sorted((ROOT / "data/scene_datasets").glob(f"**/{scene}*.navmesh"))
        if matches:
            return matches[0], "scene search"
    return None, "navmesh not found"


def load_gt_topdown(trace: dict, resolution: float, navmesh_override: str | None = None, disabled: bool = False) -> dict:
    if disabled:
        return {"available": False, "reason": "disabled by --no-gt-topdown"}
    navmesh, navmesh_source = infer_navmesh_path(trace, navmesh_override)
    origin_world, origin_source = load_origin_world(trace)
    position_frame, position_frame_source = infer_position_frame(trace, origin_world)
    if navmesh is None:
        return {
            "available": False,
            "reason": navmesh_source,
            "origin_world": origin_world,
            "origin_source": origin_source,
            "position_frame": position_frame,
            "position_frame_source": position_frame_source,
        }
    if origin_world is None:
        return {
            "available": False,
            "reason": origin_source,
            "navmesh_path": str(navmesh),
            "navmesh_source": navmesh_source,
            "position_frame": position_frame,
            "position_frame_source": position_frame_source,
        }
    try:
        import habitat_sim

        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh)):
            return {
                "available": False,
                "reason": f"failed to load navmesh: {navmesh}",
                "origin_world": origin_world,
                "origin_source": origin_source,
                "navmesh_path": str(navmesh),
                "navmesh_source": navmesh_source,
                "position_frame": position_frame,
                "position_frame_source": position_frame_source,
            }
        bounds_min, bounds_max = pathfinder.get_bounds()
        slice_height = float(bounds_min[1] + 0.1)
        topdown = np.asarray(pathfinder.get_topdown_view(resolution, slice_height), dtype=bool)
        return {
            "available": True,
            "topdown": topdown,
            "bounds_min": np.asarray(bounds_min, dtype=np.float32),
            "bounds_max": np.asarray(bounds_max, dtype=np.float32),
            "origin_world": origin_world,
            "origin_source": origin_source,
            "position_frame": position_frame,
            "position_frame_source": position_frame_source,
            "navmesh_path": str(navmesh),
            "navmesh_source": navmesh_source,
            "resolution": resolution,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"topdown unavailable: {exc}",
            "origin_world": origin_world,
            "origin_source": origin_source,
            "navmesh_path": str(navmesh),
            "navmesh_source": navmesh_source,
            "position_frame": position_frame,
            "position_frame_source": position_frame_source,
        }


def to_plot_pos(pos, gt_ctx):
    if pos is None:
        return None
    if gt_ctx.get("available"):
        p = position_to_world(pos, gt_ctx.get("origin_world"), gt_ctx.get("position_frame", "episode_start_relative"))
        return float(p[0]), float(p[2])
    return xyz_to_xz(pos)


def node_plot_pos(node, gt_ctx):
    pos = node.get("position")
    return to_plot_pos(pos, gt_ctx)


def collect_bounds(trace: dict, gt_ctx: dict):
    if gt_ctx.get("available"):
        bmin, bmax = gt_ctx["bounds_min"], gt_ctx["bounds_max"]
        return float(bmin[0]), float(bmax[0]), float(bmin[2]), float(bmax[2])
    xs, zs = [], []
    for st in trace.get("steps", []):
        for pos in (st.get("position"), st.get("target_position")):
            xz = xyz_to_xz(pos)
            if xz:
                xs.append(xz[0])
                zs.append(xz[1])
        for node in st.get("topo", {}).get("nodes", []):
            xz = node_pos(node)
            if xz:
                xs.append(xz[0])
                zs.append(xz[1])
    if not xs:
        return -5.0, 5.0, -5.0, 5.0
    margin = max(1.0, 0.08 * max(max(xs) - min(xs), max(zs) - min(zs)))
    return min(xs) - margin, max(xs) + margin, min(zs) - margin, max(zs) + margin


def draw_base(ax, gt_ctx, bounds, title):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.tick_params(labelsize=7)
    ax.grid(color="#e5e7eb", linewidth=0.35, alpha=0.7)
    if gt_ctx.get("available"):
        topdown = gt_ctx["topdown"]
        bmin, bmax = gt_ctx["bounds_min"], gt_ctx["bounds_max"]
        extent = [float(bmin[0]), float(bmax[0]), float(bmax[2]), float(bmin[2])]
        ax.imshow(topdown, cmap="gray", origin="upper", extent=extent, alpha=0.42, zorder=0)


def draw_agent(ax, step, gt_ctx):
    pos = to_plot_pos(step.get("position"), gt_ctx)
    if pos is None:
        return
    heading = step.get("world_heading", step.get("heading", 0.0)) or 0.0
    dx = 0.45 * math.sin(float(heading))
    dz = 0.45 * math.cos(float(heading))
    ax.arrow(pos[0], pos[1], dx, dz, width=0.035, head_width=0.22, head_length=0.28, color="#dc2626", zorder=12)


def draw_trajectory(ax, steps, idx, gt_ctx):
    pts = []
    for st in steps[: idx + 1]:
        pos = to_plot_pos(st.get("position"), gt_ctx)
        if pos:
            pts.append(pos)
    if len(pts) > 1:
        arr = np.asarray(pts)
        ax.plot(arr[:, 0], arr[:, 1], color="#111827", linewidth=1.2, alpha=0.72, zorder=3)


def selected_ids(step):
    ids = set()
    for key in (
        "target_node_id",
        "semantic_target_node_id",
        "reground_target_node_id",
        "anchor_waypoint_id",
        "reground_anchor_node_id",
    ):
        value = step.get(key)
        if value:
            ids.add(value)
    return ids


def draw_navigation(ax, trace, idx, gt_ctx, bounds):
    step = trace["steps"][idx]
    draw_base(ax, gt_ctx, bounds, "Navigation TopoMap")
    nodes = {n.get("id"): n for n in step.get("topo", {}).get("nodes", [])}

    for edge in step.get("topo", {}).get("edges", []):
        if edge.get("type") not in ("navigable", "observed_at", "visible_from"):
            continue
        a, b = nodes.get(edge.get("source")), nodes.get(edge.get("target"))
        if not a or not b:
            continue
        pa, pb = node_plot_pos(a, gt_ctx), node_plot_pos(b, gt_ctx)
        if not pa or not pb:
            continue
        is_nav = edge.get("type") == "navigable"
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#9ca3af", linewidth=0.7 if is_nav else 0.35, linestyle="-" if is_nav else ":", alpha=0.55, zorder=1)

    chosen = selected_ids(step)
    for node in nodes.values():
        nt = node.get("type")
        if nt not in NAV_TYPES:
            continue
        pos = node_plot_pos(node, gt_ctx)
        if not pos:
            continue
        color, marker, size = NAV_STYLE.get(nt, ("#666", "o", 16))
        if nt == "object" and (node.get("attributes") or {}).get("folded_anchor"):
            marker, color, size = "*", "#ef4444", 85
        ax.scatter(pos[0], pos[1], s=size, marker=marker, color=color, edgecolors="#111827", linewidths=0.25, alpha=0.92, zorder=5)
        if node.get("id") in chosen:
            ax.scatter(pos[0], pos[1], s=size + 90, marker="o", facecolors="none", edgecolors="#facc15", linewidths=2.2, zorder=9)

    target = to_plot_pos(step.get("target_position"), gt_ctx)
    if target:
        ax.scatter(target[0], target[1], s=120, marker="x", color="#facc15", linewidths=2.5, zorder=10)
    draw_trajectory(ax, trace["steps"], idx, gt_ctx)
    draw_agent(ax, step, gt_ctx)


def draw_structure(ax, trace, idx, gt_ctx, bounds):
    step = trace["steps"][idx]
    draw_base(ax, gt_ctx, bounds, "Structure TopoMap")
    nodes = {n.get("id"): n for n in step.get("topo", {}).get("nodes", [])}

    for edge in step.get("topo", {}).get("edges", []):
        if edge.get("type") not in ("adjacent_to", "belongs_to"):
            continue
        a, b = nodes.get(edge.get("source")), nodes.get(edge.get("target"))
        if not a or not b:
            continue
        pa, pb = node_plot_pos(a, gt_ctx), node_plot_pos(b, gt_ctx)
        if not pa or not pb:
            continue
        room_edge = edge.get("type") == "adjacent_to"
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#16a34a" if room_edge else "#a7f3d0", linewidth=1.45 if room_edge else 0.55, alpha=0.68, zorder=2)

    chosen = selected_ids(step)
    for node in nodes.values():
        nt = node.get("type")
        attrs = node.get("attributes") or {}
        pos = node_plot_pos(node, gt_ctx)
        if not pos:
            continue
        if nt == "room":
            label = node.get("label") or "room"
            ax.scatter(pos[0], pos[1], s=180, marker="o", color="#8b5cf6", alpha=0.45, edgecolors="#4c1d95", linewidths=0.9, zorder=5)
            ax.text(pos[0], pos[1] + 0.28, label[:22], fontsize=7, color="#4c1d95", ha="center", va="bottom", zorder=8)
            contains = attrs.get("contains_labels") or []
            if contains:
                ax.text(pos[0], pos[1] - 0.34, ", ".join(contains[:3])[:30], fontsize=5.5, color="#6d28d9", ha="center", va="top", zorder=8)
        elif nt == "landmark" and attrs.get("structure_role"):
            role = attrs.get("structure_role")
            marker = "D" if role == "portal" else "s"
            color = "#22c55e" if role == "portal" else "#14b8a6"
            ax.scatter(pos[0], pos[1], s=58, marker=marker, color=color, edgecolors="#064e3b", linewidths=0.5, zorder=6)
            label = node.get("label") or role
            ax.text(pos[0], pos[1] + 0.2, label[:18], fontsize=5.5, color="#065f46", ha="center", va="bottom", zorder=8)
        else:
            continue
        if node.get("id") in chosen:
            ax.scatter(pos[0], pos[1], s=245, marker="o", facecolors="none", edgecolors="#facc15", linewidths=2.2, zorder=9)

    draw_agent(ax, step, gt_ctx)


def draw_rgb(ax, trace, idx):
    step = trace["steps"][idx]
    frame = step.get("rgb_frame")
    frame_path = resolve_repo_path(frame) if frame else None
    if frame_path is not None:
        rgb = imageio.imread(frame_path)
        ax.imshow(rgb)
        title = f"First-person RGB | step {step.get('step', idx)}"
    else:
        ax.set_facecolor("#111827")
        ax.text(0.5, 0.56, "RGB frame not recorded", color="#e5e7eb", fontsize=18, fontweight="bold", ha="center", va="center")
        ax.text(0.5, 0.45, f"scene: {trace.get('scene', '?')} | step {step.get('step', idx)}", color="#9ca3af", fontsize=10, ha="center", va="center")
        title = "First-person RGB + bbox"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def compact_id(value, width=42):
    if not value:
        return "none"
    value = str(value)
    return value if len(value) <= width else value[: width - 3] + "..."


def draw_debug(ax, trace, idx, gt_ctx):
    step = trace["steps"][idx]
    topo = step.get("topo", {})
    nodes = topo.get("nodes", [])
    counts = {}
    for node in nodes:
        counts[node.get("type", "?")] = counts.get(node.get("type", "?"), 0) + 1
    goal = step.get("goal") or {}
    nav_debug = step.get("navigation_debug") or {}
    mem = step.get("memory") or {}
    target_pos = step.get("target_position")
    target_pos_txt = "none" if target_pos is None else np.asarray(target_pos, dtype=float).round(3).tolist()

    lines = [
        f"Scene: {trace.get('scene', '?')}    Frame: {idx + 1}/{len(trace['steps'])}",
        f"Task: {step.get('task_index', '?')}    Goal: {goal.get('target_object', '?')}",
        "",
        "Stage 1 structure target",
        f"  id: {compact_id(step.get('anchor_room_id') or step.get('structure_target_id') or step.get('semantic_target_node_id'))}",
        f"  type: {step.get('target_anchor_type', 'n/a')}    score: {nav_debug.get('structure_score', 'n/a')}",
        "",
        "Stage 2 navigation target",
        f"  id: {compact_id(step.get('target_node_id'))}",
        f"  semantic_id: {compact_id(step.get('semantic_target_node_id'))}",
        f"  anchor_wp: {compact_id(step.get('anchor_waypoint_id') or step.get('reground_anchor_node_id'))}",
        f"  target_pos: {target_pos_txt}",
        f"  reground: {step.get('reground_state', 'idle')}    requires: {step.get('requires_regrounding', False)}",
        "",
        "Action / progress",
        f"  plan: {step.get('plan_mode', 'n/a')}    low: {step.get('low_action', 'n/a')}    agent: {step.get('agent_action', 'n/a')}",
        f"  target_dist: {step.get('target_distance', 'n/a')}    goal_min: {step.get('goal_min_distance', 'n/a')}",
        f"  collision_like: {step.get('collision_like', False)}    topo bg: {'GT navmesh' if gt_ctx.get('available') else 'relative grid'}",
        "",
        "Topo counts",
        f"  nodes: {len(nodes)}    edges: {len(topo.get('edges', []))}",
        f"  wp: {counts.get('waypoint_visited', 0)} visited / {counts.get('waypoint_frontier', 0)} frontier / {counts.get('waypoint_candidate', 0)} candidate",
        f"  objects: {counts.get('object', 0)}    rooms: {counts.get('room', 0)}    landmarks: {counts.get('landmark', 0)}",
        "",
        "Memory",
        f"  reuse: {mem.get('memory_reuse_hits', 'n/a')}    semantic: {mem.get('semantic_reuse_hits', 'n/a')}",
        f"  heavy_calls: {mem.get('heavy_perception_calls', 'n/a')}    mean_conf: {mem.get('mean_object_confidence', 'n/a')}",
    ]
    if not gt_ctx.get("available"):
        lines += ["", f"Topdown unavailable: {gt_ctx.get('reason', 'unknown')}"]

    ax.axis("off")
    ax.set_facecolor("#f8fafc")
    ax.text(0.02, 0.97, "\n".join(lines), ha="left", va="top", fontsize=8, family="monospace", color="#111827")


def frame_indices(n_steps: int, stride: int, max_frames: int | None):
    if max_frames and n_steps > max_frames:
        return sorted(set(np.linspace(0, n_steps - 1, max_frames).astype(int).tolist()))
    return list(range(0, n_steps, max(1, stride)))


def render_frame(trace, idx, gt_ctx, bounds):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1.0, 1.0], left=0.035, right=0.985, top=0.94, bottom=0.055, wspace=0.16, hspace=0.22)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_nav = fig.add_subplot(gs[0, 1])
    ax_struct = fig.add_subplot(gs[1, 0])
    ax_debug = fig.add_subplot(gs[1, 1])

    draw_rgb(ax_rgb, trace, idx)
    draw_navigation(ax_nav, trace, idx, gt_ctx, bounds)
    draw_structure(ax_struct, trace, idx, gt_ctx, bounds)
    draw_debug(ax_debug, trace, idx, gt_ctx)

    step = trace["steps"][idx]
    goal = (step.get("goal") or {}).get("target_object", "?")
    fig.suptitle(f"GOAT multigoal dual topo | {trace.get('scene', '?')} | task {step.get('task_index', '?')} | goal: {goal}", fontsize=13, fontweight="bold")
    image = fig_to_rgb(fig)
    plt.close(fig)
    return image


def visualize_one(trace_path: Path, out_dir: Path, args):
    print(f"[load] {trace_path}")
    trace = load_json(trace_path)
    steps = trace.get("steps") or []
    if not steps:
        raise ValueError(f"No steps in {trace_path}")
    scene = trace.get("scene") or trace_path.name.replace("_multigoal.json", "")
    scene_out = out_dir / scene
    scene_out.mkdir(parents=True, exist_ok=True)

    gt_ctx = load_gt_topdown(trace, args.topdown_resolution, args.navmesh_path, args.no_gt_topdown)
    bounds = collect_bounds(trace, gt_ctx)
    indices = frame_indices(len(steps), args.stride, args.max_frames)
    frames = []
    key_frames = []
    print(f"[render] {scene}: {len(indices)} frames from {len(steps)} steps")
    for out_i, idx in enumerate(indices):
        image = render_frame(trace, idx, gt_ctx, bounds)
        frames.append(image)
        if args.save_key_frames and (out_i in {0, len(indices) // 2, len(indices) - 1}):
            frame_path = scene_out / f"frame_{idx:06d}.png"
            imageio.imwrite(frame_path, image)
            key_frames.append(str(frame_path))
        if (out_i + 1) % 25 == 0 or out_i + 1 == len(indices):
            print(f"  {scene}: {out_i + 1}/{len(indices)}")

    video_path = scene_out / f"{scene}_dual_topo.mp4"
    imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=8)
    summary = {
        "trace": str(trace_path),
        "scene": scene,
        "steps": len(steps),
        "rendered_frames": len(indices),
        "fps": args.fps,
        "video": str(video_path),
        "key_frames": key_frames,
        "gt_topdown": {
            "available": bool(gt_ctx.get("available")),
            "reason": gt_ctx.get("reason", ""),
            "navmesh_path": gt_ctx.get("navmesh_path", ""),
            "navmesh_source": gt_ctx.get("navmesh_source", ""),
            "origin_source": gt_ctx.get("origin_source", ""),
            "origin_world": None if gt_ctx.get("origin_world") is None else np.asarray(gt_ctx.get("origin_world")).round(4).tolist(),
            "resolution": gt_ctx.get("resolution", args.topdown_resolution),
            "position_frame": gt_ctx.get("position_frame", trace.get("coordinate_frame", "")),
            "position_frame_source": gt_ctx.get("position_frame_source", ""),
        },
        "rgb_frame_dir": trace.get("rgb_frame_dir"),
        "final_summary": trace.get("final_summary", {}),
    }
    (scene_out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {video_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Batch multigoal dual-topo visualizer")
    ap.add_argument("--trace", default=None, help="Single *_multigoal.json trace path, matching visualize_goat_topo_trace.py style")
    ap.add_argument("--reports", nargs="*", default=[], help="*_report.json paths; sibling *_multigoal.json traces are used")
    ap.add_argument("--traces", nargs="*", default=[], help="*_multigoal.json trace paths")
    ap.add_argument("--out-dir", default="data/logs/goat_topo/final_14scenes/dual_topo_viz")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=240, help="Sample evenly if trace has more frames than this; 0 disables")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--topdown-resolution", type=float, default=0.05)
    ap.add_argument("--navmesh-path", default=None)
    ap.add_argument("--no-gt-topdown", action="store_true")
    ap.add_argument("--save-key-frames", action="store_true")
    args = ap.parse_args()
    if args.max_frames <= 0:
        args.max_frames = None

    trace_inputs = list(args.traces)
    if args.trace:
        trace_inputs.insert(0, args.trace)
    paths = [derive_trace_path(p) for p in args.reports] + [derive_trace_path(p) for p in trace_inputs]
    if not paths:
        raise SystemExit("Provide --reports or --traces")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [visualize_one(path, out_dir, args) for path in paths]
    manifest = {"outputs": summaries}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
