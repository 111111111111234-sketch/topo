#!/usr/bin/env python3
"""ConfTopo trace visualizer: Navigation TopoMap + Structure TopoMap + Debug Panel.

Usage:
    python3 scripts/viz_topo_trace.py --trace data/logs/.../topo_trace_multigoal.json --out-dir viz_frames
    python3 scripts/viz_topo_trace.py --trace data/logs/.../topo_trace_multigoal.json --video viz.mp4
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np


# ==================== Constants ====================
NODE_COLORS = {
    "waypoint_visited": "#888888",
    "waypoint_frontier": "#4A90D9",
    "waypoint_candidate": "#7EC8E3",
    "object": "#E87A00",
    "landmark": "#3CB371",
    "room": "#9B59B6",
}

NODE_MARKERS = {
    "waypoint_visited": "o",
    "waypoint_frontier": "^",
    "waypoint_candidate": ".",
    "object": "D",
    "landmark": "s",
    "room": "o",
}

NODE_SIZES = {
    "waypoint_visited": 8,
    "waypoint_frontier": 12,
    "waypoint_candidate": 6,
    "object": 14,
    "landmark": 10,
    "room": 30,
}

EDGE_COLORS = {
    "navigable": "#AAAAAA",
    "adjacent_to": "#5DADE2",
    "belongs_to": "#D5D5D5",
    "observed_at": "#E8E8E8",
    "visible_from": "#E0E0E0",
}

# ==================== Parsing ====================
def load_trace(path):
    with open(path) as f:
        return json.load(f)


def xy(node_or_pos):
    if isinstance(node_or_pos, dict):
        p = node_or_pos.get("position", node_or_pos)
        return (float(p[0]), float(p[2]))
    return (float(node_or_pos[0]), float(node_or_pos[2]))


def get_scene_bounds(steps):
    xs, zs = [], []
    for step in steps:
        pos = step.get("position")
        if pos:
            xs.append(pos[0]); zs.append(pos[2])
        for node in step.get("topo", {}).get("nodes", []):
            p = node.get("position")
            if p:
                xs.append(p[0]); zs.append(p[2])
    if not xs:
        return -5, 5, -5, 5
    margin = max(1.0, (max(xs) - min(xs)) * 0.1)
    return min(xs) - margin, max(xs) + margin, min(zs) - margin, max(zs) + margin


# ==================== Drawing ====================
def draw_navigation_map(ax, step, agent_pos, target_pos, scene_bounds):
    xmin, xmax, zmin, zmax = scene_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.set_title("Navigation TopoMap", fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.tick_params(labelsize=6)

    topo = step.get("topo", {})
    nodes = {n["id"]: n for n in topo.get("nodes", [])}
    edges = topo.get("edges", [])

    # Edges
    for e in edges:
        src, tgt, etype = e.get("source"), e.get("target"), e.get("type", "")
        sn, tn = nodes.get(src), nodes.get(tgt)
        if sn is None or tn is None:
            continue
        if etype not in ("navigable", "observed_at", "visible_from"):
            continue
        (x1, y1), (x2, y2) = xy(sn), xy(tn)
        color = EDGE_COLORS.get(etype, "#ccc")
        lw = 0.6 if etype == "navigable" else 0.3
        ls = "-" if etype == "navigable" else ":"
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, alpha=0.5)

    # Nodes
    for nid, node in nodes.items():
        nt = node.get("type", "")
        if nt not in NODE_COLORS:
            continue
        if nt == "room":
            continue  # draw in structure map
        x, y = xy(node)
        color = NODE_COLORS.get(nt, "#ccc")
        marker = NODE_MARKERS.get(nt, "o")
        size = NODE_SIZES.get(nt, 8)

        # Highlight folded anchor
        attrs = node.get("attributes", {})
        if attrs.get("is_semantic_anchor") and attrs.get("folded_detail"):
            ax.scatter(x, y, s=size * 4, marker="*", color="red", zorder=6, edgecolors="darkred", linewidths=0.5)
            continue

        ax.scatter(x, y, s=size, marker=marker, color=color, zorder=4, edgecolors="white" if nt != "waypoint_visited" else None, linewidths=0.3)

    # Agent trajectory
    traj_x, traj_z = [], []
    # We'll draw trajectory from the step history - stored per frame
    # For now, just mark the current position
    if agent_pos:
        ax.scatter(agent_pos[0], agent_pos[2], s=60, marker=">", color="red", zorder=10, label="agent")
        ax.annotate("agent", (agent_pos[0], agent_pos[2]), fontsize=5, xytext=(3, 3),
                    textcoords="offset points", color="red")

    # Target
    if target_pos:
        ax.scatter(target_pos[0], target_pos[2], s=50, marker="+", color="yellow", zorder=9, linewidths=2)
        ax.annotate("target", (target_pos[0], target_pos[2]), fontsize=5, xytext=(3, 3),
                    textcoords="offset points", color="gold")

    ax.legend(fontsize=5, loc="upper left", framealpha=0.7).set_zorder(20)


def draw_structure_map(ax, step, agent_pos, scene_bounds):
    xmin, xmax, zmin, zmax = scene_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.set_title("Structure TopoMap", fontsize=10, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.tick_params(labelsize=6)

    topo = step.get("topo", {})
    nodes = {n["id"]: n for n in topo.get("nodes", [])}
    edges = topo.get("edges", [])

    # ADJACENT_TO edges
    for e in edges:
        if e.get("type") != "adjacent_to":
            continue
        sn, tn = nodes.get(e["source"]), nodes.get(e["target"])
        if sn is None or tn is None:
            continue
        x1, y1 = xy(sn); x2, y2 = xy(tn)
        ax.plot([x1, x2], [y1, y2], color=EDGE_COLORS["adjacent_to"], linewidth=1.0, linestyle="-", alpha=0.7)
        # Show passage type if available
        passage = e.get("passage_type", "")
        if passage:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, passage, fontsize=4, color="green", alpha=0.6,
                    ha="center", va="center", style="italic")

    # BELONGS_TO edges (light)
    for e in edges:
        if e.get("type") != "belongs_to":
            continue
        sn, tn = nodes.get(e["source"]), nodes.get(e["target"])
        if sn is None or tn is None:
            continue
        x1, y1 = xy(sn); x2, y2 = xy(tn)
        ax.plot([x1, x2], [y1, y2], color="#ddd", linewidth=0.4, linestyle="--", alpha=0.3)

    # Room nodes
    for nid, node in nodes.items():
        if node.get("type") != "room":
            continue
        attrs = node.get("attributes", {})
        if attrs.get("summary_type") != "room_region":
            continue
        x, y = xy(node)
        # Semi-transparent circle
        circle = mpatches.Circle((x, y), radius=0.8, color=NODE_COLORS["room"],
                                 alpha=0.2, zorder=2)
        ax.add_patch(circle)
        ax.scatter(x, y, s=NODE_SIZES["room"], marker="o", color=NODE_COLORS["room"],
                   zorder=4, edgecolors="darkviolet", linewidths=0.5)
        label = node.get("label", "")
        if label:
            ax.text(x, y - 0.5, label, fontsize=6, color="purple",
                    ha="center", va="top", fontweight="bold")
        # Room summary
        summary = attrs.get("semantic_summary", {})
        contains = summary.get("contains_labels", {})
        if contains:
            items = sorted(contains.items(), key=lambda x: -x[1])[:3]
            txt = ", ".join([f"{c} {l}" for l, c in items])
            ax.text(x + 0.5, y + 0.3, txt, fontsize=4, color="purple",
                    ha="left", va="center", alpha=0.7, style="italic")

    # Portal/structural landmarks
    for nid, node in nodes.items():
        if node.get("type") != "landmark":
            continue
        attrs = node.get("attributes", {})
        if not attrs.get("structure_role"):
            continue
        x, y = xy(node)
        role = attrs.get("structure_role", "")
        if role == "portal":
            ax.scatter(x, y, s=30, marker="D", color="#5DADE2", zorder=5,
                       edgecolors="blue", linewidths=0.5)
        else:
            ax.scatter(x, y, s=12, marker="s", color="#3CB371", zorder=4,
                       edgecolors="darkgreen", linewidths=0.3)
            label = node.get("label", "")
            if label:
                ax.text(x, y + 0.2, label, fontsize=4, color="green",
                        ha="center", va="bottom")

    # Agent position
    if agent_pos:
        ax.scatter(agent_pos[0], agent_pos[2], s=40, marker=">", color="red", zorder=10)

    ax.legend(fontsize=5, loc="upper left", framealpha=0.7).set_zorder(20)


def draw_debug_panel(ax, step, task_idx, n_steps):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    mem = step.get("memory", {})
    lines = []
    lines.append(f"Step: {step.get('step', 0)} / {n_steps} | Task: {task_idx}")
    lines.append(f"Goal: {step.get('goal', {}).get('target_object', '?')}")
    lines.append(f"Agent pos: ({step.get('position') or ['?']*3})"[:60])

    # Plan info
    plan_mode = step.get("plan_mode", step.get("mode", "normal"))
    lines.append(f"Plan mode: {plan_mode}")
    struct_id = step.get("structure_target_id", "none")
    lines.append(f"Structure target: {struct_id if struct_id else 'none'}")
    tgt_id = step.get("target_node_id", "none")
    lines.append(f"Target node: {tgt_id if tgt_id else 'none'}")

    # Reground info
    req_reground = step.get("requires_regrounding", False)
    reground_state = step.get("reground_state", "idle")
    lines.append(f"Reground: {reground_state} (req={req_reground})")

    # Stop debug
    stop_debug = step.get("_last_stop_debug", step.get("last_stop_debug", {}))
    if stop_debug:
        lines.append(f"Stop debug: {stop_debug}")

    # Goal proximity
    gmd = step.get("goal_min_distance")
    if gmd is not None:
        gmd_str = f"{gmd:.3f}"
    else:
        gmd_str = "inf"
    lines.append(f"Goal min dist: {gmd_str}")

    # Memory stats
    lines.append(f"Objects: {mem.get('objects', 0)} | Rooms: {mem.get('rooms', 0)} | "
                 f"Landmarks: {mem.get('landmarks', 0)}")
    lines.append(f"Frontiers: {mem.get('frontiers', 0)} | Waypoints: {mem.get('visited_waypoints', 0)}")
    lines.append(f"Semantic anchors: {mem.get('semantic_anchors', 0)} | Merges: {mem.get('object_merge_count', 0)}")
    lines.append(f"Heavy calls: {mem.get('heavy_perception_calls', 0)} | "
                 f"Mean conf: {mem.get('mean_object_confidence', 0):.3f}")
    lines.append(f"Collisions: {mem.get('collision_like_count', 0)}")

    # Granularity debug
    gd = mem.get("granularity_debug", {})
    if gd:
        lines.append(f"Far objects: {gd.get('far_object_candidates', 0)} | "
                     f"Folded marks: {gd.get('folded_anchor_marks', 0)}")

    # Topo counts
    nodes = step.get("topo", {}).get("nodes", [])
    edges = step.get("topo", {}).get("edges", [])
    lines.append(f"Topo nodes: {len(nodes)} | edges: {len(edges)}")

    # Render
    y_pos = 0.95
    for line in lines:
        if y_pos < 0.02:
            break
        ax.text(0.02, y_pos, line, fontsize=6, fontfamily="monospace",
                verticalalignment="top", color="#222")
        y_pos -= 0.035


# ==================== Main ====================
def visualize(args):
    trace = load_trace(args.trace)
    steps = trace.get("steps", [])
    if not steps:
        print("No steps found in trace.")
        return

    # Get task info
    tasks = trace.get("tasks", [])
    n_steps = len(steps)
    scene_bounds = get_scene_bounds(steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step-wise current task index and trajectory
    current_task = 0
    traj = []

    for si, step in enumerate(steps):
        # Track task transitions
        stask = step.get("task_index", current_task)
        current_task = stask

        pos = step.get("position")
        if pos:
            traj.append((pos[0], pos[2]))

        agent_pos = pos
        target_pos = step.get("target_position")
        if target_pos is not None:
            target_pos = (target_pos[0], target_pos[1], target_pos[2])
        else:
            target_pos = None

        # Create figure: 2 rows, 2 cols
        fig = plt.figure(figsize=(14, 8), facecolor="white")
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.4],
                              left=0.05, right=0.95, top=0.95, bottom=0.05,
                              hspace=0.3, wspace=0.3)

        ax_nav = fig.add_subplot(gs[0, 0])
        ax_struct = fig.add_subplot(gs[0, 1])
        ax_debug = fig.add_subplot(gs[1, :])

        # Draw navigation map with trajectory
        draw_navigation_map(ax_nav, step, agent_pos, target_pos, scene_bounds)

        # Draw trajectory overlay on navigation map
        if len(traj) > 1:
            tx, tz = zip(*traj)
            ax_nav.plot(tx, tz, color="#555", linewidth=0.8, alpha=0.4, linestyle="-")

        # Draw structure map
        draw_structure_map(ax_struct, step, agent_pos, scene_bounds)

        # Draw debug panel
        plot_task = step.get("task_index", current_task)
        draw_debug_panel(ax_debug, step, plot_task, n_steps)

        fig.suptitle(f"Step {si}/{n_steps} | Task {plot_task}",
                     fontsize=12, fontweight="bold", y=0.98)

        fname = out_dir / f"step_{si:06d}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if (si + 1) % 50 == 0:
            print(f"  Rendered {si + 1}/{n_steps}")

    print(f"Frames saved to {out_dir}")

    # Video
    if args.video:
        fps = args.fps
        import subprocess
        video_path = args.video
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-pattern_type", "glob", "-i", f"{out_dir}/step_*.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", str(video_path)
        ]
        print(f"Generating video: {video_path}")
        subprocess.run(cmd, check=True)
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ConfTopo trace visualizer")
    ap.add_argument("--trace", required=True, help="Path to trace JSON")
    ap.add_argument("--out-dir", default="viz_frames", help="Output directory for frames")
    ap.add_argument("--video", default=None, help="Output video path (optional)")
    ap.add_argument("--fps", type=int, default=4, help="Video FPS")
    args = ap.parse_args()
    visualize(args)
