"""Phase3 Dynamic Memory Visualization.

Layout (default):
  Left: first-person RGB
  Right-top: GT + two-layer map (base: room/lm, overlay: waypoints, near objects)
  Right-bottom: spatial structure base layer (room↔landmark, no waypoints)

Optional --secondary-panel adds a small inset for summary/near/object views.

Usage:
    python scripts/visualize_phase3_memory_trace.py \
        --trace data/logs/goat_topo/phase3_dynamic_memory_trace_v2.json \
        --out-dir data/logs/goat_topo/phase3_memory_viz_v2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftopo.viz.memory_trace_viz import (
    build_spatial_skeleton,
    compute_trace_limits,
    draw_memory_panel,
    filter_topo_nodes,
    is_room_summary_node,
    load_gt_topdown,
    load_origin_world,
    make_geo_validator,
    memory_panel_title,
)

ROOT = Path(__file__).resolve().parents[1]

SECONDARY_PANELS = {
    "summary_only": "Summary-only (region)",
    "near_only": "Near radius",
    "object_only": "Object-only (near)",
}


def resolve_rgb_path(trace: dict, st: dict) -> Optional[str]:
    rgb_frame = st.get("rgb_frame")
    if rgb_frame is None:
        return None
    if isinstance(rgb_frame, str):
        candidates = [
            Path(rgb_frame),
            ROOT / rgb_frame,
        ]
        if trace.get("frame_dir"):
            step = st.get("step", 0)
            candidates.append(ROOT / trace["frame_dir"] / f"rgb_{step:04d}.png")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return None


def _filter_edges_for_nodes(edges: list, node_ids: set) -> list:
    return [
        e for e in edges
        if e.get("source") in node_ids and e.get("target") in node_ids
    ]


def _smooth_skeleton_positions(
    nodes: list,
    cache: dict,
    alpha: float = 0.35,
) -> None:
    """EMA smooth skeleton node positions across animation frames."""
    for node in nodes:
        key = str(node.get("id") or node.get("label") or "")
        if not key:
            continue
        pos = np.asarray(node["position"], dtype=np.float32)
        if key not in cache:
            cache[key] = pos.copy()
        elif float(np.linalg.norm(pos[[0, 2]] - cache[key][[0, 2]])) > 5.0:
            cache[key] = pos.copy()
        else:
            cache[key] = (1.0 - alpha) * cache[key] + alpha * pos
        node["position"] = cache[key].tolist()


def draw_frame(
    trace: dict,
    idx: int,
    xlim: tuple,
    zlim: tuple,
    near_radius: float,
    gt_ctx: dict,
    origin_world: Optional[np.ndarray],
    secondary_panel: Optional[str] = None,
    legacy_viz: bool = False,
    show_env_landmarks: bool = False,
    geo_validator=None,
    anchor_cache: Optional[dict] = None,
) -> np.ndarray:
    """Render a single frame as RGBA numpy array."""
    st = trace["steps"][idx]
    pos = np.asarray(st.get("position", [0, 0, 0]), dtype=np.float32)
    topo = st.get("topo", {})
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    goal = trace.get("current_goal", {})
    goal_text = f"goal: {goal.get('target_object', 'n/a')}" if goal else ""
    heavy_reason = ""
    mem = st.get("memory", {})
    if isinstance(mem, dict):
        lh = mem.get("last_heavy", {})
        if isinstance(lh, dict) and lh.get("ran"):
            heavy_reason = lh.get("reason", "")

    has_secondary = secondary_panel in SECONDARY_PANELS
    fig_w = 15.0 if not has_secondary else 18.0
    fig = plt.figure(figsize=(fig_w, 8.5), dpi=110)
    if has_secondary:
        gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 0.72], height_ratios=[1.0, 1.0])
        rgb_ax = fig.add_subplot(gs[:, 0])
        gt_ax = fig.add_subplot(gs[0, 1])
        topo_ax = fig.add_subplot(gs[1, 1])
        sec_ax = fig.add_subplot(gs[:, 2])
    else:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.0, 1.0])
        rgb_ax = fig.add_subplot(gs[:, 0])
        gt_ax = fig.add_subplot(gs[0, 1])
        topo_ax = fig.add_subplot(gs[1, 1])
        sec_ax = None

    rgb_path = resolve_rgb_path(trace, st)
    if rgb_path:
        from PIL import Image
        rgb_ax.imshow(np.asarray(Image.open(rgb_path)))
    else:
        rgb_ax.text(0.5, 0.5, "No RGB\n(add --frame-dir when running trace)", ha="center", va="center", fontsize=8)
    rgb_ax.axis("off")
    rgb_ax.set_title(f"First person | step {st.get('step', idx)}", fontsize=9)

    far_radius = float(trace.get("config", {}).get("memory", {}).get("far_radius", 10.0))
    mem_cfg = trace.get("config", {}).get("memory", {})
    structure_edges = edges
    if legacy_viz:
        gt_nodes = filter_topo_nodes(nodes, "full", agent_pos=pos, near_radius=near_radius)
        memory_nodes = gt_nodes
        gt_title = "GT floorplan + TopoMap (visible)"
        memory_title = "Dynamic TopoMap (visible memory)"
        show_uncertainty = False
    else:
        base_layer = dict(
            agent_pos=pos, near_radius=near_radius, far_radius=far_radius,
        )
        # GT: relaxed geo filter (bbox objects may be ~1-2m off navmesh).
        gt_nodes = filter_topo_nodes(
            nodes, "two_layer", geo_validator=geo_validator, **base_layer,
        )
        memory_nodes, structure_edges = build_spatial_skeleton(
            nodes,
            edges,
            summary_radius=float(mem_cfg.get("summary_radius", 5.0)),
            room_link_max_distance=float(mem_cfg.get("room_link_max_distance", 12.0)),
            merge_radius=float(mem_cfg.get("merge_radius", 1.0)),
            origin_world=origin_world,
            scene_file=trace.get("scene_file"),
        )
        skeleton_by_id = {n["id"]: n for n in memory_nodes}
        for gt_node in gt_nodes:
            sk_node = skeleton_by_id.get(gt_node["id"])
            if sk_node is None or not is_room_summary_node(sk_node):
                continue
            gt_node["position"] = sk_node["position"]
            sk_attrs = sk_node.get("attributes") or {}
            if sk_attrs.get("contains_labels"):
                gt_node.setdefault("attributes", {})["contains_labels"] = sk_attrs["contains_labels"]
        gt_nodes.extend(
            n for n in memory_nodes
            if (n.get("attributes") or {}).get("synthetic_portal")
        )
        if anchor_cache is not None:
            _smooth_skeleton_positions(memory_nodes, anchor_cache)
            for gt_node in gt_nodes:
                sk_node = next((n for n in memory_nodes if n["id"] == gt_node["id"]), None)
                if sk_node is not None and is_room_summary_node(sk_node):
                    gt_node["position"] = sk_node["position"]
        gt_title = "GT | base(room/lm) + waypoint overlay + near obj"
        memory_title = "Traversable skeleton | walked room↔portal↔room"
        show_uncertainty = True

    title_text = memory_panel_title(
        step=st.get("step", idx),
        nodes=nodes,
        goal_text=goal_text,
        heavy_reason=heavy_reason,
        gt_nodes=gt_nodes,
        memory_nodes=memory_nodes,
    )
    fig.suptitle(title_text, fontsize=8, y=0.98)

    gt_ids = {n["id"] for n in gt_nodes}
    gt_edges = _filter_edges_for_nodes(edges, gt_ids)
    if not legacy_viz:
        gt_edges = gt_edges + structure_edges

    draw_memory_panel(
        gt_ax,
        gt_nodes,
        gt_edges,
        pos,
        title=gt_title,
        xlim=xlim,
        zlim=zlim,
        show_labels=True,
        max_labels=10,
        gt_ctx=gt_ctx,
        origin_world=origin_world,
        show_legend=True,
        panel_mode="two_layer",
        show_object_uncertainty=show_uncertainty,
    )

    draw_memory_panel(
        topo_ax,
        memory_nodes,
        structure_edges if not legacy_viz else _filter_edges_for_nodes(edges, {n["id"] for n in memory_nodes}),
        pos,
        title=memory_title,
        xlim=xlim,
        zlim=zlim,
        show_labels=True,
        max_labels=12,
        gt_ctx=gt_ctx,
        origin_world=origin_world,
        show_legend=True,
        panel_mode="spatial_structure" if not legacy_viz else "two_layer",
        show_object_uncertainty=False,
    )

    if sec_ax is not None and secondary_panel is not None:
        sec_nodes = filter_topo_nodes(nodes, secondary_panel, agent_pos=pos, near_radius=near_radius)
        sec_edges = _filter_edges_for_nodes(edges, {n["id"] for n in sec_nodes})
        draw_memory_panel(
            sec_ax,
            sec_nodes,
            sec_edges,
            pos,
            title=SECONDARY_PANELS[secondary_panel],
            xlim=xlim,
            zlim=zlim,
            show_labels=True,
            max_labels=8,
            show_legend=True,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Phase3 Dynamic Memory Visualization")
    parser.add_argument("--trace", default="data/logs/goat_topo/phase3_dynamic_memory_trace.json")
    parser.add_argument("--out-dir", default="data/logs/goat_topo/phase3_memory_viz")
    parser.add_argument("--near-radius", type=float, default=3.0,
                        help="Radius (m) for near_only secondary panel")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--step-range", type=str, default=None,
                        help="e.g. '0:50' to only render steps 0-49")
    parser.add_argument("--topdown-resolution", type=float, default=0.05)
    parser.add_argument("--navmesh-path", default=None)
    parser.add_argument("--no-gt-topdown", action="store_true")
    parser.add_argument(
        "--secondary-panel",
        choices=["none", "summary_only", "near_only", "object_only"],
        default="none",
        help="Optional small inset panel; default layout keeps only Overview (full)",
    )
    parser.add_argument(
        "--legacy-viz",
        action="store_true",
        help="Use old filters (show all visible nodes including env landmarks on GT)",
    )
    parser.add_argument(
        "--show-env-landmarks",
        action="store_true",
        help="Also draw environment CLIP view-tags (faint purple) on the memory panel",
    )
    parser.add_argument(
        "--limits-mode",
        choices=["trajectory", "trajectory_plus_near", "all"],
        default="trajectory",
        help="Axis limits: trajectory only (default), +nearby nodes, or all nodes (legacy)",
    )
    parser.add_argument("--limits-margin", type=float, default=2.0)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.is_absolute():
        trace_path = ROOT / trace_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"Loading trace: {trace_path}")
    with open(trace_path) as f:
        trace = json.load(f)

    origin_world, origin_source = load_origin_world(trace)
    print(f"  origin_world: {origin_source}")
    gt_ctx = load_gt_topdown(
        trace,
        args.topdown_resolution,
        args.navmesh_path,
        disabled=args.no_gt_topdown,
    )
    print(f"  gt_topdown: {gt_ctx.get('reason') if not gt_ctx.get('available') else gt_ctx.get('navmesh_path')}")

    steps = trace.get("steps", [])
    if args.step_range:
        parts = args.step_range.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(steps)
        step_indices = list(range(start, min(end, len(steps))))
    else:
        step_indices = list(range(len(steps)))

    secondary = None if args.secondary_panel == "none" else args.secondary_panel
    print(f"  {len(steps)} total steps, rendering {len(step_indices)}")
    viz_mode = "legacy" if args.legacy_viz else "geo_aligned"
    print(f"  layout: RGB | GT+Topo / Dynamic Topo | mode={viz_mode} | secondary={secondary or 'off'}")
    xlim, zlim = compute_trace_limits(
        trace, margin=args.limits_margin, mode=args.limits_mode,
    )
    print(f"  limits_mode={args.limits_mode} xlim={xlim}, zlim={zlim}")
    geo_validator = make_geo_validator(gt_ctx, origin_world)

    frame_paths = []
    anchor_cache: dict = {}
    for i, idx in enumerate(step_indices):
        frame = draw_frame(
            trace, idx, xlim, zlim, args.near_radius,
            gt_ctx, origin_world, secondary_panel=secondary,
            legacy_viz=args.legacy_viz,
            show_env_landmarks=args.show_env_landmarks,
            geo_validator=geo_validator,
            anchor_cache=anchor_cache,
        )
        frame_path = frames_dir / f"memory_{i:04d}.png"
        imageio.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        if (i + 1) % 20 == 0 or i == len(step_indices) - 1:
            print(f"  rendered {i + 1}/{len(step_indices)}")

    final_png = out_dir / "phase3_memory_final.png"
    if frame_paths:
        shutil.copy2(frame_paths[-1], str(final_png))

    video_path = out_dir / "phase3_memory.mp4"
    writer = imageio.get_writer(
        str(video_path), fps=args.fps, codec="libx264", pixelformat="yuv420p", quality=8,
    )
    for fp in frame_paths:
        writer.append_data(imageio.imread(fp))
    writer.close()
    print(f"  Video: {video_path}")

    if not args.no_gif:
        gif_path = out_dir / "phase3_memory.gif"
        frames_for_gif = [imageio.imread(fp) for fp in frame_paths]
        imageio.mimsave(str(gif_path), frames_for_gif, duration=1000 // max(1, args.fps), loop=0)
        print(f"  GIF: {gif_path}")

    summary = {
        "trace": str(trace_path),
        "steps_rendered": len(step_indices),
        "near_radius": args.near_radius,
        "secondary_panel": secondary,
        "viz_mode": viz_mode,
        "show_env_landmarks": args.show_env_landmarks,
        "limits_mode": args.limits_mode,
        "limits_margin": args.limits_margin,
        "gt_topdown_available": bool(gt_ctx.get("available")),
        "gt_topdown_reason": gt_ctx.get("reason", ""),
        "origin_source": origin_source,
        "video": str(video_path),
        "final_png": str(final_png),
        "frames_dir": str(frames_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
