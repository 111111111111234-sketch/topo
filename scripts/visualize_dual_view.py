#!/usr/bin/env python3
"""Three-panel navigation visualization with GT topdown backgrounds.

Layout:
  Left panel:  first-person RGB
  Right top:   two-layer map (navigation + structure), GT topdown background
  Right bottom: spatial structure skeleton (rooms + portals), GT topdown background
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftopo.viz.memory_trace_viz import (
    build_spatial_skeleton,
    compute_trace_limits,
    draw_gt_topdown_relative,
    draw_memory_panel,
    filter_topo_nodes,
    load_gt_topdown,
    load_origin_world,
    memory_panel_title,
)


def resolve_rgb_path(trace: dict, st: dict) -> Optional[str]:
    rgb_frame = st.get("rgb_frame")
    if rgb_frame is None:
        return None
    candidates = [Path(rgb_frame), ROOT / rgb_frame]
    frame_dir = trace.get("frame_dir")
    if frame_dir:
        step_idx = st.get("step", 0)
        candidates.append(ROOT / frame_dir / "rgb_{:04d}.png".format(step_idx))
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--topdown-resolution", type=float, default=0.05)
    parser.add_argument("--navmesh-path", default=None)
    parser.add_argument("--no-gt-topdown", action="store_true")
    parser.add_argument("--rgb-ratio", type=float, default=0.25,
                        help="Fraction of total width for left RGB panel (0-1)")
    parser.add_argument("--show-object-uncertainty", action="store_true")
    args = parser.parse_args()

    trace_path = ROOT / args.trace
    with open(trace_path) as f:
        trace = json.load(f)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    gt_ctx = load_gt_topdown(
        trace,
        resolution=args.topdown_resolution,
        navmesh_override=args.navmesh_path,
        disabled=args.no_gt_topdown,
    )
    origin_world, origin_source = load_origin_world(trace)

    xlim, zlim = compute_trace_limits(
        trace, margin=1.5, mode="trajectory_plus_near", max_node_distance=12.0,
    )

    steps = trace.get("steps", [])
    n_steps = len(steps)

    ratio = max(0.1, min(0.5, float(args.rgb_ratio)))
    n_cols = 100
    rgb_cols = max(10, int(round(n_cols * ratio)))
    topo_cols = n_cols - rgb_cols

    fig_width = 14.0 + 4.0 * (1.0 - ratio)
    fig = plt.figure(figsize=(fig_width, 7.5), dpi=args.dpi)
    ax_rgb   = plt.subplot2grid((2, n_cols), (0, 0), rowspan=2, colspan=rgb_cols, fig=fig)
    ax_two   = plt.subplot2grid((2, n_cols), (0, rgb_cols), rowspan=1, colspan=topo_cols, fig=fig)
    ax_struct = plt.subplot2grid((2, n_cols), (1, rgb_cols), rowspan=1, colspan=topo_cols, fig=fig)

    output_frames = []

    fig.tight_layout(pad=1.5)

    for idx in range(0, n_steps, args.stride):
        st = steps[idx]
        step_num = st.get("step", idx)

        for ax in [ax_rgb, ax_two, ax_struct]:
            ax.clear()

        # ---- Left: RGB ----
        rgb_path = resolve_rgb_path(trace, st)
        if rgb_path is not None and Path(rgb_path).is_file():
            rgb_img = imageio.imread(rgb_path)
            ax_rgb.imshow(rgb_img)
        else:
            ax_rgb.text(0.5, 0.5, "(no RGB frame)", ha="center", va="center",
                        fontsize=10, transform=ax_rgb.transAxes, color="#94a3b8")
        ax_rgb.set_title("Step {}".format(step_num), fontsize=7, fontweight="bold", pad=2)
        ax_rgb.axis("off")

        topo_data = st.get("topo", {})
        current_goal = trace.get("current_goal", {})
        goal_text = ""
        if current_goal:
            target = current_goal.get("target_object", "")
            if target:
                goal_text = "Goal: {}".format(target)
        heavy_debug = st.get("memory", {}).get("last_heavy", {})
        heavy_reason = ""
        if heavy_debug and heavy_debug.get("ran"):
            heavy_reason = "{} det={}".format(heavy_debug.get("reason", ""), heavy_debug.get("detections", 0))

        nodes = list(topo_data.get("nodes", []))
        edges = list(topo_data.get("edges", []))

        if not nodes:
            out_path = frame_dir / "frame_{:04d}.png".format(step_num)
            fig.savefig(out_path, dpi=args.dpi)
            output_frames.append(str(out_path))
            continue

        agent_pos = np.asarray(st.get("position", [0, 0, 0]), dtype=np.float32)

        # ---- Right top: Two-layer ----
        two_nodes = filter_topo_nodes(nodes, "two_layer", agent_pos=agent_pos)
        draw_memory_panel(
            ax_two, two_nodes, edges, agent_pos,
            title=memory_panel_title(step_num, two_nodes, goal_text, heavy_reason),
            xlim=xlim, zlim=zlim, show_labels=True, max_labels=10,
            gt_ctx=gt_ctx, origin_world=origin_world,
            show_legend=False, axis_label="rel", panel_mode="two_layer",
            show_object_uncertainty=args.show_object_uncertainty,
        )

        # ---- Right bottom: Structure skeleton ----
        sk_nodes, sk_edges = build_spatial_skeleton(nodes, edges)
        draw_memory_panel(
            ax_struct, sk_nodes, sk_edges, agent_pos,
            title="Spatial Structure (rooms + portals)",
            xlim=xlim, zlim=zlim, show_labels=True, max_labels=8,
            gt_ctx=gt_ctx, origin_world=origin_world,
            show_legend=True, axis_label="rel", panel_mode="spatial_structure",
            show_object_uncertainty=False,
        )

        out_path = frame_dir / "frame_{:04d}.png".format(step_num)
        fig.savefig(out_path, dpi=args.dpi)
        output_frames.append(str(out_path))
        print("Step {:3d}/{:3d} -> {}".format(step_num, n_steps, out_path))

    plt.close(fig)

    if output_frames:
        video_path = out_dir / "dual_view.mp4"
        writer = imageio.get_writer(str(video_path), fps=args.fps, format="ffmpeg")
        for fp in output_frames:
            writer.append_data(imageio.imread(fp))
        writer.close()
        print("Video: {}".format(video_path))

        gif_path = out_dir / "dual_view.gif"
        frames = [imageio.imread(p) for p in output_frames]
        imageio.mimsave(str(gif_path), frames, fps=args.fps, loop=0)
        print("GIF: {}".format(gif_path))

    summary = {
        "trace": args.trace,
        "out_dir": str(out_dir),
        "steps": n_steps,
        "stride": args.stride,
        "frames": len(output_frames),
        "video": str(video_path) if output_frames else None,
        "gif": str(gif_path) if output_frames else None,
        "gt_topdown": {
            "available": gt_ctx.get("available", False),
            "reason": gt_ctx.get("reason", ""),
        },
        "origin_world": origin_world.tolist() if origin_world is not None else None,
        "origin_source": origin_source,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary: {}".format(summary_path))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()