"""Lightweight visualizer for run_goat_agent1_smoke.py traces.

Renders a 2D trajectory (x-z) colored by goal plus a debug panel
(nav_phase / VLM / approach telemetry). No RGB or per-step topo required.

Example:
    python scripts/visualize_agent1_smoke_trace.py \\
        --trace data/logs/goat_topo/agent1_smoke_4ok3usBNeis_ep0_v4/trace.json \\
        --out-dir data/logs/goat_topo/agent1_smoke_4ok3usBNeis_ep0_v4/viz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trace_goal_markers import draw_goal_position_markers, extend_xz_limits, load_goal_instances

ROOT = Path(__file__).resolve().parents[1]

GOAL_PALETTE = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
PHASE_COLORS = {
    "SEARCH": "#94a3b8",
    "ROUTE_TO_ANCHOR": "#3b82f6",
    "SCAN_TRACK": "#f59e0b",
    "VISUAL_APPROACH": "#22c55e",
    "VERIFY_STOP": "#a855f7",
    "STOP": "#ef4444",
    "RECOVER": "#f97316",
}


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute() and path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return path


def step_xy(step: dict[str, Any], *, use_world: bool) -> np.ndarray:
    key = "world_position" if use_world else "position"
    return np.asarray(step.get(key) or step.get("position") or [0.0, 0.0, 0.0], dtype=np.float32)


def goal_colors(trace: dict[str, Any]) -> dict[str, str]:
    goals = trace.get("goals") or []
    if not goals:
        seen: list[str] = []
        for step in trace.get("steps", []):
            g = step.get("goal")
            if g and g not in seen:
                seen.append(str(g))
        goals = seen
    return {str(g): GOAL_PALETTE[i % len(GOAL_PALETTE)] for i, g in enumerate(goals)}


def collect_limits(steps: list[dict[str, Any]], trace: dict[str, Any] | None = None, *, use_world: bool, pad: float = 0.6):
    if not steps:
        return (-1.0, 1.0), (-1.0, 1.0)
    xs, zs = [], []
    for step in steps:
        p = step_xy(step, use_world=use_world)
        xs.append(float(p[0]))
        zs.append(float(p[2]))
        tgt = step.get("target_position")
        if tgt is not None:
            t = np.asarray(tgt, dtype=np.float32)
            xs.append(float(t[0]))
            zs.append(float(t[2]))
    if trace is not None:
        extend_xz_limits(
            xs,
            zs,
            trace,
            use_world=use_world,
            origin_world=trace.get("origin_world"),
            position_frame=trace.get("coordinate_frame", "episode_start_relative"),
        )
    xmin, xmax = min(xs) - pad, max(xs) + pad
    zmin, zmax = min(zs) - pad, max(zs) + pad
    if xmax - xmin < 1.0:
        cx = 0.5 * (xmin + xmax)
        xmin, xmax = cx - 0.5, cx + 0.5
    if zmax - zmin < 1.0:
        cz = 0.5 * (zmin + zmax)
        zmin, zmax = cz - 0.5, cz + 0.5
    return (xmin, xmax), (zmin, zmax)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if len(value) <= 4 and all(isinstance(x, (int, float)) for x in value):
            return "[" + ", ".join(f"{float(x):.3f}" for x in value) + "]"
        return str(value)[:80]
    text = str(value)
    return text if len(text) <= 72 else text[:69] + "..."


def debug_lines(trace: dict[str, Any], step: dict[str, Any]) -> list[str]:
    mem = step.get("memory") or {}
    lines = [
        "Agent1 Smoke Trace",
        f"scene: {trace.get('scene', 'n/a')}  ep: {trace.get('episode_id', 'n/a')}",
        f"step: {step.get('global_step', '?')}  local: {step.get('local_step', '?')}",
        f"goal [{step.get('goal_index', '?')}]: {step.get('goal', 'n/a')}",
        "",
        "Navigation",
        f"nav_phase: {step.get('nav_phase', 'n/a')}",
        f"phase_reason: {_fmt(step.get('phase_reason'))}",
        f"agent_action: {_fmt(step.get('agent_action'))}",
        f"low_action: {_fmt(step.get('low_action'))}",
        f"mode: {_fmt(step.get('mode'))}",
        f"target: {_fmt(step.get('target_node_id'))} ({_fmt(step.get('target_type'))})",
        f"anchor: {_fmt(step.get('active_anchor_id'))}",
        f"anchor_dist: {_fmt(step.get('anchor_distance'))}",
        f"goal_min_dist: {_fmt(step.get('goal_min_distance'))}",
        f"dist_to_target: {_fmt(step.get('distance_to_target'))}",
        "",
        "Proposal",
        f"type: {_fmt(step.get('proposal_type'))}",
        f"source: {_fmt(step.get('proposal_source'))}",
        f"score: {_fmt(step.get('proposal_score'))}",
        "",
        "Stop gate",
        f"stop_reason: {_fmt(step.get('stop_reason'))}",
        f"goal_visible: {_fmt(step.get('stop_goal_visible'))}",
        f"need_scan: {_fmt(step.get('stop_need_scan'))}",
        f"need_approach: {_fmt(step.get('stop_need_approach'))}",
        f"centered/close: {_fmt(step.get('stop_centered'))} / {_fmt(step.get('stop_close'))}",
        f"bbox_area: {_fmt(step.get('stop_bbox_area'))}",
        f"verified_stop: {_fmt(step.get('verified_stop'))}",
        f"scan_no_confirm: {_fmt(step.get('anchor_scan_no_confirm'))}",
        "",
        "Approach",
        f"steps: {_fmt(step.get('approach_steps'))}",
        f"forward_count: {_fmt(step.get('approach_forward_count'))}",
        f"travel_m: {_fmt(step.get('approach_travel_distance'))}",
        "",
        "VLM",
        f"fresh: {_fmt(step.get('vlm_fresh'))}",
        f"trigger: {_fmt(step.get('vlm_trigger_reason'))}",
        f"mode: {_fmt(step.get('vlm_mode'))}",
        f"goal_visible: {_fmt(step.get('vlm_goal_visible'))}",
        f"stop_candidate: {_fmt(step.get('vlm_stop_candidate'))}",
        f"best_label: {_fmt(step.get('vlm_best_label'))}",
        f"range/vis: {_fmt(step.get('vlm_range_bin'))} / {_fmt(step.get('vlm_visibility'))}",
        f"bbox: {_fmt(step.get('vlm_best_bbox'))}",
        "",
        "Memory",
        f"nodes: {mem.get('total_nodes', 0)}  visited: {mem.get('visited_waypoints', 0)}",
        f"frontiers: {mem.get('frontiers', 0)}  objects: {mem.get('objects', 0)}",
    ]
    summary = trace.get("smoke_summary")
    if summary and step.get("global_step") == len(trace.get("steps", [])) - 1:
        lines.extend([
            "",
            "Smoke summary",
            f"pipeline_ok: {_fmt(summary.get('pipeline_ok'))}",
            f"saw_visual_approach: {_fmt(summary.get('saw_visual_approach'))}",
            f"saw_verified_stop: {_fmt(summary.get('saw_verified_stop'))}",
            f"max_approach_fwd: {_fmt(summary.get('max_approach_forward_count'))}",
            f"goals_success: {_fmt(summary.get('goals_success'))}",
        ])
    return lines


def draw_colored_path(ax, steps: list[dict[str, Any]], end_idx: int, colors: dict[str, str], *, use_world: bool):
    if end_idx < 1:
        return
    for i in range(1, end_idx + 1):
        p0 = step_xy(steps[i - 1], use_world=use_world)
        p1 = step_xy(steps[i], use_world=use_world)
        color = colors.get(str(steps[i].get("goal", "")), "#6b7280")
        ax.plot([p0[0], p1[0]], [p0[2], p1[2]], color=color, linewidth=1.8, alpha=0.9, zorder=2)


def draw_goal_boundaries(ax, steps: list[dict[str, Any]], end_idx: int, *, use_world: bool):
    prev_goal = None
    for i in range(min(end_idx, len(steps) - 1) + 1):
        goal = steps[i].get("goal")
        if prev_goal is not None and goal != prev_goal:
            p = step_xy(steps[i], use_world=use_world)
            ax.axvline(p[0], color="#cbd5e1", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
            ax.axhline(p[2], color="#cbd5e1", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
        prev_goal = goal


def draw_frame(
    trace: dict[str, Any],
    idx: int,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    colors: dict[str, str],
    *,
    use_world: bool,
    out_path: Path | None = None,
) -> np.ndarray | None:
    steps = trace["steps"]
    step = steps[idx]
    fig, (ax, info) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=130, gridspec_kw={"width_ratios": [1.35, 1]})

    draw_colored_path(ax, steps, idx, colors, use_world=use_world)
    draw_goal_boundaries(ax, steps, idx, use_world=use_world)
    draw_goal_position_markers(
        ax,
        trace,
        step,
        use_world=use_world,
        origin_world=trace.get("origin_world"),
        position_frame=trace.get("coordinate_frame", "episode_start_relative"),
    )

    for goal, color in colors.items():
        ax.plot([], [], color=color, linewidth=2.0, label=goal)

    cur = step_xy(step, use_world=use_world)
    phase = str(step.get("nav_phase") or "")
    ax.scatter(
        [cur[0]], [cur[2]],
        marker="*",
        s=280,
        color=PHASE_COLORS.get(phase, "#facc15"),
        edgecolors="black",
        linewidths=0.9,
        zorder=5,
        label=f"agent ({phase or '?'})",
    )

    tgt = step.get("target_position")
    if tgt is not None:
        t = np.asarray(tgt, dtype=np.float32)
        ax.scatter([t[0]], [t[2]], marker="X", s=120, color="#ef4444", edgecolors="black", linewidths=0.7, zorder=4)
        ax.plot([cur[0], t[0]], [cur[2], t[2]], color="#ef4444", linestyle=":", linewidth=1.0, alpha=0.65, zorder=1)

    if step.get("verified_stop"):
        ax.scatter([cur[0]], [cur[2]], marker="s", s=90, facecolors="none", edgecolors="#ef4444", linewidths=2.0, zorder=6)
    if step.get("vlm_fresh"):
        ax.scatter([cur[0] + 0.05], [cur[2] + 0.05], marker="o", s=40, color="#06b6d4", alpha=0.85, zorder=6)

    coord_label = "world x-z" if use_world else "episode-relative x-z"
    ax.set_title(f"Trajectory | global_step {step.get('global_step', idx)} | {coord_label}")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=7)

    info.axis("off")
    info.text(0.02, 0.98, "\n".join(debug_lines(trace, step)), va="top", ha="left", fontsize=8.5, family="monospace")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return None

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize agent1 smoke trace (path + debug panel)")
    ap.add_argument("--trace", required=True, help="Path to trace.json from run_goat_agent1_smoke.py")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <trace_dir>/viz)")
    ap.add_argument("--stride", type=int, default=2, help="Frame stride for video")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--use-world", action="store_true", help="Plot world_position instead of episode-relative position")
    args = ap.parse_args()

    trace_path = resolve_path(args.trace)
    trace = json.loads(trace_path.read_text())
    load_goal_instances(trace)
    steps = trace.get("steps") or []
    if not steps:
        raise SystemExit(f"No steps in trace: {trace_path}")

    out_dir = Path(args.out_dir) if args.out_dir else trace_path.parent / "viz"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = goal_colors(trace)
    xlim, zlim = collect_limits(steps, trace, use_world=args.use_world)
    stride = max(1, args.stride)

    final_png = out_dir / "trajectory_final.png"
    draw_frame(trace, len(steps) - 1, xlim, zlim, colors, use_world=args.use_world, out_path=final_png)

    frames: list[np.ndarray] = []
    for i in range(0, len(steps), stride):
        frame = draw_frame(trace, i, xlim, zlim, colors, use_world=args.use_world)
        if frame is not None:
            frames.append(frame)

    if len(steps) - 1 not in range(0, len(steps), stride):
        last = draw_frame(trace, len(steps) - 1, xlim, zlim, colors, use_world=args.use_world)
        if last is not None:
            frames.append(last)

    video_path = out_dir / "agent1_smoke_audit.mp4"
    imageio.mimsave(video_path, frames, fps=args.fps)

    summary = {
        "trace": str(trace_path),
        "steps": len(steps),
        "stride": stride,
        "fps": args.fps,
        "use_world": args.use_world,
        "goals": trace.get("goals", []),
        "goal_colors": colors,
        "final_png": str(final_png),
        "video": str(video_path),
        "smoke_summary": trace.get("smoke_summary", {}),
        "task_summaries": trace.get("task_summaries", []),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
