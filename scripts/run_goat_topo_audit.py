from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GOAT exploration/topo trace capture, then generate audit visualization."
    )
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene", default="4ok3usBNeis")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="For loop mode, 0 means run until the loop completes")
    parser.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    parser.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--goal-modality", choices=["auto", "object", "instruction", "description", "image"], default="auto")
    parser.add_argument("--run-dir", default="data/logs/goat_topo/live_trace")
    parser.add_argument("--no-unique-run-dir", action="store_true", help="Write directly into --run-dir instead of creating a timestamped subdirectory")
    parser.add_argument("--trajectory-mode", choices=["agent", "loop"], default="agent")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--object-threshold", type=float, default=None)
    parser.add_argument("--room-threshold", type=float, default=None)
    parser.add_argument("--landmark-threshold", type=float, default=None)
    parser.add_argument("--use-placeholder-embed", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--topdown-resolution", type=float, default=0.05)
    parser.add_argument("--navmesh-path", default=None)
    parser.add_argument("--no-gt-topdown", action="store_true")
    parser.add_argument("--show-planned-route", action="store_true")
    parser.add_argument("--loop-anchors", type=int, default=3)
    parser.add_argument("--loop-samples", type=int, default=80)
    parser.add_argument("--loop-min-start-dist", type=float, default=4.0)
    parser.add_argument("--loop-min-extent", type=float, default=4.0)
    parser.add_argument("--loop-max-geodesic-length", type=float, default=18.0)
    parser.add_argument("--loop-floor-y-tolerance", type=float, default=0.35)
    parser.add_argument("--loop-reach-radius", type=float, default=0.45)
    parser.add_argument("--loop-return-radius", type=float, default=0.75)
    parser.add_argument("--instruction", default="Walk one loop around the room on the same floor and return to the start.")
    args = parser.parse_args()

    run_root = Path(args.run_dir)
    run_dir = run_root if args.no_unique_run_dir else run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = run_dir / "topo_trace_semantic.json"
    frame_dir = run_dir / "rgb_frames"
    viz_dir = run_dir / "viz_audit"

    trace_cmd = [
        sys.executable,
        "scripts/run_goat_loop_topo_trace.py" if args.trajectory_mode == "loop" else "scripts/run_goat_topo_trace.py",
        "--split",
        args.split,
        "--scene",
        args.scene,
        "--episode-index",
        str(args.episode_index),
        "--max-steps",
        str(args.max_steps),
        "--dataset-dir",
        args.dataset_dir,
        "--scene-root",
        args.scene_root,
        "--goal-graph-dir",
        args.goal_graph_dir,
        "--goal-modality",
        args.goal_modality,
        "--output",
        str(trace_path),
        "--frame-dir",
        str(frame_dir),
        "--clip-model",
        args.clip_model,
        "--clip-device",
        args.clip_device,
    ]
    for flag, value in [
        ("--object-threshold", args.object_threshold),
        ("--room-threshold", args.room_threshold),
        ("--landmark-threshold", args.landmark_threshold),
    ]:
        if value is not None:
            trace_cmd.extend([flag, str(value)])
    if args.use_placeholder_embed:
        trace_cmd.append("--use-placeholder-embed")
    if args.trajectory_mode == "loop":
        trace_cmd.extend([
            "--loop-anchors",
            str(args.loop_anchors),
            "--loop-samples",
            str(args.loop_samples),
            "--loop-min-start-dist",
            str(args.loop_min_start_dist),
            "--loop-min-extent",
            str(args.loop_min_extent),
            "--loop-max-geodesic-length",
            str(args.loop_max_geodesic_length),
            "--loop-floor-y-tolerance",
            str(args.loop_floor_y_tolerance),
            "--loop-reach-radius",
            str(args.loop_reach_radius),
            "--loop-return-radius",
            str(args.loop_return_radius),
            "--instruction",
            args.instruction,
        ])

    viz_cmd = [
        sys.executable,
        "scripts/visualize_goat_topo_trace.py",
        "--trace",
        str(trace_path),
        "--out-dir",
        str(viz_dir),
        "--stride",
        str(args.stride),
        "--fps",
        str(args.fps),
        "--topdown-resolution",
        str(args.topdown_resolution),
    ]
    if args.navmesh_path:
        viz_cmd.extend(["--navmesh-path", args.navmesh_path])
    if args.no_gt_topdown:
        viz_cmd.append("--no-gt-topdown")
    if args.show_planned_route:
        viz_cmd.append("--show-planned-route")

    run(trace_cmd)
    run(viz_cmd)

    print(
        "\nDone.\n"
        f"Trace: {ROOT / trace_path}\n"
        f"RGB frames: {ROOT / frame_dir}\n"
        f"Audit video: {ROOT / viz_dir / 'goat_semantic_audit_view.mp4'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
