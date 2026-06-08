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
        description="Regenerate a GOAT topo trace, render the normal audit view, and compare world vs relative topo coordinates."
    )
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene", default="4ok3usBNeis")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until the agent stops; loop mode runs until completion")
    parser.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    parser.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--goal-modality", choices=["auto", "object", "instruction", "description", "image"], default="auto")
    parser.add_argument("--run-dir", default="data/logs/goat_topo/coord_compare_audit")
    parser.add_argument("--no-unique-run-dir", action="store_true")
    parser.add_argument("--trajectory-mode", choices=["agent", "loop"], default="agent")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--object-threshold", type=float, default=None)
    parser.add_argument("--room-threshold", type=float, default=None)
    parser.add_argument("--landmark-threshold", type=float, default=None)
    parser.add_argument(
        "--env-landmarks",
        default="default",
        help="Comma-separated scene landmark labels, 'default' for built-ins, or 'none' to disable.",
    )
    parser.add_argument("--use-placeholder-embed", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--topdown-resolution", type=float, default=0.05)
    parser.add_argument("--navmesh-path", default=None)
    parser.add_argument("--no-gt-topdown", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--translation-noise-std", type=float, default=0.0)
    parser.add_argument("--heading-noise-std-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-correction", choices=["none", "gt_loop_closure", "observation_loop_closure"], default="gt_loop_closure")
    parser.add_argument("--loop-closure-radius", type=float, default=0.75)
    parser.add_argument("--loop-closure-min-interval", type=int, default=20)
    parser.add_argument("--loop-closure-min-drift", type=float, default=0.25)
    parser.add_argument("--observation-loop-sim-threshold", type=float, default=0.97)
    parser.add_argument("--observation-loop-max-pose-gap", type=float, default=2.0)
    parser.add_argument("--observation-loop-cooldown", type=int, default=30)
    parser.add_argument("--observation-loop-min-sim-margin", type=float, default=0.02)
    parser.add_argument("--observation-loop-max-heading-gap", type=float, default=3.141592653589793)
    parser.add_argument("--observation-loop-require-landmark-overlap", action="store_true")
    parser.add_argument("--merge-structural-topo", dest="merge_corrected_topo", action="store_true", default=True)
    parser.add_argument("--no-merge-structural-topo", dest="merge_corrected_topo", action="store_false")
    parser.add_argument("--merge-corrected-topo", dest="merge_corrected_topo", action="store_true")
    parser.add_argument("--no-merge-corrected-topo", dest="merge_corrected_topo", action="store_false")
    parser.add_argument("--visited-merge-radius", type=float, default=0.75)
    parser.add_argument("--room-merge-radius", type=float, default=2.0)
    parser.add_argument("--frontier-merge-radius", type=float, default=1.5)
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
    coord_dir = run_dir / "coord_compare"
    odom_dir = run_dir / "step_odometry_compare"

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
        "--env-landmarks",
        args.env_landmarks,
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
    coord_cmd = [
        sys.executable,
        "scripts/compare_world_relative_topo_trace.py",
        "--trace",
        str(trace_path),
        "--out-dir",
        str(coord_dir),
        "--stride",
        str(args.stride),
        "--fps",
        str(args.fps),
        "--topdown-resolution",
        str(args.topdown_resolution),
    ]
    odom_cmd = [
        sys.executable,
        "scripts/compare_step_odometry_topo_trace.py",
        "--trace",
        str(trace_path),
        "--out-dir",
        str(odom_dir),
        "--stride",
        str(args.stride),
        "--fps",
        str(args.fps),
        "--topdown-resolution",
        str(args.topdown_resolution),
        "--translation-noise-std",
        str(args.translation_noise_std),
        "--heading-noise-std-deg",
        str(args.heading_noise_std_deg),
        "--seed",
        str(args.seed),
        "--history-correction",
        args.history_correction,
        "--loop-closure-radius",
        str(args.loop_closure_radius),
        "--loop-closure-min-interval",
        str(args.loop_closure_min_interval),
        "--loop-closure-min-drift",
        str(args.loop_closure_min_drift),
        "--observation-loop-sim-threshold",
        str(args.observation_loop_sim_threshold),
        "--observation-loop-max-pose-gap",
        str(args.observation_loop_max_pose_gap),
        "--observation-loop-cooldown",
        str(args.observation_loop_cooldown),
        "--observation-loop-min-sim-margin",
        str(args.observation_loop_min_sim_margin),
        "--observation-loop-max-heading-gap",
        str(args.observation_loop_max_heading_gap),
        "--visited-merge-radius",
        str(args.visited_merge_radius),
        "--room-merge-radius",
        str(args.room_merge_radius),
        "--frontier-merge-radius",
        str(args.frontier_merge_radius),
    ]
    if args.observation_loop_require_landmark_overlap:
        odom_cmd.append("--observation-loop-require-landmark-overlap")
    if not args.merge_corrected_topo:
        odom_cmd.append("--no-merge-structural-topo")
    for cmd in (viz_cmd, coord_cmd, odom_cmd):
        if args.navmesh_path:
            cmd.extend(["--navmesh-path", args.navmesh_path])
        if args.no_gt_topdown:
            cmd.append("--no-gt-topdown")
    if args.show_planned_route:
        viz_cmd.append("--show-planned-route")
    if args.no_labels:
        coord_cmd.append("--no-labels")
        odom_cmd.append("--no-labels")

    run(trace_cmd)
    run(viz_cmd)
    run(coord_cmd)
    run(odom_cmd)

    print(
        "\nDone.\n"
        f"Trace: {ROOT / trace_path}\n"
        f"RGB frames: {ROOT / frame_dir}\n"
        f"Audit video: {ROOT / viz_dir / 'goat_semantic_audit_view.mp4'}\n"
        f"Coord compare video: {ROOT / coord_dir / 'world_relative_topo_compare.mp4'}\n"
        f"Coord compare summary: {ROOT / coord_dir / 'summary.json'}\n"
        f"Step odometry compare video: {ROOT / odom_dir / 'step_odometry_compare.mp4'}\n"
        f"Step odometry compare summary: {ROOT / odom_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
