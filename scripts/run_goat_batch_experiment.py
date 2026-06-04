"""Small-scale GOAT batch experiment: single-goal traces + optional multigoal runs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_goat_minimal import ROOT, load_goal_graph, pick_episode
from run_phase2_semantic_acceptance import load_json_gz, scene_name_from_content


def enumerate_episodes(args) -> list[dict[str, Any]]:
    content_dir = ROOT / args.dataset_dir / args.split / "content"
    files = sorted(content_dir.glob("*.json.gz"))[: args.max_scenes]
    rows: list[dict[str, Any]] = []
    for path in files:
        data = load_json_gz(path)
        episodes = data.get("episodes", [])[: args.episodes_per_scene]
        scene = scene_name_from_content(path)
        for idx, episode in enumerate(episodes):
            rows.append({
                "scene": scene,
                "episode_index": idx,
                "episode_id": episode.get("episode_id"),
                "num_tasks": len(episode.get("tasks", [])),
                "content_path": str(path),
            })
    return rows


def load_thresholds(args) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    summary_path = ROOT / args.phase2_summary
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        thresholds.update(summary.get("scan_summary", {}).get("thresholds", {}))
    if args.object_threshold is not None:
        thresholds["object"] = args.object_threshold
    if args.landmark_threshold is not None:
        thresholds["landmark"] = args.landmark_threshold
    if args.room_threshold is not None:
        thresholds["room"] = args.room_threshold
    return thresholds


def goal_count(row: dict[str, Any], args) -> int:
    path, episode = pick_episode(
        ROOT / args.dataset_dir,
        args.split,
        row["scene"],
        row["episode_index"],
    )
    ig = load_goal_graph(ROOT / args.goal_graph_dir, args.split, path, episode["episode_id"])
    from conftopo.core.instruction_graph import GoalNode

    return sum(1 for g in ig.goal_nodes if isinstance(g, GoalNode))


def run_single_goal(args, row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    case_dir = ROOT / args.output_dir / "single_goal" / f"{row['scene']}_ep{row['episode_index']}"
    trace_path = case_dir / "topo_trace_semantic.json"
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_goat_topo_trace.py",
        "--split",
        args.split,
        "--scene",
        row["scene"],
        "--episode-index",
        str(row["episode_index"]),
        "--max-steps",
        str(args.max_steps),
        "--dataset-dir",
        args.dataset_dir,
        "--scene-root",
        args.scene_root,
        "--goal-graph-dir",
        args.goal_graph_dir,
        "--output",
        str(trace_path.relative_to(ROOT)),
        "--clip-model",
        args.clip_model,
        "--clip-device",
        args.clip_device,
    ]
    if "object" in thresholds:
        cmd.extend(["--object-threshold", str(thresholds["object"])])
    if "landmark" in thresholds:
        cmd.extend(["--landmark-threshold", str(thresholds["landmark"])])
    if "room" in thresholds:
        cmd.extend(["--room-threshold", str(thresholds["room"])])

    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - started
    result: dict[str, Any] = {
        **row,
        "mode": "single_goal",
        "trace": str(trace_path),
        "elapsed_sec": round(elapsed, 2),
        "ok": proc.returncode == 0,
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout)[-2000:]
        return result

    trace = json.loads(trace_path.read_text())
    summary = trace.get("final_summary", {})
    memory = trace.get("final_memory", {})
    result.update({
        "target_object": (trace.get("current_goal") or {}).get("target_object"),
        "episode_length": summary.get("episode_length", len(trace.get("steps", []))),
        "object_nodes": summary.get("object_nodes", memory.get("objects", 0)),
        "room_nodes": summary.get("room_nodes", memory.get("rooms", 0)),
        "landmark_nodes": summary.get("landmark_nodes", memory.get("landmarks", 0)),
        "semantic_node_count": summary.get("semantic_node_count", 0),
        "collision_like_count": summary.get("collision_like_count", 0),
        "path_length_relative": summary.get("path_length_relative", 0.0),
        "memory_reuse_count": summary.get("memory_reuse_count", 0),
        "sr_proxy": summary.get("sr_proxy", False),
        "failure_reason": summary.get("failure_reason", ""),
        "navigation_stable": summary.get("collision_like_count", 0) == 0,
    })
    return result


def run_multigoal(args, row: dict[str, Any]) -> dict[str, Any]:
    case_dir = ROOT / args.output_dir / "multigoal" / f"{row['scene']}_ep{row['episode_index']}"
    trace_path = case_dir / "topo_trace_multigoal.json"
    report_path = case_dir / "multigoal_acceptance_report.json"
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_goat_multigoal_acceptance.py",
        "--split",
        args.split,
        "--scene",
        row["scene"],
        "--episode-index",
        str(row["episode_index"]),
        "--steps-per-goal",
        str(args.steps_per_goal),
        "--max-goals",
        str(args.max_goals),
        "--dataset-dir",
        args.dataset_dir,
        "--scene-root",
        args.scene_root,
        "--goal-graph-dir",
        args.goal_graph_dir,
        "--output",
        str(trace_path.relative_to(ROOT)),
        "--report",
        str(report_path.relative_to(ROOT)),
        "--clip-model",
        args.clip_model,
        "--clip-device",
        args.clip_device,
        "--phase2-summary",
        args.phase2_summary,
    ]

    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - started
    result: dict[str, Any] = {
        **row,
        "mode": "multigoal",
        "trace": str(trace_path),
        "report": str(report_path),
        "elapsed_sec": round(elapsed, 2),
        "ok": proc.returncode == 0,
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout)[-2000:]
        return result

    report = json.loads(report_path.read_text())
    result.update({
        "overall_passed": report.get("overall_passed", False),
        "memory_preservation_passed": report.get("memory_preservation_passed", False),
        "semantic_build_passed": report.get("semantic_build_passed", False),
        "memory_reuse_passed": report.get("memory_reuse_passed", False),
        "navigation_stable": report.get("navigation_stable", False),
        "semantic_nodes_built": report.get("semantic_nodes_built", {}),
        "final_summary": report.get("final_summary", {}),
    })
    return result


def _mean(values: list[float | int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def aggregate_results(single_rows: list[dict[str, Any]], multigoal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_single = [r for r in single_rows if r.get("ok")]
    ok_multi = [r for r in multigoal_rows if r.get("ok")]

    single_agg: dict[str, Any] = {"count": len(single_rows), "ok_count": len(ok_single)}
    if ok_single:
        single_agg.update({
            "sr_proxy_rate": sum(1 for r in ok_single if r.get("sr_proxy")) / len(ok_single),
            "navigation_stable_rate": sum(1 for r in ok_single if r.get("navigation_stable")) / len(ok_single),
            "mean_episode_length": _mean([r.get("episode_length", 0) for r in ok_single]),
            "mean_object_nodes": _mean([r.get("object_nodes", 0) for r in ok_single]),
            "mean_landmark_nodes": _mean([r.get("landmark_nodes", 0) for r in ok_single]),
            "mean_semantic_node_count": _mean([r.get("semantic_node_count", 0) for r in ok_single]),
            "mean_collision_like_count": _mean([r.get("collision_like_count", 0) for r in ok_single]),
            "mean_path_length_relative": _mean([r.get("path_length_relative", 0.0) for r in ok_single]),
            "failure_reasons": {},
        })
        reasons: dict[str, int] = {}
        for r in ok_single:
            reason = r.get("failure_reason") or "ok"
            if r.get("sr_proxy"):
                reason = "ok"
            reasons[reason] = reasons.get(reason, 0) + 1
        single_agg["failure_reasons"] = reasons

    multi_agg: dict[str, Any] = {"count": len(multigoal_rows), "ok_count": len(ok_multi)}
    if ok_multi:
        multi_agg.update({
            "overall_passed_rate": sum(1 for r in ok_multi if r.get("overall_passed")) / len(ok_multi),
            "memory_preservation_rate": sum(1 for r in ok_multi if r.get("memory_preservation_passed")) / len(ok_multi),
            "semantic_build_rate": sum(1 for r in ok_multi if r.get("semantic_build_passed")) / len(ok_multi),
            "memory_reuse_rate": sum(1 for r in ok_multi if r.get("memory_reuse_passed")) / len(ok_multi),
            "navigation_stable_rate": sum(1 for r in ok_multi if r.get("navigation_stable")) / len(ok_multi),
            "mean_episode_length": _mean([(r.get("final_summary") or {}).get("episode_length", 0) for r in ok_multi]),
            "mean_semantic_reuse_count": _mean([(r.get("final_summary") or {}).get("semantic_reuse_count", 0) for r in ok_multi]),
        })

    return {"single_goal": single_agg, "multigoal": multi_agg}


def select_multigoal_rows(rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    by_scene = {row["scene"]: row for row in rows}
    selected: list[dict[str, Any]] = []

    if args.multigoal_scenes:
        wanted = [s.strip() for s in args.multigoal_scenes.split(",") if s.strip()]
        for scene in wanted:
            if scene in by_scene:
                selected.append(by_scene[scene])
                continue
            try:
                path, episode = pick_episode(ROOT / args.dataset_dir, args.split, scene, 0)
                selected.append({
                    "scene": scene,
                    "episode_index": 0,
                    "episode_id": episode.get("episode_id"),
                    "num_tasks": len(episode.get("tasks", [])),
                    "content_path": str(path),
                })
            except Exception as exc:
                print(json.dumps({"warn": "multigoal_scene_skipped", "scene": scene, "error": repr(exc)}, ensure_ascii=False))
    else:
        for row in rows:
            try:
                if goal_count(row, args) >= args.max_goals:
                    selected.append(row)
            except Exception:
                continue
            if len(selected) >= args.multigoal_count:
                break
    return selected[: args.multigoal_count]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small GOAT batch experiment and aggregate trace stats.")
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    ap.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    ap.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    ap.add_argument("--output-dir", default="data/logs/goat_topo/batch_experiment")
    ap.add_argument("--max-scenes", type=int, default=6)
    ap.add_argument("--episodes-per-scene", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--multigoal-count", type=int, default=2)
    ap.add_argument("--multigoal-scenes", default="", help="Comma-separated scenes for multigoal; default auto-pick")
    ap.add_argument("--steps-per-goal", type=int, default=80)
    ap.add_argument("--max-goals", type=int, default=4)
    ap.add_argument("--skip-multigoal", action="store_true")
    ap.add_argument("--clip-model", default="ViT-B/32")
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument("--object-threshold", type=float, default=None)
    ap.add_argument("--room-threshold", type=float, default=None)
    ap.add_argument("--landmark-threshold", type=float, default=None)
    ap.add_argument(
        "--phase2-summary",
        default="data/logs/goat_topo/phase2_pathfinder_acceptance/summary.json",
    )
    args = ap.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args)
    episodes = enumerate_episodes(args)

    print(json.dumps({"phase": "enumerate", "episodes": len(episodes), "thresholds": thresholds}, ensure_ascii=False))

    single_results: list[dict[str, Any]] = []
    for i, row in enumerate(episodes):
        print(json.dumps({"phase": "single_goal", "index": i + 1, "total": len(episodes), "scene": row["scene"]}, ensure_ascii=False))
        single_results.append(run_single_goal(args, row, thresholds))

    multigoal_results: list[dict[str, Any]] = []
    if not args.skip_multigoal and args.multigoal_count > 0:
        multi_rows = select_multigoal_rows(episodes, args)
        for i, row in enumerate(multi_rows):
            print(json.dumps({"phase": "multigoal", "index": i + 1, "total": len(multi_rows), "scene": row["scene"]}, ensure_ascii=False))
            multigoal_results.append(run_multigoal(args, row))

    aggregate = aggregate_results(single_results, multigoal_results)
    summary = {
        "config": {
            "split": args.split,
            "max_scenes": args.max_scenes,
            "episodes_per_scene": args.episodes_per_scene,
            "max_steps": args.max_steps,
            "multigoal_count": 0 if args.skip_multigoal else args.multigoal_count,
            "steps_per_goal": args.steps_per_goal,
            "max_goals": args.max_goals,
            "thresholds": thresholds,
        },
        "aggregate": aggregate,
        "single_goal_results": single_results,
        "multigoal_results": multigoal_results,
    }

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"ok": True, "output": str(summary_path), "aggregate": aggregate}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
