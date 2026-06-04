from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from run_goat_minimal import ROOT, find_scene_file, load_goal_graph, make_sim, normalize_quat
from conftopo.acceptance.phase2 import auto_threshold
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception import ClipRuntimeEncoder
from conftopo.perception.light_perceiver import cosine_sim


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def scene_name_from_content(path: Path) -> str:
    return path.name.replace(".json.gz", "")


def landmark_names(raw) -> list[str]:
    names = []
    if isinstance(raw, dict):
        raw = list(raw.keys())
    for item in raw or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("object") or item.get("label")
            if name:
                names.append(str(name))
        else:
            names.append(str(item))
    return names


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


def score_episode(encoder, dataset_dir: Path, scene_root: Path, goal_graph_dir: Path, split: str, content_path: Path, episode_index: int) -> dict[str, Any] | None:
    data = load_json_gz(content_path)
    episodes = data.get("episodes", [])
    if episode_index >= len(episodes):
        return None
    episode = episodes[episode_index]
    try:
        ig = load_goal_graph(goal_graph_dir, split, content_path, episode["episode_id"])
        goal = ig.get_current_goal()
        if not isinstance(goal, GoalNode) or goal.target_embedding is None:
            return None
        scene_file = find_scene_file(episode["scene_id"], scene_root)
        sim = make_sim(scene_file)
        sim_agent = sim.initialize_agent(0)
        set_start_state(sim_agent, episode)
        obs = sim.get_sensor_observations()
        rgb = obs.get("color_sensor")
        embed = encoder.encode_image(rgb)
        sim.close()
        object_score = float(cosine_sim(embed, goal.target_embedding[np.newaxis, :])[0])
        landmark_score = 0.0
        landmark_label = ""
        lm_names = landmark_names(goal.landmarks)
        if lm_names:
            lm_embeds = goal.landmark_embeddings
            if lm_embeds is None:
                lm_embeds = encoder.encode_text(lm_names)
            lm_sims = cosine_sim(embed, lm_embeds)
            best_idx = int(np.argmax(lm_sims))
            landmark_score = float(lm_sims[best_idx])
            landmark_label = lm_names[best_idx]
        return {
            "scene": scene_name_from_content(content_path),
            "episode_index": episode_index,
            "episode_id": episode["episode_id"],
            "target_object": goal.target_object,
            "landmarks": landmark_names(goal.landmarks),
            "best_landmark": landmark_label,
            "object_score": object_score,
            "landmark_score": landmark_score,
        }
    except Exception as exc:
        return {"scene": scene_name_from_content(content_path), "episode_index": episode_index, "error": repr(exc), "object_score": -1.0, "landmark_score": -1.0, "landmarks": []}


def scan_candidates(args) -> list[dict[str, Any]]:
    encoder = ClipRuntimeEncoder(args.clip_model, args.clip_device)
    content_dir = ROOT / args.dataset_dir / args.split / "content"
    files = sorted(content_dir.glob("*.json.gz"))[: args.max_scenes]
    rows = []
    for path in files:
        data = load_json_gz(path)
        episodes = data.get("episodes", [])[: args.episodes_per_scene]
        for idx, _ in enumerate(episodes):
            row = score_episode(encoder, ROOT / args.dataset_dir, ROOT / args.scene_root, ROOT / args.goal_graph_dir, args.split, path, idx)
            if row:
                rows.append(row)
                print(json.dumps({"scan": row}, ensure_ascii=False))
    return rows


def run_case(args, case_name: str, row: dict[str, Any], object_threshold: float, landmark_threshold: float) -> dict[str, Any]:
    case_dir = ROOT / args.output_dir / case_name
    frame_dir = case_dir / "rgb_frames"
    trace_path = case_dir / "topo_trace_semantic.json"
    viz_dir = case_dir / "viz"
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_goat_topo_trace.py",
        "--split", args.split,
        "--scene", row["scene"],
        "--episode-index", str(row["episode_index"]),
        "--max-steps", str(args.max_steps),
        "--dataset-dir", args.dataset_dir,
        "--scene-root", args.scene_root,
        "--goal-graph-dir", args.goal_graph_dir,
        "--output", str(trace_path.relative_to(ROOT)),
        "--frame-dir", str(frame_dir.relative_to(ROOT)),
        "--clip-model", args.clip_model,
        "--clip-device", args.clip_device,
        "--object-threshold", str(object_threshold),
        "--landmark-threshold", str(landmark_threshold),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/visualize_goat_topo_trace.py", "--trace", str(trace_path.relative_to(ROOT)), "--out-dir", str(viz_dir.relative_to(ROOT)), "--stride", str(args.video_stride), "--fps", str(args.fps)], cwd=ROOT, check=True)
    trace = json.load(open(trace_path))
    return {"case": case_name, "candidate": row, "trace": str(trace_path), "viz_dir": str(viz_dir), "final_memory": trace.get("final_memory"), "final_summary": trace.get("final_summary")}



def build_navigation_report(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {cases: {}, total_collision_like_count: 0, long_stuck_targets: []}
    for result in case_results:
        trace_path = ROOT / result[trace] if not Path(result[trace]).is_absolute() else Path(result[trace])
        trace = json.load(open(trace_path))
        steps = trace.get(steps, [])
        action_counts = Counter(str(st.get(low_action)) for st in steps)
        collision_count = sum(1 for st in steps if st.get(collision_like) or (st.get(navigation_debug) or {}).get(collision_like))
        target_segments = []
        current = None
        start_idx = 0
        positions: list[list[float]] = []
        for idx, st in enumerate(steps):
            target = st.get(target_node_id) or none
            pos = st.get(position) or [0.0, 0.0, 0.0]
            if current is None:
                current = target
                start_idx = idx
                positions = [pos]
                continue
            if target != current:
                target_segments.append((current, start_idx, idx - 1, positions))
                current = target
                start_idx = idx
                positions = [pos]
            else:
                positions.append(pos)
        if current is not None:
            target_segments.append((current, start_idx, len(steps) - 1, positions))

        no_progress_segments = []
        for target, start, end, seg_positions in target_segments:
            if end - start + 1 < 5:
                continue
            arr = np.asarray(seg_positions, dtype=np.float32)
            displacement = float(np.linalg.norm(arr[-1, [0, 2]] - arr[0, [0, 2]])) if len(arr) else 0.0
            if end - start + 1 >= 10 and displacement < 0.15:
                item = {target_node_id: target, start_step: start, end_step: end, steps: end - start + 1, displacement: displacement}
                no_progress_segments.append(item)
                if end - start + 1 >= 20:
                    report[long_stuck_targets].append({case: result[case], **item})

        skipped = Counter()
        release_reasons = Counter()
        for st in steps:
            nav = st.get(navigation_debug) or {}
            if nav.get(release_reason):
                release_reasons[str(nav.get(release_reason))] += 1
            selection = st.get(selection_debug) or {}
            for item in selection.get(skipped_candidates, []):
                skipped[str(item.get(reason, unknown))] += 1
        case_report = {
            steps: len(steps),
            action_counts: dict(action_counts),
            collision_like_count: collision_count,
            no_progress_segments: no_progress_segments,
            long_stuck_target_count: sum(1 for s in no_progress_segments if s[steps] >= 20),
            skipped_candidate_reasons: dict(skipped),
            release_reasons: dict(release_reasons),
            final_summary: trace.get(final_summary, {}),
        }
        report[cases][result[case]] = case_report
        report[total_collision_like_count] += collision_count
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    ap.add_argument("--scene-root", default="data/scene_datasets/hm3d")
    ap.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    ap.add_argument("--output-dir", default="data/logs/goat_topo/phase2_semantic_acceptance")
    ap.add_argument("--max-scenes", type=int, default=12)
    ap.add_argument("--episodes-per-scene", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--clip-model", default="ViT-B/32")
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument("--object-min", type=float, default=0.045)
    ap.add_argument("--landmark-min", type=float, default=0.045)
    ap.add_argument("--ratio", type=float, default=0.85)
    ap.add_argument("--video-stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = scan_candidates(args)
    valid_object = [r for r in rows if r.get("object_score", -1) >= 0]
    valid_landmark = [r for r in rows if r.get("landmarks") and r.get("landmark_score", -1) >= 0]
    if not valid_object:
        raise RuntimeError("No object candidates found")
    if not valid_landmark:
        raise RuntimeError("No landmark candidates found; increase --max-scenes or --episodes-per-scene")
    object_row = max(valid_object, key=lambda r: r["object_score"])
    landmark_row = max(valid_landmark, key=lambda r: r["landmark_score"])
    object_threshold = auto_threshold([object_row["object_score"]], args.object_min, args.ratio)
    landmark_threshold = auto_threshold([landmark_row["landmark_score"]], args.landmark_min, args.ratio)
    scan_summary = {"rows": rows, "selected_object": object_row, "selected_landmark": landmark_row, "thresholds": {"object": object_threshold, "landmark": landmark_threshold, "ratio": args.ratio, "object_min": args.object_min, "landmark_min": args.landmark_min}}
    (out_dir / "scan_summary.json").write_text(json.dumps(scan_summary, indent=2, ensure_ascii=False))
    results = [
        run_case(args, "object_acceptance", object_row, object_threshold, args.landmark_min),
        run_case(args, "landmark_acceptance", landmark_row, args.object_min, landmark_threshold),
    ]
    summary = {"scan_summary": scan_summary, "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
