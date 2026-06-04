from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"waypoint_visited": "#2563eb", "waypoint_frontier": "#f97316", "waypoint_candidate": "#06b6d4", "object": "#16a34a", "room": "#dc2626", "landmark": "#9333ea"}
MARKERS = {"waypoint_visited": "o", "waypoint_frontier": "^", "waypoint_candidate": "v", "object": "D", "room": "s", "landmark": "P"}
LABELS = {"waypoint_visited": "visited waypoint", "waypoint_frontier": "frontier", "waypoint_candidate": "candidate/ghost", "object": "object", "room": "room", "landmark": "landmark"}


def collect_limits(steps):
    xs, zs = [], []
    for st in steps:
        p = st["position"]
        xs.append(p[0]); zs.append(p[2])
        if st.get("target_position"):
            t = st["target_position"]
            xs.append(t[0]); zs.append(t[2])
        for n in st["topo"]["nodes"]:
            xs.append(n["position"][0]); zs.append(n["position"][2])
    return (min(xs)-1.0, max(xs)+1.0), (min(zs)-1.0, max(zs)+1.0)


def fmt_scores(title, scores):
    if not scores:
        return [title + ": n/a"]
    lines = [title]
    for item in scores[:3]:
        lines.append("  {}: {:.3f}".format(item.get("label", "n/a"), item.get("score", 0.0)))
    return lines


def fmt_candidates(candidates):
    if not candidates:
        return ["Top candidates: n/a"]
    lines = ["Top candidates"]
    for item in candidates[:3]:
        lines.append("  {}: {:.3f}".format(item.get("node_id", "n/a"), float(item.get("score", 0.0))))
    return lines


def fmt_semantic_decisions(decisions):
    if not decisions:
        return ["Semantic decisions: n/a"]
    lines = ["Semantic decisions"]
    for key, short in [("objects", "obj"), ("rooms", "room"), ("landmarks", "lm")]:
        rows = decisions.get(key, [])
        if not rows:
            lines.append("  {}: n/a".format(short))
            continue
        item = rows[0]
        lines.append(
            "  {} {} {:.3f}/{:.3f} {}".format(
                short,
                item.get("label", "n/a"),
                float(item.get("score", 0.0)),
                float(item.get("threshold", 0.0)),
                item.get("decision", ""),
            )
        )
    return lines


def fig_to_rgb(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()


def resize_to(img, shape):
    from PIL import Image
    h, w = shape[:2]
    return np.asarray(Image.fromarray(img).resize((w, h)))


def draw_topo_frame(trace, idx, xlim, zlim, out_path=None):
    st = trace["steps"][idx]
    nodes = {n["id"]: n for n in st["topo"]["nodes"]}
    fig, (ax, info) = plt.subplots(1, 2, figsize=(13, 6), dpi=130, gridspec_kw={"width_ratios": [2.1, 1]})
    for e in st["topo"]["edges"]:
        a, b = nodes.get(e["source"]), nodes.get(e["target"])
        if not a or not b:
            continue
        nav = e.get("type") == "navigable"
        ax.plot([a["position"][0], b["position"][0]], [a["position"][2], b["position"][2]], color="#6b7280", linewidth=1.2 if nav else 0.8, linestyle="-" if nav else "--", alpha=0.65, zorder=1)
    selection = st.get("selection_debug", {})
    rank_labels = {}
    for rank, item in enumerate(selection.get("top_candidate_scores", [])[:5], start=1):
        node_id = item.get("node_id")
        if node_id:
            rank_labels[node_id] = "#{} {:.3f}".format(rank, float(item.get("score", 0.0)))
    for typ in ["waypoint_visited", "waypoint_frontier", "object", "room", "landmark"]:
        group = [n for n in nodes.values() if n["type"] == typ]
        if not group:
            continue
        ax.scatter([n["position"][0] for n in group], [n["position"][2] for n in group], s=[70 + 180 * float(n.get("confidence", 0.5)) for n in group], marker=MARKERS[typ], color=COLORS[typ], edgecolors="black", linewidths=0.6, alpha=0.9, label=LABELS[typ], zorder=3)
        for n in group:
            label = n["id"] if not n.get("label") else "{}:{}".format(n["id"], n["label"])
            ax.text(n["position"][0], n["position"][2] + 0.08, label, fontsize=7, ha="center")
            if n["id"] in rank_labels:
                ax.text(
                    n["position"][0],
                    n["position"][2] - 0.16,
                    rank_labels[n["id"]],
                    fontsize=7,
                    ha="center",
                    color="#111827",
                    bbox={"facecolor": "#fef3c7", "edgecolor": "#f59e0b", "boxstyle": "round,pad=0.18", "alpha": 0.78},
                    zorder=6,
                )
    path = np.array([s["position"] for s in trace["steps"][:idx+1]])
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 2], color="#111827", linewidth=1.6, alpha=0.8, label="agent path", zorder=2)
    cur = st["position"]
    ax.scatter([cur[0]], [cur[2]], marker="*", s=260, color="#facc15", edgecolors="black", linewidths=1.0, label="agent", zorder=5)
    if st.get("target_position"):
        tgt = st["target_position"]
        ax.scatter([tgt[0]], [tgt[2]], marker="X", s=150, color="#ef4444", edgecolors="black", linewidths=0.8, label="planned target", zorder=4)
        ax.plot([cur[0], tgt[0]], [cur[2], tgt[2]], color="#ef4444", linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    ax.set_title("DynamicTopoMap | step {}".format(st.get("step")))
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_xlim(*xlim); ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.22); ax.legend(loc="upper right", fontsize=7)
    mem, goal, perception = st["memory"], trace.get("current_goal", {}), st.get("perception", {})
    sticky = st.get("sticky_debug", {})
    selection = st.get("selection_debug", {})
    event = selection.get("navigation_event") or st.get("navigation_event") or {}
    blocked = selection.get("blocked_targets") or sticky.get("blocked_targets") or {}
    skipped = selection.get("skipped_candidates") or sticky.get("skipped_candidates") or []
    lines = ["Topo Memory", "total nodes: {}".format(mem["total_nodes"]), "visited: {}".format(mem["visited_waypoints"]), "frontiers: {}".format(mem["frontiers"]), "objects: {}".format(mem["objects"]), "rooms: {}".format(mem["rooms"]), "landmarks: {}".format(mem.get("landmarks", 0)), "", "Goal/Target", "goal: {}/object".format(goal.get("target_object", "n/a")), "target: {}".format(st.get("target_node_id") or "n/a"), "selected rank: {}".format(selection.get("selected_rank") or "n/a"), "sticky: {}".format(sticky.get("sticky_target_id") or "n/a"), "release: {}".format(sticky.get("sticky_release_reason") or ""), "event: {}".format(event.get("action") or event.get("reason") or ""), "blocked: {}".format(len(blocked)), "skipped: {}".format(len(skipped)), "", "Action", "low: {}".format(st.get("low_action")), "agent: {}".format(st.get("agent_action")), ""]
    lines.extend(fmt_candidates(selection.get("top_candidate_scores", [])))
    lines.append("")
    lines.extend(fmt_semantic_decisions(selection.get("semantic_decisions", {})))
    lines.append("")
    lines.extend(fmt_scores("Top object CLIP", perception.get("goal_scores", [])))
    lines.extend(fmt_scores("Top room CLIP", perception.get("room_scores", [])))
    lines.extend(fmt_scores("Top landmark CLIP", perception.get("landmark_scores", [])))
    info.axis("off"); info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig); return None
    img = fig_to_rgb(fig); plt.close(fig); return img


def draw_first_person_frame(trace, idx):
    st = trace["steps"][idx]
    frame = st.get("rgb_frame")
    if frame:
        rgb = imageio.imread(ROOT / frame)
    else:
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    fig, (ax, info) = plt.subplots(1, 2, figsize=(8, 4.5), dpi=130, gridspec_kw={"width_ratios": [1.25, 1]})
    ax.imshow(rgb); ax.axis("off"); ax.set_title("First person RGB | step {}".format(st.get("step")))
    goal = trace.get("current_goal", {})
    p = st.get("perception", {})
    selection = st.get("selection_debug", {})
    lines = ["GOAT Input", "{}/object".format(goal.get("target_object", "n/a")), "", "Action", "low: {}".format(st.get("low_action")), "target: {}".format(st.get("target_node_id") or "n/a"), "selected rank: {}".format(selection.get("selected_rank") or "n/a"), ""]
    lines.extend(fmt_candidates(selection.get("top_candidate_scores", [])))
    lines.append("")
    lines.extend(fmt_scores("Top object CLIP", p.get("goal_scores", [])))
    lines.extend(fmt_scores("Top room CLIP", p.get("room_scores", [])))
    lines.extend(fmt_scores("Top landmark CLIP", p.get("landmark_scores", [])))
    info.axis("off"); info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout(); img = fig_to_rgb(fig); plt.close(fig); return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/logs/goat_topo/topo_trace_semantic.json")
    ap.add_argument("--out-dir", default="data/logs/goat_topo/viz_semantic")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    trace = json.load(open(ROOT / args.trace if not Path(args.trace).is_absolute() else args.trace))
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlim, zlim = collect_limits(trace["steps"])
    final_png = out_dir / "topo_map_final.png"
    draw_topo_frame(trace, len(trace["steps"])-1, xlim, zlim, final_png)
    key_frames = []
    for i in sorted(set([0, min(10, len(trace["steps"])-1), min(25, len(trace["steps"])-1), min(50, len(trace["steps"])-1), len(trace["steps"])-1])):
        p = out_dir / "topo_map_step_{:03d}.png".format(i)
        draw_topo_frame(trace, i, xlim, zlim, p); key_frames.append(str(p))
    stride = max(1, args.stride)
    topo_frames, fp_frames, dual_frames = [], [], []
    for i in range(0, len(trace["steps"]), stride):
        topo = draw_topo_frame(trace, i, xlim, zlim)
        fp = draw_first_person_frame(trace, i)
        fp = resize_to(fp, topo.shape)
        dual = np.concatenate([fp, topo], axis=1)
        topo_frames.append(topo); fp_frames.append(fp); dual_frames.append(dual)
    paths = {
        "first_person_video": out_dir / "first_person_semantic.mp4",
        "topo_video": out_dir / "topo_map_semantic_growth.mp4",
        "dual_video": out_dir / "goat_semantic_dual_view.mp4",
        "gif": out_dir / "topo_map_semantic_growth.gif",
    }
    imageio.mimsave(paths["first_person_video"], fp_frames, fps=args.fps)
    imageio.mimsave(paths["topo_video"], topo_frames, fps=args.fps)
    imageio.mimsave(paths["dual_video"], dual_frames, fps=args.fps)
    imageio.mimsave(paths["gif"], topo_frames, duration=1.0/args.fps)
    summary = {"trace": args.trace, "steps": len(trace["steps"]), "final_png": str(final_png), "key_frames": key_frames, "final_memory": trace.get("final_memory", {}), "final_summary": trace.get("final_summary", {})}
    summary.update({k: str(v) for k, v in paths.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
