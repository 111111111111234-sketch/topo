from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "waypoint_visited": "#2563eb",
    "waypoint_frontier": "#f97316",
    "waypoint_candidate": "#06b6d4",
    "object": "#16a34a",
    "room": "#dc2626",
    "landmark": "#9333ea",
}
MARKERS = {
    "waypoint_visited": "o",
    "waypoint_frontier": "^",
    "waypoint_candidate": "v",
    "object": "D",
    "room": "s",
    "landmark": "P",
}
LABELS = {
    "waypoint_visited": "visited waypoint",
    "waypoint_frontier": "frontier",
    "waypoint_candidate": "candidate/ghost",
    "object": "object",
    "room": "room",
    "landmark": "landmark",
}
NODE_TYPES = [
    "waypoint_visited",
    "waypoint_frontier",
    "waypoint_candidate",
    "object",
    "room",
    "landmark",
]


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()


def resolve_repo_path(path_like):
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and (ROOT / path).exists():
        return ROOT / path
    parts = path.parts
    if "data" in parts:
        candidate = ROOT.joinpath(*parts[parts.index("data"):])
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def load_origin_world(trace):
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


def infer_navmesh_path(trace, override=None):
    if override:
        path = resolve_repo_path(override)
        return path, "override" if path else f"override not found: {override}"
    scene_file = resolve_repo_path(trace.get("scene_file"))
    if scene_file is not None:
        candidates = []
        name = scene_file.name
        if name.endswith(".basis.glb"):
            candidates.append(scene_file.with_name(name.replace(".basis.glb", ".basis.navmesh")))
        if scene_file.suffix == ".glb":
            candidates.append(scene_file.with_suffix(".navmesh"))
        candidates.append(scene_file.with_name(scene_file.stem + ".navmesh"))
        for candidate in candidates:
            if candidate.exists():
                return candidate, "scene_file"
    scene = trace.get("scene") or (Path(str(trace.get("scene_file", ""))).name.split(".")[0] if trace.get("scene_file") else "")
    scene = scene.replace(".basis", "")
    if scene:
        matches = sorted((ROOT / "data/scene_datasets").glob(f"**/{scene}*.navmesh"))
        if matches:
            return matches[0], "scene search"
    return None, "navmesh not found"


def load_gt_topdown(trace, resolution, navmesh_override=None, disabled=False):
    if disabled:
        return {"available": False, "reason": "disabled by --no-gt-topdown"}
    navmesh_path, navmesh_source = infer_navmesh_path(trace, navmesh_override)
    if navmesh_path is None:
        return {"available": False, "reason": navmesh_source}
    try:
        import habitat_sim

        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh_path)):
            return {"available": False, "reason": f"failed to load navmesh: {navmesh_path}"}
        bounds_min, bounds_max = pathfinder.get_bounds()
        slice_height = float(bounds_min[1] + 0.1)
        topdown = np.asarray(pathfinder.get_topdown_view(resolution, slice_height), dtype=bool)
        return {
            "available": True,
            "topdown": topdown,
            "bounds_min": np.asarray(bounds_min, dtype=np.float32),
            "bounds_max": np.asarray(bounds_max, dtype=np.float32),
            "navmesh_path": str(navmesh_path),
            "navmesh_source": navmesh_source,
            "resolution": float(resolution),
        }
    except Exception as exc:
        return {"available": False, "reason": f"topdown unavailable: {exc}", "navmesh_path": str(navmesh_path)}


def trace_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def trace_frame(trace: dict, origin_world: np.ndarray | None) -> str:
    frame = trace.get("coordinate_frame")
    if frame:
        return str(frame)
    steps = trace.get("steps") or []
    if not steps:
        return "episode_start_relative"
    first = np.asarray(steps[0].get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    if origin_world is not None and np.linalg.norm(first[[0, 2]] - origin_world[[0, 2]]) < 0.5:
        return "world"
    if np.linalg.norm(first[[0, 2]]) < 0.5:
        return "episode_start_relative"
    return "world"


def to_world(position, origin_world: np.ndarray, frame: str) -> np.ndarray:
    arr = np.asarray(position, dtype=np.float32)
    if frame == "episode_start_relative":
        return arr + origin_world
    return arr


def to_relative(position, origin_world: np.ndarray, frame: str) -> np.ndarray:
    arr = np.asarray(position, dtype=np.float32)
    if frame == "world":
        return arr - origin_world
    return arr


def transform_position(position, origin_world: np.ndarray, frame: str, target_frame: str) -> np.ndarray:
    if target_frame == "world":
        return to_world(position, origin_world, frame)
    if target_frame == "relative":
        return to_relative(position, origin_world, frame)
    raise ValueError(f"Unknown target_frame: {target_frame}")


def collect_limits(trace: dict, origin_world: np.ndarray, frame: str, target_frame: str) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    zs: list[float] = []
    for st in trace.get("steps", []):
        for value in [st.get("position"), st.get("target_position")]:
            if value is None:
                continue
            p = transform_position(value, origin_world, frame, target_frame)
            xs.append(float(p[0]))
            zs.append(float(p[2]))
        for node in st.get("topo", {}).get("nodes", []):
            p = transform_position(node["position"], origin_world, frame, target_frame)
            xs.append(float(p[0]))
            zs.append(float(p[2]))
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    return (min(xs) - 1.0, max(xs) + 1.0), (min(zs) - 1.0, max(zs) + 1.0)


def load_rgb_frame(step: dict) -> np.ndarray:
    frame = step.get("rgb_frame")
    if not frame:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    frame_path = resolve_repo_path(frame)
    if frame_path is None:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    rgb = imageio.imread(frame_path)
    return rgb[..., :3] if rgb.ndim == 3 and rgb.shape[-1] == 4 else rgb


def draw_topdown_background(ax, gt_ctx: dict, origin_world: np.ndarray, target_frame: str) -> None:
    if not gt_ctx.get("available"):
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "GT topdown unavailable\n{}".format(gt_ctx.get("reason", "")),
            ha="center",
            va="center",
            fontsize=9,
        )
        return
    bounds_min = gt_ctx["bounds_min"]
    bounds_max = gt_ctx["bounds_max"]
    if target_frame == "world":
        extent = [bounds_min[0], bounds_max[0], bounds_max[2], bounds_min[2]]
    else:
        extent = [
            bounds_min[0] - origin_world[0],
            bounds_max[0] - origin_world[0],
            bounds_max[2] - origin_world[2],
            bounds_min[2] - origin_world[2],
        ]
    ax.imshow(gt_ctx["topdown"], cmap="gray", origin="upper", extent=extent, alpha=0.58)


def draw_topo(
    ax,
    trace: dict,
    idx: int,
    origin_world: np.ndarray,
    frame: str,
    target_frame: str,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    title: str,
    show_labels: bool,
) -> None:
    st = trace["steps"][idx]
    nodes = {node["id"]: node for node in st.get("topo", {}).get("nodes", [])}

    def pos(value):
        return transform_position(value, origin_world, frame, target_frame)

    for edge in st.get("topo", {}).get("edges", []):
        a = nodes.get(edge.get("source"))
        b = nodes.get(edge.get("target"))
        if a is None or b is None:
            continue
        pa = pos(a["position"])
        pb = pos(b["position"])
        nav = edge.get("type") == "navigable"
        ax.plot(
            [pa[0], pb[0]],
            [pa[2], pb[2]],
            color="#6b7280",
            linewidth=1.0 if nav else 0.7,
            linestyle="-" if nav else "--",
            alpha=0.65,
            zorder=1,
        )

    for node_type in NODE_TYPES:
        group = [node for node in nodes.values() if node.get("type") == node_type]
        if not group:
            continue
        points = [pos(node["position"]) for node in group]
        ax.scatter(
            [p[0] for p in points],
            [p[2] for p in points],
            s=[45 + 120 * float(node.get("confidence", 0.5)) for node in group],
            marker=MARKERS[node_type],
            color=COLORS[node_type],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label=LABELS[node_type],
            zorder=3,
        )
        if show_labels:
            for node, p in zip(group, points):
                label = node["id"] if not node.get("label") else "{}:{}".format(node["id"], node["label"])
                ax.text(float(p[0]), float(p[2]) + 0.08, label, fontsize=6.0, ha="center", zorder=5)

    path = np.asarray([pos(step["position"]) for step in trace["steps"][: idx + 1]], dtype=np.float32)
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 2], color="#111827", linewidth=1.35, alpha=0.85, label="agent path", zorder=2)
    cur = pos(st["position"])
    ax.scatter([cur[0]], [cur[2]], marker="*", s=180, color="#facc15", edgecolors="black", linewidths=0.9, label="agent", zorder=5)
    if st.get("target_position") is not None:
        tgt = pos(st["target_position"])
        ax.scatter([tgt[0]], [tgt[2]], marker="X", s=105, color="#ef4444", edgecolors="black", linewidths=0.7, label="target", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("world x" if target_frame == "world" else "relative x")
    ax.set_ylabel("world z" if target_frame == "world" else "relative z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=6.0)


def max_roundtrip_error(trace: dict, origin_world: np.ndarray, frame: str) -> float:
    errors = []
    for st in trace.get("steps", []):
        values = [st.get("position"), st.get("target_position")]
        values.extend(node.get("position") for node in st.get("topo", {}).get("nodes", []))
        for value in values:
            if value is None:
                continue
            rel = to_relative(value, origin_world, frame)
            world = rel + origin_world
            back = world - origin_world
            errors.append(float(np.linalg.norm((back - rel)[[0, 2]])))
    return max(errors) if errors else 0.0


def planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b)[[0, 2]]))


def coordinate_invariance_metrics(trace: dict, origin_world: np.ndarray, frame: str) -> dict[str, float]:
    pairwise_deltas: list[float] = []
    edge_deltas: list[float] = []
    agent_node_deltas: list[float] = []

    for st in trace.get("steps", []):
        nodes = st.get("topo", {}).get("nodes", [])
        by_id = {node["id"]: node for node in nodes}
        world_nodes = {node["id"]: transform_position(node["position"], origin_world, frame, "world") for node in nodes}
        rel_nodes = {node["id"]: transform_position(node["position"], origin_world, frame, "relative") for node in nodes}

        ids = list(by_id)
        for i, node_a in enumerate(ids):
            for node_b in ids[i + 1:]:
                world_dist = planar_distance(world_nodes[node_a], world_nodes[node_b])
                rel_dist = planar_distance(rel_nodes[node_a], rel_nodes[node_b])
                pairwise_deltas.append(abs(world_dist - rel_dist))

        for edge in st.get("topo", {}).get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source not in by_id or target not in by_id:
                continue
            world_dist = planar_distance(world_nodes[source], world_nodes[target])
            rel_dist = planar_distance(rel_nodes[source], rel_nodes[target])
            edge_deltas.append(abs(world_dist - rel_dist))

        if st.get("position") is not None:
            world_agent = transform_position(st["position"], origin_world, frame, "world")
            rel_agent = transform_position(st["position"], origin_world, frame, "relative")
            for node_id in ids:
                world_dist = planar_distance(world_agent, world_nodes[node_id])
                rel_dist = planar_distance(rel_agent, rel_nodes[node_id])
                agent_node_deltas.append(abs(world_dist - rel_dist))

    return {
        "max_pairwise_distance_delta": max(pairwise_deltas) if pairwise_deltas else 0.0,
        "max_edge_length_delta": max(edge_deltas) if edge_deltas else 0.0,
        "max_agent_node_distance_delta": max(agent_node_deltas) if agent_node_deltas else 0.0,
    }


def draw_compare_frame(
    trace: dict,
    idx: int,
    gt_ctx: dict,
    origin_world: np.ndarray,
    frame: str,
    world_limits: tuple[tuple[float, float], tuple[float, float]],
    rel_limits: tuple[tuple[float, float], tuple[float, float]],
    show_labels: bool,
) -> np.ndarray:
    st = trace["steps"][idx]
    rgb = load_rgb_frame(st)

    fig = plt.figure(figsize=(22.0, 10.5), dpi=125)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.08, 1.0, 1.0], height_ratios=[1.0, 1.0])
    fp_ax = fig.add_subplot(gs[:, 0])
    world_topdown_ax = fig.add_subplot(gs[0, 1])
    rel_topdown_ax = fig.add_subplot(gs[0, 2])
    world_topo_ax = fig.add_subplot(gs[1, 1])
    rel_topo_ax = fig.add_subplot(gs[1, 2])

    fp_ax.imshow(rgb)
    fp_ax.axis("off")
    fp_ax.set_title("First person RGB | step {}".format(st.get("step", idx)))

    draw_topdown_background(world_topdown_ax, gt_ctx, origin_world, "world")
    draw_topo(
        world_topdown_ax,
        trace,
        idx,
        origin_world,
        frame,
        "world",
        world_limits[0],
        world_limits[1],
        "WORLD coords | GT topdown + topo",
        show_labels=False,
    )

    draw_topdown_background(rel_topdown_ax, gt_ctx, origin_world, "relative")
    draw_topo(
        rel_topdown_ax,
        trace,
        idx,
        origin_world,
        frame,
        "relative",
        rel_limits[0],
        rel_limits[1],
        "RELATIVE coords | GT topdown + topo",
        show_labels=False,
    )

    draw_topo(
        world_topo_ax,
        trace,
        idx,
        origin_world,
        frame,
        "world",
        world_limits[0],
        world_limits[1],
        "WORLD coords | DynamicTopoMap",
        show_labels=show_labels,
    )
    draw_topo(
        rel_topo_ax,
        trace,
        idx,
        origin_world,
        frame,
        "relative",
        rel_limits[0],
        rel_limits[1],
        "RELATIVE coords | DynamicTopoMap",
        show_labels=show_labels,
    )

    fig.suptitle(
        "World vs relative topo-map coordinate audit | trace frame: {} | origin_world: {}".format(
            frame,
            np.asarray(origin_world).round(4).tolist(),
        ),
        fontsize=12,
        family="monospace",
    )
    fig.tight_layout()
    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out-dir", default="data/logs/goat_topo/coord_compare")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--topdown-resolution", type=float, default=0.05)
    parser.add_argument("--navmesh-path", default=None)
    parser.add_argument("--no-gt-topdown", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    args = parser.parse_args()

    trace = json.load(open(trace_path(args.trace)))
    out_dir = trace_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    origin_world, origin_source = load_origin_world(trace)
    if origin_world is None:
        raise RuntimeError(f"Cannot compare coordinate frames without origin_world: {origin_source}")
    frame = trace_frame(trace, origin_world)
    gt_ctx = load_gt_topdown(trace, args.topdown_resolution, args.navmesh_path, args.no_gt_topdown)
    gt_ctx["origin_world"] = origin_world

    world_limits = collect_limits(trace, origin_world, frame, "world")
    rel_limits = collect_limits(trace, origin_world, frame, "relative")
    stride = max(1, args.stride)
    indices = list(range(0, len(trace.get("steps", [])), stride))
    if indices and indices[-1] != len(trace["steps"]) - 1:
        indices.append(len(trace["steps"]) - 1)

    frames = [
        draw_compare_frame(
            trace,
            idx,
            gt_ctx,
            origin_world,
            frame,
            world_limits,
            rel_limits,
            show_labels=not args.no_labels,
        )
        for idx in indices
    ]

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "world_relative_topo_compare.mp4"
    gif_path = out_dir / "world_relative_topo_compare.gif"
    final_png = out_dir / "world_relative_topo_compare_final.png"
    if frames:
        pil_frames = [Image.fromarray(frame) for frame in frames]
        for frame_idx, pil_frame in zip(indices, pil_frames):
            pil_frame.save(frames_dir / f"compare_{frame_idx:04d}.png")
        imageio.imwrite(final_png, frames[-1])
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=1)
        imageio.mimsave(gif_path, frames, duration=1.0 / max(1, args.fps))

    summary = {
        "trace": args.trace,
        "steps": len(trace.get("steps", [])),
        "frames": len(frames),
        "trace_frame": frame,
        "origin_source": origin_source,
        "origin_world": np.asarray(origin_world).round(4).tolist(),
        "world_limits": world_limits,
        "relative_limits": rel_limits,
        "max_roundtrip_planar_error": max_roundtrip_error(trace, origin_world, frame),
        "coordinate_invariance": coordinate_invariance_metrics(trace, origin_world, frame),
        "gt_topdown_available": bool(gt_ctx.get("available")),
        "gt_topdown_reason": gt_ctx.get("reason", ""),
        "video": str(video_path),
        "gif": str(gif_path),
        "final_png": str(final_png),
        "frames_dir": str(frames_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
