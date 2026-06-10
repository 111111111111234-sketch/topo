from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Optional

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
SUMMARY_COLOR = "#0f766e"
COMPRESSED_COLOR = "#64748b"
RECOVERED_COLOR = "#22c55e"
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
TEXT_LABEL_NODE_TYPES = {"object", "room", "landmark"}
MAX_TEXT_LABELS_BY_TYPE = {
    "object": 8,
    "room": 8,
    "landmark": 10,
}
NODE_TYPES = [
    "waypoint_visited",
    "waypoint_frontier",
    "waypoint_candidate",
    "object",
    "room",
    "landmark",
]
OBJECT_OBSERVED_EDGE_CLUSTER_RADIUS = 1.25


def display_node_label(node: dict) -> str:
    label = str(node.get("label") or "").strip()
    if not label:
        label = str(node.get("id", ""))
    attrs = node.get("attributes", {})
    if is_room_summary_node(node):
        contains = attrs.get("contains_labels", [])
        if contains:
            label = "{} [{}]".format(label, ",".join(str(v) for v in contains[:3]))
        label = f"{label} [S]"
    elif attrs.get("recovered_from_summary"):
        label = f"{label} [R]"
    elif attrs.get("history_compressed"):
        reason = str(attrs.get("history_compression_reason", "compressed"))
        if reason == "mid_or_far_object":
            suffix = "C:mid"
        elif reason == "far_low_confidence":
            suffix = "C:far"
        elif reason == "far_landmark":
            suffix = "C:lm"
        else:
            suffix = "C"
        label = f"{label} [{suffix}]"
    merged_count = len(node.get("attributes", {}).get("merged_from", []))
    if merged_count > 1:
        label = f"{label} (+{merged_count - 1})"
    return label


def is_room_summary_node(node: dict) -> bool:
    return node.get("type") == "room" and node.get("attributes", {}).get("summary_type") == "room_region"


def node_legend_label(node_type: str, node: dict) -> str:
    attrs = node.get("attributes", {})
    if is_room_summary_node(node):
        return "room/region summary"
    if attrs.get("recovered_from_summary") and node_type == "object":
        return "recovered object"
    if attrs.get("history_compressed") and node_type in ("object", "landmark"):
        return "compressed {}".format(LABELS[node_type])
    return LABELS[node_type]


def node_style(node_type: str, node: dict) -> dict[str, Any]:
    attrs = node.get("attributes", {})
    confidence = float(node.get("confidence", 0.5))
    style = {
        "size": 45 + 120 * confidence,
        "marker": MARKERS[node_type],
        "color": COLORS[node_type],
        "edgecolor": "black",
        "linewidth": 0.5,
        "alpha": 0.9,
        "zorder": 3,
    }
    if is_room_summary_node(node):
        style.update({
            "size": style["size"] * 1.25,
            "color": SUMMARY_COLOR,
            "edgecolor": "#0f172a",
            "linewidth": 1.2,
            "alpha": 0.84,
            "zorder": 4,
        })
    elif attrs.get("recovered_from_summary") and node_type == "object":
        style.update({
            "color": RECOVERED_COLOR,
            "edgecolor": "#14532d",
            "linewidth": 1.2,
            "alpha": 0.95,
            "zorder": 4,
        })
    elif attrs.get("history_compressed") and node_type in ("object", "landmark"):
        style.update({
            "color": COMPRESSED_COLOR,
            "edgecolor": "#334155",
            "linewidth": 0.8,
            "alpha": 0.62,
            "zorder": 2,
        })
    return style


def draw_node_group(
    ax,
    node_type: str,
    group: list[dict],
    points: list[np.ndarray],
    show_labels: bool,
    placed_labels: list[np.ndarray],
    legend_labels_seen: set[str],
) -> None:
    for rank, (node, p) in enumerate(zip(group, points)):
        style = node_style(node_type, node)
        legend_label = node_legend_label(node_type, node)
        label = legend_label if legend_label not in legend_labels_seen else None
        legend_labels_seen.add(legend_label)
        ax.scatter(
            [p[0]],
            [p[2]],
            s=style["size"],
            marker=style["marker"],
            color=style["color"],
            edgecolors=style["edgecolor"],
            linewidths=style["linewidth"],
            alpha=style["alpha"],
            label=label,
            zorder=style["zorder"],
        )
        if show_labels:
            if not should_display_node_label(node_type, node, rank):
                continue
            if label_too_close(p, placed_labels):
                continue
            placed_labels.append(np.asarray(p, dtype=np.float32))
            ax.text(float(p[0]), float(p[2]) + 0.08, display_node_label(node), fontsize=5.8, ha="center", zorder=5)


def should_display_node_label(node_type: str, node: dict, rank: int) -> bool:
    if node_type not in TEXT_LABEL_NODE_TYPES:
        return False
    return rank < MAX_TEXT_LABELS_BY_TYPE.get(node_type, 0)


def label_too_close(point: np.ndarray, placed: list[np.ndarray], min_distance: float = 0.35) -> bool:
    return any(planar_distance(point, prev) < min_distance for prev in placed)


def node_merged_ids(node: Optional[dict]) -> set[str]:
    if node is None:
        return set()
    attrs = node.get("attributes", {})
    merged = attrs.get("merged_from", [node.get("id")])
    if not isinstance(merged, list):
        merged = [merged]
    return {str(v) for v in merged if v is not None}


def best_observed_viewpoint_id(object_node: dict) -> Optional[str]:
    best = None
    best_score = -1.0
    best_step = -1
    for obs in object_node.get("attributes", {}).get("bbox_observations", []):
        viewpoint_id = obs.get("viewpoint_id")
        if viewpoint_id is None:
            continue
        score = float(obs.get("confidence", 0.0))
        step = int(obs.get("step_id", -1))
        if score > best_score or (score == best_score and step > best_step):
            best = str(viewpoint_id)
            best_score = score
            best_step = step
    return best


def topo_edge_key(edge: dict) -> tuple[str, str, str]:
    source = str(edge.get("source"))
    target = str(edge.get("target"))
    edge_type = str(edge.get("type", ""))
    return (source, target, edge_type)


def observed_at_object_and_viewpoint(edge: dict, nodes: dict[str, dict]) -> tuple[Optional[dict], Optional[dict]]:
    source = nodes.get(edge.get("source"))
    target = nodes.get(edge.get("target"))
    if source is None or target is None:
        return None, None
    if source.get("type") == "object":
        return source, target
    if target.get("type") == "object":
        return target, source
    return None, None


def observed_at_score(object_node: dict, viewpoint_node: dict) -> tuple[float, int]:
    viewpoint_ids = node_merged_ids(viewpoint_node)
    best_score = -1.0
    best_step = -1
    for obs in object_node.get("attributes", {}).get("bbox_observations", []):
        viewpoint_id = obs.get("viewpoint_id")
        if viewpoint_id is None or str(viewpoint_id) not in viewpoint_ids:
            continue
        score = float(obs.get("confidence", 0.0))
        step = int(obs.get("step_id", -1))
        if score > best_score or (score == best_score and step > best_step):
            best_score = score
            best_step = step
    if best_score >= 0.0:
        return best_score, best_step
    return float(object_node.get("confidence", 0.0)), int(object_node.get("step_id", viewpoint_node.get("step_id", -1)))


def object_observed_at_group_label(object_node: dict) -> str:
    label = str(object_node.get("label") or "").strip()
    if label:
        return label
    return str(object_node.get("id"))


def drawable_topo_edges(
    edges: list[dict],
    nodes: dict[str, dict],
    object_observed_edge_cluster_radius: float = OBJECT_OBSERVED_EDGE_CLUSTER_RADIUS,
) -> list[dict]:
    allowed_observed_at: set[tuple[str, str, str]] = set()
    object_clusters: list[dict[str, Any]] = []

    for edge in edges:
        if edge.get("type") != "observed_at":
            continue
        object_node, viewpoint_node = observed_at_object_and_viewpoint(edge, nodes)
        if object_node is None or viewpoint_node is None:
            continue
        object_label = object_observed_at_group_label(object_node)
        object_pos = node_position_array(object_node)
        score = observed_at_score(object_node, viewpoint_node)
        key = topo_edge_key(edge)

        cluster = None
        best_dist = float("inf")
        for candidate in object_clusters:
            if candidate["label"] != object_label:
                continue
            dist = planar_distance(object_pos, candidate["position"])
            if dist <= object_observed_edge_cluster_radius and dist < best_dist:
                cluster = candidate
                best_dist = dist

        if cluster is None:
            object_clusters.append({
                "label": object_label,
                "positions": [object_pos],
                "position": object_pos.copy(),
                "best_score": score,
                "best_key": key,
            })
            continue

        cluster["positions"].append(object_pos)
        cluster["position"] = np.mean(np.asarray(cluster["positions"], dtype=np.float32), axis=0).astype(np.float32)
        if score > cluster["best_score"]:
            cluster["best_score"] = score
            cluster["best_key"] = key

    for cluster in object_clusters:
        allowed_observed_at.add(cluster["best_key"])

    out = []
    for edge in edges:
        if should_draw_topo_edge(edge, nodes, allowed_observed_at):
            out.append(edge)
    return out


def should_draw_topo_edge(edge: dict, nodes: dict[str, dict], allowed_observed_at: set[tuple[str, str, str]]) -> bool:
    edge_type = edge.get("type")
    if edge_type == "navigable":
        return True

    source = nodes.get(edge.get("source"))
    target = nodes.get(edge.get("target"))
    if source is None or target is None:
        return False

    if edge_type != "observed_at":
        return edge_type == "belongs_to" and (is_room_summary_node(source) or is_room_summary_node(target))
    return topo_edge_key(edge) in allowed_observed_at


def topo_edge_style(edge: dict, source: dict, target: dict) -> dict[str, Any]:
    edge_type = edge.get("type")
    if edge_type == "navigable":
        return {"color": "#6b7280", "linewidth": 1.0, "linestyle": "-", "alpha": 0.65}
    if edge_type == "belongs_to" and (is_room_summary_node(source) or is_room_summary_node(target)):
        return {"color": SUMMARY_COLOR, "linewidth": 0.9, "linestyle": ":", "alpha": 0.62}
    return {"color": "#6b7280", "linewidth": 0.7, "linestyle": "--", "alpha": 0.65}


def trace_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


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


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()


def pad_to_even_shape(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def angle_wrap(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)


def forward_unit(heading: float) -> np.ndarray:
    return np.asarray([-math.sin(heading), -math.cos(heading)], dtype=np.float32)


def right_unit(heading: float) -> np.ndarray:
    return np.asarray([math.cos(heading), -math.sin(heading)], dtype=np.float32)


def planar_distance(a, b) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm((aa - bb)[[0, 2]]))


def step_world_positions(trace: dict, origin_world: np.ndarray) -> np.ndarray:
    positions = []
    for st in trace.get("steps", []):
        if st.get("world_position") is not None:
            positions.append(np.asarray(st["world_position"], dtype=np.float32))
        else:
            positions.append(origin_world + np.asarray(st["position"], dtype=np.float32))
    return np.asarray(positions, dtype=np.float32)


def step_headings(trace: dict) -> np.ndarray:
    return np.asarray([float(st.get("world_heading", st.get("heading", 0.0))) for st in trace.get("steps", [])], dtype=np.float32)


def compute_local_odometry(world_positions: np.ndarray, headings: np.ndarray, rng, translation_noise_std: float, heading_noise_std: float) -> list[dict[str, float]]:
    deltas = [{"delta_forward": 0.0, "delta_right": 0.0, "delta_y": 0.0, "delta_heading": 0.0}]
    for idx in range(1, len(world_positions)):
        prev_heading = float(headings[idx - 1])
        delta = world_positions[idx] - world_positions[idx - 1]
        delta_xz = np.asarray([delta[0], delta[2]], dtype=np.float32)
        delta_forward = float(np.dot(delta_xz, forward_unit(prev_heading)))
        delta_right = float(np.dot(delta_xz, right_unit(prev_heading)))
        delta_heading = angle_wrap(float(headings[idx]) - prev_heading)
        if translation_noise_std > 0.0:
            delta_forward += float(rng.normal(0.0, translation_noise_std))
            delta_right += float(rng.normal(0.0, translation_noise_std))
        if heading_noise_std > 0.0:
            delta_heading += float(rng.normal(0.0, heading_noise_std))
        deltas.append({
            "delta_forward": delta_forward,
            "delta_right": delta_right,
            "delta_y": float(delta[1]),
            "delta_heading": delta_heading,
        })
    return deltas


def integrate_odometry(deltas: list[dict[str, float]], initial_heading: float) -> tuple[np.ndarray, np.ndarray]:
    positions = [np.zeros(3, dtype=np.float32)]
    headings = [float(initial_heading)]
    for delta in deltas[1:]:
        heading = headings[-1]
        move_xz = (
            forward_unit(heading) * float(delta["delta_forward"])
            + right_unit(heading) * float(delta["delta_right"])
        )
        next_pos = positions[-1].copy()
        next_pos[0] += move_xz[0]
        next_pos[1] += float(delta["delta_y"])
        next_pos[2] += move_xz[1]
        positions.append(next_pos.astype(np.float32))
        headings.append(angle_wrap(heading + float(delta["delta_heading"])))
    return np.asarray(positions, dtype=np.float32), np.asarray(headings, dtype=np.float32)


def interpolate_angles(start: float, end: float, t: float) -> float:
    return angle_wrap(start + angle_wrap(end - start) * t)


def apply_loop_closure_history_correction(
    gt_rel: np.ndarray,
    gt_headings: np.ndarray,
    odom_positions: np.ndarray,
    odom_headings: np.ndarray,
    radius: float,
    min_interval: int,
    min_drift: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Audit-only loop closure correction.

    Closure detection uses GT proximity as an oracle. When a closure is found,
    the current pose is constrained to the previously corrected historical pose
    plus the small GT offset between the two places, then the correction is
    distributed over the segment since the previous correction anchor. This
    approximates the "rewrite history after loop closure" effect of pose graph
    optimization without changing online agent behavior.
    """
    if len(odom_positions) == 0:
        return odom_positions, odom_headings, []

    corrected_positions = odom_positions.copy()
    corrected_headings = odom_headings.copy()
    correction_events: list[dict[str, Any]] = []
    last_anchor = 0

    for idx in range(max(1, min_interval), len(odom_positions)):
        candidates = []
        for prev in range(0, idx - min_interval + 1):
            gt_dist = planar_distance(gt_rel[idx], gt_rel[prev])
            if gt_dist <= radius:
                odom_dist = planar_distance(corrected_positions[idx], corrected_positions[prev])
                candidates.append((gt_dist, -prev, prev, odom_dist))
        if not candidates:
            continue
        _, _, prev, odom_dist = min(candidates)
        drift = abs(odom_dist - planar_distance(gt_rel[idx], gt_rel[prev]))
        pose_gap = planar_distance(corrected_positions[idx], corrected_positions[prev] + (gt_rel[idx] - gt_rel[prev]))
        if max(drift, pose_gap) < min_drift:
            continue

        desired_position = corrected_positions[prev] + (gt_rel[idx] - gt_rel[prev])
        desired_heading = angle_wrap(float(corrected_headings[prev]) + angle_wrap(float(gt_headings[idx]) - float(gt_headings[prev])))
        position_delta = desired_position - corrected_positions[idx]
        heading_delta = angle_wrap(desired_heading - float(corrected_headings[idx]))
        start = max(last_anchor, prev)
        segment = max(1, idx - start)
        for k in range(start + 1, idx + 1):
            alpha = (k - start) / segment
            corrected_positions[k] = corrected_positions[k] + position_delta * alpha
            corrected_headings[k] = angle_wrap(float(corrected_headings[k]) + heading_delta * alpha)
        for k in range(idx + 1, len(corrected_positions)):
            corrected_positions[k] = corrected_positions[k] + position_delta
            corrected_headings[k] = angle_wrap(float(corrected_headings[k]) + heading_delta)
        last_anchor = idx
        correction_events.append({
            "step": int(idx),
            "matched_step": int(prev),
            "gt_distance": float(planar_distance(gt_rel[idx], gt_rel[prev])),
            "pre_correction_pose_gap": float(pose_gap),
            "pre_correction_heading_gap": float(abs(heading_delta)),
            "distributed_from_step": int(start),
        })

    return corrected_positions, corrected_headings, correction_events


def step_rgb_embeddings(trace: dict) -> np.ndarray:
    embeddings = []
    for st in trace.get("steps", []):
        emb = st.get("rgb_embedding")
        if emb is None:
            return np.empty((0, 0), dtype=np.float32)
        embeddings.append(np.asarray(emb, dtype=np.float32))
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)


def step_landmark_labels(trace: dict) -> list[set[str]]:
    labels_by_step: list[set[str]] = []
    for st in trace.get("steps", []):
        labels = {
            str(node.get("label", "")).strip()
            for node in st.get("topo", {}).get("nodes", [])
            if node.get("type") == "landmark" and str(node.get("label", "")).strip()
        }
        labels_by_step.append(labels)
    return labels_by_step


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def apply_observation_loop_closure_history_correction(
    embeddings: np.ndarray,
    odom_positions: np.ndarray,
    odom_headings: np.ndarray,
    landmark_labels_by_step: list[set[str]],
    min_interval: int,
    min_drift: float,
    similarity_threshold: float,
    max_pose_gap: float,
    cooldown_steps: int,
    min_similarity_margin: float,
    max_heading_gap: float,
    require_landmark_overlap: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Loop closure driven by current observation matching historical observations.

    This mode does not use GT pose for closure detection. It treats a high CLIP
    image-embedding match to an older observation as revisiting that old
    waypoint, then distributes the odometry correction over the segment.
    """
    if len(odom_positions) == 0 or embeddings.size == 0 or len(embeddings) != len(odom_positions):
        return odom_positions.copy(), odom_headings.copy(), []

    corrected_positions = odom_positions.copy()
    corrected_headings = odom_headings.copy()
    correction_events: list[dict[str, Any]] = []
    last_anchor = 0
    last_event_step = -max(1, cooldown_steps)
    used_matches: set[int] = set()

    for idx in range(max(1, min_interval), len(odom_positions)):
        if idx - last_event_step < cooldown_steps:
            continue
        current = embeddings[idx]
        candidates = []
        for prev in range(0, idx - min_interval + 1):
            if prev in used_matches:
                continue
            sim = cosine_similarity(current, embeddings[prev])
            if sim >= similarity_threshold:
                pose_gap = planar_distance(corrected_positions[idx], corrected_positions[prev])
                if pose_gap > max_pose_gap:
                    continue
                heading_gap = abs(angle_wrap(float(corrected_headings[idx]) - float(corrected_headings[prev])))
                if heading_gap > max_heading_gap:
                    continue
                landmark_overlap = set()
                if idx < len(landmark_labels_by_step) and prev < len(landmark_labels_by_step):
                    landmark_overlap = landmark_labels_by_step[idx] & landmark_labels_by_step[prev]
                if require_landmark_overlap and not landmark_overlap:
                    continue
                candidates.append((sim, pose_gap, heading_gap, prev, landmark_overlap))
        if not candidates:
            continue
        candidates = sorted(candidates, key=lambda row: (row[0], row[1], row[3]), reverse=True)
        sim, pose_gap, heading_gap, prev, landmark_overlap = candidates[0]
        second_sim = candidates[1][0] if len(candidates) > 1 else 0.0
        if sim - second_sim < min_similarity_margin:
            continue
        if pose_gap < min_drift:
            continue

        desired_position = corrected_positions[prev].copy()
        desired_heading = float(corrected_headings[prev])
        position_delta = desired_position - corrected_positions[idx]
        heading_delta = angle_wrap(desired_heading - float(corrected_headings[idx]))
        start = max(last_anchor, prev)
        segment = max(1, idx - start)
        for k in range(start + 1, idx + 1):
            alpha = (k - start) / segment
            corrected_positions[k] = corrected_positions[k] + position_delta * alpha
            corrected_headings[k] = angle_wrap(float(corrected_headings[k]) + heading_delta * alpha)
        for k in range(idx + 1, len(corrected_positions)):
            corrected_positions[k] = corrected_positions[k] + position_delta
            corrected_headings[k] = angle_wrap(float(corrected_headings[k]) + heading_delta)
        last_anchor = idx
        last_event_step = idx
        used_matches.add(prev)
        correction_events.append({
            "step": int(idx),
            "matched_step": int(prev),
            "observation_similarity": float(sim),
            "second_best_similarity": float(second_sim),
            "similarity_margin": float(sim - second_sim),
            "pre_correction_pose_gap": float(pose_gap),
            "observation_heading_gap": float(heading_gap),
            "landmark_overlap": sorted(landmark_overlap),
            "pre_correction_heading_gap": float(abs(heading_delta)),
            "distributed_from_step": int(start),
        })

    return corrected_positions, corrected_headings, correction_events


def load_rgb_frame(step: dict) -> np.ndarray:
    frame = step.get("rgb_frame")
    if not frame:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    frame_path = resolve_repo_path(frame)
    if frame_path is None:
        return np.zeros((360, 480, 3), dtype=np.uint8)
    rgb = imageio.imread(frame_path)
    return rgb[..., :3] if rgb.ndim == 3 and rgb.shape[-1] == 4 else rgb


def node_anchor_index(node: dict, num_steps: int, default_idx: int) -> int:
    step_id = node.get("step_id")
    if step_id is None:
        return default_idx
    return max(0, min(num_steps - 1, int(step_id) - 1))


def reproject_local_offset(point, anchor_idx: int, gt_rel: np.ndarray, gt_headings: np.ndarray, odom_positions: np.ndarray, odom_headings: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float32)
    anchor_gt = gt_rel[anchor_idx]
    offset = point - anchor_gt
    offset_xz = np.asarray([offset[0], offset[2]], dtype=np.float32)
    local_forward = float(np.dot(offset_xz, forward_unit(float(gt_headings[anchor_idx]))))
    local_right = float(np.dot(offset_xz, right_unit(float(gt_headings[anchor_idx]))))
    odom_xz = (
        forward_unit(float(odom_headings[anchor_idx])) * local_forward
        + right_unit(float(odom_headings[anchor_idx])) * local_right
    )
    out = odom_positions[anchor_idx].copy()
    out[0] += odom_xz[0]
    out[1] += offset[1]
    out[2] += odom_xz[1]
    return out.astype(np.float32)


def odom_node_position(node: dict, idx: int, gt_rel: np.ndarray, gt_headings: np.ndarray, odom_positions: np.ndarray, odom_headings: np.ndarray) -> np.ndarray:
    anchor_idx = node_anchor_index(node, len(gt_rel), idx)
    return reproject_local_offset(node["position"], anchor_idx, gt_rel, gt_headings, odom_positions, odom_headings)


def odom_target_position(target, idx: int, gt_rel: np.ndarray, gt_headings: np.ndarray, odom_positions: np.ndarray, odom_headings: np.ndarray) -> np.ndarray:
    return reproject_local_offset(target, idx, gt_rel, gt_headings, odom_positions, odom_headings)


def node_position_array(node: dict) -> np.ndarray:
    return np.asarray(node["position"], dtype=np.float32)


def project_step_topo(
    trace: dict,
    idx: int,
    gt_rel: np.ndarray,
    gt_headings: np.ndarray,
    odom_positions: np.ndarray,
    odom_headings: np.ndarray,
) -> dict[str, Any]:
    st = trace["steps"][idx]
    nodes = []
    for node in st.get("topo", {}).get("nodes", []):
        out = dict(node)
        out["position"] = odom_node_position(node, idx, gt_rel, gt_headings, odom_positions, odom_headings)
        attrs = dict(out.get("attributes", {}))
        attrs.setdefault("merged_from", [node["id"]])
        out["attributes"] = attrs
        nodes.append(out)
    return {
        "nodes": nodes,
        "edges": [dict(edge) for edge in st.get("topo", {}).get("edges", [])],
    }


def project_step_topo_frame(
    trace: dict,
    idx: int,
    mode: str,
    origin_world: np.ndarray,
    gt_rel: np.ndarray,
    gt_headings: np.ndarray,
    odom_positions: np.ndarray,
    odom_headings: np.ndarray,
) -> dict[str, Any]:
    st = trace["steps"][idx]
    nodes = []
    for node in st.get("topo", {}).get("nodes", []):
        out = dict(node)
        if mode == "world":
            out["position"] = origin_world + np.asarray(node["position"], dtype=np.float32)
        elif mode == "relative":
            out["position"] = np.asarray(node["position"], dtype=np.float32)
        elif mode == "odometry":
            out["position"] = odom_node_position(node, idx, gt_rel, gt_headings, odom_positions, odom_headings)
        else:
            raise ValueError(f"Unsupported topo projection mode: {mode}")
        attrs = dict(out.get("attributes", {}))
        attrs.setdefault("merged_from", [node["id"]])
        out["attributes"] = attrs
        nodes.append(out)
    return {
        "nodes": nodes,
        "edges": [dict(edge) for edge in st.get("topo", {}).get("edges", [])],
    }


def project_target_frame(
    target,
    idx: int,
    mode: str,
    origin_world: np.ndarray,
    gt_rel: np.ndarray,
    gt_headings: np.ndarray,
    odom_positions: np.ndarray,
    odom_headings: np.ndarray,
) -> Optional[np.ndarray]:
    if target is None:
        return None
    if mode == "world":
        return origin_world + np.asarray(target, dtype=np.float32)
    if mode == "relative":
        return np.asarray(target, dtype=np.float32)
    if mode == "odometry":
        return odom_target_position(target, idx, gt_rel, gt_headings, odom_positions, odom_headings)
    raise ValueError(f"Unsupported target projection mode: {mode}")


def mergeable_radius(node_type: str, node: dict, cluster: dict, visited_radius: float, room_radius: float, frontier_radius: float) -> Optional[float]:
    if node_type == "waypoint_visited":
        return visited_radius
    if node_type == "room":
        if node.get("label") != cluster["label"]:
            return None
        return room_radius
    if node_type in ("waypoint_frontier", "waypoint_candidate"):
        return frontier_radius
    return None


def merge_projected_topo(projected: dict[str, Any], visited_radius: float, room_radius: float, frontier_radius: float) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}

    for node in projected.get("nodes", []):
        node_type = node.get("type")
        node_pos = node_position_array(node)
        best_cluster = None
        best_distance = float("inf")
        for cluster in clusters:
            if cluster["type"] != node_type:
                continue
            radius = mergeable_radius(node_type, node, cluster, visited_radius, room_radius, frontier_radius)
            if radius is None:
                continue
            dist = planar_distance(node_pos, cluster["position"])
            if dist <= radius and dist < best_distance:
                best_cluster = cluster
                best_distance = dist
        if best_cluster is None:
            attrs = dict(node.get("attributes", {}))
            merged_from = list(attrs.get("merged_from", [node["id"]]))
            best_cluster = {
                "id": node["id"],
                "type": node_type,
                "label": node.get("label"),
                "positions": [node_pos],
                "position": node_pos.copy(),
                "confidence": float(node.get("confidence", 0.5)),
                "attributes": attrs,
                "merged_from": merged_from,
                "template": dict(node),
            }
            clusters.append(best_cluster)
        else:
            best_cluster["positions"].append(node_pos)
            best_cluster["position"] = np.mean(np.asarray(best_cluster["positions"], dtype=np.float32), axis=0).astype(np.float32)
            best_cluster["confidence"] = max(best_cluster["confidence"], float(node.get("confidence", 0.5)))
            attrs = dict(node.get("attributes", {}))
            best_cluster["merged_from"].extend(attrs.get("merged_from", [node["id"]]))
        id_map[node["id"]] = best_cluster["id"]

    merged_nodes = []
    for cluster in clusters:
        template = dict(cluster["template"])
        attrs = dict(template.get("attributes", {}))
        merged_from = sorted(set(cluster["merged_from"]))
        attrs["merged_from"] = merged_from
        template["id"] = cluster["id"]
        template["type"] = cluster["type"]
        template["label"] = cluster["label"]
        template["position"] = cluster["position"]
        template["confidence"] = cluster["confidence"]
        template["attributes"] = attrs
        merged_nodes.append(template)

    edge_keys = set()
    merged_edges = []
    for edge in projected.get("edges", []):
        source = id_map.get(edge.get("source"))
        target = id_map.get(edge.get("target"))
        if source is None or target is None or source == target:
            continue
        key = (source, target, edge.get("type", ""))
        reverse_key = (target, source, edge.get("type", ""))
        if key in edge_keys or reverse_key in edge_keys:
            continue
        out = dict(edge)
        out["source"] = source
        out["target"] = target
        edge_keys.add(key)
        merged_edges.append(out)

    return {"nodes": merged_nodes, "edges": merged_edges, "id_map": id_map}


def compute_merge_stats(corrected_topos: list[dict[str, Any]], merged_topos: list[dict[str, Any]]) -> dict[str, Any]:
    if not corrected_topos or not merged_topos:
        return {
            "final_node_count_before": 0,
            "final_node_count_after": 0,
            "final_edge_count_before": 0,
            "final_edge_count_after": 0,
            "merged_node_count_by_type": {},
        }
    before = corrected_topos[-1]
    after = merged_topos[-1]
    before_counts = {node_type: 0 for node_type in NODE_TYPES}
    after_counts = {node_type: 0 for node_type in NODE_TYPES}
    for node in before.get("nodes", []):
        before_counts[node.get("type")] = before_counts.get(node.get("type"), 0) + 1
    for node in after.get("nodes", []):
        after_counts[node.get("type")] = after_counts.get(node.get("type"), 0) + 1
    merged_by_type = {
        node_type: max(0, before_counts.get(node_type, 0) - after_counts.get(node_type, 0))
        for node_type in NODE_TYPES
    }
    return {
        "final_node_count_before": len(before.get("nodes", [])),
        "final_node_count_after": len(after.get("nodes", [])),
        "final_edge_count_before": len(before.get("edges", [])),
        "final_edge_count_after": len(after.get("edges", [])),
        "merged_node_count_by_type": merged_by_type,
    }


def topo_memory_visual_counts(topo: dict[str, Any]) -> dict[str, int]:
    nodes = topo.get("nodes", [])
    return {
        "room_summaries": sum(1 for node in nodes if is_room_summary_node(node)),
        "compressed_nodes": sum(1 for node in nodes if node.get("attributes", {}).get("history_compressed")),
        "recovered_objects": sum(
            1
            for node in nodes
            if node.get("type") == "object" and node.get("attributes", {}).get("recovered_from_summary")
        ),
    }


def draw_topdown_background(ax, gt_ctx: dict, origin_world: np.ndarray) -> None:
    if not gt_ctx.get("available"):
        ax.axis("off")
        ax.text(0.5, 0.5, "GT topdown unavailable\n{}".format(gt_ctx.get("reason", "")), ha="center", va="center", fontsize=9)
        return
    bounds_min = gt_ctx["bounds_min"]
    bounds_max = gt_ctx["bounds_max"]
    extent = [bounds_min[0], bounds_max[0], bounds_max[2], bounds_min[2]]
    ax.imshow(gt_ctx["topdown"], cmap="gray", origin="upper", extent=extent, alpha=0.58)


def draw_topdown_background_relative(ax, gt_ctx: dict, origin_world: np.ndarray) -> None:
    if not gt_ctx.get("available"):
        ax.axis("off")
        ax.text(0.5, 0.5, "GT topdown unavailable\n{}".format(gt_ctx.get("reason", "")), ha="center", va="center", fontsize=9)
        return
    bounds_min = gt_ctx["bounds_min"]
    bounds_max = gt_ctx["bounds_max"]
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
    mode: str,
    origin_world: np.ndarray,
    gt_rel: np.ndarray,
    gt_headings: np.ndarray,
    odom_positions: np.ndarray,
    odom_headings: np.ndarray,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    title: str,
    show_labels: bool,
    object_observed_edge_cluster_radius: float = OBJECT_OBSERVED_EDGE_CLUSTER_RADIUS,
) -> None:
    st = trace["steps"][idx]
    nodes = {node["id"]: node for node in st.get("topo", {}).get("nodes", [])}
    placed_labels: list[np.ndarray] = []
    legend_labels_seen: set[str] = set()

    def pos(value, node=None):
        if mode == "world":
            return origin_world + np.asarray(value, dtype=np.float32)
        if mode == "relative":
            return np.asarray(value, dtype=np.float32)
        if node is not None:
            return odom_node_position(node, idx, gt_rel, gt_headings, odom_positions, odom_headings)
        return odom_target_position(value, idx, gt_rel, gt_headings, odom_positions, odom_headings)

    for edge in drawable_topo_edges(
        st.get("topo", {}).get("edges", []),
        nodes,
        object_observed_edge_cluster_radius,
    ):
        a = nodes.get(edge.get("source"))
        b = nodes.get(edge.get("target"))
        if a is None or b is None:
            continue
        pa = pos(a["position"], a)
        pb = pos(b["position"], b)
        style = topo_edge_style(edge, a, b)
        ax.plot([pa[0], pb[0]], [pa[2], pb[2]], zorder=1, **style)

    for node_type in NODE_TYPES:
        group = [node for node in nodes.values() if node.get("type") == node_type]
        if not group:
            continue
        group = sorted(group, key=lambda node: float(node.get("confidence", 0.0)), reverse=True)
        points = [pos(node["position"], node) for node in group]
        draw_node_group(ax, node_type, group, points, show_labels, placed_labels, legend_labels_seen)

    if mode == "world":
        path = origin_world + gt_rel[: idx + 1]
        cur = origin_world + gt_rel[idx]
    elif mode == "relative":
        path = gt_rel[: idx + 1]
        cur = gt_rel[idx]
    else:
        path = odom_positions[: idx + 1]
        cur = odom_positions[idx]
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 2], color="#111827", linewidth=1.35, alpha=0.85, label="agent path", zorder=2)
    ax.scatter([cur[0]], [cur[2]], marker="*", s=180, color="#facc15", edgecolors="black", linewidths=0.9, label="agent", zorder=5)
    if st.get("target_position") is not None:
        target = pos(st["target_position"])
        ax.scatter([target[0]], [target[2]], marker="X", s=105, color="#ef4444", edgecolors="black", linewidths=0.7, label="target", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("world x" if mode == "world" else "{} x".format(mode))
    ax.set_ylabel("world z" if mode == "world" else "{} z".format(mode))
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=5.8)


def draw_projected_topo(
    ax,
    topo: dict[str, Any],
    path: np.ndarray,
    cur: np.ndarray,
    target: Optional[np.ndarray],
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    title: str,
    show_labels: bool,
    axis_label: str,
    show_path: bool = False,
    object_observed_edge_cluster_radius: float = OBJECT_OBSERVED_EDGE_CLUSTER_RADIUS,
) -> None:
    nodes = {node["id"]: node for node in topo.get("nodes", [])}
    placed_labels: list[np.ndarray] = []
    legend_labels_seen: set[str] = set()
    for edge in drawable_topo_edges(
        topo.get("edges", []),
        nodes,
        object_observed_edge_cluster_radius,
    ):
        a = nodes.get(edge.get("source"))
        b = nodes.get(edge.get("target"))
        if a is None or b is None:
            continue
        pa = node_position_array(a)
        pb = node_position_array(b)
        style = topo_edge_style(edge, a, b)
        ax.plot([pa[0], pb[0]], [pa[2], pb[2]], zorder=1, **style)

    for node_type in NODE_TYPES:
        group = [node for node in nodes.values() if node.get("type") == node_type]
        if not group:
            continue
        group = sorted(group, key=lambda node: float(node.get("confidence", 0.0)), reverse=True)
        points = [node_position_array(node) for node in group]
        draw_node_group(ax, node_type, group, points, show_labels, placed_labels, legend_labels_seen)

    if show_path and len(path) > 1:
        ax.plot(path[:, 0], path[:, 2], color="#111827", linewidth=1.35, alpha=0.85, label="agent path", zorder=2)
    ax.scatter([cur[0]], [cur[2]], marker="*", s=180, color="#facc15", edgecolors="black", linewidths=0.9, label="agent", zorder=5)
    if target is not None:
        ax.scatter([target[0]], [target[2]], marker="X", s=105, color="#ef4444", edgecolors="black", linewidths=0.7, label="target", zorder=4)

    ax.set_title(title)
    ax.set_xlabel(f"{axis_label} x")
    ax.set_ylabel(f"{axis_label} z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=5.8)


def collect_limits(trace: dict, origin_world: np.ndarray, gt_rel: np.ndarray, gt_headings: np.ndarray, odom_positions: np.ndarray, odom_headings: np.ndarray, mode: str):
    xs: list[float] = []
    zs: list[float] = []
    for idx, st in enumerate(trace.get("steps", [])):
        if mode == "world":
            points = [origin_world + gt_rel[idx]]
        elif mode == "relative":
            points = [gt_rel[idx]]
        else:
            points = [odom_positions[idx]]
        if st.get("target_position") is not None:
            if mode == "world":
                points.append(origin_world + np.asarray(st["target_position"], dtype=np.float32))
            elif mode == "relative":
                points.append(np.asarray(st["target_position"], dtype=np.float32))
            else:
                points.append(odom_target_position(st["target_position"], idx, gt_rel, gt_headings, odom_positions, odom_headings))
        for node in st.get("topo", {}).get("nodes", []):
            if mode == "world":
                points.append(origin_world + np.asarray(node["position"], dtype=np.float32))
            elif mode == "relative":
                points.append(np.asarray(node["position"], dtype=np.float32))
            else:
                points.append(odom_node_position(node, idx, gt_rel, gt_headings, odom_positions, odom_headings))
        for p in points:
            xs.append(float(p[0]))
            zs.append(float(p[2]))
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    return (min(xs) - 1.0, max(xs) + 1.0), (min(zs) - 1.0, max(zs) + 1.0)


def collect_projected_limits(topos: list[dict[str, Any]], positions: np.ndarray, targets: list[Optional[np.ndarray]]):
    xs: list[float] = []
    zs: list[float] = []
    for idx, topo in enumerate(topos):
        points = [positions[idx]]
        if idx < len(targets) and targets[idx] is not None:
            points.append(targets[idx])
        points.extend(node_position_array(node) for node in topo.get("nodes", []))
        for p in points:
            xs.append(float(p[0]))
            zs.append(float(p[2]))
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    return (min(xs) - 1.0, max(xs) + 1.0), (min(zs) - 1.0, max(zs) + 1.0)


def combine_limits(*limits: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = min(limit[0][0] for limit in limits)
    x_max = max(limit[0][1] for limit in limits)
    z_min = min(limit[1][0] for limit in limits)
    z_max = max(limit[1][1] for limit in limits)
    return (x_min, x_max), (z_min, z_max)


def compute_metrics(trace: dict, gt_rel: np.ndarray, gt_headings: np.ndarray, odom_positions: np.ndarray, odom_headings: np.ndarray) -> dict[str, float]:
    pose_drifts = [planar_distance(gt, odom) for gt, odom in zip(gt_rel, odom_positions)]
    heading_drifts = [abs(angle_wrap(float(odom) - float(gt))) for gt, odom in zip(gt_headings, odom_headings)]
    node_sq_errors: list[float] = []
    edge_deltas: list[float] = []
    agent_node_deltas: list[float] = []
    for idx, st in enumerate(trace.get("steps", [])):
        nodes = st.get("topo", {}).get("nodes", [])
        by_id = {node["id"]: node for node in nodes}
        gt_nodes = {node["id"]: np.asarray(node["position"], dtype=np.float32) for node in nodes}
        odom_nodes = {node["id"]: odom_node_position(node, idx, gt_rel, gt_headings, odom_positions, odom_headings) for node in nodes}
        for node_id in by_id:
            err = planar_distance(gt_nodes[node_id], odom_nodes[node_id])
            node_sq_errors.append(err * err)
            gt_dist = planar_distance(gt_rel[idx], gt_nodes[node_id])
            odom_dist = planar_distance(odom_positions[idx], odom_nodes[node_id])
            agent_node_deltas.append(abs(gt_dist - odom_dist))
        for edge in st.get("topo", {}).get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source not in by_id or target not in by_id:
                continue
            gt_len = planar_distance(gt_nodes[source], gt_nodes[target])
            odom_len = planar_distance(odom_nodes[source], odom_nodes[target])
            edge_deltas.append(abs(gt_len - odom_len))
    return {
        "pose_drift_final": pose_drifts[-1] if pose_drifts else 0.0,
        "pose_drift_max": max(pose_drifts) if pose_drifts else 0.0,
        "heading_drift_final": heading_drifts[-1] if heading_drifts else 0.0,
        "heading_drift_max": max(heading_drifts) if heading_drifts else 0.0,
        "topo_node_position_rmse": float(math.sqrt(sum(node_sq_errors) / len(node_sq_errors))) if node_sq_errors else 0.0,
        "topo_edge_length_delta_max": max(edge_deltas) if edge_deltas else 0.0,
        "agent_node_distance_delta_max": max(agent_node_deltas) if agent_node_deltas else 0.0,
    }


def compute_projected_topo_metrics(
    trace: dict,
    gt_rel: np.ndarray,
    gt_headings: np.ndarray,
    positions: np.ndarray,
    headings: np.ndarray,
    projected_topos: list[dict[str, Any]],
) -> dict[str, float]:
    pose_drifts = [planar_distance(gt, odom) for gt, odom in zip(gt_rel, positions)]
    heading_drifts = [abs(angle_wrap(float(odom) - float(gt))) for gt, odom in zip(gt_headings, headings)]
    node_sq_errors: list[float] = []
    edge_deltas: list[float] = []
    agent_node_deltas: list[float] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []

    for idx, topo in enumerate(projected_topos):
        st = trace["steps"][idx]
        original_nodes = {node["id"]: node for node in st.get("topo", {}).get("nodes", [])}
        node_counts.append(len(topo.get("nodes", [])))
        edge_counts.append(len(topo.get("edges", [])))
        for node in topo.get("nodes", []):
            merged_from = node.get("attributes", {}).get("merged_from", [node["id"]])
            gt_points = [
                np.asarray(original_nodes[node_id]["position"], dtype=np.float32)
                for node_id in merged_from
                if node_id in original_nodes
            ]
            if not gt_points:
                continue
            gt_center = np.mean(np.asarray(gt_points, dtype=np.float32), axis=0)
            node_pos = node_position_array(node)
            err = planar_distance(gt_center, node_pos)
            node_sq_errors.append(err * err)
            gt_dist = planar_distance(gt_rel[idx], gt_center)
            odom_dist = planar_distance(positions[idx], node_pos)
            agent_node_deltas.append(abs(gt_dist - odom_dist))

        projected_nodes = {node["id"]: node for node in topo.get("nodes", [])}
        for edge in topo.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source not in projected_nodes or target not in projected_nodes:
                continue
            source_from = projected_nodes[source].get("attributes", {}).get("merged_from", [source])
            target_from = projected_nodes[target].get("attributes", {}).get("merged_from", [target])
            source_gt = [
                np.asarray(original_nodes[node_id]["position"], dtype=np.float32)
                for node_id in source_from
                if node_id in original_nodes
            ]
            target_gt = [
                np.asarray(original_nodes[node_id]["position"], dtype=np.float32)
                for node_id in target_from
                if node_id in original_nodes
            ]
            if not source_gt or not target_gt:
                continue
            gt_len = planar_distance(np.mean(np.asarray(source_gt, dtype=np.float32), axis=0), np.mean(np.asarray(target_gt, dtype=np.float32), axis=0))
            projected_len = planar_distance(projected_nodes[source]["position"], projected_nodes[target]["position"])
            edge_deltas.append(abs(gt_len - projected_len))

    return {
        "pose_drift_final": pose_drifts[-1] if pose_drifts else 0.0,
        "pose_drift_max": max(pose_drifts) if pose_drifts else 0.0,
        "heading_drift_final": heading_drifts[-1] if heading_drifts else 0.0,
        "heading_drift_max": max(heading_drifts) if heading_drifts else 0.0,
        "topo_node_position_rmse": float(math.sqrt(sum(node_sq_errors) / len(node_sq_errors))) if node_sq_errors else 0.0,
        "topo_edge_length_delta_max": max(edge_deltas) if edge_deltas else 0.0,
        "agent_node_distance_delta_max": max(agent_node_deltas) if agent_node_deltas else 0.0,
        "node_count_final": node_counts[-1] if node_counts else 0,
        "edge_count_final": edge_counts[-1] if edge_counts else 0,
    }


def format_step_goal(trace: dict[str, Any], idx: int) -> str:
    st = trace["steps"][idx]
    task_index = st.get("task_index")
    goal = st.get("goal") or trace.get("current_goal") or {}
    if not isinstance(goal, dict):
        goal = {}
    target_object = goal.get("target_object") or goal.get("goal") or "n/a"
    room_prior = goal.get("room_prior") or []
    landmarks = goal.get("landmarks") or []
    if not isinstance(room_prior, list):
        room_prior = [room_prior]
    if not isinstance(landmarks, list):
        landmarks = [landmarks]
    room_text = ",".join(str(v) for v in room_prior[:3] if v) or "n/a"
    landmark_text = ",".join(str(v) for v in landmarks[:3] if v) or "n/a"
    if task_index is None:
        return "goal object: {} | room: {} | landmarks: {}".format(target_object, room_text, landmark_text)
    task_count = len(trace.get("task_summaries", []))
    task_text = "task {}/{}".format(int(task_index) + 1, task_count) if task_count else "task {}".format(int(task_index) + 1)
    return "{} | goal object: {} | room: {} | landmarks: {}".format(task_text, target_object, room_text, landmark_text)


def draw_frame(
    trace,
    idx,
    rgb,
    gt_ctx,
    origin_world,
    gt_rel,
    world_topos,
    world_targets,
    gt_step_topos,
    gt_step_targets,
    gt_step_positions,
    corrected_topos,
    corrected_targets,
    raw_topos,
    raw_targets,
    odom_positions,
    corrected_positions,
    limits,
    metrics,
    corrected_metrics,
    corrected_structural_metrics,
    correction_events,
    structural_merge_enabled,
    show_labels,
    object_observed_edge_cluster_radius,
):
    fig = plt.figure(figsize=(18.0, 10.5), dpi=125)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0], height_ratios=[1.0, 1.0])
    fp_ax = fig.add_subplot(gs[:, 0])
    world_ax = fig.add_subplot(gs[0, 1])
    gt_ax = fig.add_subplot(gs[0, 2])
    raw_ax = fig.add_subplot(gs[1, 1])
    corrected_ax = fig.add_subplot(gs[1, 2])

    fp_ax.imshow(rgb)
    fp_ax.axis("off")
    goal_text = format_step_goal(trace, idx)
    fp_ax.set_title("First person RGB | step {} | {}".format(trace["steps"][idx].get("step", idx), goal_text), fontsize=9)

    draw_topdown_background_relative(world_ax, gt_ctx, origin_world)
    draw_projected_topo(
        world_ax,
        gt_step_topos[idx],
        gt_step_positions[: idx + 1],
        gt_step_positions[idx],
        gt_step_targets[idx],
        limits["shared"][0],
        limits["shared"][1],
        "GT topdown | structural topo",
        False,
        "relative",
        object_observed_edge_cluster_radius=object_observed_edge_cluster_radius,
    )
    draw_projected_topo(
        gt_ax,
        gt_step_topos[idx],
        gt_step_positions[: idx + 1],
        gt_step_positions[idx],
        gt_step_targets[idx],
        limits["shared"][0],
        limits["shared"][1],
        "GT step-relative structural topo",
        show_labels,
        "relative",
        object_observed_edge_cluster_radius=object_observed_edge_cluster_radius,
    )
    draw_projected_topo(
        raw_ax,
        raw_topos[idx],
        odom_positions[: idx + 1],
        odom_positions[idx],
        raw_targets[idx],
        limits["shared"][0],
        limits["shared"][1],
        "Raw noisy step-odometry structural topo",
        show_labels,
        "odometry",
        object_observed_edge_cluster_radius=object_observed_edge_cluster_radius,
    )
    draw_projected_topo(
        corrected_ax,
        corrected_topos[idx],
        corrected_positions[: idx + 1],
        corrected_positions[idx],
        corrected_targets[idx],
        limits["shared"][0],
        limits["shared"][1],
        "Online-corrected structural topo",
        show_labels,
        "odometry",
        object_observed_edge_cluster_radius=object_observed_edge_cluster_radius,
    )

    merge_text = "on" if structural_merge_enabled else "off"
    memory_counts = topo_memory_visual_counts(corrected_topos[idx])
    fig.suptitle(
        "Step odometry audit | {} | target: {} | action: {} | raw drift: {:.3f}/{:.3f}m | corrected: {:.3f}/{:.3f}m | closures: {} | structural merge: {} | corrected nodes: {} | summaries: {} | compressed: {} | recovered: {}".format(
            goal_text,
            trace["steps"][idx].get("target_node_id") or "n/a",
            trace["steps"][idx].get("low_action") or "n/a",
            metrics["pose_drift_final"],
            metrics["pose_drift_max"],
            corrected_metrics["pose_drift_final"],
            corrected_metrics["pose_drift_max"],
            sum(1 for event in correction_events if event["step"] <= idx),
            merge_text,
            int(corrected_structural_metrics.get("node_count_final", 0)),
            memory_counts["room_summaries"],
            memory_counts["compressed_nodes"],
            memory_counts["recovered_objects"],
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
    parser.add_argument("--out-dir", default="data/logs/goat_topo/step_odometry_compare")
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
    parser.add_argument("--observation-loop-max-heading-gap", type=float, default=math.pi)
    parser.add_argument("--observation-loop-require-landmark-overlap", action="store_true")
    parser.add_argument("--merge-structural-topo", dest="merge_corrected_topo", action="store_true", default=True)
    parser.add_argument("--no-merge-structural-topo", dest="merge_corrected_topo", action="store_false")
    parser.add_argument("--merge-corrected-topo", dest="merge_corrected_topo", action="store_true")
    parser.add_argument("--no-merge-corrected-topo", dest="merge_corrected_topo", action="store_false")
    parser.add_argument("--visited-merge-radius", type=float, default=0.75)
    parser.add_argument("--room-merge-radius", type=float, default=2.0)
    parser.add_argument("--frontier-merge-radius", type=float, default=1.5)
    parser.add_argument("--object-observed-edge-cluster-radius", type=float, default=OBJECT_OBSERVED_EDGE_CLUSTER_RADIUS)
    args = parser.parse_args()

    trace = json.load(open(trace_path(args.trace)))
    out_dir = trace_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    origin_world, origin_source = load_origin_world(trace)
    if origin_world is None:
        raise RuntimeError(f"Cannot audit odometry without origin_world: {origin_source}")

    world_positions = step_world_positions(trace, origin_world)
    world_headings = step_headings(trace)
    gt_rel = world_positions - origin_world
    rng = np.random.default_rng(args.seed)
    heading_noise_std = math.radians(float(args.heading_noise_std_deg))
    gt_step_deltas = compute_local_odometry(world_positions, world_headings, np.random.default_rng(args.seed), 0.0, 0.0)
    gt_step_positions, gt_step_headings = integrate_odometry(gt_step_deltas, float(world_headings[0]) if len(world_headings) else 0.0)
    odom_deltas = compute_local_odometry(world_positions, world_headings, rng, args.translation_noise_std, heading_noise_std)
    odom_positions, odom_headings = integrate_odometry(odom_deltas, float(world_headings[0]) if len(world_headings) else 0.0)
    metrics = compute_metrics(trace, gt_rel, world_headings, odom_positions, odom_headings)
    observation_embeddings = step_rgb_embeddings(trace)
    landmark_labels_by_step = step_landmark_labels(trace)
    if args.history_correction == "gt_loop_closure":
        corrected_positions, corrected_headings, correction_events = apply_loop_closure_history_correction(
            gt_rel,
            world_headings,
            odom_positions,
            odom_headings,
            radius=args.loop_closure_radius,
            min_interval=args.loop_closure_min_interval,
            min_drift=args.loop_closure_min_drift,
        )
    elif args.history_correction == "observation_loop_closure":
        corrected_positions, corrected_headings, correction_events = apply_observation_loop_closure_history_correction(
            observation_embeddings,
            odom_positions,
            odom_headings,
            landmark_labels_by_step,
            min_interval=args.loop_closure_min_interval,
            min_drift=args.loop_closure_min_drift,
            similarity_threshold=args.observation_loop_sim_threshold,
            max_pose_gap=args.observation_loop_max_pose_gap,
            cooldown_steps=args.observation_loop_cooldown,
            min_similarity_margin=args.observation_loop_min_sim_margin,
            max_heading_gap=args.observation_loop_max_heading_gap,
            require_landmark_overlap=args.observation_loop_require_landmark_overlap,
        )
    else:
        corrected_positions = odom_positions.copy()
        corrected_headings = odom_headings.copy()
        correction_events = []
    corrected_metrics = compute_metrics(trace, gt_rel, world_headings, corrected_positions, corrected_headings)

    raw_gt_step_topos = [
        project_step_topo(trace, idx, gt_rel, world_headings, gt_step_positions, gt_step_headings)
        for idx in range(len(trace.get("steps", [])))
    ]
    raw_odom_topos = [
        project_step_topo(trace, idx, gt_rel, world_headings, odom_positions, odom_headings)
        for idx in range(len(trace.get("steps", [])))
    ]
    raw_corrected_topos = [
        project_step_topo(trace, idx, gt_rel, world_headings, corrected_positions, corrected_headings)
        for idx in range(len(trace.get("steps", [])))
    ]
    raw_world_topos = [
        project_step_topo_frame(trace, idx, "world", origin_world, gt_rel, world_headings, odom_positions, odom_headings)
        for idx in range(len(trace.get("steps", [])))
    ]

    def apply_structural_merge(topos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not args.merge_corrected_topo:
            return topos
        return [
            merge_projected_topo(
                topo,
                visited_radius=args.visited_merge_radius,
                room_radius=args.room_merge_radius,
                frontier_radius=args.frontier_merge_radius,
            )
            for topo in topos
        ]

    gt_step_topos = apply_structural_merge(raw_gt_step_topos)
    odom_topos = apply_structural_merge(raw_odom_topos)
    corrected_topos = apply_structural_merge(raw_corrected_topos)
    world_topos = apply_structural_merge(raw_world_topos)
    gt_step_metrics = compute_projected_topo_metrics(
        trace,
        gt_rel,
        world_headings,
        gt_step_positions,
        gt_step_headings,
        gt_step_topos,
    )
    raw_structural_metrics = compute_projected_topo_metrics(
        trace,
        gt_rel,
        world_headings,
        odom_positions,
        odom_headings,
        odom_topos,
    )
    corrected_structural_metrics = compute_projected_topo_metrics(
        trace,
        gt_rel,
        world_headings,
        corrected_positions,
        corrected_headings,
        corrected_topos,
    )
    merge_stats = {
        "world": compute_merge_stats(raw_world_topos, world_topos),
        "gt_step": compute_merge_stats(raw_gt_step_topos, gt_step_topos),
        "raw_odometry": compute_merge_stats(raw_odom_topos, odom_topos),
        "corrected": compute_merge_stats(raw_corrected_topos, corrected_topos),
    }
    memory_visual_counts = {
        "world": topo_memory_visual_counts(world_topos[-1]) if world_topos else {},
        "gt_step": topo_memory_visual_counts(gt_step_topos[-1]) if gt_step_topos else {},
        "raw_odometry": topo_memory_visual_counts(odom_topos[-1]) if odom_topos else {},
        "corrected": topo_memory_visual_counts(corrected_topos[-1]) if corrected_topos else {},
    }
    gt_ctx = load_gt_topdown(trace, args.topdown_resolution, args.navmesh_path, args.no_gt_topdown)
    world_targets = [
        project_target_frame(st.get("target_position"), idx, "world", origin_world, gt_rel, world_headings, odom_positions, odom_headings)
        for idx, st in enumerate(trace.get("steps", []))
    ]
    gt_step_targets = [
        None if st.get("target_position") is None else odom_target_position(st["target_position"], idx, gt_rel, world_headings, gt_step_positions, gt_step_headings)
        for idx, st in enumerate(trace.get("steps", []))
    ]
    odom_targets = [
        None if st.get("target_position") is None else odom_target_position(st["target_position"], idx, gt_rel, world_headings, odom_positions, odom_headings)
        for idx, st in enumerate(trace.get("steps", []))
    ]
    corrected_targets = [
        None if st.get("target_position") is None else odom_target_position(st["target_position"], idx, gt_rel, world_headings, corrected_positions, corrected_headings)
        for idx, st in enumerate(trace.get("steps", []))
    ]

    limits = {
        "world": collect_projected_limits(world_topos, origin_world + gt_rel, world_targets),
        "gt_step": collect_projected_limits(gt_step_topos, gt_step_positions, gt_step_targets),
        "raw": collect_projected_limits(odom_topos, odom_positions, odom_targets),
        "corrected": collect_projected_limits(corrected_topos, corrected_positions, corrected_targets),
    }
    limits["shared"] = combine_limits(limits["gt_step"], limits["raw"], limits["corrected"])
    stride = max(1, args.stride)
    indices = list(range(0, len(trace.get("steps", [])), stride))
    if indices and indices[-1] != len(trace["steps"]) - 1:
        indices.append(len(trace["steps"]) - 1)

    frames = []
    for idx in indices:
        frames.append(
            draw_frame(
                trace,
                idx,
                load_rgb_frame(trace["steps"][idx]),
                gt_ctx,
                origin_world,
                gt_rel,
                world_topos,
                world_targets,
                gt_step_topos,
                gt_step_targets,
                gt_step_positions,
                corrected_topos,
                corrected_targets,
                odom_topos,
                odom_targets,
                odom_positions,
                corrected_positions,
                limits,
                metrics,
                corrected_metrics,
                corrected_structural_metrics,
                correction_events,
                structural_merge_enabled=bool(args.merge_corrected_topo),
                show_labels=not args.no_labels,
                object_observed_edge_cluster_radius=float(args.object_observed_edge_cluster_radius),
            )
        )

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "step_odometry_compare.mp4"
    gif_path = out_dir / "step_odometry_compare.gif"
    final_png = out_dir / "step_odometry_compare_final.png"
    if frames:
        pil_frames = [Image.fromarray(frame) for frame in frames]
        for frame_idx, pil_frame in zip(indices, pil_frames):
            pil_frame.save(frames_dir / f"odometry_{frame_idx:04d}.png")
        imageio.imwrite(final_png, frames[-1])
        video_frames = [pad_to_even_shape(frame) for frame in frames]
        imageio.mimsave(video_path, video_frames, fps=args.fps, macro_block_size=1)
        imageio.mimsave(gif_path, frames, duration=1.0 / max(1, args.fps))

    summary: dict[str, Any] = {
        "trace": args.trace,
        "steps": len(trace.get("steps", [])),
        "frames": len(frames),
        "origin_source": origin_source,
        "origin_world": np.asarray(origin_world).round(4).tolist(),
        "pose_sources": trace.get("pose_sources", {}),
        "gt_step_relative_model": "agent_local_delta_integrated_from_gt_world_pose_without_noise",
        "odometry_model": "agent_local_delta_integrated_from_gt_world_pose_with_optional_noise",
        "translation_noise_std": float(args.translation_noise_std),
        "heading_noise_std_deg": float(args.heading_noise_std_deg),
        "seed": int(args.seed),
        "history_correction": {
            "mode": args.history_correction,
            "loop_closure_radius": float(args.loop_closure_radius),
            "loop_closure_min_interval": int(args.loop_closure_min_interval),
            "loop_closure_min_drift": float(args.loop_closure_min_drift),
            "observation_loop_sim_threshold": float(args.observation_loop_sim_threshold),
            "observation_loop_max_pose_gap": float(args.observation_loop_max_pose_gap),
            "observation_loop_cooldown": int(args.observation_loop_cooldown),
            "observation_loop_min_sim_margin": float(args.observation_loop_min_sim_margin),
            "observation_loop_max_heading_gap": float(args.observation_loop_max_heading_gap),
            "observation_loop_require_landmark_overlap": bool(args.observation_loop_require_landmark_overlap),
            "observation_embeddings_available": bool(observation_embeddings.size > 0),
            "events": correction_events,
            "event_count": len(correction_events),
        },
        "metrics": metrics,
        "corrected_metrics": corrected_metrics,
        "gt_step_relative_metrics": gt_step_metrics,
        "raw_structural_metrics": raw_structural_metrics,
        "corrected_structural_metrics": corrected_structural_metrics,
        "merged_corrected_metrics": corrected_structural_metrics,
        "merge_stats": {
            "enabled": bool(args.merge_corrected_topo),
            "visited_merge_radius": float(args.visited_merge_radius),
            "room_merge_radius": float(args.room_merge_radius),
            "frontier_merge_radius": float(args.frontier_merge_radius),
            "object_observed_edge_cluster_radius": float(args.object_observed_edge_cluster_radius),
            "applied_to": ["gt_step_relative", "raw_odometry", "corrected"],
            **merge_stats,
        },
        "memory_visual_counts": memory_visual_counts,
        "gt_topdown_available": bool(gt_ctx.get("available")),
        "gt_topdown_reason": gt_ctx.get("reason", ""),
        "video": str(video_path),
        "gif": str(gif_path),
        "final_png": str(final_png),
        "frames_dir": str(frames_dir),
        "odometry_deltas_preview": odom_deltas[:5],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
