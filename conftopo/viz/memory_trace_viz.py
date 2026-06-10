"""Phase3 dynamic memory visualization utilities.

Provides node styling, filtering, GT topdown background, and panel drawing.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.patches import Circle

from conftopo.core.gt_room_regions import (
    _try_load_habitat_semantic_regions,
    room_for_waypoint_gt,
    rooms_from_gt_regions,
)
from conftopo.core.region_rooms import (
    normalize_region_label,
    room_for_labeled_waypoint,
    rooms_from_waypoint_clusters,
)

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "waypoint_visited": "#2563eb",
    "waypoint_frontier": "#f97316",
    "waypoint_candidate": "#06b6d4",
    "object": "#16a34a",
    "room": "#dc2626",
    "landmark": "#9333ea",
}
SUMMARY_COLOR = "#0f766e"
CLIP_ROOM_COLOR = "#f97316"
COMPRESSED_COLOR = "#64748b"
RECOVERED_COLOR = "#22c55e"
ENV_LANDMARK_COLOR = "#c4b5fd"
UNCERTAIN_OBJECT_COLOR = "#86efac"
OBJECT_UNCERTAINTY_RADIUS = 0.55

MARKERS = {
    "waypoint_visited": "o",
    "waypoint_frontier": "^",
    "waypoint_candidate": "v",
    "object": "D",
    "room": "s",
    "landmark": "P",
    "compressed": "h",
}
LABELS = {
    "waypoint_visited": "visited waypoint",
    "waypoint_frontier": "frontier",
    "waypoint_candidate": "candidate",
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


def is_room_summary_node(node: dict) -> bool:
    return (node.get("type") == "room"
            and node.get("attributes", {}).get("summary_type") == "room_region")


def is_clip_room_node(node: dict) -> bool:
    return node.get("type") == "room" and not is_room_summary_node(node)


def is_environment_landmark(node: dict) -> bool:
    if node.get("type") != "landmark":
        return False
    attrs = node.get("attributes") or {}
    return attrs.get("landmark_source") == "environment" or attrs.get("source") == "environment"


def is_navigation_landmark(node: dict) -> bool:
    return node.get("type") == "landmark" and not is_environment_landmark(node)


def is_structure_anchor_landmark(node: dict) -> bool:
    if not is_navigation_landmark(node):
        return False
    attrs = node.get("attributes") or {}
    return attrs.get("structure_role") == "portal"


def is_portal_landmark(node: dict) -> bool:
    return is_structure_anchor_landmark(node)


def is_persistent_structure_node(node: dict) -> bool:
    """Bottom-layer room/landmark anchors that persist across waypoint motion."""
    if is_room_summary_node(node):
        return True
    if not is_navigation_landmark(node):
        return False
    attrs = node.get("attributes") or {}
    source = attrs.get("landmark_source", "")
    return source in ("goal_hint", "promoted_object") or bool(attrs.get("promoted_from_object"))


def node_distance_to_agent(node: dict, agent_pos: np.ndarray) -> float:
    pos = np.asarray(node.get("position", [0, 0, 0]), dtype=np.float32)
    return float(np.linalg.norm(pos - agent_pos))


def node_distance_tier(
    node: dict,
    agent_pos: np.ndarray,
    near_radius: float = 3.0,
    far_radius: float = 10.0,
) -> Optional[str]:
    """Return display tier: near (object), mid (landmark), far (room summary only)."""
    if node.get("type") in ("waypoint_visited", "waypoint_frontier", "waypoint_candidate"):
        return "waypoint"
    if is_room_summary_node(node):
        return "room"
    attrs = node.get("attributes") or {}
    if attrs.get("folded") and not attrs.get("recovered_from_summary"):
        return None
    if is_environment_landmark(node) or is_clip_room_node(node):
        return None

    dist = node_distance_to_agent(node, agent_pos)
    gran = attrs.get("granularity", "object" if node.get("type") == "object" else "landmark")
    ntype = node.get("type")

    if dist <= near_radius:
        if ntype == "object" and (gran in ("", "object") or attrs.get("recovered_from_summary")):
            return "near"
        if is_navigation_landmark(node):
            return "near"
    elif dist <= far_radius:
        if is_navigation_landmark(node) or attrs.get("promoted_from_object"):
            return "mid"
        if ntype == "object" and gran == "landmark":
            return "mid"
    return None


def is_bbox_object(node: dict) -> bool:
    if node.get("type") != "object":
        return False
    attrs = node.get("attributes") or {}
    return bool(attrs.get("bbox_observations")) or attrs.get("source") in ("groundingdino", "heavy")


def node_style(node_type: str, node: dict) -> dict:
    attrs = node.get("attributes", {})
    confidence = float(node.get("confidence", 0.5))
    style = {
        "size": 45 + 120 * confidence,
        "marker": MARKERS.get(node_type, "o"),
        "color": COLORS.get(node_type, "#888888"),
        "edgecolor": "black",
        "linewidth": 0.5,
        "alpha": 0.9,
        "zorder": 3,
    }
    if is_room_summary_node(node):
        style.update({
            "size": style["size"] * 1.35,
            "marker": "s",
            "color": SUMMARY_COLOR,
            "edgecolor": "#0f172a",
            "linewidth": 1.2,
            "alpha": 0.88,
            "zorder": 4,
        })
    elif is_clip_room_node(node):
        style.update({
            "size": 28,
            "marker": "o",
            "color": "none",
            "edgecolor": CLIP_ROOM_COLOR,
            "linewidth": 1.0,
            "alpha": 0.75,
            "zorder": 2,
        })
    elif is_environment_landmark(node):
        style.update({
            "size": 18,
            "marker": ".",
            "color": ENV_LANDMARK_COLOR,
            "edgecolor": "none",
            "linewidth": 0.0,
            "alpha": 0.35,
            "zorder": 1,
        })
    elif attrs.get("landmark_source") == "promoted_object" or attrs.get("promoted_from_object"):
        style.update({
            "size": style["size"] * 1.15,
            "color": "#7c3aed",
            "edgecolor": "#4c1d95",
            "linewidth": 1.0,
            "alpha": 0.92,
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
    elif is_bbox_object(node) and not attrs.get("recovered_from_summary"):
        style.update({
            "color": UNCERTAIN_OBJECT_COLOR,
            "edgecolor": "#166534",
            "linewidth": 0.9,
            "alpha": 0.9,
            "zorder": 4,
        })
    elif attrs.get("history_compressed") and node_type in ("object", "landmark"):
        style.update({
            "marker": MARKERS["compressed"],
            "color": COMPRESSED_COLOR,
            "edgecolor": "#334155",
            "linewidth": 0.8,
            "alpha": 0.62,
            "zorder": 2,
        })
    elif node_type == "waypoint_visited":
        role = attrs.get("waypoint_role", "")
        if role == "entrance":
            style.update({
                "marker": "D",
                "color": "#f59e0b",
                "edgecolor": "#92400e",
                "linewidth": 1.1,
                "size": style["size"] * 1.1,
                "zorder": 6,
            })
        elif role == "corridor_anchor":
            style.update({
                "marker": "o",
                "color": "#1d4ed8",
                "edgecolor": "#1e3a8a",
                "linewidth": 0.9,
                "size": max(35.0, style["size"] * 0.75),
                "alpha": 0.82,
                "zorder": 5,
            })
    return style


def display_node_label(node: dict) -> str:
    label = str(node.get("label") or "").strip()
    if not label:
        label = str(node.get("id", ""))
    attrs = node.get("attributes", {})
    if is_room_summary_node(node):
        contains = attrs.get("contains_labels", [])
        if contains:
            label = "{} [{}]".format(label, ",".join(str(v) for v in contains[:4]))
        label = f"{label} [S]"
    elif attrs.get("structure_role") == "portal":
        pair_labels = attrs.get("structure_pair_labels")
        if attrs.get("synthetic_portal") and pair_labels and len(pair_labels) == 2:
            label = f"{label}({pair_labels[0]}↔{pair_labels[1]})"
        label = f"{label} [P]"
    elif attrs.get("recovered_from_summary"):
        label = f"{label} [R]"
    elif attrs.get("folded"):
        label = f"{label} [F]"
    elif attrs.get("history_compressed"):
        reason = str(attrs.get("history_compression_reason", "C"))
        label = f"{label} [{reason[:3]}]"
    return label


def granularity_counts(nodes: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {"object": 0, "landmark": 0, "room_level": 0}
    for n in nodes:
        if n.get("type") not in ("object", "landmark"):
            continue
        g = (n.get("attributes") or {}).get("granularity", "none")
        counts[g] = counts.get(g, 0) + 1
    return counts


def panel_node_counts(nodes: List[dict]) -> Dict[str, int]:
    """Count nodes actually drawn in a viz panel."""
    counts = {
        "obj": 0,
        "summary": 0,
        "room_tag": 0,
        "landmark": 0,
        "env_lm": 0,
    }
    for n in nodes:
        ntype = n.get("type")
        if ntype == "object":
            counts["obj"] += 1
        elif is_room_summary_node(n):
            counts["summary"] += 1
        elif is_clip_room_node(n):
            counts["room_tag"] += 1
        elif ntype == "landmark":
            if is_environment_landmark(n):
                counts["env_lm"] += 1
            else:
                counts["landmark"] += 1
    return counts


def _format_panel_counts(label: str, counts: Dict[str, int]) -> str:
    parts = [f"obj={counts['obj']}", f"sum={counts['summary']}"]
    if counts["room_tag"]:
        parts.append(f"room={counts['room_tag']}")
    if counts["landmark"]:
        parts.append(f"lm={counts['landmark']}")
    if counts["env_lm"]:
        parts.append(f"env_lm={counts['env_lm']}")
    return f"{label}({','.join(parts)})"


def memory_panel_title(
    step: int,
    nodes: List[dict],
    goal_text: str = "",
    heavy_reason: str = "",
    gt_nodes: Optional[List[dict]] = None,
    memory_nodes: Optional[List[dict]] = None,
) -> str:
    gc = granularity_counts(nodes)
    folded = sum(1 for n in nodes if (n.get("attributes") or {}).get("folded"))
    summaries = sum(1 for n in nodes if is_room_summary_node(n))
    compressed = sum(1 for n in nodes if (n.get("attributes") or {}).get("history_compressed"))
    recovered = sum(1 for n in nodes if (n.get("attributes") or {}).get("recovered_from_summary"))
    parts = [
        f"Step {step}",
        goal_text or "",
        f"mem: obj={gc.get('object', 0)} lm={gc.get('landmark', 0)} rl={gc.get('room_level', 0)}",
        f"folded={folded} summaries={summaries} compressed={compressed} recovered={recovered}",
    ]
    if gt_nodes is not None:
        parts.append(_format_panel_counts("gt", panel_node_counts(gt_nodes)))
    if memory_nodes is not None:
        parts.append(_format_panel_counts("shown", panel_node_counts(memory_nodes)))
    if heavy_reason:
        parts.append(f"heavy: {heavy_reason}")
    return " | ".join(p for p in parts if p)


def make_geo_validator(
    gt_ctx: Optional[dict],
    origin_world: Optional[np.ndarray],
    snap_tolerance: float = 2.5,
) -> Optional[Callable[[np.ndarray], bool]]:
    """Return a callable that checks whether a relative position is geo-plausible.

    Bbox-estimated objects often sit 1-2m off the navmesh slice; ``snap_tolerance``
    allows those near the walkable surface to still render on the GT panel.
    """
    if not gt_ctx or not gt_ctx.get("available") or origin_world is None:
        return None
    pathfinder = gt_ctx.get("pathfinder")
    if pathfinder is None:
        return None
    origin = np.asarray(origin_world, dtype=np.float32)
    bounds_min = np.asarray(gt_ctx["bounds_min"], dtype=np.float32)
    bounds_max = np.asarray(gt_ctx["bounds_max"], dtype=np.float32)

    def is_geo_valid(rel_pos: np.ndarray) -> bool:
        world = origin + np.asarray(rel_pos, dtype=np.float32)
        if not (bounds_min[0] <= world[0] <= bounds_max[0] and bounds_min[2] <= world[2] <= bounds_max[2]):
            return False
        try:
            if bool(pathfinder.is_navigable(world)):
                return True
            snap = np.asarray(pathfinder.snap_point(world), dtype=np.float32)
            planar = float(np.linalg.norm((snap - world)[[0, 2]]))
            return planar <= snap_tolerance
        except Exception:
            return True

    return is_geo_valid


def _passes_geo(node: dict, geo_validator: Optional[Callable[[np.ndarray], bool]]) -> bool:
    if geo_validator is None:
        return True
    if node.get("type") not in ("object", "room", "landmark"):
        return True
    pos = np.asarray(node.get("position", [0, 0, 0]), dtype=np.float32)
    return geo_validator(pos)


def filter_topo_nodes(
    nodes: List[dict],
    mode: str,
    agent_pos: Optional[np.ndarray] = None,
    near_radius: float = 3.0,
    far_radius: float = 10.0,
    geo_validator: Optional[Callable[[np.ndarray], bool]] = None,
) -> List[dict]:
    """Filter nodes for a specific viz panel mode.

    Modes:
      full          - all non-folded semantic nodes + waypoints
      gt_aligned    - GT overlay: waypoints, room summaries, geo-valid objects
      memory        - dynamic memory: objects, summaries, non-env landmarks
      distance_layers - near=object, mid=landmark, far=room summary only
      two_layer       - bottom: all room/lm structure; top: waypoints; near objects
      spatial_structure - persistent base layer only (room + nav lm, no waypoints)
      object_only   - object granularity nodes near agent
      summary_only  - room_region summaries only
      near_only     - nodes within near_radius
    """
    if mode == "full":
        out = []
        for n in nodes:
            if n.get("type") in ("waypoint_visited", "waypoint_frontier", "waypoint_candidate"):
                out.append(n)
                continue
            if is_room_summary_node(n):
                out.append(n)
                continue
            if not (n.get("attributes") or {}).get("folded"):
                out.append(n)
        return out
    elif mode == "gt_aligned":
        out = []
        for n in nodes:
            ntype = n.get("type")
            if ntype in ("waypoint_visited", "waypoint_frontier", "waypoint_candidate"):
                out.append(n)
                continue
            if is_room_summary_node(n):
                out.append(n)
                continue
            if ntype == "object" and not (n.get("attributes") or {}).get("folded"):
                if _passes_geo(n, geo_validator):
                    out.append(n)
        return out
    elif mode == "memory":
        out = []
        for n in nodes:
            ntype = n.get("type")
            if ntype in ("waypoint_visited", "waypoint_frontier", "waypoint_candidate"):
                out.append(n)
                continue
            if (n.get("attributes") or {}).get("folded"):
                continue
            if is_room_summary_node(n) or is_clip_room_node(n):
                out.append(n)
                continue
            if ntype == "object":
                out.append(n)
                continue
            if ntype == "landmark" and not is_environment_landmark(n):
                out.append(n)
        return out
    elif mode == "two_layer":
        if agent_pos is None:
            return filter_topo_nodes(nodes, "memory", agent_pos, near_radius, far_radius, geo_validator)
        out = []
        for n in nodes:
            ntype = n.get("type")
            attrs = n.get("attributes") or {}
            if ntype in ("waypoint_visited", "waypoint_frontier", "waypoint_candidate"):
                out.append(n)
                continue
            if attrs.get("folded") and not attrs.get("recovered_from_summary"):
                continue
            if is_persistent_structure_node(n):
                out.append(n)
                continue
            if ntype == "object":
                tier = node_distance_tier(n, agent_pos, near_radius, far_radius)
                if tier != "near":
                    continue
                if not _passes_geo(n, geo_validator):
                    continue
                out.append(n)
        return out
    elif mode == "spatial_structure":
        out = []
        for n in nodes:
            ntype = n.get("type")
            if ntype in ("waypoint_visited", "waypoint_frontier", "waypoint_candidate"):
                continue
            if is_room_summary_node(n):
                out.append(n)
                continue
            if is_structure_anchor_landmark(n):
                out.append(n)
        return out
    elif mode == "distance_layers":
        if agent_pos is None:
            return filter_topo_nodes(nodes, "memory", agent_pos, near_radius, far_radius, geo_validator)
        out = []
        for n in nodes:
            tier = node_distance_tier(n, agent_pos, near_radius, far_radius)
            if tier is None:
                continue
            if tier == "waypoint":
                out.append(n)
                continue
            if tier == "room":
                out.append(n)
                continue
            if is_persistent_structure_node(n):
                out.append(n)
                continue
            if tier in ("near", "mid") and n.get("type") == "object" and not _passes_geo(n, geo_validator):
                continue
            out.append(n)
        return out
    elif mode == "memory_with_env":
        out = filter_topo_nodes(nodes, "memory", agent_pos, near_radius, geo_validator)
        seen = {n["id"] for n in out}
        for n in nodes:
            if n["id"] in seen:
                continue
            if is_environment_landmark(n) and not (n.get("attributes") or {}).get("folded"):
                out.append(n)
                seen.add(n["id"])
        return out
    elif mode == "object_only":
        return [
            n for n in nodes
            if n.get("type") == "object"
            and (n.get("attributes") or {}).get("granularity") == "object"
            and not (n.get("attributes") or {}).get("folded")
        ]
    elif mode == "summary_only":
        return [n for n in nodes if is_room_summary_node(n)]
    elif mode == "near_only":
        if agent_pos is None:
            return nodes
        out = []
        for n in nodes:
            pos = np.asarray(n.get("position", [0, 0, 0]), dtype=np.float32)
            dist = float(np.linalg.norm(pos - agent_pos))
            if n.get("type") in ("waypoint_visited", "waypoint_frontier", "waypoint_candidate"):
                if dist <= near_radius * 2:
                    out.append(n)
            elif not (n.get("attributes") or {}).get("folded") and dist <= near_radius:
                out.append(n)
        return out
    return nodes


_PORTAL_LABEL_PRIORITY = (
    "door", "hallway", "entrance", "corridor", "arch", "gate", "opening",
)


def _portal_label_score(label: str) -> int:
    lowered = str(label or "").strip().lower()
    for idx, token in enumerate(_PORTAL_LABEL_PRIORITY):
        if token in lowered:
            return idx
    return len(_PORTAL_LABEL_PRIORITY)


def _filter_summary_contains_near_room_dict(
    room: dict,
    summary_radius: float,
) -> None:
    attrs = room.setdefault("attributes", {})
    room_pos = np.asarray(room["position"], dtype=np.float32)
    near_labels: List[str] = []
    label_counts: Dict[str, int] = {}
    for obs in attrs.get("summary_observations", []):
        pos = np.asarray(obs.get("position", room_pos), dtype=np.float32)
        if float(np.linalg.norm(pos - room_pos)) > summary_radius:
            continue
        label = str(obs.get("label") or "").strip()
        if not label:
            continue
        key = label.lower()
        label_counts[key] = label_counts.get(key, 0) + 1
        if key not in {v.lower() for v in near_labels}:
            near_labels.append(label)
    if near_labels:
        attrs["contains_labels"] = near_labels[:8]
        attrs["label_counts"] = label_counts


def _pair_portal_node_id_dict(room_a_id: str, room_b_id: str) -> str:
    pair = tuple(sorted((room_a_id, room_b_id)))
    return f"portal::{pair[0]}::{pair[1]}"


def _room_for_transition_dict(
    position: np.ndarray,
    rooms: List[dict],
    summary_radius: float = 5.0,
    waypoint: Optional[dict] = None,
    nodes: Optional[List[dict]] = None,
    edges: Optional[List[dict]] = None,
) -> Optional[dict]:
    if waypoint is not None:
        label_match = room_for_labeled_waypoint(
            waypoint, rooms, nodes=nodes, edges=edges,
        )
        if label_match is not None:
            return label_match
        attrs = waypoint.get("attributes") or {}
        if normalize_region_label(attrs.get("view_room_label")):
            return None
    pos = np.asarray(position, dtype=np.float32)
    strict = [
        room for room in rooms
        if float(np.linalg.norm(np.asarray(room["position"], dtype=np.float32) - pos)) <= summary_radius
    ]
    if len(strict) == 1:
        return strict[0]
    if strict:
        return min(
            strict,
            key=lambda room: float(
                np.linalg.norm(np.asarray(room["position"], dtype=np.float32) - pos)
            ),
        )
    boundary = [
        room for room in rooms
        if float(np.linalg.norm(np.asarray(room["position"], dtype=np.float32) - pos))
        <= summary_radius * 1.35
    ]
    if boundary:
        return min(
            boundary,
            key=lambda room: float(
                np.linalg.norm(np.asarray(room["position"], dtype=np.float32) - pos)
            ),
        )
    return None


def _pick_portal_landmark_dict(
    midpoint: np.ndarray,
    landmarks: List[dict],
    max_dist: float,
    room_a: Optional[dict] = None,
    room_b: Optional[dict] = None,
) -> Optional[dict]:
    pos = np.asarray(midpoint, dtype=np.float32)
    candidates = [
        landmark for landmark in landmarks
        if float(np.linalg.norm(np.asarray(landmark["position"], dtype=np.float32) - pos)) <= max_dist
    ]
    if not candidates:
        return None

    def _betweenness(landmark: dict) -> float:
        if room_a is None or room_b is None:
            return 0.0
        ap = np.asarray(room_a["position"], dtype=np.float32)[[0, 2]]
        bp = np.asarray(room_b["position"], dtype=np.float32)[[0, 2]]
        lp = np.asarray(landmark["position"], dtype=np.float32)[[0, 2]]
        ab = bp - ap
        denom = float(np.dot(ab, ab))
        if denom < 1e-6:
            return float(np.linalg.norm(lp - ap))
        t = float(np.clip(np.dot(lp - ap, ab) / denom, 0.0, 1.0))
        proj = ap + t * ab
        return float(np.linalg.norm(lp - proj))

    return min(
        candidates,
        key=lambda landmark: (
            len((landmark.get("attributes") or {}).get("structure_room_pairs", [])),
            _portal_label_score(str(landmark.get("label", ""))),
            _betweenness(landmark),
            float(np.linalg.norm(np.asarray(landmark["position"], dtype=np.float32) - pos)),
            -float(landmark.get("confidence", 0.0)),
        ),
    )


def _nearest_room_summary_for_pos(
    position: np.ndarray,
    rooms: List[dict],
    summary_radius: float = 5.0,
) -> Optional[dict]:
    if not rooms:
        return None
    pos = np.asarray(position, dtype=np.float32)
    expanded = summary_radius * 1.8
    candidates = [
        room for room in rooms
        if float(np.linalg.norm(np.asarray(room["position"], dtype=np.float32) - pos)) <= expanded
    ]
    pool = candidates if candidates else rooms
    return min(
        pool,
        key=lambda room: float(
            np.linalg.norm(np.asarray(room["position"], dtype=np.float32) - pos)
        ),
    )


def _clone_nodes_for_skeleton(nodes: List[dict]) -> List[dict]:
    cloned = []
    for node in nodes:
        attrs = dict(node.get("attributes") or {})
        attrs.pop("structure_anchor", None)
        attrs.pop("structure_role", None)
        attrs.pop("structure_room_pairs", None)
        cloned.append({**node, "attributes": attrs})
    return cloned


def _nav_landmark_nodes(nodes: List[dict]) -> List[dict]:
    return [n for n in nodes if is_navigation_landmark(n)]


def _nearest_nav_landmark_dict(
    position: np.ndarray,
    landmarks: List[dict],
    max_dist: float,
) -> Optional[dict]:
    pos = np.asarray(position, dtype=np.float32)
    best = None
    best_dist = float(max_dist)
    for landmark in landmarks:
        lp = np.asarray(landmark["position"], dtype=np.float32)
        dist = float(np.linalg.norm(lp - pos))
        if dist < best_dist:
            best = landmark
            best_dist = dist
    return best


def _landmarks_for_room_dict(
    room: dict,
    nav_landmarks: List[dict],
    edges: List[dict],
    summary_radius: float,
) -> List[dict]:
    room_id = room["id"]
    belongs_ids = {
        e["source"] if e["target"] == room_id else e["target"]
        for e in edges
        if e.get("type") == "belongs_to"
        and room_id in (e.get("source"), e.get("target"))
    }
    members = [lm for lm in nav_landmarks if lm["id"] in belongs_ids]
    if members:
        return members
    room_pos = np.asarray(room["position"], dtype=np.float32)
    return [
        lm for lm in nav_landmarks
        if float(np.linalg.norm(np.asarray(lm["position"], dtype=np.float32) - room_pos)) <= summary_radius
    ]


def _structure_rooms_for_nodes(
    nodes: List[dict],
    edges: List[dict],
    summary_radius: float,
    origin_world: Optional[np.ndarray] = None,
    scene_file: Optional[str] = None,
    gt_regions: Optional[list] = None,
) -> Tuple[List[dict], Optional[list]]:
    object_rooms = [n for n in nodes if is_room_summary_node(n)]
    if origin_world is not None and scene_file:
        regions = gt_regions
        if regions is None:
            regions = _try_load_habitat_semantic_regions(scene_file, origin_world=origin_world)
        if regions:
            rooms = rooms_from_gt_regions(
                nodes, origin_world,
                existing_summaries=object_rooms,
                scene_file=scene_file,
            )
            if rooms:
                return rooms, regions
    return rooms_from_waypoint_clusters(
        nodes, existing_summaries=object_rooms, edges=edges,
    ), None


def _room_for_structure_waypoint(
    waypoint: dict,
    rooms: List[dict],
    summary_radius: float,
    nodes: List[dict],
    edges: List[dict],
    gt_regions: Optional[list] = None,
) -> Optional[dict]:
    if gt_regions:
        return room_for_waypoint_gt(waypoint, rooms, gt_regions)
    return _room_for_transition_dict(
        waypoint["position"], rooms, summary_radius,
        waypoint=waypoint, nodes=nodes, edges=edges,
    )


def _traversable_room_transitions_dict(
    nodes: List[dict],
    edges: List[dict],
    summary_radius: float,
    origin_world: Optional[np.ndarray] = None,
    scene_file: Optional[str] = None,
    gt_regions: Optional[list] = None,
) -> List[Tuple[dict, dict, dict, dict, float]]:
    """Room pairs crossed by navigable waypoint segments the agent actually walked."""
    rooms, gt_regions = _structure_rooms_for_nodes(
        nodes, edges, summary_radius, origin_world, scene_file, gt_regions,
    )
    waypoints = {
        n["id"]: n for n in nodes if n.get("type") == "waypoint_visited"
    }
    transitions: List[Tuple[dict, dict, dict, dict, float]] = []
    seen_pairs = set()
    for edge in edges:
        if edge.get("type") != "navigable":
            continue
        src_id = edge.get("source")
        tgt_id = edge.get("target")
        if src_id not in waypoints or tgt_id not in waypoints:
            continue
        wp_a = waypoints[src_id]
        wp_b = waypoints[tgt_id]
        room_a = _room_for_structure_waypoint(
            wp_a, rooms, summary_radius, nodes, edges, gt_regions,
        )
        room_b = _room_for_structure_waypoint(
            wp_b, rooms, summary_radius, nodes, edges, gt_regions,
        )
        if room_a is None or room_b is None or room_a["id"] == room_b["id"]:
            continue
        pair_key = tuple(sorted((room_a["id"], room_b["id"])))
        seg_dist = float(
            np.linalg.norm(
                np.asarray(wp_a["position"], dtype=np.float32)
                - np.asarray(wp_b["position"], dtype=np.float32)
            )
        )
        if pair_key in seen_pairs:
            for idx, item in enumerate(transitions):
                if tuple(sorted((item[0]["id"], item[1]["id"]))) == pair_key:
                    if seg_dist < item[4]:
                        transitions[idx] = (room_a, room_b, wp_a, wp_b, seg_dist)
                    break
            continue
        seen_pairs.add(pair_key)
        transitions.append((room_a, room_b, wp_a, wp_b, seg_dist))
    return transitions


def _mark_structure_anchors(
    nodes: List[dict],
    edges: List[dict],
    summary_radius: float = 5.0,
    merge_radius: float = 1.0,
) -> None:
    rooms = [n for n in nodes if is_room_summary_node(n)]
    nav_landmarks = _nav_landmark_nodes(nodes)
    if not rooms:
        return

    portal_max = max(merge_radius * 5.0, summary_radius * 1.2)
    for room_a, room_b, wp_a, wp_b, _ in _traversable_room_transitions_dict(
        nodes, edges, summary_radius,
    ):
        midpoint = (
            np.asarray(wp_a["position"], dtype=np.float32)
            + np.asarray(wp_b["position"], dtype=np.float32)
        ) * 0.5
        portal = _pick_portal_landmark_dict(
            midpoint, nav_landmarks, portal_max, room_a=room_a, room_b=room_b,
        )
        if portal is None:
            continue
        portal["attributes"]["structure_anchor"] = True
        portal["attributes"]["structure_role"] = "portal"
        pair_key = tuple(sorted((room_a["id"], room_b["id"])))
        portal_pairs = portal["attributes"].setdefault("structure_room_pairs", [])
        if list(pair_key) not in portal_pairs and pair_key not in portal_pairs:
            portal_pairs.append(list(pair_key))


def build_spatial_skeleton(
    nodes: List[dict],
    edges: List[dict],
    summary_radius: float = 5.0,
    room_link_max_distance: float = 12.0,
    merge_radius: float = 1.0,
    origin_world: Optional[np.ndarray] = None,
    scene_file: Optional[str] = None,
    gt_regions: Optional[list] = None,
) -> Tuple[List[dict], List[dict]]:
    """Build canonical room→portal→room chains from walked navigable segments."""
    del room_link_max_distance
    working_nodes = _clone_nodes_for_skeleton(nodes)
    working_edges = list(edges)

    rooms, gt_regions = _structure_rooms_for_nodes(
        working_nodes, working_edges, summary_radius, origin_world, scene_file, gt_regions,
    )
    for room in rooms:
        if (room.get("attributes") or {}).get("region_source") != "habitat_semantic":
            _filter_summary_contains_near_room_dict(room, summary_radius)
    nav_landmarks = _nav_landmark_nodes(working_nodes)
    portal_ids: set = set()
    seen_pairs: set = set()
    skeleton_edges: List[dict] = []
    edge_seen: set = set()

    def _append_traversable(src_id: str, tgt_id: str, weight: float) -> None:
        key = tuple(sorted((src_id, tgt_id))) + ("adjacent_to",)
        if key in edge_seen or src_id == tgt_id:
            return
        skeleton_edges.append({
            "source": src_id,
            "target": tgt_id,
            "type": "adjacent_to",
            "weight": float(weight),
            "traversable": True,
        })
        edge_seen.add(key)

    portal_max = max(merge_radius * 5.0, summary_radius * 1.2)
    transitions = sorted(
        _traversable_room_transitions_dict(
            working_nodes, working_edges, summary_radius,
            origin_world=origin_world, scene_file=scene_file, gt_regions=gt_regions,
        ),
        key=lambda item: tuple(sorted((item[0]["id"], item[1]["id"]))),
    )
    for room_a, room_b, wp_a, wp_b, seg_dist in transitions:
        pair_key = tuple(sorted((room_a["id"], room_b["id"])))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        midpoint = (
            np.asarray(wp_a["position"], dtype=np.float32)
            + np.asarray(wp_b["position"], dtype=np.float32)
        ) * 0.5
        ref_portal = _pick_portal_landmark_dict(
            midpoint, nav_landmarks, portal_max, room_a=room_a, room_b=room_b,
        )
        portal_id = _pair_portal_node_id_dict(room_a["id"], room_b["id"])
        portal_label = str(ref_portal.get("label") or "passage") if ref_portal else "passage"
        portal_node = {
            "id": portal_id,
            "type": "landmark",
            "position": midpoint.tolist(),
            "confidence": float(ref_portal.get("confidence", 0.55)) if ref_portal else 0.55,
            "label": portal_label,
            "attributes": {
                "structure_anchor": True,
                "structure_role": "portal",
                "synthetic_portal": True,
                "structure_room_pairs": [list(pair_key)],
                "portal_ref": ref_portal["id"] if ref_portal else None,
                "structure_pair_labels": [
                    str(room_a.get("label") or room_a["id"]),
                    str(room_b.get("label") or room_b["id"]),
                ],
            },
        }
        working_nodes.append(portal_node)
        portal_ids.add(portal_id)
        pa = np.asarray(room_a["position"], dtype=np.float32)
        pb = np.asarray(room_b["position"], dtype=np.float32)
        pp = np.asarray(portal_node["position"], dtype=np.float32)
        _append_traversable(room_a["id"], portal_id, float(np.linalg.norm(pa - pp)))
        _append_traversable(portal_id, room_b["id"], float(np.linalg.norm(pp - pb)))

    node_by_id = {n["id"]: n for n in working_nodes}
    skeleton_nodes = list(rooms) + [
        node_by_id[pid] for pid in portal_ids if pid in node_by_id
    ]
    return skeleton_nodes, skeleton_edges


def augment_spatial_structure_edges(
    nodes: List[dict],
    edges: List[dict],
    summary_radius: float = 5.0,
    room_link_max_distance: float = 12.0,
) -> List[dict]:
    """Backward-compatible wrapper returning skeleton edges only."""
    _, skeleton_edges = build_spatial_skeleton(
        nodes, edges,
        summary_radius=summary_radius,
        room_link_max_distance=room_link_max_distance,
    )
    return skeleton_edges


def compute_trace_limits(
    trace: dict,
    margin: float = 2.0,
    mode: str = "trajectory",
    max_node_distance: float = 15.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute axis limits for map panels.

    Default ``trajectory`` uses only agent path positions so bbox outliers
    (e.g. z=-50) do not shrink the map to a tiny corner.
    """
    traj_x, traj_z = [], []
    for st in trace.get("steps", []):
        pos = st.get("position") or st.get("world_position")
        if pos:
            traj_x.append(float(pos[0]))
            traj_z.append(float(pos[2]))

    if not traj_x:
        return (-10.0, 10.0), (-10.0, 10.0)

    xs = list(traj_x)
    zs = list(traj_z)

    if mode == "all":
        for st in trace.get("steps", []):
            for n in st.get("topo", {}).get("nodes", []):
                p = n.get("position")
                if p:
                    xs.append(float(p[0]))
                    zs.append(float(p[2]))
    elif mode == "trajectory_plus_near":
        cx = float(np.mean(traj_x))
        cz = float(np.mean(traj_z))
        traj_span = max(max(traj_x) - min(traj_x), max(traj_z) - min(traj_z), 1.0)
        keep_radius = max(max_node_distance, traj_span * 1.5 + margin)
        last_nodes = trace["steps"][-1].get("topo", {}).get("nodes", []) if trace.get("steps") else []
        for n in last_nodes:
            if n.get("type") not in ("object", "landmark", "room"):
                continue
            p = n.get("position")
            if not p:
                continue
            px, pz = float(p[0]), float(p[2])
            if abs(px - cx) <= keep_radius and abs(pz - cz) <= keep_radius:
                xs.append(px)
                zs.append(pz)

    xlim = (min(xs) - margin, max(xs) + margin)
    zlim = (min(zs) - margin, max(zs) + margin)

    # Keep a sane aspect ratio so GT panels do not look stretched.
    x_span = max(1e-3, xlim[1] - xlim[0])
    z_span = max(1e-3, zlim[1] - zlim[0])
    if z_span > x_span * 1.35:
        pad = (z_span - x_span * 1.35) * 0.5
        xlim = (xlim[0] - pad, xlim[1] + pad)
    elif x_span > z_span * 1.35:
        pad = (x_span - z_span * 1.35) * 0.5
        zlim = (zlim[0] - pad, zlim[1] + pad)

    return xlim, zlim


def resolve_repo_path(path_like) -> Optional[Path]:
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


def load_origin_world(trace: dict) -> Tuple[Optional[np.ndarray], str]:
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


def infer_navmesh_path(trace: dict, override: Optional[str] = None) -> Tuple[Optional[Path], str]:
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
    scene = trace.get("scene") or (
        Path(str(trace.get("scene_file", ""))).name.split(".")[0] if trace.get("scene_file") else ""
    )
    scene = scene.replace(".basis", "")
    if scene:
        matches = sorted((ROOT / "data/scene_datasets").glob(f"**/{scene}*.navmesh"))
        if matches:
            return matches[0], "scene search"
    return None, "navmesh not found"


def load_gt_topdown(
    trace: dict,
    resolution: float = 0.05,
    navmesh_override: Optional[str] = None,
    disabled: bool = False,
) -> dict:
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
            "pathfinder": pathfinder,
        }
    except Exception as exc:
        return {"available": False, "reason": f"topdown unavailable: {exc}", "navmesh_path": str(navmesh_path)}


def draw_gt_topdown_relative(ax, gt_ctx: dict, origin_world: np.ndarray) -> bool:
    if not gt_ctx.get("available"):
        ax.text(
            0.5, 0.5,
            "GT topdown unavailable\n{}".format(gt_ctx.get("reason", "")),
            ha="center", va="center", fontsize=8, transform=ax.transAxes,
        )
        return False
    bounds_min = gt_ctx["bounds_min"]
    bounds_max = gt_ctx["bounds_max"]
    extent = [
        bounds_min[0] - origin_world[0],
        bounds_max[0] - origin_world[0],
        bounds_max[2] - origin_world[2],
        bounds_min[2] - origin_world[2],
    ]
    ax.imshow(gt_ctx["topdown"], cmap="gray", origin="upper", extent=extent, alpha=0.58, zorder=0)
    return True


def draw_memory_panel(
    ax,
    nodes: List[dict],
    edges: List[dict],
    agent_pos: np.ndarray,
    title: str,
    xlim: Tuple[float, float],
    zlim: Tuple[float, float],
    show_labels: bool = True,
    max_labels: int = 12,
    gt_ctx: Optional[dict] = None,
    origin_world: Optional[np.ndarray] = None,
    show_legend: bool = True,
    axis_label: str = "relative",
    panel_mode: str = "default",
    show_object_uncertainty: bool = False,
):
    """Draw a single topomap panel with memory-aware styling."""
    if gt_ctx is not None and origin_world is not None:
        draw_gt_topdown_relative(ax, gt_ctx, origin_world)

    node_map = {n["id"]: n for n in nodes}

    for edge in edges:
        src = node_map.get(edge.get("source"))
        tgt = node_map.get(edge.get("target"))
        if src is None or tgt is None:
            continue
        sp = np.asarray(src["position"], dtype=np.float32)
        tp = np.asarray(tgt["position"], dtype=np.float32)
        etype = edge.get("type", "")
        if panel_mode == "spatial_structure":
            if etype != "adjacent_to":
                continue
            src_room = is_room_summary_node(src)
            tgt_room = is_room_summary_node(tgt)
            if src_room and tgt_room:
                color, width, alpha = "#15803d", 3.4, 0.95
            else:
                color, width, alpha = "#4ade80", 2.4, 0.92
            ax.plot(
                [sp[0], tp[0]], [sp[2], tp[2]],
                color=color, linewidth=width, linestyle="-", alpha=alpha, zorder=2,
            )
            continue
        if etype == "navigable":
            ax.plot([sp[0], tp[0]], [sp[2], tp[2]], color="#6b7280",
                    linewidth=0.8, linestyle="-", alpha=0.5, zorder=1)
        elif etype == "adjacent_to" and (
            is_room_summary_node(src) or is_room_summary_node(tgt)
        ):
            ax.plot(
                [sp[0], tp[0]], [sp[2], tp[2]],
                color="#22c55e", linewidth=1.4, linestyle="-", alpha=0.65, zorder=1,
            )
        elif etype == "belongs_to" and (is_room_summary_node(src) or is_room_summary_node(tgt)):
            ax.plot([sp[0], tp[0]], [sp[2], tp[2]], color=SUMMARY_COLOR,
                    linewidth=0.9, linestyle="-", alpha=0.55, zorder=1)

    legend_seen = set()
    label_count = 0
    for node_type in NODE_TYPES:
        group = [n for n in nodes if n.get("type") == node_type]
        if not group:
            continue
        group.sort(key=lambda n: float(n.get("confidence", 0)), reverse=True)
        for n in group:
            pos = np.asarray(n["position"], dtype=np.float32)
            style = node_style(node_type, n)
            if panel_mode == "spatial_structure" and is_room_summary_node(n):
                style = dict(style)
                style["size"] = float(style.get("size", 80)) * 1.35
                style["zorder"] = 8
            elif panel_mode == "spatial_structure" and node_type == "landmark":
                style = dict(style)
                role = (n.get("attributes") or {}).get("structure_role", "")
                style["size"] = float(style.get("size", 50)) * (1.35 if role == "portal" else 1.15)
                style["zorder"] = 9 if role == "portal" else 7
                if role == "portal":
                    style["edgecolor"] = "#5b21b6"
                    style["linewidth"] = 1.4
            elif panel_mode == "two_layer" and node_type in (
                "waypoint_visited", "waypoint_frontier", "waypoint_candidate",
            ):
                style = dict(style)
                style["zorder"] = 6
                style["alpha"] = min(1.0, float(style.get("alpha", 0.9)) + 0.05)
            legend_label = None
            lkey = _legend_key(node_type, n)
            if lkey not in legend_seen:
                legend_label = lkey
                legend_seen.add(lkey)
            scatter_kwargs = {
                "s": style["size"],
                "marker": style["marker"],
                "color": style["color"],
                "edgecolors": style["edgecolor"],
                "linewidths": style["linewidth"],
                "alpha": style["alpha"],
                "zorder": style["zorder"],
                "label": legend_label,
            }
            if style["color"] == "none":
                scatter_kwargs.pop("color")
                scatter_kwargs["facecolors"] = "none"
            ax.scatter([pos[0]], [pos[2]], **scatter_kwargs)
            if (
                show_object_uncertainty
                and node_type == "object"
                and is_bbox_object(n)
            ):
                ax.add_patch(Circle(
                    (pos[0], pos[2]),
                    OBJECT_UNCERTAINTY_RADIUS,
                    fill=False,
                    edgecolor="#16a34a",
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.45,
                    zorder=2,
                ))
            if show_labels and label_count < max_labels and node_type in ("object", "room", "landmark"):
                ax.text(pos[0], pos[2] + 0.08, display_node_label(n),
                        fontsize=5.5, ha="center", zorder=5)
                label_count += 1

    ax.scatter([agent_pos[0]], [agent_pos[2]], marker="*", s=180,
               color="#facc15", edgecolors="black", linewidths=0.9,
               label="agent", zorder=5)

    ax.set_title(title, fontsize=8)
    ax.set_xlabel(f"{axis_label} x")
    ax.set_ylabel(f"{axis_label} z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    if show_legend:
        ax.legend(loc="upper right", fontsize=5.5)


def _legend_key(node_type: str, node: dict) -> str:
    attrs = node.get("attributes", {})
    if is_room_summary_node(node):
        return "room/region summary"
    if is_clip_room_node(node):
        return "CLIP room tag (view)"
    if is_environment_landmark(node):
        return "env landmark (view tag)"
    if attrs.get("structure_role") == "portal":
        return "portal landmark"
    if attrs.get("structure_role") == "hub":
        return "hub landmark"
    if attrs.get("landmark_source") == "promoted_object" or attrs.get("promoted_from_object"):
        return "nav landmark (fused)"
    if is_navigation_landmark(node):
        return "nav landmark"
    if attrs.get("recovered_from_summary") and node_type == "object":
        return "recovered object"
    if is_bbox_object(node) and not attrs.get("recovered_from_summary"):
        return "detected object (bbox est.)"
    if attrs.get("history_compressed") and node_type in ("object", "landmark"):
        return f"compressed {LABELS.get(node_type, node_type)}"
    if node_type == "waypoint_visited" and attrs.get("waypoint_role") == "entrance":
        return "entrance waypoint"
    if node_type == "waypoint_visited" and attrs.get("waypoint_role") == "corridor_anchor":
        return "corridor anchor"
    return LABELS.get(node_type, node_type)
