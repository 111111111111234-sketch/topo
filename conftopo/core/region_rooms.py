"""Trajectory-derived region rooms for the semantic-structural layer."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REGION_LABEL_ALIASES = {
    "laundry room": "closet",
    "laundry": "closet",
    "kitchen": "hallway",
}

INSTANCE_CLUSTER_DIST_M = 3.0


def normalize_region_label(room_label: Optional[str]) -> Optional[str]:
    if room_label is None:
        return None
    room = str(room_label).strip().lower()
    if not room or room == "unknown":
        return None
    if "@" in room:
        room = room.split("@", 1)[0].strip()
    return REGION_LABEL_ALIASES.get(room, room)


def instance_direction_suffix(centroid: np.ndarray, ref: np.ndarray) -> str:
    dx = float(centroid[0] - ref[0])
    dz = float(centroid[2] - ref[2])
    if abs(dx) < 0.5 and abs(dz) < 0.5:
        return "main"
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz < 0 else "north"


def spatial_clusters(
    positions: List[np.ndarray],
    max_dist: float = INSTANCE_CLUSTER_DIST_M,
) -> List[List[np.ndarray]]:
    """Group positions into spatially connected clusters on the xz plane."""
    if not positions:
        return []
    pts = [np.asarray(pos, dtype=np.float32) for pos in positions]
    used = [False] * len(pts)
    clusters: List[List[np.ndarray]] = []
    for seed in range(len(pts)):
        if used[seed]:
            continue
        group = [seed]
        used[seed] = True
        queue = [seed]
        while queue:
            current = queue.pop()
            for idx in range(len(pts)):
                if used[idx]:
                    continue
                delta = pts[idx][[0, 2]] - pts[current][[0, 2]]
                if float(np.linalg.norm(delta)) <= max_dist:
                    used[idx] = True
                    group.append(idx)
                    queue.append(idx)
        clusters.append([pts[idx] for idx in group])
    return clusters


def cluster_waypoints_spatial(
    waypoints: List[dict],
    max_dist: float = INSTANCE_CLUSTER_DIST_M,
) -> List[List[dict]]:
    if not waypoints:
        return []
    positions = [np.asarray(wp["position"], dtype=np.float32) for wp in waypoints]
    pts = positions
    used = [False] * len(pts)
    clusters: List[List[dict]] = []
    for seed in range(len(pts)):
        if used[seed]:
            continue
        group_idx = [seed]
        used[seed] = True
        queue = [seed]
        while queue:
            current = queue.pop()
            for idx in range(len(pts)):
                if used[idx]:
                    continue
                delta = pts[idx][[0, 2]] - pts[current][[0, 2]]
                if float(np.linalg.norm(delta)) <= max_dist:
                    used[idx] = True
                    group_idx.append(idx)
                    queue.append(idx)
        clusters.append([waypoints[i] for i in group_idx])
    return clusters


def robust_region_centroid(
    positions: List[np.ndarray],
    max_dist: float = INSTANCE_CLUSTER_DIST_M,
    entrance_positions: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Anchor at cluster mean, preferring entrance waypoints when reliable."""
    if not positions:
        return np.zeros(3, dtype=np.float32)
    cluster_mean = np.mean(np.stack(positions), axis=0).astype(np.float32)
    if entrance_positions:
        entrances_in_cluster = [np.asarray(e, dtype=np.float32) for e in entrance_positions]
        if entrances_in_cluster:
            entrance_mean = np.mean(np.stack(entrances_in_cluster), axis=0).astype(np.float32)
            entrance_ok = (
                len(entrances_in_cluster) >= 2
                or float(np.linalg.norm(entrance_mean[[0, 2]] - cluster_mean[[0, 2]])) <= 1.5
            )
            if entrance_ok:
                return entrance_mean
    return cluster_mean


def _waypoint_clip_label(waypoint: dict) -> Optional[str]:
    attrs = waypoint.get("attributes") or {}
    return normalize_region_label(attrs.get("view_room_label"))


def object_room_votes_for_waypoint(
    waypoint_id: str,
    nodes: List[dict],
    edges: List[dict],
) -> Counter:
    """Room-context votes from GroundingDINO objects linked to this waypoint."""
    node_by_id = {n["id"]: n for n in nodes}
    votes: Counter = Counter()
    for edge in edges:
        if edge.get("type") not in ("observed_at", "visible_from"):
            continue
        src, tgt = edge.get("source"), edge.get("target")
        if waypoint_id not in (src, tgt):
            continue
        other_id = tgt if src == waypoint_id else src
        other = node_by_id.get(other_id)
        if other is None or other.get("type") not in ("object", "landmark"):
            continue
        attrs = other.get("attributes") or {}
        for key in ("room_context",):
            norm = normalize_region_label(attrs.get(key))
            if norm:
                votes[norm] += 1
        for value in attrs.get("room_contexts", []) or []:
            norm = normalize_region_label(value)
            if norm:
                votes[norm] += 1
    return votes


def effective_waypoint_region_label(
    waypoint: dict,
    nodes: Optional[List[dict]] = None,
    edges: Optional[List[dict]] = None,
) -> Optional[str]:
    """Fuse CLIP view tag with GroundingDINO-object room_context votes."""
    clip_label = _waypoint_clip_label(waypoint)
    if not nodes or not edges:
        return clip_label
    votes = object_room_votes_for_waypoint(waypoint["id"], nodes, edges)
    if not votes:
        return clip_label
    obj_label, obj_count = votes.most_common(1)[0]
    if clip_label is None:
        return obj_label
    if obj_label == clip_label:
        return clip_label
    if obj_count >= 2 and votes[clip_label] == 0:
        return obj_label
    return clip_label


def region_instances_from_waypoints(
    waypoints: List[dict],
    nodes: Optional[List[dict]] = None,
    edges: Optional[List[dict]] = None,
    max_dist: float = INSTANCE_CLUSTER_DIST_M,
) -> List[Dict[str, Any]]:
    """One instance per spatial cluster (>max_dist apart) within each region label."""
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for waypoint in waypoints:
        label = effective_waypoint_region_label(waypoint, nodes, edges)
        if label:
            by_label[label].append(waypoint)

    if not by_label:
        return []

    all_positions = [
        np.asarray(wp["position"], dtype=np.float32) for wp in waypoints
    ]
    ref = np.mean(np.stack(all_positions), axis=0)

    instances: List[Dict[str, Any]] = []
    used_display: set = set()
    for base_label in sorted(by_label):
        spatial_groups = cluster_waypoints_spatial(by_label[base_label], max_dist=max_dist)
        for group in spatial_groups:
            positions = [np.asarray(wp["position"], dtype=np.float32) for wp in group]
            entrances = [
                np.asarray(wp["position"], dtype=np.float32)
                for wp in group
                if (wp.get("attributes") or {}).get("waypoint_role") == "entrance"
            ]
            centroid = robust_region_centroid(
                positions, max_dist=max_dist, entrance_positions=entrances or None,
            )
            suffix = instance_direction_suffix(centroid, ref)
            display = f"{base_label}@{suffix}"
            if display in used_display:
                idx = 2
                while f"{base_label}@{suffix}{idx}" in used_display:
                    idx += 1
                display = f"{base_label}@{suffix}{idx}"
            used_display.add(display)
            instances.append({
                "base_label": base_label,
                "label": display,
                "instance_suffix": suffix,
                "centroid": centroid,
                "waypoint_ids": [wp["id"] for wp in group],
            })
    return instances


def rooms_from_waypoint_clusters(
    nodes: List[dict],
    existing_summaries: Optional[List[dict]] = None,
    edges: Optional[List[dict]] = None,
    max_dist: float = INSTANCE_CLUSTER_DIST_M,
) -> List[dict]:
    """Build region rooms: same label, spatially separated clusters become separate instances."""
    waypoints = [n for n in nodes if n.get("type") == "waypoint_visited"]
    instances = region_instances_from_waypoints(
        waypoints, nodes=nodes, edges=edges, max_dist=max_dist,
    )
    if not instances:
        return list(existing_summaries or [])

    summary_by_base: Dict[str, dict] = {}
    if existing_summaries:
        for room in existing_summaries:
            attrs = room.get("attributes") or {}
            key = normalize_region_label(attrs.get("base_label") or room.get("label"))
            if key and key not in summary_by_base:
                summary_by_base[key] = room

    rooms: List[dict] = []
    for idx, inst in enumerate(instances):
        base_label = inst["base_label"]
        display = inst["label"]
        centroid = inst["centroid"]
        base = summary_by_base.get(base_label)
        room_id = f"region::{base_label}::{idx}"
        if base is not None:
            attrs = dict(base.get("attributes") or {})
            attrs["summary_type"] = "room_region"
            attrs["region_source"] = "waypoint_cluster"
            attrs["base_label"] = base_label
            attrs["region_instance"] = inst["instance_suffix"]
            attrs["waypoint_ids"] = inst["waypoint_ids"]
            room = {
                **base,
                "id": room_id,
                "label": display,
                "position": centroid.tolist(),
                "attributes": attrs,
            }
        else:
            room = {
                "id": room_id,
                "type": "room",
                "position": centroid.tolist(),
                "label": display,
                "confidence": 0.65,
                "attributes": {
                    "summary_type": "room_region",
                    "region_source": "waypoint_cluster",
                    "base_label": base_label,
                    "region_instance": inst["instance_suffix"],
                    "waypoint_ids": inst["waypoint_ids"],
                    "contains_labels": [],
                },
            }
        rooms.append(room)
    return rooms


def room_for_labeled_waypoint(
    waypoint: dict,
    rooms: List[dict],
    nodes: Optional[List[dict]] = None,
    edges: Optional[List[dict]] = None,
    max_snap_dist: float = 6.0,
) -> Optional[dict]:
    """Map waypoint to the nearest room instance with matching base label."""
    label = effective_waypoint_region_label(waypoint, nodes, edges)
    if not label:
        return None
    wp_id = waypoint.get("id")
    for room in rooms:
        attrs = room.get("attributes") or {}
        if wp_id and wp_id in (attrs.get("waypoint_ids") or []):
            return room
    pos = np.asarray(waypoint["position"], dtype=np.float32)
    candidates = []
    for room in rooms:
        base = normalize_region_label(
            (room.get("attributes") or {}).get("base_label") or room.get("label")
        )
        if base != label:
            continue
        room_pos = np.asarray(room["position"], dtype=np.float32)
        dist = float(np.linalg.norm(pos[[0, 2]] - room_pos[[0, 2]]))
        if dist <= max_snap_dist:
            candidates.append((dist, room))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]
