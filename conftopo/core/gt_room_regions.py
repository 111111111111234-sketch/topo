"""GT room regions from habitat semantic annotations (when available)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from conftopo.core.region_rooms import normalize_region_label


@dataclass
class GTRoomRegion:
    region_id: int
    label: str
    centroid: np.ndarray
    aabb_min: Optional[np.ndarray] = None
    aabb_max: Optional[np.ndarray] = None
    source: str = "habitat_semantic"


def _try_load_habitat_semantic_regions(
    scene_file: Optional[str],
    origin_world: Optional[np.ndarray] = None,
) -> List[GTRoomRegion]:
    if not scene_file:
        return []
    scene_path = Path(scene_file)
    candidates = [
        scene_path.with_name(scene_path.name.replace(".basis.glb", ".basis.scn")),
        scene_path.parent / "info_semantic.json",
    ]
    descriptor = next((path for path in candidates if path.is_file()), None)
    if descriptor is None or descriptor.suffix == ".json":
        return []
    try:
        import habitat_sim

        semantic_scene = habitat_sim.scene.SemanticScene()
        if not semantic_scene.load_mp3d_house(
            str(descriptor), semantic_scene, np.zeros(4, dtype=np.float32),
        ):
            return []
        origin = np.asarray(origin_world, dtype=np.float32).reshape(-1) if origin_world is not None else np.zeros(3, dtype=np.float32)
        regions: List[GTRoomRegion] = []
        for region in semantic_scene.regions:
            if region.aabb is None:
                continue
            center = np.asarray(region.aabb.center, dtype=np.float32)
            rel_center = center - origin
            rel_center[1] = center[1]
            aabb_min = np.asarray(region.aabb.min, dtype=np.float32) - origin
            aabb_max = np.asarray(region.aabb.max, dtype=np.float32) - origin
            label = region.category.name() if region.category else f"region_{region.id}"
            regions.append(GTRoomRegion(
                region_id=int(region.id),
                label=str(label),
                centroid=rel_center.astype(np.float32),
                aabb_min=aabb_min.astype(np.float32),
                aabb_max=aabb_max.astype(np.float32),
                source="habitat_semantic",
            ))
        return regions
    except Exception:
        return []


def _position_in_aabb(position: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
    pos = np.asarray(position, dtype=np.float32).reshape(-1)
    x, y, z = float(pos[0]), float(pos[1] if pos.size > 1 else 0.0), float(pos[2] if pos.size > 2 else 0.0)
    return (
        aabb_min[0] <= x <= aabb_max[0]
        and aabb_min[1] <= y <= aabb_max[1]
        and aabb_min[2] <= z <= aabb_max[2]
    )


def rooms_from_gt_regions(
    nodes: List[dict],
    origin_world: np.ndarray,
    existing_summaries: Optional[List[dict]] = None,
    scene_file: Optional[str] = None,
) -> List[dict]:
    """Build room nodes from habitat semantic regions; empty if annotations missing."""
    del existing_summaries
    waypoints = [n for n in nodes if n.get("type") == "waypoint_visited"]
    if not waypoints or not scene_file:
        return []
    gt_regions = _try_load_habitat_semantic_regions(scene_file, origin_world=origin_world)
    if not gt_regions:
        return []

    visited_ids = set()
    for waypoint in waypoints:
        for region in gt_regions:
            if region.aabb_min is None or region.aabb_max is None:
                continue
            if _position_in_aabb(waypoint["position"], region.aabb_min, region.aabb_max):
                visited_ids.add(region.region_id)

    rooms: List[dict] = []
    for region in gt_regions:
        if region.region_id not in visited_ids:
            continue
        rooms.append({
            "id": f"gt_room::{region.region_id}",
            "type": "room",
            "position": region.centroid.tolist(),
            "label": region.label,
            "confidence": 0.8,
            "attributes": {
                "summary_type": "room_region",
                "region_source": region.source,
                "gt_region_id": region.region_id,
                "contains_labels": [],
            },
        })
    return rooms


def room_for_waypoint_gt(
    waypoint: dict,
    rooms: List[dict],
    gt_regions: List[GTRoomRegion],
) -> Optional[dict]:
    pos = np.asarray(waypoint["position"], dtype=np.float32)
    matched_id = None
    for region in gt_regions:
        if region.aabb_min is None or region.aabb_max is None:
            continue
        if _position_in_aabb(pos, region.aabb_min, region.aabb_max):
            matched_id = region.region_id
            break
    if matched_id is None:
        best_dist = float("inf")
        for region in gt_regions:
            dist = float(np.linalg.norm(pos - region.centroid))
            if dist < best_dist:
                best_dist = dist
                matched_id = region.region_id
        if best_dist > 3.0:
            return None
    room_id = f"gt_room::{matched_id}"
    for room in rooms:
        if room["id"] == room_id:
            return room
    return None
