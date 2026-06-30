"""Resolve and draw GT goal instance positions for GOAT trace visualizations."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GOAL_MARKER_COLORS = ("#e11d48", "#db2777", "#c026d3", "#7c3aed", "#ea580c", "#0891b2")


def resolve_repo_path(path_like: str | Path | None, root: Path = ROOT) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and (root / path).exists():
        return root / path
    parts = path.parts
    if "data" in parts:
        candidate = root.joinpath(*parts[parts.index("data") :])
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def _scene_basename(episode: dict[str, Any]) -> str:
    scene_id = str(episode.get("scene_id", ""))
    return scene_id.split("/")[-1].replace(".basis.glb", "").replace(".glb", "")


def _resolve_goal_entries(
    dataset_goals: dict[str, Any],
    episode: dict[str, Any],
    task_index: int,
) -> list[dict[str, Any]]:
    tasks = episode.get("tasks") or []
    if task_index >= len(tasks) or len(tasks[task_index]) < 2:
        return []
    goal_category, goal_type, *rest = tasks[task_index]
    goal_inst_id = rest[0] if rest else None

    matching = [
        entries
        for entries in dataset_goals.values()
        if isinstance(entries, list)
        and entries
        and isinstance(entries[0], dict)
        and entries[0].get("object_category") == goal_category
    ]
    if not matching:
        return []

    goal_entries = list(matching[0])
    scene_name = _scene_basename(episode)
    for child_cat in matching[0][0].get("children_object_categories") or []:
        child = dataset_goals.get(f"{scene_name}_{child_cat}")
        if isinstance(child, list):
            goal_entries.extend(child)

    if goal_type == "object":
        return [g for g in goal_entries if isinstance(g, dict)]
    if goal_inst_id is None:
        return []
    return [
        g
        for g in goal_entries
        if isinstance(g, dict) and str(g.get("object_id")) == str(goal_inst_id)
    ]


def _dedupe_positions(positions: list[list[float]]) -> list[list[float]]:
    seen: set[tuple] = set()
    out: list[list[float]] = []
    for pos in positions:
        key = tuple(np.round(np.asarray(pos, dtype=np.float32), 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append([float(x) for x in pos])
    return out


def world_to_relative(world_pos: Any, origin_world: Any) -> np.ndarray:
    rel = np.asarray(world_pos, dtype=np.float32) - np.asarray(origin_world, dtype=np.float32)
    rel[1] = 0.0
    return rel


def relative_to_world(relative_pos: Any, origin_world: Any) -> np.ndarray:
    return np.asarray(relative_pos, dtype=np.float32) + np.asarray(origin_world, dtype=np.float32)


def _positions_from_entries(
    goal_entries: list[dict[str, Any]],
    origin_world: list[float] | np.ndarray | None,
) -> tuple[list[list[float]], list[list[float]]]:
    world_positions: list[list[float]] = []
    for entry in goal_entries:
        pos_raw = entry.get("position")
        if pos_raw is None:
            continue
        world_positions.append(np.asarray(pos_raw, dtype=np.float32).round(4).tolist())
    world_positions = _dedupe_positions(world_positions)
    if origin_world is None:
        return world_positions, []
    relative_positions = [
        world_to_relative(pos, origin_world).round(4).tolist() for pos in world_positions
    ]
    return world_positions, relative_positions


def _instances_from_task_summaries(trace: dict[str, Any]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for summary in trace.get("task_summaries") or []:
        world = summary.get("goal_instance_positions_world")
        relative = summary.get("goal_instance_positions")
        if not world and not relative:
            continue
        instances.append(
            {
                "goal_index": int(summary.get("goal_index", len(instances))),
                "target_object": str(summary.get("target_object", "")),
                "positions_world": world or [],
                "positions_relative": relative or [],
            }
        )
    return instances


def _instances_from_episode(trace: dict[str, Any]) -> list[dict[str, Any]]:
    episode_file = resolve_repo_path(trace.get("episode_file"))
    if episode_file is None:
        return []

    try:
        with gzip.open(episode_file, "rt") as handle:
            dataset = json.load(handle)
    except Exception:
        return []

    episode_id = trace.get("episode_id")
    episode = None
    for item in dataset.get("episodes", []):
        if str(item.get("episode_id")) == str(episode_id):
            episode = item
            break
    if episode is None:
        return []

    origin_world = trace.get("origin_world")
    goals = trace.get("goals") or []
    instances: list[dict[str, Any]] = []
    for goal_index, target_object in enumerate(goals):
        entries = _resolve_goal_entries(dataset.get("goals", {}), episode, goal_index)
        world_positions, relative_positions = _positions_from_entries(entries, origin_world)
        if not world_positions:
            continue
        instances.append(
            {
                "goal_index": goal_index,
                "target_object": str(target_object),
                "positions_world": world_positions,
                "positions_relative": relative_positions,
            }
        )
    return instances


def load_goal_instances(trace: dict[str, Any]) -> list[dict[str, Any]]:
    cached = trace.get("_goal_instances_cache")
    if isinstance(cached, list):
        return cached

    instances = trace.get("goal_instances")
    if isinstance(instances, list) and instances:
        trace["_goal_instances_cache"] = instances
        return instances

    instances = _instances_from_task_summaries(trace)
    if instances:
        trace["_goal_instances_cache"] = instances
        return instances

    instances = _instances_from_episode(trace)
    trace["_goal_instances_cache"] = instances
    return instances


def goal_color_map(trace: dict[str, Any]) -> dict[str, str]:
    goals = trace.get("goals") or []
    if not goals:
        seen: list[str] = []
        for step in trace.get("steps", []):
            name = step.get("goal")
            if name and name not in seen:
                seen.append(str(name))
        goals = seen
    return {str(goal): GOAL_MARKER_COLORS[i % len(GOAL_MARKER_COLORS)] for i, goal in enumerate(goals)}


def plot_positions(
    instance: dict[str, Any],
    *,
    use_world: bool,
    origin_world: Any = None,
    position_frame: str = "episode_start_relative",
    pos_fn: Callable[[Any], np.ndarray] | None = None,
) -> list[np.ndarray]:
    if use_world:
        raw = instance.get("positions_world") or []
        if pos_fn is not None:
            return [pos_fn(p) for p in raw]
        if position_frame == "episode_start_relative" and origin_world is not None:
            return [relative_to_world(p, origin_world) for p in instance.get("positions_relative") or raw]
        return [np.asarray(p, dtype=np.float32) for p in raw]

    raw = instance.get("positions_relative") or []
    if raw:
        return [np.asarray(p, dtype=np.float32) for p in raw]
    if origin_world is None:
        return []
    return [world_to_relative(p, origin_world) for p in instance.get("positions_world") or []]


def extend_xz_limits(
    xs: list[float],
    zs: list[float],
    trace: dict[str, Any],
    *,
    use_world: bool,
    origin_world: Any = None,
    position_frame: str = "episode_start_relative",
) -> None:
    for instance in load_goal_instances(trace):
        for pos in plot_positions(
            instance,
            use_world=use_world,
            origin_world=origin_world,
            position_frame=position_frame,
        ):
            xs.append(float(pos[0]))
            zs.append(float(pos[2]))


def draw_goal_position_markers(
    ax,
    trace: dict[str, Any],
    step: dict[str, Any],
    *,
    use_world: bool = False,
    origin_world: Any = None,
    position_frame: str = "episode_start_relative",
    pos_fn: Callable[[Any], np.ndarray] | None = None,
    highlight_current: bool = True,
    show_labels: bool = True,
) -> None:
    instances = load_goal_instances(trace)
    if not instances:
        return

    colors = goal_color_map(trace)
    current_index = step.get("goal_index")
    current_name = str(step.get("goal") or "")

    def is_current(instance: dict[str, Any]) -> bool:
        return (
            current_index is not None and instance.get("goal_index") == current_index
        ) or (bool(current_name) and instance.get("target_object") == current_name)

    labeled_current = False
    labeled_other = False

    for instance in instances:
        current = is_current(instance)
        if highlight_current and not current:
            continue

        positions = plot_positions(
            instance,
            use_world=use_world,
            origin_world=origin_world,
            position_frame=position_frame,
            pos_fn=pos_fn,
        )
        if not positions:
            continue

        color = colors.get(str(instance.get("target_object", "")), GOAL_MARKER_COLORS[0])
        alpha = 1.0 if current or not highlight_current else 0.45
        size = 190 if current else 110
        label = None
        if current and not labeled_current:
            label = "goal GT ({})".format(instance.get("target_object", "goal"))
            labeled_current = True
        elif highlight_current and not current and not labeled_other:
            label = "other goal GT"
            labeled_other = True

        xs = [float(p[0]) for p in positions]
        zs = [float(p[2]) for p in positions]
        ax.scatter(
            xs,
            zs,
            marker="H",
            s=size,
            color=color,
            edgecolors="white",
            linewidths=1.0,
            alpha=alpha,
            label=label,
            zorder=6,
        )
        if show_labels and current:
            for pos in positions:
                ax.text(
                    float(pos[0]),
                    float(pos[2]) + 0.14,
                    str(instance.get("target_object", "goal")),
                    fontsize=7,
                    ha="center",
                    color=color,
                    zorder=7,
                )

    if highlight_current:
        for instance in instances:
            if is_current(instance):
                continue
            positions = plot_positions(
                instance,
                use_world=use_world,
                origin_world=origin_world,
                position_frame=position_frame,
                pos_fn=pos_fn,
            )
            if not positions:
                continue
            color = colors.get(str(instance.get("target_object", "")), GOAL_MARKER_COLORS[0])
            ax.scatter(
                [float(p[0]) for p in positions],
                [float(p[2]) for p in positions],
                marker="H",
                s=110,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.45,
                label=None if labeled_other else "other goal GT",
                zorder=5,
            )
            labeled_other = True
