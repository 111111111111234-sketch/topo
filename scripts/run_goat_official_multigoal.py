"""Run ConfTopo-GOAT inside the official GOAT/Habitat multigoal environment.

This runner is intentionally different from ``run_goat_multigoal_acceptance.py``:

* Habitat/GOAT owns the episode loop, task transitions, and measurements.
* ConfTopo only consumes observations and proposes low-level actions or a
  ``subtask_stop``.
* The output metrics are copied from ``env.get_metrics()``; this script does
  not reimplement GOAT Success/SPL.

Expected usage:

    python scripts/run_goat_official_multigoal.py \
        --config-path path/to/goat_eval.yaml \
        --split val_seen \
        --goal-graph-dir data/goal_graphs/goat \
        --episodes 10 \
        --output data/logs/goat_topo/official_multigoal/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from run_goat_minimal import ROOT, quat_to_heading, rgb_to_embedding

from conftopo.agents import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.instruction_graph import GoalNode, InstructionGraph
from conftopo.perception import GoatModalityClipEncoder, encode_agent_rgb_embed


LOW_ACTIONS = ("move_forward", "turn_left", "turn_right")
STOP_ACTION_CANDIDATES = ("subtask_stop", "stop")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def load_habitat_config(config_path: str, opts: list[str]):
    """Load a Habitat/GOAT config without tying this script to one API version."""
    try:
        from habitat_baselines.config.default import get_config

        return get_config(config_path, opts)
    except Exception:
        from habitat.config.default import get_config

        try:
            return get_config(config_path, opts)
        except TypeError:
            return get_config(config_paths=config_path, opts=opts)


def make_env(config):
    """Construct the official Habitat environment from the provided config."""
    try:
        import habitat

        return habitat.Env(config=config)
    except Exception:
        from habitat_baselines.common.environments import get_env_class

        env_name = None
        for path in (
            ("habitat_baselines", "env_name"),
            ("ENV_NAME",),
            ("env_name",),
        ):
            cur = config
            for key in path:
                cur = getattr(cur, key, None)
                if cur is None:
                    break
            if cur:
                env_name = str(cur)
                break
        env_cls = get_env_class(env_name or "NavRLEnv")
        return env_cls(config=config)


def thaw_config(config):
    if hasattr(config, "defrost"):
        config.defrost()


def freeze_config(config):
    if hasattr(config, "freeze"):
        config.freeze()


def maybe_set(config, dotted_key: str, value: Any) -> bool:
    """Best-effort config mutation across Habitat config versions."""
    cur = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if hasattr(cur, key):
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return False
    last = parts[-1]
    if hasattr(cur, last):
        setattr(cur, last, value)
        return True
    if isinstance(cur, dict):
        cur[last] = value
        return True
    return False


def configure_paths(config, args) -> None:
    """Patch common dataset/simulator paths when the config exposes them."""
    thaw_config(config)
    if args.split:
        for key in (
            "habitat.dataset.split",
            "DATASET.SPLIT",
            "dataset.split",
        ):
            maybe_set(config, key, args.split)
    if args.dataset_path:
        for key in (
            "habitat.dataset.data_path",
            "DATASET.DATA_PATH",
            "dataset.data_path",
        ):
            maybe_set(config, key, args.dataset_path)
    if args.scene_dataset:
        for key in (
            "habitat.simulator.scene_dataset",
            "SIMULATOR.SCENE_DATASET",
            "simulator.scene_dataset",
        ):
            maybe_set(config, key, args.scene_dataset)
    freeze_config(config)


def normalize_action_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


def get_possible_actions(config) -> list[str]:
    """Extract action names from the config, falling back to GOAT defaults."""
    candidates: list[Any] = []
    for path in (
        ("habitat", "task", "actions"),
        ("TASK", "ACTIONS"),
        ("habitat", "task", "possible_actions"),
        ("TASK", "POSSIBLE_ACTIONS"),
    ):
        cur = config
        for key in path:
            cur = getattr(cur, key, None)
            if cur is None:
                break
        if cur:
            candidates.append(cur)

    for actions in candidates:
        if isinstance(actions, dict):
            return [normalize_action_name(k) for k in actions.keys()]
        if hasattr(actions, "keys"):
            return [normalize_action_name(k) for k in actions.keys()]
        if isinstance(actions, (list, tuple)):
            return [normalize_action_name(x) for x in actions]

    return [
        "stop",
        "move_forward",
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
        "subtask_stop",
    ]


def build_action_map(config) -> dict[str, int]:
    actions = get_possible_actions(config)
    return {normalize_action_name(name): idx for idx, name in enumerate(actions)}


def choose_stop_action(action_map: dict[str, int]) -> str:
    for name in STOP_ACTION_CANDIDATES:
        if name in action_map:
            return name
    raise KeyError("Official GOAT action space must expose 'subtask_stop' or 'stop'.")


def step_env(env, action_name: str, action_map: dict[str, int]):
    if action_name not in action_map:
        raise KeyError(f"Action '{action_name}' not in action space: {sorted(action_map)}")
    action_id = action_map[action_name]
    try:
        return env.step(action_id)
    except Exception:
        return env.step({"action": action_id})


def extract_rgb(observations: dict[str, Any]) -> np.ndarray | None:
    preferred = ("rgb", "color_sensor", "rgb_sensor", "head_rgb", "panoramic_rgb")
    for key in preferred:
        value = observations.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return value[..., :3]
    for value in observations.values():
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in (3, 4):
            return value[..., :3]
    return None


def sim_pose(env) -> tuple[np.ndarray | None, float | None]:
    sim = getattr(env, "sim", None) or getattr(env, "_sim", None)
    if sim is None or not hasattr(sim, "get_agent_state"):
        return None, None
    state = sim.get_agent_state()
    pos = np.asarray(state.position, dtype=np.float32)
    rot = state.rotation
    heading = quat_to_heading([rot.real, *list(rot.imag)])
    return pos, heading


def extract_pose(
    env,
    observations: dict[str, Any],
    allow_sim_pose: bool,
) -> tuple[np.ndarray, float, str]:
    """Return position and heading for ConfTopo observations.

    Prefer official GPS/Compass observations.  ``--allow-sim-pose-fallback`` is
    useful for debugging configs that omit pose sensors, but official runs
    should provide pose through observations.
    """
    position = None
    heading = None
    source = "observation"

    for key in ("gps", "gps_compass", "position"):
        value = observations.get(key)
        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                position = arr[:3]
                break
            if arr.size == 2:
                position = np.array([arr[0], 0.0, arr[1]], dtype=np.float32)
                break

    for key in ("compass", "heading"):
        value = observations.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size:
                heading = float(arr[0])
                break

    if (position is None or heading is None) and allow_sim_pose:
        sim_position, sim_heading = sim_pose(env)
        if position is None and sim_position is not None:
            position = sim_position
            source = "sim_state_fallback"
        if heading is None and sim_heading is not None:
            heading = sim_heading
            source = "sim_state_fallback"

    if position is None:
        raise RuntimeError("No GPS/position observation found. Add a pose sensor or pass --allow-sim-pose-fallback.")
    if heading is None:
        heading = 0.0
    return position.astype(np.float32), float(heading), source


def angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def local_controller(
    target_rel: np.ndarray | None,
    agent_rel: np.ndarray,
    heading: float,
    stop_radius: float,
    turn_threshold: float,
) -> str:
    if target_rel is None:
        return "turn_right"
    delta = np.asarray(target_rel, dtype=np.float32) - np.asarray(agent_rel, dtype=np.float32)
    dx = float(delta[0])
    dz = float(delta[2])
    distance = math.sqrt(dx * dx + dz * dz)
    if distance <= stop_radius:
        return "turn_right"
    desired_heading = math.atan2(-dx, -dz)
    err = angle_diff(desired_heading, heading)
    if err > turn_threshold:
        return "turn_left"
    if err < -turn_threshold:
        return "turn_right"
    return "move_forward"


def current_episode(env):
    if hasattr(env, "current_episode"):
        return env.current_episode
    task = getattr(env, "_task", None)
    return getattr(task, "current_episode", None)


def episode_id_of(episode) -> str:
    return str(getattr(episode, "episode_id", ""))


def scene_name_of(episode) -> str:
    scene_id = str(getattr(episode, "scene_id", ""))
    name = Path(scene_id).name
    return name.replace(".basis.glb", "").replace(".glb", "")


def load_goal_graph_for_episode(goal_graph_dir: Path, split: str, episode) -> InstructionGraph:
    path = goal_graph_dir / f"{split}_goal_graphs.json"
    all_goals = json.loads(path.read_text())
    ep_id = episode_id_of(episode)
    scene_name = scene_name_of(episode)

    preferred = f"{scene_name}_{ep_id}"
    if preferred in all_goals:
        return InstructionGraph.from_dict(all_goals[preferred])

    suffix = f"_{ep_id}"
    matches = [
        key for key in all_goals
        if key.endswith(suffix) and (not scene_name or key.startswith(scene_name))
    ]
    if not matches:
        matches = [key for key in all_goals if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"Could not uniquely match GoalGraph for scene='{scene_name}', episode_id='{ep_id}'. "
            f"Matches: {matches[:5]}"
        )
    return InstructionGraph.from_dict(all_goals[matches[0]])


def make_conftopo_config(args) -> ConfTopoConfig:
    config = ConfTopoConfig()
    config.perception.clip_model = args.clip_model
    config.perception.clip_image_model = getattr(args, "clip_image_model", "RN50")
    config.perception.clip_device = args.clip_device
    config.perception.backend = args.perception_backend
    config.perception.vlm_api_base = args.vlm_api_base
    config.perception.vlm_model = args.vlm_model
    config.perception.vlm_timeout = args.vlm_timeout
    config.perception.heavy_enabled = bool(args.heavy_enabled or args.perception_backend == "vlm")
    if args.heavy_interval is not None:
        config.perception.heavy_interval = args.heavy_interval
    return config


def get_goal_nodes(graph: InstructionGraph) -> list[GoalNode]:
    return [g for g in graph.goal_nodes if isinstance(g, GoalNode)]


def maybe_sync_goal(agent: ConfTopoGOATAgent, goals: list[GoalNode], goal_idx: int) -> None:
    if 0 <= goal_idx < len(goals):
        agent.set_new_goal(goals[goal_idx])


def run_episode(env, args, action_map: dict[str, int], encoder) -> dict[str, Any]:
    observations = env.reset()
    episode = current_episode(env)
    graph = load_goal_graph_for_episode(ROOT / args.goal_graph_dir, args.split, episode)
    goals = get_goal_nodes(graph)
    if not goals:
        raise RuntimeError("GoalGraph has no GOAT GoalNode entries.")

    agent = ConfTopoGOATAgent(make_conftopo_config(args))
    agent.set_goal(graph)
    if encoder is not None:
        agent.perceiver.room_text_embeds = encoder.encode_text(agent.perceiver.room_labels)
    maybe_sync_goal(agent, goals, 0)

    stop_action = choose_stop_action(action_map)
    trace_steps: list[dict[str, Any]] = []
    local_goal_idx = 0
    total_steps = 0
    done = False

    while not done and total_steps < args.max_steps_per_episode:
        rgb = extract_rgb(observations)
        if rgb is None:
            raise RuntimeError("No RGB observation found in official env observations.")
        position, heading, pose_source = extract_pose(env, observations, args.allow_sim_pose_fallback)
        rgb_embed = encode_agent_rgb_embed(
            encoder, rgb, agent,
            goal_type=goals[local_goal_idx].goal_type if local_goal_idx < len(goals) else "category",
            use_placeholder=args.use_placeholder_embed,
            placeholder_fn=rgb_to_embedding,
        )

        out = agent.step({
            "rgb": rgb,
            "rgb_embed": rgb_embed,
            "position": position,
            "heading": heading,
        })

        direct = normalize_action_name(out.get("action", "navigate"))
        target = out.get("target_position")
        if direct in LOW_ACTIONS:
            action_name = direct
        elif direct == "stop" or normalize_action_name(out.get("plan_action", "")) == "stop":
            action_name = stop_action
        else:
            agent_rel = np.asarray(getattr(agent, "_position", position), dtype=np.float32)
            action_name = local_controller(
                np.asarray(target, dtype=np.float32) if target is not None else None,
                agent_rel,
                heading,
                stop_radius=args.controller_stop_radius,
                turn_threshold=args.controller_turn_threshold,
            )

        observations = step_env(env, action_name, action_map)
        total_steps += 1
        done = bool(getattr(env, "episode_over", False))

        if action_name == stop_action:
            local_goal_idx += 1
            maybe_sync_goal(agent, goals, local_goal_idx)

        metrics = env.get_metrics() if hasattr(env, "get_metrics") else {}
        trace_steps.append({
            "step": total_steps - 1,
            "episode_id": episode_id_of(episode),
            "scene": scene_name_of(episode),
            "goal_index": local_goal_idx,
            "goal": _to_builtin(goals[min(local_goal_idx, len(goals) - 1)].to_dict())
            if local_goal_idx < len(goals) and hasattr(goals[local_goal_idx], "to_dict")
            else None,
            "agent_action": out.get("action"),
            "plan_action": out.get("plan_action"),
            "env_action": action_name,
            "target_node_id": out.get("target_node_id"),
            "target_position": _to_builtin(target),
            "pose_source": pose_source,
            "metrics": _to_builtin(metrics) if args.record_step_metrics else None,
            "memory": _to_builtin(agent.memory_stats),
        })

    final_metrics = env.get_metrics() if hasattr(env, "get_metrics") else {}
    return {
        "episode_id": episode_id_of(episode),
        "scene": scene_name_of(episode),
        "steps": total_steps,
        "episode_over": done,
        "goals_total": len(goals),
        "final_goal_index_local": local_goal_idx,
        "official_metrics": _to_builtin(final_metrics),
        "trace": trace_steps if args.record_trace else [],
        "final_memory": _to_builtin(agent.memory_stats),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    numeric: dict[str, list[float]] = {}
    for result in results:
        for key, value in (result.get("official_metrics") or {}).items():
            if isinstance(value, (int, float, bool)):
                numeric.setdefault(key, []).append(float(value))
    return {
        "episodes": len(results),
        "metric_means": {
            key: float(np.mean(values)) for key, values in sorted(numeric.items())
            if values
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, help="Official GOAT/Habitat eval config.")
    parser.add_argument("--opts", nargs="*", default=[], help="Extra Habitat config overrides.")
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--scene-dataset", default=None)
    parser.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=2000)
    parser.add_argument("--output", default="data/logs/goat_topo/official_multigoal/results.json")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--record-step-metrics", action="store_true")
    parser.add_argument("--allow-sim-pose-fallback", action="store_true")

    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clip-image-model", default="RN50", help="CLIP model for image-goal rgb_embed (GOAT official: RN50)")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--use-placeholder-embed", action="store_true")
    parser.add_argument("--perception-backend", choices=["clip_groundingdino", "vlm"], default="clip_groundingdino")
    parser.add_argument("--vlm-api-base", default="http://localhost:8000/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--vlm-timeout", type=float, default=5.0)
    parser.add_argument("--heavy-enabled", action="store_true")
    parser.add_argument("--heavy-interval", type=int, default=None)

    parser.add_argument("--controller-stop-radius", type=float, default=0.35)
    parser.add_argument("--controller-turn-threshold", type=float, default=0.25)
    args = parser.parse_args()

    config = load_habitat_config(args.config_path, args.opts)
    configure_paths(config, args)
    action_map = build_action_map(config)
    env = make_env(config)
    encoder = None if args.use_placeholder_embed else GoatModalityClipEncoder(
        args.clip_model, args.clip_image_model, args.clip_device,
    )

    results: list[dict[str, Any]] = []
    try:
        for _ in range(args.episodes):
            results.append(run_episode(env, args, action_map, encoder))
    finally:
        env.close()

    payload = {
        "runner": "run_goat_official_multigoal",
        "config_path": args.config_path,
        "split": args.split,
        "action_map": action_map,
        "results": results,
        "summary": aggregate(results),
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({
        "ok": True,
        "output": str(out),
        "summary": payload["summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
