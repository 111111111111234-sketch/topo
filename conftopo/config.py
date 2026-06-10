"""ConfTopo-Agent configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryConfig:
    # 置信度衰减系数
    confidence_decay: float = 0.95
    near_radius: float = 3.0
    far_radius: float = 10.0
    # 剪枝阈值
    prune_threshold: float = 0.1
    max_nodes: int = 500
    # 合并半径
    merge_radius: float = 1.0
    # Room/region summary settings for coarse semantic memory
    summary_radius: float = 5.0
    summary_min_distance: float = 6.0
    summary_low_detail_threshold: float = 0.35
    summary_mid_detail_threshold: float = 0.65
    summary_max_observations: int = 20
    # Fold/visibility: distance beyond which compressed nodes are hidden from planning/viz
    fold_distance: float = 3.0
    # Distance-aware pruning thresholds
    far_prune_distance: float = 10.0
    far_prune_threshold: float = 0.18
    mid_prune_distance: float = 6.0
    mid_prune_threshold: float = 0.12
    # Relaxed room_level trigger conditions
    room_level_min_distance: float = 6.0
    room_level_confidence_max: float = 0.65
    room_level_detail_max: float = 0.55
    # Bottom-layer spatial graph: connect nearby room summaries
    room_link_max_distance: float = 12.0
    # Distant waypoint compression (navigation layer)
    waypoint_compress_enabled: bool = True
    waypoint_compress_distance: float = 5.0
    waypoint_compress_keep_near: float = 3.0
    waypoint_compress_collinear_deg: float = 20.0


@dataclass
class PerceptionConfig:
    light_every_step: bool = True
    heavy_enabled: bool = False
    heavy_interval: int = 7
    heavy_on_frontier: bool = True
    heavy_goal_warmup_steps: int = 1
    heavy_goal_sim_threshold: float = 0.35
    heavy_low_object_confidence: float = 0.35
    # Summary-context heavy perception: separate cooldown and label budget
    heavy_summary_cooldown: int = 8
    heavy_summary_max_labels: int = 12
    # Phase 3.7: tighten heavy labels with the planner's structure target.
    # When the planner has chosen a structure target (room / portal /
    # structural landmark), heavy detection prefers labels relevant to
    # that anchor:
    #   * room target  -> contains_labels + goal labels (drops the broad
    #                     default vocabulary)
    #   * portal /     -> structural label vocabulary + goal labels
    #     structural
    # If False, heavy labels are always derived from goal + perception +
    # default vocabulary (legacy behaviour).
    heavy_align_with_structure_target: bool = True
    heavy_backend: str = "groundingdino"
    groundingdino_config: Optional[str] = None
    groundingdino_checkpoint: Optional[str] = None
    groundingdino_device: str = "cpu"
    groundingdino_text_threshold: float = 0.25
    clip_model: str = "ViT-B/32"
    clip_device: str = "auto"
    object_threshold: float = 0.28
    object_detection_threshold: float = 0.40
    room_threshold: float = 0.20
    landmark_threshold: float = 0.22
    # 房间类型标签
    room_labels: List[str] = field(default_factory=lambda: [
        "kitchen", "living room", "bedroom", "bathroom",
        "hallway", "dining room", "office", "garage",
        "laundry room", "closet", "staircase", "balcony",
    ])


@dataclass
class PlanningConfig:
    alpha: float = 0.0
    beta: float = 0.0
    normalize_scores: bool = True
    use_retrieval: bool = False
    exploration_threshold: float = 0.3
    retrieval_hidden_dim: int = 256
    sticky_target_enabled: bool = True
    sticky_reach_radius: float = 0.75
    sticky_release_after_no_progress: int = 5
    sticky_min_progress: float = 0.05
    frontier_consume_radius: float = 0.75
    target_too_close_radius: float = 0.45
    blocked_target_ttl: int = 20
    # --- Phase 3: two-stage room-centric planning -----------------------
    # When enabled, plan() first picks a structure target (room / portal /
    # structural landmark) and then expands navigation candidates anchored
    # to that target. Falls back to the legacy single-stage behaviour when
    # no structure target clears the score threshold.
    two_stage_enabled: bool = True
    structure_target_score_threshold: float = 0.05
    structure_anchor_radius: float = 6.0
    structure_anchor_bonus: float = 0.25
    # Far semantic objects (not the goal object) are pruned when a
    # structure target is active and they sit outside the anchor radius.
    far_object_skip_when_anchored: bool = True


@dataclass
class ConfTopoConfig:
    enabled: bool = True
    goal_graph_dir: str = "data/goal_graphs"

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
