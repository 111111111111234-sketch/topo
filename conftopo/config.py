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


@dataclass
class PerceptionConfig:
    light_every_step: bool = True
    heavy_interval: int = 5
    heavy_on_frontier: bool = True
    clip_model: str = "ViT-B/32"
    clip_device: str = "auto"
    object_threshold: float = 0.23
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


@dataclass
class ConfTopoConfig:
    enabled: bool = True
    goal_graph_dir: str = "data/goal_graphs"

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
