"""LightPerceiver: CLIP-based lightweight semantic perception (每步运行).

从当前视觉 embedding 中提取语义信息:
- 房间类型分类 (kitchen / bedroom / ...)
- 目标物体相似度打分
- landmark 匹配

注意: 这个模块不加载 CLIP 模型本身 (ETPNav 已有 CLIPEncoder),
而是直接对已有的 CLIP 视觉 embedding 做文本对齐打分。
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity. a: [D] or [N,D], b: [M,D] -> [N,M] or [M]."""
    if a.ndim == 1:
        a = a[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    # normalize
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    sim = a_norm @ b_norm.T  # [N, M]
    if squeeze:
        return sim.squeeze(0)  # [M]
    return sim


class LightPerceiver:
    """CLIP-based room/object/landmark scoring.

    在 ETPNav 场景中，CLIP 视觉编码器已经在 policy 中运行 (CLIPEncoder)。
    LightPerceiver 接收已编码的视觉 embedding，与预计算的文本 embedding 做余弦相似度。

    Usage:
        perceiver = LightPerceiver(room_labels, clip_encode_fn)
        perceiver.set_goal_labels(["sink", "table"])

        # 每步
        result = perceiver.perceive(visual_embed)
        # result = {"room_scores": [("kitchen", 0.31), ...],
        #           "room_label": "kitchen",
        #           "goal_scores": [("sink", 0.45), ...],
        #           "best_goal_sim": 0.45}
    """

    def __init__(
        self,
        room_labels: Optional[List[str]] = None,
        room_text_embeds: Optional[np.ndarray] = None,
        clip_encode_text_fn=None,
    ):
        """
        Args:
            room_labels: list of room type strings
            room_text_embeds: pre-computed CLIP text embeddings [N_rooms, D],
                              if None, will compute from room_labels using clip_encode_text_fn
            clip_encode_text_fn: function(list[str]) -> np.ndarray [N, D]
        """
        if room_labels is None:
            room_labels = [
                "kitchen", "living room", "bedroom", "bathroom",
                "hallway", "dining room", "office", "garage",
                "laundry room", "closet", "staircase", "balcony",
            ]
        self.room_labels = room_labels
        self._clip_encode_text = clip_encode_text_fn

        # Room text embeddings
        if room_text_embeds is not None:
            self.room_text_embeds = room_text_embeds
        elif clip_encode_text_fn is not None:
            room_prompts = [f"a photo of a {r}" for r in room_labels]
            self.room_text_embeds = clip_encode_text_fn(room_prompts)
        else:
            self.room_text_embeds = None

        # Goal text embeddings (set per episode/goal)
        self.goal_labels: List[str] = []
        self.goal_text_embeds: Optional[np.ndarray] = None

        # Landmark text embeddings
        self.landmark_labels: List[str] = []
        self.landmark_text_embeds: Optional[np.ndarray] = None

    def set_goal_labels(
        self,
        labels: List[str],
        embeddings: Optional[np.ndarray] = None,
    ):
        """Set target object/goal labels for current episode."""
        self.goal_labels = labels
        if embeddings is not None:
            self.goal_text_embeds = embeddings
        elif self._clip_encode_text is not None and labels:
            prompts = [f"a photo of a {l}" for l in labels]
            self.goal_text_embeds = self._clip_encode_text(prompts)
        else:
            self.goal_text_embeds = None

    def set_landmark_labels(
        self,
        labels: List[str],
        embeddings: Optional[np.ndarray] = None,
    ):
        """Set landmark labels from instruction graph."""
        self.landmark_labels = labels
        if embeddings is not None:
            self.landmark_text_embeds = embeddings
        elif self._clip_encode_text is not None and labels:
            prompts = [f"a photo of a {l}" for l in labels]
            self.landmark_text_embeds = self._clip_encode_text(prompts)
        else:
            self.landmark_text_embeds = None

    def perceive(self, visual_embed: np.ndarray) -> Dict:
        """Score current visual embedding against all semantic categories.

        Args:
            visual_embed: CLIP visual embedding [D] or [V, D] (V views)

        Returns:
            dict with room_scores, room_label, goal_scores, landmark_scores, etc.
        """
        result = {}

        # Handle multi-view: average pool
        if visual_embed.ndim == 2:
            pooled = visual_embed.mean(axis=0)
        else:
            pooled = visual_embed

        # Room classification
        if self.room_text_embeds is not None:
            room_sims = cosine_sim(pooled, self.room_text_embeds)  # [N_rooms]
            room_scores = list(zip(self.room_labels, room_sims.tolist()))
            room_scores.sort(key=lambda x: x[1], reverse=True)
            result["room_scores"] = room_scores
            result["room_label"] = room_scores[0][0]
            result["room_confidence"] = float(room_scores[0][1])
        else:
            result["room_scores"] = []
            result["room_label"] = "unknown"
            result["room_confidence"] = 0.0

        # Goal object matching
        if self.goal_text_embeds is not None:
            goal_sims = cosine_sim(pooled, self.goal_text_embeds)  # [N_goals]
            goal_scores = list(zip(self.goal_labels, goal_sims.tolist()))
            goal_scores.sort(key=lambda x: x[1], reverse=True)
            result["goal_scores"] = goal_scores
            result["best_goal_sim"] = float(goal_scores[0][1]) if goal_scores else 0.0
        else:
            result["goal_scores"] = []
            result["best_goal_sim"] = 0.0

        # Landmark matching
        if self.landmark_text_embeds is not None:
            lm_sims = cosine_sim(pooled, self.landmark_text_embeds)  # [N_landmarks]
            lm_scores = list(zip(self.landmark_labels, lm_sims.tolist()))
            lm_scores.sort(key=lambda x: x[1], reverse=True)
            result["landmark_scores"] = lm_scores
            result["best_landmark_sim"] = float(lm_scores[0][1]) if lm_scores else 0.0
        else:
            result["landmark_scores"] = []
            result["best_landmark_sim"] = 0.0

        # Per-view scoring (for multi-view input)
        if visual_embed.ndim == 2 and self.goal_text_embeds is not None:
            per_view_goal_sims = cosine_sim(visual_embed, self.goal_text_embeds)  # [V, N_goals]
            result["per_view_goal_sims"] = per_view_goal_sims  # for directional preference
            result["best_view_idx"] = int(per_view_goal_sims.max(axis=1).argmax())
        else:
            result["per_view_goal_sims"] = None
            result["best_view_idx"] = 0

        return result

    def perceive_pano(
        self,
        pano_embeds: np.ndarray,
    ) -> Dict:
        """Score panoramic embeddings (12 views) for directional semantic info.

        Args:
            pano_embeds: [12, D] CLIP embeddings for each panoramic view direction

        Returns:
            dict with per-direction room/goal/landmark scores
        """
        result = {"per_view": []}
        for v in range(pano_embeds.shape[0]):
            view_result = self.perceive(pano_embeds[v])
            result["per_view"].append(view_result)

        # Aggregate: best room across views
        if self.room_text_embeds is not None:
            all_room_sims = cosine_sim(pano_embeds, self.room_text_embeds)  # [12, N_rooms]
            avg_room_sims = all_room_sims.mean(axis=0)
            best_room_idx = int(avg_room_sims.argmax())
            result["room_label"] = self.room_labels[best_room_idx]
            result["room_confidence"] = float(avg_room_sims[best_room_idx])

        # Aggregate: best goal direction
        if self.goal_text_embeds is not None:
            all_goal_sims = cosine_sim(pano_embeds, self.goal_text_embeds)  # [12, N_goals]
            max_per_view = all_goal_sims.max(axis=1)  # [12]
            result["best_goal_direction"] = int(max_per_view.argmax())
            result["best_goal_sim"] = float(max_per_view.max())

        return result

    def classify_room(self, visual_embed: np.ndarray) -> Tuple[str, float]:
        """Quick room classification.

        Returns: (room_label, confidence)
        """
        if self.room_text_embeds is None:
            return "unknown", 0.0
        if visual_embed.ndim == 2:
            visual_embed = visual_embed.mean(axis=0)
        sims = cosine_sim(visual_embed, self.room_text_embeds)
        idx = int(sims.argmax())
        return self.room_labels[idx], float(sims[idx])

    def match_goal(self, visual_embed: np.ndarray) -> Tuple[str, float]:
        """Quick goal matching.

        Returns: (best_goal_label, similarity)
        """
        if self.goal_text_embeds is None:
            return "", 0.0
        if visual_embed.ndim == 2:
            visual_embed = visual_embed.mean(axis=0)
        sims = cosine_sim(visual_embed, self.goal_text_embeds)
        idx = int(sims.argmax())
        return self.goal_labels[idx], float(sims[idx])
