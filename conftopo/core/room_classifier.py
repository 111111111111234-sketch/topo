"""Temporally-smoothed room classification for ConfTopo agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


_OBJECT_ROOM_PRIORS: List[Tuple[Tuple[str, ...], Tuple[str, ...], float]] = [
    (("bed", "mattress", "pillow", "wardrobe", "nightstand"), ("bedroom",), 0.15),
    (("toilet", "bathtub", "shower", "towel"), ("bathroom",), 0.15),
    (("sofa", "couch", "tv", "television", "coffee table"), ("living room",), 0.12),
    (("stove", "oven", "fridge", "refrigerator", "microwave", "counter", "sink", "dishwasher"), ("kitchen",), 0.12),
    (("dining table", "dining chair"), ("dining room",), 0.10),
    (("desk", "monitor", "bookshelf", "office chair"), ("office",), 0.10),
    (("rack", "shelf", "hanger"), ("bedroom", "closet", "laundry room"), 0.08),
    (("washer", "dryer", "laundry"), ("laundry room",), 0.12),
    (("stairs", "staircase"), ("hallway", "staircase"), 0.06),
]


def _object_room_boost(object_labels: List[str]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    obj_lower = [str(lbl).strip().lower() for lbl in object_labels]
    for obj_tokens, room_tokens, boost in _OBJECT_ROOM_PRIORS:
        if any(any(tok in obj for tok in obj_tokens) for obj in obj_lower):
            for room in room_tokens:
                boosts[room] = boosts.get(room, 0.0) + boost
    return boosts


@dataclass
class RoomClassifierConfig:
    min_raw_confidence: float = 0.30
    vote_window: int = 3
    confirm_vote_fraction: float = 0.60
    transition_min_displacement: float = 0.6
    transition_holdoff_steps: int = 3
    score_decay: float = 0.90
    object_prior_weight: float = 1.0


@dataclass
class RoomClassificationResult:
    confirmed_label: Optional[str]
    scores: Dict[str, float]
    is_transition: bool
    vote_counts: Dict[str, int]
    top_confidence: float


class RoomClassifier:
    """Stateful temporally-smoothed room classifier."""

    def __init__(self, config: Optional[RoomClassifierConfig] = None) -> None:
        self.config = config or RoomClassifierConfig()
        self._window: Deque[Tuple[str, float]] = deque(maxlen=self.config.vote_window)
        self._scores: Dict[str, float] = {}
        self._confirmed: Optional[str] = None
        self._last_transition_pos: Optional[np.ndarray] = None
        self._holdoff_remaining: int = 0
        self._step: int = 0
        self._prev_position: Optional[np.ndarray] = None

    def update(
        self,
        raw_label: str,
        raw_confidence: float,
        position: Optional[np.ndarray] = None,
        object_labels: Optional[List[str]] = None,
    ) -> RoomClassificationResult:
        self._step += 1
        cfg = self.config

        # 1. Decay
        for k in list(self._scores):
            self._scores[k] *= cfg.score_decay
            if self._scores[k] < 0.005:
                del self._scores[k]

        # 2. Gate + feed window
        clean_label = str(raw_label or "").strip().lower()
        if clean_label and clean_label != "unknown" and float(raw_confidence) >= cfg.min_raw_confidence:
            self._window.append((clean_label, float(raw_confidence)))
            self._scores[clean_label] = self._scores.get(clean_label, 0.0) + float(raw_confidence)

        # 3. Object prior
        if object_labels and cfg.object_prior_weight > 0:
            for room, boost in _object_room_boost(object_labels).items():
                self._scores[room] = self._scores.get(room, 0.0) + boost * cfg.object_prior_weight

        # 4. Vote
        vote_counts: Dict[str, int] = {}
        for lbl, _ in self._window:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1
        winner: Optional[str] = None
        winner_votes = 0
        for lbl, cnt in vote_counts.items():
            if cnt > winner_votes:
                winner, winner_votes = lbl, cnt
        vote_total = len(self._window)
        winner_fraction = (winner_votes / vote_total) if vote_total > 0 else 0.0

        # 5. Displacement from last confirmed position
        pos = np.array(position, dtype=np.float32) if position is not None else None
        anchor = self._last_transition_pos
        if pos is not None and anchor is not None:
            displacement = float(np.linalg.norm(pos[[0, 2]] - anchor[[0, 2]]))
        else:
            # No anchor yet: treat as having crossed minimum displacement
            displacement = float(cfg.transition_min_displacement)
        if pos is not None:
            self._prev_position = pos.copy()

        # 6. Holdoff countdown
        is_transition = False
        if self._holdoff_remaining > 0:
            self._holdoff_remaining -= 1

        # 7. Confirm / transition
        candidate = winner if winner_fraction >= cfg.confirm_vote_fraction else None
        if candidate is not None and candidate != self._confirmed and self._holdoff_remaining == 0:
            window_half = max(1, cfg.vote_window // 2)
            window_full_enough = len(self._window) >= window_half
            need_displacement = self._confirmed is not None
            if window_full_enough and (not need_displacement or displacement >= cfg.transition_min_displacement):
                is_transition = True
                self._confirmed = candidate
                self._holdoff_remaining = cfg.transition_holdoff_steps
                if pos is not None:
                    self._last_transition_pos = pos.copy()

        # 8. Normalise
        total = sum(self._scores.values())
        normalised = {k: v / total for k, v in self._scores.items()} if total > 1e-6 else dict(self._scores)
        top_conf = normalised.get(self._confirmed, 0.0) if self._confirmed else 0.0

        return RoomClassificationResult(
            confirmed_label=self._confirmed,
            scores=dict(normalised),
            is_transition=is_transition,
            vote_counts=dict(vote_counts),
            top_confidence=top_conf,
        )

    def reset(self) -> None:
        self._window.clear()
        self._scores.clear()
        self._confirmed = None
        self._last_transition_pos = None
        self._holdoff_remaining = 0
        self._step = 0
        self._prev_position = None

    def state_dict(self) -> Dict[str, Any]:
        return {"confirmed": self._confirmed, "scores": dict(self._scores),
                "holdoff": self._holdoff_remaining, "step": self._step}
