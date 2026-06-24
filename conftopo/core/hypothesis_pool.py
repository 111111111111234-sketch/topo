"""Short-lived weak semantic hypotheses for ConfTopo.

HypothesisPool intentionally is not object memory. It stores weak evidence that
should guide verification and exploration, then expires or is promoted when a
confirmed perception path creates a real map node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class Hypothesis:
    id: str
    goal_id: Optional[str]
    kind: str
    label: str
    source: str
    anchor_node_id: Optional[str]
    position: Optional[np.ndarray]
    score: float = 0.0
    confidence: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    weak_bbox: Optional[List[float]] = None
    weak_relations: List[Any] = field(default_factory=list)
    first_seen_step: int = 0
    last_seen_step: int = 0
    last_decay_step: int = 0
    seen_count: int = 1
    ttl: int = 20
    status: str = "active"
    trigger_reason: Optional[str] = None
    promoted_node_id: Optional[str] = None
    reject_reason: Optional[str] = None


class HypothesisPool:
    """TTL-managed pool of weak observations awaiting verification."""

    def __init__(
        self,
        *,
        default_ttl: int = 20,
        decay_rate: float = 0.92,
        verify_seen_count: int = 2,
        verify_confidence: float = 0.45,
        rejected_cooldown: int = 12,
    ):
        self.default_ttl = int(default_ttl)
        self.decay_rate = float(decay_rate)
        self.verify_seen_count = int(verify_seen_count)
        self.verify_confidence = float(verify_confidence)
        self.rejected_cooldown = int(rejected_cooldown)
        self._items: Dict[str, Hypothesis] = {}
        self._next_id = 0

    def clear(self, goal_id: Optional[str] = None) -> None:
        if goal_id is None:
            self._items.clear()
            return
        for hyp_id in list(self._items):
            if self._items[hyp_id].goal_id == goal_id:
                del self._items[hyp_id]

    def add_or_update(self, hyp: Hypothesis) -> Hypothesis:
        existing = self._find_match(hyp)
        if existing is None:
            if not hyp.id:
                hyp.id = self._new_id()
            if hyp.ttl <= 0:
                hyp.ttl = self.default_ttl
            if hyp.last_decay_step <= 0:
                hyp.last_decay_step = hyp.last_seen_step
            hyp.label = self._clean_label(hyp.label)
            hyp.kind = str(hyp.kind or "object").strip().lower()
            hyp.source = str(hyp.source or "unknown").strip().lower()
            hyp.status = hyp.status or "active"
            self._maybe_mark_needs_verify(hyp)
            self._items[hyp.id] = hyp
            return hyp

        existing.score = max(float(existing.score), float(hyp.score))
        existing.confidence = max(float(existing.confidence), float(hyp.confidence))
        existing.last_seen_step = max(int(existing.last_seen_step), int(hyp.last_seen_step))
        existing.last_decay_step = max(int(existing.last_decay_step), int(existing.last_seen_step))
        existing.seen_count += max(1, int(hyp.seen_count))
        if hyp.position is not None:
            new_pos = np.asarray(hyp.position, dtype=np.float32)
            if existing.position is None:
                existing.position = new_pos
            else:
                existing.position = (0.7 * existing.position + 0.3 * new_pos).astype(np.float32)
        if hyp.weak_bbox is not None:
            existing.weak_bbox = list(hyp.weak_bbox)
        if hyp.weak_relations:
            existing.weak_relations = self._dedupe(existing.weak_relations + list(hyp.weak_relations))
        existing.attributes.update(dict(hyp.attributes or {}))
        if hyp.trigger_reason:
            existing.trigger_reason = hyp.trigger_reason
        if existing.status == "expired":
            existing.status = "active"
        self._maybe_mark_needs_verify(existing)
        return existing

    def decay(self, current_step: int) -> None:
        for hyp in self._items.values():
            age = int(current_step) - int(hyp.last_seen_step)
            if hyp.status in ("promoted", "expired"):
                continue
            if hyp.status == "rejected":
                if age > self.rejected_cooldown:
                    hyp.status = "expired"
                continue
            if age > hyp.ttl:
                hyp.status = "expired"
                continue
            delta = int(current_step) - int(hyp.last_decay_step)
            if delta > 0:
                hyp.confidence = float(hyp.confidence) * (self.decay_rate ** delta)
                hyp.last_decay_step = int(current_step)

    def get_active(self, goal_id: Optional[str] = None, include_needs_verify: bool = True) -> List[Hypothesis]:
        statuses = {"active", "needs_verify"} if include_needs_verify else {"active"}
        return [
            h for h in self._items.values()
            if h.status in statuses and (goal_id is None or h.goal_id == goal_id)
        ]

    def get_top_for_vlm(self, goal_id: Optional[str], k: int = 3) -> List[Hypothesis]:
        active = self.get_active(goal_id=goal_id, include_needs_verify=True)
        active.sort(key=lambda h: (h.status == "needs_verify", h.confidence, h.seen_count), reverse=True)
        return active[: max(0, int(k))]

    def promote(self, hyp_id: str, object_node_id: str) -> Optional[Hypothesis]:
        hyp = self._items.get(hyp_id)
        if hyp is None:
            return None
        hyp.status = "promoted"
        hyp.promoted_node_id = object_node_id
        return hyp

    def promote_matching(
        self,
        *,
        goal_id: Optional[str],
        label: str,
        anchor_node_id: Optional[str],
        object_node_id: str,
    ) -> List[Hypothesis]:
        label = self._clean_label(label)
        promoted: List[Hypothesis] = []
        for hyp in self.get_active(goal_id=goal_id, include_needs_verify=True):
            if hyp.kind != "object" or hyp.label != label:
                continue
            if anchor_node_id is not None and hyp.anchor_node_id not in (None, anchor_node_id):
                continue
            promoted_hyp = self.promote(hyp.id, object_node_id)
            if promoted_hyp is not None:
                promoted.append(promoted_hyp)
        return promoted

    def reject(self, hyp_id: str, reason: str) -> Optional[Hypothesis]:
        hyp = self._items.get(hyp_id)
        if hyp is None:
            return None
        hyp.status = "rejected"
        hyp.reject_reason = str(reason)
        return hyp

    def reject_goal_near_anchor(
        self,
        *,
        goal_id: Optional[str],
        anchor_node_id: Optional[str],
        reason: str,
    ) -> List[Hypothesis]:
        rejected: List[Hypothesis] = []
        for hyp in self.get_active(goal_id=goal_id, include_needs_verify=True):
            if anchor_node_id is not None and hyp.anchor_node_id != anchor_node_id:
                continue
            rejected_hyp = self.reject(hyp.id, reason)
            if rejected_hyp is not None:
                rejected.append(rejected_hyp)
        return rejected

    def to_debug_list(self, goal_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        items = list(self._items.values())
        if goal_id is not None:
            items = [h for h in items if h.goal_id == goal_id]
        items.sort(key=lambda h: (h.status == "needs_verify", h.confidence, h.last_seen_step), reverse=True)
        out = []
        for h in items[: max(0, int(limit))]:
            out.append({
                "id": h.id,
                "goal_id": h.goal_id,
                "kind": h.kind,
                "label": h.label,
                "source": h.source,
                "anchor_node_id": h.anchor_node_id,
                "score": round(float(h.score), 4),
                "confidence": round(float(h.confidence), 4),
                "seen_count": int(h.seen_count),
                "status": h.status,
                "trigger_reason": h.trigger_reason,
                "promoted_node_id": h.promoted_node_id,
                "reject_reason": h.reject_reason,
            })
        return out

    def _find_match(self, hyp: Hypothesis) -> Optional[Hypothesis]:
        label = self._clean_label(hyp.label)
        kind = str(hyp.kind or "object").strip().lower()
        source = str(hyp.source or "unknown").strip().lower()
        for existing in self._items.values():
            if existing.status in ("promoted", "expired"):
                continue
            if existing.goal_id != hyp.goal_id:
                continue
            if existing.kind != kind or existing.label != label or existing.source != source:
                continue
            if existing.anchor_node_id == hyp.anchor_node_id:
                return existing
        return None

    def _maybe_mark_needs_verify(self, hyp: Hypothesis) -> None:
        if hyp.status not in ("active", "needs_verify"):
            return
        if hyp.seen_count >= self.verify_seen_count and hyp.confidence >= self.verify_confidence:
            hyp.status = "needs_verify"
            if not hyp.trigger_reason:
                hyp.trigger_reason = "multi_frame_stable"

    def _new_id(self) -> str:
        self._next_id += 1
        return f"hyp_{self._next_id}"


    def promote_by_goal_and_label(
        self,
        *,
        goal_id: str,
        label: str,
        object_node_id: str,
    ) -> Optional[Hypothesis]:
        label = self._clean_label(label)
        for hyp in self.get_active(goal_id=goal_id, include_needs_verify=True):
            if hyp.kind == "object" and hyp.label == label:
                return self.promote(hyp.id, object_node_id)
        return None

    def reject_active_by_goal(
        self,
        *,
        goal_id: str,
        reason: str,
        exclude_object_node_ids: Optional[Set[str]] = None,
    ) -> List[Hypothesis]:
        excluded = exclude_object_node_ids or set()
        rejected: List[Hypothesis] = []
        for hyp in self.get_active(goal_id=goal_id, include_needs_verify=True):
            if hyp.promoted_node_id in excluded:
                continue
            if hyp.status in ("active", "needs_verify"):
                r = self.reject(hyp.id, reason)
                if r is not None:
                    rejected.append(r)
        return rejected

    @staticmethod
    def _clean_label(label: str) -> str:
        return str(label or "").strip().lower()

    @staticmethod
    def _dedupe(values: List[Any]) -> List[Any]:
        out = []
        seen = set()
        for value in values:
            key = repr(value)
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out
