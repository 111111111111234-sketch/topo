"""InstructionGraph: unified goal representation for R2R / GOAT / SOON tasks."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import json
import re
import numpy as np


_FIND_RE = re.compile(
    r"(?:find|locate|look for)\s+(?:me\s+)?(?:a|an|the|some)?\s*(.+?)(?:\s+(?:on|in|at|near|opposite|which|that|next|between|behind|under|above|below|beside|with)\b)",
    re.IGNORECASE,
)


_STOP_WORDS = {"in", "on", "at", "near", "which", "that", "is", "it", "and", "or", "to", "with", "behind", "under", "above", "below", "beside", "next", "between", "opposite"}
_ARTICLES = {"the", "a", "an", "some"}


def _extract_head_noun(text: str) -> str:
    """Quick extraction of head noun from a description sentence (for backward compat)."""
    text = text.strip()
    m = _FIND_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            words = candidate.split()
            if len(words) <= 2:
                return candidate
            segments = re.split(r",\s*|\s+and\s+", candidate)
            last_seg = segments[-1].strip()
            seg_words = last_seg.split()
            if len(seg_words) <= 2 and len(segments) > 1:
                return seg_words[-1]
            if len(seg_words) <= 2:
                return last_seg
            return " ".join(seg_words[-2:])
    # Fallback: collect leading content words until a stop word/preposition
    words = text.split()
    result_words = []
    for w in words:
        clean = w.lower().rstrip(".,;:")
        if clean in _ARTICLES and not result_words:
            continue  # skip leading articles
        if clean in _STOP_WORDS and result_words:
            break
        result_words.append(w.rstrip(".,;:"))
        if len(result_words) >= 3:
            break
    return " ".join(result_words) if result_words else text[:40]


@dataclass
class SubGoal:
    """R2R-style route sub-goal."""
    id: int
    action: str  # go_forward / turn_left / turn_right / stop
    landmark: Optional[str] = None
    landmark_embedding: Optional[np.ndarray] = None
    spatial_relation: str = "at"  # past / at / towards / near
    implied_room: Optional[str] = None
    termination_condition: str = ""
    status: str = "pending"  # pending / active / completed


@dataclass
class Relation:
    """Spatial relation between goal and a reference object."""
    relation_type: str  # near / left_of / right_of / in_front_of / behind / on / under
    reference: str  # reference object name


@dataclass
class GoalNode:
    """GOAT / SOON-style object goal."""
    target_object: str  # clean category/noun, e.g. "picture", "sink"
    description: Optional[str] = None  # raw full instruction text
    target_embedding: Optional[np.ndarray] = None
    attributes: List[str] = field(default_factory=list)
    room_prior: List[str] = field(default_factory=list)
    room_prior_embeddings: Optional[np.ndarray] = None
    landmarks: List[str] = field(default_factory=list)
    landmark_embeddings: Optional[np.ndarray] = None
    relations: List[Relation] = field(default_factory=list)
    goal_type: str = "category"  # category / description / image
    confidence: float = 1.0
    status: str = "pending"  # pending / active / completed / failed


class InstructionGraph:
    """Unified instruction/goal representation supporting both route and object-goal tasks."""

    def __init__(
        self,
        goal_type: str = "route",
        sub_goals: Optional[List[SubGoal]] = None,
        goal_nodes: Optional[List[GoalNode]] = None,
    ):
        """
        Args:
            goal_type: "route" (R2R) or "object_goal" (GOAT/SOON)
            sub_goals: ordered list of sub-goals for route instructions
            goal_nodes: list of goal nodes for object-goal tasks
        """
        self.goal_type = goal_type
        self.sub_goals = sub_goals or []
        self.goal_nodes = goal_nodes or []
        self._current_idx = 0

    def get_current_goal(self) -> Optional[Union[SubGoal, GoalNode]]:
        if self.goal_type == "route":
            if self._current_idx < len(self.sub_goals):
                return self.sub_goals[self._current_idx]
            return None
        else:
            if self._current_idx < len(self.goal_nodes):
                return self.goal_nodes[self._current_idx]
            return None

    def advance(self) -> bool:
        """Mark current goal as completed and advance to next.
        Returns True if there's a next goal, False if all done."""
        current = self.get_current_goal()
        if current is not None:
            current.status = "completed"
        self._current_idx += 1

        next_goal = self.get_current_goal()
        if next_goal is not None:
            next_goal.status = "active"
            return True
        return False

    def set_current_goal(self, goal: GoalNode):
        """For GOAT multi-goal: switch to a specific goal (memory not cleared)."""
        for i, g in enumerate(self.goal_nodes):
            if g is goal:
                self._current_idx = i
                g.status = "active"
                return
        self.goal_nodes.append(goal)
        self._current_idx = len(self.goal_nodes) - 1
        goal.status = "active"

    def set_current_goal_by_index(self, idx: int):
        """Switch to goal at given index."""
        if 0 <= idx < len(self.goal_nodes):
            self._current_idx = idx
            self.goal_nodes[idx].status = "active"

    def is_complete(self) -> bool:
        if self.goal_type == "route":
            return self._current_idx >= len(self.sub_goals)
        else:
            return self._current_idx >= len(self.goal_nodes)

    @property
    def current_idx(self) -> int:
        return self._current_idx

    @property
    def total_goals(self) -> int:
        if self.goal_type == "route":
            return len(self.sub_goals)
        return len(self.goal_nodes)

    @property
    def completed_goals(self) -> int:
        if self.goal_type == "route":
            return sum(1 for g in self.sub_goals if g.status == "completed")
        return sum(1 for g in self.goal_nodes if g.status == "completed")

    def get_all_landmark_embeddings(self) -> Optional[np.ndarray]:
        """Get all landmark embeddings (for CLIP matching)."""
        embeddings = []
        if self.goal_type == "route":
            for sg in self.sub_goals:
                if sg.landmark_embedding is not None:
                    embeddings.append(sg.landmark_embedding)
        else:
            for gn in self.goal_nodes:
                if gn.landmark_embeddings is not None:
                    embeddings.append(gn.landmark_embeddings)
        if not embeddings:
            return None
        return np.concatenate(embeddings, axis=0) if embeddings else None

    def get_target_embeddings(self) -> Optional[np.ndarray]:
        """Get target object embeddings (for object-goal tasks)."""
        embeddings = []
        for gn in self.goal_nodes:
            if gn.target_embedding is not None:
                embeddings.append(gn.target_embedding)
        if not embeddings:
            return None
        return np.stack(embeddings, axis=0)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        data = {"goal_type": self.goal_type, "current_idx": self._current_idx}
        if self.goal_type == "route":
            data["sub_goals"] = [
                {
                    "id": sg.id,
                    "action": sg.action,
                    "landmark": sg.landmark,
                    "spatial_relation": sg.spatial_relation,
                    "implied_room": sg.implied_room,
                    "termination_condition": sg.termination_condition,
                    "status": sg.status,
                }
                for sg in self.sub_goals
            ]
        else:
            data["goal_nodes"] = [
                {
                    "target_object": gn.target_object,
                    **({"description": gn.description} if gn.description else {}),
                    "attributes": gn.attributes,
                    "room_prior": gn.room_prior,
                    "landmarks": gn.landmarks,
                    "relations": [
                        {"relation_type": r.relation_type, "reference": r.reference}
                        for r in gn.relations
                    ],
                    "goal_type": gn.goal_type,
                    "confidence": gn.confidence,
                    "status": gn.status,
                }
                for gn in self.goal_nodes
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "InstructionGraph":
        """Deserialize from dict."""
        goal_type = data["goal_type"]
        ig = cls(goal_type=goal_type)
        ig._current_idx = data.get("current_idx", 0)

        if goal_type == "route":
            for sg_data in data.get("sub_goals", []):
                ig.sub_goals.append(SubGoal(
                    id=sg_data["id"],
                    action=sg_data["action"],
                    landmark=sg_data.get("landmark"),
                    spatial_relation=sg_data.get("spatial_relation", "at"),
                    implied_room=sg_data.get("implied_room"),
                    termination_condition=sg_data.get("termination_condition", ""),
                    status=sg_data.get("status", "pending"),
                ))
        else:
            for gn_data in data.get("goal_nodes", []):
                relations = [
                    Relation(r["relation_type"], r["reference"])
                    for r in gn_data.get("relations", [])
                ]
                target_embedding = gn_data.get("target_embedding")
                if target_embedding is not None:
                    target_embedding = np.asarray(target_embedding, dtype=np.float32)

                room_prior_embeddings = gn_data.get("room_prior_embeddings")
                if room_prior_embeddings is not None:
                    room_prior_embeddings = np.asarray(room_prior_embeddings, dtype=np.float32)

                landmark_embeddings = gn_data.get("landmark_embeddings")
                if landmark_embeddings is not None:
                    landmark_embeddings = np.asarray(landmark_embeddings, dtype=np.float32)

                raw_target = gn_data["target_object"]
                description = gn_data.get("description")
                goal_type_val = gn_data.get("goal_type", "category")

                # Backward compat: if target_object looks like a full sentence
                # (old format), migrate it to description and extract clean noun
                if description is None and goal_type_val == "description" and len(raw_target) > 60:
                    description = raw_target
                    raw_target = _extract_head_noun(raw_target)

                ig.goal_nodes.append(GoalNode(
                    target_object=raw_target,
                    description=description,
                    target_embedding=target_embedding,
                    attributes=gn_data.get("attributes", []),
                    room_prior=gn_data.get("room_prior", []),
                    room_prior_embeddings=room_prior_embeddings,
                    landmarks=gn_data.get("landmarks", []),
                    landmark_embeddings=landmark_embeddings,
                    relations=relations,
                    goal_type=goal_type_val,
                    confidence=gn_data.get("confidence", 1.0),
                    status=gn_data.get("status", "pending"),
                ))
        return ig

    @classmethod
    def load(cls, path: str) -> "InstructionGraph":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
