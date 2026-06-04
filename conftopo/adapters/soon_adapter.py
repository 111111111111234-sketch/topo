"""SOON adapter skeleton for Phase 2 interface validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from conftopo.core.instruction_graph import GoalNode, InstructionGraph, Relation


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("type") or item.get("label")
                if name:
                    out.append(str(name))
            else:
                out.append(str(item))
        return out
    if isinstance(value, dict):
        return [str(v) for v in value.values() if isinstance(v, str)]
    return [str(value)]


def _attributes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        attrs = []
        for k, v in value.items():
            if isinstance(v, (list, tuple)):
                attrs.extend(str(x) for x in v)
            elif v:
                attrs.append(str(v))
        return attrs
    return _as_list(value)



def _embedding(value: Any):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)

def _relations(value: Any) -> List[Relation]:
    rels = []
    for row in value or []:
        if isinstance(row, dict):
            rels.append(Relation(str(row.get("relation_type", row.get("type", "near"))), str(row.get("reference", row.get("name", "")))))
    return rels


import re

_FIND_PATTERN = re.compile(
    r"(?:find|locate|look for)\s+(?:me\s+)?(?:a|an|the|some)?\s*(.+?)(?:\s+(?:on|in|at|near|opposite|which|that|next|between|behind|under|above|below|beside|with)\b)",
    re.IGNORECASE,
)

_TRIGGERS = ("find ", "locate ", "look for ", "go to find ", "navigate to find ", "go to ", "navigate to ", "reach ")

_STRIP_PHRASES = re.compile(
    r"^(?:(?:made of|consisting of)\s+[\w\s,]+\s+)?",
    re.IGNORECASE,
)


def _strip_leading_adjectives(phrase: str) -> str:
    """Extract core noun(s) from an adjective-heavy noun phrase.

    "tall, woody, gray cabinet" -> "cabinet"
    "short, colorful, made of metal and cotton chair" -> "chair"
    "rectangular, brown, wooden photo frame" -> "photo frame"
    "black chair" -> "black chair"  (keep short adj+noun combos)
    "long, big and soft bed" -> "bed"
    """
    words = phrase.split()
    if len(words) <= 2:
        return phrase

    # Split by commas + "and" to isolate segments, then take the tail noun(s)
    # from the last segment that doesn't look purely adjectival
    segments = re.split(r",\s*|\s+and\s+", phrase)
    last_seg = segments[-1].strip() if segments else phrase
    # The last segment's last word(s) are the head noun
    seg_words = last_seg.split()
    if len(seg_words) <= 2:
        # "wooden photo frame" -> take last 2 words; "soft bed" -> take last word
        # If last segment has a typical adjective + noun, just return last word
        if len(seg_words) == 2 and len(segments) > 1:
            return seg_words[-1]
        return last_seg
    # 3+ word last segment: return last 2 words (likely compound noun)
    return " ".join(seg_words[-2:])


def _extract_clean_target(text: str, attributes_hint: Any = None) -> str:
    """Extract a clean short noun phrase from a full description sentence.

    Heuristics (in priority order):
      1. If attributes_hint is a dict with a 'type' key, use it
      2. Regex: "find/locate/look for a <noun> on/in/at..."
      3. Fallback: first 3 words after longest matching trigger
      4. Last resort: first word
    """
    if isinstance(attributes_hint, dict) and attributes_hint.get("type"):
        return str(attributes_hint["type"])

    text = text.strip()
    m = _FIND_PATTERN.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate and len(candidate) < 60:
            return _strip_leading_adjectives(candidate)

    lower = text.lower()
    for trigger in sorted(_TRIGGERS, key=len, reverse=True):
        idx = lower.find(trigger)
        if idx >= 0:
            after = text[idx + len(trigger):].strip()
            for art in ("a ", "an ", "the ", "some "):
                if after.lower().startswith(art):
                    after = after[len(art):]
                    break
            words = after.split()[:3]
            if words:
                result = " ".join(words).rstrip(".,;:")
                return _strip_leading_adjectives(result)

    words = text.split()
    return words[0] if words else "object"


class SOONConfTopoAdapter:
    """Load SOON episodes/goals into the shared GoalNode/InstructionGraph format.

    Phase 2 only validates the interface: no full SOON navigation loop is required.
    """

    def __init__(self, dataset_dir: str | Path = "data/datasets/soon", goal_graph_dir: str | Path = "data/goal_graphs/soon"):
        self.dataset_dir = Path(dataset_dir)
        self.goal_graph_dir = Path(goal_graph_dir)

    def load_episodes(self, split: str = "val_unseen_house") -> List[Dict[str, Any]]:
        path = self.dataset_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.load(open(path))
        if not isinstance(data, list):
            raise ValueError(f"SOON split must be a list of episodes: {path}")
        return data

    def load_goal_graphs(self, split: str = "val_unseen_house") -> Dict[str, Any]:
        path = self.goal_graph_dir / f"{split}_goal_graphs.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return json.load(open(path))

    def goal_from_graph_dict(self, data: Dict[str, Any]) -> GoalNode:
        graph = data if data.get("goal_nodes") is not None else {"goal_nodes": [data]}
        nodes = graph.get("goal_nodes", [])
        if not nodes:
            raise ValueError("SOON goal graph has no goal_nodes")
        row = nodes[0]
        raw_target = str(row.get("target_object", row.get("description", "target")))
        # If target_object looks like a full sentence, treat it as description and
        # extract a clean noun from attributes or fall back to first word
        description = row.get("description") or None
        target_object = raw_target
        if len(raw_target) > 60 or raw_target.lower().startswith(("i'd", "i want", "find", "go")):
            description = raw_target
            target_object = _extract_clean_target(raw_target, row.get("attributes"))
        return GoalNode(
            target_object=target_object,
            description=description,
            target_embedding=_embedding(row.get("target_embedding")),
            attributes=_attributes(row.get("attributes")),
            room_prior=_as_list(row.get("room_prior")),
            room_prior_embeddings=_embedding(row.get("room_prior_embeddings")),
            landmarks=_as_list(row.get("landmarks")),
            landmark_embeddings=_embedding(row.get("landmark_embeddings")),
            relations=_relations(row.get("relations")),
            goal_type=str(row.get("goal_type", "description")),
            confidence=float(row.get("confidence", 1.0)),
            status=str(row.get("status", "pending")),
        )

    def goal_from_episode(self, episode: Dict[str, Any]) -> GoalNode:
        instructions = episode.get("instructions") or []
        raw_desc = instructions[-1] if instructions else episode.get("instr_id", "target")
        raw_desc = str(raw_desc)
        target_object = _extract_clean_target(raw_desc, instructions[:-1] if len(instructions) > 1 else None)
        return GoalNode(
            target_object=target_object,
            description=raw_desc,
            attributes=_attributes(instructions[:-1]),
            room_prior=[],
            landmarks=[],
            relations=[],
            goal_type="description",
        )

    def instruction_graph_from_goal(self, goal: GoalNode) -> InstructionGraph:
        goal.status = "active"
        return InstructionGraph(goal_type="object_goal", goal_nodes=[goal])

    def load_instruction_graph(self, split: str = "val_unseen_house", key: Optional[str] = None, episode_index: int = 0) -> InstructionGraph:
        if key is not None:
            graphs = self.load_goal_graphs(split)
            if key not in graphs:
                raise KeyError(key)
            return self.instruction_graph_from_goal(self.goal_from_graph_dict(graphs[key]))
        episodes = self.load_episodes(split)
        if episode_index >= len(episodes):
            raise IndexError(episode_index)
        return self.instruction_graph_from_goal(self.goal_from_episode(episodes[episode_index]))
