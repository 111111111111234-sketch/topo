"""Precompute GoalGraphs for all task types (R2R / GOAT / SOON).

Usage:
    python scripts/precompute_goal_graphs.py --task soon --split val_unseen_instrs --no-llm
    python scripts/precompute_goal_graphs.py --task all --llm-url http://localhost:8000/v1
    python scripts/precompute_goal_graphs.py --task all --embed-clip
"""

import argparse
import gzip
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
sys.path.insert(0, str(ROOT))

from conftopo.core.instruction_graph import InstructionGraph, SubGoal, GoalNode, Relation


# ---------------------------------------------------------------------------
# Lightweight LLM client (HTTP only — does not import ETPNav / vllm_server env)
# vLLM service runs separately: conda activate vllm_server && bash ETPNav/scripts/start_vllm.sh
# ---------------------------------------------------------------------------

def _extract_json_from_llm_text(text: str) -> Optional[dict]:
    """Parse JSON from LLM output; tolerate Qwen3 thinking blocks and markdown fences."""
    if not text:
        return None
    text = text.strip()
    # Strip Qwen3-style thinking blocks
    for tag in ("think", "redacted_thinking"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            chunk = part.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                text = chunk
                break
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _normalize_attributes(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[str] = []
        for k, v in value.items():
            if isinstance(v, bool) and v:
                out.append(str(k))
            elif isinstance(v, (list, tuple)):
                out.extend(str(x) for x in v)
            elif v is not None and not isinstance(v, dict):
                out.append(str(v))
        return out
    return _normalize_str_list(value)


def _apply_parsed_description(parsed: dict) -> Tuple[List[str], List[str], List[str], List[Relation], Optional[str]]:
    """Normalize LLM JSON into GoalNode fields."""
    attributes = _normalize_attributes(parsed.get("attributes"))
    room_prior = _normalize_str_list(parsed.get("room_prior"))
    landmarks = _normalize_str_list(parsed.get("landmarks"))
    relations = _relations_from_llm(parsed.get("relations"))
    target_object = parsed.get("target_object")
    if target_object is not None:
        target_object = str(target_object).strip()[:80]
    return attributes, room_prior, landmarks, relations, target_object or None


INSTRUCTION_PARSER_PROMPT = """You are a navigation instruction parser. Given a natural language navigation instruction, decompose it into a structured sequence of sub-goals.

Output a JSON object with:
- "sub_goals": a list of objects, each containing:
  - "id": integer index starting from 0
  - "action": the movement action (e.g., "turn_left", "turn_right", "go_forward", "go_up", "go_down", "stop")
  - "landmark": the key landmark or reference object mentioned (null if none)
  - "spatial_relation": spatial relationship to the landmark (e.g., "past", "towards", "left_of", "right_of", "at", null)
  - "implied_room": room type if mentioned (e.g., "kitchen", null)
  - "termination_condition": what signals this sub-goal is complete

Output valid JSON only."""


class GoalGraphLLM:
    """Call vLLM OpenAI API; runs in conftopo env while server uses vllm_server env."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen2.5-7B-Instruct",
        timeout: float = 120.0,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=timeout,
            max_retries=2,
        )
        self.model = model
        self._instruction_cache: Dict[str, dict] = {}
        self._description_cache: Dict[str, dict] = {}

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def parse_instruction(self, instruction: str, use_cache: bool = True) -> dict:
        if use_cache and instruction in self._instruction_cache:
            return self._instruction_cache[instruction]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INSTRUCTION_PARSER_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        except Exception:
            result = {"sub_goals": []}
        if use_cache:
            self._instruction_cache[instruction] = result
        return result

    def parse_description(self, description: str, use_cache: bool = True) -> Optional[dict]:
        if use_cache and description in self._description_cache:
            return self._description_cache[description]
        prompt = (
            f'Parse: "{description}"\n'
            "Return JSON: target_object (short noun), attributes (string list), "
            "room_prior (string list), landmarks (string list), "
            'relations (list of {"type","reference"}).'
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "/no_think\nOutput only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            result = _extract_json_from_llm_text(text)
        except Exception:
            result = None
        if use_cache and result is not None:
            self._description_cache[description] = result
        return result


def _find_goal_entry(goals: dict, category: str, ref) -> Optional[dict]:
    """Resolve GOAT goal dict entry from task category and optional ref id."""
    if ref is not None:
        ref_s = str(ref)
        for key, entries in goals.items():
            if ref_s in key and category.replace(" ", "_") in key.replace(" ", "_"):
                if entries:
                    return entries[0]
        for key, entries in goals.items():
            if ref_s in key and entries:
                return entries[0]

    suffix = category.replace(" ", "_")
    for key, entries in goals.items():
        if key.endswith(f"_{suffix}") or key.endswith(f"_{category}"):
            if entries:
                return entries[0]
    for key, entries in goals.items():
        if category in key and entries:
            return entries[0]
    return None


def _goal_node_from_goat_task(
    category: str,
    modality: str,
    ref,
    goals: dict,
    llm_client=None,
) -> GoalNode:
    """Build GoalNode from one GOAT sub-task."""
    entry = _find_goal_entry(goals, category, ref)
    target = category
    raw_description: Optional[str] = None
    attributes: List[str] = []
    room_prior: List[str] = []
    landmarks: List[str] = []
    relations: List[Relation] = []
    goal_type = modality  # object / description / image

    if entry:
        target = entry.get("object_category", category)
        lang = entry.get("lang_desc") or entry.get("description") or ""
        if lang and modality == "description":
            raw_description = lang.strip()[:500]
            # target_object stays as clean category noun; fallback to first noun phrase
            if llm_client:
                parsed = _parse_description(lang, llm_client)
                if parsed:
                    attributes, room_prior, landmarks, relations, parsed_target = _apply_parsed_description(parsed)
                    if parsed_target:
                        target = parsed_target
        elif modality == "object":
            goal_type = "category"
        elif modality == "image":
            goal_type = "image"
    else:
        if modality == "description" and ref:
            raw_description = str(ref)
        goal_type = modality if modality in ("object", "description", "image") else "category"

    return GoalNode(
        target_object=target,
        description=raw_description,
        attributes=attributes,
        room_prior=room_prior,
        landmarks=landmarks,
        relations=relations,
        goal_type=goal_type,
    )


def _parse_description(description: str, llm_client) -> Optional[dict]:
    if hasattr(llm_client, "parse_description"):
        return llm_client.parse_description(description)
    return None


def _relations_from_llm(parsed_relations) -> List[Relation]:
    """Build Relation list from LLM JSON; tolerate missing/alternate keys."""
    out: List[Relation] = []
    if not parsed_relations:
        return out
    if isinstance(parsed_relations, dict):
        for rel_type, refs in parsed_relations.items():
            for ref in _normalize_str_list(refs):
                out.append(Relation(str(rel_type), ref[:120]))
        return out
    for r in parsed_relations:
        if isinstance(r, str) and r.strip():
            out.append(Relation("near", r.strip()[:120]))
            continue
        if not isinstance(r, dict):
            continue
        ref = (
            r.get("reference")
            or r.get("object")
            or r.get("landmark")
            or r.get("target")
            or r.get("name")
        )
        if not ref:
            continue
        rel_type = r.get("type") or r.get("relation_type") or "near"
        out.append(Relation(str(rel_type), str(ref)[:120]))
    return out


# ==================== R2R ====================

def process_r2r(data_dir: str, output_dir: str, splits: List[str], llm_client=None):
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        input_path = os.path.join(data_dir, split, f"{split}.json.gz")
        if not os.path.exists(input_path):
            print(f"  [SKIP] {input_path}")
            continue

        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        episodes = data.get("episodes", data if isinstance(data, list) else [])
        results = {}

        for i, ep in enumerate(episodes):
            ep_id = ep.get("episode_id", i)
            instr = ep.get("instruction", {})
            if isinstance(instr, dict):
                text = instr.get("instruction_text", "")
            else:
                text = str(instr)

            ig = InstructionGraph(goal_type="route")
            if llm_client and text:
                try:
                    parsed = llm_client.parse_instruction(text)
                    sg_list = parsed if isinstance(parsed, list) else parsed.get("sub_goals", [parsed])
                except Exception:
                    sg_list = None
            else:
                sg_list = None

            if not sg_list:
                sg_list = [{
                    "action": "go_forward",
                    "landmark": text[:80] if text else None,
                    "spatial_relation": "towards",
                    "implied_room": None,
                    "termination_condition": "end",
                }]

            for j, sg_data in enumerate(sg_list):
                if isinstance(sg_data, str):
                    continue
                ig.sub_goals.append(SubGoal(
                    id=j,
                    action=sg_data.get("action", "go_forward"),
                    landmark=sg_data.get("landmark"),
                    spatial_relation=sg_data.get("spatial_relation", "at"),
                    implied_room=sg_data.get("implied_room"),
                    termination_condition=sg_data.get("termination_condition", ""),
                ))

            results[str(ep_id)] = ig.to_dict()

            if (i + 1) % 500 == 0:
                print(f"  [{split}] {i+1}/{len(episodes)}")

        out_path = os.path.join(output_dir, f"{split}_goal_graphs.json")
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  [{split}] {len(results)} episodes -> {out_path}")


# ==================== GOAT ====================

def process_goat(goat_root: str, output_dir: str, splits: List[str], llm_client=None):
    """goat_root: .../goat_bench/hm3d/v1"""
    os.makedirs(output_dir, exist_ok=True)

    split_map = {
        "train": "train",
        "val": "val_seen",
        "val_seen": "val_seen",
        "val_unseen": "val_unseen",
        "val_seen_synonyms": "val_seen_synonyms",
    }

    for split in splits:
        folder = split_map.get(split, split)
        content_dir = os.path.join(goat_root, folder, "content")
        if not os.path.isdir(content_dir):
            print(f"  [SKIP] {content_dir}")
            continue

        results = {}
        scene_files = sorted(glob.glob(os.path.join(content_dir, "*.json.gz")))
        ep_count = 0

        for si, sf in enumerate(scene_files):
            scene_name = os.path.basename(sf).replace(".json.gz", "")
            with gzip.open(sf, "rt", encoding="utf-8") as f:
                data = json.load(f)

            goals = data.get("goals", {})
            for ep in data.get("episodes", []):
                ep_id = ep.get("episode_id", 0)
                key = f"{scene_name}_{ep_id}"
                ig = InstructionGraph(goal_type="object_goal")

                for task in ep.get("tasks", []):
                    if len(task) < 2:
                        continue
                    category, modality = task[0], task[1]
                    ref = task[2] if len(task) > 2 else None
                    try:
                        gn = _goal_node_from_goat_task(category, modality, ref, goals, llm_client)
                    except Exception as e:
                        print(f"  [WARN] GOAT parse failed {key}: {e}", flush=True)
                        continue
                    ig.goal_nodes.append(gn)

                if ig.goal_nodes:
                    results[key] = ig.to_dict()
                    ep_count += 1

            if (si + 1) % 5 == 0 or si == len(scene_files) - 1:
                print(f"  [{folder}] {si+1}/{len(scene_files)} scenes, {ep_count} eps", flush=True)

        out_path = os.path.join(output_dir, f"{folder}_goal_graphs.json")
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  [{folder}] DONE {ep_count} episodes from {len(scene_files)} scenes -> {out_path}", flush=True)


# ==================== SOON / FAO ====================

SOON_FILES = {
    "train": "train.json",
    "val_unseen_instrs": "val_unseen_instrs.json",
    "val_unseen_house": "val_unseen_house.json",
    "test_release": "test_release.json",
}


def _split_csv_phrases(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    return [p.strip() for p in str(text).replace("\n", " ").split(",") if p.strip()]


def _relations_from_spatial(text: str) -> List[Relation]:
    if not text or not str(text).strip():
        return []
    t = str(text).strip().lower()
    rules = [
        ("left_of", ["left of", "left side"]),
        ("right_of", ["right of", "right side"]),
        ("in_front_of", ["in front", "opposite to", "opposite"]),
        ("behind", ["behind"]),
        ("on", [" on ", "above"]),
        ("under", ["under", "below"]),
        ("near", ["near", "next to", "beside"]),
    ]
    for rel_type, keywords in rules:
        if any(k in t for k in keywords):
            return [Relation(rel_type, str(text).strip()[:120])]
    return [Relation("near", str(text).strip()[:120])]


def _goal_node_from_soon_layers(
    layers: List[str],
    obj_name: Optional[str] = None,
    llm_client=None,
) -> GoalNode:
    """Map FAO 6-layer instruction group to GoalNode."""
    layers = list(layers)
    attr_text = layers[0] if len(layers) > 0 else ""
    rel_text = layers[1] if len(layers) > 1 else ""
    room_text = layers[2] if len(layers) > 2 else ""
    landmark_text = layers[3] if len(layers) > 3 else ""
    full_text = layers[4] if len(layers) > 4 else ""

    attributes = _split_csv_phrases(attr_text)
    relations = _relations_from_spatial(rel_text)
    room_prior = [str(room_text).strip()] if str(room_text).strip() else []
    landmarks = [str(landmark_text).strip()] if str(landmark_text).strip() else []
    raw_description = str(full_text).strip()[:500] if str(full_text).strip() else None
    target = obj_name or "object"

    if llm_client and str(full_text).strip():
        parsed = _parse_description(str(full_text).strip(), llm_client)
        if parsed:
            attrs, rp, lms, rels, parsed_target = _apply_parsed_description(parsed)
            if parsed_target:
                target = parsed_target
            if attrs:
                attributes = attrs
            if rp:
                room_prior = rp
            if lms:
                landmarks = lms
            if rels:
                relations = rels

    return GoalNode(
        target_object=target,
        description=raw_description,
        attributes=attributes,
        room_prior=room_prior,
        landmarks=landmarks,
        relations=relations,
        goal_type="description",
    )


def _iter_soon_instruction_groups(item: dict) -> List[Tuple[List[str], int]]:
    """Yield (layer_strings, group_index). Handles flat test_release vs nested train/val."""
    instructions = item.get("instructions") or []
    if not instructions:
        return []
    if isinstance(instructions[0], str):
        return [(instructions, 0)]
    groups: List[Tuple[List[str], int]] = []
    for gi, group in enumerate(instructions):
        if isinstance(group, list) and group:
            groups.append((group, gi))
    return groups


def _soon_episode_key(item: dict, group: List[str], group_idx: int, item_idx: int) -> str:
    scan = item.get("scan")
    if not scan and item.get("bboxes"):
        scan = item["bboxes"][0].get("scan", "unknown")
    if not scan:
        scan = "unknown"
    instr_id = item.get("instr_id", item_idx)
    uid = str(group[5]) if len(group) > 5 else str(group_idx)
    return f"{scan}_{instr_id}_{uid}"


def process_soon(
    soon_dir: str,
    output_dir: str,
    splits: Optional[List[str]] = None,
    llm_client=None,
):
    """Process FAO JSON files under data/datasets/soon/."""
    os.makedirs(output_dir, exist_ok=True)

    if splits:
        selected = {}
        for name in splits:
            if name in SOON_FILES:
                selected[name] = SOON_FILES[name]
            elif name.endswith(".json"):
                key = name.replace(".json", "")
                selected[key] = name
            else:
                selected[name] = SOON_FILES.get(name, f"{name}.json")
    else:
        selected = dict(SOON_FILES)

    for split_name, json_name in selected.items():
        input_path = os.path.join(soon_dir, json_name)
        if not os.path.isfile(input_path):
            print(f"  [SKIP] {input_path}")
            continue

        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        results: Dict[str, dict] = {}
        for item_idx, item in enumerate(data):
            if (item_idx + 1) % 50 == 0:
                print(f"  [{split_name}] {item_idx+1}/{len(data)}")

            obj_name = None
            bboxes = item.get("bboxes") or []
            if bboxes:
                obj_name = bboxes[0].get("obj_name")

            for group, gi in _iter_soon_instruction_groups(item):
                try:
                    gn = _goal_node_from_soon_layers(group, obj_name, llm_client)
                    key = _soon_episode_key(item, group, gi, item_idx)
                    ig = InstructionGraph(goal_type="object_goal")
                    ig.goal_nodes.append(gn)
                    results[key] = ig.to_dict()
                except Exception as e:
                    key = _soon_episode_key(item, group, gi, item_idx)
                    print(f"  [WARN] SOON parse failed key={key}: {type(e).__name__}: {e}")
                    continue

        out_path = os.path.join(output_dir, f"{split_name}_goal_graphs.json")
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  [{split_name}] {len(results)} goals from {len(data)} items -> {out_path}")


# ==================== CLIP embeddings ====================

def _resolve_clip_pretrained(clip_cache: str) -> str:
    """Prefer local checkpoint to avoid HuggingFace download timeouts."""
    cache = Path(clip_cache)
    candidates = [
        cache / "ViT-B-32-laion2b_s34b_b79k.bin",
        cache / "open_clip_pytorch_model.bin",
        DATA_ROOT.parent / "models" / "clip" / "ViT-B-32-laion2b_s34b_b79k.bin",
        DATA_ROOT.parent / "models" / "open_clip_pytorch_model.bin",
    ]
    for path in candidates:
        if path.is_file():
            print(f"  Using local CLIP weights: {path}")
            return str(path)
    print(f"  No local CLIP weights under {clip_cache}; will try HuggingFace download")
    return "laion2b_s34b_b79k"


def embed_goal_graphs(
    goal_graph_dir: str,
    clip_cache: Optional[str] = None,
    include_train: bool = False,
):
    try:
        import open_clip
        import torch
    except ImportError:
        print("open_clip not installed, skip embedding")
        return

    if clip_cache is None:
        clip_cache = str(DATA_ROOT.parent / "models" / "clip")
    os.makedirs(clip_cache, exist_ok=True)
    pretrained = _resolve_clip_pretrained(clip_cache)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=pretrained, cache_dir=clip_cache,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    text_cache: Dict[str, List[float]] = {}

    def encode_texts(texts: List[str]) -> Dict[str, np.ndarray]:
        unique = list({t for t in texts if t and t not in text_cache})
        if not unique:
            return {}
        batch_size = 64
        for i in range(0, len(unique), batch_size):
            batch = unique[i : i + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for t, f in zip(batch, feats.cpu().numpy()):
                text_cache[t] = f.tolist()
        return {t: np.array(text_cache[t], dtype=np.float32) for t in texts if t in text_cache}

    json_files = sorted(glob.glob(os.path.join(goal_graph_dir, "**", "*_goal_graphs.json"), recursive=True))
    for jf in json_files:
        basename = os.path.basename(jf)
        # Skip train / large splits unless --embed-clip-train
        if not include_train and (
            "train_goal_graphs" in basename
            or "test_release_goal_graphs" in basename
        ):
            print(f"  [SKIP embed] {jf} (use --embed-clip-train to include)")
            continue
        print(f"  Embedding {jf}...")
        with open(jf) as f:
            data = json.load(f)

        texts_to_encode = []
        for ep_data in data.values():
            if ep_data.get("goal_type") == "route":
                for sg in ep_data.get("sub_goals", []):
                    if sg.get("landmark") and isinstance(sg["landmark"], str):
                        texts_to_encode.append(sg["landmark"])
                    if sg.get("implied_room") and isinstance(sg["implied_room"], str):
                        texts_to_encode.append(sg["implied_room"])
            else:
                for gn in ep_data.get("goal_nodes", []):
                    t = gn.get("target_object", "")
                    if isinstance(t, str) and t:
                        # Encode enriched label: "red rectangular picture" for better CLIP matching
                        attrs = gn.get("attributes", [])
                        if isinstance(attrs, list) and attrs:
                            enriched = " ".join(str(a) for a in attrs[:3]) + " " + t
                        else:
                            enriched = t
                        texts_to_encode.append(enriched)
                    for rp in gn.get("room_prior", []):
                        if isinstance(rp, str) and rp:
                            texts_to_encode.append(rp)
                    for lm in gn.get("landmarks", []):
                        if isinstance(lm, str) and lm:
                            texts_to_encode.append(lm)

        encode_texts(texts_to_encode)

        for ep_data in data.values():
            if ep_data.get("goal_type") == "route":
                for sg in ep_data.get("sub_goals", []):
                    lm = sg.get("landmark")
                    if isinstance(lm, str) and lm in text_cache:
                        sg["landmark_embedding"] = text_cache[lm]
            else:
                for gn in ep_data.get("goal_nodes", []):
                    t = gn.get("target_object", "")
                    if isinstance(t, str) and t:
                        attrs = gn.get("attributes", [])
                        if isinstance(attrs, list) and attrs:
                            enriched = " ".join(str(a) for a in attrs[:3]) + " " + t
                        else:
                            enriched = t
                        if enriched in text_cache:
                            gn["target_embedding"] = text_cache[enriched]
                    rp = gn.get("room_prior", [])
                    if rp:
                        rp_emb = [text_cache[r] for r in rp
                                  if isinstance(r, str) and r in text_cache]
                        if rp_emb:
                            gn["room_prior_embeddings"] = rp_emb

        with open(jf, "w") as f:
            json.dump(data, f)
        print(f"    saved with embeddings")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["r2r", "goat", "soon", "all"], default="all")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["r2r", "goat", "soon"],
        help="Run only these tasks (overrides --task), e.g. --tasks goat soon",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--r2r-dir", default=str(DATA_ROOT / "datasets/r2r"))
    parser.add_argument("--goat-dir", default=str(DATA_ROOT / "datasets/goat/hm3d/v1"))
    parser.add_argument("--soon-dir", default=str(DATA_ROOT / "datasets/soon"))
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "goal_graphs"))
    parser.add_argument("--llm-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--embed-only", action="store_true", help="Only run CLIP embedding on existing JSON")
    parser.add_argument("--embed-clip", action="store_true", help="Add CLIP text embeddings to output JSON")
    parser.add_argument("--embed-clip-train", action="store_true", help="Also embed train / test_release splits")
    parser.add_argument(
        "--clip-cache",
        default=str(DATA_ROOT.parent / "models" / "clip"),
        help="Directory with ViT-B-32-laion2b_s34b_b79k.bin (avoids HuggingFace download)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip large splits (train, test_release); only val/dev for each task",
    )
    args = parser.parse_args()

    r2r_default = ["val_seen", "val_unseen"] if args.no_train else ["train", "val_seen", "val_unseen"]
    goat_default = (
        ["val_seen", "val_unseen", "val_seen_synonyms"]
        if args.no_train
        else ["train", "val_seen", "val_unseen", "val_seen_synonyms"]
    )
    soon_default = (
        ["val_unseen_instrs", "val_unseen_house"]
        if args.no_train
        else None
    )

    if args.embed_only:
        print("\n=== CLIP embeddings (embed-only) ===")
        embed_goal_graphs(args.output_dir, clip_cache=args.clip_cache, include_train=args.embed_clip_train)
        print("\nDone!")
        return

    llm_client = None
    if not args.no_llm:
        try:
            llm_client = GoalGraphLLM(base_url=args.llm_url, model=args.llm_model)
            if llm_client.is_available():
                print(f"LLM API OK at {args.llm_url} (model={args.llm_model})")
                print("  (vLLM server runs in vllm_server env; this script only uses HTTP)")
            else:
                print(f"LLM API not reachable at {args.llm_url} -> fallback mode")
                print("  Start server: conda activate vllm_server && bash ETPNav/scripts/start_vllm.sh")
                llm_client = None
        except ImportError as e:
            print(f"openai not installed ({e}) -> fallback mode; use: conda activate conftopo")

    active_tasks = args.tasks if args.tasks else (
        ["r2r", "goat", "soon"] if args.task == "all" else [args.task]
    )

    if "r2r" in active_tasks:
        print("\n=== R2R ===")
        splits = [args.split] if args.split else r2r_default
        process_r2r(args.r2r_dir, os.path.join(args.output_dir, "r2r"), splits, llm_client)

    if "goat" in active_tasks:
        print("\n=== GOAT ===")
        splits = [args.split] if args.split else goat_default
        process_goat(args.goat_dir, os.path.join(args.output_dir, "goat"), splits, llm_client)

    if "soon" in active_tasks:
        soon_dir = args.soon_dir
        if os.path.isdir(soon_dir):
            print("\n=== SOON ===")
            soon_splits = [args.split] if args.split else soon_default
            process_soon(
                soon_dir,
                os.path.join(args.output_dir, "soon"),
                splits=soon_splits,
                llm_client=llm_client,
            )
        else:
            print(f"\n=== SOON [SKIP] no data at {soon_dir} ===")

    if args.embed_clip:
        print("\n=== CLIP embeddings ===")
        embed_goal_graphs(args.output_dir, clip_cache=args.clip_cache, include_train=args.embed_clip_train)

    print("\nDone!")


if __name__ == "__main__":
    main()
