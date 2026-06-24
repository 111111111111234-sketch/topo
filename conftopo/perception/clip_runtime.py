"""Runtime CLIP encoder for ConfTopo semantic smoke tests."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Union

import numpy as np
from PIL import Image

GOAT_OFFICIAL_IMAGE_CLIP_MODEL = "RN50"
DEFAULT_CLIP_MODEL = "ViT-B/32"


class ClipRuntimeEncoder:
    """Small wrapper around OpenAI CLIP for image/text embeddings."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str = "auto"):
        import clip
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self._clip = clip
        self._torch = torch
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def encode_image(self, rgb: np.ndarray) -> np.ndarray:
        """Encode one RGB/RGBA frame into a normalized CLIP image embedding."""
        arr = np.asarray(rgb)
        if arr.ndim != 3:
            raise ValueError(f"Expected HxWxC image, got shape {arr.shape}")
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        image = Image.fromarray(arr)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            embed = self.model.encode_image(tensor)
            embed = embed / embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return embed.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def encode_text(self, labels: Iterable[str], prompt: str = "a photo of a {}") -> np.ndarray:
        """Encode labels/prompts into normalized CLIP text embeddings."""
        labels = list(labels)
        if not labels:
            return np.empty((0, 0), dtype=np.float32)
        texts: List[str] = [prompt.format(label) for label in labels]
        tokens = self._clip.tokenize(texts).to(self.device)
        with self._torch.no_grad():
            embeds = self.model.encode_text(tokens)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return embeds.detach().cpu().numpy().astype(np.float32)


class GoatModalityClipEncoder:
    """GOAT-aware dual CLIP: ViT-B/32 for object/description, RN50 for image goals."""

    def __init__(
        self,
        text_model: str = DEFAULT_CLIP_MODEL,
        image_model: str = GOAT_OFFICIAL_IMAGE_CLIP_MODEL,
        device: str = "auto",
    ):
        self.text_model_name = text_model
        self.image_model_name = image_model
        self._text_encoder = ClipRuntimeEncoder(text_model, device)
        if image_model == text_model:
            self._image_encoder = self._text_encoder
        else:
            self._image_encoder = ClipRuntimeEncoder(image_model, device)

    @property
    def model_name(self) -> str:
        return self.text_model_name

    def encode_image(self, rgb: np.ndarray, goal_type: str = "category") -> np.ndarray:
        if (goal_type or "category").lower() == "image":
            return self._image_encoder.encode_image(rgb)
        return self._text_encoder.encode_image(rgb)

    def encode_text(self, labels: Iterable[str], prompt: str = "a photo of a {}") -> np.ndarray:
        return self._text_encoder.encode_text(labels, prompt=prompt)


def agent_current_goal_type(agent: Any) -> str:
    """Return active GoalNode.goal_type from a ConfTopo GOAT agent."""
    goal_mgr = getattr(agent, "goal_manager", None)
    if goal_mgr is not None:
        current = getattr(goal_mgr, "current_goal", None)
        if current is not None:
            return getattr(current, "goal_type", None) or "category"
    ig = getattr(agent, "instruction_graph", None)
    if ig is not None:
        current = ig.get_current_goal()
        if current is not None:
            return getattr(current, "goal_type", None) or "category"
    return "category"


def encode_agent_rgb_embed(
    encoder: Optional[Union[ClipRuntimeEncoder, GoatModalityClipEncoder]],
    rgb: np.ndarray,
    agent: Any,
    *,
    goal_type: Optional[str] = None,
    use_placeholder: bool = False,
    placeholder_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Encode RGB for the agent step using the modality-appropriate CLIP model."""
    if use_placeholder:
        if placeholder_fn is None:
            raise ValueError("placeholder_fn is required when use_placeholder=True")
        return placeholder_fn(rgb)
    if encoder is None:
        raise ValueError("encoder is required when use_placeholder=False")
    # Light perception (room/landmark/category) always uses the text CLIP visual space.
    if isinstance(encoder, GoatModalityClipEncoder):
        return encoder._text_encoder.encode_image(rgb)
    return encoder.encode_image(rgb)


def encode_agent_image_goal_embed(
    encoder: Optional[Union[ClipRuntimeEncoder, GoatModalityClipEncoder]],
    rgb: np.ndarray,
    agent: Any,
    *,
    goal_type: Optional[str] = None,
) -> Optional[np.ndarray]:
    """RN50 image embedding for image-goal CLIP matching (1024-dim)."""
    if encoder is None or not isinstance(encoder, GoatModalityClipEncoder):
        return None
    modality = (goal_type or agent_current_goal_type(agent) or "category").lower()
    if modality != "image":
        return None
    return encoder.encode_image(rgb, goal_type="image")
