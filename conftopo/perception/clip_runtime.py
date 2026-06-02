"""Runtime CLIP encoder for ConfTopo semantic smoke tests."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from PIL import Image


class ClipRuntimeEncoder:
    """Small wrapper around OpenAI CLIP for image/text embeddings."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
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
