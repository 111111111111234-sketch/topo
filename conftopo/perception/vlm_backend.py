"""VLM backend abstraction, fake backend for testing, and Qwen3-VL backend."""

from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from conftopo.perception.vlm_prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_vlm_json,
)

logger = logging.getLogger(__name__)


class VLMBackendBase(ABC):
    """Abstract interface for Vision-Language Model backends."""

    @abstractmethod
    def query(
        self,
        rgb: np.ndarray,
        goal_text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an image + goal description to the VLM and return parsed JSON.

        Returns a dict with the canonical schema::

            {
                "room": {"label": str, "confidence": float},
                "objects": [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...],
                "goal_visible": bool,
                "goal_reason": str,
                "portals": [str, ...],
                "scene_summary": str,
                "uncertainty": float,
            }
        """


class FakeVLMBackend(VLMBackendBase):
    """Deterministic fake backend for unit / integration tests.

    Always returns a fixed response.  Optionally accepts a ``response``
    dict to customise the output.
    """

    def __init__(self, response: Optional[Dict[str, Any]] = None):
        self._response = response or {
            "room": {"label": "unknown", "confidence": 0.5},
            "objects": [],
            "goal_visible": False,
            "goal_reason": "",
            "portals": [],
            "scene_summary": "",
            "uncertainty": 0.5,
        }
        self.calls: int = 0

    def query(
        self,
        rgb: np.ndarray,
        goal_text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.calls += 1
        return dict(self._response)


def _rgb_to_base64_jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    """Encode an HWC uint8 RGB array to a base64 JPEG string."""
    from PIL import Image  # lazy import to avoid hard dependency

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Qwen3VLBackend(VLMBackendBase):
    """Qwen3-VL backend via OpenAI-compatible chat API (vLLM / SGLang).

    Expects the VLM server to be running at ``api_base`` and serving the
    model named ``model``.  Uses the ``openai`` Python SDK under the hood.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        timeout: float = 5.0,
        max_tokens: int = 1024,
    ):
        self._api_base = api_base
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._client: Any = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for Qwen3VLBackend. "
                "Install it with: pip install openai"
            )
        self._client = OpenAI(
            base_url=self._api_base,
            api_key="EMPTY",
            timeout=self._timeout,
        )

    def query(
        self,
        rgb: np.ndarray,
        goal_text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_client()
        b64_img = _rgb_to_base64_jpeg(rgb)
        user_text = build_user_prompt(goal_text, context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=0.1,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("VLM query failed: %s", exc)
            raw_text = ""

        return parse_vlm_json(raw_text)
