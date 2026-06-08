"""Object-level heavy perception interface.

Heavy perception is optional at runtime. The default GroundingDINO backend is
loaded lazily so Phase 2 CLIP-only execution and tests do not require model
weights or the GroundingDINO package.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence
import hashlib
import os

import numpy as np


@dataclass
class ObjectObservation:
    """Normalized object-level detection result."""

    label: str
    bbox: List[float]
    confidence: float
    embedding: Optional[np.ndarray] = None
    source: str = "heavy"
    view_heading: float = 0.0
    step_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "bbox": [float(v) for v in self.bbox],
            "confidence": float(self.confidence),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "source": self.source,
            "view_heading": float(self.view_heading),
            "step_id": int(self.step_id),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectObservation":
        embedding = data.get("embedding")
        return cls(
            label=str(data.get("label", "")),
            bbox=[float(v) for v in data.get("bbox", [0, 0, 0, 0])],
            confidence=float(data.get("confidence", 0.0)),
            embedding=np.array(embedding, dtype=np.float32) if embedding is not None else None,
            source=str(data.get("source", "heavy")),
            view_heading=float(data.get("view_heading", 0.0)),
            step_id=int(data.get("step_id", 0)),
        )


class GroundingDINOBackend:
    """Lazy GroundingDINO adapter.

    This class intentionally keeps imports out of module import time. A caller
    may provide an already constructed backend with the same `detect` method.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        self.model = model
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._predict_fn = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        config_path = self.config_path or os.environ.get("GROUNDINGDINO_CONFIG")
        checkpoint_path = self.checkpoint_path or os.environ.get("GROUNDINGDINO_CHECKPOINT")
        if not config_path or not checkpoint_path:
            raise RuntimeError(
                "GroundingDINO backend requires config/checkpoint paths "
                "or GROUNDINGDINO_CONFIG/GROUNDINGDINO_CHECKPOINT"
            )
        try:
            from groundingdino.util.inference import load_model, predict
        except ImportError as exc:
            raise RuntimeError("GroundingDINO package is not installed") from exc
        self.model = load_model(config_path, checkpoint_path, device=self.device)
        self._predict_fn = predict

    def detect(self, image: Any, labels: Sequence[str]) -> List[Dict[str, Any]]:
        self._ensure_model()
        if hasattr(self.model, "detect"):
            return self.model.detect(image=image, labels=list(labels))
        if self._predict_fn is None:
            from groundingdino.util.inference import predict
            self._predict_fn = predict
        caption = " . ".join(str(label) for label in labels)
        boxes, logits, phrases = self._predict_fn(
            model=self.model,
            image=np.asarray(image),
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes_np = _as_numpy(boxes)
        logits_np = _as_numpy(logits)
        height, width = np.asarray(image).shape[:2]
        rows = []
        for box, score, phrase in zip(boxes_np, logits_np, phrases):
            rows.append({
                "label": _clean_phrase_label(str(phrase), labels),
                "bbox": _cxcywh_to_xyxy(box, width, height),
                "confidence": float(score),
                "source": "groundingdino",
            })
        return rows


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _cxcywh_to_xyxy(box: np.ndarray, width: int, height: int) -> List[float]:
    cx, cy, bw, bh = [float(v) for v in box]
    if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
        cx *= width
        bw *= width
        cy *= height
        bh *= height
    return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]


def _clean_phrase_label(phrase: str, labels: Sequence[str]) -> str:
    lower = phrase.lower()
    for label in labels:
        if str(label).lower() in lower:
            return str(label)
    return phrase.split(",")[0].strip()


class FakeGroundingDINOBackend:
    """Deterministic test backend that does not require model weights."""

    def __init__(self, detections: Optional[Iterable[Dict[str, Any]]] = None):
        self._detections = list(detections) if detections is not None else None
        self.calls = 0

    def detect(self, image: Any, labels: Sequence[str]) -> List[Dict[str, Any]]:
        self.calls += 1
        if self._detections is not None:
            return [dict(row) for row in self._detections]
        rows = []
        for idx, label in enumerate(labels):
            digest = hashlib.sha1(str(label).encode("utf-8")).digest()
            x1 = 5.0 + float(digest[0] % 20)
            y1 = 6.0 + float(digest[1] % 20)
            rows.append({
                "label": str(label),
                "bbox": [x1, y1, x1 + 24.0 + idx, y1 + 18.0 + idx],
                "confidence": 0.75,
                "source": "fake_groundingdino",
            })
        return rows


class HeavyPerceiver:
    """Normalize object-level detections from a heavy backend."""

    def __init__(self, backend: Optional[Any] = None, min_confidence: float = 0.25):
        self.backend = backend if backend is not None else GroundingDINOBackend()
        self.min_confidence = float(min_confidence)

    def perceive(
        self,
        rgb: Any,
        labels: Sequence[str],
        *,
        visual_embedding: Optional[np.ndarray] = None,
        view_heading: float = 0.0,
        step_id: int = 0,
    ) -> List[ObjectObservation]:
        if rgb is None or not labels:
            return []
        raw_rows = self.backend.detect(rgb, labels)
        observations = []
        for row in raw_rows:
            label = str(row.get("label", "")).strip()
            conf = float(row.get("confidence", row.get("score", 0.0)))
            if not label or conf < self.min_confidence:
                continue
            embedding = row.get("embedding")
            if embedding is None:
                embedding = visual_embedding
            observations.append(ObjectObservation(
                label=label,
                bbox=[float(v) for v in row.get("bbox", [0, 0, 0, 0])],
                confidence=conf,
                embedding=np.array(embedding, dtype=np.float32) if embedding is not None else None,
                source=str(row.get("source", "groundingdino")),
                view_heading=float(row.get("view_heading", view_heading)),
                step_id=int(row.get("step_id", step_id)),
            ))
        return observations
