"""Phase2 semantic acceptance utilities."""

from __future__ import annotations

from typing import Iterable


def auto_threshold(scores: Iterable[float], min_threshold: float = 0.045, ratio: float = 0.85) -> float:
    """Return a stable per-run threshold from observed CLIP scores."""
    vals = [float(s) for s in scores if s is not None]
    if not vals:
        return float(min_threshold)
    return float(max(min_threshold, max(vals) * ratio))


def score_summary(scores: Iterable[float]) -> dict:
    vals = [float(s) for s in scores if s is not None]
    if not vals:
        return {"count": 0, "max": 0.0, "mean": 0.0}
    return {"count": len(vals), "max": max(vals), "mean": sum(vals) / len(vals)}
