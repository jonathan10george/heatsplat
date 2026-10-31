"""Thermal colour palettes. No matplotlib dependency: small interpolated LUTs."""

from __future__ import annotations

import numpy as np

# FLIR-style "ironbow": black -> purple -> red -> orange -> yellow -> white.
_IRONBOW_STOPS = np.array(
    [
        [0.00, 0.00, 0.00, 0.00],
        [0.15, 0.20, 0.00, 0.45],
        [0.35, 0.60, 0.00, 0.60],
        [0.55, 0.90, 0.25, 0.10],
        [0.75, 1.00, 0.65, 0.00],
        [0.90, 1.00, 0.92, 0.35],
        [1.00, 1.00, 1.00, 1.00],
    ],
    dtype=np.float32,
)


def ironbow(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB in [0, 1] with the ironbow palette. Shape (N,) -> (N, 3)."""
    t = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    stops = _IRONBOW_STOPS[:, 0]
    rgb = np.stack([np.interp(t, stops, _IRONBOW_STOPS[:, c + 1]) for c in range(3)], axis=1)
    return rgb.astype(np.float32)


def normalize(values: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Robustly rescale values to [0, 1] using percentile clipping."""
    v = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(v, [low_pct, high_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def grayscale(rgb: np.ndarray, brightness: float = 1.0) -> np.ndarray:
    """Luma-weighted grayscale of (N, 3) RGB, scaled by brightness."""
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.repeat((luma * brightness)[:, None], 3, axis=1).astype(np.float32)
