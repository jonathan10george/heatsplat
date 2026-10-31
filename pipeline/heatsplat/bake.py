"""Bake per-Gaussian temperature from posed thermal frames (the real-capture path).

Each Gaussian centre is projected into every thermal frame. Samples that are occluded (a nearer
Gaussian lands on the same pixel) are rejected with a coarse depth buffer, and the surviving
samples are averaged per Gaussian. Poses are camera-to-world 4x4 matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEPTH_BUFFER_SCALE = 4  # depth test at 1/4 image resolution
DEPTH_TOLERANCE_M = 0.08


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class BakeResult:
    temperature: np.ndarray  # (N,) degrees C; NaN where never observed
    coverage: np.ndarray  # (N,) number of frames that observed each Gaussian


def opengl_to_opencv(c2w: np.ndarray) -> np.ndarray:
    """Convert nerfstudio / Blender style poses (Y up, Z back) to OpenCV (Y down, Z forward)."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ flip


def project(positions: np.ndarray, c2w: np.ndarray, K: Intrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (u, v, depth) in an OpenCV camera. Points behind the camera get depth <= 0."""
    w2c = np.linalg.inv(c2w)
    cam = positions @ w2c[:3, :3].T + w2c[:3, 3]
    depth = cam[:, 2]
    safe = np.where(depth > 1e-6, depth, 1e-6)
    u = K.fx * cam[:, 0] / safe + K.cx
    v = K.fy * cam[:, 1] / safe + K.cy
    return u, v, depth


def _visible_samples(u: np.ndarray, v: np.ndarray, depth: np.ndarray, K: Intrinsics) -> np.ndarray:
    in_frame = (depth > 1e-6) & (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
    idx = np.flatnonzero(in_frame)
    if idx.size == 0:
        return idx
    bw, bh = K.width // DEPTH_BUFFER_SCALE + 1, K.height // DEPTH_BUFFER_SCALE + 1
    cell = (v[idx] / DEPTH_BUFFER_SCALE).astype(int) * bw + (u[idx] / DEPTH_BUFFER_SCALE).astype(int)
    nearest = np.full(bw * bh, np.inf)
    np.minimum.at(nearest, cell, depth[idx])
    return idx[depth[idx] <= nearest[cell] + DEPTH_TOLERANCE_M]


def bake_temperature(positions: np.ndarray, poses_c2w: np.ndarray, K: Intrinsics, frames: np.ndarray) -> BakeResult:
    """positions (N,3); poses (F,4,4) OpenCV convention; frames (F,H,W) in degrees C."""
    if poses_c2w.shape[0] != frames.shape[0]:
        raise ValueError("one pose per thermal frame is required")
    if frames.shape[1:] != (K.height, K.width):
        raise ValueError(f"frames are {frames.shape[1:]}, intrinsics say {(K.height, K.width)}")
    total = np.zeros(len(positions), dtype=np.float64)
    count = np.zeros(len(positions), dtype=np.int32)
    for c2w, frame in zip(poses_c2w, frames):
        u, v, depth = project(positions.astype(np.float64), c2w, K)
        idx = _visible_samples(u, v, depth, K)
        if idx.size == 0:
            continue
        samples = frame[v[idx].astype(int), u[idx].astype(int)]
        total[idx] += samples
        count[idx] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        temperature = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return BakeResult(temperature.astype(np.float32), count)


def fill_unobserved(positions: np.ndarray, temperature: np.ndarray, k: int = 8) -> np.ndarray:
    """Replace NaN temperatures with the mean of the nearest observed Gaussians."""
    from scipy.spatial import cKDTree

    observed = np.isfinite(temperature)
    if observed.all() or not observed.any():
        return temperature
    tree = cKDTree(positions[observed])
    _, idx = tree.query(positions[~observed], k=min(k, int(observed.sum())))
    filled = temperature.copy()
    filled[~observed] = temperature[observed][idx].mean(axis=1)
    return filled
