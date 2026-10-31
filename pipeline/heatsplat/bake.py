"""Bake per-Gaussian temperature from posed thermal frames (the real-capture path).

Each Gaussian centre is projected into every thermal frame. Samples that are occluded (a nearer
Gaussian lands on the same pixel) are rejected with a coarse depth buffer, and the surviving
samples are averaged per Gaussian. Poses are camera-to-world 4x4 matrices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

DEPTH_BUFFER_SCALE = 4  # depth test at 1/4 image resolution
DEPTH_TOLERANCE_M = 0.08
DEPTH_TOLERANCE_CELLS = 3.0  # a surface seen obliquely spans several cell widths of depth within one cell
DEPTH_TOLERANCE_MAX_M = 0.3  # never looser than typical recess / eave offsets
OCCLUDER_MIN_OPACITY = 0.1  # near-transparent floaters must not hide the wall behind them
FILL_MAX_DISTANCE_M = 1.0  # unobserved Gaussians farther than this from any observation get the scene median


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


def _visible_samples(u: np.ndarray, v: np.ndarray, depth: np.ndarray, K: Intrinsics, occluder: np.ndarray | None = None) -> np.ndarray:
    in_frame = (depth > 1e-6) & (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
    idx = np.flatnonzero(in_frame)
    if idx.size == 0:
        return idx
    bw, bh = K.width // DEPTH_BUFFER_SCALE + 1, K.height // DEPTH_BUFFER_SCALE + 1
    cell = (v[idx] / DEPTH_BUFFER_SCALE).astype(int) * bw + (u[idx] / DEPTH_BUFFER_SCALE).astype(int)
    nearest = np.full(bw * bh, np.inf)
    writers = idx if occluder is None else idx[occluder[idx]]
    writer_cells = cell if occluder is None else cell[occluder[idx]]
    np.minimum.at(nearest, writer_cells, depth[writers])
    # Tolerance grows with the cell's footprint in metres so grazing-angle surfaces are kept,
    # while surfaces metres behind an occluder are still rejected.
    cell_world_m = DEPTH_BUFFER_SCALE * depth[idx] / K.fx
    tolerance = np.clip(DEPTH_TOLERANCE_CELLS * cell_world_m, DEPTH_TOLERANCE_M, DEPTH_TOLERANCE_MAX_M)
    return idx[depth[idx] <= nearest[cell] + tolerance]


def bake_temperature(positions: np.ndarray, poses_c2w: np.ndarray, K: Intrinsics, frames: np.ndarray, opacities: np.ndarray | None = None) -> BakeResult:
    """positions (N,3); poses (F,4,4) OpenCV convention; frames (F,H,W) in degrees C.

    `opacities` (N,), when given, keeps near-transparent Gaussians out of the depth buffer so
    floaters do not occlude real surfaces (they are still sampled themselves).
    """
    if poses_c2w.shape[0] != frames.shape[0]:
        raise ValueError("one pose per thermal frame is required")
    if frames.shape[1:] != (K.height, K.width):
        raise ValueError(f"frames are {frames.shape[1:]}, intrinsics say {(K.height, K.width)}")
    occluder = None if opacities is None else np.asarray(opacities) >= OCCLUDER_MIN_OPACITY
    total = np.zeros(len(positions), dtype=np.float64)
    count = np.zeros(len(positions), dtype=np.int32)
    for c2w, frame in zip(poses_c2w, frames):
        u, v, depth = project(positions.astype(np.float64), c2w, K)
        idx = _visible_samples(u, v, depth, K, occluder)
        if idx.size == 0:
            continue
        samples = frame[v[idx].astype(int), u[idx].astype(int)]
        finite = np.isfinite(samples)  # a dead pixel must not poison the Gaussian for all frames
        total[idx[finite]] += samples[finite]
        count[idx[finite]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        temperature = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return BakeResult(temperature.astype(np.float32), count)


def fill_unobserved(positions: np.ndarray, temperature: np.ndarray, k: int = 8, max_distance: float = FILL_MAX_DISTANCE_M) -> np.ndarray:
    """Replace NaN temperatures: the mean of the nearest observed Gaussians when one is within
    `max_distance`, otherwise the scene median (neutral, so it never reads as a leak)."""
    from scipy.spatial import cKDTree

    observed = np.isfinite(temperature)
    if observed.all() or not observed.any():
        return temperature
    tree = cKDTree(positions[observed])
    dist, idx = tree.query(positions[~observed], k=min(k, int(observed.sum())))
    dist, idx = np.atleast_2d(dist.T).T, np.atleast_2d(idx.T).T  # k == 1 returns 1-D arrays
    neighbour_temps = np.where(dist <= max_distance, temperature[observed][idx], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected: they take the median below
        local = np.nanmean(neighbour_temps, axis=1)
    filled = temperature.copy()
    filled[~observed] = np.where(np.isfinite(local), local, float(np.median(temperature[observed])))
    return filled
