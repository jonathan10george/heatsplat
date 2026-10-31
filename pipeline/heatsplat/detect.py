"""Unsupervised heat-leak detection on a temperature-baked Gaussian scene.

Each Gaussian is compared with the median temperature of its surroundings (a local baseline
estimated from a spatial subsample). Gaussians that are robustly warmer than that baseline are
clustered with DBSCAN; each cluster is a leak candidate with geometry and severity statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from heatsplat.scene import GaussianScene, surface_normals

SIGMA_FLOOR_C = 0.3
MAD_TO_SIGMA = 1.4826
MIN_SUPPORT_NEIGHBOURS = 8
QUERY_CHUNK = 8192
VOXEL_KEY_STRIDE = 1 << 20
SILL_MAX_HEIGHT_M = 0.35
EAVE_BAND_M = 0.6
RING_HOLE_MIN_M = 0.25

CAUSE_SILL = "Gap under a door or at the sill plate"
CAUSE_EAVE = "Air leak at the eave / top plate (attic bypass)"
CAUSE_FRAME = "Failed seal around a window or door frame"
CAUSE_INSULATION = "Missing or settled wall insulation"


@dataclass(frozen=True)
class DetectionParams:
    baseline_radius_m: float = 2.0
    baseline_neighbours: int = 384
    min_delta_c: float = 1.5
    min_zscore: float = 2.5
    cluster_eps_m: float = 0.2
    cluster_min_points: int = 12
    hysteresis_ratio: float = 0.6  # above edge_floor_c, clusters grow down to max(edge_floor_c, min_delta_c * ratio) but must contain a seed above min_delta_c
    edge_floor_c: float = 1.5
    normal_cos_threshold: float = 0.8  # only Gaussians facing within ~37 deg count as 'same surface'
    sign: int = 1  # +1 warm anomalies (exterior scan), -1 cold anomalies (interior scan)

    @property
    def support_voxel_m(self) -> float:
        """Cell size that puts ~baseline_neighbours support points inside baseline_radius_m on a surface."""
        return float(self.baseline_radius_m * np.sqrt(np.pi / self.baseline_neighbours))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Leak:
    id: str
    rank: int
    centroid: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    n_points: int
    mean_delta_c: float
    max_delta_c: float
    area_m2: float
    severity: float  # mean delta (C) x area (m2)
    likely_cause: str
    point_indices: np.ndarray

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("point_indices")
        return d


@dataclass(frozen=True)
class DetectionResult:
    leaks: tuple[Leak, ...]
    residual_c: np.ndarray  # signed per-Gaussian deviation from local baseline
    candidate_mask: np.ndarray
    params: DetectionParams


def voxel_support(positions: np.ndarray, temperature: np.ndarray, normals: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample to one averaged point per voxel. Support density is then the same on every
    surface, so the baseline radius means the same thing on a dense wall and a sparse lawn."""
    keys = np.floor((positions - positions.min(axis=0)) / voxel_m).astype(np.int64)
    flat = (keys[:, 0] * VOXEL_KEY_STRIDE + keys[:, 1]) * VOXEL_KEY_STRIDE + keys[:, 2]
    _, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    mean = lambda a: np.bincount(inverse, a) / counts  # noqa: E731
    pos = np.stack([mean(positions[:, i]) for i in range(3)], axis=1)
    temp = mean(temperature.astype(np.float64))
    nrm = np.stack([mean(normals[:, i]) for i in range(3)], axis=1)
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(length > 1e-6, nrm / np.maximum(length, 1e-9), normals[first])
    return pos, temp.astype(np.float32), nrm.astype(np.float32)


def local_baseline(positions: np.ndarray, temperature: np.ndarray, normals: np.ndarray, params: DetectionParams) -> np.ndarray:
    """Per-Gaussian median temperature of same-surface neighbours within `baseline_radius_m`.

    Neighbours come from a voxel-downsampled support set so the k-NN query stays cheap while
    still spanning a radius large enough to see past a leak-sized patch. Support points whose
    surface normal differs from the query's are ignored, so a wall is never compared with the
    roof that meets it.
    """
    n = len(positions)
    support_pos, support_temp, support_normals = voxel_support(positions, temperature, normals, params.support_voxel_m)
    tree = cKDTree(support_pos)
    k = min(params.baseline_neighbours, len(support_pos))
    global_median = float(np.median(temperature))

    baseline = np.empty(n, dtype=np.float32)
    for start in range(0, n, QUERY_CHUNK):
        stop = min(n, start + QUERY_CHUNK)
        dist, idx = tree.query(positions[start:stop], k=k, distance_upper_bound=params.baseline_radius_m)
        valid = np.isfinite(dist)
        safe_idx = np.minimum(idx, len(support_pos) - 1)
        facing = np.abs(np.einsum("qk,qnk->qn", normals[start:stop], support_normals[safe_idx]))
        valid &= facing >= params.normal_cos_threshold
        temps = np.where(valid, support_temp[safe_idx], np.nan)
        enough = valid.sum(axis=1) >= MIN_SUPPORT_NEIGHBOURS
        with np.errstate(all="ignore"):
            med = np.nanmedian(temps, axis=1)
        baseline[start:stop] = np.where(enough, med, global_median)
    return baseline


def residual_noise_sigma(residual: np.ndarray) -> float:
    """Robust scene-wide noise level of the residual field (MAD scaled to a normal sigma)."""
    centre = float(np.median(residual))
    mad = float(np.median(np.abs(residual - centre)))
    return max(MAD_TO_SIGMA * mad, SIGMA_FLOOR_C)


def _area_from_points(points: np.ndarray) -> float:
    """Surface area covered by a cluster of points assuming roughly uniform sampling."""
    if len(points) < 2:
        return 0.0
    d, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(d[:, 1]))
    return float(len(points) * 4.0 * spacing**2)


def _likely_cause(points: np.ndarray, centroid: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, wall_height: float) -> str:
    """Geometry-only heuristic, so it works on real scans with no semantic labels."""
    if centroid[1] < SILL_MAX_HEIGHT_M:
        return CAUSE_SILL
    if centroid[1] > wall_height - EAVE_BAND_M:
        return CAUSE_EAVE
    centre = (bbox_min + bbox_max) / 2.0
    hole = float(np.min(np.linalg.norm(points - centre, axis=1)))
    if hole > RING_HOLE_MIN_M:
        return CAUSE_FRAME
    return CAUSE_INSULATION


def compute_residual(scene: GaussianScene, params: DetectionParams = DetectionParams()) -> np.ndarray:
    """Unsigned per-Gaussian deviation from the local same-surface baseline (degrees C).

    This is the expensive half of detection and depends only on the baseline parameters, so
    callers may cache it and re-run `cluster_leaks` with different thresholds.
    """
    if scene.temperature is None:
        raise ValueError("scene has no baked temperature; run the bake or synth step first")
    baseline = local_baseline(scene.positions.astype(np.float64), scene.temperature.astype(np.float32), surface_normals(scene), params)
    return (scene.temperature.astype(np.float32) - baseline).astype(np.float32)


def baseline_signature(params: DetectionParams) -> dict:
    """The subset of parameters a cached residual depends on."""
    return {
        "baseline_radius_m": params.baseline_radius_m,
        "baseline_neighbours": params.baseline_neighbours,
        "normal_cos_threshold": params.normal_cos_threshold,
    }


def cluster_leaks(scene: GaussianScene, residual_unsigned: np.ndarray, params: DetectionParams = DetectionParams(), wall_height_m: float | None = None) -> DetectionResult:
    """Threshold a residual field and group the survivors into ranked leaks. Fast."""
    positions = scene.positions.astype(np.float64)
    wall_height = float(wall_height_m) if wall_height_m else float(np.percentile(positions[:, 1], 55))
    residual = (params.sign * residual_unsigned).astype(np.float32)
    zscore = residual / residual_noise_sigma(residual)
    seed = (residual >= params.min_delta_c) & (zscore >= params.min_zscore)
    # Hysteresis: cluster everything above the lower edge threshold, keep clusters that contain a
    # seed. A leak's soft edges then stay attached to it instead of fragmenting as the slider rises.
    edge_delta = min(params.min_delta_c, max(params.edge_floor_c, params.min_delta_c * params.hysteresis_ratio))
    candidate = (residual >= edge_delta) & (zscore >= params.min_zscore)
    if seed.sum() == 0 or candidate.sum() < params.cluster_min_points:
        return DetectionResult((), residual, seed, params)

    cand_idx = np.flatnonzero(candidate)
    labels = DBSCAN(eps=params.cluster_eps_m, min_samples=params.cluster_min_points).fit(positions[cand_idx]).labels_

    raw = []
    for cluster_id in np.unique(labels[labels >= 0]):
        members = cand_idx[labels == cluster_id]
        if not seed[members].any():
            continue
        pts = positions[members]
        deltas = residual[members]
        bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
        centroid = pts.mean(axis=0)
        area = _area_from_points(pts)
        raw.append((members, pts, deltas, bbox_min, bbox_max, centroid, area))

    ranked = sorted(raw, key=lambda r: float(r[2].mean()) * r[6], reverse=True)
    leaks = tuple(
        Leak(
            id=f"L{rank}",
            rank=rank,
            centroid=tuple(map(float, centroid)),
            bbox_min=tuple(map(float, bbox_min)),
            bbox_max=tuple(map(float, bbox_max)),
            n_points=int(len(members)),
            mean_delta_c=float(deltas.mean()),
            max_delta_c=float(deltas.max()),
            area_m2=area,
            severity=float(deltas.mean()) * area,
            likely_cause=_likely_cause(pts, centroid, bbox_min, bbox_max, wall_height),
            point_indices=members,
        )
        for rank, (members, pts, deltas, bbox_min, bbox_max, centroid, area) in enumerate(ranked, start=1)
    )
    return DetectionResult(leaks, residual, seed, params)


def detect_leaks(scene: GaussianScene, params: DetectionParams = DetectionParams(), wall_height_m: float | None = None) -> DetectionResult:
    """Full detection: baseline residual, then thresholding and clustering."""
    return cluster_leaks(scene, compute_residual(scene, params), params, wall_height_m)
