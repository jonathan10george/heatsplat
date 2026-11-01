"""Build a Gaussian scene from a plain point cloud when no splat trainer is available.

A COLMAP sparse model (points3D.txt) or a coloured PLY point cloud becomes a set of flat
disk Gaussians: each point gets a disk whose radius follows its nearest-neighbour spacing
and whose normal comes from a local PCA, so the detector's surface-normal baseline works
the same way it does on a trained splat. Far sparser than a real splat, but it needs no GPU
and gives a real capture something to bake temperature onto within minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .ply_io import _parse_header
from .scene import GaussianScene, quaternions_from_normals

NORMAL_NEIGHBOURS = 16       # PCA neighbourhood for the surface normal
RADIUS_NEIGHBOURS = 4        # disk radius follows the mean distance to this many neighbours
RADIUS_FACTOR = 0.5          # ... multiplied by this: sigma of half a spacing reads as a crisp point cloud rather than a smear
THICKNESS_RATIO = 0.1        # disk thickness relative to its radius (thin axis = normal)
RADIUS_MAX_SPACING_RATIO = 2.0  # cap the radius at this many median spacings: SfM points are clumpy and a few huge disks would smear the view
FLOATER_SPACING_RATIO = 4.0  # drop points whose spacing exceeds this multiple of the median
MAX_REPROJ_ERROR_PX = 2.0    # COLMAP points with larger reprojection error are noise
MIN_TRACK_LENGTH = 3         # COLMAP points seen in fewer images are unreliable
POINT_OPACITY = 0.9


@dataclass(frozen=True)
class PointCloud:
    positions: np.ndarray  # (N, 3) float64
    colors: np.ndarray     # (N, 3) float32 in [0, 1]

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def load_colmap_points(path: Path, max_error_px: float = MAX_REPROJ_ERROR_PX, min_track: int = MIN_TRACK_LENGTH) -> PointCloud:
    """Parse COLMAP points3D.txt (ID X Y Z R G B ERROR TRACK...), keeping well-observed points."""
    positions: list[list[float]] = []
    colors: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        track_length = (len(parts) - 8) // 2
        if float(parts[7]) > max_error_px or track_length < min_track:
            continue
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
    if not positions:
        raise ValueError(f"no usable 3D points in {path} (error <= {max_error_px}px, track >= {min_track})")
    return PointCloud(np.array(positions, dtype=np.float64), (np.array(colors, dtype=np.float32) / 255.0))


def load_ply_points(path: Path) -> PointCloud:
    """Read a binary little-endian PLY point cloud with x, y, z and red, green, blue."""
    raw = Path(path).read_bytes()
    count, properties, header_end = _parse_header(raw)
    dtype = np.dtype(list(properties))  # header parser already maps PLY types to numpy dtypes
    rows = np.frombuffer(raw, dtype=dtype, count=count, offset=header_end)
    names = set(rows.dtype.names or ())
    if not {"x", "y", "z"} <= names:
        raise ValueError(f"{path} has no x/y/z vertex properties")
    positions = np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float64)
    if {"red", "green", "blue"} <= names:
        rgb = np.stack([rows["red"], rows["green"], rows["blue"]], axis=1).astype(np.float32)
        colors = rgb / 255.0 if rgb.max(initial=0.0) > 1.0 else rgb
    else:
        colors = np.full((count, 3), 0.6, dtype=np.float32)
    finite = np.isfinite(positions).all(axis=1)
    return PointCloud(positions[finite], np.clip(colors[finite], 0.0, 1.0))


def load_points(path: Path) -> PointCloud:
    """Dispatch on extension: .txt is a COLMAP points3D file, .ply a point cloud."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return load_ply_points(path)
    return load_colmap_points(path)


def pca_normals(positions: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Unit normal per point: the least-variance axis of its neighbourhood (N, k, 3)."""
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred) / max(neighbours.shape[1] - 1, 1)
    _, vectors = np.linalg.eigh(covariance)  # ascending eigenvalues; column 0 = normal
    normals = vectors[:, :, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def gaussians_from_points(cloud: PointCloud, radius_factor: float = RADIUS_FACTOR) -> GaussianScene:
    """Turn a point cloud into surface-aligned disk Gaussians, dropping isolated floaters."""
    n = len(cloud)
    if n <= NORMAL_NEIGHBOURS:
        raise ValueError(f"need more than {NORMAL_NEIGHBOURS} points, got {n}")
    tree = cKDTree(cloud.positions)
    distances, indices = tree.query(cloud.positions, k=NORMAL_NEIGHBOURS + 1)
    spacing = distances[:, 1:RADIUS_NEIGHBOURS + 1].mean(axis=1)
    keep = spacing <= np.median(spacing) * FLOATER_SPACING_RATIO
    normals = pca_normals(cloud.positions, cloud.positions[indices])
    radius = np.minimum(spacing, np.median(spacing) * RADIUS_MAX_SPACING_RATIO) * radius_factor
    radius = radius.astype(np.float32)
    scales = np.stack([radius, radius, radius * THICKNESS_RATIO], axis=1)
    scene = GaussianScene(
        positions=cloud.positions.astype(np.float32),
        scales=scales,
        rotations=quaternions_from_normals(normals),
        colors=cloud.colors.astype(np.float32),
        opacities=np.full(n, POINT_OPACITY, dtype=np.float32),
    )
    return scene.subset(keep)
