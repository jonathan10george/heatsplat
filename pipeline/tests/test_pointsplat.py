"""Point cloud -> disk Gaussians (the no-GPU path for real captures)."""
from pathlib import Path

import numpy as np
import pytest

from heatsplat.pointsplat import (
    RADIUS_FACTOR,
    THICKNESS_RATIO,
    gaussians_from_points,
    load_colmap_points,
    load_ply_points,
    load_points,
)
from heatsplat.scene import surface_normals


def _write_points3d(path: Path, rows: list[str]) -> Path:
    path.write_text("# 3D point list\n" + "\n".join(rows) + "\n")
    return path


def test_load_colmap_points_filters_noisy_and_short_track_points(tmp_path):
    # Arrange: good point (error 0.5, 3 observations), noisy point, short-track point
    path = _write_points3d(tmp_path / "points3D.txt", [
        "1 0 0 0 255 0 0 0.5 1 0 2 0 3 0",
        "2 1 0 0 0 255 0 5.0 1 0 2 0 3 0",
        "3 2 0 0 0 0 255 0.1 1 0",
    ])
    # Act
    cloud = load_colmap_points(path)
    # Assert
    assert len(cloud) == 1
    np.testing.assert_allclose(cloud.positions[0], [0, 0, 0])
    np.testing.assert_allclose(cloud.colors[0], [1.0, 0.0, 0.0])


def test_load_colmap_points_raises_when_nothing_usable(tmp_path):
    path = _write_points3d(tmp_path / "points3D.txt", ["1 0 0 0 1 2 3 9.0 1 0 2 0 3 0"])
    with pytest.raises(ValueError, match="no usable"):
        load_colmap_points(path)


def test_load_ply_points_reads_binary_xyz_rgb(tmp_path):
    # Arrange: a 3-vertex binary PLY with uchar colours
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows = np.array([(0, 0, 0, 255, 0, 0), (1, 0, 0, 0, 255, 0), (0, 1, 0, 0, 0, 255)], dtype=dtype)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    path = tmp_path / "sparse_pc.ply"
    path.write_bytes(header.encode() + rows.tobytes())
    # Act
    cloud = load_points(path)
    # Assert
    assert len(cloud) == 3
    np.testing.assert_allclose(cloud.positions[1], [1, 0, 0])
    np.testing.assert_allclose(cloud.colors[2], [0, 0, 1])


def test_load_ply_points_rejects_cloud_without_xyz(tmp_path):
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float a\nend_header\n"
    path = tmp_path / "bad.ply"
    path.write_bytes(header.encode() + np.zeros(1, dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="x/y/z"):
        load_ply_points(path)


def _plane_cloud(spacing: float, n: int = 20, seed: int = 0):
    from heatsplat.pointsplat import PointCloud
    rng = np.random.default_rng(seed)
    xs, zs = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    positions = np.stack([xs.ravel(), np.zeros(n * n), zs.ravel()], axis=1)  # plane y = 0, normal +/- Y
    positions = positions + rng.normal(scale=spacing * 0.02, size=positions.shape)
    colors = np.full((n * n, 3), 0.5, dtype=np.float32)
    return PointCloud(positions, colors)


def test_gaussians_from_points_makes_disks_lying_on_the_surface():
    # Arrange
    spacing = 0.1
    cloud = _plane_cloud(spacing)
    # Act
    scene = gaussians_from_points(cloud)
    # Assert: thin axis is the plane normal (Y), radius follows the point spacing
    normals = surface_normals(scene)
    assert np.mean(np.abs(normals[:, 1]) > 0.95) > 0.95
    radii = scene.scales[:, 0]
    assert np.median(radii) == pytest.approx(spacing * RADIUS_FACTOR, rel=0.15)
    np.testing.assert_allclose(scene.scales[:, 2], radii * THICKNESS_RATIO, rtol=1e-5)
    assert len(scene) == len(cloud)


def test_gaussians_from_points_drops_isolated_floaters():
    from heatsplat.pointsplat import PointCloud
    cloud = _plane_cloud(0.1)
    floater = np.array([[50.0, 50.0, 50.0]])
    with_floater = PointCloud(np.vstack([cloud.positions, floater]), np.vstack([cloud.colors, [[1, 1, 1]]]).astype(np.float32))
    scene = gaussians_from_points(with_floater)
    assert len(scene) == len(cloud)
    assert not np.any(np.all(np.isclose(scene.positions, floater), axis=1))


def test_gaussians_from_points_rejects_tiny_clouds():
    from heatsplat.pointsplat import PointCloud
    cloud = PointCloud(np.zeros((5, 3)), np.zeros((5, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="need more than"):
        gaussians_from_points(cloud)


def test_gaussians_from_points_caps_disks_in_sparse_patches():
    """A point 2.5 spacings from the grid is kept (not a floater) but must not become a giant disk."""
    from heatsplat.pointsplat import RADIUS_MAX_SPACING_RATIO, PointCloud
    cloud = _plane_cloud(0.1)
    far = np.array([[-0.25, 0.0, 0.5]])  # 2.5 spacings beyond the grid's edge
    with_far = PointCloud(np.vstack([cloud.positions, far]), np.vstack([cloud.colors, [[1, 1, 1]]]).astype(np.float32))
    scene = gaussians_from_points(with_far)
    assert len(scene) == len(with_far)
    radii = scene.scales[:, 0]
    assert radii.max() <= np.median(radii) * RADIUS_MAX_SPACING_RATIO * 1.01
