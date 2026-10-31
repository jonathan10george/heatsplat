import numpy as np
import pytest

from heatsplat.scene import GaussianScene, quaternions_from_normals, rotation_matrices, surface_normals
from heatsplat.synth import build_plane


def _tiny_scene(n: int = 5) -> GaussianScene:
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(n, 3))
    scales = np.tile(np.array([0.02, 0.02, 0.003], dtype=np.float32), (n, 1))
    return GaussianScene(
        positions=rng.normal(size=(n, 3)).astype(np.float32),
        scales=scales,
        rotations=quaternions_from_normals(normals),
        colors=rng.uniform(size=(n, 3)).astype(np.float32),
        opacities=np.full(n, 0.9, np.float32),
        temperature=np.arange(n, dtype=np.float32),
        labels=np.zeros(n, np.int16),
    ), normals


def test_quaternions_rotate_z_axis_onto_normals():
    rng = np.random.default_rng(3)
    normals = rng.normal(size=(200, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals[0] = [0, 0, 1]  # parallel edge case
    normals[1] = [0, 0, -1]  # anti-parallel edge case
    q = quaternions_from_normals(normals)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    rotated = rotation_matrices(q.astype(np.float64))[:, :, 2]
    assert np.allclose(rotated, normals, atol=1e-5)


def test_surface_normals_recover_thin_axis():
    scene, normals = _tiny_scene()
    expected = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    got = surface_normals(scene)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", got, expected)), 1.0, atol=1e-4)


def test_save_and_load_round_trip(tmp_path):
    scene, _ = _tiny_scene()
    scene.save(tmp_path / "s.npz")
    back = GaussianScene.load(tmp_path / "s.npz")
    assert len(back) == len(scene)
    assert np.array_equal(back.temperature, scene.temperature)
    assert np.array_equal(back.labels, scene.labels)
    assert np.allclose(back.rotations, scene.rotations)


def test_with_temperature_returns_new_scene_without_mutating():
    scene, _ = _tiny_scene()
    original = scene.temperature.copy()
    updated = scene.with_temperature(scene.temperature + 10)
    assert np.array_equal(scene.temperature, original)
    assert np.allclose(updated.temperature, original + 10)


def test_subset_keeps_all_arrays_aligned():
    scene, _ = _tiny_scene(8)
    mask = np.array([True, False] * 4)
    sub = scene.subset(mask)
    assert len(sub) == 4
    assert np.array_equal(sub.temperature, scene.temperature[mask])
    assert np.array_equal(sub.positions, scene.positions[mask])


def test_rejects_mismatched_shapes_and_log_scales():
    scene, _ = _tiny_scene()
    with pytest.raises(ValueError, match="colors"):
        GaussianScene(scene.positions, scene.scales, scene.rotations, scene.colors[:2], scene.opacities)
    with pytest.raises(ValueError, match="scales"):
        GaussianScene(scene.positions, scene.scales - 1.0, scene.rotations, scene.colors, scene.opacities)


def test_build_plane_has_patch_at_expected_temperature():
    plane = build_plane(size_m=4.0, density=100.0, patch=(-0.5, 0.5, 1.0, 2.0), patch_delta_c=3.0)
    inside = (np.abs(plane.positions[:, 0]) < 0.5) & (plane.positions[:, 1] > 1.0) & (plane.positions[:, 1] < 2.0)
    assert plane.temperature[inside].mean() == pytest.approx(8.0, abs=0.2)
    assert plane.temperature[~inside].mean() == pytest.approx(5.0, abs=0.2)
