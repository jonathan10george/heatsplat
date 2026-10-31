import numpy as np
import pytest

from heatsplat.bake import Intrinsics, bake_temperature, fill_unobserved, opengl_to_opencv, project

K = Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)


def _camera_at(z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[2, 3] = z
    return pose


def test_project_centre_point_lands_on_principal_point():
    u, v, d = project(np.array([[0.0, 0.0, 5.0]]), _camera_at(0.0), K)
    assert (u[0], v[0], d[0]) == (50.0, 50.0, 5.0)


def test_bake_samples_frame_gradient():
    xs = np.linspace(-2, 2, 41)
    positions = np.stack([xs, np.zeros_like(xs), np.full_like(xs, 5.0)], axis=1)
    frame = np.tile(np.arange(100, dtype=np.float32)[None, :], (100, 1))  # temperature == column index
    result = bake_temperature(positions, _camera_at(0.0)[None], K, frame[None])
    expected_u = 100.0 * xs / 5.0 + 50.0
    assert np.all(result.coverage == 1)
    assert np.allclose(result.temperature, np.floor(expected_u), atol=1.0)


def test_occluded_point_is_not_sampled_and_can_be_filled():
    positions = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 6.0], [0.3, 0.0, 2.0]])
    frame = np.full((100, 100), 7.0, np.float32)
    result = bake_temperature(positions, _camera_at(0.0)[None], K, frame[None])
    assert list(result.coverage) == [1, 0, 1]
    assert np.isnan(result.temperature[1])
    filled = fill_unobserved(positions, result.temperature)
    assert filled[1] == pytest.approx(7.0)


def test_point_behind_camera_is_ignored():
    result = bake_temperature(np.array([[0.0, 0.0, -1.0]]), _camera_at(0.0)[None], K, np.zeros((1, 100, 100), np.float32))
    assert result.coverage[0] == 0


def test_opengl_to_opencv_flips_y_and_z():
    out = opengl_to_opencv(np.eye(4))
    assert np.allclose(np.diag(out), [1, -1, -1, 1])


def test_mismatched_frames_and_poses_rejected():
    with pytest.raises(ValueError):
        bake_temperature(np.zeros((1, 3)), np.eye(4)[None], K, np.zeros((2, 100, 100)))
