import numpy as np

from heatsplat.scene import GaussianScene
from heatsplat.splat_io import BYTES_PER_GAUSSIAN, read_splat_positions, scene_to_splat_bytes, write_splat


def _scene() -> GaussianScene:
    return GaussianScene(
        positions=np.array([[1.0, -2.0, 3.5], [0.0, 0.0, 0.0]], np.float32),
        scales=np.array([[0.1, 0.2, 0.3], [0.01, 0.01, 0.01]], np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], np.float32),
        colors=np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 1.0]], np.float32),
        opacities=np.array([1.0, 0.5], np.float32),
    )


def test_layout_is_32_bytes_per_gaussian():
    data = scene_to_splat_bytes(_scene())
    assert len(data) == 2 * BYTES_PER_GAUSSIAN


def test_fields_are_encoded_as_viewer_expects():
    raw = np.frombuffer(scene_to_splat_bytes(_scene()), np.uint8).reshape(2, BYTES_PER_GAUSSIAN)
    first = raw[0]
    assert np.allclose(first[:12].copy().view("<f4"), [1.0, -2.0, 3.5])
    assert np.allclose(first[12:24].copy().view("<f4"), [0.1, 0.2, 0.3])
    assert list(first[24:28]) == [255, 128, 0, 255]  # rgb + alpha
    assert list(first[28:32]) == [255, 128, 128, 128]  # identity quaternion (w, x, y, z)
    assert raw[1][27] == 128  # opacity 0.5


def test_color_override_and_round_trip(tmp_path):
    scene = _scene()
    n = write_splat(tmp_path / "x.splat", scene, colors=np.zeros((2, 3), np.float32))
    assert n == 2
    assert np.allclose(read_splat_positions(tmp_path / "x.splat"), scene.positions)
    raw = np.frombuffer((tmp_path / "x.splat").read_bytes(), np.uint8)
    assert list(raw[24:27]) == [0, 0, 0]
