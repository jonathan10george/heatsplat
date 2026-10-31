import numpy as np
import pytest

from heatsplat.ply_io import read_ply, write_ply
from heatsplat.synth import build_plane


def test_ply_round_trip_preserves_gaussians(tmp_path):
    scene = build_plane(size_m=2.0, density=50.0)
    n = write_ply(tmp_path / "s.ply", scene)
    back = read_ply(tmp_path / "s.ply")
    assert n == len(back) == len(scene)
    assert np.allclose(back.positions, scene.positions, atol=1e-6)
    assert np.allclose(back.scales, scene.scales, rtol=1e-4)
    assert np.allclose(back.opacities, scene.opacities, atol=1e-4)
    assert np.allclose(back.colors, scene.colors, atol=2e-3)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", back.rotations, scene.rotations)), 1.0, atol=1e-5)
    assert back.temperature is None


def test_rejects_non_gaussian_ply(tmp_path):
    (tmp_path / "bad.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n" + np.zeros(3, np.float32).tobytes())
    with pytest.raises(ValueError, match="missing"):
        read_ply(tmp_path / "bad.ply")


def test_reader_accepts_double_types_crlf_headers_and_drops_nan_rows(tmp_path):
    scene = build_plane(size_m=1.0, density=20.0)
    write_ply(tmp_path / "s.ply", scene)
    raw = (tmp_path / "s.ply").read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode().replace("\n", "\r\n").encode()
    body = bytearray(raw[header_end:])
    body[0:4] = np.array([np.nan], "<f4").tobytes()  # first Gaussian's x becomes NaN
    (tmp_path / "crlf.ply").write_bytes(header + bytes(body))
    back = read_ply(tmp_path / "crlf.ply")
    assert len(back) == len(scene) - 1
    assert np.isfinite(back.positions).all()

    with pytest.raises(ValueError, match="unsupported PLY property type"):
        bad = raw.replace(b"property float x\n", b"property quad x\n", 1)
        (tmp_path / "bad.ply").write_bytes(bad)
        read_ply(tmp_path / "bad.ply")
