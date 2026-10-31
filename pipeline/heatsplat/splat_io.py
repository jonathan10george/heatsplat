"""Write the compact `.splat` format consumed by browser splat viewers.

Layout per Gaussian (32 bytes): position float32 x3, scale float32 x3,
colour rgba uint8 x4, rotation quaternion (w, x, y, z) uint8 x4 as q * 128 + 128.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from heatsplat.fsutil import atomic_write_bytes
from heatsplat.scene import GaussianScene

BYTES_PER_GAUSSIAN = 32


def scene_to_splat_bytes(scene: GaussianScene, colors: np.ndarray | None = None) -> bytes:
    """Serialise a scene. `colors` (N, 3) in [0, 1] overrides the scene's own colours."""
    rgb = scene.colors if colors is None else np.asarray(colors, dtype=np.float32)
    if rgb.shape != (len(scene), 3):
        raise ValueError(f"colors must be ({len(scene)}, 3), got {rgb.shape}")

    n = len(scene)
    out = np.empty(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    out["pos"] = scene.positions.astype(np.float32)
    out["scale"] = scene.scales.astype(np.float32)
    out["rgba"][:, :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out["rgba"][:, 3] = np.clip(scene.opacities * 255.0 + 0.5, 0, 255).astype(np.uint8)
    q = scene.rotations / np.linalg.norm(scene.rotations, axis=1, keepdims=True)
    out["rot"] = np.clip(q * 128.0 + 128.0, 0, 255).astype(np.uint8)
    return out.tobytes()


def write_splat(path: Path, scene: GaussianScene, colors: np.ndarray | None = None) -> int:
    """Write a .splat file and return the number of Gaussians written."""
    data = scene_to_splat_bytes(scene, colors)
    atomic_write_bytes(path, data)
    return len(data) // BYTES_PER_GAUSSIAN


def read_splat_positions(path: Path) -> np.ndarray:
    """Read only the positions back from a .splat file (used by tests and sanity checks)."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(-1, BYTES_PER_GAUSSIAN)
    return raw[:, :12].copy().view("<f4").reshape(-1, 3)
