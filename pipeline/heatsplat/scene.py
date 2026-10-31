"""Immutable container for a Gaussian splat scene with optional per-Gaussian temperature."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from heatsplat.fsutil import atomic_savez

SCALE_FLOOR = 1e-4


@dataclass(frozen=True)
class GaussianScene:
    """One Gaussian per row. All arrays share the first dimension.

    positions:   (N, 3) float32, world space, Y up
    scales:      (N, 3) float32, linear (already exp'd), metres
    rotations:   (N, 4) float32, unit quaternions as (w, x, y, z)
    colors:      (N, 3) float32 in [0, 1]
    opacities:   (N,)   float32 in [0, 1]
    temperature: (N,)   float32 in degrees C, or None if not yet baked
    labels:      (N,)   int16 semantic label (synthetic scenes only), or None
    """

    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    temperature: np.ndarray | None = None
    labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        checks = {
            "positions": (self.positions.shape, (n, 3)),
            "scales": (self.scales.shape, (n, 3)),
            "rotations": (self.rotations.shape, (n, 4)),
            "colors": (self.colors.shape, (n, 3)),
            "opacities": (self.opacities.shape, (n,)),
        }
        for name, (got, want) in checks.items():
            if got != want:
                raise ValueError(f"{name} has shape {got}, expected {want}")
        if self.temperature is not None and self.temperature.shape != (n,):
            raise ValueError(f"temperature has shape {self.temperature.shape}, expected {(n,)}")
        if self.labels is not None and self.labels.shape != (n,):
            raise ValueError(f"labels has shape {self.labels.shape}, expected {(n,)}")
        if np.any(self.scales < SCALE_FLOOR):
            raise ValueError(f"scales must be >= {SCALE_FLOOR} (linear, not log)")

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def has_temperature(self) -> bool:
        return self.temperature is not None

    def with_temperature(self, temperature: np.ndarray) -> GaussianScene:
        return replace(self, temperature=np.asarray(temperature, dtype=np.float32))

    def with_colors(self, colors: np.ndarray) -> GaussianScene:
        return replace(self, colors=np.asarray(colors, dtype=np.float32))

    def subset(self, mask: np.ndarray) -> GaussianScene:
        pick = lambda a: None if a is None else a[mask]  # noqa: E731
        return GaussianScene(
            positions=self.positions[mask],
            scales=self.scales[mask],
            rotations=self.rotations[mask],
            colors=self.colors[mask],
            opacities=self.opacities[mask],
            temperature=pick(self.temperature),
            labels=pick(self.labels),
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def save(self, path: Path) -> None:
        arrays = {
            "positions": self.positions.astype(np.float32),
            "scales": self.scales.astype(np.float32),
            "rotations": self.rotations.astype(np.float32),
            "colors": self.colors.astype(np.float32),
            "opacities": self.opacities.astype(np.float32),
        }
        if self.temperature is not None:
            arrays["temperature"] = self.temperature.astype(np.float32)
        if self.labels is not None:
            arrays["labels"] = self.labels.astype(np.int16)
        atomic_savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> GaussianScene:
        with np.load(path) as data:
            return cls(
                positions=data["positions"],
                scales=data["scales"],
                rotations=data["rotations"],
                colors=data["colors"],
                opacities=data["opacities"],
                temperature=data["temperature"] if "temperature" in data else None,
                labels=data["labels"] if "labels" in data else None,
            )


def quaternions_from_normals(normals: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z) rotating local +Z onto each given normal.

    Used to orient flat disk Gaussians so they lie on a surface.
    """
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dots = np.clip(n @ z, -1.0, 1.0)
    axes = np.cross(np.broadcast_to(z, n.shape), n)
    axis_norm = np.linalg.norm(axes, axis=1)
    quats = np.zeros((n.shape[0], 4), dtype=np.float64)
    # General case: rotation about cross(z, n) by angle acos(dot).
    ok = axis_norm > 1e-8
    half = np.arccos(dots[ok]) * 0.5
    quats[ok, 0] = np.cos(half)
    quats[ok, 1:] = axes[ok] / axis_norm[ok, None] * np.sin(half)[:, None]
    # Parallel: identity. Anti-parallel: 180 degrees about X.
    same = (~ok) & (dots > 0)
    flipped = (~ok) & (dots <= 0)
    quats[same] = [1.0, 0.0, 0.0, 0.0]
    quats[flipped] = [0.0, 1.0, 0.0, 0.0]
    return quats.astype(np.float32)


def rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """(N, 4) unit quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices."""
    q = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def surface_normals(scene: GaussianScene) -> np.ndarray:
    """Per-Gaussian unit normal: the rotated axis along which the Gaussian is thinnest.

    Works for synthetic disks and for trained splats, which flatten onto surfaces.
    """
    thin_axis = np.argmin(scene.scales, axis=1)
    rot = rotation_matrices(scene.rotations.astype(np.float64))
    normals = rot[np.arange(len(scene)), :, thin_axis]
    return (normals / np.linalg.norm(normals, axis=1, keepdims=True)).astype(np.float32)
