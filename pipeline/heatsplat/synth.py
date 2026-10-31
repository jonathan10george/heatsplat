"""Synthetic house scene with a plausible night-time exterior thermal signature and injected leaks.

The geometry is a gabled house (walls, roof, gable ends, windows, door, ground) sampled as
flat disk Gaussians. The temperature field is a base per-surface temperature plus subtle stud
thermal bridging, plus a handful of ground-truth leak regions, plus sensor noise and blur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from heatsplat.detect import CAUSE_EAVE, CAUSE_FRAME, CAUSE_INSULATION, CAUSE_SILL
from heatsplat.scene import GaussianScene, quaternions_from_normals

LABEL_WALL, LABEL_ROOF, LABEL_GLASS, LABEL_FRAME, LABEL_DOOR, LABEL_GROUND = 0, 1, 2, 3, 4, 5
LABEL_NAMES = {0: "wall", 1: "roof", 2: "glass", 3: "frame", 4: "door", 5: "ground"}

WALL_HEIGHT = 3.0
HALF_X, HALF_Z = 5.0, 4.0
ROOF_RISE, OVERHANG = 2.2, 0.4
FRAME_WIDTH, FRAME_PROUD, GLASS_INSET = 0.10, 0.04, 0.05
LEAK_EDGE_SOFTNESS_M = 0.06
SENSOR_NOISE_C = 0.25
BLUR_NEIGHBOURS = 10
SPLAT_RADIUS_FACTOR = 1.1  # Gaussian sigma = factor / sqrt(density): >1 means neighbours overlap, so surfaces read solid from afar


@dataclass(frozen=True)
class SynthParams:
    density: float = 400.0  # Gaussians per square metre on the house
    ground_density: float = 40.0
    seed: int = 7
    ambient_c: float = 3.0
    wall_c: float = 5.0
    roof_c: float = 3.2
    glass_c: float = 0.5
    door_c: float = 5.5
    ground_c: float = 2.5


@dataclass(frozen=True)
class Rect:
    """A planar rectangle: origin corner, unit edge directions u and v, extents, outward normal."""

    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    width: float
    height: float
    normal: np.ndarray

    def to_world(self, a: np.ndarray, b: np.ndarray, proud: float = 0.0) -> np.ndarray:
        return self.origin + a[:, None] * self.u + b[:, None] * self.v + proud * self.normal

    def sample(self, rng: np.random.Generator, density: float) -> tuple[np.ndarray, np.ndarray]:
        n = max(1, int(round(density * self.width * self.height)))
        return rng.uniform(0, self.width, n), rng.uniform(0, self.height, n)


@dataclass(frozen=True)
class Opening:
    a0: float
    a1: float
    b0: float
    b1: float
    kind: int  # LABEL_GLASS or LABEL_DOOR

    def contains(self, a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (a >= self.a0 - margin) & (a <= self.a1 + margin) & (b >= self.b0 - margin) & (b <= self.b1 + margin)


@dataclass(frozen=True)
class LeakBox:
    """Ground-truth leak: axis-aligned world box with a temperature rise, excluding some labels."""

    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    delta_c: float
    cause: str
    exclude_labels: tuple[int, ...] = (LABEL_GLASS, LABEL_ROOF, LABEL_GROUND)

    def center(self) -> list[float]:
        return [float((l + h) / 2) for l, h in zip(self.lo, self.hi)]


@dataclass(frozen=True)
class Surface:
    positions: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    base_temp: np.ndarray
    scale: float
    opacity: float = 0.95


@dataclass(frozen=True)
class SynthResult:
    scene: GaussianScene
    leaks: tuple[LeakBox, ...]
    meta: dict = field(default_factory=dict)


def _walls() -> list[tuple[Rect, list[Opening]]]:
    y = np.array([0.0, 1.0, 0.0])
    win = lambda a0, b0=1.2, w=1.2, h=1.2: Opening(a0, a0 + w, b0, b0 + h, LABEL_GLASS)  # noqa: E731
    front = Rect(np.array([-HALF_X, 0, HALF_Z]), np.array([1.0, 0, 0]), y, 2 * HALF_X, WALL_HEIGHT, np.array([0, 0, 1.0]))
    back = Rect(np.array([HALF_X, 0, -HALF_Z]), np.array([-1.0, 0, 0]), y, 2 * HALF_X, WALL_HEIGHT, np.array([0, 0, -1.0]))
    right = Rect(np.array([HALF_X, 0, HALF_Z]), np.array([0, 0, -1.0]), y, 2 * HALF_Z, WALL_HEIGHT, np.array([1.0, 0, 0]))
    left = Rect(np.array([-HALF_X, 0, -HALF_Z]), np.array([0, 0, 1.0]), y, 2 * HALF_Z, WALL_HEIGHT, np.array([-1.0, 0, 0]))
    door = Opening(4.5, 5.5, 0.0, 2.1, LABEL_DOOR)
    return [
        (front, [win(1.4), win(7.4), door]),
        (back, [win(2.0), win(6.8)]),
        (right, [win(3.8)]),
        (left, [win(3.4)]),
    ]


def ground_truth_leaks() -> tuple[LeakBox, ...]:
    """Leak regions in world space. These are what the detector should recover."""
    return (
        LeakBox("front-left window seal", (-3.72, 1.08, 3.9), (-2.28, 2.52, 4.1), 5.0, CAUSE_FRAME),
        LeakBox("right window seal", (4.9, 1.08, -1.02), (5.1, 2.52, 0.42), 4.5, CAUSE_FRAME),
        LeakBox("back eave bypass", (-3.5, 2.55, -4.1), (-0.5, 3.0, -3.9), 4.0, CAUSE_EAVE),
        LeakBox("right wall insulation void", (4.9, 1.0, 0.8), (5.1, 1.9, 2.0), 4.5, CAUSE_INSULATION),
        LeakBox("door bottom gap", (-0.6, 0.0, 3.9), (0.6, 0.15, 4.1), 5.5, CAUSE_SILL),
    )


def _wall_surfaces(rect: Rect, openings: list[Opening], p: SynthParams, rng: np.random.Generator) -> list[Surface]:
    a, b = rect.sample(rng, p.density)
    in_opening = np.zeros_like(a, dtype=bool)
    for op in openings:
        in_opening |= op.contains(a, b)
    a_w, b_w = a[~in_opening], b[~in_opening]
    pos = rect.to_world(a_w, b_w)
    siding = 0.92 - 0.10 * ((b_w / 0.22) % 1.0 < 0.18)  # horizontal siding shadow lines
    color = np.stack([siding * 0.86, siding * 0.80, siding * 0.68], axis=1)
    studs = 0.6 * np.exp(-(((a_w / 0.4) % 1.0) - 0.5) ** 2 / (2 * 0.06**2))  # faint thermal bridging
    temp = p.wall_c + 0.3 * (b_w / WALL_HEIGHT) + studs
    scale = SPLAT_RADIUS_FACTOR / np.sqrt(p.density)
    out = [Surface(pos, np.repeat(rect.normal[None], len(pos), 0), color, np.full(len(pos), LABEL_WALL, np.int16), temp, scale)]

    for op in openings:
        n_ring = int(p.density * (op.a1 - op.a0 + 2 * FRAME_WIDTH) * (op.b1 - op.b0 + 2 * FRAME_WIDTH))
        a_o, b_o = rng.uniform(op.a0 - FRAME_WIDTH, op.a1 + FRAME_WIDTH, n_ring), rng.uniform(op.b0 - FRAME_WIDTH, op.b1 + FRAME_WIDTH, n_ring)
        ring = ~op.contains(a_o, b_o)
        a_f, b_f = a_o[ring], b_o[ring]
        if op.kind == LABEL_DOOR:
            n_door = int(p.density * (op.a1 - op.a0) * (op.b1 - op.b0))
            a_d, b_d = rng.uniform(op.a0, op.a1, n_door), rng.uniform(op.b0, op.b1, n_door)
            panel = 0.9 - 0.15 * (((b_d / 0.7) % 1.0 < 0.1) | (np.abs(a_d - (op.a0 + op.a1) / 2) < 0.03))
            door_col = np.stack([0.45 * panel, 0.25 * panel, 0.12 * panel], axis=1)
            out.append(Surface(rect.to_world(a_d, b_d, 0.01), np.repeat(rect.normal[None], len(a_d), 0), door_col, np.full(len(a_d), LABEL_DOOR, np.int16), np.full(len(a_d), p.door_c), scale))
        else:
            a_g, b_g = rng.uniform(op.a0 + GLASS_INSET, op.a1 - GLASS_INSET, int(p.density * 1.44)), rng.uniform(op.b0 + GLASS_INSET, op.b1 - GLASS_INSET, int(p.density * 1.44))
            glass_col = np.stack([0.30 + 0.1 * b_g / 2.4, 0.42 + 0.1 * b_g / 2.4, 0.58 + 0.15 * b_g / 2.4], axis=1)
            mullion = (np.abs(a_g - (op.a0 + op.a1) / 2) < 0.025) | (np.abs(b_g - (op.b0 + op.b1) / 2) < 0.025)
            glass_col = np.where(mullion[:, None], 0.95, glass_col)
            out.append(Surface(rect.to_world(a_g, b_g, -GLASS_INSET), np.repeat(rect.normal[None], len(a_g), 0), glass_col, np.full(len(a_g), LABEL_GLASS, np.int16), np.full(len(a_g), p.glass_c), scale, 0.9))
        frame_col = np.full((len(a_f), 3), 0.95, dtype=np.float32)
        out.append(Surface(rect.to_world(a_f, b_f, FRAME_PROUD), np.repeat(rect.normal[None], len(a_f), 0), frame_col, np.full(len(a_f), LABEL_FRAME, np.int16), np.full(len(a_f), p.wall_c - 0.5), scale))
    return out


def _roof_surfaces(p: SynthParams, rng: np.random.Generator) -> list[Surface]:
    out = []
    slope_len = float(np.hypot(ROOF_RISE, HALF_Z + OVERHANG))
    for sign in (1.0, -1.0):
        origin = np.array([-(HALF_X + OVERHANG), WALL_HEIGHT - 0.05, sign * (HALF_Z + OVERHANG)])
        v = np.array([0.0, ROOF_RISE, -sign * (HALF_Z + OVERHANG)]) / slope_len
        u = np.array([1.0, 0.0, 0.0])
        normal = np.cross(u, v)
        rect = Rect(origin, u, v, 2 * (HALF_X + OVERHANG), slope_len, normal / np.linalg.norm(normal))
        a, b = rect.sample(rng, p.density)
        shingle = 0.30 - 0.08 * ((b / 0.3) % 1.0 < 0.15) + rng.normal(0, 0.015, len(a))
        color = np.stack([shingle * 1.05, shingle, shingle * 0.95], axis=1)
        temp = p.roof_c + 0.2 * (b / slope_len)
        out.append(Surface(rect.to_world(a, b), np.repeat(rect.normal[None], len(a), 0), color, np.full(len(a), LABEL_ROOF, np.int16), temp, SPLAT_RADIUS_FACTOR / np.sqrt(p.density)))
    for sign in (1.0, -1.0):  # gable end triangles
        n = int(p.density * HALF_Z * ROOF_RISE * 1.2)
        z = rng.uniform(-HALF_Z, HALF_Z, n)
        y = rng.uniform(WALL_HEIGHT, WALL_HEIGHT + ROOF_RISE, n)
        keep = np.abs(z) <= HALF_Z * (1.0 - (y - WALL_HEIGHT) / ROOF_RISE)
        z, y = z[keep], y[keep]
        pos = np.stack([np.full(len(z), sign * HALF_X), y, z], axis=1)
        normal = np.array([sign, 0.0, 0.0])
        siding = 0.92 - 0.10 * ((y / 0.22) % 1.0 < 0.18)
        color = np.stack([siding * 0.86, siding * 0.80, siding * 0.68], axis=1)
        out.append(Surface(pos, np.repeat(normal[None], len(z), 0), color, np.full(len(z), LABEL_WALL, np.int16), np.full(len(z), p.wall_c - 0.3), SPLAT_RADIUS_FACTOR / np.sqrt(p.density)))
    return out


def _ground_surface(p: SynthParams, rng: np.random.Generator) -> Surface:
    half = 12.0
    n = int(p.ground_density * (2 * half) ** 2)
    x, z = rng.uniform(-half, half, n), rng.uniform(-half, half, n)
    keep = ~((np.abs(x) < HALF_X - 0.2) & (np.abs(z) < HALF_Z - 0.2))
    x, z = x[keep], z[keep]
    pos = np.stack([x, np.zeros_like(x), z], axis=1)
    path = (np.abs(x) < 0.9) & (z > HALF_Z)  # front walkway
    g = 0.35 + 0.08 * rng.normal(size=len(x))
    color = np.where(path[:, None], np.array([[0.55, 0.53, 0.50]]), np.stack([g * 0.75, g, g * 0.55], axis=1))
    return Surface(pos, np.repeat(np.array([[0.0, 1.0, 0.0]]), len(x), 0), color, np.full(len(x), LABEL_GROUND, np.int16), np.full(len(x), p.ground_c), SPLAT_RADIUS_FACTOR / np.sqrt(p.ground_density), 0.85)


def _apply_leaks(positions: np.ndarray, labels: np.ndarray, temp: np.ndarray, leaks: tuple[LeakBox, ...]) -> np.ndarray:
    delta = np.zeros_like(temp)
    for leak in leaks:
        lo, hi = np.array(leak.lo), np.array(leak.hi)
        outside = np.maximum(0.0, np.maximum(lo - positions, positions - hi))
        d = np.linalg.norm(outside, axis=1)
        weight = np.exp(-((d / LEAK_EDGE_SOFTNESS_M) ** 2))
        weight[np.isin(labels, leak.exclude_labels)] = 0.0
        delta = np.maximum(delta, leak.delta_c * weight)
    return temp + delta


def _blur(positions: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    _, idx = cKDTree(positions).query(positions, k=k)
    return values[idx].mean(axis=1)


def build_house(params: SynthParams = SynthParams()) -> SynthResult:
    rng = np.random.default_rng(params.seed)
    surfaces: list[Surface] = []
    for rect, openings in _walls():
        surfaces.extend(_wall_surfaces(rect, openings, params, rng))
    surfaces.extend(_roof_surfaces(params, rng))
    surfaces.append(_ground_surface(params, rng))

    positions = np.concatenate([s.positions for s in surfaces]).astype(np.float32)
    normals = np.concatenate([s.normals for s in surfaces]).astype(np.float32)
    colors = np.clip(np.concatenate([s.colors for s in surfaces]) + rng.normal(0, 0.02, (len(positions), 3)), 0, 1).astype(np.float32)
    labels = np.concatenate([s.labels for s in surfaces]).astype(np.int16)
    base_temp = np.concatenate([s.base_temp for s in surfaces]).astype(np.float32)
    scale_xy = np.concatenate([np.full(len(s.positions), s.scale) for s in surfaces]).astype(np.float32)
    opacities = np.concatenate([np.full(len(s.positions), s.opacity) for s in surfaces]).astype(np.float32)

    leaks = ground_truth_leaks()
    temp = _apply_leaks(positions, labels, base_temp, leaks)
    temp = temp + rng.normal(0, SENSOR_NOISE_C, len(temp)).astype(np.float32)
    temp = _blur(positions, temp, BLUR_NEIGHBOURS).astype(np.float32)

    scales = np.stack([scale_xy, scale_xy, scale_xy * 0.15], axis=1).astype(np.float32)
    scene = GaussianScene(positions, scales, quaternions_from_normals(normals), colors, opacities, temp, labels)
    meta = {
        "name": "Synthetic gabled house, night exterior scan",
        "wall_height_m": WALL_HEIGHT,
        "ambient_c": params.ambient_c,
        "ground_truth_leaks": [
            {"name": l.name, "center": l.center(), "delta_c": l.delta_c, "cause": l.cause} for l in leaks
        ],
        "synthetic": True,
        "label_names": LABEL_NAMES,
    }
    return SynthResult(scene, leaks, meta)


def build_plane(size_m: float = 6.0, density: float = 300.0, base_c: float = 5.0, patch: tuple[float, float, float, float] | None = (-0.5, 0.5, 1.5, 2.5), patch_delta_c: float = 4.0, seed: int = 1) -> GaussianScene:
    """A single flat wall (normal +Z) standing on the ground (y in [0, size]) with one optional
    warm rectangular patch (x0, x1, y0, y1).

    Small and fast: the fixture for unit tests and the simplest possible real-world analogue.
    """
    rng = np.random.default_rng(seed)
    n = int(density * size_m * size_m)
    x, y = rng.uniform(-size_m / 2, size_m / 2, n), rng.uniform(0.0, size_m, n)
    positions = np.stack([x, y, np.zeros(n)], axis=1).astype(np.float32)
    temp = np.full(n, base_c, dtype=np.float32) + rng.normal(0, 0.1, n).astype(np.float32)
    if patch is not None:
        x0, x1, y0, y1 = patch
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        temp = np.where(inside, temp + patch_delta_c, temp).astype(np.float32)
    s = SPLAT_RADIUS_FACTOR / np.sqrt(density)
    scales = np.tile(np.array([s, s, s * 0.15], dtype=np.float32), (n, 1))
    rotations = quaternions_from_normals(np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1)))
    colors = np.tile(np.array([[0.8, 0.75, 0.65]], dtype=np.float32), (n, 1))
    return GaussianScene(positions, scales, rotations, colors, np.full(n, 0.95, np.float32), temp)
