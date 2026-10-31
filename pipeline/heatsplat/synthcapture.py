"""Fake a phone + thermal-camera capture of a baked scene.

Renders posed thermal frames from a scene's per-Gaussian temperature (nearest-surface
z-buffer) and writes the files the ingest step expects: scene.ply, transforms.json and one
thermal .npy per frame. Used to prove the real-capture path before any hardware exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from heatsplat.bake import Intrinsics, opengl_to_opencv, project
from heatsplat.fsutil import atomic_write_text
from heatsplat.ply_io import write_ply
from heatsplat.scene import GaussianScene, quaternion_from_matrix


@dataclass(frozen=True)
class CaptureLayout:
    n_frames: int = 12
    radius_m: float = 13.0
    height_m: float = 2.5
    look_at: tuple[float, float, float] = (0.0, 1.5, 0.0)
    width: int = 320
    height: int = 240
    hfov_deg: float = 50.0


def look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """OpenCV camera-to-world (x right, y down, z forward) looking from eye at target."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, eye
    return c2w


def arc_poses(layout: CaptureLayout) -> list[np.ndarray]:
    target = np.array(layout.look_at, dtype=np.float64)
    poses = []
    for i in range(layout.n_frames):
        angle = 2 * np.pi * i / layout.n_frames
        eye = np.array([layout.radius_m * np.sin(angle), layout.height_m, layout.radius_m * np.cos(angle)])
        poses.append(look_at_pose(eye, target))
    return poses


def intrinsics_for(layout: CaptureLayout) -> Intrinsics:
    fx = 0.5 * layout.width / np.tan(np.radians(layout.hfov_deg) / 2)
    return Intrinsics(fx, fx, layout.width / 2, layout.height / 2, layout.width, layout.height)


def render_thermal_frame(scene: GaussianScene, c2w_opencv: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Nearest-Gaussian temperature per pixel with a z-buffer, holes filled from the nearest pixel."""
    if scene.temperature is None:
        raise ValueError("scene has no temperature to render")
    u, v, depth = project(scene.positions.astype(np.float64), c2w_opencv, K)
    ok = (depth > 1e-6) & (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
    idx = np.flatnonzero(ok)
    pixel = v[idx].astype(int) * K.width + u[idx].astype(int)
    nearest = np.full(K.width * K.height, np.inf)
    np.minimum.at(nearest, pixel, depth[idx])
    winners = idx[depth[idx] <= nearest[pixel] + 1e-6]
    frame = np.full(K.width * K.height, np.nan, dtype=np.float32)
    frame[v[winners].astype(int) * K.width + u[winners].astype(int)] = scene.temperature[winners]
    frame = frame.reshape(K.height, K.width)
    missing = np.isnan(frame)
    if missing.any() and not missing.all():
        _, (iy, ix) = ndimage.distance_transform_edt(missing, return_indices=True)
        frame = frame[iy, ix]
    return frame.astype(np.float32)


def write_capture(scene: GaussianScene, out_dir: Path, layout: CaptureLayout = CaptureLayout()) -> dict:
    """Write scene.ply, transforms.json (OpenGL poses, Y-up world) and thermal/<frame>.npy files."""
    out_dir = Path(out_dir)
    (out_dir / "thermal").mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / "scene.ply", scene)
    K = intrinsics_for(layout)
    frames = []
    for i, c2w in enumerate(arc_poses(layout)):
        name = f"frame_{i:03d}"
        np.save(out_dir / "thermal" / f"{name}.npy", render_thermal_frame(scene, c2w, K))
        frames.append({"file_path": f"images/{name}.jpg", "transform_matrix": opengl_to_opencv(c2w).tolist()})  # the flip is an involution: OpenCV -> OpenGL
    transforms = {"fl_x": K.fx, "fl_y": K.fy, "cx": K.cx, "cy": K.cy, "w": K.width, "h": K.height, "frames": frames}
    atomic_write_text(out_dir / "transforms.json", json.dumps(transforms, indent=2))
    write_colmap_model(out_dir / "colmap", arc_poses(layout), K)
    return transforms


def write_colmap_model(model_dir: Path, c2w_opencv: list[np.ndarray], K: Intrinsics) -> None:
    """Write cameras.txt + images.txt (COLMAP text format, world-to-camera quaternions)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(model_dir / "cameras.txt", f"# Camera list\n1 PINHOLE {K.width} {K.height} {K.fx} {K.fy} {K.cx} {K.cy}\n")
    lines = ["# Image list with two lines of data per image"]
    for i, c2w in enumerate(c2w_opencv):
        rot_w2c = c2w[:3, :3].T
        t = -rot_w2c @ c2w[:3, 3]
        q = quaternion_from_matrix(rot_w2c)
        lines.append(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 frame_{i:03d}.jpg")
        lines.append("")  # no 2D points
    atomic_write_text(model_dir / "images.txt", "\n".join(lines) + "\n")
