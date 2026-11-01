"""Real-capture ingest: a trained splat (.ply) + camera poses (nerfstudio transforms.json)
+ posed thermal frames -> a baked, exported Heatsplat scene.

Conventions
- transforms.json poses are camera-to-world in the OpenGL convention (Y up, Z back), as
  written by nerfstudio, Polycam's nerfstudio export and most phone capture apps.
- A COLMAP text model (cameras.txt + images.txt) is the alternative pose source. It is in the
  OpenCV convention and shares the world frame of a splat trained by OpenSplat on that model,
  which is the recommended Mac-only path.
- `up_axis="auto"` estimates world up from the cameras: phones are held upright, so the mean
  camera "up" direction is gravity-opposed. Use it for COLMAP models, whose frame is arbitrary.
- The .ply may live in nerfstudio's dataparser frame (splatfacto exports do); pass
  `dataparser_transforms` so the poses are mapped into that frame.
- The thermal camera is rigidly attached to the phone. `thermal_in_rgb` is its pose in the
  RGB camera's frame (identity = same optical centre and orientation, a fine first guess).
- Thermal frames are matched to RGB frames by file stem and may be .npy (degrees C),
  radiometric .csv, or 8/16-bit grayscale images scaled into `thermal_range_c`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from heatsplat.bake import Intrinsics, bake_temperature, fill_unobserved, opengl_to_opencv
from heatsplat.ply_io import read_ply
from heatsplat.scene import SCALE_FLOOR, Z_UP_TO_Y_UP, GaussianScene, rotate_scene

THERMAL_EXTENSIONS = (".npy", ".csv", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".PNG", ".TIF", ".TIFF", ".JPG", ".JPEG")  # FLIR exports upper-case
MIN_COVERAGE_FRACTION = 0.2  # below this the poses almost certainly do not match the splat
UP_SEPARATION_RATIO = 0.25  # eigenvalue ratio under which the camera X axes pin down the up direction
PATH_PLANARITY_RATIO = 0.05  # camera centres: smallest/middle eigenvalue ratio under which the walk lies in a plane
PATH_MIN_SPREAD_RATIO = 0.1  # ... and middle/largest ratio above which that path is 2-D rather than a line
PATH_MIN_FRAMES = 6
PLANE_UP_MAX_TILT_DEG = 45.0  # a 'ground plane' further than this from the image-up is not one
BELOW_GROUND_CLIP_M = 1.5  # with a known camera height, Gaussians this far under the ground are reconstruction junk
FAR_FLOATER_PERCENTILE = 99.0  # distance from the scene's median position that counts as its extent
FAR_FLOATER_RATIO = 2.0  # Gaussians more than this many extents out are debris (sky, mismatched matches)
MAX_GAUSSIAN_SIGMA_M = 2.0  # a trained Gaussian wider than this (one sigma) is sky or ground fog, not a facade feature
NEEDLE_MIN_SIGMA_M = 1.0  # needles: long thin Gaussians that streak across the view
NEEDLE_ASPECT = 8.0  # longest axis over the middle axis


@dataclass(frozen=True)
class Frame:
    name: str
    c2w: np.ndarray  # (4, 4) camera-to-world
    intrinsics: Intrinsics
    convention: str = "opengl"  # "opengl" (transforms.json) or "opencv" (COLMAP)

    def c2w_opencv(self) -> np.ndarray:
        return opengl_to_opencv(self.c2w) if self.convention == "opengl" else self.c2w.copy()


@dataclass(frozen=True)
class CaptureConfig:
    ply: Path
    transforms: Path | None  # nerfstudio transforms.json, or None when `colmap` is given
    thermal_dir: Path
    colmap: Path | None = None  # COLMAP text model directory (cameras.txt, images.txt)
    thermal_map: Path | None = None  # optional CSV "rgb_stem,thermal_file" when names differ
    thermal_range_c: tuple[float, float] | None = None  # for grayscale images: value 0..max -> lo..hi
    thermal_hfov_deg: float | None = None  # thermal lens horizontal FOV; default assumes the RGB FOV
    thermal_in_rgb: np.ndarray = field(default_factory=lambda: np.eye(4))
    dataparser_transforms: Path | None = None
    up_axis: str = "z"  # "z" (nerfstudio), "y" (already Y up) or "auto" (from camera orientations)
    scale: float | None = None  # multiply positions and Gaussian sizes by this to reach metres
    camera_height_m: float | None = None  # phone height above ground; shifts the scene so the ground sits at y=0
    min_coverage: float = MIN_COVERAGE_FRACTION
    name: str = "Real capture"


@dataclass(frozen=True)
class IngestResult:
    scene: GaussianScene
    frames_used: int
    frames_skipped: tuple[str, ...]
    coverage: np.ndarray  # frames that observed each Gaussian (before up-axis rotation; order is unchanged)
    coverage_fraction: float
    meta: dict


def load_transforms(path: Path) -> list[Frame]:
    """Parse nerfstudio-style transforms.json (global or per-frame intrinsics)."""
    data = json.loads(Path(path).read_text())
    frames = []
    for entry in data.get("frames", []):
        merged = {**{k: v for k, v in data.items() if k != "frames"}, **entry}
        width, height = int(merged.get("w", 0)), int(merged.get("h", 0))
        if "fl_x" in merged:
            fx, fy = float(merged["fl_x"]), float(merged.get("fl_y", merged["fl_x"]))
        elif "camera_angle_x" in merged and width:
            fx = 0.5 * width / math.tan(0.5 * float(merged["camera_angle_x"]))
            fy = fx
        else:
            raise ValueError("transforms.json needs fl_x/fl_y or camera_angle_x with w")
        if width <= 0 or height <= 0:
            raise ValueError("transforms.json needs positive w and h")
        cx, cy = float(merged.get("cx", width / 2)), float(merged.get("cy", height / 2))
        frames.append(Frame(Path(str(entry["file_path"])).stem, np.array(entry["transform_matrix"], dtype=np.float64), Intrinsics(fx, fy, cx, cy, width, height), "opengl"))
    if not frames:
        raise ValueError("transforms.json has no frames")
    return frames


def _colmap_intrinsics(parts: list[str]) -> Intrinsics:
    model, width, height = parts[1], int(parts[2]), int(parts[3])
    params = [float(v) for v in parts[4:]]
    # Brown-Conrady coefficients are carried where the model has them; fisheye models project through
    # their own distortion, which is not modelled, so they keep the pinhole part only.
    if model == "SIMPLE_PINHOLE":
        return Intrinsics(params[0], params[0], params[1], params[2], width, height)
    if model == "SIMPLE_RADIAL":
        return Intrinsics(params[0], params[0], params[1], params[2], width, height, k1=params[3])
    if model == "RADIAL":
        return Intrinsics(params[0], params[0], params[1], params[2], width, height, k1=params[3], k2=params[4])
    if model == "SIMPLE_RADIAL_FISHEYE":
        return Intrinsics(params[0], params[0], params[1], params[2], width, height)
    if model in ("PINHOLE", "OPENCV_FISHEYE"):
        return Intrinsics(params[0], params[1], params[2], params[3], width, height)
    if model in ("OPENCV", "FULL_OPENCV"):
        return Intrinsics(*params[:4], width, height, k1=params[4], k2=params[5], p1=params[6], p2=params[7])
    raise ValueError(f"unsupported COLMAP camera model {model}")


def load_colmap(model_dir: Path) -> list[Frame]:
    """Parse a COLMAP text model (cameras.txt, images.txt) into camera-to-world OpenCV frames."""
    model_dir = Path(model_dir)
    cameras: dict[int, Intrinsics] = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cameras[int(parts[0])] = _colmap_intrinsics(parts)
    frames = []
    # Each image has a header line (exactly 10 tokens, integer id first) followed by a 2D-point
    # line (3k tokens, often empty, sometimes stripped by tools). Select headers by shape.
    for line in (model_dir / "images.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 10 or line.startswith("#") or not parts[0].isdigit():
            continue
        qw, qx, qy, qz = (float(v) for v in parts[1:5])
        t = np.array([float(v) for v in parts[5:8]])
        rot = _quat_to_matrix(qw, qx, qy, qz)  # world-to-camera
        c2w = np.eye(4)
        c2w[:3, :3] = rot.T
        c2w[:3, 3] = -rot.T @ t
        frames.append(Frame(Path(parts[9]).stem, c2w, cameras[int(parts[8])], "opencv"))
    if not frames:
        raise ValueError(f"no images in {model_dir / 'images.txt'}")
    return frames


def _quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_thermal_map(path: Path) -> dict[str, Path]:
    """CSV of `rgb_stem,thermal_file` pairs (thermal paths relative to the CSV's directory)."""
    mapping = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        stem, thermal = (part.strip() for part in line.split(",", 1))
        mapping[Path(stem).stem] = Path(path).parent / thermal
    return mapping


def drop_below_ground(scene: GaussianScene, clip_m: float) -> tuple[GaussianScene, int]:
    """Remove Gaussians more than `clip_m` below the ground plane y=0 (SfM floaters, sky matches)."""
    keep = scene.positions[:, 1] >= -clip_m
    dropped = int((~keep).sum())
    return (scene.subset(keep) if dropped else scene), dropped


def drop_far_floaters(scene: GaussianScene, ratio: float = FAR_FLOATER_RATIO) -> tuple[GaussianScene, int]:
    """Remove Gaussians far outside the scene's bulk: beyond `ratio` times the 99th-percentile distance from its median position.

    Trained splats park a few sky and mismatch Gaussians a hundred metres out; they would stretch the scene bounds the viewer frames."""
    centre = np.median(scene.positions, axis=0)
    distance = np.linalg.norm(scene.positions - centre, axis=1)
    keep = distance <= np.percentile(distance, FAR_FLOATER_PERCENTILE) * ratio
    dropped = int((~keep).sum())
    return (scene.subset(keep) if dropped else scene), dropped


def drop_oversized(scene: GaussianScene) -> tuple[GaussianScene, int]:
    """Remove trained-splat debris from a metric scene: Gaussians wider than MAX_GAUSSIAN_SIGMA_M and
    needles longer than NEEDLE_MIN_SIGMA_M with an aspect ratio above NEEDLE_ASPECT. Flat wall patches
    (two long axes, one thin) under the cap survive."""
    ordered = np.sort(scene.scales, axis=1)
    longest, middle = ordered[:, 2], ordered[:, 1]
    oversized = longest > MAX_GAUSSIAN_SIGMA_M
    needle = (longest > NEEDLE_MIN_SIGMA_M) & (longest > NEEDLE_ASPECT * middle)
    keep = ~(oversized | needle)
    dropped = int((~keep).sum())
    return (scene.subset(keep) if dropped else scene), dropped


def _up_from_camera_plane(centres: np.ndarray, mean_up: np.ndarray) -> np.ndarray | None:
    """Normal of the plane the camera centres lie in, when the walk spans a 2-D patch of ground."""
    if len(centres) < PATH_MIN_FRAMES:
        return None
    centred = centres - centres.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred / len(centred))
    if eigenvalues[2] <= 1e-12 or eigenvalues[1] / eigenvalues[2] < PATH_MIN_SPREAD_RATIO:
        return None  # a straight line does not define a plane
    if eigenvalues[0] / eigenvalues[1] > PATH_PLANARITY_RATIO:
        return None  # heights vary too much (stairs, drone) for the path to be a ground plane
    up = eigenvectors[:, 0]
    up = up if float(np.dot(up, mean_up)) >= 0 else -up
    tilt = np.degrees(np.arccos(np.clip(float(np.dot(up, mean_up)), -1.0, 1.0)))
    return up if tilt <= PLANE_UP_MAX_TILT_DEG else None


def estimate_up_from_poses(c2w_opencv: np.ndarray) -> np.ndarray:
    """World up from the camera path and orientations, robust to pitch and to a steady roll.

    A person walking a building holds the phone at a near-constant height, so when the camera
    centres span a 2-D patch they lie in a plane whose normal is gravity: that is the first choice.
    Otherwise, with near-zero roll every camera X axis is horizontal, so up is the direction with
    the least variance across the X axes (an eigenvector); that needs the heading to vary (an arc
    or a corner). For a straight walk we fall back to the mean image-up, biased by the pitch.
    """
    rights = c2w_opencv[:, :3, 0]
    ups = -c2w_opencv[:, :3, 1]
    mean_up = ups.mean(axis=0)
    if np.linalg.norm(mean_up) < 1e-6:
        raise ValueError("camera orientations cancel out; pass up_axis explicitly")
    mean_up /= np.linalg.norm(mean_up)
    plane_up = _up_from_camera_plane(c2w_opencv[:, :3, 3], mean_up)
    if plane_up is not None:
        return plane_up
    eigenvalues, eigenvectors = np.linalg.eigh(rights.T @ rights / len(rights))
    separated = eigenvalues[1] > 1e-9 and eigenvalues[0] / eigenvalues[1] < UP_SEPARATION_RATIO
    if not separated:
        return mean_up
    up = eigenvectors[:, 0]
    return up if float(np.dot(up, mean_up)) >= 0 else -up


def rotation_aligning(vector: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking unit `vector` onto unit `target` (Rodrigues)."""
    v, t = np.asarray(vector, float), np.asarray(target, float)
    axis = np.cross(v, t)
    s, c = np.linalg.norm(axis), float(np.dot(v, t))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


def apply_dataparser(c2w: np.ndarray, dataparser_path: Path) -> np.ndarray:
    """Map a world pose into nerfstudio's dataparser frame (transform then uniform scale)."""
    data = json.loads(Path(dataparser_path).read_text())
    transform = np.eye(4)
    transform[:3, :4] = np.array(data["transform"], dtype=np.float64)
    out = transform @ c2w
    out[:3, 3] *= float(data.get("scale", 1.0))
    return out


def load_thermal_frame(path: Path, thermal_range_c: tuple[float, float] | None) -> np.ndarray:
    """Load one thermal frame as float32 degrees C.

    .npy / .csv / float images are taken as degrees already. Integer images are scaled by their
    dtype's full range (not the frame's own maximum, which would give every frame a different
    scale) into `thermal_range_c`.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=np.float32)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("Pillow is required to read image thermal frames") from exc
    image = Image.open(path)
    if image.mode not in ("L", "I;16", "I;16B", "I;16L", "I", "F"):
        image = image.convert("L")  # palette / RGB / grey+alpha: brightness only, never the alpha channel
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    if thermal_range_c is None:
        raise ValueError(f"{path.name}: integer image thermal frames need thermal_range_c (min, max) to map to degrees C")
    lo, hi = thermal_range_c
    full_scale = float(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else 255.0
    if image.mode == "I" and array.max() <= 65535:
        full_scale = 65535.0  # Pillow widens 16-bit PNGs to int32
    return (lo + (hi - lo) * array.astype(np.float32) / full_scale).astype(np.float32)


def find_thermal_frame(thermal_dir: Path, stem: str) -> Path | None:
    for ext in THERMAL_EXTENSIONS:
        candidate = thermal_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def thermal_intrinsics(rgb: Intrinsics, frame_shape: tuple[int, int], hfov_deg: float | None) -> Intrinsics:
    """Intrinsics for the thermal image: either from its own FOV or the RGB FOV rescaled."""
    height, width = frame_shape
    if hfov_deg is not None:
        fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
        return Intrinsics(fx, fx, width / 2, height / 2, width, height)
    if (width, height) == (rgb.width, rgb.height):
        # Thermal frames registered onto the RGB image grid (ThermoScenes, FLIR MSX exports): the whole
        # calibration carries over, including fy, the principal point and the distortion.
        return rgb
    # Otherwise assume the thermal lens shares the RGB horizontal FOV; keep pixels square (fy = fx)
    # because the aspect ratios differ (e.g. 256x192 against 16:9 video).
    fx = rgb.fx * width / rgb.width
    return Intrinsics(fx, fx, width / 2, height / 2, width, height)


def ingest_capture(config: CaptureConfig) -> IngestResult:
    scene = read_ply(config.ply)
    if config.colmap is not None:
        frames = load_colmap(config.colmap)
    elif config.transforms is not None:
        frames = load_transforms(config.transforms)
    else:
        raise ValueError("pass either transforms (nerfstudio) or colmap (text model)")
    name_map = load_thermal_map(config.thermal_map) if config.thermal_map else {}

    poses, images, used, skipped = [], [], 0, []
    intrinsics: Intrinsics | None = None
    for frame in frames:
        path = name_map.get(frame.name) or find_thermal_frame(config.thermal_dir, frame.name)
        if path is None or not path.exists():
            skipped.append(frame.name)
            continue
        image = load_thermal_frame(path, config.thermal_range_c)
        candidate = thermal_intrinsics(frame.intrinsics, image.shape, config.thermal_hfov_deg)
        if intrinsics is None:
            intrinsics = candidate
        elif (candidate.width, candidate.height) != (intrinsics.width, intrinsics.height):
            raise ValueError(f"{path.name}: thermal frames must share one resolution")
        elif not np.allclose([candidate.fx, candidate.fy, candidate.cx, candidate.cy], [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy], rtol=1e-3):
            raise ValueError(f"{frame.name}: per-frame RGB intrinsics vary; pass thermal_hfov_deg so the fixed thermal lens is used")
        c2w = frame.c2w
        if config.dataparser_transforms is not None:
            c2w = apply_dataparser(c2w, config.dataparser_transforms)
        c2w = c2w @ config.thermal_in_rgb  # rigid offset of the thermal lens in the RGB camera frame
        poses.append(opengl_to_opencv(c2w) if frame.convention == "opengl" else c2w)
        images.append(image)
        used += 1
    if used == 0 or intrinsics is None:
        raise ValueError(f"no thermal frames in {config.thermal_dir} matched the {len(frames)} poses by file stem")

    poses_cv = np.stack(poses)
    baked = bake_temperature(scene.positions, poses_cv, intrinsics, np.stack(images), scene.opacities)
    coverage_fraction = float(np.mean(baked.coverage > 0))
    if coverage_fraction < config.min_coverage:
        raise ValueError(
            f"only {coverage_fraction:.0%} of Gaussians were seen by any thermal frame; the poses probably do not match the splat "
            "(wrong --dataparser-transforms, wrong pose set, or thermal frames paired with the wrong RGB frames)"
        )
    scene = scene.with_temperature(fill_unobserved(scene.positions, baked.temperature))

    estimated_up = None
    if config.up_axis == "z":
        rotation = Z_UP_TO_Y_UP
    elif config.up_axis == "auto":
        estimated_up = estimate_up_from_poses(poses_cv)
        rotation = rotation_aligning(estimated_up, np.array([0.0, 1.0, 0.0]))
    elif config.up_axis == "y":
        rotation = None
    else:
        raise ValueError("up_axis must be 'y', 'z' or 'auto'")
    if rotation is not None:
        scene = rotate_scene(scene, rotation)
    if config.scale is not None:
        if config.scale <= 0:
            raise ValueError("scale must be positive")
        # Trained splats carry Gaussians already clamped to the floor; a scale below 1 must not push them under it.
        scene = replace(scene, positions=(scene.positions * config.scale).astype(np.float32), scales=np.maximum(scene.scales * config.scale, SCALE_FLOOR).astype(np.float32))
    dropped_oversized = 0
    if config.scale is not None:  # the size thresholds are metric
        scene, dropped_oversized = drop_oversized(scene)
    centres = poses_cv[:, :3, 3]
    if rotation is not None:
        centres = centres @ np.asarray(rotation, dtype=np.float64).T
    if config.scale is not None:
        centres = centres * config.scale
    ground_shift_m = None
    dropped_below_ground = 0
    if config.camera_height_m is not None:
        if config.camera_height_m <= 0:
            raise ValueError("camera_height_m must be positive")
        ground_shift_m = float(config.camera_height_m - np.median(centres[:, 1]))
        centres = centres + np.array([0.0, ground_shift_m, 0.0])
        scene = replace(scene, positions=(scene.positions + np.array([0.0, ground_shift_m, 0.0], dtype=np.float32)).astype(np.float32))
        scene, dropped_below_ground = drop_below_ground(scene, BELOW_GROUND_CLIP_M)
    scene, dropped_far = drop_far_floaters(scene)

    meta = {
        "name": config.name,
        "synthetic": False,
        "source": {"ply": str(config.ply), "poses": str(config.colmap or config.transforms), "thermal_dir": str(config.thermal_dir)},
        "frames_used": used,
        "frames_skipped": skipped,
        "coverage_fraction": coverage_fraction,
        "unobserved_fraction": 1.0 - coverage_fraction,
        "thermal_intrinsics": intrinsics.__dict__,
        "thermal_fov_assumed_from_rgb": config.thermal_hfov_deg is None,
        "up_axis_in": config.up_axis,
        "estimated_up": None if estimated_up is None else estimated_up.tolist(),
        "applied_rotation": None if rotation is None else np.asarray(rotation).tolist(),
        "scale_to_metres": config.scale,
        "camera_height_m": config.camera_height_m,
        "ground_shift_m": ground_shift_m,
        "dropped_below_ground": dropped_below_ground,
        "dropped_far": dropped_far,
        "dropped_oversized": dropped_oversized,
        "camera_centroid": centres.mean(axis=0).tolist(),  # where the scan was taken from; the viewer frames the scene from this side
        "units": "metres" if (config.scale is not None or config.colmap is None) else "unscaled (reconstruction units; areas and costs are not metric)",
    }
    return IngestResult(scene, used, tuple(skipped), baked.coverage, coverage_fraction, meta)
