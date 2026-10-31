"""Turn a baked scene into everything the viewer needs: three .splat files plus leaks.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heatsplat.colormap import grayscale, ironbow, normalize
from heatsplat.cost import CostModel, estimate_cost
from heatsplat.detect import DetectionParams, DetectionResult, baseline_signature, cluster_leaks, compute_residual
from heatsplat.fsutil import atomic_savez, atomic_write_text
from heatsplat.scene import GaussianScene
from heatsplat.splat_io import write_splat

GAUSSIANS_FILE = "gaussians.npz"
META_FILE = "meta.json"
LEAKS_FILE = "leaks.json"
RESIDUAL_FILE = "residual.npz"
SPLAT_MODES = ("rgb", "thermal", "leaks")
REQUIRED_SCENE_FILES = (GAUSSIANS_FILE, META_FILE, LEAKS_FILE, *(f"{m}.splat" for m in SPLAT_MODES))
COLOR_PERCENTILES = (2.0, 98.0)
DIM_BRIGHTNESS = 0.35
LEAK_COLOUR_FLOOR = 0.4


@dataclass(frozen=True)
class SceneSummary:
    scene_id: str
    name: str
    n_gaussians: int
    temp_min_c: float
    temp_max_c: float
    n_leaks: int
    total_cost_per_year: float
    total_watts: float
    currency: str


def thermal_colors(scene: GaussianScene) -> np.ndarray:
    return ironbow(normalize(scene.temperature, *COLOR_PERCENTILES))


def leak_colors(scene: GaussianScene, result: DetectionResult) -> np.ndarray:
    colors = grayscale(scene.colors, DIM_BRIGHTNESS)
    if not result.leaks:
        return colors
    members = np.concatenate([leak.point_indices for leak in result.leaks])
    deltas = result.residual_c[members]
    strength = LEAK_COLOUR_FLOOR + (1.0 - LEAK_COLOUR_FLOOR) * normalize(deltas, 5.0, 95.0)
    return _assign(colors, members, ironbow(strength))


def _assign(base: np.ndarray, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[indices] = values
    return out


def leaks_payload(scene_id: str, result: DetectionResult, cost_model: CostModel, meta: dict) -> dict:
    leaks = []
    for leak in result.leaks:
        cost = estimate_cost(leak, cost_model)
        leaks.append({**leak.to_dict(), "cost": cost.to_dict()})
    return {
        "scene_id": scene_id,
        "params": result.params.to_dict(),
        "cost_model": cost_model.to_dict(),
        "summary": {
            "n_leaks": len(leaks),
            "total_cost_per_year": round(sum(l["cost"]["cost_per_year"] for l in leaks), 2),
            "total_watts": round(sum(l["cost"]["watts"] for l in leaks), 1),
            "currency": cost_model.currency,
            "n_candidates": int(result.candidate_mask.sum()),
        },
        "ground_truth_leaks": meta.get("ground_truth_leaks", []),
        "leaks": leaks,
    }


def cached_residual(scene: GaussianScene, scene_dir: Path, params: DetectionParams) -> np.ndarray:
    """Load the residual field if it was computed with the same baseline parameters, else compute and store it."""
    path = scene_dir / RESIDUAL_FILE
    signature = baseline_signature(params)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if json.loads(str(cached["signature"])) == signature and cached["residual"].shape == (len(scene),):
                return cached["residual"]
    residual = compute_residual(scene, params)
    atomic_savez(path, residual=residual, signature=np.array(json.dumps(signature)))
    return residual


def analyze_scene(scene_dir: Path, params: DetectionParams = DetectionParams(), cost_model: CostModel = CostModel()) -> dict:
    """Re-run detection for an exported scene, rewrite leaks.splat and leaks.json, return the JSON."""
    scene = GaussianScene.load(scene_dir / GAUSSIANS_FILE)
    meta = json.loads((scene_dir / META_FILE).read_text())
    result = cluster_leaks(scene, cached_residual(scene, scene_dir, params), params, meta.get("wall_height_m"))
    write_splat(scene_dir / "leaks.splat", scene, leak_colors(scene, result))
    payload = leaks_payload(scene_dir.name, result, cost_model, meta)
    atomic_write_text(scene_dir / LEAKS_FILE, json.dumps(payload, indent=2))
    return payload


def export_scene(scene: GaussianScene, scene_dir: Path, meta: dict, params: DetectionParams = DetectionParams(), cost_model: CostModel = CostModel()) -> dict:
    """Write a full scene directory from a baked scene. Returns the leaks payload."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / RESIDUAL_FILE).unlink(missing_ok=True)  # a new scene invalidates any cached residual
    scene.save(scene_dir / GAUSSIANS_FILE)
    lo, hi = scene.bounds()
    color_lo, color_hi = np.percentile(scene.temperature, COLOR_PERCENTILES)
    full_meta = {
        **meta,
        "scene_id": scene_dir.name,
        "n_gaussians": len(scene),
        "bounds": {"min": lo.tolist(), "max": hi.tolist()},
        "temp_range_c": [float(scene.temperature.min()), float(scene.temperature.max())],
        "color_range_c": [float(color_lo), float(color_hi)],  # what the thermal palette actually spans
    }
    atomic_write_text(scene_dir / META_FILE, json.dumps(full_meta, indent=2))
    write_splat(scene_dir / "rgb.splat", scene)
    write_splat(scene_dir / "thermal.splat", scene, thermal_colors(scene))
    return analyze_scene(scene_dir, params, cost_model)


def is_complete_scene(scene_dir: Path) -> bool:
    return all((scene_dir / name).exists() for name in REQUIRED_SCENE_FILES)


def load_summary(scene_dir: Path) -> SceneSummary:
    meta = json.loads((scene_dir / META_FILE).read_text())
    leaks = json.loads((scene_dir / LEAKS_FILE).read_text())
    return SceneSummary(
        scene_id=scene_dir.name,
        name=meta.get("name", scene_dir.name),
        n_gaussians=meta["n_gaussians"],
        temp_min_c=meta["temp_range_c"][0],
        temp_max_c=meta["temp_range_c"][1],
        n_leaks=leaks["summary"]["n_leaks"],
        total_cost_per_year=leaks["summary"]["total_cost_per_year"],
        total_watts=leaks["summary"]["total_watts"],
        currency=leaks["summary"]["currency"],
    )
