"""Command line: `synth` (synthetic demo), `analyze` (re-run detection), `ingest` (real capture)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from heatsplat.detect import DetectionParams
from heatsplat.export import analyze_scene, export_scene
from heatsplat.ingest import CaptureConfig, ingest_capture
from heatsplat.paths import scene_dir
from heatsplat.synth import SynthParams, build_house


def _detection_params(args: argparse.Namespace) -> DetectionParams:
    return DetectionParams(
        min_delta_c=args.min_delta,
        min_zscore=args.min_z,
        cluster_eps_m=args.eps,
        cluster_min_points=args.min_points,
    )


def _print_leaks(payload: dict) -> None:
    s = payload["summary"]
    print(f"{s['n_leaks']} leaks, {s['n_candidates']} candidate Gaussians, est. {s['total_watts']} W, {s['currency']} {s['total_cost_per_year']}/yr")
    for leak in payload["leaks"]:
        c = leak["centroid"]
        print(f"  {leak['id']:>3}  dT={leak['mean_delta_c']:.1f}C  area={leak['area_m2']:.2f}m2  ${leak['cost']['cost_per_year']:>6}/yr  at ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})  {leak['likely_cause']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="heatsplat")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("synth", "analyze", "ingest"):
        p = sub.add_parser(name)
        p.add_argument("--scene", required=True, help="scene id, e.g. demo-house")
        p.add_argument("--min-delta", type=float, default=DetectionParams.min_delta_c)
        p.add_argument("--min-z", type=float, default=DetectionParams.min_zscore)
        p.add_argument("--eps", type=float, default=DetectionParams.cluster_eps_m)
        p.add_argument("--min-points", type=int, default=DetectionParams.cluster_min_points)
        if name == "synth":
            p.add_argument("--density", type=float, default=SynthParams.density)
            p.add_argument("--seed", type=int, default=SynthParams.seed)
        if name == "ingest":
            p.add_argument("--ply", required=True, type=Path, help="trained Gaussian splat .ply")
            p.add_argument("--transforms", type=Path, default=None, help="nerfstudio transforms.json with RGB poses")
            p.add_argument("--colmap", type=Path, default=None, help="COLMAP text model dir (cameras.txt, images.txt) instead of --transforms")
            p.add_argument("--thermal-map", type=Path, default=None, help="CSV rgb_stem,thermal_file when names differ")
            p.add_argument("--thermal-dir", required=True, type=Path, help="thermal frames named like the RGB frames")
            p.add_argument("--thermal-range", type=str, default=None, help="min,max degrees C for 8/16-bit grayscale frames")
            p.add_argument("--thermal-hfov", type=float, default=None, help="thermal lens horizontal FOV in degrees")
            p.add_argument("--dataparser-transforms", type=Path, default=None, help="nerfstudio dataparser_transforms.json if the ply is in that frame")
            p.add_argument("--up", choices=("y", "z", "auto"), default=None, help="world up axis of the capture (default: auto with --colmap, z with --transforms)")
            p.add_argument("--scale", type=float, default=None, help="multiply the reconstruction by this to reach metres (measure a known length)")
            p.add_argument("--name", type=str, default="Real capture")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    target = scene_dir(args.scene)
    if args.command == "synth":
        result = build_house(SynthParams(density=args.density, seed=args.seed))
        print(f"built {len(result.scene):,} Gaussians in {time.perf_counter() - t0:.1f}s")
        payload = export_scene(result.scene, target, result.meta, _detection_params(args))
    elif args.command == "ingest":
        thermal_range = None
        if args.thermal_range:
            values = [float(v) for v in args.thermal_range.split(",")]
            if len(values) != 2 or values[0] >= values[1]:
                print("--thermal-range must be min,max with min < max", file=sys.stderr)
                return 2
            thermal_range = (values[0], values[1])
        up_axis = args.up or ("auto" if args.colmap is not None else "z")
        if (args.transforms is None) == (args.colmap is None):
            print("pass exactly one of --transforms or --colmap", file=sys.stderr)
            return 2
        result = ingest_capture(CaptureConfig(args.ply, args.transforms, args.thermal_dir, colmap=args.colmap, thermal_map=args.thermal_map, thermal_range_c=thermal_range, thermal_hfov_deg=args.thermal_hfov, up_axis=up_axis, scale=args.scale, dataparser_transforms=args.dataparser_transforms, name=args.name))
        print(f"baked {len(result.scene):,} Gaussians from {result.frames_used} thermal frames ({result.coverage_fraction:.0%} observed); skipped {len(result.frames_skipped)} poses without frames")
        if result.meta["units"] != "metres":
            print("WARNING: reconstruction units are not metres; pass --scale (metres per unit) or areas and costs will be off", file=sys.stderr)
        if result.meta["thermal_fov_assumed_from_rgb"]:
            print("note: thermal lens FOV assumed equal to the RGB camera; pass --thermal-hfov for the real lens", file=sys.stderr)
        payload = export_scene(result.scene, target, result.meta, _detection_params(args))
    else:
        if not target.exists():
            print(f"no scene at {target}", file=sys.stderr)
            return 1
        payload = analyze_scene(target, _detection_params(args))
    print(f"wrote {target} in {time.perf_counter() - t0:.1f}s")
    _print_leaks(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
