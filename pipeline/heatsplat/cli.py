"""Command line: `python -m heatsplat.cli synth --scene demo-house` and `analyze`."""

from __future__ import annotations

import argparse
import sys
import time

from heatsplat.detect import DetectionParams
from heatsplat.export import analyze_scene, export_scene
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
    for name in ("synth", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--scene", required=True, help="scene id, e.g. demo-house")
        p.add_argument("--min-delta", type=float, default=DetectionParams.min_delta_c)
        p.add_argument("--min-z", type=float, default=DetectionParams.min_zscore)
        p.add_argument("--eps", type=float, default=DetectionParams.cluster_eps_m)
        p.add_argument("--min-points", type=int, default=DetectionParams.cluster_min_points)
        if name == "synth":
            p.add_argument("--density", type=float, default=SynthParams.density)
            p.add_argument("--seed", type=int, default=SynthParams.seed)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    target = scene_dir(args.scene)
    if args.command == "synth":
        result = build_house(SynthParams(density=args.density, seed=args.seed))
        print(f"built {len(result.scene):,} Gaussians in {time.perf_counter() - t0:.1f}s")
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
