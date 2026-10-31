import json

from heatsplat.detect import DetectionParams
from heatsplat.export import analyze_scene, export_scene, load_summary
from heatsplat.synth import build_plane


def test_export_writes_all_viewer_files_and_analyze_reruns(tmp_path):
    scene = build_plane()
    scene_dir = tmp_path / "plane"
    payload = export_scene(scene, scene_dir, {"name": "plane"})
    for name in ("gaussians.npz", "meta.json", "rgb.splat", "thermal.splat", "leaks.splat", "leaks.json"):
        assert (scene_dir / name).exists(), name
    assert payload["summary"]["n_leaks"] == 1
    assert payload["leaks"][0]["cost"]["cost_per_year"] > 0
    meta = json.loads((scene_dir / "meta.json").read_text())
    assert meta["n_gaussians"] == len(scene)
    lo, hi = meta["color_range_c"]
    assert meta["temp_range_c"][0] <= lo < hi <= meta["temp_range_c"][1]
    assert not list(scene_dir.glob(".*.tmp")), "atomic writes must leave no temp files"

    strict = analyze_scene(scene_dir, DetectionParams(min_delta_c=10.0))
    assert strict["summary"]["n_leaks"] == 0
    assert json.loads((scene_dir / "leaks.json").read_text())["params"]["min_delta_c"] == 10.0

    summary = load_summary(scene_dir)
    assert (scene_dir / "residual.npz").exists()
    assert summary.scene_id == "plane" and summary.n_leaks == 0


def test_cached_residual_is_reused_until_baseline_params_change(tmp_path, monkeypatch):
    import heatsplat.export as export_module

    scene = build_plane()
    scene_dir = tmp_path / "plane"
    export_scene(scene, scene_dir, {"name": "plane"})
    calls = []
    real = export_module.compute_residual
    monkeypatch.setattr(export_module, "compute_residual", lambda s, p: calls.append(p) or real(s, p))
    analyze_scene(scene_dir, DetectionParams(min_delta_c=2.0))
    assert calls == []  # thresholds changed, baseline did not: cache hit
    analyze_scene(scene_dir, DetectionParams(baseline_radius_m=1.0))
    assert len(calls) == 1  # baseline changed: recomputed
