import numpy as np
import pytest

from heatsplat.detect import CAUSE_INSULATION, DetectionParams, detect_leaks
from heatsplat.scene import GaussianScene
from heatsplat.synth import build_plane

MATCH_RADIUS_M = 0.7


def _nearest(point, leaks):
    return min(leaks, key=lambda l: np.linalg.norm(np.array(l.centroid) - np.array(point)))


def test_recovers_every_planted_leak_with_no_false_positives(house):
    result = detect_leaks(house.scene, wall_height_m=house.meta["wall_height_m"])
    truth = house.meta["ground_truth_leaks"]
    assert len(result.leaks) == len(truth)
    for gt in truth:
        nearest = _nearest(gt["center"], result.leaks)
        assert np.linalg.norm(np.array(nearest.centroid) - np.array(gt["center"])) < MATCH_RADIUS_M, gt["name"]
        assert nearest.likely_cause == gt["cause"], gt["name"]
        assert nearest.mean_delta_c > 2.5


def test_leaks_are_ranked_by_severity(house):
    result = detect_leaks(house.scene, wall_height_m=house.meta["wall_height_m"])
    severities = [l.severity for l in result.leaks]
    assert severities == sorted(severities, reverse=True)
    assert [l.rank for l in result.leaks] == list(range(1, len(result.leaks) + 1))


def test_flat_plane_single_patch_geometry():
    plane = build_plane(size_m=6.0, density=300.0, patch=(-0.5, 0.5, 1.5, 2.5), patch_delta_c=4.0)
    result = detect_leaks(plane)
    assert len(result.leaks) == 1
    leak = result.leaks[0]
    assert leak.area_m2 == pytest.approx(1.0, abs=0.3)
    assert leak.mean_delta_c == pytest.approx(4.0, abs=0.5)
    assert np.allclose(leak.centroid, [0, 2.0, 0], atol=0.1)
    assert leak.likely_cause == CAUSE_INSULATION


def test_cold_anomalies_found_with_negative_sign():
    plane = build_plane(patch_delta_c=-4.0)
    assert len(detect_leaks(plane).leaks) == 0
    assert len(detect_leaks(plane, DetectionParams(sign=-1)).leaks) == 1


def test_uniform_plane_has_no_leaks():
    plane = build_plane(patch=None)
    assert detect_leaks(plane).leaks == ()


def test_requires_temperature():
    plane = build_plane()
    bare = GaussianScene(plane.positions, plane.scales, plane.rotations, plane.colors, plane.opacities)
    with pytest.raises(ValueError, match="temperature"):
        detect_leaks(bare)


def test_stricter_threshold_flags_fewer_candidates(house):
    loose = detect_leaks(house.scene, DetectionParams(min_delta_c=1.0))
    strict = detect_leaks(house.scene, DetectionParams(min_delta_c=3.5))
    assert strict.candidate_mask.sum() < loose.candidate_mask.sum()
    assert len(strict.leaks) >= 1


def test_strict_threshold_never_fragments_into_more_leaks(house):
    counts = [len(detect_leaks(house.scene, DetectionParams(min_delta_c=d)).leaks) for d in (1.5, 2.5, 3.5, 4.0)]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == 5


def test_effective_cluster_eps_keeps_configured_radius_on_dense_scenes():
    from heatsplat.detect import DetectionParams, effective_cluster_eps
    xs, zs = np.meshgrid(np.arange(40) * 0.05, np.arange(40) * 0.05)
    dense = np.stack([xs.ravel(), np.zeros(1600), zs.ravel()], axis=1)
    params = DetectionParams()
    assert effective_cluster_eps(dense, params) == params.cluster_eps_m


def test_sparse_point_scene_still_clusters_a_warm_patch():
    """A COLMAP-sparse scene (0.3 m spacing) must not silently return zero leaks."""
    from heatsplat.detect import DetectionParams, detect_leaks, effective_cluster_eps
    from heatsplat.pointsplat import PointCloud, gaussians_from_points
    spacing = 0.3  # window-sized patch (1.5 m) still holds >= cluster_min_points at this spacing
    xs, zs = np.meshgrid(np.arange(60) * spacing, np.arange(40) * spacing)
    positions = np.stack([xs.ravel(), zs.ravel(), np.zeros(xs.size)], axis=1)  # wall in the XY plane
    rng = np.random.default_rng(1)
    positions = positions + rng.normal(scale=0.01, size=positions.shape)
    temperature = rng.normal(loc=-5.0, scale=0.3, size=len(positions)).astype(np.float32)
    patch = (positions[:, 0] > 8) & (positions[:, 0] < 9.5) & (positions[:, 1] > 4) & (positions[:, 1] < 5.5)
    temperature[patch] += 6.0
    cloud = PointCloud(positions, np.full((len(positions), 3), 0.5, dtype=np.float32))
    scene = gaussians_from_points(cloud).with_temperature(temperature)
    params = DetectionParams()
    assert effective_cluster_eps(scene.positions.astype(np.float64), params) > params.cluster_eps_m
    result = detect_leaks(scene, params)
    assert len(result.leaks) == 1
    leak = result.leaks[0]
    assert 8 < leak.centroid[0] < 9.5 and 4 < leak.centroid[1] < 5.5
    assert leak.n_points >= int(patch.sum()) * 0.8


def test_cluster_settings_lower_the_cluster_size_only_for_sparse_scenes():
    from heatsplat.detect import MIN_CLUSTER_POINTS_FLOOR, DetectionParams, cluster_settings
    params = DetectionParams()
    xs, zs = np.meshgrid(np.arange(40) * 0.05, np.arange(40) * 0.05)
    dense = np.stack([xs.ravel(), np.zeros(1600), zs.ravel()], axis=1)
    xs, zs = np.meshgrid(np.arange(40) * 0.3, np.arange(40) * 0.3)
    sparse = np.stack([xs.ravel(), np.zeros(1600), zs.ravel()], axis=1)
    assert cluster_settings(dense, params) == (params.cluster_eps_m, params.cluster_min_points)
    eps, min_points = cluster_settings(sparse, params)
    assert eps > params.cluster_eps_m
    assert min_points == MIN_CLUSTER_POINTS_FLOOR
