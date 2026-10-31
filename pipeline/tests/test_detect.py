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
