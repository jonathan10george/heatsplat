import json

from dataclasses import replace

import numpy as np
import pytest

from heatsplat.detect import detect_leaks
from heatsplat.bake import Intrinsics
from heatsplat.ingest import CaptureConfig, _colmap_intrinsics, drop_far_floaters, drop_oversized, estimate_up_from_poses, ingest_capture, load_colmap, load_thermal_frame, load_transforms, rotation_aligning, thermal_intrinsics
from heatsplat.ply_io import read_ply, write_ply
from heatsplat.scene import SCALE_FLOOR, Z_UP_TO_Y_UP, GaussianScene, rotate_scene, surface_normals
from heatsplat.synth import LABEL_GROUND
from heatsplat.synthcapture import look_at_pose
from heatsplat.synthcapture import CaptureLayout, write_capture

MATCH_RADIUS_M = 0.8


@pytest.fixture(scope="module")
def capture(house, tmp_path_factory):
    out = tmp_path_factory.mktemp("capture")
    write_capture(house.scene, out, CaptureLayout(n_frames=12))
    return out


def test_capture_files_match_ingest_contract(capture):
    frames = load_transforms(capture / "transforms.json")
    assert len(frames) == 12 and frames[0].intrinsics.width == 320
    assert sorted(p.stem for p in (capture / "thermal").glob("*.npy")) == sorted(f.name for f in frames)


def test_ingest_recovers_temperature_and_leaks(house, capture):
    result = ingest_capture(CaptureConfig(capture / "scene.ply", capture / "transforms.json", capture / "thermal", up_axis="y", name="synthetic capture"))
    assert result.frames_used == 12 and result.frames_skipped == ()
    house_surfaces = house.scene.labels != LABEL_GROUND  # the lawn's far corners are never in frame
    assert np.mean(result.coverage[house_surfaces] > 0) > 0.9
    truth = house.scene.temperature
    baked = result.scene.temperature
    assert np.corrcoef(truth, baked)[0, 1] > 0.95
    assert np.median(np.abs(truth - baked)) < 0.5

    leaks = detect_leaks(result.scene, wall_height_m=house.meta["wall_height_m"]).leaks
    found = 0
    for gt in house.meta["ground_truth_leaks"]:
        if any(np.linalg.norm(np.array(l.centroid) - np.array(gt["center"])) < MATCH_RADIUS_M for l in leaks):
            found += 1
    assert found >= 4, [l.centroid for l in leaks]


def test_missing_thermal_frames_are_skipped_not_fatal(capture, tmp_path):
    thermal = tmp_path / "thermal-partial"
    thermal.mkdir()
    for p in sorted((capture / "thermal").glob("*.npy"))[:6]:
        (thermal / p.name).write_bytes(p.read_bytes())
    result = ingest_capture(CaptureConfig(capture / "scene.ply", capture / "transforms.json", thermal, up_axis="y"))
    assert result.frames_used == 6 and len(result.frames_skipped) == 6


def test_grayscale_image_frames_need_a_range(tmp_path):
    from PIL import Image

    Image.fromarray(np.full((4, 4), 128, np.uint8)).save(tmp_path / "f.png")
    with pytest.raises(ValueError, match="thermal_range_c"):
        load_thermal_frame(tmp_path / "f.png", None)
    frame = load_thermal_frame(tmp_path / "f.png", (0.0, 25.5))
    assert frame.shape == (4, 4) and frame[0, 0] == pytest.approx(12.8, abs=0.01)


def test_z_up_rotation_keeps_normals_consistent(house):
    rotated = rotate_scene(house.scene, Z_UP_TO_Y_UP)
    expected = surface_normals(house.scene) @ Z_UP_TO_Y_UP.T
    got = surface_normals(rotated)
    assert np.allclose(np.abs(np.einsum("ij,ij->i", got, expected)), 1.0, atol=1e-4)
    assert np.allclose(rotated.positions[:, 1], house.scene.positions[:, 2])


def test_transforms_json_accepts_camera_angle_x(tmp_path):
    (tmp_path / "t.json").write_text(json.dumps({"camera_angle_x": 1.0, "w": 100, "h": 50, "frames": [{"file_path": "a.jpg", "transform_matrix": np.eye(4).tolist()}]}))
    frames = load_transforms(tmp_path / "t.json")
    assert frames[0].intrinsics.fx == pytest.approx(50 / np.tan(0.5))


def test_colmap_model_matches_transforms_json(capture):
    from_json = load_transforms(capture / "transforms.json")
    from_colmap = load_colmap(capture / "colmap")
    assert [f.name for f in from_colmap] == [f.name for f in from_json]
    for a, b in zip(from_json, from_colmap):
        assert np.allclose(a.c2w_opencv(), b.c2w_opencv(), atol=1e-6)
        assert b.intrinsics == a.intrinsics


def test_ingest_from_colmap_with_auto_up_matches_truth(house, capture):
    result = ingest_capture(CaptureConfig(capture / "scene.ply", None, capture / "thermal", colmap=capture / "colmap", up_axis="auto"))
    # The synthetic world is already Y up, so auto must recover (almost) the identity rotation.
    assert np.allclose(result.scene.positions, house.scene.positions, atol=0.05)
    assert np.corrcoef(house.scene.temperature, result.scene.temperature)[0, 1] > 0.95


def test_auto_up_recovers_a_tilted_world():
    rng = np.random.default_rng(4)
    target = rng.normal(size=3)
    tilt = rotation_aligning(np.array([0.0, 1.0, 0.0]), target / np.linalg.norm(target))
    poses = []
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        c2w = np.eye(4)
        c2w[:3, 1] = [0, -1, 0]  # OpenCV camera: y down, so "up" is -y = +Y world
        c2w[:3, 2] = [np.sin(angle), 0, np.cos(angle)]
        c2w[:3, 0] = np.cross(c2w[:3, 1], c2w[:3, 2])
        c2w[:3, :3] = tilt @ c2w[:3, :3]
        poses.append(c2w)
    up = estimate_up_from_poses(np.stack(poses))
    assert np.allclose(up, tilt @ np.array([0, 1.0, 0]), atol=1e-6)
    assert np.allclose(rotation_aligning(up, np.array([0, 1.0, 0])) @ up, [0, 1, 0], atol=1e-6)


def test_poses_that_miss_the_splat_are_rejected(capture, tmp_path):
    data = json.loads((capture / "transforms.json").read_text())
    for frame in data["frames"]:
        frame["transform_matrix"][0][3] += 500.0  # slide every camera 500 m sideways
    (tmp_path / "far.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="do not match the splat"):
        ingest_capture(CaptureConfig(capture / "scene.ply", tmp_path / "far.json", capture / "thermal", up_axis="y"))


def test_scale_converts_reconstruction_units_to_metres(capture):
    base = ingest_capture(CaptureConfig(capture / "scene.ply", capture / "transforms.json", capture / "thermal", up_axis="y"))
    scaled = ingest_capture(CaptureConfig(capture / "scene.ply", capture / "transforms.json", capture / "thermal", up_axis="y", scale=2.0))
    assert np.allclose(scaled.scene.positions, base.scene.positions * 2)
    assert np.allclose(scaled.scene.scales, base.scene.scales * 2)
    assert scaled.meta["units"] == "metres" and scaled.meta["scale_to_metres"] == 2.0


def test_colmap_without_scale_is_flagged_as_unscaled(capture):
    result = ingest_capture(CaptureConfig(capture / "scene.ply", None, capture / "thermal", colmap=capture / "colmap", up_axis="auto"))
    assert result.meta["units"].startswith("unscaled")
    assert result.meta["estimated_up"] is not None and result.meta["applied_rotation"] is not None


def test_integer_frames_scale_by_dtype_not_frame_maximum(tmp_path):
    from PIL import Image

    Image.fromarray(np.full((4, 4), 200, np.uint16)).save(tmp_path / "dark16.png")
    Image.fromarray(np.full((4, 4), 20000, np.uint16)).save(tmp_path / "bright16.png")
    dark = load_thermal_frame(tmp_path / "dark16.png", (0.0, 65.535))
    bright = load_thermal_frame(tmp_path / "bright16.png", (0.0, 65.535))
    assert dark[0, 0] == pytest.approx(0.2, abs=1e-3)
    assert bright[0, 0] == pytest.approx(20.0, abs=1e-3)
    Image.fromarray(np.dstack([np.full((4, 4), 100, np.uint8), np.full((4, 4), 255, np.uint8)]), "LA").save(tmp_path / "la.png")
    assert load_thermal_frame(tmp_path / "la.png", (0.0, 25.5))[0, 0] == pytest.approx(10.0, abs=0.01)


def test_auto_up_ignores_camera_pitch_on_an_arc():
    target = np.array([0.0, 0.0, 0.0])
    poses = [look_at_pose(np.array([10 * np.sin(a), 6.0, 10 * np.cos(a)]), target) for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
    up = estimate_up_from_poses(np.stack(poses))  # cameras pitched ~31 degrees down
    assert np.allclose(up, [0, 1, 0], atol=1e-6)


def test_auto_up_straight_walk_falls_back_to_mean_up():
    poses = [look_at_pose(np.array([x, 1.5, 8.0]), np.array([x, 1.5, 0.0])) for x in np.linspace(-3, 3, 7)]
    assert np.allclose(estimate_up_from_poses(np.stack(poses)), [0, 1, 0], atol=1e-6)


def test_colmap_images_without_blank_point_lines(capture, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "cameras.txt").write_text((capture / "colmap" / "cameras.txt").read_text())
    stripped = "\n".join(ln for ln in (capture / "colmap" / "images.txt").read_text().splitlines() if ln.strip())
    (model / "images.txt").write_text(stripped + "\n")
    assert len(load_colmap(model)) == len(load_colmap(capture / "colmap")) == 12


def test_auto_up_uses_the_walking_plane_when_heading_barely_changes():
    """A zig-zag walk at phone height, always facing the facade and pitched 25 degrees up: the
    camera X axes are all parallel, so only the plane of camera centres can give gravity."""
    pitch = np.radians(25.0)
    poses = []
    for row, z in enumerate((9.0, 7.5, 6.0, 4.5)):
        for x in np.linspace(-4, 4, 5):
            eye = np.array([x, 1.5 + 0.02 * (row % 2), z])
            target = eye + np.array([0.0, np.sin(pitch) * 10, -np.cos(pitch) * 10])
            poses.append(look_at_pose(eye, target))
    up = estimate_up_from_poses(np.stack(poses))
    assert np.degrees(np.arccos(np.clip(up @ np.array([0, 1.0, 0]), -1, 1))) < 1.0


def test_auto_up_ignores_the_walking_plane_on_stairs():
    """Climbing while shooting: centres are not coplanar, so the orientation cue must take over."""
    target = np.array([0.0, 0.0, 0.0])
    poses = [look_at_pose(np.array([10 * np.sin(a), 2.0 + 3.0 * a, 10 * np.cos(a)]), target) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    up = estimate_up_from_poses(np.stack(poses))
    assert np.degrees(np.arccos(np.clip(abs(up @ np.array([0, 1.0, 0])), -1, 1))) < 10.0


def test_camera_height_puts_the_ground_at_zero(capture):
    """Cameras in the synthetic capture fly at 2.5 m; declaring that height leaves the scene in
    place, declaring 1.2 m slides everything down by 1.3 m (the lawn stays above the below-ground
    clip), and the shift is recorded."""
    base = CaptureConfig(capture / "scene.ply", capture / "transforms.json", capture / "thermal", up_axis="y")
    plain = ingest_capture(base).scene
    same = ingest_capture(replace(base, camera_height_m=2.5))
    lower = ingest_capture(replace(base, camera_height_m=1.2))
    assert np.allclose(same.scene.positions, plain.positions, atol=1e-3)
    assert np.allclose(lower.scene.positions[:, 1], plain.positions[:, 1] - 1.3, atol=1e-3)
    assert np.allclose(lower.scene.positions[:, [0, 2]], plain.positions[:, [0, 2]], atol=1e-6)
    assert lower.meta["ground_shift_m"] == pytest.approx(-1.3, abs=1e-3)
    assert lower.meta["dropped_below_ground"] == 0
    with pytest.raises(ValueError, match="camera_height_m"):
        ingest_capture(replace(base, camera_height_m=0.0))


def test_drop_below_ground_removes_only_deep_floaters():
    from heatsplat.ingest import drop_below_ground
    positions = np.array([[0, 2.0, 0], [0, -0.4, 0], [0, -3.0, 0], [1, 5.0, 1]], dtype=np.float32)
    scene = GaussianScene(positions, np.full((4, 3), 0.1, np.float32), np.tile([1, 0, 0, 0], (4, 1)).astype(np.float32), np.full((4, 3), 0.5, np.float32), np.ones(4, np.float32))
    kept, dropped = drop_below_ground(scene, 1.5)
    assert dropped == 1 and len(kept) == 3
    assert kept.positions[:, 1].min() == pytest.approx(-0.4)
    same, none_dropped = drop_below_ground(kept, 1.5)
    assert none_dropped == 0 and same is kept


def test_scale_below_one_keeps_floor_sized_gaussians_valid(capture, tmp_path):
    # Trained splats carry Gaussians already clamped to SCALE_FLOOR; shrinking the scene to metres must not reject them.
    scene = read_ply(capture / "scene.ply")
    scales = scene.scales.copy()
    scales[:5] = SCALE_FLOOR
    write_ply(tmp_path / "floor.ply", replace(scene, scales=scales))
    result = ingest_capture(CaptureConfig(tmp_path / "floor.ply", capture / "transforms.json", capture / "thermal", up_axis="y", scale=0.5))
    assert result.scene.scales.min() >= SCALE_FLOOR
    assert np.allclose(result.scene.scales[:5], SCALE_FLOOR)
    assert np.allclose(result.scene.scales[5:], scene.scales[5:] * 0.5)


def test_drop_far_floaters_removes_only_distant_debris():
    rng = np.random.default_rng(0)
    bulk = rng.uniform(-10.0, 10.0, (500, 3)).astype(np.float32)
    debris = np.array([[300.0, 0.0, 0.0], [0.0, -250.0, 0.0], [0.0, 0.0, 400.0]], dtype=np.float32)
    positions = np.concatenate([bulk, debris])
    n = len(positions)
    scene = GaussianScene(
        positions=positions,
        scales=np.full((n, 3), 0.05, dtype=np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        colors=np.full((n, 3), 0.5, dtype=np.float32),
        opacities=np.ones(n, dtype=np.float32),
    )
    cleaned, dropped = drop_far_floaters(scene)
    assert dropped == 3 and len(cleaned) == 500
    assert np.all(np.abs(cleaned.positions) <= 10.0)
    assert drop_far_floaters(cleaned)[1] == 0


def _scene_with_scales(scales: np.ndarray) -> GaussianScene:
    n = len(scales)
    return GaussianScene(
        positions=np.zeros((n, 3), dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        colors=np.full((n, 3), 0.5, dtype=np.float32),
        opacities=np.ones(n, dtype=np.float32),
    )


def test_drop_oversized_removes_fog_and_needles_but_keeps_wall_patches():
    scene = _scene_with_scales([
        [0.05, 0.05, 0.01],  # ordinary surface Gaussian
        [1.5, 1.5, 0.01],  # flat wall patch under the cap: keep
        [0.8, 0.05, 0.05],  # short frame rail: keep
        [3.0, 3.0, 0.5],  # sky / ground fog: drop
        [2.5, 0.04, 0.04],  # needle streak: drop
    ])
    cleaned, dropped = drop_oversized(scene)
    assert dropped == 2 and len(cleaned) == 3
    assert cleaned.scales.max() == pytest.approx(1.5)


def test_meta_records_where_the_cameras_were(capture):
    result = ingest_capture(CaptureConfig(capture / "scene.ply", capture / "transforms.json", capture / "thermal", up_axis="y", scale=1.0, camera_height_m=1.5))
    centroid = result.meta["camera_centroid"]
    assert len(centroid) == 3 and abs(centroid[1] - 1.5) < 0.5
    assert result.meta["dropped_oversized"] == 0


def test_thermal_frames_sized_like_the_rgb_keep_its_full_calibration():
    rgb = Intrinsics(358.5, 627.6, 239.6, 320.9, 480, 640, k1=0.0068, p2=0.06)
    assert thermal_intrinsics(rgb, (640, 480), None) == rgb
    small = thermal_intrinsics(rgb, (192, 256), None)
    assert small.fx == small.fy == pytest.approx(358.5 * 256 / 480)
    assert small.k1 == 0.0 and small.p2 == 0.0


def test_colmap_intrinsics_carry_distortion_coefficients():
    opencv = _colmap_intrinsics("1 OPENCV 480 640 358.5 627.6 239.6 320.9 0.0068 -1.5e-06 -0.0207 0.0616".split())
    assert (opencv.fx, opencv.fy, opencv.k1, opencv.k2, opencv.p1, opencv.p2) == (358.5, 627.6, 0.0068, -1.5e-06, -0.0207, 0.0616)
    radial = _colmap_intrinsics("1 SIMPLE_RADIAL 1920 1080 1500 960 540 -0.05".split())
    assert radial.fx == radial.fy == 1500 and radial.k1 == -0.05 and radial.p1 == 0.0
