"""Physical bounds, rigid poses and persistent coverage of the finite beam."""

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from probe_tracking.kidney_scan import KidneyScan
from probe_tracking.kidney_voxels import VoxelKidney, intersect_voxels_with_fan, validate_voxel_size, voxelize_kidney
from probe_tracking.probe_scan_profile import load_convex_scan_profile
from probe_tracking.ultrasound_geometry import make_fan_geometry, validate_fan_parameters

ROOT = Path(__file__).resolve().parents[1]
KIDNEY = ROOT / "assets/kidney/kidney_right.stl"
PROBE = ROOT / "assets/Vscan_Air_CL/Vscan_Air_CL.obj"


def fan(**kwargs):
    return make_fan_geometry([0, 0, 0], curvature_radius_mm=10, **{"depth_cm": 10, **kwargs})


def pose(translation=(0, 0, 0), rotation=None):
    transform = np.eye(4)
    transform[:3, 3] = translation
    if rotation is not None:
        transform[:3, :3] = rotation
    return transform


def hits(centers, *, kidney_pose=None, probe_pose=None, beam=None):
    kidney_pose = np.eye(4) if kidney_pose is None else kidney_pose
    probe_pose = np.eye(4) if probe_pose is None else probe_pose
    result = intersect_voxels_with_fan(
        VoxelKidney(centers, 2),
        kidney_pose[:3, 3],
        kidney_pose[:3, :3],
        probe_pose[:3, 3],
        probe_pose[:3, :3],
        fan() if beam is None else beam,
    )
    return result.hit_mask


def test_voxelization_fills_volume_without_mutating_source_stl_vertices():
    box = trimesh.creation.box(extents=[8, 8, 8])
    triangles = box.triangles + [20, 40, 60]
    mesh = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(36).reshape(-1, 3),
        process=False,
    )
    before = mesh.vertices.copy()
    assert not mesh.is_watertight
    voxels = voxelize_kidney(mesh, 2)
    assert np.any(np.all(voxels.centers == [20, 40, 60], axis=1))
    assert voxels.voxel_count == 125
    np.testing.assert_array_equal(mesh.vertices, before)
    assert not mesh.is_watertight


def test_voxelization_rejects_open_mesh_and_excessive_grid():
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="closed and watertight"):
        voxelize_kidney(mesh)
    with pytest.raises(ValueError, match="increase the voxel size"):
        voxelize_kidney(trimesh.creation.box(extents=[1000, 1000, 1000]), 1)


@pytest.mark.parametrize("invalid", [0, -1, 0.5, 11, float("nan"), float("inf"), True, "bad"])
def test_invalid_voxel_resolution_is_rejected(invalid):
    with pytest.raises(ValueError, match="between 1 and 10"):
        validate_voxel_size(invalid)


@pytest.mark.parametrize("depth,angle", [(0, 60), (25, 60), (15, 0), (15, 61), (np.nan, 60), (15, np.inf), (True, 60)])
def test_invalid_beam_parameters_are_rejected(depth, angle):
    with pytest.raises(ValueError):
        validate_fan_parameters(depth, angle)


def test_fan_depth_starts_at_lens_and_rays_follow_finite_aperture():
    beam = make_fan_geometry([7, 11, 65.5], curvature_radius_mm=52)
    source = np.asarray(beam["source_arc"])
    origin = np.asarray(beam["virtual_origin"])
    np.testing.assert_allclose(source[32], [7, 11, 65.5])
    np.testing.assert_allclose(np.linalg.norm(source - origin, axis=1), 52)
    assert not any(np.allclose(vertex, origin) for vertex in beam["vertices"])
    rays = np.asarray(beam["scan_lines"])
    np.testing.assert_allclose(np.linalg.norm(rays[:, 1] - rays[:, 0], axis=1), 150)
    np.testing.assert_allclose(rays[4], [[7, 11, 65.5], [7, 11, 215.5]])
    assert beam["outline"][0] == beam["outline"][-1]
    assert validate_fan_parameters(24, 60) == (24, 60)
    assert validate_voxel_size(1) == 1
    assert validate_voxel_size(10) == 10


def test_plane_depth_angle_and_lens_exclude_outside_cubes():
    mask = hits([[0, 0, 50], [0, 1.01, 50], [0, 0, 110], [60, 0, 20], [0, 0, -5]])
    np.testing.assert_array_equal(mask, [True, False, False, False, False])


def test_rendered_depth_and_angle_boundaries_count_only_positive_overlap():
    # At z=100 the polygonal outer arc passes through the cube; at z=101
    # the cube only touches its outermost vertex and must remain uncolored.
    np.testing.assert_array_equal(hits([[0, 0, 100], [0, 0, 101]]), [True, False])
    edge_x = 40 / math.sqrt(3)  # The 30-degree boundary at z=30.
    np.testing.assert_array_equal(hits([[edge_x, 0, 30], [edge_x + 2, 0, 30]]), [True, False])
    narrow = fan(angle_deg=20)
    assert hits([[20, 0, 50]])[0]
    assert not hits([[20, 0, 50]], beam=narrow)[0]
    assert not hits([[0, 0, 50]], beam=fan(depth_cm=4))[0]


def test_point_and_edge_tangency_are_excluded_but_shared_cube_face_counts():
    edge_rotation = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    assert not hits([[0, 0, 0]], kidney_pose=pose([0, math.sqrt(2), 50], edge_rotation))[0]
    x = np.array([1, -1, 0]) / math.sqrt(2)
    y = np.array([1, 1, 1]) / math.sqrt(3)
    point_rotation = np.stack([x, y, np.cross(x, y)])
    assert not hits([[0, 0, 0]], kidney_pose=pose([0, math.sqrt(3), 50], point_rotation))[0]
    np.testing.assert_array_equal(hits([[0, -1, 50], [0, 1, 50]]), [True, True])


def test_oblique_body_section_and_common_rigid_transform_preserve_hits():
    x = np.array([1, -1, 0]) / math.sqrt(2)
    y = np.array([1, 1, 1]) / math.sqrt(3)
    assert hits([[0, 0, 0]], kidney_pose=pose([0, 0, 50], np.stack([x, y, np.cross(x, y)])))[0]
    centers = [[0, 0, 50], [4, 0, 50], [80, 0, 20]]
    world_pose = pose([31, -19, 72], Rotation.from_euler("xyz", [25, -35, 45], degrees=True).as_matrix())
    np.testing.assert_array_equal(hits(centers, kidney_pose=world_pose, probe_pose=world_pose), hits(centers))


def test_independent_probe_and_kidney_poses_match_relative_transform():
    kidney_pose = pose([2, 3, 5], Rotation.from_euler("xyz", [0.15, 0.1, -0.25]).as_matrix())
    probe_pose = pose([1, -2, 1], Rotation.from_euler("xyz", [-0.1, 0.05, 0.2]).as_matrix())
    centers = [[0, 0, 50], [2, 2, 50]]
    actual = hits(centers, kidney_pose=kidney_pose, probe_pose=probe_pose)
    expected = hits(centers, kidney_pose=np.linalg.inv(probe_pose) @ kidney_pose)
    np.testing.assert_array_equal(actual, expected)


def test_probe_rotation_moves_plane_and_direction_together():
    centers = [[0, -50, 0], [0, 0, 50]]
    np.testing.assert_array_equal(hits(centers), [False, True])
    rotated = pose(rotation=Rotation.from_euler("x", 90, degrees=True).as_matrix())
    np.testing.assert_array_equal(hits(centers, probe_pose=rotated), [True, False])


def test_history_loss_reacquisition_and_reset_leave_old_frames_unchanged():
    scan = KidneyScan(VoxelKidney([[0, 0, 50], [0, 4, 50], [100, 0, 50]], 2), fan())
    first = scan.update(np.eye(4))
    second = scan.update(pose([0, 4, 0]))
    lost = scan.update(None)
    assert first.tracked and first.hit_count == first.history_count == 1
    assert second.hit_count == 1 and second.history_count == 2
    assert lost.hit_count == 0 and lost.history_count == 2 and not lost.tracked
    assert lost.coverage_percent == pytest.approx(200 / 3)
    reacquired = scan.update(np.eye(4))
    assert reacquired.tracked and reacquired.hit_count == 1 and reacquired.history_count == 2
    reset = scan.reset_history()
    assert reset.tracked and reset.hit_count == 1 and reset.history_count == 0
    assert reset.coverage_percent == 0
    assert first.history_count == 1 and second.history_count == 2  # Stable snapshots.
    with pytest.raises(ValueError, match="read-only"):
        reset.current_mask[0] = False
    resumed = scan.update(np.eye(4))
    assert resumed.history_count == 1
    np.testing.assert_array_equal(resumed.current_mask, reset.current_mask)
    scan.update(None)
    assert not scan.reset_history().tracked


@pytest.mark.parametrize(
    "invalid", [np.zeros((3, 3)), np.eye(4) * 2, np.full((4, 4), np.nan), pose(rotation=np.eye(3) * 2)]
)
def test_invalid_probe_pose_is_rejected_without_changing_history(invalid):
    scan = KidneyScan(VoxelKidney([[0, 0, 50]], 2), fan())
    good = scan.update(np.eye(4))
    with pytest.raises(ValueError, match="rigid"):
        scan.update(invalid)
    assert scan.frame is good


def test_empty_voxel_set_has_zero_coverage():
    scan = KidneyScan(VoxelKidney([], 2), fan())
    frame = scan.update(np.eye(4))
    assert frame.tracked and frame.hit_count == frame.history_count == frame.coverage_percent == 0


def test_bundled_kidney_loading_and_rotation_about_its_bounding_box_center():
    mesh = trimesh.load_mesh(KIDNEY, process=False)
    center = mesh.bounds.mean(axis=0)
    position, angles = [15, -22, 401], [35, -20, 10]
    scan = KidneyScan.load(KIDNEY, PROBE, position_mm=position, rotation_deg=angles)
    rx, ry, rz = [Rotation.from_euler(axis, angle, degrees=True).as_matrix() for axis, angle in zip("xyz", angles)]
    np.testing.assert_allclose(scan.kidney_rotation, rz @ ry @ rx)
    np.testing.assert_allclose(scan.kidney_rotation @ center + scan.kidney_translation, position)
    assert 3000 < scan.voxels.voxel_count < 8000
    np.testing.assert_allclose(np.asarray(scan.fan["source_arc"])[32], [0.110413, 0, 65.5], atol=1e-5)
    assert 52 < scan.fan["curvature_radius_mm"] < 53
    assert not scan.frame.tracked and scan.frame.history_count == 0
    scan.set_placement(scale=1.4)
    np.testing.assert_allclose(scan.source_center, center)
    np.testing.assert_allclose(scan.kidney_rotation @ (1.4 * center) + scan.kidney_translation, position)


@pytest.mark.parametrize(
    "keyword,value", [("position_mm", [0, np.nan, 0]), ("rotation_deg", [0, np.inf, 0]), ("rotation_deg", [0])]
)
def test_invalid_kidney_position_or_rotation_is_rejected(keyword, value):
    with pytest.raises(ValueError, match="three finite"):
        KidneyScan.load(KIDNEY, PROBE, **{keyword: value})


def test_probe_without_named_lens_reports_filename_and_missing_material(tmp_path):
    path = tmp_path / "unknown_probe.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match="unknown_probe.obj.*convex_acoustic_lens"):
        load_convex_scan_profile(path)
