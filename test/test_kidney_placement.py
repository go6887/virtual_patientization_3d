"""Placement changes move the same organ voxels, including their scan history."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from probe_tracking.kidney_scan import KidneyScan
from probe_tracking.kidney_voxels import VoxelKidney, intersect_voxels_with_fan, validate_kidney_scale
from probe_tracking.ultrasound_geometry import make_fan_geometry


def fan():
    return make_fan_geometry([0, 0, 0], curvature_radius_mm=10, depth_cm=10)


def test_scale_and_translation_preserve_mesh_pivot_under_rotation_and_source_voxels():
    voxels = VoxelKidney([[5, 8, 12], [9, 12, 18]], 2)
    rotation = Rotation.from_euler("xyz", [30, 50, -20], degrees=True).as_matrix()
    source_center = np.array([7.3, 9.1, 14.7])  # Mesh center can differ from voxel center.
    scan = KidneyScan(
        voxels,
        fan(),
        kidney_translation=[12, -25, 45],
        kidney_rotation=rotation,
        source_center=source_center,
    )
    original_position = scan.kidney_position.copy()
    original_centers = voxels.centers.copy()
    scan.set_placement(scale=2.5)
    np.testing.assert_allclose(rotation @ (2.5 * source_center) + scan.kidney_translation, original_position)
    np.testing.assert_allclose(scan.kidney_position, original_position)
    scan.set_placement(position_mm=[25, 60, 300], scale=0.25)
    np.testing.assert_allclose(rotation @ (0.25 * source_center) + scan.kidney_translation, [25, 60, 300])
    np.testing.assert_array_equal(scan.kidney_rotation, rotation)
    np.testing.assert_array_equal(voxels.centers, original_centers)
    assert scan.voxels is voxels and voxels.pitch_mm == 2
    assert scan.kidney_scale == 0.25
    for array in [scan.kidney_position, scan.kidney_translation, scan.source_center]:
        assert not array.flags.writeable


def test_constructor_keeps_translation_semantics_with_scale_and_default_voxel_center():
    voxels = VoxelKidney([[2, 4, 6], [6, 8, 10]], 2)
    scan = KidneyScan(voxels, fan(), kidney_translation=[10, 20, 30], scale=2)
    np.testing.assert_array_equal(scan.source_center, [4, 6, 8])
    np.testing.assert_array_equal(scan.kidney_translation, [10, 20, 30])
    np.testing.assert_array_equal(scan.kidney_position, [18, 32, 46])
    empty = KidneyScan(VoxelKidney([], 2), fan(), kidney_translation=[1, 2, 3])
    np.testing.assert_array_equal(empty.source_center, [0, 0, 0])
    np.testing.assert_array_equal(empty.kidney_position, [1, 2, 3])


def test_scale_changes_physical_cube_size_and_center_in_intersection():
    voxels = VoxelKidney([[0, 0, 0]], 2)

    def hit(scale):
        return intersect_voxels_with_fan(
            voxels, [0, 1.5, 50], np.eye(3), [0, 0, 0], np.eye(3), fan(), kidney_scale=scale
        ).hit_mask[0]

    assert not hit(1)
    assert hit(2)  # Half-size grows from 1 to 2 mm and crosses the beam plane.
    # A scaled source center at depth 150 mm exceeds this fan's 100 mm depth.
    result = intersect_voxels_with_fan(
        VoxelKidney([[0, 0, 50]], 2),
        [0, 0, 0],
        np.eye(3),
        [0, 0, 0],
        np.eye(3),
        fan(),
        kidney_scale=3,
    )
    assert not result.hit_mask[0]


def test_scaled_rotated_cube_edge_tangency_remains_excluded():
    rotation = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    result = intersect_voxels_with_fan(
        VoxelKidney([[0, 0, 0]], 2),
        [0, 2 * math.sqrt(2), 50],
        rotation,
        [0, 0, 0],
        np.eye(3),
        fan(),
        kidney_scale=2,
    )
    assert result.hit_count == 0


def test_movement_recomputes_current_on_next_frame_and_keeps_voxel_history():
    scan = KidneyScan(VoxelKidney([[0, 0, 50], [0, 4, 50]], 2), fan())
    first = scan.update(np.eye(4))
    np.testing.assert_array_equal(first.current_mask, [True, False])
    scan.set_placement(position_mm=[0, -2, 50])
    assert scan.frame is first
    second = scan.update(np.eye(4))
    np.testing.assert_array_equal(second.current_mask, [False, True])
    np.testing.assert_array_equal(second.history_mask, [True, True])
    np.testing.assert_array_equal(first.history_mask, [True, False])


@pytest.mark.parametrize(
    "placement",
    [
        {"position_mm": [1, 2, np.nan], "scale": 2},
        {"position_mm": [1, 2], "scale": 2},
        {"position_mm": [1, 2, 3], "scale": 0},
        {"position_mm": [1, 2, 3], "scale": 6},
        {"position_mm": [1, 2, 3], "scale": np.inf},
        {"position_mm": [1, 2, 3], "scale": True},
    ],
)
def test_invalid_placement_is_atomic(placement):
    scan = KidneyScan(VoxelKidney([[0, 0, 50]], 2), fan())
    frame = scan.update(np.eye(4))
    position, translation, scale = scan.kidney_position.copy(), scan.kidney_translation.copy(), scan.kidney_scale
    with pytest.raises(ValueError):
        scan.set_placement(**placement)
    np.testing.assert_array_equal(scan.kidney_position, position)
    np.testing.assert_array_equal(scan.kidney_translation, translation)
    assert scan.kidney_scale == scale and scan.frame is frame


@pytest.mark.parametrize("invalid", [0, -1, 0.09, 5.01, np.inf, np.nan, True, np.bool_(False), "bad"])
def test_invalid_scale_rejected_by_core_and_intersection(invalid):
    with pytest.raises(ValueError, match="between 0.1 and 5.0"):
        validate_kidney_scale(invalid)
    with pytest.raises(ValueError, match="between 0.1 and 5.0"):
        KidneyScan(VoxelKidney([], 2), fan(), scale=invalid)
    with pytest.raises(ValueError, match="between 0.1 and 5.0"):
        intersect_voxels_with_fan(
            VoxelKidney([], 2), [0, 0, 0], np.eye(3), [0, 0, 0], np.eye(3), fan(), kidney_scale=invalid
        )


def test_supported_scale_endpoints_are_inclusive():
    assert validate_kidney_scale(0.1) == 0.1
    assert validate_kidney_scale(5) == 5


def test_placement_while_lost_preserves_history_and_reacquisition_uses_new_transform():
    scan = KidneyScan(VoxelKidney([[0, 0, 50]], 2), fan())
    assert scan.update(np.eye(4)).hit_count == 1
    lost = scan.update(None)
    scan.set_placement(position_mm=[0, 10, 50], scale=2)
    assert scan.frame is lost and lost.history_count == 1 and lost.hit_count == 0
    still_lost = scan.update(None)
    assert not still_lost.tracked and still_lost.history_count == 1 and still_lost.hit_count == 0
    assert scan.update(np.eye(4)).hit_count == 0
    probe = np.eye(4)
    probe[:3, 3] = [0, 10, 0]
    acquired = scan.update(probe)
    assert acquired.tracked and acquired.hit_count == acquired.history_count == 1

