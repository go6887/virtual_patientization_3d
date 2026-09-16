"""Scan colors and loss/recovery must serialize as replayable temporal data."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from scipy.spatial.transform import Rotation

from probe_tracking import viewer
from probe_tracking.camera import CameraIntrinsics
from probe_tracking.geometry import MarkerGeometry
from probe_tracking.kidney_scan import KidneyScan, ScanFrame
from probe_tracking.kidney_view import CUBE_SIZE_FACTOR, KIDNEY_ENTITY, KidneyView
from probe_tracking.kidney_voxels import VoxelKidney
from probe_tracking.tracking import Pose
from probe_tracking.ultrasound_geometry import make_fan_geometry


def values(archetype, component):
    return getattr(archetype, component).as_arrow_array().to_pylist()


def transform_parts(archetype):
    # Rerun stores matrix columns consecutively.
    linear = np.asarray(values(archetype, "mat3x3")[0]).reshape(3, 3).T
    translation = np.asarray(values(archetype, "translation")[0])
    return linear, translation


@pytest.fixture
def scan():
    centers = np.array(list(product([-3.0, 0.0, 3.0], repeat=3))) + [0, 0, 70]
    return KidneyScan(VoxelKidney(centers, 3), make_fan_geometry([0, 0, 0]))


@pytest.fixture
def events(monkeypatch):
    """Capture semantic contents while retaining actual strict SDK serialization."""
    captured = []
    original_log = rr.log

    def log(path, archetype, *extra, **kwargs):
        captured.append((path, archetype, kwargs))
        return original_log(path, archetype, *extra, **kwargs)

    monkeypatch.setattr(rr, "log", log)
    return captured


def voxel_batches(events):
    return {path.rsplit("/", 1)[1]: archetype for path, archetype, _ in events if "/voxels/" in path}


def test_colors_are_exclusive_boundary_is_hollow_and_replacements_are_temporal(scan, events):
    previous_strict = rr.strict_mode()
    rr.set_strict_mode(True)
    try:
        with rr.RecordingStream("kidney-temporal-colors") as recording:
            memory = recording.memory_recording()
            scene = KidneyView(scan)
            rr.set_time("frame", sequence=0)
            scene.log_frame(scan.frame)
            batches = voxel_batches(events)
            assert len(values(batches["context"], "centers")) == 26
            assert [0, 0, 70] not in values(batches["context"], "centers")

            rr.set_time("frame", sequence=1)
            scene.log_frame(scan.update(np.eye(4)))
            batches = voxel_batches(events)
            assert len(values(batches["current"], "centers")) == 9
            assert values(batches["current"], "colors") == [0x55DC8CFF] * 9
            assert values(batches["current"], "fill_mode") == [3]
            assert values(batches["history"], "centers") == []
            assert len(values(batches["context"], "centers")) == 18
            assert values(batches["context"], "fill_mode") == [1]

            moved = np.eye(4)
            moved[1, 3] = 3
            rr.set_time("frame", sequence=2)
            scene.log_frame(scan.update(moved))
            batches = voxel_batches(events)
            assert values(batches["history"], "colors") == [0xFFE146FF] * 9
            selected_sets = [
                set(map(tuple, values(batches[name], "centers"))) for name in ("context", "history", "current")
            ]
            assert all(not (first & second) for i, first in enumerate(selected_sets) for second in selected_sets[i + 1:])
            for boxes in batches.values():
                count = len(values(boxes, "centers"))
                np.testing.assert_allclose(values(boxes, "half_sizes"), np.full((count, 3), 1.5 * CUBE_SIZE_FACTOR))

            rr.set_time("frame", sequence=3)
            scene.log_frame(scan.reset_history())
            batches = voxel_batches(events)
            assert len(values(batches["current"], "centers")) == 9
            for component in ("centers", "half_sizes", "colors"):
                assert values(batches["history"], component) == []
            assert "**0 / 27**" in values(events[-1][1], "text")[0]
            assert all(not kwargs.get("static", False) for path, _, kwargs in events if "/voxels/" in path)
            recording.flush()
            assert memory.num_msgs() > 0
            assert len(memory.drain_as_bytes()) > 100
    finally:
        rr.set_strict_mode(previous_strict)


def test_viewer_loss_retains_movable_kidney_and_restores_exact_temporal_fan(scan, events, monkeypatch, tmp_path):
    mesh = rr.Mesh3D(vertex_positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], triangle_indices=[[0, 1, 2]])
    monkeypatch.setattr(viewer, "load_meshes", lambda path: [("part_00", mesh)])
    monkeypatch.setattr(viewer, "marker_mesh", lambda geometry, marker_id: mesh)
    geometry_path = Path(__file__).resolve().parents[1] / "assets/Vscan_marker_attachment/marker_geometry_42mm.json"
    geometry = MarkerGeometry.load(geometry_path)
    scene = viewer.RerunViewer(
        geometry, tmp_path / "probe.obj", CameraIntrinsics.approximate(640, 360), spawn=False, scan=scan,
    )
    recording = rr.get_global_data_recording()
    memory = recording.memory_recording()
    pose = Pose(np.eye(3), np.array([0, 0, 410]), (0,), 0.1)
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    frame = scan.update(np.eye(4))
    try:
        # A static placement would shadow all subsequent manual movements.
        assert not any(path == KIDNEY_ENTITY for path, _, _ in events)
        events.clear()
        scene.log_frame(0, 0, image, {}, pose, scan_frame=frame)
        visible = {path: archetype for path, archetype, _ in events if "/ultrasound/" in path}
        assert len(visible) == 6
        assert all(not kwargs.get("static", False) for path, _, kwargs in events if "/ultrasound/" in path)
        np.testing.assert_allclose(
            values(visible["world/tracked/probe/ultrasound/fill"], "vertex_positions"), scan.fan["vertices"], rtol=1e-6,
        )
        probe_transform = next(archetype for path, archetype, _ in events if path == "world/tracked/probe")
        np.testing.assert_allclose(values(probe_transform, "translation"), [geometry.cube_from_probe[:3, 3]])

        events.clear()
        original_position = scan.kidney_position.copy()
        scan.set_placement(position_mm=[50, -20, 100], scale=1.5)
        lost = scan.update(None)
        scene.log_frame(1, 0.1, image, {}, None, scan_frame=lost)
        cleared = {path for path, archetype, _ in events if isinstance(archetype, rr.Clear)}
        assert cleared == {"world/tracked", "world/trajectory"}
        batches = voxel_batches(events)
        assert values(batches["current"], "centers") == []
        assert len(values(batches["history"], "centers")) == frame.hit_count
        assert "LOST - UNAVAILABLE" in values(events[-1][1], "text")[0]
        assert "Center: 50, -20, 100 mm · Size: **150%**" in values(events[-1][1], "text")[0]
        kidney_transform = next(archetype for path, archetype, _ in events if path == KIDNEY_ENTITY)
        linear, translation = transform_parts(kidney_transform)
        np.testing.assert_allclose(linear, np.eye(3) * 1.5)
        np.testing.assert_allclose(translation, scan.kidney_translation)
        # Placement changes carry history along with the original voxel identities.
        np.testing.assert_array_equal(
            values(batches["history"], "centers"), scan.voxels.centers[frame.history_mask],
        )
        assert not any("/ultrasound/" in path for path, _, _ in events)

        events.clear()
        scan.set_placement(position_mm=original_position, scale=1.0)
        scene.log_frame(2, 0.2, image, {}, pose, scan_frame=scan.update(np.eye(4)))
        restored = {path: archetype for path, archetype, _ in events if "/ultrasound/" in path}
        assert restored.keys() == visible.keys()
        for suffix, component in (("fill", "vertex_positions"), ("depth_labels", "labels")):
            path = f"world/tracked/probe/ultrasound/{suffix}"
            assert values(restored[path], component) == values(visible[path], component)

        events.clear()
        scene.log_scan_frame(scan.reset_history())
        scene.log_image(image)
        assert not any(path.startswith("world/tracked") or path == "world/trajectory" for path, _, _ in events)
        assert len(values(voxel_batches(events)["current"], "centers")) == frame.hit_count
        # An ordinary tracked frame with no current hits is a MISS, even after history reset.
        events.clear()
        scene.log_scan_frame(ScanFrame(np.zeros(27, bool), np.zeros(27, bool), True))
        assert "MISS - no intersection" in values(events[-1][1], "text")[0]
        recording.flush()
        assert memory.num_msgs() > 0
        assert len(memory.drain_as_bytes()) > 100
    finally:
        scene.close()


def test_placement_scale_and_history_are_recorded_together_for_replay(events, monkeypatch, tmp_path):
    """Each timeline row restores an independent center, size, and voxel history."""
    source_center = np.array([5.0, -8.0, 11.0])
    centers = np.array(list(product([-3.0, 0.0, 3.0], repeat=3))) + source_center
    rotation = Rotation.from_euler("xyz", [20, 35, -10], degrees=True).as_matrix()
    scan = KidneyScan(
        VoxelKidney(centers, 3), make_fan_geometry([0, 0, 0]),
        kidney_rotation=rotation, source_center=source_center,
    )
    timeline_rows = []
    selected_frame = [None]
    original_time, original_log = rr.set_time, rr.log

    def set_time(timeline, **kwargs):
        if timeline == "frame":
            selected_frame[0] = kwargs["sequence"]
        return original_time(timeline, **kwargs)

    def log(path, archetype, *extra, **kwargs):
        if path == KIDNEY_ENTITY:
            timeline_rows.append((selected_frame[0], archetype, kwargs))
        return original_log(path, archetype, *extra, **kwargs)

    monkeypatch.setattr(rr, "set_time", set_time)
    monkeypatch.setattr(rr, "log", log)
    placements = [([10, 20, 90], 1.0), ([50, -25, 120], 2.0), ([-30, 10, 75], 0.5)]
    recording_path = tmp_path / "placements.rrd"
    previous_strict = rr.strict_mode()
    rr.set_strict_mode(True)
    try:
        with rr.RecordingStream("kidney-placement-history") as recording:
            recording.save(recording_path)
            scene = KidneyView(scan)
            for frame_index, (position, scale) in enumerate(placements):
                rr.set_time("frame", sequence=frame_index)
                rr.set_time("elapsed", duration=frame_index * 0.1)
                scan.set_placement(position_mm=position, scale=scale)
                scene.log_frame(scan.frame)
                status = values(events[-1][1], "text")[0]
                assert f"Size: **{scale * 100:.0f}%**" in status

                # Parent scale changes physical cubes, while source-local geometry
                # remains unchanged and can safely retain accumulated voxel IDs.
                boxes = voxel_batches(events)["context"]
                local_half_sizes = np.asarray(values(boxes, "half_sizes"))
                linear, translation = transform_parts(timeline_rows[-1][1])
                np.testing.assert_allclose(linear @ source_center + translation, position, atol=1e-5)
                np.testing.assert_allclose(
                    np.linalg.norm(linear, axis=0) * local_half_sizes,
                    np.full_like(local_half_sizes, 1.5 * CUBE_SIZE_FACTOR * scale), rtol=1e-6,
                )
            recording.flush()
        assert recording_path.stat().st_size > 100
    finally:
        rr.set_strict_mode(previous_strict)

    assert [frame for frame, _, _ in timeline_rows] == [0, 1, 2]
    assert all(not kwargs.get("static", False) for _, _, kwargs in timeline_rows)
    # Earlier serialized archetypes remain unchanged after later placement edits.
    # With no static transform, Rerun's latest-at query restores these time rows.
    for (_, transform, _), (position, scale) in zip(timeline_rows, placements, strict=True):
        linear, translation = transform_parts(transform)
        np.testing.assert_allclose(linear, rotation * scale, rtol=1e-6)
        np.testing.assert_allclose(linear @ source_center + translation, position, atol=1e-5)
