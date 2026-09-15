"""End-to-end checks that images, tracking state and the Rerun scene agree."""

import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from probe_tracking.camera import CameraIntrinsics
from probe_tracking.cli import main
from probe_tracking.geometry import MarkerGeometry
from probe_tracking.synthetic import demo_pose, render_markers
from probe_tracking.tracking import Tracker

GEOMETRY_PATH = Path(__file__).resolve().parents[1] / "assets/Vscan_marker_attachment/marker_geometry_42mm.json"


def sample_frames():
    geometry = MarkerGeometry.load(GEOMETRY_PATH)
    intrinsics = CameraIntrinsics.approximate(960, 540)
    frames, ground_truth = [], []
    for timestamp in [0.0, 0.2, 5.5]:
        rotation, translation = demo_pose(timestamp)
        frames.append(render_markers(geometry, intrinsics.camera_matrix, rotation, translation, 960, 540))
        ground_truth.append((rotation, translation))
    return geometry, intrinsics, frames, ground_truth


def test_render_detect_estimate_loss_and_reacquisition():
    geometry, intrinsics, frames, ground_truth = sample_frames()
    tracker = Tracker(geometry, intrinsics.camera_matrix, intrinsics.dist_coeffs)
    recovered = []
    for frame, (rotation, translation) in zip(frames, ground_truth):
        detections = tracker.detect(frame)
        assert len(detections) >= 2
        pose = tracker.estimate(detections)
        assert pose is not None
        assert len(pose.marker_ids) >= 2
        # Rasterization and detector subpixel refinement add real corner error.
        assert np.linalg.norm(pose.translation - translation) < 5.0
        assert Rotation.from_matrix(rotation.T @ pose.rotation).magnitude() < np.deg2rad(1.5)
        assert pose.reprojection_error_px < 1.0
        recovered.append(pose.translation)
        blank_detections = tracker.detect(np.full_like(frame, 238))
        assert blank_detections == {}
        assert tracker.estimate(blank_detections) is None
    assert np.linalg.norm(recovered[0] - recovered[-1]) > 20.0


class FakeVideo:
    """Supply uncompressed deterministic images while exercising the actual CLI."""

    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        frame = next(self.frames, None)
        return (False, None) if frame is None else (True, frame.copy())

    def get(self, prop):
        assert prop == cv2.CAP_PROP_FPS
        return 30.0

    def release(self):
        self.released = True


def test_cli_rerun_clears_lost_geometry_and_restores_it_without_stale_smoothing(tmp_path, monkeypatch):
    geometry, intrinsics, images, _ = sample_frames()
    blank = np.full_like(images[0], 238)
    video = FakeVideo([images[0], images[1], blank, blank, images[2]])
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: video)
    calibration_path = tmp_path / "camera.json"
    intrinsics.save(calibration_path)
    # A minimal genuine OBJ still exercises the mesh loader and Rerun serializer.
    model_path = tmp_path / "probe.obj"
    model_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    recording_path = tmp_path / "loss-and-reacquisition.rrd"
    diagnostics_path = tmp_path / "tracking.jsonl"

    logged = []
    current_frame = [None]
    real_log, real_set_time = rr.log, rr.set_time

    def capture_log(entity_path, *entities, **kwargs):
        logged.append((current_frame[0], entity_path, entities, kwargs))
        return real_log(entity_path, *entities, **kwargs)

    def capture_time(timeline, **kwargs):
        if timeline == "frame":
            current_frame[0] = kwargs["sequence"]
        return real_set_time(timeline, **kwargs)

    monkeypatch.setattr(rr, "log", capture_log)
    monkeypatch.setattr(rr, "set_time", capture_time)
    assert (
        main(
            [
                "track",
                "--video",
                str(tmp_path / "deterministic-source.avi"),
                "--geometry",
                str(GEOMETRY_PATH),
                "--model",
                str(model_path),
                "--calibration",
                str(calibration_path),
                "--no-viewer",
                "--no-realtime",
                "--save",
                str(recording_path),
                "--smooth",
                "0.05",
                "--diagnostics",
                str(diagnostics_path),
            ]
        )
        == 0
    )
    assert video.released
    assert recording_path.is_file() and recording_path.stat().st_size > 1000
    records = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    assert records[0]["type"] == "session"
    assert records[0]["image_size"] == [960, 540]
    assert records[0]["corners_coordinate_space"] == "undistorted_pixels"
    assert records[0]["dist_coeffs_used_for_pose"] == [0.0] * 5
    frames = records[1:-1]
    assert [frame["reason"] for frame in frames] == ["tracking", "tracking", "no_markers", "no_markers", "tracking"]
    assert frames[0]["corners_px"]
    assert frames[0]["shortest_edge_px"]
    assert frames[2]["corners_px"] == {}
    assert frames[2]["reprojection_error_px"] is None
    assert records[-1] == {"type": "summary", "total_frames": 5, "counts": {"tracking": 3, "no_markers": 2}}

    clears = [
        (frame, path, entity)
        for frame, path, entities, _ in logged
        for entity in entities
        if isinstance(entity, rr.Clear)
    ]
    assert [(frame, path) for frame, path, _ in clears] == [
        (2, "world/tracked"),
        (2, "world/trajectory"),
    ]
    assert all(entity.is_recursive.as_arrow_array().to_pylist() == [True] for _, _, entity in clears)

    # Static geometry would survive a temporal Clear. Require temporal mesh logs
    # and an identical geometry restoration on the first reacquired frame.
    mesh_events = [
        (frame, path, kwargs)
        for frame, path, entities, kwargs in logged
        if path.startswith("world/tracked/") and any(isinstance(entity, rr.Mesh3D) for entity in entities)
    ]
    first_paths = {path for frame, path, _ in mesh_events if frame == 0}
    restored_paths = {path for frame, path, _ in mesh_events if frame == 4}
    assert first_paths == restored_paths
    assert "world/tracked/probe/part_00" in first_paths
    assert {f"world/tracked/markers/id_{marker_id}" for marker_id in geometry.markers} <= first_paths
    assert all(frame in (0, 4) and not kwargs.get("static", False) for frame, _, kwargs in mesh_events)

    transforms = [
        (frame, entity)
        for frame, path, entities, _ in logged
        if path == "world/tracked"
        for entity in entities
        if isinstance(entity, rr.Transform3D)
    ]
    assert [frame for frame, _ in transforms] == [0, 1, 4]
    fresh_tracker = Tracker(geometry, intrinsics.camera_matrix, intrinsics.dist_coeffs)
    fresh_pose = fresh_tracker.estimate(fresh_tracker.detect(images[2]))
    assert fresh_pose is not None
    reacquired_translation = transforms[-1][1].translation.as_arrow_array().to_pylist()[0]
    np.testing.assert_allclose(reacquired_translation, fresh_pose.translation, atol=1e-3)
    # The trail must restart after loss rather than connect through missing data.
    assert [
        frame
        for frame, path, entities, _ in logged
        if path == "world/trajectory" and any(isinstance(entity, rr.LineStrips3D) for entity in entities)
    ] == [1]
