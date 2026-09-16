"""Exercise real detection, scan geometry and temporal Rerun serialization together."""

from threading import Event, get_ident, local

import cv2
import numpy as np
import pytest
import rerun as rr

from probe_tracking import synthetic
from probe_tracking.camera import CameraIntrinsics
from probe_tracking.cli import DEFAULT_GEOMETRY, main
from probe_tracking.geometry import MarkerGeometry
from probe_tracking.kidney_scan import KidneyScan
from probe_tracking.tracking import Tracker


class FakeVideo:
    """Supply deterministic, uncompressed frames through the camera interface."""

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


@pytest.fixture(scope="module")
def sample_frames():
    geometry = MarkerGeometry.load(DEFAULT_GEOMETRY)
    intrinsics = CameraIntrinsics.approximate(960, 540)
    frames = []
    for timestamp in [0.0, 0.2, 5.5]:
        rotation, translation = synthetic.demo_pose(timestamp)
        frames.append(
            synthetic.render_markers(geometry, intrinsics.camera_matrix, rotation, translation, 960, 540)
        )
    return geometry, intrinsics, frames


@pytest.fixture
def recorded(monkeypatch):
    """Observe calls while retaining the real scan implementation and serializer."""
    result = {"logs": [], "updates": [], "resets": [], "log_threads": []}
    # Rerun's timeline is thread-local. A shared frame counter would conceal a
    # reset accidentally logged on the UI thread without the worker's time.
    timeline_state = local()
    real_log, real_set_time = rr.log, rr.set_time
    real_update, real_reset = KidneyScan.update, KidneyScan.reset_history

    def log(path, *entities, **kwargs):
        result["logs"].extend((getattr(timeline_state, "frame", None), path, entity, kwargs) for entity in entities)
        result["log_threads"].extend((path, get_ident()) for _ in entities)
        return real_log(path, *entities, **kwargs)

    def set_time(timeline, **kwargs):
        if timeline == "frame":
            timeline_state.frame = kwargs["sequence"]
        return real_set_time(timeline, **kwargs)

    def update(scan, camera_from_probe):
        scan_frame = real_update(scan, camera_from_probe)
        result["updates"].append(
            (None if camera_from_probe is None else camera_from_probe.copy(), scan_frame)
        )
        return scan_frame

    def reset(scan):
        scan_frame = real_reset(scan)
        result["resets"].append(scan_frame)
        return scan_frame

    monkeypatch.setattr(rr, "log", log)
    monkeypatch.setattr(rr, "set_time", set_time)
    monkeypatch.setattr(KidneyScan, "update", update)
    monkeypatch.setattr(KidneyScan, "reset_history", reset)
    return result


def entities(recorded, path, kind):
    return [
        (frame, entity, kwargs)
        for frame, entity_path, entity, kwargs in recorded["logs"]
        if path == entity_path and isinstance(entity, kind)
    ]


def matrix(transform):
    result = np.eye(4)
    # Rerun's matrix component stores columns consecutively.
    result[:3, :3] = np.asarray(transform.mat3x3.as_arrow_array().to_pylist()[0]).reshape(3, 3).T
    result[:3, 3] = transform.translation.as_arrow_array().to_pylist()[0]
    return result


def run_video(tmp_path, monkeypatch, intrinsics, frames, *options, video=None):
    video = FakeVideo(frames) if video is None else video
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: video)
    calibration = tmp_path / "camera.json"
    intrinsics.save(calibration)
    recording = tmp_path / "kidney.rrd"
    assert main(
        [
            "track", "--kidney", "--video", str(tmp_path / "source.avi"),
            "--calibration", str(calibration), "--no-viewer", "--no-realtime", "--save", str(recording),
            *options,
        ]
    ) == 0
    assert video.released
    assert recording.is_file() and recording.stat().st_size > 1000


def test_kidney_tracks_displayed_pose_and_survives_loss(tmp_path, monkeypatch, sample_frames, recorded):
    geometry, intrinsics, images = sample_frames
    blank = np.full_like(images[0], 238)
    run_video(
        tmp_path, monkeypatch, intrinsics,
        [images[0], images[1], blank, blank, images[2]], "--smooth", "0.05",
    )
    updates = recorded["updates"]
    scans = [scan for _, scan in updates]
    assert len(scans) == 5
    assert [scan.tracked for scan in scans] == [True, True, False, False, True]
    assert scans[0].hit_count > 0
    assert scans[2].hit_count == scans[3].hit_count == 0
    np.testing.assert_array_equal(scans[1].history_mask, scans[2].history_mask)
    np.testing.assert_array_equal(scans[1].history_mask, scans[3].history_mask)
    assert scans[4].history_count >= scans[3].history_count > 0

    transforms = entities(recorded, "world/tracked", rr.Transform3D)
    assert [frame for frame, _, _ in transforms] == [0, 1, 4]
    for frame, transform, _ in transforms:
        np.testing.assert_allclose(
            updates[frame][0], matrix(transform) @ geometry.cube_from_probe, atol=3e-5,
        )
    fresh = Tracker(geometry, intrinsics.camera_matrix, intrinsics.dist_coeffs)
    second_pose = fresh.estimate(fresh.detect(images[1]))
    fresh.reset()
    recovered_pose = fresh.estimate(fresh.detect(images[2]))
    assert second_pose is not None and recovered_pose is not None
    # A heavy smoothing setting must affect both the visible probe and the scan.
    assert np.linalg.norm(matrix(transforms[1][1])[:3, 3] - second_pose.translation) > 1
    # The first pose after LOST starts fresh rather than blending with old data.
    np.testing.assert_allclose(updates[4][0], recovered_pose.matrix @ geometry.cube_from_probe, atol=1e-6)

    kidney_transforms = entities(recorded, "world/kidney", rr.Transform3D)
    assert [frame for frame, _, _ in kidney_transforms] == list(range(5))
    assert all(not kwargs.get("static", False) for _, _, kwargs in kidney_transforms)
    for _, transform, _ in kidney_transforms[1:]:
        np.testing.assert_allclose(matrix(transform), matrix(kidney_transforms[0][1]))
    clears = [(frame, path) for frame, path, entity, _ in recorded["logs"] if isinstance(entity, rr.Clear)]
    assert clears == [(2, "world/tracked"), (2, "world/trajectory")]
    for name in ("current", "history", "context"):
        boxes = entities(recorded, f"world/kidney/voxels/{name}", rr.Boxes3D)
        assert [frame for frame, _, _ in boxes] == list(range(5))
        assert all(not kwargs.get("static", False) for _, _, kwargs in boxes)
        if name == "current":
            assert [len(box.centers.as_arrow_array()) for _, box, _ in boxes] == [scan.hit_count for scan in scans]
        if name == "history":
            for frame in (2, 3):
                assert len(boxes[frame][1].centers.as_arrow_array()) == scans[frame].history_count
    statuses = entities(recorded, "kidney_status", rr.TextDocument)
    assert [frame for frame, _, _ in statuses] == list(range(5))
    assert all("UNAVAILABLE" in statuses[frame][1].text.as_arrow_array().to_pylist()[0] for frame in (2, 3))
    fan_meshes = entities(recorded, "world/tracked/probe/ultrasound/fill", rr.Mesh3D)
    assert [frame for frame, _, _ in fan_meshes] == [0, 4]
    assert all(not kwargs.get("static", False) for _, _, kwargs in fan_meshes)
    assert [frame for frame, _, _ in entities(recorded, "world/trajectory", rr.LineStrips3D)] == [1]


@pytest.mark.parametrize("with_controls", [False, True])
def test_preview_reset_keeps_current_hit_and_resumes_history(
    tmp_path, monkeypatch, sample_frames, recorded, with_controls,
):
    from probe_tracking import kidney_controls

    _, intrinsics, images = sample_frames
    ui_thread = get_ident()
    controls = []

    class PassiveControls:
        def __init__(self, **kwargs):
            self.closed = False
            controls.append(self)

        def pump(self):
            assert get_ident() == ui_thread

        def close(self):
            assert get_ident() == ui_thread
            self.closed = True

    monkeypatch.setattr(kidney_controls, "KidneyControls", PassiveControls)
    keys = iter([ord("r"), -1, ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: shown.append(name))
    destroyed = []
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: destroyed.append(True))
    options = ["--preview", "--kidney-controls" if with_controls else "--no-kidney-controls"]
    run_video(tmp_path, monkeypatch, intrinsics, [images[0]] * 3, *options)
    scans = [scan for _, scan in recorded["updates"]]
    assert len(scans) == 3 and len(recorded["resets"]) == 1
    reset = recorded["resets"][0]
    assert reset.tracked and reset.hit_count > 0 and reset.history_count == 0
    np.testing.assert_array_equal(reset.current_mask, scans[0].current_mask)
    assert scans[0].history_count > 0  # The earlier recorded snapshot stays immutable.
    assert scans[1].history_count > 0 and scans[2].history_count >= scans[1].history_count
    current = entities(recorded, "world/kidney/voxels/current", rr.Boxes3D)
    history = entities(recorded, "world/kidney/voxels/history", rr.Boxes3D)
    assert [frame for frame, _, _ in current] == [0, 0, 1, 2]
    assert len(current[1][1].centers.as_arrow_array()) == scans[0].hit_count
    assert len(history[1][1].centers.as_arrow_array()) == 0
    assert [frame for frame, _, _ in entities(recorded, "world/tracked", rr.Transform3D)] == [0, 1, 2]
    assert [frame for frame, _, _ in entities(recorded, "world/trajectory", rr.LineStrips3D)] == [1, 2]
    assert len(shown) == 4 and destroyed == [True]
    log_threads = {thread for path, thread in recorded["log_threads"] if path == "world/kidney/voxels/current"}
    assert len(log_threads) == 1
    assert (ui_thread not in log_threads) == with_controls
    assert all(window.closed for window in controls)


def test_headless_demo_samples_hit_and_miss_over_ten_seconds(tmp_path, monkeypatch, recorded):
    original_pose = synthetic.demo_pose
    times = iter([0.0, 1.0, 5.0, 9.97])
    monkeypatch.setattr(synthetic, "demo_pose", lambda timestamp: original_pose(next(times)))
    recording = tmp_path / "demo.rrd"
    assert main(
        [
            "demo", "--kidney", "--smooth", "1", "--max-frames", "4", "--no-viewer", "--no-realtime",
            "--save", str(recording),
        ]
    ) == 0
    scans = [scan for _, scan in recorded["updates"]]
    assert len(scans) == 4 and all(scan.tracked for scan in scans)
    assert scans[0].hit_count > 0 and scans[-1].hit_count == 0
    assert scans[-1].history_count > 0
    assert scans[-1].coverage_percent > 0
    assert recording.stat().st_size > 1000


@pytest.mark.parametrize(
    "options",
    [
        ["--kidney-position-mm", "nan", "0", "0"],
        ["--kidney-rotation-deg", "0", "inf", "0"],
        ["--voxel-mm", "nan"],
        ["--voxel-mm", "0"],
        ["--beam-depth-cm", "inf"],
        ["--beam-depth-cm", "0"],
        ["--beam-angle-deg", "nan"],
        ["--beam-angle-deg", "61"],
        ["--kidney-scale", "nan"],
        ["--kidney-scale", "0"],
        ["--kidney-scale", "5.1"],
    ],
)
def test_invalid_scan_parameters_fail_before_camera_access(tmp_path, monkeypatch, capsys, options):
    def unexpected_camera(source):
        pytest.fail("Invalid kidney options must be rejected before opening the camera")

    monkeypatch.setattr(cv2, "VideoCapture", unexpected_camera)
    recording = tmp_path / "invalid.rrd"
    assert main(["track", "--kidney", "--no-viewer", "--save", str(recording), *options]) == 1
    assert "Error:" in capsys.readouterr().err
    assert not recording.exists()


def test_live_controls_move_scale_reset_and_stop_on_frame_boundaries(
    tmp_path, monkeypatch, sample_frames, recorded,
):
    from probe_tracking import kidney_controls
    from probe_tracking.viewer import RerunViewer

    _, intrinsics, images = sample_frames
    blank = np.full_like(images[0], 238)
    windows = []
    completed_frames = [0]
    real_log_frame = RerunViewer.log_frame

    def log_frame(viewer, *args, **kwargs):
        real_log_frame(viewer, *args, **kwargs)
        completed_frames[0] += 1

    monkeypatch.setattr(RerunViewer, "log_frame", log_frame)

    class ScriptedControls:
        def __init__(self, *, position_mm, scale, on_change, on_reset_history, on_stop):
            self.initial = tuple(position_mm)
            assert scale == 1
            self.change, self.reset, self.stop = on_change, on_reset_history, on_stop
            self.pumps = 0
            self.handled_frames = set()
            self.closed = False
            windows.append(self)

        def pump(self):
            self.pumps += 1
            completed = completed_frames[0]
            if completed in self.handled_frames:
                return
            self.handled_frames.add(completed)
            if completed == 1:
                self.change((1000, -180, 400), 2)
            elif completed == 2:
                self.change(self.initial, 0.5)
            elif completed == 3:
                self.change(self.initial, 1)
                self.reset()
            elif completed == 5:
                self.stop()

        def close(self):
            self.closed = True

    monkeypatch.setattr(kidney_controls, "KidneyControls", ScriptedControls)
    run_video(
        tmp_path, monkeypatch, intrinsics,
        [images[0], images[0], blank, images[0], images[1], images[1]], "--kidney-controls",
    )
    assert len(windows) == 1 and windows[0].closed and windows[0].pumps >= 6
    scans = [frame for _, frame in recorded["updates"]]
    assert len(scans) == 5 and scans[0].hit_count > 0
    assert scans[1].hit_count == 0 and scans[1].history_count == scans[0].history_count
    assert not scans[2].tracked and scans[2].history_count == scans[0].history_count
    assert recorded["resets"][0].hit_count > 0 and recorded["resets"][0].history_count == 0
    assert scans[4].history_count > 0
    transforms = entities(recorded, "world/kidney", rr.Transform3D)
    assert [frame for frame, _, _ in transforms] == list(range(5))
    matrices = [matrix(transform) for _, transform, _ in transforms]
    np.testing.assert_allclose(
        [np.linalg.det(value[:3, :3]) for value in matrices], [1, 8, 0.125, 1, 1], atol=1e-6,
    )
    assert all(not kwargs.get("static", False) for _, _, kwargs in transforms)
    # Both the current hit and accumulated colors follow the same transformed parent.
    assert len(entities(recorded, "world/kidney/voxels/history", rr.Boxes3D)) == 5


def test_controls_are_not_opened_for_headless_recording(tmp_path, monkeypatch):
    from probe_tracking import kidney_controls

    def unexpected_controls(**kwargs):
        pytest.fail("Headless recording must not initialize a GUI")

    monkeypatch.setattr(kidney_controls, "KidneyControls", unexpected_controls)
    assert main([
        "demo", "--kidney", "--max-frames", "1", "--no-viewer", "--no-realtime",
        "--save", str(tmp_path / "headless.rrd"),
    ]) == 0


@pytest.mark.parametrize("stop_during_capture", [False, True])
def test_controls_accept_repeated_moves_while_capture_is_blocked(
    tmp_path, monkeypatch, sample_frames, recorded, stop_during_capture,
):
    from probe_tracking import kidney_controls

    _, intrinsics, images = sample_frames
    capturing, release_capture = Event(), Event()
    ui_thread = get_ident()
    windows, placements = [], []
    real_set_placement = KidneyScan.set_placement

    def set_placement(scan, *, position_mm=None, scale=None):
        placements.append((tuple(position_mm), scale))
        return real_set_placement(scan, position_mm=position_mm, scale=scale)

    monkeypatch.setattr(KidneyScan, "set_placement", set_placement)

    class BlockedVideo(FakeVideo):
        def __init__(self):
            super().__init__([images[0]] * 2)
            self.read_count = 0

        def read(self):
            self.read_count += 1
            if self.read_count == 2:
                assert get_ident() != ui_thread
                capturing.set()
                # The frame can only finish once the main thread has accepted
                # every simulated click. No speed-dependent sleep is needed.
                if not release_capture.wait(timeout=5):
                    raise RuntimeError("Controls were not serviced during a blocked camera read")
            return super().read()

    class RepeatedControls:
        def __init__(self, *, position_mm, scale, on_change, on_reset_history, on_stop):
            self.position = list(position_mm)
            self.scale = scale
            self.change, self.stop = on_change, on_stop
            self.click_count = 0
            self.closed = False
            windows.append(self)

        def pump(self):
            assert get_ident() == ui_thread
            if not capturing.is_set() or self.click_count == 20:
                return
            self.position[0] += 5
            self.click_count += 1
            self.change(tuple(self.position), self.scale)
            if self.click_count == 20:
                if stop_during_capture:
                    self.stop()
                release_capture.set()

        def close(self):
            assert get_ident() == ui_thread
            self.closed = True

    video = BlockedVideo()
    monkeypatch.setattr(kidney_controls, "KidneyControls", RepeatedControls)
    run_video(
        tmp_path, monkeypatch, intrinsics, [], "--kidney-controls", "--max-frames", "2", video=video,
    )
    assert windows[0].closed and windows[0].click_count == 20
    transforms = entities(recorded, "world/kidney", rr.Transform3D)
    assert len(transforms) == (1 if stop_during_capture else 2)
    if stop_during_capture:
        assert placements == []
    else:
        # All twenty 5 mm clicks reach the scene through one final placement;
        # obsolete intermediate positions do not form a geometry-work backlog.
        assert placements == [((100, -180, 400), 1)]
        np.testing.assert_allclose(
            matrix(transforms[1][1])[:3, 3] - matrix(transforms[0][1])[:3, 3], [100, 0, 0], atol=1e-5,
        )
