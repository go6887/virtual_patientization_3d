import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from probe_tracking.calibration import calibrate_from_corners, create_chessboard, run_calibration
from probe_tracking.camera import CameraIntrinsics


def test_intrinsics_round_trip_and_resize(tmp_path):
    camera = CameraIntrinsics(
        np.array([[800, 0, 640], [0, 810, 360], [0, 0, 1]]),
        np.array([0.1, -0.02, 0.001, 0.002, 0]),
        (1280, 720),
        rms_error_px=0.25,
    )
    destination = tmp_path / "camera.json"
    camera.save(destination)
    restored = CameraIntrinsics.load(destination)
    np.testing.assert_array_equal(restored.camera_matrix, camera.camera_matrix)
    np.testing.assert_array_equal(restored.dist_coeffs, camera.dist_coeffs)
    assert restored.image_size == (1280, 720)
    assert restored.calibrated and restored.rms_error_px == 0.25
    small = restored.for_size(640, 360)
    np.testing.assert_allclose(small.camera_matrix, [[400, 0, 320], [0, 405, 180], [0, 0, 1]])
    np.testing.assert_array_equal(small.dist_coeffs, camera.dist_coeffs)
    with pytest.raises(ValueError, match="aspect ratio"):
        restored.for_size(640, 480)


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera_matrix", [[0, 0, 320], [0, 500, 240], [0, 0, 1]]),
        ("camera_matrix", [[500, 0, float("nan")], [0, 500, 240], [0, 0, 1]]),
        ("camera_matrix", [[500, 1, 320], [0, 500, 240], [0, 0, 1]]),
        ("camera_matrix", [[500, 0, 320], [0, 500, 240], [0, 0, 2]]),
        ("dist_coeffs", [0, 0, 0, float("inf"), 0]),
        ("dist_coeffs", ["0", "0", "0", "0", "0"]),
        ("dist_coeffs", [0, 0]),
        ("image_size", [640.5, 480]),
        ("image_size", [True, 480]),
        ("image_size", [640, 0]),
        ("calibrated", "true"),
        ("rms_error_px", -1),
        ("rms_error_px", float("nan")),
    ],
)
def test_invalid_intrinsics_are_rejected(field, value):
    data = {
        "camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        "dist_coeffs": [0, 0, 0, 0, 0],
        "image_size": (640, 480),
    }
    data[field] = value
    with pytest.raises(ValueError):
        CameraIntrinsics(**data)


def test_bad_camera_json_has_readable_error(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"image_size": [640, 480]}))
    with pytest.raises(ValueError, match="missing camera_matrix"):
        CameraIntrinsics.load(path)


def test_approximation_is_explicitly_uncalibrated():
    camera = CameraIntrinsics.approximate(640, 480, 90)
    assert not camera.calibrated
    assert camera.rms_error_px is None
    np.testing.assert_allclose(camera.camera_matrix, [[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    for fov in (0, 180, -10, float("inf"), True):
        with pytest.raises(ValueError):
            CameraIntrinsics.approximate(640, 480, fov)


def test_synthetic_chessboards_recover_intrinsics():
    columns, rows, square = 9, 6, 20.0
    objects = np.zeros((columns * rows, 3), np.float32)
    objects[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square
    true_matrix = np.array([[900.0, 0, 640], [0, 910.0, 360], [0, 0, 1]])
    distortion = np.array([0.04, -0.02, 0.001, -0.001, 0.01])
    views = []
    rng = np.random.default_rng(42)
    for _ in range(20):
        rvec = rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.3])
        tvec = rng.uniform([-140, -100, 500], [-20, 0, 850])
        points, _ = cv2.projectPoints(objects, rvec, tvec, true_matrix, distortion)
        views.append(points)
    solved = calibrate_from_corners(views, (1280, 720), columns, rows, square)
    np.testing.assert_allclose(solved.camera_matrix, true_matrix, atol=0.03)
    np.testing.assert_allclose(solved.dist_coeffs, distortion, atol=0.01)
    assert solved.calibrated
    assert solved.rms_error_px < 0.001


def test_invalid_calibration_views_are_rejected():
    with pytest.raises(ValueError, match="three"):
        calibrate_from_corners([], (640, 480), 9, 6, 20)
    with pytest.raises(ValueError, match="distinct"):
        calibrate_from_corners([np.zeros((54, 2))] * 3, (640, 480), 9, 6, 20)
    with pytest.raises(ValueError, match="finite"):
        calibrate_from_corners([np.full((54, 2), np.nan)] * 3, (640, 480), 9, 6, 20)


def test_chessboard_svg_has_physical_square_size_and_margin(tmp_path):
    path = create_chessboard(tmp_path / "chessboard.svg", 9, 6, 20)
    root = ET.parse(path).getroot()
    assert root.attrib["width"] == "240mm"
    assert root.attrib["height"] == "180mm"
    squares = [child for child in root if child.attrib.get("fill") == "black"]
    assert len(squares) == 35  # 10 x 7 squares around 9 x 6 inner corners.
    assert all(square.attrib["width"] == "20" and square.attrib["height"] == "20" for square in squares)
    assert min(float(square.attrib["x"]) for square in squares) == 20
    assert min(float(square.attrib["y"]) for square in squares) == 20


@pytest.mark.parametrize("opened", [True, False])
def test_calibration_releases_camera_on_capture_failure(monkeypatch, tmp_path, opened):
    class FakeCapture:
        released = False

        def isOpened(self):
            return opened

        def set(self, *_):
            return True

        def read(self):
            return False, None

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_: None)
    with pytest.raises(RuntimeError, match="camera|Camera"):
        run_calibration(0, 640, 480, 9, 6, 20, tmp_path / "camera.json")
    assert capture.released
    assert not (tmp_path / "camera.json").exists()
