"""Loss reasons describe existing acceptance rules without changing them."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from probe_tracking.camera import CameraIntrinsics
from probe_tracking.geometry import MarkerGeometry
from probe_tracking.synthetic import demo_pose
from probe_tracking.tracking import Tracker


@pytest.fixture
def tracker():
    geometry_path = Path(__file__).resolve().parents[1] / "assets/Vscan_marker_attachment/marker_geometry_42mm.json"
    geometry = MarkerGeometry.load(geometry_path)
    intrinsics = CameraIntrinsics.approximate(1280, 720)
    return Tracker(geometry, intrinsics.camera_matrix, intrinsics.dist_coeffs)


def projected_faces(tracker, rotation=None, translation=None):
    if rotation is None:
        rotation, translation = demo_pose(0.0)
    return {
        marker_id: cv2.projectPoints(
            tracker.geometry.markers[marker_id],
            cv2.Rodrigues(rotation)[0],
            translation,
            tracker.camera_matrix,
            tracker.dist_coeffs,
        )[0].reshape(4, 2)
        for marker_id in [0, 1]
    }


def test_reason_initialization_validation_and_reset(tracker):
    assert tracker.last_reason == "not_started"
    assert tracker.detected_ids == ()
    assert tracker.duplicate_ids == ()
    assert tracker.rejected_candidate_count == 0
    tracker.reset()
    assert tracker.last_reason == "not_started"
    assert tracker.estimate({}) is None
    assert tracker.last_reason == "no_markers"
    tracker.reset()
    assert tracker.last_reason == "no_markers"
    assert tracker.estimate({0: np.full((4, 2), np.nan)}) is None
    assert tracker.last_reason == "invalid_corners"
    tracker.reset()
    assert tracker.last_reason == "invalid_corners"
    # A syntactically valid projection of the printed face's back is rejected.
    backside = projected_faces(tracker, np.eye(3), np.array([20.0, 10.0, 500.0]))
    assert tracker.estimate({0: backside[0]}) is None
    assert tracker.last_reason == "pose_rejected"
    tracker.reset()
    assert tracker.last_reason == "pose_rejected"


def test_inconsistent_face_reason_explains_loss_and_single_face_recovery(tracker):
    valid = projected_faces(tracker)
    conflicting = {marker_id: corners.copy() for marker_id, corners in valid.items()}
    conflicting[1] += [40.0, 0.0]
    assert tracker.estimate(valid) is not None
    assert tracker.last_reason == "tracking"
    # History can select the good face while both detected faces disagree.
    assert tracker.estimate(conflicting) is not None
    assert tracker.last_reason == "tracking"
    assert tracker.estimate({}) is None
    assert tracker.last_reason == "no_markers"
    for _ in range(2):
        assert tracker.estimate(conflicting) is None
        assert tracker.last_reason == "inconsistent_faces"
        tracker.reset()  # The CLI does this after every lost frame.
        assert tracker.last_reason == "inconsistent_faces"
    assert tracker.estimate({0: valid[0]}) is not None
    assert tracker.last_reason == "tracking"
    tracker.reset()
    assert tracker.last_reason == "tracking"
    # Explicit reset still clears the prior despite preserving its explanation.
    assert tracker.estimate(conflicting) is None
    assert tracker.last_reason == "inconsistent_faces"


def test_detect_metadata_keeps_unknown_duplicates_and_refreshes_on_empty_frame(tracker):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.full((300, 1000), 255, dtype=np.uint8)
    for index, marker_id in enumerate([0, 1, 1, 49]):
        x = 30 + 240 * index
        canvas[70:230, x : x + 160] = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
    _, _, rejected = tracker.detector.detectMarkers(canvas)
    detections = tracker.detect(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    assert set(detections) == {0}
    assert tracker.detected_ids == (0, 1, 1, 49)
    assert tracker.duplicate_ids == (1,)
    assert tracker.rejected_candidate_count == len(rejected)
    assert tracker.rejected_candidate_count > 0
    assert tracker.detect(np.full_like(canvas, 255)) == {}
    assert tracker.detected_ids == ()
    assert tracker.duplicate_ids == ()
    assert tracker.rejected_candidate_count == 0
