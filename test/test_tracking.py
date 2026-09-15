"""Numerical regression tests for cube/probe frame conventions and pose rejection."""

import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from probe_tracking.geometry import MarkerGeometry
from probe_tracking.tracking import Pose, PoseSmoother, Tracker

GEOMETRY_PATH = Path(__file__).resolve().parents[1] / "assets/Vscan_marker_attachment/marker_geometry_42mm.json"
CAMERA = np.array([[850.0, 0.0, 640.0], [0.0, 840.0, 360.0], [0.0, 0.0, 1.0]])
DISTORTION = np.array([0.05, -0.02, 0.001, -0.002, 0.003])


def project(geometry, rotation, translation, marker_ids):
    return {
        marker_id: cv2.projectPoints(
            geometry.markers[marker_id],
            cv2.Rodrigues(rotation)[0],
            translation,
            CAMERA,
            DISTORTION,
        )[0].reshape(4, 2)
        for marker_id in marker_ids
    }


def camera_looking_at_cube(camera_center):
    center = np.asarray(camera_center, dtype=float)
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    return rotation, -rotation @ center


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.geometry = MarkerGeometry.load(GEOMETRY_PATH)
        self.tracker = Tracker(self.geometry, CAMERA, DISTORTION)

    def assert_pose_close(self, pose, rotation, translation, position_tolerance=1e-4, angle_tolerance=1e-5):
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.translation, translation, atol=position_tolerance)
        self.assertLess(Rotation.from_matrix(pose.rotation.T @ rotation).magnitude(), angle_tolerance)
        self.assertAlmostEqual(np.linalg.det(pose.rotation), 1.0)
        self.assertLess(pose.reprojection_error_px, 1e-5)

    def test_nominal_transform_is_inverted_for_probe_mesh(self):
        geometry = self.geometry
        np.testing.assert_allclose(geometry.cube_from_probe[:3, 3], [0, 8, -49])
        # JSON says cube origin lands at probe [0, 49, -8]. Inverse must undo it.
        np.testing.assert_allclose(geometry.cube_from_probe @ [0, 49, -8, 1], [0, 0, 0, 1])
        self.assertEqual(geometry.marker_size_mm, 32)
        self.assertEqual(geometry.dictionary_name, "DICT_4X4_50")
        self.assertEqual(set(geometry.markers), {0, 1, 2, 3, 4})

    def test_all_five_single_faces_with_distortion_and_rotation(self):
        tilt = Rotation.from_euler("xyz", [0.17, -0.23, 0.31]).as_matrix()
        translation = np.array([12.0, -7.0, 550.0])
        for marker_id, corners in self.geometry.markers.items():
            with self.subTest(marker_id=marker_id):
                right = corners[1] - corners[0]
                right /= np.linalg.norm(right)
                up = corners[0] - corners[3]
                up /= np.linalg.norm(up)
                basis = np.column_stack([right, up, np.cross(right, up)])
                rotation = tilt @ np.diag([1, -1, -1]) @ basis.T
                self.tracker.reset()
                pose = self.tracker.estimate(project(self.geometry, rotation, translation, [marker_id]))
                self.assert_pose_close(pose, rotation, translation)
                self.assertEqual(pose.marker_ids, (marker_id,))

    def test_three_faces_recover_camera_from_cube(self):
        rotation, translation = camera_looking_at_cube([360, 280, 490])
        pose = self.tracker.estimate(project(self.geometry, rotation, translation, [0, 1, 2]))
        self.assert_pose_close(pose, rotation, translation)
        self.assertEqual(pose.marker_ids, (0, 1, 2))
        homogeneous_point = np.array([10, -5, 7, 1])
        np.testing.assert_allclose(
            (pose.matrix @ homogeneous_point)[:3], rotation @ homogeneous_point[:3] + translation
        )

    def test_directly_front_facing_single_marker(self):
        rotation = np.diag([1.0, -1.0, -1.0])
        translation = np.array([0.0, 0.0, 500.0])
        pose = self.tracker.estimate(project(self.geometry, rotation, translation, [0]))
        self.assert_pose_close(pose, rotation, translation, angle_tolerance=1e-4)

    def test_wrong_id_face_is_excluded_as_a_whole(self):
        rotation, translation = camera_looking_at_cube([360, 280, 490])
        detections = project(self.geometry, rotation, translation, [0, 1, 2])
        detections[4] = detections.pop(2)  # A +Y face wrongly decoded as -Y.
        pose = self.tracker.estimate(detections)
        self.assert_pose_close(pose, rotation, translation)
        self.assertEqual(pose.marker_ids, (0, 1))

    def test_bad_corner_excludes_face_and_refines_remaining_faces(self):
        rotation, translation = camera_looking_at_cube([360, 280, 490])
        detections = project(self.geometry, rotation, translation, [0, 1, 2])
        detections[2][0] += [12, 8]
        pose = self.tracker.estimate(detections)
        self.assert_pose_close(pose, rotation, translation)
        self.assertEqual(pose.marker_ids, (0, 1))

    def test_noisy_faces_are_stable_and_error_is_in_pixels(self):
        rotation, translation = camera_looking_at_cube([360, 280, 490])
        rng = np.random.default_rng(123)
        detections = project(self.geometry, rotation, translation, [0, 1, 2])
        detections = {key: pixels + rng.normal(0, 0.15, pixels.shape) for key, pixels in detections.items()}
        pose = self.tracker.estimate(detections)
        self.assertIsNotNone(pose)
        self.assertEqual(pose.marker_ids, (0, 1, 2))
        self.assertLess(np.linalg.norm(pose.translation - translation), 2.5)
        self.assertLess(Rotation.from_matrix(pose.rotation.T @ rotation).magnitude(), 0.02)
        self.assertLess(pose.reprojection_error_px, 0.4)

    def test_unknown_invalid_and_missing_detections_never_return_stale_pose(self):
        rotation, translation = camera_looking_at_cube([360, 280, 490])
        valid = project(self.geometry, rotation, translation, [0, 1])
        self.assertIsNotNone(self.tracker.estimate(valid))
        self.assertIsNone(self.tracker.estimate({}))
        self.assertIsNone(self.tracker.estimate({49: valid[0]}))
        self.assertIsNone(self.tracker.estimate({0: np.full((4, 2), np.nan)}))
        self.assertIsNone(self.tracker.estimate({0: np.zeros((4, 2))}))
        self.assertIsNone(self.tracker.estimate({0: np.zeros((2, 2))}))
        self.assertIsNotNone(self.tracker.estimate(valid))

    def test_back_of_printed_face_is_rejected(self):
        rotation = np.eye(3)
        translation = np.array([20.0, 10.0, 500.0])
        self.assertIsNone(self.tracker.estimate(project(self.geometry, rotation, translation, [0])))

    def test_detect_ignores_unknown_and_duplicate_ids(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        canvas = np.full((300, 1000), 255, dtype=np.uint8)
        for index, marker_id in enumerate([0, 1, 1, 49]):
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
            x = 30 + 240 * index
            canvas[70:230, x : x + 160] = marker
        detected = self.tracker.detect(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
        self.assertEqual(set(detected), {0})
        np.testing.assert_allclose(detected[0], [[30, 70], [189, 70], [189, 229], [30, 229]], atol=1)

    def test_inconsistent_single_face_tie_is_rejected_without_history(self):
        rotation, translation = camera_looking_at_cube([360, 280, 490])
        detections = project(self.geometry, rotation, translation, [0, 1])
        detections[1] += [180, 60]
        self.assertIsNone(self.tracker.estimate(detections))


class SmootherTests(unittest.TestCase):
    def test_rotation_uses_shortest_arc_across_180_degrees(self):
        smoother = PoseSmoother(0.5)
        first = Pose(Rotation.from_euler("z", 179, degrees=True).as_matrix(), np.array([0.0, 0, 400]), (0,), 0.1)
        second = Pose(Rotation.from_euler("z", -179, degrees=True).as_matrix(), np.array([10.0, 2, 420]), (0, 1), 0.7)
        self.assertIs(smoother.update(first), first)
        result = smoother.update(second)
        np.testing.assert_allclose(result.rotation, Rotation.from_euler("z", 180, degrees=True).as_matrix(), atol=1e-10)
        np.testing.assert_allclose(result.translation, [5, 1, 410])
        self.assertEqual(result.reprojection_error_px, 0.7)
        self.assertEqual(result.marker_ids, (0, 1))
        smoother.reset()
        self.assertIs(smoother.update(second), second)


if __name__ == "__main__":
    unittest.main()
