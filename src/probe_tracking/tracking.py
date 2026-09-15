"""ArUco cube tracking in the OpenCV camera frame (right/down/forward)."""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import MarkerGeometry


@dataclass(frozen=True)
class Pose:
    """Camera-from-cube rigid transform; translation is in millimetres."""

    rotation: np.ndarray
    translation: np.ndarray
    marker_ids: tuple[int, ...]
    reprojection_error_px: float

    @property
    def matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.translation
        return transform


class Tracker:
    def __init__(
        self,
        geometry: MarkerGeometry,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None,
        max_reprojection_error_px: float = 4.0,
    ) -> None:
        self.geometry = geometry
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        if (
            self.camera_matrix.shape != (3, 3)
            or not np.isfinite(self.camera_matrix).all()
            or min(self.camera_matrix[0, 0], self.camera_matrix[1, 1]) <= 0
            or not np.allclose(self.camera_matrix[2], [0, 0, 1])
        ):
            raise ValueError("Camera matrix must be a finite 3x3 pinhole intrinsic matrix.")
        if self.dist_coeffs.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.dist_coeffs).all():
            raise ValueError("Distortion coefficients must contain 4, 5, 8, 12, or 14 finite values.")
        if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0:
            raise ValueError("Reprojection threshold must be positive and finite.")
        self.max_reprojection_error_px = max_reprojection_error_px
        dictionary_id = getattr(cv2.aruco, geometry.dictionary_name, None)
        if not isinstance(dictionary_id, int):
            raise ValueError(f"Unknown OpenCV ArUco dictionary: {geometry.dictionary_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if any(marker_id >= len(dictionary.bytesList) for marker_id in geometry.markers):
            raise ValueError("Geometry contains a marker ID outside its ArUco dictionary.")
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._previous: Pose | None = None
        self.last_reason = "not_started"
        self.detected_ids: tuple[int, ...] = ()
        self.duplicate_ids: tuple[int, ...] = ()
        self.rejected_candidate_count = 0

    def reset(self) -> None:
        """Clear the pose prior while preserving the latest diagnostic reason."""
        self._previous = None

    def detect(self, frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        corners, ids, rejected = self.detector.detectMarkers(gray)
        self.rejected_candidate_count = len(rejected)
        if ids is None:
            self.detected_ids = ()
            self.duplicate_ids = ()
            return {}
        flat_ids = ids.reshape(-1)
        self.detected_ids = tuple(sorted(int(marker_id) for marker_id in flat_ids))
        # Multiple copies of the same ID have no unique geometry correspondence.
        unique, counts = np.unique(flat_ids, return_counts=True)
        self.duplicate_ids = tuple(int(marker_id) for marker_id, count in zip(unique, counts) if count > 1)
        accepted = {int(marker_id) for marker_id, count in zip(unique, counts) if count == 1}
        return {
            int(marker_id): np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            for marker_id, marker_corners in zip(flat_ids, corners)
            if int(marker_id) in accepted and int(marker_id) in self.geometry.markers
        }

    def _points(self, ids: tuple[int, ...], detections: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate([self.geometry.markers[marker_id] for marker_id in ids]),
            np.concatenate([detections[marker_id] for marker_id in ids]),
        )

    def _single_face_candidates(self, marker_id: int, corners: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        world = self.geometry.markers[marker_id]
        center = world.mean(axis=0)
        right = world[1] - world[0]
        right /= np.linalg.norm(right)
        up = world[0] - world[3]
        up /= np.linalg.norm(up)
        normal = np.cross(right, up)
        normal /= np.linalg.norm(normal)
        up = np.cross(normal, right)
        basis = np.column_stack([right, up, normal])
        local = np.ascontiguousarray((world - center) @ basis)
        # General IPPE supports calibrated face corner updates as well as squares.
        try:
            result = cv2.solvePnPGeneric(local, corners, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return []
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            rotation = cv2.Rodrigues(rvec)[0] @ basis.T
            translation = tvec.reshape(3) - rotation @ center
            candidates.append((rotation, translation))
        return candidates

    def _face_errors(
        self, rotation: np.ndarray, translation: np.ndarray, detections: dict[int, np.ndarray]
    ) -> dict[int, float]:
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return {}
        rvec = cv2.Rodrigues(rotation)[0]
        errors = {}
        for marker_id, observed in detections.items():
            points = self.geometry.markers[marker_id]
            camera_points = points @ rotation.T + translation
            if np.any(camera_points[:, 2] <= 1e-6):
                continue
            camera_normal = rotation @ self.geometry.normals[marker_id]
            # Printed outward face must point back towards the camera center.
            if np.dot(camera_normal, camera_points.mean(axis=0)) >= -1e-6:
                continue
            projected = cv2.projectPoints(points, rvec, translation, self.camera_matrix, self.dist_coeffs)[0].reshape(
                4, 2
            )
            corner_errors = np.linalg.norm(projected - observed, axis=1)
            rms = float(np.sqrt(np.mean(corner_errors**2)))
            if rms <= self.max_reprojection_error_px and np.max(corner_errors) <= 2 * self.max_reprojection_error_px:
                errors[marker_id] = rms
        return errors

    def _refine(
        self, rotation: np.ndarray, translation: np.ndarray, ids: tuple[int, ...], detections: dict[int, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        points, pixels = self._points(ids, detections)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                points,
                pixels,
                self.camera_matrix,
                self.dist_coeffs,
                cv2.Rodrigues(rotation)[0],
                translation.reshape(3, 1).copy(),
            )
            return cv2.Rodrigues(rvec)[0], tvec.reshape(3)
        except cv2.error:
            return rotation, translation

    def estimate(self, detections: dict[int, np.ndarray]) -> Pose | None:
        """Estimate this frame only. Missing/invalid frames clear the pose prior.

        Each entire face is an inlier or outlier, so a wrong decoded face cannot
        contribute a handful of corners to an otherwise plausible cube pose.
        """
        valid = {}
        for marker_id, corners in detections.items():
            points = np.asarray(corners, dtype=np.float64)
            if marker_id not in self.geometry.markers or points.shape != (4, 2) or not np.isfinite(points).all():
                continue
            polygon = points.astype(np.float32)
            if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 16:
                continue
            valid[marker_id] = np.ascontiguousarray(points)
        if not valid:
            self.last_reason = "invalid_corners" if detections else "no_markers"
            self.reset()
            return None

        candidates = []
        for marker_id, corners in valid.items():
            candidates.extend(self._single_face_candidates(marker_id, corners))
        # At most five attachment faces: all pairs are inexpensive and avoid a
        # random RANSAC result when an entire marker is incorrectly identified.
        for ids in combinations(sorted(valid), 2):
            points, pixels = self._points(ids, valid)
            if np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
                continue
            try:
                result = cv2.solvePnPGeneric(
                    points, pixels, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
                )
            except cv2.error:
                continue
            candidates.extend((cv2.Rodrigues(rvec)[0], tvec.reshape(3)) for rvec, tvec in zip(result[1], result[2]))

        scored: list[Pose] = []
        for rotation, translation in candidates:
            errors = self._face_errors(rotation, translation, valid)
            if not errors:
                continue
            rotation, translation = self._refine(rotation, translation, tuple(sorted(errors)), valid)
            errors = self._face_errors(rotation, translation, valid)
            if not errors:
                continue
            # Refine once more if another face joined the consensus.
            rotation, translation = self._refine(rotation, translation, tuple(sorted(errors)), valid)
            errors = self._face_errors(rotation, translation, valid)
            if errors:
                scored.append(
                    Pose(
                        rotation,
                        translation,
                        tuple(sorted(errors)),
                        float(np.sqrt(np.mean(np.square(list(errors.values()))))),
                    )
                )
        if not scored:
            self.last_reason = "pose_rejected"
            self.reset()
            return None

        consensus = max(len(pose.marker_ids) for pose in scored)
        eligible = [pose for pose in scored if len(pose.marker_ids) == consensus]
        best_error = min(pose.reprojection_error_px for pose in eligible)
        # History only breaks near-equal reprojection solutions, never overrides
        # a larger geometric consensus or rescues a failed current frame.
        eligible = [pose for pose in eligible if pose.reprojection_error_px <= best_error + 0.25]
        if self._previous is not None:
            previous = self._previous

            def continuity(pose: Pose) -> float:
                angle = Rotation.from_matrix(previous.rotation.T @ pose.rotation).magnitude()
                distance = np.linalg.norm(pose.translation - previous.translation)
                return float(angle + distance / max(np.linalg.norm(previous.translation), 1.0))

            best = min(eligible, key=continuity)
        else:
            best = min(eligible, key=lambda pose: pose.reprojection_error_px)
        if len(valid) > 1 and consensus == 1:
            # Two inconsistent individually plausible faces cannot be resolved
            # without history. A previous nearby pose may resolve this tie.
            if self._previous is None or continuity(best) > 0.25:
                self.last_reason = "inconsistent_faces"
                self.reset()
                return None
        self._previous = best
        self.last_reason = "tracking"
        return best


class PoseSmoother:
    """Exponential smoothing with geodesic interpolation of rigid rotations."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Smoothing alpha must be in (0, 1].")
        self.alpha = float(alpha)
        self._previous: Pose | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, pose: Pose) -> Pose:
        if self._previous is None:
            self._previous = pose
            return pose
        previous = self._previous
        delta = Rotation.from_matrix(previous.rotation.T @ pose.rotation).as_rotvec()
        rotation = previous.rotation @ Rotation.from_rotvec(self.alpha * delta).as_matrix()
        translation = (1 - self.alpha) * previous.translation + self.alpha * pose.translation
        self._previous = Pose(rotation, translation, pose.marker_ids, pose.reprojection_error_px)
        return self._previous
