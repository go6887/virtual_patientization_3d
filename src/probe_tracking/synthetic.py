"""Deterministic camera images for exercising the real ArUco pose estimator.

The renderer uses the same millimetre geometry as the physical attachment.  It
does not supply its ground-truth pose to the estimator: callers detect the
rendered black-and-white markers just as they would in a webcam frame.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


class _MarkerGeometry(Protocol):
    markers: dict[int, NDArray[np.float64]]
    normals: dict[int, NDArray[np.float64]]
    dictionary_name: str
    marker_size_mm: float


@lru_cache(maxsize=64)
def _marker_texture(dictionary_name: str, marker_id: int) -> NDArray[np.uint8]:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    # An integral number of pixels per module avoids uneven marker cell widths.
    side = (dictionary.markerSize + 2) * 48
    return cv2.aruco.generateImageMarker(dictionary, marker_id, side, borderBits=1)


def demo_pose(time_seconds: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a smooth marker-cube-to-OpenCV-camera pose, in millimetres.

    The +Z face and two side faces generally face the camera.  Translation and
    all three angles change, while the labels remain large enough to detect at
    typical 720p webcam intrinsics.  Camera axes are X right, Y down, Z forward.
    """
    t = float(time_seconds)
    ax = np.pi + 0.46 + 0.13 * np.sin(0.63 * t)
    ay = 0.45 + 0.24 * np.sin(0.47 * t + 0.6)
    az = 0.12 * np.sin(0.38 * t)
    sx, cx = np.sin(ax), np.cos(ax)
    sy, cy = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    translation = np.array(
        [42.0 * np.sin(0.52 * t), 24.0 * np.sin(0.73 * t), 400.0 + 32.0 * np.sin(0.35 * t)],
        dtype=np.float64,
    )
    return rz @ ry @ rx, translation


def render_markers(
    geometry: _MarkerGeometry,
    camera_matrix: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    """Render visible cube markers as a BGR image with a pale background.

    ``rotation`` maps cube coordinates into camera coordinates, and
    ``translation`` is the cube origin in camera millimetres.  The image is
    pinhole/undistorted; use zero distortion when detecting these frames.
    Marker corners are ordered top-left, top-right, bottom-right, bottom-left
    as viewed from outside the printed face.  Faces are backface-culled and
    painted from far to near, with a 4 mm white paper margin around each marker.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    frame = np.full((height, width, 3), 238, dtype=np.uint8)
    visible: list[tuple[float, int, NDArray[np.float64]]] = []
    for marker_id, points in geometry.markers.items():
        camera_points = np.asarray(points, dtype=np.float64) @ rotation.T + translation
        center = camera_points.mean(axis=0)
        normal = rotation @ np.asarray(geometry.normals[marker_id], dtype=np.float64)
        if np.any(camera_points[:, 2] <= 1e-3) or float(normal @ center) >= 0.0:
            continue
        visible.append((float(center[2]), marker_id, camera_points))

    def project(points: NDArray[np.float64]) -> NDArray[np.float32]:
        pixels = points @ camera_matrix.T
        return (pixels[:, :2] / pixels[:, 2:3]).astype(np.float32)

    for _, marker_id, camera_points in sorted(visible, key=lambda item: item[0], reverse=True):
        center = camera_points.mean(axis=0)
        offsets = camera_points - center
        # 42 mm plastic face and 40 mm paper for the supplied 32 mm markers.
        plastic = center + offsets * ((geometry.marker_size_mm + 10.0) / geometry.marker_size_mm)
        paper = center + offsets * ((geometry.marker_size_mm + 8.0) / geometry.marker_size_mm)
        if np.any(plastic[:, 2] <= 1e-3):
            continue
        cv2.fillConvexPoly(frame, np.rint(project(plastic)).astype(np.int32), (155, 173, 153))
        cv2.fillConvexPoly(frame, np.rint(project(paper)).astype(np.int32), (255, 255, 255))
        texture = _marker_texture(geometry.dictionary_name, marker_id)
        side = texture.shape[0]
        # Pixel centres are integer coordinates; physical border corners lie
        # half a pixel beyond the first and last texture pixel centres.
        source = np.array(
            [[-0.5, -0.5], [side - 0.5, -0.5], [side - 0.5, side - 0.5], [-0.5, side - 0.5]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, project(camera_points))
        warped = cv2.warpPerspective(texture, homography, (width, height), flags=cv2.INTER_NEAREST, borderValue=255)
        mask = cv2.warpPerspective(np.full_like(texture, 255), homography, (width, height), flags=cv2.INTER_NEAREST)
        frame[mask != 0] = warped[mask != 0, None]
    return frame
