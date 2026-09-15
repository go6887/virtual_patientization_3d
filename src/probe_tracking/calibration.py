"""Chessboard printing and interactive webcam calibration with OpenCV."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraIntrinsics, validate_image_size


def _validate_board(columns: int, rows: int, square_size_mm: float) -> None:
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 2 for v in (columns, rows)):
        raise ValueError("Chessboard columns and rows are inner corner counts, each at least 2")
    if (
        isinstance(square_size_mm, bool)
        or not isinstance(square_size_mm, Real)
        or not math.isfinite(square_size_mm)
        or square_size_mm <= 0
    ):
        raise ValueError("Chessboard square size must be a finite positive number in mm")


def create_chessboard(path: str | Path, columns: int, rows: int, square_size_mm: float) -> Path:
    """Create a physically sized SVG; rows/columns count *inner corners*."""
    _validate_board(columns, rows, square_size_mm)
    square = float(square_size_mm)
    margin = square
    width, height = (columns + 3) * square, (rows + 3) * square
    rectangles = []
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                rectangles.append(
                    f'<rect x="{margin + column * square:g}" y="{margin + row * square:g}" '
                    f'width="{square:g}" height="{square:g}" fill="black"/>'
                )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}mm" height="{height:g}mm" '
        f'viewBox="0 0 {width:g} {height:g}">\n'
        f"<title>{columns} x {rows} inner corners, {square:g} mm squares. Print at 100%.</title>\n"
        f'<rect width="{width:g}" height="{height:g}" fill="white"/>\n' + "\n".join(rectangles) + "\n</svg>\n",
        encoding="utf-8",
    )
    return destination


def calibrate_from_corners(
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    columns: int,
    rows: int,
    square_size_mm: float,
) -> CameraIntrinsics:
    """Solve standard pinhole intrinsics from at least three board views.

    Each view contains ``rows * columns`` pixel coordinates in the order
    returned by OpenCV's chessboard detector. Use many tilted views spread
    across the frame; three is only the numerical minimum accepted here.
    """
    _validate_board(columns, rows, square_size_mm)
    width, height = validate_image_size(*image_size)
    if len(image_points) < 3:
        raise ValueError("At least three chessboard views are needed for calibration")
    objects = np.zeros((rows * columns, 3), np.float32)
    objects[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_mm
    views = []
    for points in image_points:
        points = np.asarray(points)
        if points.dtype.kind not in "iuf" or points.shape not in ((rows * columns, 2), (rows * columns, 1, 2)):
            raise ValueError(f"Each calibration view must contain {rows * columns} numeric corner pairs")
        points = points.astype(np.float32).reshape(-1, 1, 2)
        if not np.isfinite(points).all():
            raise ValueError("Calibration corners must be finite")
        if (
            np.any(points[..., 0] < 0)
            or np.any(points[..., 0] >= width)
            or np.any(points[..., 1] < 0)
            or np.any(points[..., 1] >= height)
        ):
            raise ValueError("Calibration corners must lie inside the image")
        if np.unique(points.reshape(-1, 2), axis=0).shape[0] != rows * columns:
            raise ValueError("Calibration corners must be distinct")
        views.append(points)
    try:
        rms, matrix, distortion, _, _ = cv2.calibrateCamera(
            [objects.copy() for _ in views], views, (width, height), None, None
        )
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not calibrate these views: {exc}") from exc
    return CameraIntrinsics(matrix, distortion, (width, height), calibrated=True, rms_error_px=rms)


def run_calibration(
    camera: int,
    width: int,
    height: int,
    columns: int,
    rows: int,
    square_size_mm: float,
    output: Path,
    min_views: int = 15,
) -> CameraIntrinsics | None:
    """Capture boards with Space; C solves/saves; Q or Escape cancels."""
    _validate_board(columns, rows, square_size_mm)
    validate_image_size(width, height)
    if isinstance(min_views, bool) or not isinstance(min_views, Integral) or min_views < 3:
        raise ValueError("min_views must be an integer of at least 3")
    if isinstance(camera, bool) or not isinstance(camera, Integral) or camera < 0:
        raise ValueError("Camera index must be a nonnegative integer")
    capture = cv2.VideoCapture(int(camera))
    window = "Probe camera calibration"
    image_points: list[np.ndarray] = []
    actual_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {camera}; check camera access and other camera apps")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print(
            f"Calibration: {columns} x {rows} INNER corners, {square_size_mm:g} mm squares.\n"
            "Print at 100% and measure a square. Keep the board flat.\n"
            "Move it to frame corners and center; change distance and tilt in both axes.\n"
            f"SPACE: capture a detected board; C: solve/save after {min_views} views; Q/ESC: cancel."
        )
        status = "Move and tilt the board; SPACE captures"
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped providing images during calibration")
            current_size = (frame.shape[1], frame.shape[0])
            if actual_size is None:
                actual_size = current_size
                print(f"Actual camera resolution: {actual_size[0]} x {actual_size[1]}")
            elif current_size != actual_size:
                raise RuntimeError("Camera resolution changed during calibration; restart capture")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                (columns, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
            )
            if found and corners is not None:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                found = bool(np.isfinite(corners).all())
            if found:
                cv2.drawChessboardCorners(frame, (columns, rows), corners, found)
            cv2.putText(
                frame,
                f"Views: {len(image_points)}/{min_views}  Board: {'OK' if found else 'not found'}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 220, 50),
                2,
            )
            cv2.putText(
                frame,
                "SPACE: capture  C: calibrate/save  Q: quit",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(frame, status, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                print("Calibration cancelled; no file saved.")
                return None
            if key == ord(" "):
                if not found or corners is None:
                    status = "Board not detected: show all inner corners"
                elif any(float(np.sqrt(np.mean((corners - previous) ** 2))) < 12 for previous in image_points):
                    status = "Similar view already captured: move/tilt the board"
                else:
                    image_points.append(corners.copy())
                    status = f"Captured {len(image_points)} views; change tilt and position"
                    print(status)
            elif key in (ord("c"), ord("C")):
                if len(image_points) < min_views:
                    status = f"Need {min_views - len(image_points)} more varied views"
                    continue
                try:
                    intrinsics = calibrate_from_corners(image_points, actual_size, columns, rows, square_size_mm)
                except ValueError as exc:
                    status = "Calibration failed; capture more varied views"
                    print(f"{status}: {exc}")
                    continue
                intrinsics.save(output)
                print(f"Saved camera calibration: {output} (RMS {intrinsics.rms_error_px:.3f} px)")
                if intrinsics.rms_error_px is not None and intrinsics.rms_error_px > 1:
                    print("RMS is above 1 px; consider recapturing sharp, varied views for better tracking.")
                return intrinsics
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV calibration window or capture failed: {exc}") from exc
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
