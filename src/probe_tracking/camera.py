"""Validated OpenCV pinhole camera intrinsics and JSON persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np


def validate_image_size(width: int, height: int) -> tuple[int, int]:
    """Return positive integer dimensions, rejecting silent truncation."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in (width, height)):
        raise ValueError("Image width and height must be positive integers")
    return int(width), int(height)


def _numeric_array(value: object, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        result = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    return result


@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrinsics for an unmirrored image; ``image_size`` is (width, height)."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    calibrated: bool = True
    rms_error_px: float | None = None

    def __post_init__(self) -> None:
        matrix = _numeric_array(self.camera_matrix, "camera_matrix")
        distortion = _numeric_array(self.dist_coeffs, "dist_coeffs")
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("Camera focal lengths must be positive")
        if not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0) or not np.allclose(
            [matrix[0, 1], matrix[1, 0]], 0, atol=1e-12, rtol=0
        ):
            raise ValueError("camera_matrix must be an OpenCV pinhole matrix with zero skew")
        if distortion.ndim not in (1, 2) or (distortion.ndim == 2 and 1 not in distortion.shape):
            raise ValueError("dist_coeffs must be a vector")
        if distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError("dist_coeffs must contain 4, 5, 8, 12, or 14 coefficients")
        if not isinstance(self.image_size, (tuple, list)) or len(self.image_size) != 2:
            raise ValueError("image_size must be [width, height]")
        size = validate_image_size(*self.image_size)
        if not isinstance(self.calibrated, bool):
            raise ValueError("calibrated must be a boolean")  # noqa: TRY004 - malformed JSON value
        rms = self.rms_error_px
        if rms is not None:
            if isinstance(rms, bool) or not isinstance(rms, Real) or not math.isfinite(rms) or rms < 0:
                raise ValueError("rms_error_px must be a finite nonnegative number or null")
            rms = float(rms)
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "dist_coeffs", distortion.reshape(-1))
        object.__setattr__(self, "image_size", size)
        object.__setattr__(self, "rms_error_px", rms)

    @classmethod
    def load(cls, path: str | Path) -> CameraIntrinsics:
        """Read the JSON format produced by :meth:`save`."""
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid camera JSON in {source}: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError("Camera JSON must contain an object")  # noqa: TRY004 - malformed JSON value
        try:
            return cls(
                camera_matrix=data["camera_matrix"],
                dist_coeffs=data["dist_coeffs"],
                image_size=data["image_size"],
                calibrated=data.get("calibrated", True),
                rms_error_px=data.get("rms_error_px"),
            )
        except KeyError as exc:
            raise ValueError(f"Camera JSON is missing {exc.args[0]}") from exc

    def save(self, path: str | Path) -> None:
        """Save readable JSON, keeping metric quality and calibration status."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "camera_matrix": self.camera_matrix.tolist(),
                    "dist_coeffs": self.dist_coeffs.tolist(),
                    "image_size": list(self.image_size),
                    "calibrated": self.calibrated,
                    "rms_error_px": self.rms_error_px,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def for_size(self, width: int, height: int) -> CameraIntrinsics:
        """Scale intrinsics only for an equal-aspect-ratio image resize.

        A camera mode that crops the image needs its own calibration even when
        the output happens to have the same aspect ratio.
        """
        width, height = validate_image_size(width, height)
        old_width, old_height = self.image_size
        if width * old_height != height * old_width:
            raise ValueError(
                f"Camera aspect ratio changed: calibrated {old_width}x{old_height}, "
                f"received {width}x{height}. Calibrate at the capture resolution."
            )
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= width / old_width
        matrix[1, :] *= height / old_height
        return CameraIntrinsics(matrix, self.dist_coeffs, (width, height), self.calibrated, self.rms_error_px)

    @classmethod
    def approximate(cls, width: int, height: int, fov_degrees: float = 60.0) -> CameraIntrinsics:
        """Create explicitly uncalibrated intrinsics from horizontal field of view."""
        width, height = validate_image_size(width, height)
        if (
            isinstance(fov_degrees, bool)
            or not isinstance(fov_degrees, Real)
            or not math.isfinite(fov_degrees)
            or not 0 < fov_degrees < 180
        ):
            raise ValueError("Horizontal field of view must be finite and between 0 and 180 degrees")
        focal = width / (2 * math.tan(math.radians(fov_degrees) / 2))
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=float)
        return cls(matrix, np.zeros(5), (width, height), calibrated=False)
