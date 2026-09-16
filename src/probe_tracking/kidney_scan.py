"""Camera-fixed kidney coverage, independent of tracking and rendering backends."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .kidney_voxels import VoxelKidney, _pose, intersect_voxels_with_fan, validate_kidney_scale, voxelize_kidney
from .probe_scan_profile import load_convex_scan_profile
from .ultrasound_geometry import FanGeometry, make_fan_geometry, validate_fan_parameters


@dataclass(frozen=True)
class ScanFrame:
    """An immutable snapshot: later updates and resets cannot change old frames."""

    current_mask: np.ndarray
    history_mask: np.ndarray
    tracked: bool

    def __post_init__(self) -> None:
        current = np.array(self.current_mask, dtype=bool, copy=True)
        history = np.array(self.history_mask, dtype=bool, copy=True)
        if current.ndim != 1 or history.shape != current.shape:
            raise ValueError("Current and history masks must be equal-length one-dimensional arrays")
        current.setflags(write=False)
        history.setflags(write=False)
        object.__setattr__(self, "current_mask", current)
        object.__setattr__(self, "history_mask", history)

    @property
    def hit_count(self) -> int:
        return int(np.count_nonzero(self.current_mask))

    @property
    def history_count(self) -> int:
        return int(np.count_nonzero(self.history_mask))

    @property
    def coverage_percent(self) -> float:
        return 100.0 * self.history_count / len(self.history_mask) if len(self.history_mask) else 0.0


def _coordinates(values, name: str) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return coordinates


class KidneyScan:
    """Intersect the displayed probe-local fan with a camera-space kidney.

    ``kidney_translation`` is the actual source-mesh transform, including the
    compensation needed to rotate and scale about the kidney bounding-box centre.
    ``kidney_position`` is that centre's camera-space position. Source voxel
    indices remain stable when placement changes so history follows the organ.
    ``update`` expects camera_from_probe, including the attachment registration.
    """

    def __init__(
        self,
        voxels: VoxelKidney,
        fan: FanGeometry,
        *,
        kidney_translation=(0.0, 0.0, 0.0),
        kidney_rotation=None,
        source_center=None,
        scale: float = 1.0,
    ) -> None:
        translation, rotation = _pose(kidney_translation, np.eye(3) if kidney_rotation is None else kidney_rotation)
        scale = validate_kidney_scale(scale)
        if source_center is None:
            source_center = (
                (voxels.centers.min(axis=0) + voxels.centers.max(axis=0)) / 2
                if voxels.voxel_count
                else np.zeros(3)
            )
        self.source_center = _coordinates(source_center, "Kidney source center").copy()
        self.source_center.setflags(write=False)
        self.voxels = voxels
        self.fan = fan
        self.kidney_translation = translation.copy()
        self.kidney_rotation = rotation.copy()
        self.kidney_position = _coordinates(
            translation + rotation @ (scale * self.source_center), "Kidney position"
        ).copy()
        self.kidney_scale = scale
        self.kidney_translation.setflags(write=False)
        self.kidney_rotation.setflags(write=False)
        self.kidney_position.setflags(write=False)
        empty = np.zeros(voxels.voxel_count, dtype=bool)
        self.frame = ScanFrame(empty, empty, False)

    @classmethod
    def load(
        cls,
        kidney_path: Path,
        probe_path: Path,
        *,
        position_mm=(0.0, -180.0, 400.0),
        rotation_deg=(0.0, 0.0, 0.0),
        voxel_mm: float = 3.0,
        beam_depth_cm: float = 15.0,
        beam_angle_deg: float = 60.0,
        scale: float = 1.0,
    ) -> "KidneyScan":
        position = _coordinates(position_mm, "Kidney position")
        angles = _coordinates(rotation_deg, "Kidney rotation")
        scale = validate_kidney_scale(scale)
        validate_fan_parameters(beam_depth_cm, beam_angle_deg)
        # Lowercase axes are fixed/extrinsic: X then Y then Z => Rz @ Ry @ Rx.
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        kidney_path, probe_path = Path(kidney_path), Path(probe_path)
        if not kidney_path.is_file():
            raise FileNotFoundError(f"Kidney mesh not found: {kidney_path}")
        scene = trimesh.load_scene(kidney_path, process=False, skip_materials=True)
        mesh = scene.to_mesh()
        voxels = voxelize_kidney(mesh, voxel_mm)
        source_center = mesh.bounds.mean(axis=0)
        profile = load_convex_scan_profile(probe_path)
        fan = make_fan_geometry(
            profile.apex,
            beam_depth_cm,
            beam_angle_deg,
            curvature_radius_mm=profile.curvature_radius_mm,
        )
        return cls(
            voxels,
            fan,
            kidney_translation=position - rotation @ (scale * source_center),
            kidney_rotation=rotation,
            source_center=source_center,
            scale=scale,
        )

    def set_placement(self, *, position_mm=None, scale: float | None = None) -> None:
        """Atomically move/resize the organ; the next update recomputes current hits.

        The pivot stays at ``position_mm`` when scaling. Current and historical
        masks are retained here; voxel identities never change with placement.
        """
        position = _coordinates(
            self.kidney_position if position_mm is None else position_mm, "Kidney position"
        ).copy()
        new_scale = self.kidney_scale if scale is None else validate_kidney_scale(scale)
        translation = _coordinates(
            position - self.kidney_rotation @ (new_scale * self.source_center), "Kidney translation"
        ).copy()
        position.setflags(write=False)
        translation.setflags(write=False)
        self.kidney_position = position
        self.kidney_scale = new_scale
        self.kidney_translation = translation

    def update(self, camera_from_probe: np.ndarray | None) -> ScanFrame:
        if camera_from_probe is None:
            current = np.zeros(self.voxels.voxel_count, dtype=bool)
        else:
            transform = np.asarray(camera_from_probe, dtype=float)
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            ):
                raise ValueError("Probe pose must be a finite rigid 4x4 transform")
            current = intersect_voxels_with_fan(
                self.voxels,
                self.kidney_translation,
                self.kidney_rotation,
                transform[:3, 3],
                transform[:3, :3],
                self.fan,
                kidney_scale=self.kidney_scale,
            ).hit_mask
        self.frame = ScanFrame(
            current,
            self.frame.history_mask | current,
            camera_from_probe is not None,
        )
        return self.frame

    def reset_history(self) -> ScanFrame:
        """Erase history now, retaining current hits until the next valid frame."""
        self.frame = ScanFrame(
            self.frame.current_mask,
            np.zeros(self.voxels.voxel_count, dtype=bool),
            self.frame.tracked,
        )
        return self.frame
