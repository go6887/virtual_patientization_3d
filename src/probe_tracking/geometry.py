"""Read the attachment's measured/nominal geometry without changing its units."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MarkerGeometry:
    """Marker corners and the nominal probe assembly, all in millimetres.

    Corners follow ArUco's decoded top-left, top-right, bottom-right,
    bottom-left order. ``cube_from_probe`` maps OBJ coordinates into the cube.
    """

    dictionary_name: str
    markers: dict[int, np.ndarray]
    normals: dict[int, np.ndarray]
    cube_from_probe: np.ndarray
    marker_size_mm: float
    cube_side_mm: float

    @classmethod
    def load(cls, path: str | Path) -> "MarkerGeometry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("units") != "mm":
            raise ValueError("Marker geometry must use millimetres (units: mm).")
        marker_size = float(data["marker_black_border_outer_side_mm"])
        cube_side = float(data["cube_side_mm"])
        if not np.isfinite([marker_size, cube_side]).all() or min(marker_size, cube_side) <= 0:
            raise ValueError("Marker and cube dimensions must be positive and finite.")
        markers: dict[int, np.ndarray] = {}
        normals: dict[int, np.ndarray] = {}
        for marker in data["markers"]:
            marker_id = int(marker["id"])
            corners = np.asarray(marker["black_border_outer_corners_mm"], dtype=np.float64)
            normal = np.asarray(marker["outward_normal"], dtype=np.float64)
            if marker_id < 0 or marker_id in markers:
                raise ValueError(f"Invalid or duplicate marker ID: {marker_id}")
            if corners.shape != (4, 3) or not np.isfinite(corners).all():
                raise ValueError(f"Marker {marker_id} must contain four finite 3D corners.")
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
                raise ValueError(f"Marker {marker_id} has an invalid outward normal.")
            normal = normal / np.linalg.norm(normal)
            centered = corners - corners.mean(axis=0)
            if np.linalg.matrix_rank(centered, tol=1e-6) != 2:
                raise ValueError(f"Marker {marker_id} must contain nondegenerate coplanar corners.")
            if np.max(np.abs(centered @ normal)) > 1e-4:
                raise ValueError(f"Marker {marker_id} normal is not perpendicular to its face.")
            edges = np.roll(corners, -1, axis=0) - corners
            turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
            if np.any(turns >= -1e-6):
                raise ValueError(f"Marker {marker_id} corners must be convex and clockwise from outside.")
            markers[marker_id] = corners
            normals[marker_id] = normal
        if not markers:
            raise ValueError("Marker geometry contains no markers.")
        assembly = data["nominal_probe_assembly_transform"]
        if assembly.get("from_frame") != "marker_cube" or assembly.get("to_frame") != "probe_model":
            raise ValueError("Assembly transform must map marker_cube to probe_model.")
        probe_from_cube = np.asarray(assembly["homogeneous_4x4"], dtype=np.float64)
        if probe_from_cube.shape != (4, 4) or not np.isfinite(probe_from_cube).all():
            raise ValueError("Assembly transform must be a finite 4x4 matrix.")
        rotation = probe_from_cube[:3, :3]
        if (
            not np.allclose(probe_from_cube[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("Assembly transform must be a rigid, right-handed transform.")
        return cls(
            dictionary_name=str(data["dictionary"]),
            markers=markers,
            normals=normals,
            cube_from_probe=np.linalg.inv(probe_from_cube),
            marker_size_mm=marker_size,
            cube_side_mm=cube_side,
        )
