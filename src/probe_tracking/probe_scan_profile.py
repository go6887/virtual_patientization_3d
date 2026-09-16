"""Estimate a convex scanning surface from the supplied probe's exterior mesh.

This is a geometric fit to a photograph-derived OBJ, not a measurement of the
internal transducer array or an acoustic calibration from the manufacturer.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class ConvexScanProfile:
    """Local +Z convex surface used to attach the displayed scanning section.

    ``apex`` is the top of an enclosing circle in the lens's central XZ plane.
    ``curvature_radius_mm`` is its radius; ``fit_error_mm`` is the radial RMS
    residual of the initial least-squares fit, before enlarging it to clear the
    visible lens. Neither quantity specifies the actual device's acoustic array.
    """

    apex: list[float]
    curvature_radius_mm: float
    fit_error_mm: float


def load_convex_scan_profile(path: Path) -> ConvexScanProfile:
    """Fit the supplied OBJ's named convex acoustic lens in its central plane.

    The outer envelope excludes the lens's rear face. The fitted radius is then
    enlarged to enclose every envelope vertex, keeping the displayed surface out
    of the lens. On the supplied Vscan Air CL model the radius is about 52.665 mm
    and the maximum gap over the central 60-degree scan is about 0.535 mm. The
    source model uses millimeters and places the convex end along local +Z.
    """
    try:
        if path.suffix.lower() != ".obj":
            raise ValueError("a material-bearing OBJ is required")
        scene = trimesh.load_scene(path, file_type="obj", process=False)
        lenses = []
        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node]
            geometry = scene.geometry[geometry_name]
            material = getattr(geometry.visual, "material", None)
            material_name = getattr(material, "name", "") or ""
            if not material_name.lower().endswith("convex_acoustic_lens"):
                continue
            if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
                raise ValueError("convex lens has no triangle geometry")
            lens = geometry.copy()
            lens.apply_transform(transform)
            lenses.append(lens)
        if not lenses:
            raise ValueError("convex_acoustic_lens material was not found; check the OBJ and MTL")

        bounds = np.array([lens.bounds for lens in lenses], dtype=float)
        if not np.isfinite(bounds).all():
            raise ValueError("convex lens bounds are not finite")
        plane_y = float((bounds[:, 0, 1].min() + bounds[:, 1, 1].max()) / 2)
        segments = np.concatenate(
            [trimesh.intersections.mesh_plane(lens, [0, 1, 0], [0, plane_y, 0]) for lens in lenses]
        )
        if not len(segments) or not np.isfinite(segments).all():
            raise ValueError("convex lens has no valid central XZ cross section")

        # OBJ values have six decimal places. Merge duplicate plane intersections
        # before taking the +Z surface for each lateral position.
        points = np.unique(np.round(segments.reshape(-1, 3), 6), axis=0)[:, [0, 2]]
        front = np.array([(x, points[points[:, 0] == x, 1].max()) for x in np.unique(points[:, 0])])
        if len(front) < 5:
            raise ValueError("convex lens cross section has too few surface points")

        # Translate before fitting so the solve remains stable if the model's
        # source origin is moved. Circle equation: 2*c.p + k = p.p.
        origin = front.mean(axis=0)
        centered = front - origin
        system = np.column_stack((2 * centered, np.ones(len(front))))
        solution, _, rank, _ = np.linalg.lstsq(system, np.einsum("ij,ij->i", centered, centered), rcond=None)
        if rank < 3:
            raise ValueError("convex lens surface does not define a curved profile")
        circle_center = solution[:2] + origin
        radius_squared = solution[2] + np.dot(solution[:2], solution[:2])
        if not np.isfinite(radius_squared) or radius_squared <= 0:
            raise ValueError("convex lens circle fit has an invalid radius")
        fitted_radius = np.sqrt(radius_squared)
        distances = np.linalg.norm(front - circle_center, axis=1)
        enclosing_radius = float(distances.max())
        fit_error = float(np.sqrt(np.mean((distances - fitted_radius) ** 2)))
        if (
            not np.isfinite(circle_center).all()
            or circle_center[1] >= front[:, 1].min()
            or not front[:, 0].min() < circle_center[0] < front[:, 0].max()
            or not 0 < enclosing_radius < np.finfo(np.float32).max
        ):
            raise ValueError("convex lens profile must curve outward along local +Z")
        end_angles = np.arctan2(front[[0, -1], 0] - circle_center[0], front[[0, -1], 1] - circle_center[1])
        if end_angles[0] > -np.pi / 6 or end_angles[1] < np.pi / 6:
            raise ValueError("convex lens surface does not cover a 60-degree scan")

        return ConvexScanProfile(
            apex=[float(circle_center[0]), plane_y, float(circle_center[1] + enclosing_radius)],
            curvature_radius_mm=enclosing_radius,
            fit_error_mm=fit_error,
        )
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        raise ValueError(f"Cannot determine the convex scan surface of {path.name}: {error}") from error
