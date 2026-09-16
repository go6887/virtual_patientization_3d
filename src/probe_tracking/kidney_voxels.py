"""Solid kidney cubes intersected with the finite displayed ultrasound plane.

Voxels remain in the original mesh's millimeter coordinates. Only positive-area
overlap with the ideal zero-thickness fan counts as a hit.
"""

import math
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh

from .ultrasound_geometry import FanGeometry

MAX_GRID_CELLS = 2_000_000
_DISTANCE_EPSILON = 1e-8  # millimeters
_CORNERS = np.array(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=float,
)
_EDGES = np.array(
    [
        [first, second]
        for first in range(8)
        for second in range(first + 1, 8)
        if np.count_nonzero(_CORNERS[first] != _CORNERS[second]) == 1
    ]
)


def validate_voxel_size(pitch_mm: float) -> float:
    """Keep interactive volume resolution within the supported 1–10 mm range."""
    try:
        pitch = float(pitch_mm)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Voxel size must be a finite number between 1 and 10 mm") from error
    if isinstance(pitch_mm, bool) or not math.isfinite(pitch) or not 1 <= pitch <= 10:
        raise ValueError("Voxel size must be a finite number between 1 and 10 mm")
    return pitch


def validate_kidney_scale(scale: float) -> float:
    """Validate a uniform organ scale without changing source voxel resolution."""
    try:
        value = float(scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Kidney scale must be a finite number between 0.1 and 5.0") from error
    if isinstance(scale, (bool, np.bool_)) or not math.isfinite(value) or not 0.1 <= value <= 5.0:
        raise ValueError("Kidney scale must be a finite number between 0.1 and 5.0")
    return value


@dataclass(frozen=True)
class VoxelKidney:
    centers: np.ndarray
    pitch_mm: float

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=float, copy=True)
        if centers.shape == (0,):
            centers = centers.reshape((0, 3))
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
            raise ValueError("Voxel centers must be finite XYZ coordinates")
        centers.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "pitch_mm", validate_voxel_size(self.pitch_mm))

    @property
    def voxel_count(self) -> int:
        return len(self.centers)

    @property
    def volume_mm3(self) -> float:
        """Volume of the cube approximation, including its stepped boundary."""
        return self.voxel_count * self.pitch_mm**3


@dataclass(frozen=True)
class VoxelIntersection:
    hit_mask: np.ndarray

    @property
    def hit_count(self) -> int:
        return int(np.count_nonzero(self.hit_mask))


def voxelize_kidney(mesh: trimesh.Trimesh, pitch_mm: float = 3.0) -> VoxelKidney:
    """Fill a closed mesh with source-aligned cubes without altering the mesh.

    STL's duplicate vertices are welded on a copy. Open or nonmanifold meshes
    are rejected rather than presenting a surface shell as a solid organ.
    Boundary cubes approximate the source surface at the chosen resolution.
    The dense fill grid is bounded before allocation to prevent tiny-pitch OOMs.
    """
    pitch = validate_voxel_size(pitch_mm)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Kidney voxelization requires a nonempty triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Kidney mesh vertices must be finite")
    shape = np.ceil(np.asarray(mesh.extents) / pitch) + 3
    if not np.isfinite(shape).all() or math.prod(int(value) for value in shape) > MAX_GRID_CELLS:
        raise ValueError(f"Voxel grid exceeds {MAX_GRID_CELLS:,} cells; increase the voxel size")
    solid = mesh.copy()
    solid.merge_vertices()
    solid.update_faces(solid.nondegenerate_faces() & solid.unique_faces())
    solid.remove_unreferenced_vertices()
    if not solid.is_watertight:
        raise ValueError("Kidney mesh must be closed and watertight to fill its interior with cubes")
    solid.fix_normals(multibody=True)
    if not math.isfinite(solid.volume) or solid.volume <= 0:
        raise ValueError("Kidney mesh must enclose a positive finite volume")
    filled = solid.voxelized(pitch).fill()
    return VoxelKidney(filled.points, pitch)


def _pose(translation, rotation) -> tuple[np.ndarray, np.ndarray]:
    translation_array = np.asarray(translation, dtype=float)
    rotation_array = np.asarray(rotation, dtype=float)
    if (
        translation_array.shape != (3,)
        or rotation_array.shape != (3, 3)
        or not np.isfinite(translation_array).all()
        or not np.isfinite(rotation_array).all()
        or not np.allclose(rotation_array.T @ rotation_array, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation_array), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError("Voxel intersection requires finite translations and rigid 3D rotations")
    return translation_array, rotation_array


def intersect_voxels_with_fan(
    voxels: VoxelKidney,
    kidney_translation: list[float],
    kidney_rotation: list[list[float]],
    probe_translation: list[float],
    probe_rotation: list[list[float]],
    fan: FanGeometry,
    *,
    kidney_scale: float = 1.0,
) -> VoxelIntersection:
    """Intersect transformed cubes with the finite rendered fan.

    Translation arguments are actual model transforms, including pivot
    compensation. Uniform scale applies to source centers and cube sizes before
    the rigid transform. A cube is highlighted only for positive-area overlap: point
    and edge contacts are excluded. If the plane lies on a shared cube face,
    both cubes can be highlighted because each has a positive-area section.
    """
    kidney_t, kidney_r = _pose(kidney_translation, kidney_rotation)
    probe_t, probe_r = _pose(probe_translation, probe_rotation)
    scale = validate_kidney_scale(kidney_scale)
    relative_r = probe_r.T @ kidney_r
    relative_t = probe_r.T @ (kidney_t - probe_t)
    outline = np.asarray(fan["outline"], dtype=float)
    if outline.ndim != 2 or outline.shape[1] != 3 or not np.isfinite(outline).all():
        raise ValueError("Fan outline must contain finite XYZ coordinates")
    fan_polygon = shapely.Polygon(outline[:, [0, 2]])
    if fan_polygon.is_empty or not fan_polygon.is_valid:
        raise ValueError("Fan outline must enclose a valid scan region")
    plane_y = float(outline[0, 1])
    if not np.allclose(outline[:, 1], plane_y, atol=_DISTANCE_EPSILON, rtol=0):
        raise ValueError("Fan outline must lie in the probe-local X/Z plane")
    hit_mask = np.zeros(voxels.voxel_count, dtype=bool)
    empty = VoxelIntersection(hit_mask)
    if not voxels.voxel_count:
        return empty

    # A vectorized conservative plane and fan-bounds test removes almost all
    # cubes before constructing any planar polygons.
    scaled_pitch = voxels.pitch_mm * scale
    centers = (voxels.centers * scale) @ relative_r.T + relative_t
    offsets = (_CORNERS * (scaled_pitch / 2)) @ relative_r.T
    half_bounds = np.sum(np.abs(relative_r), axis=1) * (scaled_pitch / 2)
    low_x, low_z, high_x, high_z = fan_polygon.bounds
    candidates = np.flatnonzero(
        (np.abs(centers[:, 1] - plane_y) <= half_bounds[1] + _DISTANCE_EPSILON)
        & (centers[:, 0] + half_bounds[0] >= low_x)
        & (centers[:, 0] - half_bounds[0] <= high_x)
        & (centers[:, 2] + half_bounds[2] >= low_z)
        & (centers[:, 2] - half_bounds[2] <= high_z)
    )
    if not len(candidates):
        return empty
    vertices = centers[candidates, None, :] + offsets[None, :, :]
    distances = vertices[:, :, 1] - plane_y
    starts, ends = _EDGES[:, 0], _EDGES[:, 1]
    distance_start, distance_end = distances[:, starts], distances[:, ends]
    denominator = distance_end - distance_start
    nonparallel = np.abs(denominator) > _DISTANCE_EPSILON
    parameters = np.divide(
        -distance_start,
        denominator,
        out=np.full_like(distance_start, np.nan),
        where=nonparallel,
    )
    crosses = nonparallel & (parameters >= 0) & (parameters <= 1)
    edge_sections = vertices[:, starts] + parameters[:, :, None] * (offsets[ends] - offsets[starts])
    edge_groups, _ = np.nonzero(crosses)
    on_plane = np.abs(distances) <= _DISTANCE_EPSILON
    vertex_groups, _ = np.nonzero(on_plane)
    points = np.concatenate([edge_sections[crosses], vertices[on_plane]])[:, [0, 2]]
    groups = np.concatenate([edge_groups, vertex_groups])
    if not len(points):
        return empty

    # Hulls handle duplicate vertices, hexagonal oblique sections, and tangent
    # lines/points in one batched GEOS call. Only positive-area clips count.
    active_groups, compact_groups = np.unique(groups, return_inverse=True)
    order = np.argsort(compact_groups, kind="stable")
    sections = shapely.convex_hull(shapely.multipoints(points[order], indices=compact_groups[order]))
    clipped = shapely.intersection(sections, fan_polygon)
    areas = shapely.area(clipped)
    positive = areas > max(1e-12, scaled_pitch**2 * 1e-10)
    if not np.any(positive):
        return empty
    hit_mask[candidates[active_groups[positive]]] = True
    return VoxelIntersection(hit_mask)
