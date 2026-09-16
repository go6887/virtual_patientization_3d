"""A convex-array scan cross-section in probe-local millimeter coordinates.

The lens is approximated by a circular arc; scan lines begin on that finite
surface and diverge along its radii. The center of curvature is a geometric
reference inside the probe, never an emitting point or rendered vertex.
"""

import math
from typing import TypedDict

# Coordinates are eventually rendered as float32 in Rerun.
MAX_COORDINATE = float.fromhex("0x1.fffffep+127")


MAX_SCAN_DEPTH_CM = 24.0
MAX_SCAN_ANGLE_DEG = 60.0
# Enclosing circle of the supplied approximate OBJ lens, not a GE calibration.
# KidneyScan measures the loaded OBJ and supplies its fitted value explicitly.
DEFAULT_CURVATURE_RADIUS_MM = 52.66506676


class FanGeometry(TypedDict):
    vertices: list[list[float]]
    triangles: list[list[int]]
    outline: list[list[float]]
    source_arc: list[list[float]]
    virtual_origin: list[float]
    curvature_radius_mm: float
    depth_arcs: list[list[list[float]]]
    depth_labels: list[str]
    depth_label_positions: list[list[float]]
    scan_lines: list[list[list[float]]]
    depth_mm: float
    angle_deg: float


def validate_fan_parameters(depth_cm: float = 15.0, angle_deg: float = 60.0) -> tuple[float, float]:
    """Bound the display to the Vscan Air CL curved array's published range."""
    try:
        depth, angle = float(depth_cm), float(angle_deg)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Scan depth and angle must be finite numbers") from exc
    if isinstance(depth_cm, bool) or not math.isfinite(depth) or not 0 < depth <= MAX_SCAN_DEPTH_CM:
        raise ValueError("Vscan Air CL: 深さは0より大きく24cm以下にしてください")
    if isinstance(angle_deg, bool) or not math.isfinite(angle) or not 0 < angle <= MAX_SCAN_ANGLE_DEG:
        raise ValueError("Vscan Air CL: 表示角は0より大きく60度以下にしてください")
    return depth, angle


def make_fan_geometry(
    tip: list[float],
    depth_cm: float = 15.0,
    angle_deg: float = 60.0,
    segments: int = 64,
    *,
    curvature_radius_mm: float = DEFAULT_CURVATURE_RADIUS_MM,
) -> FanGeometry:
    """Triangulate the region between the lens arc and the requested depth arc.

    tip is the apex of the lens's enclosing circle, pointing along local +Z.
    Depth is distance from each point on the lens arc along its scan direction.
    A smaller angle crops the display around +Z without changing lens curvature.
    Guide rays indicate selected scan directions, not individual elements or
    simultaneous transmissions. This plane does not model acoustic intensity,
    focusing, refraction, or the finite elevational thickness of a real beam.
    """
    depth, angle = validate_fan_parameters(depth_cm, angle_deg)
    if type(segments) is not int or not 2 <= segments <= 4096:
        raise ValueError("Fan segments must be an integer between 2 and 4096")
    try:
        if len(tip) != 3 or any(isinstance(value, bool) for value in tip):
            raise ValueError("Lens apex must have three finite coordinates")
        apex = [float(value) for value in tip]
        radius = float(curvature_radius_mm)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Lens apex and curvature radius must be finite numbers") from exc
    if any(not math.isfinite(value) or abs(value) > MAX_COORDINATE for value in apex):
        raise ValueError("Lens apex must fit in the 3D coordinate range")
    if isinstance(curvature_radius_mm, bool) or not math.isfinite(radius) or not 0 < radius <= MAX_COORDINATE:
        raise ValueError("Lens curvature radius must be positive and fit in the 3D coordinate range")

    depth_mm = depth * 10.0
    half_angle = math.radians(angle) / 2.0
    angles = [-half_angle + 2.0 * half_angle * index / segments for index in range(segments + 1)]
    virtual_origin = [apex[0], apex[1], apex[2] - radius]

    def point(distance: float, theta: float) -> list[float]:
        result = [
            virtual_origin[0] + (radius + distance) * math.sin(theta),
            virtual_origin[1],
            virtual_origin[2] + (radius + distance) * math.cos(theta),
        ]
        if any(not math.isfinite(value) or abs(value) > MAX_COORDINATE for value in result):
            raise ValueError("Scan plane would exceed the 3D coordinate range")
        return result

    def arc(distance: float) -> list[list[float]]:
        return [point(distance, theta) for theta in angles]

    source_arc = arc(0.0)
    outer_arc = arc(depth_mm)
    guide_depths = [5.0 * index for index in range(1, int(depth / 5.0) + 1)]
    if not guide_depths or guide_depths[-1] < depth:
        guide_depths.append(depth)
    count = segments + 1
    triangles = []
    for index in range(segments):
        # Both triangles face local +Y. No triangle reaches the virtual origin.
        triangles.extend(
            [
                [index, count + index, count + index + 1],
                [index, count + index + 1, index + 1],
            ]
        )

    return {
        "vertices": source_arc + outer_arc,
        "triangles": triangles,
        "outline": [source_arc[0]] + outer_arc + list(reversed(source_arc)),
        "source_arc": source_arc,
        "virtual_origin": virtual_origin,
        "curvature_radius_mm": radius,
        "depth_arcs": [arc(guide * 10.0) for guide in guide_depths],
        "depth_labels": [f"{guide:g} cm" for guide in guide_depths],
        "depth_label_positions": [point(guide * 10.0, half_angle) for guide in guide_depths],
        "scan_lines": [
            [point(0, theta), point(depth_mm, theta)]
            for theta in [-half_angle + 2.0 * half_angle * index / 8.0 for index in range(9)]
        ],
        "depth_mm": depth_mm,
        "angle_deg": angle,
    }
