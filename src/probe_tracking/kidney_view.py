"""Temporal Rerun rendering for a movable kidney and a tracked scan plane."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rerun as rr

if TYPE_CHECKING:
    from .kidney_scan import KidneyScan, ScanFrame


CURRENT_COLOR = [85, 220, 140, 255]
HISTORY_COLOR = [255, 225, 70, 255]
CONTEXT_COLOR = [235, 150, 85, 255]
CUBE_SIZE_FACTOR = 0.94
KIDNEY_ENTITY = "world/kidney"
SCAN_STATUS_ENTITY = "kidney_status"


def boundary_mask(centers: np.ndarray, pitch_mm: float) -> np.ndarray:
    """Select occupied cubes with at least one empty face neighbor."""
    boundary = np.ones(len(centers), dtype=bool)
    if len(centers):
        indices = np.rint((centers - centers.min(axis=0)) / pitch_mm).astype(np.int64)
        occupied = {tuple(index) for index in indices}
        for index, (x, y, z) in enumerate(indices):
            boundary[index] = not (
                (x - 1, y, z) in occupied
                and (x + 1, y, z) in occupied
                and (x, y - 1, z) in occupied
                and (x, y + 1, z) in occupied
                and (x, y, z - 1) in occupied
                and (x, y, z + 1) in occupied
            )
    return boundary


class KidneyView:
    """Keep the kidney outside the subtree cleared when tracking is lost."""

    def __init__(self, scan: KidneyScan):
        self.scan = scan
        self.boundary = boundary_mask(scan.voxels.centers, scan.voxels.pitch_mm)

    def log_frame(self, frame: ScanFrame) -> None:
        """Replace all box batches on the caller's frame and elapsed timelines."""
        voxels = self.scan.voxels
        current = np.asarray(frame.current_mask)
        history = np.asarray(frame.history_mask)
        for mask in (current, history):
            if mask.dtype != np.bool_ or mask.shape != (len(voxels.centers),):
                raise ValueError("Scan masks must contain one boolean per kidney voxel")
        # Keep source-local cube sizes and positions unchanged: their common
        # parent carries placement and physical scale. This must be temporal so
        # scrubbing a recording restores the placement at the selected frame.
        rr.log(
            KIDNEY_ENTITY,
            rr.Transform3D(
                mat3x3=self.scan.kidney_rotation * self.scan.kidney_scale,
                translation=self.scan.kidney_translation,
            ),
        )
        # A current hit is always green, even when it is also in the history.
        # After a history reset it may temporarily be absent from that history.
        context = self.boundary & ~(current | history)
        half_size = voxels.pitch_mm * CUBE_SIZE_FACTOR * 0.5
        for name, selected, color, fill in (
            ("context", context, CONTEXT_COLOR, rr.components.FillMode.MajorWireframe),
            ("history", history & ~current, HISTORY_COLOR, rr.components.FillMode.Solid),
            ("current", current, CURRENT_COLOR, rr.components.FillMode.Solid),
        ):
            count = np.count_nonzero(selected)
            rr.log(
                f"{KIDNEY_ENTITY}/voxels/{name}",
                rr.Boxes3D(
                    centers=voxels.centers[selected],
                    half_sizes=np.full((count, 3), half_size, dtype=np.float32),
                    colors=np.tile(np.asarray(color, dtype=np.uint8), (count, 1)),
                    fill_mode=fill,
                    # Exterior edges must remain visible at the default camera
                    # distance; keep the open faces so internal hits stay visible.
                    radii=voxels.pitch_mm * (0.06 if name == "context" else 0.018),
                    show_labels=False,
                ),
            )
        # Rerun's bundled fonts do not include Japanese glyphs. Keep the live
        # status legible in both native and web viewers; README explains it in Japanese.
        state = "HIT - intersecting" if frame.hit_count else "MISS - no intersection"
        current_text = f"**{frame.hit_count}** cubes"
        if not frame.tracked:
            state = "LOST - UNAVAILABLE"
            current_text = "unavailable"
        x, y, z = self.scan.kidney_position
        rr.log(
            SCAN_STATUS_ENTITY,
            rr.TextDocument(
                f"**{state}**\n\nCurrent: {current_text} · "
                f"History: **{frame.history_count} / {len(voxels.centers)}**"
                f" · **{frame.coverage_percent:.1f}%**\n\n"
                f"Center: {x:.0f}, {y:.0f}, {z:.0f} mm · Size: **{self.scan.kidney_scale * 100:.0f}%**\n\n"
                "Green: current · Yellow: history · Outline: unscanned\n\n"
                "Preview: R clears history",
                media_type=rr.MediaType.MARKDOWN,
            ),
        )

    def log_fan(self) -> None:
        """Log the exact intersection geometry in the probe's local frame.

        These entities must be temporal: the tracked subtree is recursively
        cleared on loss, and recreated on reacquisition.
        """
        fan = self.scan.fan
        root = "world/tracked/probe/ultrasound"
        rr.log(
            f"{root}/fill",
            rr.Mesh3D(
                vertex_positions=fan["vertices"],
                triangle_indices=fan["triangles"],
                albedo_factor=[62, 213, 240, 38],
            ),
        )
        for name, strips, radius, color in (
            ("outline", [fan["outline"]], 0.45, [98, 223, 248, 210]),
            ("source_arc", [fan["source_arc"]], 0.8, [147, 245, 255, 255]),
            ("scan_lines", fan["scan_lines"], 0.15, [88, 203, 226, 65]),
            ("depth_arcs", fan["depth_arcs"], 0.25, [109, 221, 240, 135]),
        ):
            rr.log(f"{root}/{name}", rr.LineStrips3D(strips, radii=radius, colors=color))
        rr.log(
            f"{root}/depth_labels",
            rr.Points3D(
                fan["depth_label_positions"],
                radii=0.15,
                colors=[158, 231, 246],
                labels=fan["depth_labels"],
                show_labels=True,
            ),
        )
