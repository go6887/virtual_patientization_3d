"""Rerun scene: camera coordinates and all physical geometry use millimetres."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh

from .camera import CameraIntrinsics
from .diagnostics import REASON_LABELS
from .geometry import MarkerGeometry
from .tracking import Pose


def load_meshes(path: Path) -> list[tuple[str, rr.Mesh3D]]:
    """Resolve OBJ/MTL once; retain material colours and Blender millimetres."""
    if not path.is_file():
        raise FileNotFoundError(f"Probe/attachment mesh not found: {path}")
    scene = trimesh.load_scene(path, process=False)
    meshes = []
    for index, node_name in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        colors = None
        material = getattr(mesh.visual, "material", None)
        if material is not None and hasattr(material, "diffuse"):
            color = np.asarray(material.diffuse, dtype=np.uint8)
        elif mesh.visual.kind in ("vertex", "face"):
            colors = (
                np.asarray(mesh.visual.to_color().vertex_colors)
                if hasattr(mesh.visual, "to_color")
                else mesh.visual.vertex_colors
            )
            color = np.array([255, 255, 255, 255], np.uint8)
        else:
            color = np.array([207, 216, 226, 255], np.uint8)
        meshes.append(
            (
                f"part_{index:02d}",
                rr.Mesh3D(
                    vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
                    triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
                    vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
                    vertex_colors=colors,
                    albedo_factor=color,
                ),
            )
        )
    if not meshes:
        raise ValueError(f"No triangle meshes in {path}")
    return meshes


def marker_mesh(geometry: MarkerGeometry, marker_id: int) -> rr.Mesh3D:
    """Coloured marker cells avoid external textures and preserve printed orientation."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, geometry.dictionary_name))
    n = dictionary.markerSize + 2
    pattern = cv2.aruco.generateImageMarker(dictionary, marker_id, n)
    corners = geometry.markers[marker_id]
    normal = geometry.normals[marker_id]
    center = corners.mean(axis=0)
    right, down = corners[1] - corners[0], corners[3] - corners[0]
    vertices, faces, colors = [], [], []

    def quad(points, value):
        i = len(vertices)
        vertices.extend(points)
        # Outward winding; Rerun also renders backs by default.
        faces.extend([[i, i + 2, i + 1], [i, i + 3, i + 2]])
        colors.extend([[value, value, value, 255]] * 4)

    quad(center + (corners - center) * (40.0 / geometry.marker_size_mm) + normal * 0.05, 250)
    for row in range(n):
        for col in range(n):
            origin = corners[0] + (col * right + row * down) / n + normal * 0.10
            quad([origin, origin + right / n, origin + (right + down) / n, origin + down / n], int(pattern[row, col]))
    return rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, vertex_colors=colors)


class RerunViewer:
    def __init__(
        self,
        geometry: MarkerGeometry,
        model_path: Path,
        intrinsics: CameraIntrinsics,
        *,
        spawn: bool = True,
        save_path: Path | None = None,
        demo: bool = False,
    ):
        self.geometry = geometry
        self.intrinsics = intrinsics
        self.demo = demo
        self.visible = False
        self.trail = deque(maxlen=240)
        self.probe_meshes = load_meshes(model_path)
        attachment_path = (
            Path(__file__).resolve().parents[2] / "assets/Vscan_marker_attachment/Vscan_marker_attachment.obj"
        )
        # Printable asset is in a bed-oriented frame, verified against the .blend assembly.
        self.attachment_meshes = load_meshes(attachment_path) if attachment_path.is_file() else []
        self.marker_meshes = {i: marker_mesh(geometry, i) for i in geometry.markers}
        rr.init("vscan_probe_tracking", strict=True)
        if spawn:
            rr.spawn()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            sinks = [rr.FileSink(str(save_path))]
            if spawn:
                sinks.append(rr.GrpcSink())
            rr.set_sinks(*sinks)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        width, height = intrinsics.image_size
        rr.log(
            "world/camera",
            rr.Pinhole(
                image_from_camera=intrinsics.camera_matrix,
                resolution=[width, height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=160,
            ),
            static=True,
        )
        rr.log(
            "world/camera_axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * 35,
                colors=[[244, 94, 98], [85, 205, 142], [93, 151, 255]],
                radii=0.65,
            ),
            static=True,
        )
        rr.log(
            "metrics/reprojection_px", rr.SeriesLines(colors=[255, 184, 90], names="Reprojection RMS (px)"), static=True
        )
        rr.log("metrics/visible_markers", rr.SeriesLines(colors=[84, 212, 172], names="Used markers"), static=True)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            name="Probe pose · mm",
                            origin="world",
                            contents=["world/tracked/**", "world/trajectory"],
                            eye_controls=rrb.EyeControls3D(
                                position=[270, -170, -50],
                                look_target=[0, 0, 410],
                                eye_up=[0, -1, 0],
                            ),
                        ),
                        rrb.Spatial2DView(
                            name="Camera · marker detections", origin="world/camera", contents=["world/camera/**"]
                        ),
                        column_shares=[1, 1],
                    ),
                    rrb.Horizontal(
                        rrb.TextDocumentView(name="Tracking status", origin="status"),
                        rrb.TimeSeriesView(name="Reprojection error", origin="metrics/reprojection_px"),
                        column_shares=[1, 1],
                    ),
                    row_shares=[3, 1],
                ),
                collapse_panels=True,
            )
        )

    def _show_geometry(self):
        base = "world/tracked"
        cube_from_probe = self.geometry.cube_from_probe
        rr.log(f"{base}/probe", rr.Transform3D(mat3x3=cube_from_probe[:3, :3], translation=cube_from_probe[:3, 3]))
        for name, mesh in self.probe_meshes:
            rr.log(f"{base}/probe/{name}", mesh)
        if self.attachment_meshes:
            rr.log(f"{base}/attachment", rr.Transform3D(mat3x3=np.diag([-1.0, 1.0, -1.0]), translation=[0, 0, 21]))
            for name, mesh in self.attachment_meshes:
                rr.log(f"{base}/attachment/{name}", mesh)
        for marker_id, mesh in self.marker_meshes.items():
            rr.log(f"{base}/markers/id_{marker_id}", mesh)
        rr.log(
            f"{base}/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * 50,
                colors=[[255, 85, 90], [85, 220, 140], [90, 150, 255]],
                radii=0.7,
            ),
        )
        self.visible = True

    def log_frame(
        self,
        frame_index: int,
        timestamp: float,
        image_bgr: np.ndarray,
        detections: dict[int, np.ndarray],
        pose: Pose | None,
        *,
        loss_reason: str = "no_markers",
        diagnostic_summary: str = "",
    ):
        rr.set_time("frame", sequence=frame_index)
        rr.set_time("elapsed", duration=timestamp)
        rr.log("world/camera/image", rr.Image(image_bgr, color_model="BGR").compress(jpeg_quality=85))
        label = "SYNTHETIC DEMO" if self.demo else "WEBCAM / VIDEO"
        intrinsics = "calibrated" if self.intrinsics.calibrated else "approximate (FOV estimate)"
        if pose is None:
            if self.visible:
                # Geometry is temporal, so Clear actually removes it until reacquired.
                rr.log("world/tracked", rr.Clear(recursive=True))
                rr.log("world/trajectory", rr.Clear(recursive=True))
                self.visible = False
            self.trail.clear()
            reason = REASON_LABELS.get(loss_reason, loss_reason)
            text = f"# LOST · {reason}\n\n{label} · detected IDs: {sorted(detections)}\n\nCamera: {intrinsics}. Probe registration: nominal CAD."
            rr.log("metrics/reprojection_px", rr.Scalars(float("nan")))
            rr.log("metrics/visible_markers", rr.Scalars(0))
        else:
            rr.log("world/tracked", rr.Transform3D(mat3x3=pose.rotation, translation=pose.translation))
            if not self.visible:
                self._show_geometry()
            probe_origin = (pose.matrix @ self.geometry.cube_from_probe)[:3, 3]
            self.trail.append(probe_origin)
            if len(self.trail) > 1:
                rr.log("world/trajectory", rr.LineStrips3D([np.array(self.trail)], colors=[66, 210, 196], radii=0.7))
            x, y, z = probe_origin
            text = (
                f"# TRACKING\n\n{label} · used IDs: {list(pose.marker_ids)}\n\n"
                f"Raw reprojection RMS: **{pose.reprojection_error_px:.2f} px**\n\n"
                f"Probe origin (smoothed): {x:.1f}, {y:.1f}, {z:.1f} mm\n\n"
                f"Camera: {intrinsics}. Probe registration: nominal CAD.\n\n"
                "Camera axes: X right · Y down · Z forward."
            )
            rr.log("metrics/reprojection_px", rr.Scalars(pose.reprojection_error_px))
            rr.log("metrics/visible_markers", rr.Scalars(len(pose.marker_ids)))
        if diagnostic_summary:
            text += f"\n\n{diagnostic_summary}"
        rr.log("status", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))

    def close(self):
        rr.get_global_data_recording().flush()
        rr.disconnect()
