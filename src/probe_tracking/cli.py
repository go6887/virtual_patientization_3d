"""Webcam/video tracking and an end-to-end synthetic demonstration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraIntrinsics
from .diagnostics import REASON_LABELS, TrackingDiagnostics
from .geometry import MarkerGeometry
from .tracking import PoseSmoother, Tracker

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
DEFAULT_GEOMETRY = ASSETS / "Vscan_marker_attachment/marker_geometry_42mm.json"
DEFAULT_MODEL = ASSETS / "Vscan_Air_CL/Vscan_Air_CL.obj"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vscan Air CL ArUco tracking → Rerun (mm)")
    sub = parser.add_subparsers(dest="command")
    for command, help_text in (
        ("track", "Track a webcam, video or still image"),
        ("demo", "Detect moving synthetic markers without a camera"),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
        p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--height", type=int, default=720)
        p.add_argument("--fps", type=float, default=30)
        p.add_argument("--fov", type=float, default=60, help="Approximate horizontal FOV in degrees")
        p.add_argument("--smooth", type=float, default=0.35, help="Pose smoothing alpha; 1 disables smoothing")
        p.add_argument("--max-error", type=float, default=4, help="Maximum raw reprojection RMS in pixels")
        p.add_argument("--max-frames", type=int, default=300 if command == "demo" else 0)
        p.add_argument("--save", type=Path, help="Save a Rerun .rrd recording")
        p.add_argument("--no-viewer", action="store_true", help="Record without opening Rerun (requires --save)")
        p.add_argument("--no-realtime", action="store_true", help="Process demo/video without frame pacing")
        p.add_argument("--preview", action="store_true", help="Also open an OpenCV preview; Q/Esc stops")
        p.add_argument("--snapshot", type=Path, help="Write the final annotated frame as an image")
        p.add_argument(
            "--diagnostics", type=Path, help="Append per-frame loss reasons and marker corners to a JSONL file"
        )
        if command == "track":
            source = p.add_mutually_exclusive_group()
            source.add_argument("--camera", type=int, default=0)
            source.add_argument("--video", type=Path)
            source.add_argument("--image", type=Path)
            p.add_argument(
                "--calibration", type=Path, help="Camera calibration JSON; auto-loads config/camera.json if present"
            )
    p = sub.add_parser("calibrate", help="Collect chessboard views and calibrate the camera")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--columns", type=int, default=9, help="Number of inner corners horizontally")
    p.add_argument("--rows", type=int, default=6, help="Number of inner corners vertically")
    p.add_argument("--square-mm", type=float, default=20)
    p.add_argument("--min-views", type=int, default=15)
    p.add_argument("--output", type=Path, default=ROOT / "config/camera.json")
    p = sub.add_parser("board", help="Create a chessboard SVG at physical size")
    p.add_argument("--columns", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square-mm", type=float, default=20)
    p.add_argument("--output", type=Path, default=ROOT / "output/chessboard.svg")
    return parser


def annotate(
    frame: np.ndarray,
    detections: dict,
    pose,
    geometry,
    intrinsics,
    *,
    demo: bool,
    loss_reason: str = "no_markers",
    diagnostic_summary: str = "",
) -> np.ndarray:
    result = frame.copy()
    if detections:
        cv2.aruco.drawDetectedMarkers(
            result,
            [v.reshape(1, 4, 2).astype(np.float32) for v in detections.values()],
            np.array(list(detections), dtype=np.int32).reshape(-1, 1),
        )
    if pose is not None:
        cv2.drawFrameAxes(
            result, intrinsics.camera_matrix, np.zeros(5), cv2.Rodrigues(pose.rotation)[0], pose.translation, 32
        )
        # The probe model origin uses the inverse of the supplied cube→probe transform.
        origin = geometry.cube_from_probe[:3, 3].reshape(1, 3)
        pixel, _ = cv2.projectPoints(
            origin, cv2.Rodrigues(pose.rotation)[0], pose.translation, intrinsics.camera_matrix, np.zeros(5)
        )
        if np.isfinite(pixel).all():
            x, y = np.clip(pixel.reshape(2), -100000, 100000).astype(int)
            cv2.drawMarker(result, (x, y), (0, 180, 255), cv2.MARKER_CROSS, 16, 2)
        text = f"TRACKING  IDs {','.join(map(str, pose.marker_ids))}  RMS {pose.reprojection_error_px:.2f} px"
        color = (100, 245, 140)
    else:
        text, color = f"LOST - {REASON_LABELS.get(loss_reason, loss_reason)}", (80, 160, 255)
    cv2.rectangle(result, (0, 0), (result.shape[1], 106 if diagnostic_summary else 78), (26, 30, 36), -1)
    cv2.putText(result, text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    label = (
        "SYNTHETIC DEMO"
        if demo
        else ("CALIBRATED CAMERA" if intrinsics.calibrated else "APPROXIMATE INTRINSICS - calibrate for scale")
    )
    cv2.putText(
        result,
        label + " | nominal CAD registration",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (215, 220, 228),
        1,
        cv2.LINE_AA,
    )
    if diagnostic_summary:
        cv2.putText(
            result, diagnostic_summary, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 225, 235), 1, cv2.LINE_AA
        )
    return result


def run_tracking(args: argparse.Namespace) -> int:
    from .synthetic import demo_pose, render_markers
    from .viewer import RerunViewer

    if args.width <= 0 or args.height <= 0 or not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("width, height and fps must be positive")
    if args.max_frames < 0 or not np.isfinite(args.max_error) or args.max_error <= 0:
        raise ValueError("max-frames must be nonnegative and max-error positive")
    if not np.isfinite(args.smooth) or not 0 < args.smooth <= 1:
        raise ValueError("smooth must be in (0, 1]")
    if args.no_viewer and args.save is None:
        raise ValueError("--no-viewer requires --save output/session.rrd")
    geometry = MarkerGeometry.load(args.geometry)
    demo = args.command == "demo"
    capture = None
    viewer = None
    diagnostics = None
    static_frame = None
    frame_count = tracked_count = 0
    try:
        if demo:
            static_frame = np.zeros((args.height, args.width, 3), np.uint8)
        elif args.image:
            static_frame = cv2.imread(str(args.image))
            if static_frame is None:
                raise ValueError(f"Cannot read image: {args.image}")
        else:
            source = str(args.video) if args.video else args.camera
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(
                    f"Cannot open camera/video {source}. On macOS, allow camera access for your terminal/Codex in System Settings > Privacy & Security > Camera; try --camera 1 if needed."
                )
            if args.video is None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                capture.set(cv2.CAP_PROP_FPS, args.fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, static_frame = capture.read()
            if not ok:
                raise RuntimeError("Camera/video opened but did not return a frame")
        height, width = static_frame.shape[:2]
        path = None if demo else args.calibration
        if not demo and path is None and (ROOT / "config/camera.json").is_file():
            path = ROOT / "config/camera.json"
        intrinsics = (
            CameraIntrinsics.load(path).for_size(width, height)
            if path
            else CameraIntrinsics.approximate(width, height, args.fov)
        )
        if path:
            print(f"Calibration: {path} ({width}x{height})")
        elif not demo:
            print("Camera intrinsics are approximate. Run 'probe-tracker calibrate' for metric scale.", file=sys.stderr)
        maps = cv2.initUndistortRectifyMap(
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            None,
            intrinsics.camera_matrix,
            (width, height),
            cv2.CV_32FC1,
        )
        tracker = Tracker(geometry, intrinsics.camera_matrix, np.zeros(5), max_reprojection_error_px=args.max_error)
        smoother = PoseSmoother(alpha=args.smooth)
        diagnostics = TrackingDiagnostics(
            args.diagnostics,
            {
                "image_size": [width, height],
                "camera_matrix": intrinsics.camera_matrix.tolist(),
                "dist_coeffs_used_for_pose": [0.0] * 5,
                "corners_coordinate_space": "undistorted_pixels",
                "source_calibration_rms_px": intrinsics.rms_error_px,
                "max_reprojection_error_px": args.max_error,
                "geometry_path": str(args.geometry.resolve()),
                "geometry_corners_mm": {str(i): points.tolist() for i, points in geometry.markers.items()},
            },
        )
        viewer = RerunViewer(geometry, args.model, intrinsics, spawn=not args.no_viewer, save_path=args.save, demo=demo)
        print("Tracking started. Stop with Ctrl+C" + (" or Q/Esc in preview." if args.preview else "."), flush=True)
        fps = args.fps
        if not demo and args.video:
            reported = capture.get(cv2.CAP_PROP_FPS)
            if np.isfinite(reported) and reported > 0:
                fps = reported
        start = time.perf_counter()
        last_status = None
        last_report = start
        while args.max_frames == 0 or frame_count < args.max_frames:
            timestamp = (
                frame_count / fps if demo or (not demo and (args.video or args.image)) else time.perf_counter() - start
            )
            if demo:
                rotation, translation = demo_pose(timestamp)
                frame = render_markers(geometry, intrinsics.camera_matrix, rotation, translation, width, height)
            elif frame_count == 0 or capture is None:
                frame = static_frame
            else:
                ok, frame = capture.read()
                if not ok:
                    if args.video:
                        break
                    raise RuntimeError("Webcam stopped returning frames")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    "Capture resolution changed during tracking; restart and recalibrate at the new resolution"
                )
            frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
            detections = tracker.detect(frame)
            pose = tracker.estimate(detections)
            if pose is not None:
                display_pose = smoother.update(pose)
                tracked_count += 1
            else:
                smoother.reset()
                tracker.reset()
                display_pose = None
            diagnostics.record(frame_count, timestamp, tracker, detections, pose)
            summary = diagnostics.recent_summary()
            annotated = annotate(
                frame,
                detections,
                pose,
                geometry,
                intrinsics,
                demo=demo,
                loss_reason=tracker.last_reason,
                diagnostic_summary=summary,
            )
            viewer.log_frame(
                frame_count,
                timestamp,
                annotated,
                detections,
                display_pose,
                loss_reason=tracker.last_reason,
                diagnostic_summary=summary,
            )
            status = "TRACKING" if pose else "LOST"
            if status != last_status:
                detail = (
                    f"; IDs {pose.marker_ids}"
                    if pose
                    else f"; reason={tracker.last_reason}; detected IDs={sorted(detections)}"
                )
                print(f"{status} at frame {frame_count}{detail}", flush=True)
                last_status = status
            now = time.perf_counter()
            if now - last_report >= 2.0:
                print(diagnostics.summary(), flush=True)
                last_report = now
            frame_count += 1
            if args.snapshot:
                snapshot = annotated
            if args.preview:
                cv2.imshow("Vscan probe tracking - Q to stop", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if not demo and args.image:
                break
            if not args.no_realtime and (demo or (not demo and args.video)):
                delay = start + frame_count / fps - time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, 1.0))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if capture is not None:
            capture.release()
        if diagnostics is not None:
            diagnostics.close()
            print(diagnostics.summary(), flush=True)
        if viewer is not None:
            viewer.close()
        if args.preview:
            cv2.destroyAllWindows()
    if args.snapshot and frame_count:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), snapshot):
            raise RuntimeError(f"Could not save snapshot: {args.snapshot}")
    print(f"Processed {frame_count} frames; tracked {tracked_count}.")
    if args.save:
        print(f"Rerun recording: {args.save.resolve()}")
    if args.diagnostics:
        print(f"Tracking diagnostics: {args.diagnostics.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv or ["track"])
    try:
        if args.command == "calibrate":
            from .calibration import run_calibration

            run_calibration(
                args.camera,
                args.width,
                args.height,
                args.columns,
                args.rows,
                args.square_mm,
                args.output,
                args.min_views,
            )
        elif args.command == "board":
            from .calibration import create_chessboard

            create_chessboard(args.output, args.columns, args.rows, args.square_mm)
            print(f"Print at 100% scale: {args.output.resolve()}")
        else:
            return run_tracking(args)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
