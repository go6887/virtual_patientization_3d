"""GUI startup must call the installed Rerun API with supported arguments."""

from pathlib import Path
from unittest.mock import create_autospec

import pytest
import rerun as rr

from probe_tracking import viewer
from probe_tracking.camera import CameraIntrinsics
from probe_tracking.geometry import MarkerGeometry


@pytest.mark.parametrize("spawn", [True, False], ids=["gui", "headless"])
@pytest.mark.parametrize("save", [False, True], ids=["no-save", "save"])
def test_viewer_startup_uses_supported_spawn_signature_and_keeps_sinks(tmp_path, monkeypatch, spawn, save):
    # Autospec preserves the *installed* rr.spawn signature. Unlike an ordinary
    # MagicMock it rejects unsupported kwargs (e.g. extra_args in Rerun 0.29).
    spawn_mock = create_autospec(rr.spawn, spec_set=True)
    sinks_mock = create_autospec(rr.set_sinks, spec_set=True)
    monkeypatch.setattr(rr, "spawn", spawn_mock)
    monkeypatch.setattr(rr, "set_sinks", sinks_mock)
    mesh = rr.Mesh3D(vertex_positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], triangle_indices=[[0, 1, 2]])
    monkeypatch.setattr(viewer, "load_meshes", lambda path: [("part_00", mesh)])
    monkeypatch.setattr(viewer, "marker_mesh", lambda geometry, marker_id: mesh)
    geometry_path = Path(__file__).resolve().parents[1] / "assets/Vscan_marker_attachment/marker_geometry_42mm.json"
    geometry = MarkerGeometry.load(geometry_path)
    save_path = tmp_path / "session.rrd" if save else None

    # Keep real initialization, archetype validation and blueprint serialization;
    # only GUI process launch and external sink connections are intercepted.
    scene = viewer.RerunViewer(
        geometry,
        tmp_path / "probe.obj",
        CameraIntrinsics.approximate(640, 360),
        spawn=spawn,
        save_path=save_path,
    )
    try:
        if spawn:
            spawn_mock.assert_called_once()
        else:
            spawn_mock.assert_not_called()
        if save:
            sinks_mock.assert_called_once()
            expected_types = [rr.FileSink, rr.GrpcSink] if spawn else [rr.FileSink]
            assert [type(sink) for sink in sinks_mock.call_args.args] == expected_types
        else:
            sinks_mock.assert_not_called()
    finally:
        scene.close()
