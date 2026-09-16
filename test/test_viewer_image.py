"""Preview JPEGs keep camera dimensions and BGR colors without PIL conversion."""

import cv2
import numpy as np
import pytest
import rerun as rr

from probe_tracking import viewer


def test_preview_jpeg_preserves_dimensions_colors_and_quality(monkeypatch):
    image = np.zeros((96, 192, 3), dtype=np.uint8)
    image[:, :64] = [255, 0, 0]
    image[:, 64:128] = [0, 255, 0]
    image[:, 128:] = [0, 0, 255]
    original = image.copy()
    encoded_calls = []
    logged = []
    original_encode = cv2.imencode

    def encode(extension, data, parameters):
        encoded_calls.append((extension, parameters))
        return original_encode(extension, data, parameters)

    monkeypatch.setattr(cv2, "imencode", encode)
    monkeypatch.setattr(rr, "log", lambda *args: logged.append(args))
    scene = viewer.RerunViewer.__new__(viewer.RerunViewer)
    scene.log_image(image)

    assert encoded_calls == [(".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85])]
    assert len(logged) == 1
    path, encoded = logged[0]
    assert path == "world/camera/image"
    assert isinstance(encoded, rr.EncodedImage)
    assert encoded.media_type.as_arrow_array().to_pylist() == ["image/jpeg"]
    jpeg = np.asarray(encoded.blob.as_arrow_array().to_pylist()[0], dtype=np.uint8)
    decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape
    # Sample inside each solid block to avoid JPEG chroma mixing at boundaries.
    np.testing.assert_allclose(decoded[48, [32, 96, 160]], image[48, [32, 96, 160]], atol=3)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("exception", [None, cv2.error("encoder failed")])
def test_preview_encoding_failure_does_not_log_invalid_image(monkeypatch, exception):
    logged = []

    def fail(*args):
        if exception is not None:
            raise exception
        return False, None

    monkeypatch.setattr(cv2, "imencode", fail)
    monkeypatch.setattr(rr, "log", lambda *args: logged.append(args))
    scene = viewer.RerunViewer.__new__(viewer.RerunViewer)
    expected = RuntimeError if exception is None else cv2.error
    with pytest.raises(expected, match="JPEG" if exception is None else "encoder failed"):
        scene.log_image(np.zeros((24, 32, 3), dtype=np.uint8))
    assert logged == []
