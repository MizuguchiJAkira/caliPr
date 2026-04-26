"""Ruler calibration tests — the manual path is pure Python; the auto path
is exercised with a synthetic ruler image so the test runs without any real
lab photo."""

import numpy as np
import pytest

from fish_morpho.ruler_calibration import (
    calibrate,
    detect_ruler_scale,
    scale_from_known_span,
)


def test_manual_calibration_basic():
    calib = scale_from_known_span((0.0, 0.0), (200.0, 0.0), known_mm=100.0)
    assert calib.method == "manual"
    assert calib.px_per_mm == pytest.approx(2.0)
    assert calib.confidence == 1.0


def test_calibrate_falls_back_to_manual_when_image_missing():
    result = calibrate(
        manual_span=((0.0, 0.0), (50.0, 0.0), 25.0),
    )
    assert result.method == "manual"
    assert result.px_per_mm == pytest.approx(2.0)


def test_calibrate_requires_something():
    with pytest.raises(ValueError):
        calibrate()


def _synthetic_ruler(px_per_mm: float = 8.0, length_mm: int = 100) -> np.ndarray:
    """Build a 1 mm-spaced tick pattern on a bright ruler strip, embedded
    in a larger white canvas so the ROI finder has to work for it."""
    cv2 = pytest.importorskip("cv2")
    strip_w = int(px_per_mm * length_mm) + 40
    strip_h = 80
    ruler = np.full((strip_h, strip_w), 240, dtype=np.uint8)
    # Tick marks every 1 mm: a thin dark vertical stripe.
    for i in range(length_mm + 1):
        x = int(round(20 + i * px_per_mm))
        cv2.line(ruler, (x, 20), (x, strip_h - 20), 40, 1)

    # Embed in a bigger canvas.
    canvas = np.full((200, strip_w + 100, 3), 255, dtype=np.uint8)
    canvas[60 : 60 + strip_h, 50 : 50 + strip_w, :] = ruler[:, :, None]
    return canvas


def test_detect_ruler_scale_on_synthetic_image():
    pytest.importorskip("cv2")
    img = _synthetic_ruler(px_per_mm=8.0, length_mm=120)
    result = detect_ruler_scale(img, min_ticks=10)
    assert result.method == "auto"
    # FFT gives us the mean tick period; allow ~5% error.
    assert result.px_per_mm == pytest.approx(8.0, rel=0.05)
    assert result.confidence > 0.2
    assert result.roi is not None


def test_detect_ruler_fails_loudly_on_noise():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(200, 300, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        detect_ruler_scale(noise, min_ticks=10)
