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


# ---------------------------------------------------------------------------
# locate_ticks: a dot per millimetre, and whether it stays on the ticks
# ---------------------------------------------------------------------------

def _ruler(positions, w=3000, h=600, y0=280, tick_h=40, long_every=10):
    """Foam with a strip of ruler: a thin dark tick at each x, longer every 10th."""
    img = np.full((h, w), 225, np.uint8)
    img[y0 - 60:y0 + tick_h + 20, :] = 238                    # the ruler's plastic
    for i, x in enumerate(positions):
        length = tick_h + (30 if i % long_every == 0 else 0)
        xi = int(round(x))
        img[y0 + tick_h - length:y0 + tick_h, xi - 1:xi + 2] = 40
    return img


def test_dots_stay_on_the_ticks_of_an_even_ruler():
    from fish_morpho.ruler_calibration import locate_ticks
    P = 25.0
    img = _ruler([200 + k * P for k in range(100)])
    t = locate_ticks(img, P)
    assert t["span_mm"] >= 95
    assert t["within_quarter_mm"] >= 0.95 * len(t["ticks"])
    assert t["max_offset_mm"] <= 0.1
    assert abs(t["px_per_mm_first_end"] - P) < 0.2 and abs(t["px_per_mm_last_end"] - P) < 0.2


def test_millimetres_widening_along_the_ruler_show_as_drift_at_the_ends():
    """One end nearer the camera: spacing grows from 24.4 to 26.0 px/mm."""
    from fish_morpho.ruler_calibration import locate_ticks
    xs, x = [], 200.0
    for k in range(120):
        xs.append(x)
        x += 24.4 + 1.6 * k / 119
    img = _ruler(xs, w=int(xs[-1]) + 300)
    t = locate_ticks(img, 25.2)                                # the one number the detector gives
    assert t["px_per_mm_last_end"] - t["px_per_mm_first_end"] > 1.0
    assert t["max_offset_mm"] > 0.7                             # far past a quarter millimetre at an end
    mid = t["ticks"][len(t["ticks"]) // 2]
    assert abs(mid[2]) < 0.25 * 25.2                            # and on the ticks in the middle


def test_a_photograph_without_a_ruler_says_so():
    from fish_morpho.ruler_calibration import locate_ticks
    rng = np.random.default_rng(3)
    img = (rng.random((600, 3000)) * 30 + 200).astype(np.uint8)
    with pytest.raises(RuntimeError):
        locate_ticks(img, 25.0)
