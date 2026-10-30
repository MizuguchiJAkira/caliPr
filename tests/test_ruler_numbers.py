"""The scale read from the numbers printed along a ruler, a centimetre apart.

Drawn rulers: numbers 0-10 every 60 px (6 px/mm), dark on light and light on a
dark bar, are found to a fraction of a percent; the same digits unevenly spaced,
as handwriting is, are not taken for a ruler; nothing numbered, nothing found.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fish_morpho.ruler_calibration import detect_numbered_scale  # noqa: E402


def ruler(xs, dark_bar=False, size=(500, 900)):
    img = np.full((*size, 3), 235, np.uint8)
    ink = (20, 20, 20)
    if dark_bar:
        cv2.rectangle(img, (20, 300), (880, 420), (25, 25, 25), -1)
        ink = (230, 230, 230)
    for i, x in enumerate(xs):
        cv2.putText(img, str(i), (int(x) - (14 if i > 9 else 7), 380), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ink, 2)
    return img


@pytest.mark.parametrize("dark_bar", [False, True])
def test_numbers_a_centimetre_apart_give_the_scale(dark_bar):
    r = detect_numbered_scale(ruler([60 + 60 * i for i in range(11)], dark_bar))
    assert abs(r["px_per_mm"] - 6.0) < 0.06 and len(r["numbers"]) == 11


def test_unevenly_spaced_marks_are_not_a_ruler():
    rng = np.random.default_rng(1)
    xs = np.cumsum(rng.uniform(40, 80, 11)) + 40
    with pytest.raises(RuntimeError):
        detect_numbered_scale(ruler(xs))


def test_no_numbers_no_scale():
    with pytest.raises(RuntimeError):
        detect_numbered_scale(np.full((400, 800, 3), 235, np.uint8))
