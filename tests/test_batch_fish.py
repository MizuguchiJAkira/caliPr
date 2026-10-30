"""Finding each fish in a photograph of several.

A made-up tray: fish (long, tapered) on a light box with a ruler, a label and its
printed letters beside them. The fish are found, in reading order, and nothing
else is; a crowded tray loses none to the background estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fish_morpho.batch_fish import find_fish  # noqa: E402


def draw_fish(img, cx, cy, length, colour=(60, 110, 150)):
    """A spindle: an ellipse for the body and a forked triangle for the tail."""
    half = length // 2
    cv2.ellipse(img, (cx, cy), (half, max(4, length // 10)), 0, 0, 360, colour, -1)
    tail = np.array([[cx + half - 4, cy], [cx + half + length // 6, cy - length // 9],
                     [cx + half + length // 6, cy + length // 9]], np.int32)
    cv2.fillPoly(img, [tail], colour)


def tray(fish_at, size=(900, 1400)):
    img = np.full((*size, 3), 245, np.uint8)
    for cx, cy, length in fish_at:
        draw_fish(img, cx, cy, length)
    cv2.rectangle(img, (100, 780), (900, 840), (20, 20, 20), -1)         # a ruler
    cv2.rectangle(img, (1000, 700), (1350, 860), (250, 250, 235), -1)    # a label...
    cv2.rectangle(img, (1000, 700), (1350, 860), (90, 90, 90), 2)
    for i in range(8):                                                   # ...and its text
        cv2.putText(img, "Notropis", (1015, 735 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    return img


def test_the_fish_are_found_in_reading_order_and_nothing_else():
    at = [(300, 150, 300), (800, 160, 280), (320, 420, 260), (820, 430, 320)]
    found = find_fish(tray(at))
    assert len(found) == 4
    for f, (cx, cy, _) in zip(found, at):
        x0, y0, x1, y1 = f["box"]
        assert x0 < cx < x1 and y0 < cy < y1
        assert len(f["outline"]) >= 4 and f["length_px"] > 200


def test_a_crowded_tray_keeps_every_fish():
    at = [(220 + 330 * c, 110 + 120 * r, 250) for r in range(5) for c in range(3)]
    assert len(find_fish(tray(at))) == 15


def test_an_empty_tray_has_no_fish():
    img = np.full((600, 900, 3), 245, np.uint8)
    cv2.rectangle(img, (100, 450), (700, 500), (20, 20, 20), -1)
    assert find_fish(img) == []
