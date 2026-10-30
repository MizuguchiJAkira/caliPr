"""Body depth from an outline: square to the fish's own axis, fins bridged."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("auto_body_traits", SCRIPTS / "auto_body_traits.py")
abt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abt)


def fish(depth=200.0, length=1000.0, angle=0.0, x0=500.0, y0=800.0, belly_bump=None):
    """A spindle-shaped outline, its deepest point at 40% of its length, turned by
    ``angle``; with ``belly_bump`` = (u_from, u_to, extra) a fin hanging below."""
    top, bot = [], []
    for i in range(201):
        u = length * i / 200
        h = depth / 2 * math.sin(math.pi * (u / length) ** (math.log(0.5) / math.log(0.4)))
        extra = belly_bump[2] if belly_bump and belly_bump[0] <= u <= belly_bump[1] else 0.0
        top.append((u, -h)); bot.append((u, h + extra))
    c, s = math.cos(angle), math.sin(angle)
    place = lambda u, v: [x0 + c * u - s * v, y0 + s * u + c * v]  # noqa: E731
    poly = [place(u, v) for u, v in top + bot[::-1]]
    kps = {"premaxilla_tip": place(0, 0), "caudal_base": place(length, 0),
           "operculum_posterior": place(0.2 * length, 0),
           "pectoral_insertion_upper": place(0.22 * length, 0.3 * depth)}
    return poly, kps, place


def test_the_greatest_depth_is_found_where_the_fish_is_deepest():
    poly, kps, _ = fish()
    mbd, at, _ = abt.body_depths(poly, kps)
    assert mbd == pytest.approx(200, rel=0.01) and at == pytest.approx(0.4, abs=0.02)


def test_a_fish_lying_at_a_slant_measures_the_same():
    poly, kps, _ = fish(angle=math.radians(20))
    assert abt.body_depths(poly, kps)[0] == pytest.approx(200, rel=0.01)


def test_a_pectoral_fin_hanging_below_the_belly_is_bridged():
    poly, kps, place = fish(belly_bump=(380, 480, 60))           # the fin, under the deepest part
    fin = [place(372, 100), place(488, 100), place(488, 170), place(372, 170)]  # encloses its bulge
    assert abt.body_depths(poly, kps)[0] > 240                   # counted, without the fin
    assert abt.body_depths(poly, kps, fin)[0] == pytest.approx(200, rel=0.02)


def test_the_typical_tip_is_placed_along_and_across_the_fish():
    _, kps, place = fish(angle=math.radians(10))
    tip = abt.at_offset(kps, kps["pectoral_insertion_upper"], (0.145, 0.032))
    want = place(0.22 * 1000 + 145, 0.3 * 200 + 32)
    assert math.dist(tip, want) < 0.5
