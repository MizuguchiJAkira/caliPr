"""Anatomical limits on predicted fin outlines.

The property that matters is asymmetric: a constraint must never remove real fin,
and should remove anatomically impossible area. So most of these tests check that
a *plausible* outline survives untouched, and only a few check that a violating
one gets clipped.
"""

from __future__ import annotations

import math

import pytest

from fish_morpho.anatomy_constraints import (
    DORSAL_DISTAL_ALLOWANCE_SL,
    DORSAL_VENTRAL_ALLOWANCE_SL,
    PECTORAL_ANTERIOR_ALLOWANCE_SL,
    constrain_fin_polygon,
    describe_limits,
    insertion_chord,
    limits_for,
    standard_length_px,
)
from fish_morpho.landmark_config import FIN_POLYGONS
from fish_morpho.measurement_engine import clip_polygon_to_halfplane, shoelace_area

# A schematic fish along +x: snout at 0, caudal base at 1000, back at y=0,
# belly at y=200. Image y grows downward, so "dorsal" is smaller y.
SL = 1000.0
KPS = {
    "premaxilla_tip": (0.0, 100.0),
    "caudal_base": (SL, 100.0),
    "operculum_posterior": (250.0, 100.0),
    "dorsal_base_center": (500.0, 0.0),
    "dorsal_tip": (520.0, -120.0),
}
# rectangular body outline, traced dorsal edge left->right then ventral back
BODY = [(0.0, 0.0), (SL, 0.0), (SL, 200.0), (0.0, 200.0)]


def area(p):
    return shoelace_area(p)


def test_standard_length_comes_from_the_axis_keypoints():
    assert standard_length_px(KPS) == pytest.approx(SL)
    assert standard_length_px({"premaxilla_tip": (0.0, 0.0)}) is None


def test_no_limits_encoded_for_pelvic_or_anal():
    assert limits_for("pelvic", KPS, BODY) == []
    assert limits_for("anal", KPS, BODY) == []
    # ...and describe_limits says so rather than staying silent
    assert "no anatomical limit" in describe_limits()["pelvic"][0]


def test_describe_limits_covers_every_fin():
    assert set(describe_limits()) == set(FIN_POLYGONS)


# ---------------------------------------------------------------------------
# Pectoral: anterior bound at the gill opening
# ---------------------------------------------------------------------------


def test_pectoral_overlapping_the_operculum_a_little_is_untouched():
    # real fins overlap: reach 2% of SL anterior of the operculum, inside the 4%
    # allowance, so nothing may be removed
    x0 = KPS["operculum_posterior"][0] - 0.02 * SL
    poly = [(x0, 120.0), (400.0, 120.0), (400.0, 200.0), (x0, 200.0)]
    out, removed, applied = constrain_fin_polygon("pectoral", poly, KPS)
    assert removed == pytest.approx(0.0)
    assert applied == []
    assert area(out) == pytest.approx(area(poly))


def test_pectoral_running_far_over_the_gill_cover_is_clipped():
    x0 = KPS["operculum_posterior"][0] - 0.12 * SL      # well past the allowance
    poly = [(x0, 120.0), (400.0, 120.0), (400.0, 200.0), (x0, 200.0)]
    out, removed, applied = constrain_fin_polygon("pectoral", poly, KPS)
    assert removed > 0.1
    assert applied == ["anterior of operculum"]
    limit_x = KPS["operculum_posterior"][0] - PECTORAL_ANTERIOR_ALLOWANCE_SL * SL
    assert min(p[0] for p in out) >= limit_x - 1e-6


def test_pectoral_bound_follows_the_fish_not_the_image():
    # rotate the whole specimen 90 degrees; the same fin must clip the same way
    def rot(p):
        return (-p[1], p[0])
    kps = {k: rot(v) for k, v in KPS.items()}
    x0 = KPS["operculum_posterior"][0] - 0.12 * SL
    poly = [(x0, 120.0), (400.0, 120.0), (400.0, 200.0), (x0, 200.0)]
    _, flat, _ = constrain_fin_polygon("pectoral", poly, KPS)
    _, turned, _ = constrain_fin_polygon("pectoral", [rot(p) for p in poly], kps)
    assert turned == pytest.approx(flat, abs=1e-6)


# ---------------------------------------------------------------------------
# Dorsal: ventral bound at the insertion, distal bound at the tip
# ---------------------------------------------------------------------------


def test_insertion_chord_picks_points_flanking_the_base():
    chord = insertion_chord(BODY, KPS["dorsal_base_center"], SL)
    assert chord is not None
    a, b = chord
    # both on the traced dorsal edge, on opposite sides of the fin base
    assert a[1] == pytest.approx(0.0) and b[1] == pytest.approx(0.0)
    assert min(a[0], b[0]) <= KPS["dorsal_base_center"][0] <= max(a[0], b[0])


def test_plausible_dorsal_fin_is_untouched():
    poly = [(460.0, 0.0), (540.0, 0.0), (520.0, -110.0)]     # sits on the back
    out, removed, applied = constrain_fin_polygon("dorsal", poly, KPS, BODY)
    assert removed == pytest.approx(0.0)
    assert applied == []
    assert area(out) == pytest.approx(area(poly))


def test_dorsal_dipping_into_the_flank_is_clipped():
    poly = [(460.0, 0.0), (540.0, 0.0), (540.0, 60.0), (460.0, 60.0)]  # below the back
    out, removed, applied = constrain_fin_polygon("dorsal", poly, KPS, BODY)
    assert removed > 0.3
    assert "ventral of insertion" in applied
    limit_y = DORSAL_VENTRAL_ALLOWANCE_SL * SL
    assert max(p[1] for p in out) <= limit_y + 1e-6


def test_dorsal_beyond_the_tip_is_clipped_this_is_the_specimen_pin():
    # a lobe far above the fin tip, like a pin head
    poly = [(460.0, 0.0), (540.0, 0.0), (540.0, -400.0), (460.0, -400.0)]
    out, removed, applied = constrain_fin_polygon("dorsal", poly, KPS, BODY)
    assert "beyond dorsal tip" in applied
    limit_y = KPS["dorsal_tip"][1] - DORSAL_DISTAL_ALLOWANCE_SL * SL
    assert min(p[1] for p in out) >= limit_y - 1e-6
    assert removed > 0.3


def test_dorsal_distal_bound_is_skipped_without_a_tip():
    kps = {k: v for k, v in KPS.items() if k != "dorsal_tip"}
    names = [n for *_, n in limits_for("dorsal", kps, BODY)]
    assert names == ["ventral of insertion"]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_landmarks_leave_the_polygon_alone():
    poly = [(460.0, 0.0), (540.0, 0.0), (520.0, -110.0)]
    for kps in ({}, {"premaxilla_tip": (0.0, 100.0)}):
        out, removed, applied = constrain_fin_polygon("dorsal", poly, kps, BODY)
        assert out == poly and removed == 0.0 and applied == []


def test_unknown_fin_and_degenerate_polygon_are_passed_through():
    poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert constrain_fin_polygon("caudal", poly, KPS, BODY)[0] == poly
    assert constrain_fin_polygon("dorsal", [(0.0, 0.0), (1.0, 1.0)], KPS, BODY)[0] == \
        [(0.0, 0.0), (1.0, 1.0)]


def test_a_limit_that_would_erase_the_fin_is_refused_not_obeyed():
    # entirely on the forbidden side: bad landmarks, not a nonexistent fin. Better
    # to keep a suspect area than to report a confident zero.
    poly = [(460.0, 300.0), (540.0, 300.0), (540.0, 340.0), (460.0, 340.0)]
    out, removed, applied = constrain_fin_polygon("dorsal", poly, KPS, BODY)
    assert out == poly
    assert removed == 0.0
    assert applied == []


# ---------------------------------------------------------------------------
# The public clipper this is built on
# ---------------------------------------------------------------------------


def test_clip_to_halfplane_keeps_the_side_the_reference_point_is_on():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    lower = clip_polygon_to_halfplane(square, (0.0, 5.0), (10.0, 5.0), (5.0, 1.0))
    upper = clip_polygon_to_halfplane(square, (0.0, 5.0), (10.0, 5.0), (5.0, 9.0))
    assert area(lower) == pytest.approx(50.0)
    assert area(upper) == pytest.approx(50.0)
    assert max(p[1] for p in lower) == pytest.approx(5.0)
    assert min(p[1] for p in upper) == pytest.approx(5.0)


def test_clip_with_the_reference_point_on_the_line_is_degenerate():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert clip_polygon_to_halfplane(square, (0.0, 5.0), (10.0, 5.0), (5.0, 5.0)) == []


def test_clip_leaves_a_polygon_that_does_not_cross_the_line():
    tri = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)]
    out = clip_polygon_to_halfplane(tri, (0.0, 20.0), (10.0, 20.0), (5.0, 1.0))
    assert area(out) == pytest.approx(area(tri))


def test_rotating_the_scene_does_not_change_the_clipped_area():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    th = 0.7
    c, s = math.cos(th), math.sin(th)

    def rot(p):
        return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)

    flat = clip_polygon_to_halfplane(square, (0.0, 5.0), (10.0, 5.0), (5.0, 1.0))
    turned = clip_polygon_to_halfplane(
        [rot(p) for p in square], rot((0.0, 5.0)), rot((10.0, 5.0)), rot((5.0, 1.0))
    )
    assert area(turned) == pytest.approx(area(flat))
