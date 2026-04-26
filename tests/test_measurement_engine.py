"""Geometry tests for the MorFishJ-port measurement engine."""

import math

import pytest

from fish_morpho.landmark_config import TRAITS, Unit, View
from fish_morpho.measurement_engine import (
    Annotation,
    _split_polygon_along_line_a,
    _sutherland_hodgman,
    _vertical_extent_at_x,
    compute_all,
    shoelace_area,
)
from fish_morpho.ruler_calibration import CalibrationResult, scale_from_known_span


# ---------------------------------------------------------------------------
# Fixtures — synthetic fish that exercises every computer
# ---------------------------------------------------------------------------


def _unit_calib(px_per_mm: float = 10.0) -> CalibrationResult:
    return CalibrationResult(
        px_per_mm=px_per_mm, method="manual", confidence=1.0
    )


def _synthetic_fish() -> Annotation:
    """Stylized left-facing fish with all polygons and keypoints present.

    Coordinates picked so that round-number expected values fall out of
    the MorFishJ formulas at 10 px/mm calibration:
      * body extent 0..200 px  → TL = 20 mm
      * caudal base at x=165   → SL = 16.5 mm
      * body vertical 10..90   → MBd = 8 mm
      * operculum at x=45      → Hl = 4.5 mm
      * peduncle line A at x=160 y in (30, 70) → CPd = 4 mm
    """
    body_plus_caudal = [
        (0, 50),     # snout tip (min x → reference line D)
        (20, 20),
        (60, 10),
        (120, 10),   # top of back
        (150, 15),
        (160, 30),   # peduncle narrowest dorsal (on line A)
        (180, 10),   # top of caudal fin
        (200, 50),   # caudal fin tip (max x → reference line E)
        (180, 90),
        (160, 70),   # peduncle narrowest ventral (on line A)
        (150, 85),
        (120, 90),
        (60, 90),
        (20, 80),
    ]
    return Annotation(
        polygons={
            "body_plus_caudal": body_plus_caudal,
            "pectoral": [(40, 60), (55, 75), (35, 80), (30, 70)],
            "dorsal": [(80, 10), (100, -5), (115, 10)],
            "pelvic": [(90, 90), (100, 100), (85, 95)],
            "anal": [(130, 90), (145, 100), (125, 95)],
        },
        keypoints={
            "eye_anterior": (20, 45),
            "eye_posterior": (30, 45),
            "eye_dorsal": (25, 40),
            "eye_ventral": (25, 50),
            "premaxilla_tip": (0, 50),
            "maxilla_mandible_intersection": (15, 55),
            "lower_jaw_tip": (0, 55),
            "operculum_posterior": (45, 55),
            "pectoral_insertion_upper": (40, 60),
            "pectoral_ray_tip": (55, 75),
            "peduncle_narrowest_dorsal": (160, 30),
            "peduncle_narrowest_ventral": (160, 70),
            "caudal_base": (165, 50),
            "dorsal_base_center": (100, 10),
            "dorsal_tip": (100, -5),
            "pelvic_base_center": (95, 90),
            "pelvic_tip": (100, 100),
            "anal_base_center": (135, 90),
            "anal_tip": (145, 100),
            "mouth_left": (1000, 500),
            "mouth_right": (1050, 500),
        },
    )


def _both_calibs() -> dict:
    return {
        View.LATERAL: _unit_calib(10.0),
        View.FRONTAL: _unit_calib(5.0),
    }


# ---------------------------------------------------------------------------
# Primitive geometry helpers
# ---------------------------------------------------------------------------


def test_shoelace_unit_square():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert shoelace_area(square) == pytest.approx(1.0)


def test_shoelace_is_orientation_invariant():
    cw = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    ccw = list(reversed(cw))
    assert shoelace_area(cw) == pytest.approx(shoelace_area(ccw))


def test_shoelace_handles_degenerate_inputs():
    assert shoelace_area([(0.0, 0.0), (1.0, 1.0)]) == 0.0
    assert shoelace_area([]) == 0.0


def test_shoelace_triangle():
    triangle = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
    assert shoelace_area(triangle) == pytest.approx(6.0)


def test_vertical_extent_crosses_simple_box():
    box = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _vertical_extent_at_x(box, 5) == (0, 10)


def test_vertical_extent_returns_none_when_outside():
    box = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _vertical_extent_at_x(box, 20) is None


def test_vertical_extent_ignores_degenerate_edges():
    # Vertical edge at x=10 shouldn't crash (division by zero guarded).
    box = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ext = _vertical_extent_at_x(box, 5)
    assert ext == (0, 10)


# ---------------------------------------------------------------------------
# Polygon clipping + line-A split
# ---------------------------------------------------------------------------


def test_sutherland_hodgman_clips_square_in_half_vertically():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    line_a = (5, -5)
    line_b = (5, 15)
    # Keep the left half (vertices with +/- sign depending on orientation).
    # Just verify both halves together sum to the whole and the anchor
    # side semantics work via _split_polygon_along_line_a.
    left, right = _split_polygon_along_line_a(square, line_a, line_b)
    assert shoelace_area(left) + shoelace_area(right) == pytest.approx(100.0)
    # Left half should roughly be 50 (two equal halves).
    assert shoelace_area(left) == pytest.approx(50.0)
    assert shoelace_area(right) == pytest.approx(50.0)


def test_split_is_area_additive_on_synthetic_fish():
    ann = _synthetic_fish()
    poly = ann.polygons["body_plus_caudal"]
    la = ann.keypoints["peduncle_narrowest_dorsal"]
    lb = ann.keypoints["peduncle_narrowest_ventral"]
    body, caudal = _split_polygon_along_line_a(poly, la, lb)
    assert shoelace_area(body) + shoelace_area(caudal) == pytest.approx(
        shoelace_area(poly)
    )
    # Body should be much larger than caudal for this shape.
    assert shoelace_area(body) > shoelace_area(caudal)


def test_split_degenerate_returns_whole_as_body():
    # A polygon entirely to one side of "line A" — caudal half collapses.
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    la = (50, 0)
    lb = (50, 10)
    body, caudal = _split_polygon_along_line_a(poly, la, lb)
    assert caudal == []
    assert body == list(poly)


def test_sutherland_hodgman_rejects_bad_keep_sign():
    poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert _sutherland_hodgman(poly, (0, 0), (1, 1), keep_sign=0) == []


# ---------------------------------------------------------------------------
# Trait computation — spot checks on the synthetic fish
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_measurements():
    ann = _synthetic_fish()
    return compute_all("SYNTH-01", ann, _both_calibs())


def test_every_trait_computes_cleanly_on_synthetic_fish(synthetic_measurements):
    ms = synthetic_measurements
    assert len(ms.values) == len(TRAITS)
    for code, mv in ms.values.items():
        assert not math.isnan(mv.value), f"{code}: {mv.missing_landmarks}"
        assert mv.missing_landmarks == ()
        assert mv.is_valid


def test_TL_matches_polygon_width(synthetic_measurements):
    # (200 - 0) px / 10 px/mm = 20 mm
    assert synthetic_measurements.values["TL"].value == pytest.approx(20.0)


def test_SL_uses_caudal_base_keypoint(synthetic_measurements):
    # (165 - 0) / 10 = 16.5 mm
    assert synthetic_measurements.values["SL"].value == pytest.approx(16.5)


def test_MBd_is_body_half_vertical_extent(synthetic_measurements):
    # body half y ranges 10..90 → 80 px → 8 mm
    assert synthetic_measurements.values["MBd"].value == pytest.approx(8.0)


def test_CPd_is_peduncle_line_a_length(synthetic_measurements):
    # 40 px / 10 = 4 mm
    assert synthetic_measurements.values["CPd"].value == pytest.approx(4.0)


def test_Ed_is_horizontal_eye_span(synthetic_measurements):
    # eye anterior (20, 45) → posterior (30, 45) = 10 px
    assert synthetic_measurements.values["Ed"].value == pytest.approx(1.0)


def test_Jl_is_euclidean(synthetic_measurements):
    # premax (0, 50) → jaw_joint (15, 55): hypot(15, 5) = 15.811...
    expected = math.hypot(15, 5) / 10
    assert synthetic_measurements.values["Jl"].value == pytest.approx(expected)


def test_EMa_is_positive_when_eye_above_mouth(synthetic_measurements):
    # eye centroid (25, 45) is visually above premaxilla (0, 50) → positive.
    assert synthetic_measurements.values["EMa"].value > 0
    assert synthetic_measurements.values["EMa"].unit == "deg"


def test_MW_uses_frontal_calibration(synthetic_measurements):
    # mouth pixels 50, frontal calib 5 px/mm → 10 mm
    assert synthetic_measurements.values["MW"].value == pytest.approx(10.0)
    assert synthetic_measurements.values["MW"].view == View.FRONTAL


def test_Bs_plus_CFs_equals_total_body_area():
    ann = _synthetic_fish()
    ms = compute_all("SYNTH-01", ann, _both_calibs())
    total = shoelace_area(ann.polygons["body_plus_caudal"]) / (10**2)
    assert ms.values["Bs"].value + ms.values["CFs"].value == pytest.approx(
        total
    )


# ---------------------------------------------------------------------------
# Missing / degenerate inputs
# ---------------------------------------------------------------------------


def test_missing_polygon_flags_trait_missing():
    ann = _synthetic_fish()
    ann.polygons.pop("body_plus_caudal")
    ms = compute_all("SYNTH-02", ann, _both_calibs())
    tl = ms.values["TL"]
    assert math.isnan(tl.value)
    assert "polygon:body_plus_caudal" in tl.missing_landmarks


def test_missing_keypoint_flags_trait_missing():
    ann = _synthetic_fish()
    ann.keypoints.pop("caudal_base")
    ms = compute_all("SYNTH-03", ann, _both_calibs())
    sl = ms.values["SL"]
    assert math.isnan(sl.value)
    assert "keypoint:caudal_base" in sl.missing_landmarks


def test_polygon_with_two_vertices_is_missing():
    ann = _synthetic_fish()
    ann.polygons["pectoral"] = [(0, 0), (1, 1)]  # not a polygon
    ms = compute_all("SYNTH-04", ann, _both_calibs())
    pfs = ms.values["PFs"]
    assert math.isnan(pfs.value)
    assert "polygon:pectoral" in pfs.missing_landmarks


def test_other_traits_still_compute_when_one_is_missing():
    ann = _synthetic_fish()
    ann.polygons.pop("pectoral")
    ms = compute_all("SYNTH-05", ann, _both_calibs())
    # Pectoral-area trait should NaN out …
    assert math.isnan(ms.values["PFs"].value)
    # … but body-level traits are unaffected.
    assert not math.isnan(ms.values["TL"].value)
    assert not math.isnan(ms.values["Bs"].value)


def test_frontal_missing_calibration_raises():
    ann = _synthetic_fish()
    # Only lateral calibration — MW needs frontal.
    with pytest.raises(KeyError):
        compute_all("SYNTH-06", ann, {View.LATERAL: _unit_calib(10.0)})


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def test_deg_traits_bypass_calibration():
    ann = _synthetic_fish()
    # Halve the px/mm: deg traits should be unchanged.
    coarse = {View.LATERAL: _unit_calib(5.0), View.FRONTAL: _unit_calib(5.0)}
    fine = {View.LATERAL: _unit_calib(10.0), View.FRONTAL: _unit_calib(10.0)}
    assert (
        compute_all("A", ann, coarse).values["EMa"].value
        == compute_all("B", ann, fine).values["EMa"].value
    )


def test_area_scales_with_px_per_mm_squared():
    ann = _synthetic_fish()
    ms_a = compute_all("A", ann, _both_calibs())
    other = {View.LATERAL: _unit_calib(20.0), View.FRONTAL: _unit_calib(5.0)}
    ms_b = compute_all("B", ann, other)
    # Doubling px/mm → quartering mm^2.
    assert ms_b.values["Bs"].value == pytest.approx(
        ms_a.values["Bs"].value / 4
    )


# ---------------------------------------------------------------------------
# Calibration helpers (kept from the old suite for safety)
# ---------------------------------------------------------------------------


def test_scale_from_known_span_round_trip():
    calib = scale_from_known_span((0.0, 0.0), (100.0, 0.0), known_mm=50.0)
    assert calib.px_per_mm == pytest.approx(2.0)
    assert calib.px_to_mm(200.0) == pytest.approx(100.0)
    assert calib.area_px_to_mm2(400.0) == pytest.approx(100.0)


def test_scale_from_known_span_rejects_coincident_points():
    with pytest.raises(ValueError):
        scale_from_known_span((1.0, 1.0), (1.0, 1.0), known_mm=10.0)


def test_scale_from_known_span_rejects_nonpositive_mm():
    with pytest.raises(ValueError):
        scale_from_known_span((0.0, 0.0), (1.0, 0.0), known_mm=0.0)
