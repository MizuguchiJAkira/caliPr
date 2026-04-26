"""Schema-level tests for the MorFishJ-port landmark/trait definitions."""

import pytest

from fish_morpho.landmark_config import (
    CALIBRATION_KEYPOINTS,
    DERIVED_REFERENCE_LINES,
    KEYPOINTS,
    POLYGONS,
    TRAITS,
    USER_REFERENCE_LINES,
    TraitSource,
    Unit,
    View,
    calibration_keypoint_by_name,
    calibration_keypoint_names,
    keypoint_by_name,
    keypoint_names,
    polygon_by_name,
    polygon_names,
    trait_by_code,
    trait_column_order,
    trait_labels,
    traits_by_source,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_schema_validates():
    """The schema passes its internal consistency check (runs at import too)."""
    validate_schema()


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------


def test_five_polygons_cover_body_and_four_fins():
    names = set(polygon_names())
    assert names == {
        "body_plus_caudal",
        "pectoral",
        "dorsal",
        "pelvic",
        "anal",
    }


def test_all_polygons_are_lateral():
    for p in POLYGONS:
        assert p.view == View.LATERAL


def test_polygon_by_name_round_trips():
    for p in POLYGONS:
        assert polygon_by_name(p.name) is p


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------


def test_lateral_and_frontal_keypoint_sets_dont_overlap():
    lateral = set(keypoint_names(View.LATERAL))
    frontal = set(keypoint_names(View.FRONTAL))
    assert not (lateral & frontal)


def test_frontal_has_two_mouth_keypoints():
    assert set(keypoint_names(View.FRONTAL)) == {"mouth_left", "mouth_right"}


def test_lateral_keypoint_count():
    """19 lateral landmarks: 4 eye cardinals + 3 mouth + 1 operculum +
    2 pectoral + 3 peduncle/caudal + 6 fin base/tip extras. Calibration
    keypoints live in a separate tuple and must NOT inflate this count."""
    assert len(keypoint_names(View.LATERAL)) == 19


def test_every_keypoint_has_labeling_hint():
    for k in KEYPOINTS:
        assert k.labeling_hint
        assert k.description


def test_keypoint_by_name_round_trips():
    for k in KEYPOINTS:
        assert keypoint_by_name(k.name) is k


# ---------------------------------------------------------------------------
# Calibration keypoints (ruler clicks; separate from measurement schema)
# ---------------------------------------------------------------------------


def test_calibration_keypoints_are_ruler_point_a_and_b():
    assert {k.name for k in CALIBRATION_KEYPOINTS} == {
        "ruler_point_a",
        "ruler_point_b",
    }


def test_calibration_keypoints_are_lateral_only():
    for k in CALIBRATION_KEYPOINTS:
        assert k.view == View.LATERAL
    assert set(calibration_keypoint_names(View.FRONTAL)) == set()
    assert set(calibration_keypoint_names(View.LATERAL)) == {
        "ruler_point_a",
        "ruler_point_b",
    }


def test_calibration_keypoints_disjoint_from_anatomical_keypoints():
    """Ruler clicks must NOT pollute the measurement schema — the whole
    reason they're in a separate tuple is so trait validation and
    missing-landmark tracking don't see them."""
    assert not set(calibration_keypoint_names()) & set(keypoint_names())


def test_no_trait_depends_on_a_calibration_keypoint():
    cal_names = set(calibration_keypoint_names())
    for t in TRAITS:
        assert not set(t.required_keypoints) & cal_names, (
            f"Trait {t.code} must not require a calibration keypoint"
        )


def test_calibration_keypoint_by_name_round_trips():
    for k in CALIBRATION_KEYPOINTS:
        assert calibration_keypoint_by_name(k.name) is k
    with pytest.raises(KeyError):
        calibration_keypoint_by_name("not_a_ruler_point")


def test_calibration_keypoints_have_labeling_hints():
    for k in CALIBRATION_KEYPOINTS:
        assert k.description
        assert k.labeling_hint


# ---------------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------------


def test_user_reference_lines_are_A_H_I():
    assert {ul.name for ul in USER_REFERENCE_LINES} == {"A", "H", "I"}


def test_derived_reference_lines_cover_B_through_G():
    # J, K, L are keypoint-anchored at measurement time and don't live in
    # DERIVED_REFERENCE_LINES — they're implicit in the engine.
    assert {dl.name for dl in DERIVED_REFERENCE_LINES} == {
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
    }


def test_line_F_and_G_restrict_to_caudal_half():
    by_name = {dl.name: dl for dl in DERIVED_REFERENCE_LINES}
    assert by_name["F"].polygon_half == "caudal"
    assert by_name["G"].polygon_half == "caudal"
    assert by_name["B"].polygon_half == "body"
    assert by_name["C"].polygon_half == "body"


def test_user_reference_lines_reference_real_keypoints():
    known = set(keypoint_names())
    for ul in USER_REFERENCE_LINES:
        for kp in ul.source_keypoints:
            assert kp in known


# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------


def test_trait_counts():
    """22 MorFishJ traits + 8 Cornell extras = 30 total."""
    assert len(TRAITS) == 30
    assert len(traits_by_source(TraitSource.MORFISHJ)) == 22
    assert len(traits_by_source(TraitSource.EXTRAS)) == 8


def test_all_trait_codes_unique():
    codes = [t.code for t in TRAITS]
    assert len(codes) == len(set(codes))


def test_every_trait_has_a_view_unit_and_source():
    for t in TRAITS:
        assert isinstance(t.view, View)
        assert isinstance(t.unit, Unit)
        assert isinstance(t.source, TraitSource)


def test_extras_carry_reference_sheet_numbers():
    for t in traits_by_source(TraitSource.EXTRAS):
        assert t.number is not None, t.code


def test_morfishj_traits_have_no_reference_sheet_numbers():
    for t in traits_by_source(TraitSource.MORFISHJ):
        assert t.number is None, t.code


def test_every_trait_references_real_shapes():
    poly_names = set(polygon_names())
    kp_names = set(keypoint_names())
    for t in TRAITS:
        for p in t.required_polygons:
            assert p in poly_names, f"{t.code} needs unknown polygon {p}"
        for k in t.required_keypoints:
            assert k in kp_names, f"{t.code} needs unknown keypoint {k}"


def test_trait_views_match_required_shape_views():
    for t in TRAITS:
        for p in t.required_polygons:
            assert polygon_by_name(p).view == t.view, (
                f"{t.code} view mismatch with polygon {p}"
            )
        for k in t.required_keypoints:
            assert keypoint_by_name(k).view == t.view, (
                f"{t.code} view mismatch with keypoint {k}"
            )


def test_mouth_width_is_frontal():
    assert trait_by_code("MW").view == View.FRONTAL


def test_total_length_requires_body_polygon():
    tl = trait_by_code("TL")
    assert "body_plus_caudal" in tl.required_polygons


def test_emangle_is_degrees():
    assert trait_by_code("EMa").unit == Unit.DEG


def test_body_and_caudal_surface_areas_are_mm2():
    assert trait_by_code("Bs").unit == Unit.MM2
    assert trait_by_code("CFs").unit == Unit.MM2


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def test_trait_column_order_groups_morfishj_before_extras():
    order = trait_column_order()
    morfishj_codes = {t.code for t in traits_by_source(TraitSource.MORFISHJ)}
    extras_codes = {t.code for t in traits_by_source(TraitSource.EXTRAS)}
    last_morfishj = max(i for i, c in enumerate(order) if c in morfishj_codes)
    first_extra = min(i for i, c in enumerate(order) if c in extras_codes)
    assert last_morfishj < first_extra


def test_trait_labels_include_unit():
    labels = trait_labels()
    assert labels["TL"].endswith("(mm)")
    assert labels["Bs"].endswith("(mm^2)")
    assert labels["EMa"].endswith("(deg)")
