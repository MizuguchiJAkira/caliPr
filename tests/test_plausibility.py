"""Bands must catch the impossible and leave correct landmarks alone."""

from __future__ import annotations

import json
import pathlib

import pytest

from fish_morpho import plausibility as P

# A rectangle 1000 wide: fraction-of-body is then just x / 1000.
BODY = [[0, 0], [1000, 0], [1000, 200], [0, 200]]


def _record(**named):
    return {"keypoints": {k: [v, 100] for k, v in named.items()},
            "polygons": {P.BODY: BODY}}


def _spread(name, lo, hi, n=6):
    """``n`` fish placing ``name`` evenly between ``lo`` and ``hi``."""
    step = (hi - lo) / (n - 1)
    return [_record(**{name: lo + step * i, "premaxilla_tip": 5,
                       "caudal_base": 900}) for i in range(n)]


def test_fit_records_counts_and_bands():
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    entry = bands["landmarks"]["pelvic_base_center"]
    assert entry["n"] == 6
    assert entry["axial_observed"] == [0.4, 0.46]
    assert bands["fish"] == 6


def test_landmark_below_min_samples_gets_no_band():
    bands = P.fit(_spread("dorsal_tip", 400, 500, n=P.MIN_SAMPLES - 1))
    entry = bands["landmarks"]["dorsal_tip"]
    assert entry["n"] == P.MIN_SAMPLES - 1
    assert "axial" not in entry


def test_margin_scales_with_observed_spread():
    tight = P.fit(_spread("anal_tip", 500, 520))["landmarks"]["anal_tip"]["axial"]
    loose = P.fit(_spread("anal_tip", 400, 700))["landmarks"]["anal_tip"]["axial"]
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])


def test_margin_has_a_floor():
    """A landmark labelled identically every time still gets room to vary."""
    band = P.fit(_spread("caudal_base", 800, 800))["landmarks"]["caudal_base"]["axial"]
    assert band[1] - band[0] == pytest.approx(2 * P.MIN_MARGIN)


def test_point_inside_the_band_is_not_flagged():
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    kps = {"pelvic_base_center": [430, 100], "premaxilla_tip": [5, 100],
           "caudal_base": [900, 100]}
    assert P.check(kps, BODY, bands) == {}


def test_point_far_outside_the_band_is_flagged():
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    kps = {"pelvic_base_center": [20, 100], "premaxilla_tip": [5, 100],
           "caudal_base": [900, 100]}
    bad = P.check(kps, BODY, bands)
    assert "pelvic_base_center" in bad
    assert "0.02 along the body" in bad["pelvic_base_center"]


def test_mirrored_frame_is_reported_once_not_nineteen_times():
    """Head-right invalidates every axial position, so say that and stop."""
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    kps = {"premaxilla_tip": [900, 100], "caudal_base": [5, 100],
           "pelvic_base_center": [20, 100]}
    bad = P.check(kps, BODY, bands)
    assert list(bad) == ["_frame"]
    assert "head-left" in bad["_frame"]


@pytest.mark.parametrize("polygon", [None, [], [[0, 0], [1, 1]]])
def test_no_usable_outline_withholds_judgement(polygon):
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    assert P.check({"pelvic_base_center": [20, 100]}, polygon, bands) == {}


def test_unfitted_dataset_is_not_checked():
    """The right default for a taxon nobody has measured."""
    assert P.check({"pelvic_base_center": [20, 100]}, BODY, None) == {}
    assert P.check({"pelvic_base_center": [20, 100]}, BODY, {"landmarks": {}}) == {}


def test_load_missing_or_malformed_returns_none(tmp_path):
    assert P.load(tmp_path) is None
    (tmp_path / "plausibility.json").write_text("{ not json")
    assert P.load(tmp_path) is None
    (tmp_path / "plausibility.json").write_text(json.dumps({"landmarks": {}}))
    assert P.load(tmp_path) is None


def test_load_roundtrips_a_fitted_file(tmp_path):
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    (tmp_path / "plausibility.json").write_text(json.dumps(bands))
    assert P.load(tmp_path)["landmarks"]["pelvic_base_center"]["n"] == 6


def test_describe_names_what_is_not_measurable():
    bands = P.fit(_spread("dorsal_tip", 400, 500, n=2)
                  + _spread("pelvic_base_center", 400, 460))
    text = " ".join(P.describe(bands))
    assert "dorsal_tip" in text and "head-left" in text
    assert "no plausibility bands" in " ".join(P.describe(None))


# A long fat blob is fish-shaped; these are not.
TALL = [[0, 0], [400, 0], [400, 300], [0, 300]]          # aspect 1.3
SPIKY = [[0, 0], [1000, 0], [1000, 200], [500, 40], [0, 200]]   # fill ~0.6, but thin


def test_fit_records_what_a_fish_outline_looks_like():
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    assert bands["outline"]["n"] == 6
    assert bands["outline"]["aspect"][0] > 0


def test_outline_that_is_not_fish_shaped_stops_the_check():
    """A wrong axis rescales every landmark, so check nothing instead."""
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    kps = {"pelvic_base_center": [20, 100], "premaxilla_tip": [5, 100],
           "caudal_base": [380, 100]}
    bad = P.check(kps, TALL, bands)
    assert list(bad) == ["_axis"]
    assert "not fish-shaped" in bad["_axis"]


def test_a_fish_shaped_outline_still_checks_normally():
    bands = P.fit(_spread("pelvic_base_center", 400, 460))
    kps = {"pelvic_base_center": [20, 100], "premaxilla_tip": [5, 100],
           "caudal_base": [900, 100]}
    assert "pelvic_base_center" in P.check(kps, BODY, bands)


def test_outline_shape_rejects_degenerate_input():
    assert P.outline_shape(None) is None
    assert P.outline_shape([[0, 0], [1, 1]]) is None
    assert P.outline_shape([[0, 0], [10, 0], [20, 0]]) is None   # zero height


def test_fit_script_ignores_unreviewed_points_and_model_outlines(tmp_path):
    """Bands are what the model is allowed to output; they cannot come from it."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "fit_plausibility", pathlib.Path(__file__).parent.parent / "scripts/fit_plausibility.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    (tmp_path / "a.json").write_text(json.dumps({
        "fish_id": "a",
        "metadata": {"assist": {"unreviewed": ["dorsal_tip"],
                                "polygons_from_model": ["body_plus_caudal"]}},
        "lateral": {"keypoints": {"dorsal_tip": [1, 1], "eye_dorsal": [2, 2]},
                    "polygons": {"body_plus_caudal": BODY}}}))
    [lat] = list(mod.hand_labelled(tmp_path))
    assert "dorsal_tip" not in lat["keypoints"]
    assert "eye_dorsal" in lat["keypoints"]
    assert "body_plus_caudal" not in lat["polygons"]
